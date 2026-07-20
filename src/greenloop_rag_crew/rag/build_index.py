"""Build and validate the persistent dense Chroma index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from greenloop_rag_crew.rag.chroma_store import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PERSIST_DIRECTORY,
    ChromaStore,
)
from greenloop_rag_crew.rag.chunker import MAX_EMBEDDING_TOKENS
from greenloop_rag_crew.rag.document_registry import DOCUMENT_REGISTRY, validate_knowledge_pack
from greenloop_rag_crew.rag.embedder import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    EXPECTED_EMBEDDING_DIMENSION,
    GreenLoopEmbedder,
    _env_int,
)
from greenloop_rag_crew.rag.schemas import DocumentChunk
from greenloop_rag_crew.runtime_paths import (
    chroma_persist_dir,
    chunks_file as configured_chunks_file,
    knowledge_dir as configured_knowledge_dir,
    manifest_file,
)

INDEX_SCHEMA_VERSION = 1
MANIFEST_PATH = manifest_file()
DEFAULT_CHUNKS_FILE = "storage/chunks.jsonl"
NORMALIZED_EMBEDDINGS = True
DISTANCE_METRIC = "cosine"


class ChunkValidationError(ValueError):
    """Raised when chunks.jsonl is not safe to index."""


def load_chunks(chunks_file: str | Path) -> list[DocumentChunk]:
    """Load and validate DocumentChunk records before model loading."""

    path = Path(chunks_file)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {path}")

    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChunkValidationError(
                    f"Malformed JSON on line {line_number} of {path}."
                ) from exc
            try:
                chunk = DocumentChunk.model_validate(payload)
            except Exception as exc:
                raise ChunkValidationError(
                    f"Invalid DocumentChunk on line {line_number} of {path}: {exc}"
                ) from exc
            if chunk.chunk_id in seen_ids:
                raise ChunkValidationError(f"Duplicate chunk_id found: {chunk.chunk_id}")
            if not chunk.embedding_text.strip():
                raise ChunkValidationError(f"{chunk.chunk_id}: embedding_text is empty.")
            if chunk.token_count > MAX_EMBEDDING_TOKENS:
                raise ChunkValidationError(
                    f"{chunk.chunk_id}: token_count {chunk.token_count} exceeds "
                    f"{MAX_EMBEDDING_TOKENS}."
                )
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)

    if not chunks:
        raise ChunkValidationError(f"No chunks found in {path}.")
    return chunks


def build_index(
    chunks_file: str | Path | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    batch_size: int | None = None,
    rebuild: bool = False,
    knowledge_dir: str | Path | None = None,
) -> str:
    """Build or skip a current dense index and return an action string."""

    chunks_path = Path(chunks_file) if chunks_file is not None else configured_chunks_file()
    persist_path = Path(persist_dir) if persist_dir is not None else chroma_persist_dir()
    knowledge_path = Path(knowledge_dir) if knowledge_dir is not None else configured_knowledge_dir()
    batch_size = batch_size or _env_int("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
    embedding_model = _configured_embedding_model()

    chunks = load_chunks(chunks_path)
    manifest_candidate = build_manifest(
        chunks=chunks,
        chunks_file=chunks_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        knowledge_dir=knowledge_path,
    )
    store = ChromaStore(persist_dir=persist_path, collection_name=collection_name)

    if not rebuild and is_index_current(manifest_candidate, store):
        print(
            f"Index is up to date: {collection_name} has {len(chunks)} records. "
            "Skipping re-embedding."
        )
        return "skipped"

    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    collection = store.recreate_collection()
    if collection.count() != 0:
        raise RuntimeError(f"Collection {collection_name!r} was not empty after recreation.")

    embedder = GreenLoopEmbedder(
        model_name=embedding_model,
        batch_size=batch_size,
    )
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_embeddings = embedder.embed_documents([chunk.embedding_text for chunk in batch])
        embeddings.extend(batch_embeddings)

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")
    for index, embedding in enumerate(embeddings):
        if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding {index} has dimension {len(embedding)}, expected "
                f"{EXPECTED_EMBEDDING_DIMENSION}."
            )

    store.add_chunks(chunks, embeddings, batch_size=batch_size)
    final_count = store.count()
    if final_count != len(chunks):
        raise RuntimeError(
            f"Collection count mismatch: expected {len(chunks)}, found {final_count}."
        )

    final_manifest = {
        **manifest_candidate,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Indexed {len(chunks)} chunks into {collection_name} "
        f"at {persist_path} using {manifest_candidate['embedding_model']}."
    )
    return "rebuilt"


def build_manifest(
    chunks: list[DocumentChunk],
    chunks_file: str | Path,
    collection_name: str,
    embedding_model: str | None = None,
    knowledge_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the comparable manifest payload. created_at is added on success."""

    chunks_path = Path(chunks_file)
    knowledge_path = Path(knowledge_dir) if knowledge_dir is not None else configured_knowledge_dir()
    validate_knowledge_pack(knowledge_path)

    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": None,
        "collection_name": collection_name,
        "embedding_model": embedding_model or _configured_embedding_model(),
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "normalized_embeddings": NORMALIZED_EMBEDDINGS,
        "distance_metric": DISTANCE_METRIC,
        "chunk_count": len(chunks),
        "chunks_file": _relative_posix(chunks_path),
        "chunks_file_sha256": sha256_file(chunks_path),
        "pdfs": [
            {
                "filename": metadata.filename,
                "document_id": metadata.document_id,
                "sha256": sha256_file(knowledge_path / metadata.filename),
            }
            for metadata in DOCUMENT_REGISTRY
        ],
        "package_versions": package_versions(),
    }


def is_index_current(expected_manifest: dict[str, Any], store: ChromaStore) -> bool:
    """Return True when manifest and Chroma count match the expected state."""

    actual_manifest = read_manifest()
    if actual_manifest is None:
        return False

    comparable_expected = _manifest_without_created_at(expected_manifest)
    comparable_actual = _manifest_without_created_at(actual_manifest)
    if comparable_actual != comparable_expected:
        return False
    return store.count() == expected_manifest["chunk_count"]


def read_manifest(path: str | Path | None = None) -> dict[str, Any] | None:
    manifest_path = Path(path) if path is not None else MANIFEST_PATH
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def package_versions() -> dict[str, str]:
    names = [
        "sentence-transformers",
        "transformers",
        "chromadb",
        "numpy",
        "pydantic",
        "python-dotenv",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--knowledge-dir", default=None)
    parser.add_argument("--collection", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    build_index(
        chunks_file=args.chunks,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
        knowledge_dir=args.knowledge_dir,
    )


def _manifest_without_created_at(manifest: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(manifest)
    comparable["created_at"] = None
    return comparable


def _relative_posix(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _configured_embedding_model() -> str:
    load_dotenv()
    return os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


if __name__ == "__main__":
    main()
