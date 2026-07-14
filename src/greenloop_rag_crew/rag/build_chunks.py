"""CLI for building deterministic page-aware chunk JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from greenloop_rag_crew.rag.chunker import create_chunks
from greenloop_rag_crew.rag.document_registry import DOCUMENT_REGISTRY, validate_knowledge_pack
from greenloop_rag_crew.rag.pdf_loader import extract_pages
from greenloop_rag_crew.rag.schemas import DocumentChunk


def build_chunks(
    knowledge_dir: str | Path = "knowledge", output: str | Path = "storage/chunks.jsonl"
) -> list[DocumentChunk]:
    """Validate PDFs, extract pages, create chunks, and write JSON Lines."""

    knowledge_path = Path(knowledge_dir)
    output_path = Path(output)

    registry = validate_knowledge_pack(knowledge_path)
    pages = extract_pages(knowledge_path)
    chunks = create_chunks(pages)

    registry_order = {metadata.document_id: index for index, metadata in enumerate(registry)}
    chunks.sort(key=lambda chunk: (registry_order[chunk.document_id], chunk.page, chunk.chunk_id))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            payload = chunk.model_dump(mode="json")
            file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    _print_summary(registry, pages, chunks, output_path)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", default="knowledge")
    parser.add_argument("--output", default="storage/chunks.jsonl")
    args = parser.parse_args()

    build_chunks(knowledge_dir=args.knowledge_dir, output=args.output)


def _print_summary(registry, pages, chunks, output_path: Path) -> None:
    counts = Counter(chunk.document_id for chunk in chunks)
    title_by_id = {metadata.document_id: metadata.title for metadata in DOCUMENT_REGISTRY}
    max_tokens = max((chunk.token_count for chunk in chunks), default=0)

    print(f"PDFs: {len(registry)}")
    print(f"Total pages: {len(pages)}")
    print("Chunks per document:")
    for metadata in registry:
        print(f"  {metadata.document_id} ({title_by_id[metadata.document_id]}): {counts[metadata.document_id]}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Max embedding_text token count: {max_tokens}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
