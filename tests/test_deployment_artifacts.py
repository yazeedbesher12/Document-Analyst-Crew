import json
from pathlib import Path

from greenloop_rag_crew import runtime_paths


def test_prebuilt_index_manifest_is_portable_and_uses_relative_artifacts():
    manifest = json.loads(Path("storage/index_manifest.json").read_text(encoding="utf-8"))

    assert manifest["chunks_file"] == "storage/chunks.jsonl"
    assert all(":" not in value for value in [manifest["chunks_file"]])
    assert Path("storage/chroma").is_dir()
    assert any(Path("storage/chroma").rglob("*"))


def test_runtime_default_paths_are_portable_on_windows_and_linux(monkeypatch):
    for variable in ("KNOWLEDGE_DIR", "CHROMA_PERSIST_DIR", "OUTPUT_DIR"):
        monkeypatch.delenv(variable, raising=False)

    assert runtime_paths.knowledge_dir() == Path("knowledge")
    assert runtime_paths.chroma_persist_dir() == Path("storage/chroma")
    assert runtime_paths.output_dir() == Path("output")
