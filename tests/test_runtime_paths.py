from pathlib import Path

from greenloop_rag_crew import runtime_paths


def test_default_runtime_paths_remain_project_relative(monkeypatch):
    for variable in (
        "KNOWLEDGE_DIR",
        "CHROMA_PERSIST_DIR",
        "CHROMA_PERSIST_DIRECTORY",
        "OUTPUT_DIR",
    ):
        monkeypatch.delenv(variable, raising=False)

    assert runtime_paths.knowledge_dir() == Path("knowledge")
    assert runtime_paths.chroma_persist_dir() == Path("storage/chroma")
    assert runtime_paths.output_dir() == Path("output")


def test_configured_runtime_paths_are_created_and_safe(tmp_path, monkeypatch):
    knowledge = tmp_path / "pdfs"
    chroma = tmp_path / "state" / "chroma"
    reports = tmp_path / "reports"
    hf_cache = tmp_path / "cache" / "hf"
    sentence_cache = tmp_path / "cache" / "sentence-transformers"
    for variable, value in {
        "KNOWLEDGE_DIR": knowledge,
        "CHROMA_PERSIST_DIR": chroma,
        "OUTPUT_DIR": reports,
        "HF_HOME": hf_cache,
        "SENTENCE_TRANSFORMERS_HOME": sentence_cache,
    }.items():
        monkeypatch.setenv(variable, str(value))

    assert runtime_paths.knowledge_dir() == knowledge
    assert runtime_paths.chroma_persist_dir() == chroma
    assert runtime_paths.output_dir() == reports
    assert chroma.is_dir()
    assert reports.is_dir()
    assert set(runtime_paths.prepare_model_cache_dirs()) == {hf_cache, sentence_cache}
    assert hf_cache.is_dir()
    assert sentence_cache.is_dir()
    assert runtime_paths.resolve_configured_output_path("output/report.md") == reports / "report.md"
