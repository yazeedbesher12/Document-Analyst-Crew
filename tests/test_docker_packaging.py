from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_locked_dependencies_and_streamlit_entrypoint():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.13-slim" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "COPY knowledge ./knowledge" in dockerfile
    assert "all-mpnet-base-v2" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "_stcore/health" in dockerfile
    assert "OLLAMA_BASE_URL" not in dockerfile
    assert "COPY .env" not in dockerfile


def test_project_selects_cpu_torch_wheels_for_linux_containers():
    project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in project
    assert "sys_platform == 'linux'" in project


def test_dockerignore_excludes_runtime_state_but_keeps_required_inputs():
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for required_rule in (".git/", ".venv/", ".env", "output/", "storage/"):
        assert required_rule in dockerignore
    for prohibited_rule in ("knowledge/", "*.pdf", "src/", "pyproject.toml", "uv.lock"):
        assert prohibited_rule not in dockerignore


def test_container_startup_builds_local_index_then_runs_streamlit():
    entrypoint = (PROJECT_ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")

    assert "greenloop_rag_crew.rag.build_chunks" in entrypoint
    assert "greenloop_rag_crew.rag.build_index" in entrypoint
    assert "streamlit run /app/streamlit_app.py" in entrypoint
    assert "--server.fileWatcherType=none" in entrypoint


def test_compose_uses_host_ollama_without_packaging_an_ollama_service():
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "host.docker.internal:11434" in compose
    assert "8501:8501" in compose
    assert "./output:/app/output" in compose
    assert "./storage/chroma:/app/storage/chroma" in compose
    assert "_stcore/health" in compose
    assert "ollama:" not in compose


def test_docker_environment_example_contains_placeholders_only():
    environment_example = (PROJECT_ROOT / "docker.env.example").read_text(encoding="utf-8")

    assert "LLM_PROVIDER=ollama" in environment_example
    assert "OLLAMA_BASE_URL=http://host.docker.internal:11434" in environment_example
    assert "OPENROUTER_API_KEY=" in environment_example
    assert "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1" in environment_example
