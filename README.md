# GreenLoop RAG Document Analyst Crew

This project answers GreenLoop document questions with a CrewAI workflow. It searches only the three PDFs in `knowledge/`; it does not use web search. Ollama is the local default, with an optional Azure runtime provider.

## Architecture

```text
PDFs -> page extraction -> deterministic chunks -> Chroma + BM25
     -> document_search -> Researcher -> Fact Checker -> Report Writer
```

The Crew runs these agents sequentially:

1. **Document Researcher** retrieves focused local evidence and records source, page, section, and chunk ID.
2. **Fact Checker** independently re-queries each key claim and produces verdicts: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `NOT_DISCLOSED`, or `RETRIEVAL_ERROR`.
3. **Report Writer** uses the Fact Checker's corrected claims to create a concise Markdown report.

`RETRIEVAL_ERROR` means the tool failed; it is never treated as `NOT_DISCLOSED`.

## Knowledge Pack

`knowledge/` must contain exactly these source files:

- `GreenLoop_Employee_Handbook_2025.pdf` (`HR-HBK-2025-v1.4`, 22 pages)
- `GreenLoop_Sorter_X1_Product_Specification.pdf` (`PRD-GLX1-2025-v2.1`, 31 pages)
- `GreenLoop_Q3_FY2025_Report.pdf` (`FIN-Q3-2025-v1.0`, 22 pages)

## Setup

Requirements:

- Python `>=3.10,<3.14`
- `uv`
- Ollama with `qwen3:8b` installed and serving at `http://localhost:11434`

Install the project dependencies:

```powershell
uv sync
```

Optional local settings are documented in `.env.example`. Do not commit a real `.env` file.

`LLM_PROVIDER=ollama` is the default and uses `OLLAMA_MODEL=qwen3:8b`. To use Azure at runtime, set `LLM_PROVIDER=azure` plus `AZURE_LLM_MODEL`, `AZURE_API_KEY`, and `AZURE_ENDPOINT`; `AZURE_API_VERSION` is optional. These settings are validated before a crew starts, and API keys are never logged.

## Build the Local Index

The checked-in storage artifacts already describe the current knowledge pack. Rebuild only after intentionally changing a PDF or the chunking/index configuration:

```powershell
uv run python -m greenloop_rag_crew.rag.build_chunks
uv run python -m greenloop_rag_crew.rag.build_index
```

`storage/chunks.jsonl` contains the page-aware chunks. `storage/index_manifest.json` records source hashes and prevents accidental use of a stale Chroma index.

## Run the Configured Reports

List the three configured questions:

```powershell
uv run python -m greenloop_rag_crew.main --list
```

Validate orchestration without calling Ollama:

```powershell
uv run python -m greenloop_rag_crew.main --all --dry-run
```

Run one official report at a time:

```powershell
uv run python -m greenloop_rag_crew.main --question-id remote_work_and_revenue
uv run python -m greenloop_rag_crew.main --question-id accuracy_comparison
uv run python -m greenloop_rag_crew.main --question-id sla_and_revenue_loss
```

The official deliverables are tracked under `output/`:

- `report_01_remote_work_and_revenue.md`
- `report_02_accuracy_comparison.md`
- `report_03_sla_and_revenue_loss.md`

Use `--overwrite` only when replacing an existing official report deliberately.

## Ask an Ad-Hoc Question

```powershell
uv run python manual_run.py
```

Each manual question receives a timestamped, unique Markdown file in `output/`. These ad-hoc reports stay ignored by Git.

## Streamlit Web Interface

Run the local web interface with:

```powershell
uv run streamlit run streamlit_app.py
```

Each submitted question creates a fresh CrewAI crew and a unique Markdown report in `output/`.

## Docker Preparation

The Docker image packages the three PDFs from `knowledge/` with the application source and JSONC configuration. It deliberately does not include the host Chroma database: the container creates its own chunks and Chroma index from those packaged PDFs at startup. The `sentence-transformers/all-mpnet-base-v2` cache is prepared during the image build so startup does not need to download it again.

Generated reports and model/index caches are container-local unless you mount volumes for `output/` and `storage/`. Ollama and `qwen3:8b` are not included in the image; run Ollama separately and provide its reachable URL at runtime, for example `OLLAMA_BASE_URL=http://host.docker.internal:11434` on supported Docker Desktop installations.

The image runs the existing Streamlit entrypoint on port `8501`. Copy `docker.env.example` to a local untracked environment file before supplying any runtime secret. A local Compose run uses the host Ollama service without packaging it:

```powershell
docker compose up --build
```

For Azure, pass `AZURE_API_KEY`, `AZURE_ENDPOINT`, `AZURE_LLM_MODEL`, and optional `AZURE_API_VERSION` only at runtime. No API key or `.env` file is copied into an image layer.

## Validate Deliverables

```powershell
uv run python -m greenloop_rag_crew.diagnostics.validate_reports
```

This validator does not call Ollama. It checks that all three official reports exist, include the expected Markdown structure, contain required values and citations, state the Q3 revenue-loss limitation correctly, and contain no TODO-style placeholders.

## Tests and Diagnostics

Run focused tests with `uv run pytest -q <test paths>`. The repository excludes real Ollama tests from the default suite.

Useful checks:

```powershell
uv run python -m greenloop_rag_crew.diagnostics.agents_check
uv run python -m greenloop_rag_crew.diagnostics.crew_check
uv run python -m greenloop_rag_crew.diagnostics.runner_check
uv run python -m greenloop_rag_crew.diagnostics.ollama_tool_check
```
