# Optimization Progress

## Step 1: Persistent Index and Retrieval Controls

Status: complete

- The persistent manifest records source PDF filenames and SHA-256 fingerprints,
  embedding model, chunking configuration, and index schema version.
- A matching manifest loads the current Chroma index with the
  `manifest_current` reason; changed documents, embedding configuration,
  chunk settings, or schema version trigger a rebuild path.
- Index lifecycle logging records whether the index was loaded or rebuilt and
  why. Chunk IDs and citation metadata are retained.
- CrewAI iteration limits are Document Researcher `4`, Fact Checker `4`, and
  Report Writer `2`.
- `RAG_TOP_K` defaults to `6` and accepts a bounded environment override.
  Hybrid vector plus BM25 retrieval remains in place, and repeated chunk IDs
  are deduplicated before evidence reaches an agent.

### Focused Verification

Command run:

```powershell
uv run pytest -q tests/test_retrieval_service.py::test_current_manifest_skips_chunking_and_chroma_rebuild tests/test_retrieval_service.py::test_changed_pdf_fingerprint_rebuilds_chunks_and_index tests/test_document_search_tool.py::test_rag_top_k_environment_default_is_respected tests/test_agent_factory.py::test_agents_use_conservative_explicit_iteration_limits tests/test_retrieval_service.py::test_retrieval_service_deduplicates_chunks_without_losing_citation_metadata
```

Result: `6 passed, 9 warnings in 13.25s`.

The warnings are existing CrewAI deprecations during agent construction. No
cache tests, benchmarks, Docker builds, PDF/index rebuilds, or live LLM calls
were run for this step.

## Step 2: Timing, Local LLM Controls, and Streamlit Progress

Status: complete

- `time.perf_counter` timing is recorded for request initialization, lazy
  embedding load, PDF extraction, chunk creation, index lifecycle, each
  retrieval, each sequential agent task, total Crew execution, and total
  request execution. Timing logs contain only stage names, durations, and
  safe counts.
- Local LLM controls are `LLM_TEMPERATURE=0.1`, `LLM_MAX_TOKENS=900`, and
  `OLLAMA_THINK=false` by default. The local model remains `qwen3:8b`.
- Streamlit now retains completed public stages with their elapsed time:
  Preparing document index, Researching documents, Verifying claims, Writing
  report, and Completed. It does not display prompts, retrieved text, hidden
  reasoning, or internal errors.
- CrewAI `1.15.2` uses its native OpenAI-compatible Ollama adapter in this
  environment. LiteLLM and the Ollama Python package are not installed and
  are not needed for this adapter.

### Focused Verification

Unit-test command run:

```powershell
uv run pytest -q tests/test_llm_config.py::test_latency_controls_are_read_from_the_environment tests/test_execution_timing.py tests/test_streamlit_progress.py
```

Result: `3 passed in 20.62s`.

One short local Ollama smoke request was run with `OLLAMA_THINK=false` and a
temporary `LLM_MAX_TOKENS=256`. It completed successfully in `94s` using
`OpenAICompatibleCompletion`; the configured `extra_body={"think": false}`
was present on the accepted request path. Prompt and response content were not
recorded. No RAG workflow, index rebuild, benchmark, Docker build, or full test
suite was run.

## Step 3: CPU-Only PyTorch Lock Verification

Status: complete

Timestamp: `2026-07-21T09:11:39.9568819+02:00`

- The existing CPU-only configuration was already correct, so no dependency
  resolution or package download was repeated. `uv lock --check` completed
  successfully.
- `pyproject.toml` selects the explicit `pytorch-cpu` index only for Linux:
  `https://download.pytorch.org/whl/cpu`. The lock resolves Linux to
  `torch 2.13.0+cpu`; Windows retains its normal platform-specific selection.
- Lock-file inspection confirms that `nvidia-*`, CUDA runtime, and Triton
  packages are absent from the Linux dependency graph.
- Runtime dependency verification succeeded: `torch=2.13.0+cpu`,
  `torch_cuda_available=False`, `sentence_transformers=5.6.0`,
  `chromadb=1.1.1`, and `crewai=1.15.2`.

### Complete Python Test Suite

Command run:

```powershell
uv run pytest -q
```

Result: `173 passed, 0 failed, 1 deselected, 210 warnings in 86.46s`.

The warnings are existing CrewAI deprecations. No benchmark, Docker build,
push, deployment, index rebuild, or additional test run was performed.

## Step 4: Cold/Warm Benchmark Attempt

Status: stopped after cold-run timeout

Timestamp: `2026-07-21T09:23:54.0494751+02:00`

The benchmark command started one clean Python process, cleared only in-memory
retrieval and embedding caches, retained the persistent Chroma index, and set
`OLLAMA_THINK=false`. The cold run did not complete after more than six minutes,
so it was stopped. The warm run was not started, as required by timeout handling.

| Measurement | Cold run | Warm run |
| --- | --- | --- |
| Application initialization | Unavailable: no completed result before timeout | Not run |
| Index load or rebuild | Unavailable from the terminated process; manifest remained unchanged | Not run |
| Retrieval time | Unavailable: no completed result before timeout | Not run |
| Researcher time | Unavailable: no completed result before timeout | Not run |
| Fact-checker time | Unavailable: no completed result before timeout | Not run |
| Report-writer time | Unavailable: no completed result before timeout | Not run |
| Total request time | `> 360s`, then terminated | Not run |
| LLM call count | Unavailable: Crew metrics were not finalized | Not run |
| PDFs reindexed | No. `index_manifest.json` remained at `2026-07-20T13:49:50.823094+00:00` | Not run |
| Embedding model loaded again | Unavailable from the terminated process | Not run |
| `OLLAMA_THINK=false` active | Yes, set explicitly for the benchmark process | Not run |

Safe timeout observations:

- No stage-marker output reached the benchmark console before termination, so
  the exact Crew task stage cannot be determined reliably.
- The final active-model snapshot was `qwen3:8b`, `6.0 GB`, `61%/39% CPU/GPU`,
  context `4096`, from `ollama ps`.
- The benchmark Python child processes were stopped; the Ollama service itself
  was left running. No benchmark retry, Docker build, index rebuild, or test
  run was performed.

## Step 5: Final Optimized Docker Build Validation

Status: blocked before build

Timestamp: `2026-07-21T09:38:24.7537560+02:00`

- The recorded complete Python suite from Step 3 remains the most recent full
  suite result: `173 passed, 0 failed, 1 deselected, 210 warnings in 86.46s`.
- The Step 4 benchmark result was confirmed without rerunning it: its cold run
  exceeded 360 seconds and the required warm run was not started.
- The Docker dependency preflight remains CPU-only: Linux resolves
  `torch 2.13.0+cpu` from the explicit PyTorch CPU index, and `uv.lock` has no
  `nvidia-*`, CUDA runtime, or Triton package entries. No dependency resolution
  was performed.
- The only code changed after the recorded full suite was the benchmark
  diagnostic. Minimum validation passed:

```powershell
uv run python -m compileall -q src\greenloop_rag_crew\diagnostics\latency_benchmark.py
```

Result: `compile-smoke: passed`.

- Docker CLI `29.6.1` is installed, but Docker Desktop's active
  `desktop-linux` context cannot reach its daemon. Exact error:

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

- Therefore `docker build -t greenloop-document-analyst:optimized .` was not
  attempted. Image size, previous-image comparison, in-container imports,
  CUDA verification, health check, and containerized end-to-end validation
  remain unavailable. No images or temporary containers were created or
  removed.
