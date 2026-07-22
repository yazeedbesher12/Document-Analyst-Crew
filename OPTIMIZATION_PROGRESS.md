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

## Step 6: Single-Generation Fast Request Pipeline

Status: complete

- Before this change, the default path always constructed the sequential
  three-agent CrewAI crew. The Document Researcher task, Fact Checker task,
  and Report Writer task each required an LLM generation; the two retrieval
  agents could also make additional Agent iterations around `document_search`.
  `document_search` itself does not call an LLM. This explains the observed
  approximately 532-second researcher and 537-second fact-checker stages.
- `RAG_PIPELINE_MODE=fast` is now the default. It performs persistent-index
  preparation, hybrid Chroma/BM25 retrieval, deduplication, citation-metadata
  verification, and citation rendering deterministically, then makes exactly
  one direct Ollama `/api/chat` request. It does not construct a CrewAI agent,
  task, or tool loop.
- `RAG_PIPELINE_MODE=strict` requires `STRICT_LLM_VERIFICATION=true` and uses
  at most two direct requests: answer generation plus one strict citation
  review. `RAG_PIPELINE_MODE=legacy` retains the existing three-agent CrewAI
  workflow and its previous LLM-call behavior.
- The fast prompt is bounded by `RAG_TOP_K_VECTOR=4`, `RAG_TOP_K_BM25=4`,
  `RAG_FINAL_CONTEXT_CHUNKS=5`, and `RAG_MAX_CONTEXT_CHARS=12000`. Citation
  headers are never truncated. Exact-answer cache keys include normalized
  question, index signature, model, generation settings, mode, and pipeline
  version. The cache is enabled by default with `RAG_ANSWER_CACHE=true`.
- Direct Ollama controls are `LLM_TEMPERATURE=0.1`, `LLM_MAX_TOKENS=512`,
  `LLM_NUM_CTX=4096`, `OLLAMA_THINK=false`, and `OLLAMA_KEEP_ALIVE=30m`.
  Safe metrics record mode, retrieval count/duration, token counts when
  returned by Ollama, request count, model-load/prompt-evaluation/generation
  durations, cache hit/miss, and total request duration. No prompt, document,
  secret, or reasoning text is logged.

### Focused Verification

Command run:

```powershell
uv run pytest -q tests/test_fast_pipeline.py tests/test_ollama_client.py tests/test_question_execution.py tests/test_execution_timing.py tests/test_llm_config.py tests/test_hybrid_retriever.py
```

Result: `35 passed in 54.95s`.

The tests cover one-request fast mode, zero-request deterministic retrieval and
metadata verification, strict two-request ceiling, retained legacy mode,
absence of a CrewAI tool loop in fast mode, exact-cache hit and invalidation,
citation-ID containment, metadata-preserving truncation, and outgoing
`think=false`/`keep_alive` request payloads. `uv run python -m compileall -q
src streamlit_app.py` and `git diff --check` also passed.

### One Real Ollama Smoke Request

The diagnostic process explicitly selected local Ollama without changing the
Azure-oriented `.env` file. One short request completed with these safe
metrics:

- request count: `1`
- `think_requested=false`; returned thinking content: `false`
- keep-alive requested: `30m`
- input/output tokens: `21` / `2`
- model load: `106.870s`; prompt evaluation: `2.281s`; generation: `0.148s`
- total request: `111.410s`

`ollama ps` immediately afterward showed `qwen3:8b`, `6.0 GB`,
`61%/39% CPU/GPU`, context `4096`, resident for approximately 29 more minutes.
The remaining local latency bottleneck is cold model loading; subsequent
requests during the keep-alive window avoid that load cost.

## Step 7: Streamed Concise Answer Generation

Status: implementation complete; real Ollama smoke blocked

- Fast-mode answer generation now uses one streamed Ollama `/api/chat` request.
  Streamlit displays answer tokens through an updating placeholder as they
  arrive, retains the completed report in session state, and keeps the public
  Writing answer timer updating until the stream ends.
- The answer format is now `## Direct Answer`, `## Evidence` (at most four
  inline-cited bullets), and `## Limitation` only when evidence is incomplete.
  The former executive summary, repeated findings, and bibliography section
  are not generated or rendered. Output is deterministically bounded to 300
  words without splitting a citation-bearing line.
- Defaults are `LLM_MAX_TOKENS=400`, `LLM_TEMPERATURE=0.1`,
  `LLM_NUM_CTX=3072`, `OLLAMA_THINK=false`, `OLLAMA_KEEP_ALIVE=30m`,
  `RAG_FINAL_CONTEXT_CHUNKS=4`, and `RAG_MAX_CONTEXT_CHARS=7000`.
  The exact-answer cache pipeline version changed so stale long-form results
  cannot be reused.
- Safe metrics now include time to first token, total generation duration,
  generated token count, approximate tokens per second, total LLM requests,
  and whether Ollama reported the model as already loaded.

### Focused Verification

```powershell
uv run pytest -q tests/test_ollama_client.py tests/test_fast_pipeline.py tests/test_streamlit_progress.py tests/test_llm_config.py tests/test_question_execution.py
```

Result: `29 passed in 12.75s`.

The mocked coverage verifies `stream=true`, `think=false`, keep-alive and
generation options in the outgoing request; incremental token delivery;
completed-answer replacement; one fast-mode request; a 400-token request
limit; inline citation containment; no bibliography; four Evidence bullets;
and safe UI metrics. Compile and diff checks also passed.

### One Real Concise Smoke Request

The one permitted real request used the question "What were GreenLoop's Q3
FY2025 revenue and growth compared with Q2?" with the exact-answer cache
disabled for that process. Index/retrieval completed, but Ollama returned HTTP
500 before the first streamed token. It took `140.8s` wall-clock including
local retrieval/embedding initialization; no answer, first-token time,
generated token count, or tokens-per-second measurement is available.

No retry was issued. `ollama ps` after the failure showed no active model;
`ollama list` still showed `qwen3:8b` installed. The remaining blocker is the
local Ollama HTTP 500, not the index, retrieval, streaming client, or citation
rendering code.
