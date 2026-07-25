FROM ghcr.io/astral-sh/uv:0.11.28 AS uv

FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    KNOWLEDGE_DIR=/app/knowledge \
    CHROMA_PERSIST_DIR=/app/storage/chroma \
    OUTPUT_DIR=/app/output \
    HF_HOME=/app/storage/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/storage/cache/sentence_transformers \
    HF_XET_HIGH_PERFORMANCE=1 \
    EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 \
    OLLAMA_THINK=false \
    OLLAMA_KEEP_ALIVE=30m \
    LLM_TEMPERATURE=0.1 \
    LLM_MAX_TOKENS=400 \
    LLM_NUM_CTX=3072 \
    RAG_PIPELINE_MODE=legacy \
    STRICT_LLM_VERIFICATION=false \
    RAG_ANSWER_CACHE=true \
    RAG_TOP_K_VECTOR=4 \
    RAG_TOP_K_BM25=4 \
    RAG_FINAL_CONTEXT_CHUNKS=4 \
    RAG_MAX_CONTEXT_CHARS=7000 \
    RAG_ANSWER_CACHE_DIR=/app/storage/answer_cache

WORKDIR /app

# Create the non-root application user.
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app

# Install dependencies in a cacheable layer.
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# Prepare model cache directories.
RUN install -d -o app -g app \
        /app/storage/cache/huggingface \
        /app/storage/cache/sentence_transformers

# Download the embedding model in an independent cached layer.
USER app

RUN python -c \
    "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'], device='cpu', trust_remote_code=False)"

# Copy application files only after downloading the model.
USER root

COPY --chown=app:app src ./src
COPY --chown=app:app streamlit_app.py docker-entrypoint.sh ./
COPY --chown=app:app .streamlit/config.toml ./.streamlit/config.toml
COPY --chown=app:app knowledge ./knowledge

# Install the project and create writable runtime directories.
RUN uv sync --frozen --no-dev \
    && install -d -o app -g app \
        /app/output \
        /app/storage/chroma \
        /app/storage/answer_cache \
    && chmod +x /app/docker-entrypoint.sh

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["/app/docker-entrypoint.sh"]