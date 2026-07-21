FROM ghcr.io/astral-sh/uv:0.11.28 AS uv

FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    KNOWLEDGE_DIR=/app/knowledge \
    CHROMA_PERSIST_DIR=/app/storage/chroma \
    OUTPUT_DIR=/app/output \
    HF_HOME=/app/storage/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/storage/cache/sentence_transformers \
    EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 \
    OLLAMA_THINK=false \
    LLM_TEMPERATURE=0.1 \
    LLM_MAX_TOKENS=900 \
    RAG_TOP_K=6

WORKDIR /app

# Lockfile-controlled production dependencies. Runtime secrets are passed with
# environment variables; .env files and API keys are intentionally never copied.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY streamlit_app.py docker-entrypoint.sh ./
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY knowledge ./knowledge

RUN uv sync --frozen --no-dev \
    && /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu', trust_remote_code=False)" \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app \
    && mkdir -p /app/output /app/storage/cache/huggingface /app/storage/cache/sentence_transformers \
    && chmod +x /app/docker-entrypoint.sh \
    && chown -R app:app /app

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["/app/docker-entrypoint.sh"]
