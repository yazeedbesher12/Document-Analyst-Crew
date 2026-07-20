#!/bin/sh
set -eu

# The image contains PDFs and code only. Runtime storage is built inside the
# container so no host-created Chroma database is reused.
mkdir -p "$CHROMA_PERSIST_DIR" "$OUTPUT_DIR" "$HF_HOME" "$SENTENCE_TRANSFORMERS_HOME"

/app/.venv/bin/python -m greenloop_rag_crew.rag.build_chunks
/app/.venv/bin/python -m greenloop_rag_crew.rag.build_index

exec /app/.venv/bin/streamlit run /app/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true \
  --server.fileWatcherType=none
