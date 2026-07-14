"""Deterministic page-aware chunking for future embedding."""

from __future__ import annotations

import re
from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from greenloop_rag_crew.rag.schemas import DocumentChunk, ExtractedPage

TOKENIZER_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_EMBEDDING_TOKENS = 384
MAX_CONTENT_TOKENS = 320
OVERLAP_TOKENS = 50

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_CHUNK_ID_RE = re.compile(r"^(?P<document_id>.+)_p(?P<page>\d{2,})_c(?P<chunk>\d{2})$")


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerBase:
    """Load only the tokenizer used by all-mpnet-base-v2, not the embedding model."""

    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def count_tokens(text: str) -> int:
    """Count tokenizer tokens the same way chunk validation does."""

    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=True))


def create_chunks(pages: list[ExtractedPage]) -> list[DocumentChunk]:
    """Create deterministic chunks, restarting chunk indexes on every page."""

    chunks: list[DocumentChunk] = []
    for page in pages:
        page_chunks = _chunk_page(page)
        chunks.extend(page_chunks)
    return chunks


def chunk_sort_key(chunk: DocumentChunk) -> tuple[str, int, int]:
    """Return a stable fallback sort key for already-created chunks."""

    match = _CHUNK_ID_RE.fullmatch(chunk.chunk_id)
    chunk_index = int(match.group("chunk")) if match else 0
    return chunk.document_id, chunk.page, chunk_index


def _chunk_page(page: ExtractedPage) -> list[DocumentChunk]:
    units = _split_text_units(page.text)
    contents = _pack_units(units)
    chunks: list[DocumentChunk] = []

    for chunk_index, content in enumerate(contents, start=1):
        chunk_id = f"{page.document_id}_p{page.page:02d}_c{chunk_index:02d}"
        content = _fit_content_to_embedding_budget(page, content)
        embedding_text = format_embedding_text(page, content)
        chunks.append(
            DocumentChunk(
                source=page.source,
                document_id=page.document_id,
                title=page.title,
                page=page.page,
                section=page.section,
                chunk_id=chunk_id,
                text=content,
                embedding_text=embedding_text,
                token_count=count_tokens(embedding_text),
            )
        )

    return chunks


def format_embedding_text(page: ExtractedPage, content: str) -> str:
    """Build the stable embedding text envelope used for token accounting."""

    return (
        f"Document: {page.title}\n"
        f"Document ID: {page.document_id}\n"
        f"Page: {page.page}\n"
        f"Section: {page.section}\n"
        "Content:\n"
        f"{content.strip()}"
    )


def _split_text_units(text: str) -> list[str]:
    """Split by paragraph, sentence, line, then tokenizer-safe fallback."""

    units: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

    for paragraph in paragraphs:
        if _content_token_count(paragraph) <= MAX_CONTENT_TOKENS:
            units.append(paragraph)
            continue

        for sentence in _split_sentences(paragraph):
            if _content_token_count(sentence) <= MAX_CONTENT_TOKENS:
                units.append(sentence)
                continue

            for line in [line.strip() for line in sentence.splitlines() if line.strip()]:
                if _content_token_count(line) <= MAX_CONTENT_TOKENS:
                    units.append(line)
                else:
                    units.extend(_token_windows(line, MAX_CONTENT_TOKENS, 0))

    return units or [text.strip()]


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(line) if part.strip()]
        sentences.extend(parts or [line])
    return sentences


def _pack_units(units: list[str]) -> list[str]:
    """Pack units into <=320-token content chunks with approximate overlap."""

    chunks: list[str] = []
    current_units: list[str] = []

    for unit in units:
        candidate_units = current_units + [unit]
        candidate = _join_units(candidate_units)
        if current_units and _content_token_count(candidate) > MAX_CONTENT_TOKENS:
            emitted = _join_units(current_units)
            chunks.append(emitted)
            overlap = _overlap_text(emitted, OVERLAP_TOKENS)
            current_units = [overlap, unit] if overlap else [unit]
            if _content_token_count(_join_units(current_units)) > MAX_CONTENT_TOKENS:
                current_units = [unit]
        else:
            current_units = candidate_units

    if current_units:
        chunks.append(_join_units(current_units))

    return chunks


def _fit_content_to_embedding_budget(page: ExtractedPage, content: str) -> str:
    """Trim content deterministically if metadata pushes the envelope over 384 tokens."""

    if count_tokens(format_embedding_text(page, content)) <= MAX_EMBEDDING_TOKENS:
        return content.strip()

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(content, add_special_tokens=False)
    low = 1
    high = len(token_ids)
    best = ""

    while low <= high:
        mid = (low + high) // 2
        decoded = tokenizer.decode(token_ids[:mid], skip_special_tokens=True).strip()
        if count_tokens(format_embedding_text(page, decoded)) <= MAX_EMBEDDING_TOKENS:
            best = decoded
            low = mid + 1
        else:
            high = mid - 1

    return best.strip() or content.strip()


def _token_windows(text: str, window_size: int, overlap: int) -> list[str]:
    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    windows: list[str] = []
    step = max(1, window_size - overlap)

    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + window_size]
        if not window:
            break
        windows.append(tokenizer.decode(window, skip_special_tokens=True).strip())

    return [window for window in windows if window]


def _overlap_text(text: str, token_count: int) -> str:
    """Return a suffix overlap without tokenizer-decoding away original casing."""

    text = text.strip()
    if _content_token_count(text) <= token_count:
        return text

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    suffix: list[str] = []
    for paragraph in reversed(paragraphs):
        candidate = "\n\n".join([paragraph] + suffix)
        if _content_token_count(candidate) <= token_count:
            suffix.insert(0, paragraph)
        elif not suffix:
            return _word_suffix(paragraph, token_count)
        else:
            break

    return "\n\n".join(suffix).strip()


def _word_suffix(text: str, token_count: int) -> str:
    words = text.split()
    suffix: list[str] = []
    for word in reversed(words):
        candidate = " ".join([word] + suffix)
        if _content_token_count(candidate) <= token_count:
            suffix.insert(0, word)
        elif suffix:
            break
    return " ".join(suffix).strip()


def _content_token_count(text: str) -> int:
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _join_units(units: list[str]) -> str:
    return "\n\n".join(unit.strip() for unit in units if unit.strip()).strip()
