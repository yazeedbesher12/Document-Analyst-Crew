"""Page-aware PDF extraction for the GreenLoop knowledge pack."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from greenloop_rag_crew.rag.document_registry import (
    DocumentMetadata,
    validate_knowledge_pack,
)
from greenloop_rag_crew.rag.schemas import ExtractedPage
from greenloop_rag_crew.runtime_paths import knowledge_dir as configured_knowledge_dir

_DECORATIVE_LINE_RE = re.compile(r"^[\s._\-=\u2014\u2013]+$")
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d{1,3}$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^section\s+\d+[a-z]?$", re.IGNORECASE)
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTICOLUMN_GAP_RE = re.compile(r"\s{2,}")


@dataclass(frozen=True)
class _LineCandidate:
    text: str
    y0: float
    y1: float
    max_font_size: float
    span_count: int


def extract_pages(knowledge_dir: str | Path | None = None) -> list[ExtractedPage]:
    """Validate and extract all pages in registry order."""

    knowledge_path = Path(knowledge_dir) if knowledge_dir is not None else configured_knowledge_dir()
    registry = validate_knowledge_pack(knowledge_path)
    pages: list[ExtractedPage] = []

    for metadata in registry:
        pages.extend(extract_document_pages(knowledge_path / metadata.filename, metadata))

    return pages


def extract_document_pages(
    pdf_path: str | Path, metadata: DocumentMetadata
) -> list[ExtractedPage]:
    """Extract every page from one registered PDF without crossing page boundaries."""

    path = Path(pdf_path)
    extracted_pages: list[ExtractedPage] = []

    with pymupdf.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            text = _normalize_page_text(page.get_text("text", sort=True))
            section = _detect_section(page, page_index)
            extracted_pages.append(
                ExtractedPage(
                    source=metadata.filename,
                    document_id=metadata.document_id,
                    title=metadata.title,
                    page=page_index,
                    section=section,
                    text=text,
                )
            )

    return extracted_pages


def _normalize_page_text(raw_text: str) -> str:
    """Normalize page text while keeping rows and meaningful line breaks visible."""

    normalized_lines: list[str] = []
    previous_blank = False

    for raw_line in raw_text.replace("\u00a0", " ").splitlines():
        line = raw_line.strip()
        if not line or _DECORATIVE_LINE_RE.fullmatch(line):
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
                previous_blank = True
            continue

        line = _normalize_line(line)
        if not line:
            continue
        normalized_lines.append(line)
        previous_blank = False

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines).strip()


def _normalize_line(line: str) -> str:
    """Collapse broken whitespace and mark obvious table columns with separators."""

    if line.startswith(("\u2022", "*")):
        return _HORIZONTAL_SPACE_RE.sub(" ", line.replace("\u2022", "*")).strip()

    cells = [cell.strip() for cell in _MULTICOLUMN_GAP_RE.split(line) if cell.strip()]
    if len(cells) >= 2:
        return " | ".join(_HORIZONTAL_SPACE_RE.sub(" ", cell) for cell in cells)
    return _HORIZONTAL_SPACE_RE.sub(" ", line).strip()


def _detect_section(page: pymupdf.Page, page_number: int) -> str:
    """Detect a page section using visible text geometry, with deterministic fallback."""

    candidates = _extract_line_candidates(page)
    if not candidates:
        return f"Page {page_number}"

    for index, line in enumerate(candidates):
        if _SECTION_RE.fullmatch(line.text):
            for following in candidates[index + 1 : index + 5]:
                if _is_plausible_heading(following, page):
                    return following.text

    top_limit = page.rect.height * 0.35
    top_candidates = [
        line
        for line in candidates
        if line.y0 <= top_limit and _is_plausible_heading(line, page)
    ]
    if top_candidates:
        largest_size = max(line.max_font_size for line in top_candidates)
        largest = [
            line
            for line in top_candidates
            if abs(line.max_font_size - largest_size) <= 0.5
        ]
        return min(largest, key=lambda line: (line.y0, line.y1, line.text)).text

    return f"Page {page_number}"


def _extract_line_candidates(page: pymupdf.Page) -> list[_LineCandidate]:
    """Convert PyMuPDF text spans into simple heading candidates."""

    text_dict = page.get_text("dict", sort=True)
    lines: list[_LineCandidate] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            pieces = [span.get("text", "") for span in spans]
            text = _normalize_line(" ".join(piece for piece in pieces if piece).strip())
            if not text:
                continue
            bbox = line.get("bbox", (0, 0, 0, 0))
            max_font_size = max((float(span.get("size", 0)) for span in spans), default=0.0)
            lines.append(
                _LineCandidate(
                    text=text,
                    y0=float(bbox[1]),
                    y1=float(bbox[3]),
                    max_font_size=max_font_size,
                    span_count=len(spans),
                )
            )

    return lines


def _is_plausible_heading(candidate: _LineCandidate, page: pymupdf.Page) -> bool:
    """Reject page furniture, page numbers, and prose-like candidates."""

    text = candidate.text.strip()
    lowered = text.lower()
    word_count = len(re.findall(r"[A-Za-z0-9]+", text))

    if not text or len(text) > 90 or word_count > 12:
        return False
    if _PAGE_NUMBER_RE.fullmatch(text) or _SECTION_RE.fullmatch(text):
        return False
    if "greenloop robotics" in lowered:
        return False
    if "fictional internal document" in lowered:
        return False
    if "page " in lowered and candidate.y0 > page.rect.height * 0.75:
        return False
    if "|" in text and candidate.span_count > 1:
        return False
    if text.endswith(".") and word_count > 6:
        return False

    return True
