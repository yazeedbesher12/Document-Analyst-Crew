"""Registry and validation for the fixed GreenLoop knowledge pack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class DocumentMetadata:
    """Stable metadata for one required source PDF."""

    filename: str
    document_id: str
    title: str
    expected_pages: int


DOCUMENT_REGISTRY: tuple[DocumentMetadata, ...] = (
    DocumentMetadata(
        filename="GreenLoop_Employee_Handbook_2025.pdf",
        document_id="HR-HBK-2025-v1.4",
        title="GreenLoop Employee Handbook 2025",
        expected_pages=22,
    ),
    DocumentMetadata(
        filename="GreenLoop_Sorter_X1_Product_Specification.pdf",
        document_id="PRD-GLX1-2025-v2.1",
        title="GreenLoop Sorter X1 Product Specification",
        expected_pages=31,
    ),
    DocumentMetadata(
        filename="GreenLoop_Q3_FY2025_Report.pdf",
        document_id="FIN-Q3-2025-v1.0",
        title="GreenLoop Q3 FY2025 Report",
        expected_pages=22,
    ),
)

_REGISTRY_BY_FILENAME = {doc.filename: doc for doc in DOCUMENT_REGISTRY}


def get_document_metadata(filename: str) -> DocumentMetadata:
    """Return registry metadata for a known PDF filename."""

    try:
        return _REGISTRY_BY_FILENAME[filename]
    except KeyError as exc:
        expected = ", ".join(_REGISTRY_BY_FILENAME)
        raise ValueError(f"Unknown PDF {filename!r}. Expected one of: {expected}") from exc


def validate_knowledge_pack(knowledge_dir: str | Path) -> tuple[DocumentMetadata, ...]:
    """Validate that the knowledge directory exactly matches the registry."""

    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {knowledge_path}")
    if not knowledge_path.is_dir():
        raise NotADirectoryError(f"Knowledge path is not a directory: {knowledge_path}")

    found = {path.name for path in knowledge_path.glob("*.pdf")}
    expected = set(_REGISTRY_BY_FILENAME)

    missing = sorted(expected - found)
    unexpected = sorted(found - expected)

    errors: list[str] = []
    if missing:
        errors.append("missing required PDF(s): " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected PDF(s): " + ", ".join(unexpected))

    for metadata in DOCUMENT_REGISTRY:
        path = knowledge_path / metadata.filename
        if not path.exists():
            continue
        try:
            with pymupdf.open(path) as document:
                actual_pages = document.page_count
        except Exception as exc:  # pragma: no cover - defensive error context
            errors.append(f"{metadata.filename}: could not be opened by PyMuPDF ({exc})")
            continue
        if actual_pages != metadata.expected_pages:
            errors.append(
                f"{metadata.filename}: expected {metadata.expected_pages} pages, "
                f"found {actual_pages}"
            )

    if errors:
        raise ValueError("Invalid GreenLoop knowledge pack: " + "; ".join(errors))

    return DOCUMENT_REGISTRY
