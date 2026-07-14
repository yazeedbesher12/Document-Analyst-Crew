import json
import re

from greenloop_rag_crew.rag.build_chunks import build_chunks
from greenloop_rag_crew.rag.chunker import MAX_EMBEDDING_TOKENS, count_tokens, create_chunks
from greenloop_rag_crew.rag.document_registry import DOCUMENT_REGISTRY
from greenloop_rag_crew.rag.pdf_loader import extract_pages


CHUNK_ID_RE = re.compile(
    r"^(?:HR-HBK-2025-v1\.4|PRD-GLX1-2025-v2\.1|FIN-Q3-2025-v1\.0)_p\d{2,}_c\d{2}$"
)


def _normalized(text: str) -> str:
    return " ".join(text.split())


def test_chunks_are_page_bounded_unique_ordered_and_token_safe():
    pages = extract_pages("knowledge")
    chunks = create_chunks(pages)

    assert chunks
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    registry_order = {
        metadata.document_id: index for index, metadata in enumerate(DOCUMENT_REGISTRY)
    }
    assert chunks == sorted(
        chunks,
        key=lambda chunk: (
            registry_order[chunk.document_id],
            chunk.page,
            int(chunk.chunk_id.rsplit("_c", 1)[1]),
        ),
    )

    valid_pages = {(page.document_id, page.page): page.text for page in pages}
    for chunk in chunks:
        assert CHUNK_ID_RE.fullmatch(chunk.chunk_id)
        assert (chunk.document_id, chunk.page) in valid_pages
        assert f"_p{chunk.page:02d}_" in chunk.chunk_id
        assert f"Document ID: {chunk.document_id}" in chunk.embedding_text
        assert f"Page: {chunk.page}" in chunk.embedding_text
        assert chunk.token_count == count_tokens(chunk.embedding_text)
        assert chunk.token_count <= MAX_EMBEDDING_TOKENS


def test_chunk_generation_is_byte_deterministic(tmp_path):
    first = tmp_path / "chunks-first.jsonl"
    second = tmp_path / "chunks-second.jsonl"

    build_chunks(output=first)
    build_chunks(output=second)

    assert first.read_bytes() == second.read_bytes()
    records = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert records
    assert len({record["chunk_id"] for record in records}) == len(records)


def test_known_facts_remain_in_chunks_from_expected_pages():
    chunks = create_chunks(extract_pages("knowledge"))
    by_page = {}
    for chunk in chunks:
        by_page.setdefault((chunk.source, chunk.page), []).append(_normalized(chunk.text))

    def page_text(source: str, page: int) -> str:
        return " ".join(by_page[(source, page)])

    handbook_p6 = page_text("GreenLoop_Employee_Handbook_2025.pdf", 6)
    assert "software and AI employees may work remotely up to three days per week" in handbook_p6

    spec_p12 = page_text("GreenLoop_Sorter_X1_Product_Specification.pdf", 12)
    assert "Macro classification accuracy | 93.2%" in spec_p12

    report_p12 = page_text("GreenLoop_Q3_FY2025_Report.pdf", 12)
    assert "Macro average | 89.1% | 93.2%" in report_p12
    assert "increased from 87.9% in Q2 to 89.1% in Q3" in report_p12

    spec_p27 = page_text("GreenLoop_Sorter_X1_Product_Specification.pdf", 27)
    assert "99.5% availability target" in spec_p27

    report_p14 = page_text("GreenLoop_Q3_FY2025_Report.pdf", 14)
    assert "99.72% for the quarter" in report_p14
