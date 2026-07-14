from pathlib import Path

from greenloop_rag_crew.rag.document_registry import DOCUMENT_REGISTRY, validate_knowledge_pack
from greenloop_rag_crew.rag.pdf_loader import extract_pages


def test_knowledge_pack_has_exact_required_pdfs_and_page_counts():
    registry = validate_knowledge_pack("knowledge")

    assert registry == DOCUMENT_REGISTRY
    assert len(registry) == 3
    assert [entry.expected_pages for entry in registry] == [22, 31, 22]
    assert sum(entry.expected_pages for entry in registry) == 75
    assert sorted(path.name for path in Path("knowledge").glob("*.pdf")) == sorted(
        entry.filename for entry in registry
    )


def test_extract_pages_are_one_based_and_complete():
    pages = extract_pages("knowledge")

    assert len(pages) == 75
    for page in pages:
        assert page.page >= 1
        assert page.source
        assert page.document_id
        assert page.title
        assert page.section
        assert page.text.strip()

    by_source = {}
    for page in pages:
        by_source.setdefault(page.source, []).append(page.page)

    for metadata in DOCUMENT_REGISTRY:
        assert by_source[metadata.filename] == list(range(1, metadata.expected_pages + 1))


def test_extracted_pages_keep_expected_page_specific_facts():
    pages = {
        (page.source, page.page): " ".join(page.text.split())
        for page in extract_pages("knowledge")
    }

    handbook_p6 = pages[("GreenLoop_Employee_Handbook_2025.pdf", 6)]
    assert "software and AI employees may work remotely up to three days per week" in handbook_p6

    spec_p12 = pages[("GreenLoop_Sorter_X1_Product_Specification.pdf", 12)]
    assert "Macro classification accuracy | 93.2%" in spec_p12

    report_p12 = pages[("GreenLoop_Q3_FY2025_Report.pdf", 12)]
    assert "Macro average | 89.1% | 93.2%" in report_p12
    assert "increased from 87.9% in Q2 to 89.1% in Q3" in report_p12

    spec_p27 = pages[("GreenLoop_Sorter_X1_Product_Specification.pdf", 27)]
    assert "99.5% availability target" in spec_p27

    report_p14 = pages[("GreenLoop_Q3_FY2025_Report.pdf", 14)]
    assert "99.72% for the quarter" in report_p14
