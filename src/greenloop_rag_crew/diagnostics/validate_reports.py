"""Validate the three tracked GreenLoop report deliverables without Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from greenloop_rag_crew.runtime_paths import resolve_configured_output_path


REQUIRED_HEADINGS = (
    "# GreenLoop Document Analysis",
    "## Executive Summary",
    "## Findings",
    "## Limitations and Undisclosed Information",
    "## Citations",
)
PLACEHOLDERS = ("TODO", "TBD", "[INSERT", "{{", "}}")


@dataclass(frozen=True)
class ReportRequirement:
    path: Path
    required_fragments: tuple[str, ...]
    source_filenames: tuple[str, ...]
    pages: tuple[str, ...]


REPORT_REQUIREMENTS = (
    ReportRequirement(
        path=Path("output/report_01_remote_work_and_revenue.md"),
        required_fragments=("three days", "$2.40M", "$3.06M", "27.5%"),
        source_filenames=(
            "GreenLoop_Employee_Handbook_2025.pdf",
            "GreenLoop_Q3_FY2025_Report.pdf",
        ),
        pages=("Page 6",),
    ),
    ReportRequirement(
        path=Path("output/report_02_accuracy_comparison.md"),
        required_fragments=("93.2%", "89.1%", "4.1", "87.9%", "1.2"),
        source_filenames=(
            "GreenLoop_Sorter_X1_Product_Specification.pdf",
            "GreenLoop_Q3_FY2025_Report.pdf",
        ),
        pages=("Page 12",),
    ),
    ReportRequirement(
        path=Path("output/report_03_sla_and_revenue_loss.md"),
        required_fragments=("99.5%", "99.72%", "0.22", "not disclosed"),
        source_filenames=(
            "GreenLoop_Sorter_X1_Product_Specification.pdf",
            "GreenLoop_Q3_FY2025_Report.pdf",
        ),
        pages=("Page 27", "Page 14", "Page 13"),
    ),
)


def validate_reports() -> dict[str, list[str]]:
    """Return validation errors grouped by report path."""

    failures: dict[str, list[str]] = {}
    for requirement in REPORT_REQUIREMENTS:
        errors = _validate_report(requirement)
        if errors:
            failures[str(requirement.path)] = errors
    return failures


def _validate_report(requirement: ReportRequirement) -> list[str]:
    path = resolve_configured_output_path(requirement.path)
    if not path.exists():
        return ["file is missing"]

    content = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if not content.strip():
        errors.append("file is empty")
        return errors

    for heading in REQUIRED_HEADINGS:
        if heading not in content:
            errors.append(f"missing heading: {heading}")
    for fragment in requirement.required_fragments:
        if fragment.casefold() not in content.casefold():
            errors.append(f"missing required value or statement: {fragment}")
    for source in requirement.source_filenames:
        if source not in content:
            errors.append(f"missing citation source: {source}")
    for page in requirement.pages:
        if page not in content:
            errors.append(f"missing citation page: {page}")
    if "Chunk ID" not in content:
        errors.append("missing citation chunk ID")
    for placeholder in PLACEHOLDERS:
        if placeholder.casefold() in content.casefold():
            errors.append(f"placeholder remains: {placeholder}")

    if path.name == "report_03_sla_and_revenue_loss.md" and "$" in content:
        errors.append("report must not state a dollar amount for multi-item revenue loss")
    return errors


def main() -> None:
    failures = validate_reports()
    if failures:
        print("Report validation: FAIL")
        for path, errors in failures.items():
            print(f"{path}:")
            for error in errors:
                print(f"  - {error}")
        raise SystemExit(1)

    print("Report validation: PASS")
    for requirement in REPORT_REQUIREMENTS:
        print(f"  - {requirement.path}")


if __name__ == "__main__":
    main()
