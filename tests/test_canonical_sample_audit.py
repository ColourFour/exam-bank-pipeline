from __future__ import annotations

from exam_bank.canonical_sample_audit import audit_canonical_sample


def _row(year: int, paper: str, component: str, *, text: bool = True, topic: bool = True) -> dict[str, object]:
    row: dict[str, object] = {
        "board": "CAIE",
        "component": component,
        "id": f"caie-9709-{paper}-s{str(year)[-2:]}-{component}",
        "mark_scheme_url": f"https://example.test/9709_s{str(year)[-2:]}_ms_{component}.pdf",
        "paper": paper,
        "paper_name": paper.replace("_", " ").title(),
        "qualification": "A Level",
        "question_paper_url": f"https://example.test/9709_s{str(year)[-2:]}_qp_{component}.pdf",
        "session": "May/June",
        "session_code": "s",
        "source": "fixture",
        "source_page": f"https://example.test/{year}/s",
        "subject": "Mathematics",
        "syllabus": "9709",
        "variant": None,
        "year": year,
    }
    if text:
        row["question_text"] = "Solve the equation."
    if topic:
        row["topic_routing_topic_ids"] = ["9709_p1_topic_algebra"]
    return row


def test_audit_flags_missing_text_topic_and_mark_scheme_without_rewriting() -> None:
    rows = [_row(2008, "pure_math_1", "1", text=False, topic=False)]
    rows[0]["mark_scheme_url"] = ""

    audit = audit_canonical_sample(rows, year_start=2008, year_end=2008)

    assert audit["summary"]["total_items_checked"] == 1
    assert audit["summary"]["valid_items"] == 0
    assert audit["summary"]["invalid_items"] == 1
    assert audit["issue_counts"]["missing_question_text"] == 1
    assert audit["issue_counts"]["missing_topic_fields"] == 1
    assert audit["issue_counts"]["missing_mark_scheme"] == 1
    assert rows[0]["mark_scheme_url"] == ""


def test_audit_reports_coverage_and_duplicate_year_type_violations() -> None:
    rows = [
        _row(2008, "pure_math_1", "1"),
        _row(2008, "pure_math_1", "1"),
        _row(2008, "pure_math_3", "3"),
        _row(2008, "mechanics_1", "4"),
    ]

    audit = audit_canonical_sample(rows, year_start=2008, year_end=2008)

    assert audit["coverage_matrix"]["missing_count"] == 2
    assert audit["coverage_matrix"]["years"]["2008"]["S1"]["status"] == "missing"
    assert audit["coverage_matrix"]["years"]["2008"]["P1"]["count"] == 2
    assert audit["issue_counts"]["duplicate_year_paper_type"] == 2
    assert audit["issue_counts"]["duplicate_id"] == 2
