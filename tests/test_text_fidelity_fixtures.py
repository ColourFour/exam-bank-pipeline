import json
from pathlib import Path

import pytest

from exam_bank.text_fidelity import build_fixture_report, load_fixture_manifest, render_markdown, score_fixture_record


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "text_fidelity" / "bad_text_records.json"


def test_load_fixture_manifest_validates_schema_and_record_count(tmp_path: Path) -> None:
    manifest = load_fixture_manifest(FIXTURE_PATH)
    assert manifest["record_count"] == len(manifest["records"])

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"schema_name": "wrong", "record_count": 0, "records": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected fixture schema"):
        load_fixture_manifest(bad_path)


def test_score_fixture_record_detects_structural_reason_codes() -> None:
    record = {
        "record_id": "sample_q05",
        "paper_id": "sample",
        "paper_family": "p1",
        "question_number": "5",
        "native_pdf_text_raw": "5 (a) Solve sin theta = 0. [2] (b) Hence find x. [3]",
        "ocr_text_raw": "5 (a) Solve sin θ = 0. [2] (b) Hence find x. [3]",
        "currently_selected_text": "Solve sin 0 = 0.",
        "expected_normalized_text_or_structural_expectations": {
            "type": "structural_expectations",
            "expectations": [
                "contains question number 5",
                "preserves theta symbols",
                "preserves subpart labels",
                "contains mark bracket [3]",
            ],
        },
        "failure_tags": ["math_notation", "greek_symbol"],
        "text_fidelity_status": "degraded",
        "text_fidelity_flags": ["math_text_corruption_detected"],
        "source_profile": "hybrid",
        "question_text_role": "untrusted_math_text",
        "question_text_trust": "low",
    }

    scored = score_fixture_record(record)
    codes = {issue["code"] for issue in scored["issues"]}
    assert scored["status"] == "fail"
    assert "missing_question_number" in codes
    assert "missing_mark_bracket" in codes
    assert "missing_specific_mark_bracket" in codes
    assert "missing_subpart_labels" in codes
    assert "selected_source_rejected_by_structural_checks" in codes
    assert "expected_structural_requirement_missing" in codes
    assert "math_symbol_loss" in scored["measurable_improvement_targets"]


def test_build_fixture_report_includes_counts_and_per_fixture_details() -> None:
    manifest = load_fixture_manifest(FIXTURE_PATH)
    report = build_fixture_report(manifest)

    assert report["schema_name"] == "exam_bank.text_fidelity.fixture_report"
    assert report["record_count"] == manifest["record_count"]
    assert len(report["records"]) == manifest["record_count"]
    assert sum(report["status_counts"].values()) == manifest["record_count"]
    assert report["failure_type_counts"]
    assert report["issue_code_counts"]
    assert any(record["issues"] for record in report["records"])
    assert "mark_bracket" in report["failure_type_counts"]
    assert report["normalized_advisory_candidates_included"] is False


def test_build_fixture_report_can_include_advisory_normalized_candidates() -> None:
    manifest = load_fixture_manifest(FIXTURE_PATH)
    report = build_fixture_report(manifest, include_normalized=True)

    assert report["normalized_advisory_candidates_included"] is True
    assert report["normalization_summary"]["measurable_improvement_count"] >= 10
    candidate = next(record for record in report["records"] if record["record_id"] == "35summer25_q04")
    assert candidate["selected_text_raw"] == "4 Find the exact coordinates of the stationary point of the curve with equation y = 3x^{3} ln x^{4}, for x20."
    assert candidate["native_pdf_text_raw"] == ""
    assert candidate["ocr_text_raw"].startswith("4 Find the exact coordinates")
    assert "for x > 0" in candidate["question_text_normalized"]
    assert "ln(x^{4})" in candidate["question_text_normalized"]
    assert candidate["normalization_is_advisory"] is True
    assert "inequality_notation_normalized" in candidate["normalization_flags"]
    assert candidate["normalization_confidence"] < 1.0


def test_render_markdown_contains_summary_and_per_fixture_table() -> None:
    manifest = load_fixture_manifest(FIXTURE_PATH)
    report = build_fixture_report(manifest)
    markdown = render_markdown(report)

    assert "# Text Fidelity Fixture Baseline" in markdown
    assert "## Summary" in markdown
    assert "| Record | Status | Issues | Measurable targets |" in markdown
    assert manifest["records"][0]["record_id"] in markdown


def test_render_markdown_describes_normalized_candidates_as_advisory() -> None:
    manifest = load_fixture_manifest(FIXTURE_PATH)
    report = build_fixture_report(manifest, include_normalized=True)
    markdown = render_markdown(report)

    assert "Advisory normalized candidates: included" in markdown
    assert "These normalized strings are candidates for review only" in markdown
    assert "| Record | Status | Issues | Measurable targets | Normalized classification | Flags | Confidence |" in markdown
