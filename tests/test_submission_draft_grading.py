from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import fitz

from exam_bank.submissions.draft_grading import build_submission_draft_grades
from exam_bank.submissions.ingest import ingest_assignment_submissions
from exam_bank.submissions.review_queue import build_submission_review_queue


FIXTURES = Path("tests/fixtures/submissions")


def _copy_fixture_tree(tmp_path: Path) -> Path:
    target = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, target)
    return target


def _write_native_text_pdf(path: Path, text: str) -> None:
    path.unlink(missing_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _make_inbox_pdfs_native_text(fixtures: Path) -> None:
    _write_native_text_pdf(fixtures / "inbox" / "S0001.pdf", "Synthetic vector answer for S0001.")
    _write_native_text_pdf(fixtures / "inbox" / "S0002_vectors_hw1.pdf", "Synthetic vector answer for S0002.")


def _set_source_question_ids(fixtures: Path, question_ids: list[str]) -> None:
    assignment_path = fixtures / "assignment_p3_vectors_hw1.json"
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    assignment["source_question_ids"] = question_ids
    assignment_path.write_text(json.dumps(assignment, indent=2, sort_keys=True), encoding="utf-8")


def _run_phase2(fixtures: Path, tmp_path: Path) -> None:
    ingest_assignment_submissions(
        assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
        roster_path=fixtures / "roster_class_12a.csv",
        submissions_dir=fixtures / "inbox",
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )
    build_submission_review_queue(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_draft_grading_outputs_are_written_and_fail_closed_for_blank_pdfs(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    _run_phase2(fixtures, tmp_path)
    review_queue_path = tmp_path / "output" / "submissions" / "p3_vectors_hw1" / "review" / "review_queue.json"
    review_queue_before = review_queue_path.read_text(encoding="utf-8")

    result = build_submission_draft_grades(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        reviewed_rubrics_path=tmp_path / "missing_reviewed_rubrics.json",
        mark_events_path=tmp_path / "missing_mark_events.json",
    )

    extraction_path = Path(result["extraction_results"])
    draft_path = Path(result["draft_grading_results"])
    summary_path = Path(result["draft_grading_summary"])
    packet_path = Path(result["teacher_grading_review_packet"])
    csv_path = Path(result["draft_grading_summary_csv"])

    assert extraction_path.is_file()
    assert draft_path.is_file()
    assert summary_path.is_file()
    assert packet_path.is_file()
    assert csv_path.is_file()
    assert review_queue_path.read_text(encoding="utf-8") == review_queue_before

    extractions = _json(extraction_path)
    drafts = _json(draft_path)
    summary = _json(summary_path)

    assert len(extractions) == 2
    assert {record["status"] for record in extractions} == {"partial"}
    assert all(record["text_extractable"] is False for record in extractions)
    assert len(drafts) == 2
    assert all(record["grading_mode"] == "draft_auto" for record in drafts)
    assert all(record["teacher_review_required"] is True for record in drafts)
    assert all(record["student_facing"] is False for record in drafts)
    assert all(record["draft_score"] is None for record in drafts)
    assert all(record["confidence"] == "none" for record in drafts)
    assert summary["student_facing_count"] == 0
    assert summary["teacher_review_required_count"] == 2
    assert result["draft_scores_assigned"] == 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["student_facing"] for row in rows} == {"False"}
    assert "No student-facing feedback" in packet_path.read_text(encoding="utf-8")


def test_missing_reviewed_rubric_mapping_does_not_receive_scores(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    _make_inbox_pdfs_native_text(fixtures)
    _set_source_question_ids(fixtures, ["q1"])
    _run_phase2(fixtures, tmp_path)

    result = build_submission_draft_grades(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        reviewed_rubrics_path=tmp_path / "missing_reviewed_rubrics.json",
        mark_events_path=tmp_path / "missing_mark_events.json",
    )

    drafts = _json(Path(result["draft_grading_results"]))
    assert {record["confidence"] for record in drafts} == {"low"}
    assert {record["status"] for record in drafts} == {"needs_review"}
    assert all(record["draft_score"] is None for record in drafts)
    assert all("missing_reviewed_rubric_mapping" in record["confidence_reasons"] for record in drafts)


def test_advisory_mark_events_alone_do_not_receive_scores(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    _make_inbox_pdfs_native_text(fixtures)
    _set_source_question_ids(fixtures, ["q1"])
    _run_phase2(fixtures, tmp_path)
    mark_events_path = tmp_path / "mark_events.json"
    mark_events_path.write_text(json.dumps({"records": [{"question_id": "q1", "mark_events": []}]}), encoding="utf-8")

    result = build_submission_draft_grades(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        reviewed_rubrics_path=tmp_path / "missing_reviewed_rubrics.json",
        mark_events_path=mark_events_path,
    )

    drafts = _json(Path(result["draft_grading_results"]))
    assert all(record["draft_score"] is None for record in drafts)
    assert all("advisory_mark_events_not_scoring_contract" in record["confidence_reasons"] for record in drafts)


def test_reviewed_rubric_without_answer_mapping_stays_medium_confidence_null_score(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    _make_inbox_pdfs_native_text(fixtures)
    _set_source_question_ids(fixtures, ["q1"])
    _run_phase2(fixtures, tmp_path)
    reviewed_rubrics_path = tmp_path / "reviewed_rubrics.json"
    reviewed_rubrics_path.write_text(
        json.dumps(
            {
                "rubrics": [
                    {
                        "rubric_id": "rr_q1",
                        "source_question_id": "q1",
                        "total_marks": 5,
                        "reviewed_by": "teacher",
                        "reviewed_at": "2026-06-22T00:00:00Z",
                        "review_status": "approved",
                        "safe_for_auto_grade_lab": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_submission_draft_grades(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        reviewed_rubrics_path=reviewed_rubrics_path,
        mark_events_path=tmp_path / "missing_mark_events.json",
    )

    drafts = _json(Path(result["draft_grading_results"]))
    assert {record["confidence"] for record in drafts} == {"medium"}
    assert all(record["draft_score"] is None for record in drafts)
    assert all(record["draft_max_score"] == 5.0 for record in drafts)
    assert all("deterministic_student_answer_mapping_missing" in record["confidence_reasons"] for record in drafts)


def test_draft_grading_appends_phase3_audit_events(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    _run_phase2(fixtures, tmp_path)

    build_submission_draft_grades(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        reviewed_rubrics_path=tmp_path / "missing_reviewed_rubrics.json",
        mark_events_path=tmp_path / "missing_mark_events.json",
    )

    audit_path = tmp_path / "output" / "submissions" / "p3_vectors_hw1" / "audit.jsonl"
    event_types = [event["event_type"] for event in _jsonl(audit_path)]

    assert "draft_grading_started" in event_types
    assert "legacy_grading_salvage_audited" in event_types
    assert "submission_extraction_started" in event_types
    assert "submission_extraction_finished" in event_types
    assert "draft_grading_created" in event_types
    assert "draft_grading_skipped" in event_types
    assert "draft_grading_summary_written" in event_types
    assert "teacher_grading_review_packet_written" in event_types
    assert "draft_grading_finished" in event_types


def test_salvage_audit_report_documents_decisions() -> None:
    report = Path("reports/SUBMISSION_DRAFT_GRADING_SALVAGE_AUDIT_2026_06_22.md")
    text = report.read_text(encoding="utf-8")

    assert "src/exam_bank/auto_grade/schemas.py" in text
    assert "reuse_now" in text
    assert "advisory_mark_events_not_scoring_contract" in text


def test_phase3_modules_do_not_add_email_ocr_or_final_grade_behavior() -> None:
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "src/exam_bank/submissions/draft_grading.py",
            "src/exam_bank/submissions/draft_grading_cli.py",
            "src/exam_bank/submissions/extraction.py",
            "src/exam_bank/submissions/grading_packets.py",
        ]
    ).lower()

    for forbidden in [
        "smtplib",
        "imaplib",
        "poplib",
        "outlook",
        "gmail",
        "sendmail",
        "pytesseract",
        "final_score",
        "final_grade",
        "student_self_check",
    ]:
        assert forbidden not in source
