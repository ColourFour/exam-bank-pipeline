from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from exam_bank.submissions.ingest import ingest_assignment_submissions


FIXTURES = Path("tests/fixtures/submissions")


def _copy_fixture_tree(tmp_path: Path) -> Path:
    target = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, target)
    return target


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_ingest_local_submission_folder_writes_outputs(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    result = ingest_assignment_submissions(
        assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
        roster_path=fixtures / "roster_class_12a.csv",
        submissions_dir=fixtures / "inbox",
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    accepted = result["accepted"]
    rejected = result["rejected"]
    rows = result["completion_rows"]

    assert {submission.student_id for submission in accepted} == {"S0001", "S0002"}
    assert any("unknown_student" in submission.validation_reasons for submission in rejected)
    assert any("not_pdf" in submission.validation_reasons for submission in rejected)
    assert {row.student_id: row.status for row in rows} == {
        "S0001": "submitted",
        "S0002": "submitted",
        "S0003": "missing",
    }
    assert Path(result["manifest"]).is_file()
    assert Path(result["audit_log"]).is_file()
    assert Path(result["completion_report"]).is_file()
    assert (tmp_path / "output" / "submissions" / "p3_vectors_hw1" / "accepted_pdfs" / "S0001.pdf").is_file()
    assert (tmp_path / "output" / "submissions" / "p3_vectors_hw1" / "rejected_pdfs" / "not_a_pdf.txt").is_file()


def test_completion_csv_and_audit_jsonl_are_written(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    result = ingest_assignment_submissions(
        assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
        roster_path=fixtures / "roster_class_12a.csv",
        submissions_dir=fixtures / "inbox",
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    with Path(result["completion_report"]).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[2]["student_id"] == "S0003"
    assert rows[2]["status"] == "missing"

    event_types = [event["event_type"] for event in _jsonl(Path(result["audit_log"]))]
    assert "assignment_loaded" in event_types
    assert "roster_loaded" in event_types
    assert "file_seen" in event_types
    assert "file_rejected" in event_types
    assert "file_accepted" in event_types
    assert "completion_report_written" in event_types
    assert "feedback_draft_created" in event_types
    assert "ingest_finished" in event_types


def test_draft_messages_are_draft_only_and_cover_ack_resend_reminder(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    result = ingest_assignment_submissions(
        assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
        roster_path=fixtures / "roster_class_12a.csv",
        submissions_dir=fixtures / "inbox",
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )
    draft_paths = result["draft_paths"]

    acknowledgement = _jsonl(draft_paths["acknowledgement"])
    resend = _jsonl(draft_paths["resend"])
    reminder = _jsonl(draft_paths["reminder"])

    assert {draft["draft_type"] for draft in acknowledgement} == {"acknowledgement"}
    assert {draft["draft_type"] for draft in resend} == {"resend"}
    assert {draft["draft_type"] for draft in reminder} == {"missing_reminder"}
    assert all(draft["send_allowed"] is False for draft in acknowledgement + resend + reminder)


def test_duplicate_submissions_keep_newest_and_reject_others(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    inbox = fixtures / "inbox"
    shutil.copy2(inbox / "S0001.pdf", inbox / "S0001_vectors_hw1.pdf")
    older = datetime(2026, 6, 29, 8, 0, tzinfo=timezone.utc).timestamp()
    newer = datetime(2026, 6, 29, 9, 0, tzinfo=timezone.utc).timestamp()
    (inbox / "S0001.pdf").touch()
    (inbox / "S0001_vectors_hw1.pdf").touch()

    os.utime(inbox / "S0001.pdf", (older, older))
    os.utime(inbox / "S0001_vectors_hw1.pdf", (newer, newer))

    result = ingest_assignment_submissions(
        assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
        roster_path=fixtures / "roster_class_12a.csv",
        submissions_dir=inbox,
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    row_by_student = {row.student_id: row for row in result["completion_rows"]}
    assert row_by_student["S0001"].source_filename == "S0001_vectors_hw1.pdf"
    assert row_by_student["S0001"].notes == "duplicates_found"
    assert any("duplicate_submission" in submission.validation_reasons for submission in result["rejected"])
    assert "duplicate_detected" in [event["event_type"] for event in _jsonl(Path(result["audit_log"]))]


def test_late_submissions_are_marked_and_can_be_rejected_when_not_allowed(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    late_received_at = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    result = ingest_assignment_submissions(
        assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
        roster_path=fixtures / "roster_class_12a.csv",
        submissions_dir=fixtures / "inbox",
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        received_at_override=late_received_at,
    )

    row_by_student = {row.student_id: row for row in result["completion_rows"]}
    assert row_by_student["S0001"].status == "late"
    assert row_by_student["S0001"].late is True

    assignment_data = json.loads((fixtures / "assignment_p3_vectors_hw1.json").read_text(encoding="utf-8"))
    assignment_data["allow_late"] = False
    no_late_assignment = fixtures / "assignment_no_late.json"
    no_late_assignment.write_text(json.dumps(assignment_data), encoding="utf-8")
    no_late_result = ingest_assignment_submissions(
        assignment_path=no_late_assignment,
        roster_path=fixtures / "roster_class_12a.csv",
        submissions_dir=fixtures / "inbox",
        output_root=tmp_path / "no_late" / "output" / "submissions",
        reports_root=tmp_path / "no_late" / "reports" / "submissions",
        received_at_override=late_received_at,
    )

    assert all("late_not_allowed" in submission.validation_reasons for submission in no_late_result["rejected"] if submission.student_id)


def test_phase_1_has_no_email_ocr_or_grading_behavior() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/exam_bank/submissions").glob("*.py"))

    for forbidden in ["smtplib", "imaplib", "poplib", "outlook", "gmail", "sendmail", "pytesseract", "grade_submission"]:
        assert forbidden not in source.lower()


def test_private_outputs_must_stay_under_submission_roots(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)

    with pytest.raises(ValueError, match="output/submissions"):
        ingest_assignment_submissions(
            assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
            roster_path=fixtures / "roster_class_12a.csv",
            submissions_dir=fixtures / "inbox",
            output_root=tmp_path / "unsafe_output",
            reports_root=tmp_path / "reports" / "submissions",
        )

    with pytest.raises(ValueError, match="reports/submissions"):
        ingest_assignment_submissions(
            assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
            roster_path=fixtures / "roster_class_12a.csv",
            submissions_dir=fixtures / "inbox",
            output_root=tmp_path / "output" / "submissions",
            reports_root=tmp_path / "unsafe_reports",
        )
