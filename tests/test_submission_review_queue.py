from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from exam_bank.submissions.ingest import ingest_assignment_submissions
from exam_bank.submissions.review_queue import build_submission_review_queue, create_review_record


FIXTURES = Path("tests/fixtures/submissions")


def _copy_fixture_tree(tmp_path: Path) -> Path:
    target = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, target)
    return target


def _ingest(fixtures: Path, tmp_path: Path, *, received_at_override: datetime | None = None) -> dict[str, object]:
    return ingest_assignment_submissions(
        assignment_path=fixtures / "assignment_p3_vectors_hw1.json",
        roster_path=fixtures / "roster_class_12a.csv",
        submissions_dir=fixtures / "inbox",
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        received_at_override=received_at_override,
    )


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_create_review_record_from_accepted_submission() -> None:
    submission = {
        "assignment_id": "p3_vectors_hw1",
        "student_id": "S0001",
        "submission_id": "p3_vectors_hw1:S0001.pdf:abc123",
        "stored_pdf_path": "output/submissions/p3_vectors_hw1/accepted_pdfs/S0001.pdf",
        "late": False,
    }

    review = create_review_record(submission)

    assert review.status == "needs_review"
    assert review.priority == "normal"
    assert review.teacher_notes == ""
    assert review.manual_completion_status == "not_reviewed"
    assert review.grading_result_id.endswith(":manual_placeholder")
    assert review.review_reasons == [
        "phase2_initial_teacher_review_required",
        "grading_not_started",
        "ocr_not_run",
        "score_not_assigned",
    ]


def test_review_queue_artifacts_are_written_for_accepted_submissions(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    _ingest(fixtures, tmp_path)

    result = build_submission_review_queue(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    assert Path(result["review_queue"]).is_file()
    assert Path(result["grading_prep"]).is_file()
    assert Path(result["review_summary"]).is_file()
    assert Path(result["review_queue_csv"]).is_file()
    assert Path(result["teacher_notes_template"]).is_file()

    review_queue = _json(Path(result["review_queue"]))
    grading_prep = _json(Path(result["grading_prep"]))
    summary = _json(Path(result["review_summary"]))

    assert len(review_queue) == 2
    assert len(grading_prep) == 2
    assert summary["accepted_count"] == 2
    assert summary["needs_review_count"] == 2
    assert summary["manual_grading_placeholders_count"] == 2
    assert result["rejected_count"] == 3
    assert result["missing_count"] == 1


def test_rejected_and_missing_submissions_do_not_get_grading_prep(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    ingest_result = _ingest(fixtures, tmp_path)
    result = build_submission_review_queue(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    rejected_student_ids = {submission.student_id for submission in ingest_result["rejected"] if submission.student_id}
    missing_student_ids = {row.student_id for row in ingest_result["completion_rows"] if row.status == "missing"}
    grading_prep = _json(Path(result["grading_prep"]))

    grading_student_ids = {record["student_id"] for record in grading_prep}
    assert not (rejected_student_ids - {"S0002"}) & grading_student_ids
    assert not missing_student_ids & grading_student_ids
    assert grading_student_ids == {"S0001", "S0002"}


def test_late_accepted_submission_gets_late_review_reason(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    _ingest(fixtures, tmp_path, received_at_override=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc))
    result = build_submission_review_queue(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    review_queue = _json(Path(result["review_queue"]))

    assert all("late_submission" in record["review_reasons"] for record in review_queue)
    assert all(record["priority"] == "high" for record in review_queue)


def test_duplicate_winner_gets_duplicate_review_reason(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    inbox = fixtures / "inbox"
    shutil.copy2(inbox / "S0001.pdf", inbox / "S0001_vectors_hw1.pdf")
    older = datetime(2026, 6, 29, 8, 0, tzinfo=timezone.utc).timestamp()
    newer = datetime(2026, 6, 29, 9, 0, tzinfo=timezone.utc).timestamp()
    os.utime(inbox / "S0001.pdf", (older, older))
    os.utime(inbox / "S0001_vectors_hw1.pdf", (newer, newer))
    _ingest(fixtures, tmp_path)

    result = build_submission_review_queue(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )
    review_by_student = {record["student_id"]: record for record in _json(Path(result["review_queue"]))}

    assert "duplicate_submission_seen" in review_by_student["S0001"]["review_reasons"]
    assert review_by_student["S0001"]["priority"] == "high"


def test_review_queue_csv_and_teacher_notes_template_are_teacher_facing(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    _ingest(fixtures, tmp_path)
    result = build_submission_review_queue(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    with Path(result["review_queue_csv"]).open("r", encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    with Path(result["teacher_notes_template"]).open("r", encoding="utf-8", newline="") as handle:
        notes_rows = list(csv.DictReader(handle))

    assert len(review_rows) == 2
    assert len(notes_rows) == 2
    assert review_rows[0]["score"] == ""
    assert review_rows[0]["max_score"] == ""
    assert review_rows[0]["review_required"] == "True"
    assert notes_rows[0]["manual_score"] == ""
    assert notes_rows[0]["max_score"] == ""
    assert notes_rows[0]["teacher_notes"] == ""


def test_review_queue_appends_phase2_audit_events(tmp_path: Path) -> None:
    fixtures = _copy_fixture_tree(tmp_path)
    ingest_result = _ingest(fixtures, tmp_path)
    build_submission_review_queue(
        assignment_id="p3_vectors_hw1",
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    event_types = [event["event_type"] for event in _jsonl(Path(ingest_result["audit_log"]))]

    assert "review_queue_started" in event_types
    assert "review_record_created" in event_types
    assert "grading_prep_created" in event_types
    assert "review_queue_written" in event_types
    assert "review_queue_csv_written" in event_types
    assert "teacher_notes_template_written" in event_types
    assert "review_queue_finished" in event_types
