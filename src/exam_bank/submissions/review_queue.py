from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exam_bank.submissions.audit_log import AuditLog
from exam_bank.submissions.models import dataclass_to_json_dict


REVIEW_STATUSES = {"needs_review", "reviewed", "needs_resubmission", "accepted_for_grading", "excluded"}
REVIEW_PRIORITIES = {"normal", "high"}


@dataclass(frozen=True)
class SubmissionReviewRecord:
    review_id: str
    assignment_id: str
    student_id: str
    submission_id: str
    stored_pdf_path: str
    status: str
    priority: str
    review_reasons: list[str]
    teacher_notes: str
    manual_completion_status: str
    grading_result_id: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ManualGradingPrepRecord:
    grading_result_id: str
    assignment_id: str
    student_id: str
    submission_id: str
    grading_mode: str
    status: str
    score: float | None
    max_score: float | None
    rubric_id: str
    question_notes: list[str]
    teacher_notes: str
    review_required: bool
    reviewed_by: str
    reviewed_at: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ReviewQueueSummary:
    assignment_id: str
    accepted_count: int
    needs_review_count: int
    needs_resubmission_count: int
    accepted_for_grading_count: int
    excluded_count: int
    manual_grading_placeholders_count: int
    created_at: datetime


REVIEW_QUEUE_CSV_FIELDS = [
    "assignment_id",
    "student_id",
    "display_name",
    "email",
    "submission_id",
    "status",
    "priority",
    "manual_completion_status",
    "review_reasons",
    "stored_pdf_path",
    "teacher_notes",
    "grading_result_id",
    "grading_status",
    "score",
    "max_score",
    "review_required",
]

TEACHER_NOTES_TEMPLATE_FIELDS = [
    "assignment_id",
    "student_id",
    "submission_id",
    "review_status",
    "teacher_notes",
    "manual_completion_status",
    "needs_resubmission_reason",
    "manual_score",
    "max_score",
]


def _require_review_private_roots(submission_output_root: Path, reports_root: Path) -> None:
    output_parts = submission_output_root.parts
    reports_parts = reports_root.parts
    if len(output_parts) < 2 or output_parts[-2:] != ("output", "submissions"):
        raise ValueError("submission_output_root must end with output/submissions")
    if len(reports_parts) < 2 or reports_parts[-2:] != ("reports", "submissions"):
        raise ValueError("reports_root must end with reports/submissions")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_completion_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["student_id"]: row for row in rows}


def _review_reasons(submission: dict[str, Any], completion_row: dict[str, str] | None) -> list[str]:
    reasons = [
        "phase2_initial_teacher_review_required",
        "grading_not_started",
        "ocr_not_run",
        "score_not_assigned",
    ]
    if bool(submission.get("late")):
        reasons.append("late_submission")
    if completion_row and "duplicates_found" in completion_row.get("notes", ""):
        reasons.append("duplicate_submission_seen")
    return reasons


def create_review_record(
    submission: dict[str, Any],
    *,
    completion_row: dict[str, str] | None = None,
    created_at: datetime | None = None,
) -> SubmissionReviewRecord:
    timestamp = created_at or datetime.now(timezone.utc)
    reasons = _review_reasons(submission, completion_row)
    priority = "high" if "late_submission" in reasons or "duplicate_submission_seen" in reasons else "normal"
    status = "needs_review"
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review status: {status}")
    if priority not in REVIEW_PRIORITIES:
        raise ValueError(f"Invalid review priority: {priority}")
    assignment_id = str(submission["assignment_id"])
    student_id = str(submission["student_id"])
    submission_id = str(submission["submission_id"])
    return SubmissionReviewRecord(
        review_id=f"{assignment_id}:{student_id}:{submission_id}:review",
        assignment_id=assignment_id,
        student_id=student_id,
        submission_id=submission_id,
        stored_pdf_path=str(submission["stored_pdf_path"]),
        status=status,
        priority=priority,
        review_reasons=reasons,
        teacher_notes="",
        manual_completion_status="not_reviewed",
        grading_result_id=f"{assignment_id}:{student_id}:{submission_id}:manual_placeholder",
        created_at=timestamp,
        updated_at=timestamp,
    )


def create_grading_prep_record(review: SubmissionReviewRecord, *, created_at: datetime | None = None) -> ManualGradingPrepRecord:
    timestamp = created_at or datetime.now(timezone.utc)
    return ManualGradingPrepRecord(
        grading_result_id=review.grading_result_id,
        assignment_id=review.assignment_id,
        student_id=review.student_id,
        submission_id=review.submission_id,
        grading_mode="manual_placeholder",
        status="not_started",
        score=None,
        max_score=None,
        rubric_id="",
        question_notes=[],
        teacher_notes="",
        review_required=True,
        reviewed_by="",
        reviewed_at=None,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _write_json_list(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclass_to_json_dict(record) for record in records]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_object(path: Path, record: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclass_to_json_dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_review_csv(
    path: Path,
    reviews: list[SubmissionReviewRecord],
    grading_by_id: dict[str, ManualGradingPrepRecord],
    completion_rows: dict[str, dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_QUEUE_CSV_FIELDS)
        writer.writeheader()
        for review in reviews:
            grading = grading_by_id[review.grading_result_id]
            completion = completion_rows.get(review.student_id, {})
            writer.writerow(
                {
                    "assignment_id": review.assignment_id,
                    "student_id": review.student_id,
                    "display_name": completion.get("display_name", ""),
                    "email": completion.get("email", ""),
                    "submission_id": review.submission_id,
                    "status": review.status,
                    "priority": review.priority,
                    "manual_completion_status": review.manual_completion_status,
                    "review_reasons": ";".join(review.review_reasons),
                    "stored_pdf_path": review.stored_pdf_path,
                    "teacher_notes": review.teacher_notes,
                    "grading_result_id": review.grading_result_id,
                    "grading_status": grading.status,
                    "score": "" if grading.score is None else grading.score,
                    "max_score": "" if grading.max_score is None else grading.max_score,
                    "review_required": grading.review_required,
                }
            )


def _write_teacher_notes_template(path: Path, reviews: list[SubmissionReviewRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEACHER_NOTES_TEMPLATE_FIELDS)
        writer.writeheader()
        for review in reviews:
            writer.writerow(
                {
                    "assignment_id": review.assignment_id,
                    "student_id": review.student_id,
                    "submission_id": review.submission_id,
                    "review_status": review.status,
                    "teacher_notes": "",
                    "manual_completion_status": review.manual_completion_status,
                    "needs_resubmission_reason": "",
                    "manual_score": "",
                    "max_score": "",
                }
            )


def build_submission_review_queue(
    *,
    assignment_id: str,
    submission_output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
) -> dict[str, object]:
    _require_review_private_roots(submission_output_root, reports_root)
    assignment_output = submission_output_root / assignment_id
    manifest_path = assignment_output / "manifest.json"
    manifest = _load_json(manifest_path)
    audit = AuditLog(assignment_output / "audit.jsonl", assignment_id)
    audit.write("review_queue_started", status="started")

    completion_report = Path(str(manifest.get("completion_report") or reports_root / f"{assignment_id}_completion.csv"))
    completion_rows = _load_completion_rows(completion_report)
    created_at = datetime.now(timezone.utc)

    reviews: list[SubmissionReviewRecord] = []
    grading_prep: list[ManualGradingPrepRecord] = []
    for submission in manifest.get("accepted_submissions", []):
        review = create_review_record(submission, completion_row=completion_rows.get(str(submission.get("student_id", ""))), created_at=created_at)
        grading = create_grading_prep_record(review, created_at=created_at)
        reviews.append(review)
        grading_prep.append(grading)
        audit.write("review_record_created", student_id=review.student_id, status=review.status, reasons=review.review_reasons, submission_id=review.submission_id)
        audit.write("grading_prep_created", student_id=review.student_id, status=grading.status, submission_id=review.submission_id)

    summary = ReviewQueueSummary(
        assignment_id=assignment_id,
        accepted_count=len(manifest.get("accepted_submissions", [])),
        needs_review_count=sum(1 for review in reviews if review.status == "needs_review"),
        needs_resubmission_count=sum(1 for review in reviews if review.status == "needs_resubmission"),
        accepted_for_grading_count=sum(1 for review in reviews if review.status == "accepted_for_grading"),
        excluded_count=sum(1 for review in reviews if review.status == "excluded"),
        manual_grading_placeholders_count=len(grading_prep),
        created_at=created_at,
    )

    review_dir = assignment_output / "review"
    review_queue_path = review_dir / "review_queue.json"
    grading_prep_path = review_dir / "grading_prep.json"
    review_summary_path = review_dir / "review_summary.json"
    teacher_notes_template_path = review_dir / "teacher_review_notes_template.csv"
    review_csv_path = reports_root / f"{assignment_id}_review_queue.csv"

    _write_json_list(review_queue_path, reviews)
    _write_json_list(grading_prep_path, grading_prep)
    _write_json_object(review_summary_path, summary)
    audit.write("review_queue_written", status="written", path=review_queue_path.as_posix())
    audit.write("review_queue_written", status="written", path=grading_prep_path.as_posix())
    audit.write("review_queue_written", status="written", path=review_summary_path.as_posix())

    grading_by_id = {record.grading_result_id: record for record in grading_prep}
    _write_review_csv(review_csv_path, reviews, grading_by_id, completion_rows)
    audit.write("review_queue_csv_written", status="written", path=review_csv_path.as_posix())
    _write_teacher_notes_template(teacher_notes_template_path, reviews)
    audit.write("teacher_notes_template_written", status="written", path=teacher_notes_template_path.as_posix())
    audit.write("review_queue_finished", status="finished")

    return {
        "assignment_id": assignment_id,
        "review_queue": review_queue_path,
        "grading_prep": grading_prep_path,
        "review_summary": review_summary_path,
        "review_queue_csv": review_csv_path,
        "teacher_notes_template": teacher_notes_template_path,
        "summary": summary,
        "reviews": reviews,
        "grading_prep_records": grading_prep,
        "rejected_count": len(manifest.get("rejected_submissions", [])),
        "missing_count": int((manifest.get("counts") or {}).get("missing") or 0),
    }
