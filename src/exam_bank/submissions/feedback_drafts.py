from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from exam_bank.submissions.models import Assignment, FeedbackDraft, Student, Submission, dataclass_to_json_dict


def build_acknowledgement_draft(assignment: Assignment, student: Student, submission: Submission) -> FeedbackDraft:
    return FeedbackDraft(
        draft_id=f"{assignment.assignment_id}:{student.student_id}:acknowledgement",
        assignment_id=assignment.assignment_id,
        student_id=student.student_id,
        draft_type="acknowledgement",
        recipient_email=student.email,
        subject=f"Draft acknowledgement: {assignment.title}",
        body_text=(
            f"Draft only. We received your submission for {assignment.title}: "
            f"{submission.source_filename}."
        ),
        send_allowed=False,
        created_at=datetime.now(timezone.utc),
    )


def build_resend_draft(assignment: Assignment, student: Student, submission: Submission) -> FeedbackDraft:
    reasons = ", ".join(submission.validation_reasons) or "validation_failed"
    return FeedbackDraft(
        draft_id=f"{assignment.assignment_id}:{student.student_id}:resend:{submission.submission_id}",
        assignment_id=assignment.assignment_id,
        student_id=student.student_id,
        draft_type="resend",
        recipient_email=student.email,
        subject=f"Draft resend request: {assignment.title}",
        body_text=(
            f"Draft only. Your submission for {assignment.title} needs review or resend. "
            f"Reason: {reasons}."
        ),
        send_allowed=False,
        created_at=datetime.now(timezone.utc),
    )


def build_missing_reminder_draft(assignment: Assignment, student: Student) -> FeedbackDraft:
    return FeedbackDraft(
        draft_id=f"{assignment.assignment_id}:{student.student_id}:missing_reminder",
        assignment_id=assignment.assignment_id,
        student_id=student.student_id,
        draft_type="missing_reminder",
        recipient_email=student.email,
        subject=f"Draft reminder: {assignment.title}",
        body_text=f"Draft only. Our records do not show a submission for {assignment.title}.",
        send_allowed=False,
        created_at=datetime.now(timezone.utc),
    )


def write_drafts_jsonl(drafts: list[FeedbackDraft], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for draft in drafts:
            handle.write(json.dumps(dataclass_to_json_dict(draft), sort_keys=True) + "\n")
