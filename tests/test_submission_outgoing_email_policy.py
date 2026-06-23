from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from exam_bank.submissions.outgoing_email import (
    OutgoingEmailDraft,
    build_outgoing_email_queue,
    normalize_outgoing_email_draft,
)


ASSIGNMENT_ID = "p3_vectors_hw1"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_approval_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "draft_id",
        "assignment_id",
        "student_id",
        "recipient_email",
        "message_type",
        "subject",
        "approval_status",
        "approved_by",
        "teacher_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_draft(*, draft_id: str = "draft-safe", message_type: str = "submission_acknowledgement") -> dict[str, object]:
    return {
        "draft_id": draft_id,
        "assignment_id": ASSIGNMENT_ID,
        "student_id": "S0001",
        "recipient_email": "student.one@example.invalid",
        "message_type": message_type,
        "subject": "Draft acknowledgement",
        "body_text": "Draft only.",
        "send_allowed": False,
        "created_at": "2026-06-22T00:00:00+00:00",
    }


def _approval_for(draft: dict[str, object], **overrides: str) -> dict[str, str]:
    row = {
        "draft_id": str(draft["draft_id"]),
        "assignment_id": str(draft["assignment_id"]),
        "student_id": str(draft["student_id"]),
        "recipient_email": str(draft["recipient_email"]),
        "message_type": str(draft["message_type"]),
        "subject": str(draft["subject"]),
        "approval_status": "approved",
        "approved_by": "teacher@example.invalid",
        "teacher_note": "",
    }
    row.update(overrides)
    return row


def test_outgoing_models_reject_send_allowed_without_approval() -> None:
    with pytest.raises(ValueError, match="send_allowed=true"):
        OutgoingEmailDraft(
            draft_id="draft-1",
            assignment_id=ASSIGNMENT_ID,
            student_id="S0001",
            recipient_email="student.one@example.invalid",
            subject="Draft",
            body_text="Draft only.",
            message_type="submission_acknowledgement",
            source_phase="phase1_submission_tracker",
            send_allowed=True,
            approval_status="draft",
            approved_by="",
            approved_at="",
            blocked_reasons=[],
            created_at="",
        )


def test_blocked_message_types_cannot_be_queued_even_when_approved(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    blocked_draft = _base_draft(draft_id="draft-final-grade", message_type="final_grade")
    _write_jsonl(output_root / ASSIGNMENT_ID / "drafts" / "custom_drafts.jsonl", [blocked_draft])
    approval_csv = tmp_path / "approval.csv"
    _write_approval_csv(approval_csv, [_approval_for(blocked_draft)])

    result = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    queue = _json(Path(result["outgoing_queue"]))
    blocked = _json(Path(result["blocked_drafts"]))
    assert queue == []
    assert blocked[0]["message_type"] == "final_grade"
    assert "blocked_message_type" in blocked[0]["blocked_reasons"]


def test_phase3_draft_auto_feedback_cannot_be_sent_as_final_feedback(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    ai_draft = _base_draft(draft_id="draft-ai-feedback", message_type="draft_auto_grading_feedback")
    ai_draft["source_phase"] = "phase3_draft_auto_grading"
    ai_draft["grading_mode"] = "draft_auto"
    _write_jsonl(output_root / ASSIGNMENT_ID / "drafts" / "custom_drafts.jsonl", [ai_draft])
    approval_csv = tmp_path / "approval.csv"
    _write_approval_csv(approval_csv, [_approval_for(ai_draft)])

    result = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    blocked = _json(Path(result["blocked_drafts"]))
    assert _json(Path(result["outgoing_queue"])) == []
    assert "phase3_draft_auto_feedback_blocked" in blocked[0]["blocked_reasons"]


def test_generic_teacher_approved_feedback_from_phase3_source_can_queue(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    feedback = _base_draft(draft_id="draft-teacher-feedback", message_type="generic_teacher_approved_feedback")
    feedback["source_phase"] = "phase3_draft_auto_grading"
    feedback["grading_mode"] = "draft_auto"
    _write_jsonl(output_root / ASSIGNMENT_ID / "drafts" / "custom_drafts.jsonl", [feedback])
    approval_csv = tmp_path / "approval.csv"
    _write_approval_csv(approval_csv, [_approval_for(feedback)])

    result = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    queue = _json(Path(result["outgoing_queue"]))
    assert len(queue) == 1
    assert queue[0]["message_type"] == "generic_teacher_approved_feedback"


def test_recipient_email_mismatch_is_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    draft = _base_draft()
    _write_jsonl(output_root / ASSIGNMENT_ID / "drafts" / "custom_drafts.jsonl", [draft])
    approval_csv = tmp_path / "approval.csv"
    _write_approval_csv(approval_csv, [_approval_for(draft, recipient_email="other@example.invalid")])

    result = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    blocked = _json(Path(result["blocked_drafts"]))
    assert _json(Path(result["outgoing_queue"])) == []
    assert "recipient_email_mismatch" in blocked[0]["blocked_reasons"]


def test_unknown_draft_id_is_audited_and_not_queued(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    draft = _base_draft()
    _write_jsonl(output_root / ASSIGNMENT_ID / "drafts" / "custom_drafts.jsonl", [draft])
    approval_csv = tmp_path / "approval.csv"
    unknown = _approval_for(draft, draft_id="missing-draft")
    _write_approval_csv(approval_csv, [unknown])

    result = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    audit = [json.loads(line) for line in Path(result["outgoing_email_audit"]).read_text(encoding="utf-8").splitlines()]
    rejected = [event for event in audit if event["event_type"] == "outgoing_approval_rejected"]
    assert rejected[0]["draft_id"] == "missing-draft"
    assert "unknown_draft_id" in rejected[0]["reasons"]
    assert _json(Path(result["outgoing_queue"])) == []


def test_approved_status_without_approver_is_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    draft = _base_draft()
    _write_jsonl(output_root / ASSIGNMENT_ID / "drafts" / "custom_drafts.jsonl", [draft])
    approval_csv = tmp_path / "approval.csv"
    _write_approval_csv(approval_csv, [_approval_for(draft, approved_by="")])

    result = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    blocked = _json(Path(result["blocked_drafts"]))
    assert "approved_by_required" in blocked[0]["blocked_reasons"]
    assert _json(Path(result["outgoing_queue"])) == []


def test_no_live_credentials_or_network_sending_behavior_exists() -> None:
    source = Path("src/exam_bank/submissions/outgoing_email.py").read_text(encoding="utf-8").lower()
    for forbidden in ["smtplib", "imaplib", "poplib", "password", "client_secret", "sendmail", "requests.", "urllib"]:
        assert forbidden not in source


def test_normalization_ignores_source_send_allowed_until_approval() -> None:
    payload = _base_draft()
    payload["send_allowed"] = True
    draft = normalize_outgoing_email_draft(payload, source_phase="phase1_submission_tracker")
    assert draft.send_allowed is False
    assert draft.approval_status == "blocked"
    assert "source_draft_send_allowed_ignored" in draft.blocked_reasons
