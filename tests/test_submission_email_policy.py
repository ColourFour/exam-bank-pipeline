from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from exam_bank.submissions.email_models import InboundEmailAttachment, InboundEmailMessage
from exam_bank.submissions.email_policy import build_email_dry_run_decision, duplicate_or_resend_reasons
from exam_bank.submissions.ingest import load_assignment, load_roster


FIXTURE_TIME = datetime(2026, 6, 22, 10, 0, tzinfo=timezone.utc)
FIXTURES = Path("tests/fixtures/submissions")


def _assignment():
    return load_assignment(FIXTURES / "assignment_p3_vectors_hw1.json")


def _roster():
    return load_roster(FIXTURES / "roster_class_12a.csv")


def _attachment(
    *,
    filename: str = "S0001_vectors_hw1.pdf",
    content_type: str = "application/pdf",
    sha256: str = "abc123",
    size_bytes: int = 128,
    index: int = 0,
) -> InboundEmailAttachment:
    return InboundEmailAttachment(
        attachment_id=f"msg-1:{index}",
        message_id="msg-1",
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        sha256=sha256,
        stored_path="",
        attachment_index=index,
        status="accepted",
        reasons=[],
    )


def _message(
    *,
    message_id: str = "msg-1",
    from_email: str = "student.one@example.invalid",
    attachments: list[InboundEmailAttachment] | None = None,
    received_at: datetime | None = FIXTURE_TIME,
    assignment_id: str = "p3_vectors_hw1",
) -> InboundEmailMessage:
    attachments = attachments if attachments is not None else [_attachment()]
    return InboundEmailMessage(
        message_id=message_id,
        thread_id="thread-1",
        assignment_id=assignment_id,
        received_at=received_at,
        from_email=from_email,
        from_name="Student One",
        subject="p3_vectors_hw1",
        body_preview="Synthetic fixture body only.",
        attachment_count=len(attachments),
        attachments=attachments,
        source="fixture",
        status="accepted",
        reasons=[],
        created_at=FIXTURE_TIME,
    )


def test_no_pdf_message_yields_quarantine_reason_no_pdf_attachment() -> None:
    decision = build_email_dry_run_decision(
        message=_message(attachments=[_attachment(filename="notes.txt", content_type="text/plain")]),
        roster=_roster(),
        assignment=_assignment(),
    )

    assert decision.status == "quarantined"
    assert "no_pdf_attachment" in decision.reasons
    assert decision.rejected_attachments[0].reasons == ["non_pdf_attachment"]


def test_too_many_pdfs_decision_exists() -> None:
    decision = build_email_dry_run_decision(
        message=_message(
            attachments=[
                _attachment(filename="S0001_a.pdf", sha256="a", index=0),
                _attachment(filename="S0001_b.pdf", sha256="b", index=1),
            ]
        ),
        roster=_roster(),
        assignment=_assignment(),
    )

    assert decision.status == "quarantined"
    assert "too_many_pdf_attachments" in decision.reasons


def test_duplicate_message_id_policy_exists() -> None:
    message = _message()

    assert duplicate_or_resend_reasons(message=message, student_id="S0001", seen_message_ids={"msg-1"}) == [
        "duplicate_message_id"
    ]

    decision = build_email_dry_run_decision(
        message=message,
        roster=_roster(),
        assignment=_assignment(),
        seen_message_ids={"msg-1"},
    )
    assert "duplicate_message_id" in decision.reasons
    assert decision.status == "quarantined"


def test_duplicate_attachment_hash_policy_exists() -> None:
    decision = build_email_dry_run_decision(
        message=_message(),
        roster=_roster(),
        assignment=_assignment(),
        seen_attachment_hashes={"abc123"},
    )

    assert "duplicate_attachment_hash" in decision.reasons
    assert decision.quarantined_attachments[0].reasons == ["duplicate_attachment_hash"]


def test_resubmission_after_accepted_submission_is_reported() -> None:
    decision = build_email_dry_run_decision(
        message=_message(),
        roster=_roster(),
        assignment=_assignment(),
        accepted_student_ids={"S0001"},
    )

    assert "resubmission_detected" in decision.reasons
    assert decision.would_create_draft is True


def test_dry_run_decision_does_not_stage_or_create_submissions() -> None:
    decision = build_email_dry_run_decision(message=_message(), roster=_roster(), assignment=_assignment())

    assert decision.status == "accepted"
    assert decision.would_stage_pdf is False
    assert decision.would_create_submission is False
    assert decision.would_create_draft is True
    assert "dry_run_only" in decision.reasons


def test_unknown_student_dry_run_does_not_create_draft() -> None:
    decision = build_email_dry_run_decision(
        message=_message(from_email="unknown@example.invalid", attachments=[_attachment(filename="homework.pdf")]),
        roster=_roster(),
        assignment=_assignment(),
    )

    assert decision.student_match_status == "unknown"
    assert decision.would_create_draft is False


def test_no_email_credentials_or_live_connector_behavior_exists() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/exam_bank/submissions").glob("email_*.py"))

    for forbidden in ["smtplib", "imaplib", "poplib", "outlook", "gmail", "smtp", "password", "client_secret"]:
        assert forbidden not in source.lower()
