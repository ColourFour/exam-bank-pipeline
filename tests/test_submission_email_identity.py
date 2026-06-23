from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from exam_bank.submissions.email_identity import match_student_for_email
from exam_bank.submissions.email_models import InboundEmailAttachment, InboundEmailMessage
from exam_bank.submissions.ingest import load_roster


FIXTURE_TIME = datetime(2026, 6, 22, 10, 0, tzinfo=timezone.utc)


def _attachment(filename: str = "homework.pdf") -> InboundEmailAttachment:
    return InboundEmailAttachment(
        attachment_id="msg-1:0",
        message_id="msg-1",
        filename=filename,
        content_type="application/pdf",
        size_bytes=128,
        sha256="abc123",
        stored_path="",
        attachment_index=0,
        status="accepted",
        reasons=[],
    )


def _message(
    *,
    from_email: str = "unknown@example.invalid",
    subject: str = "p3_vectors_hw1",
    body_preview: str = "",
    attachment_filename: str = "homework.pdf",
    from_name: str = "Student One",
) -> InboundEmailMessage:
    return InboundEmailMessage(
        message_id="msg-1",
        thread_id="thread-1",
        assignment_id="p3_vectors_hw1",
        received_at=FIXTURE_TIME,
        from_email=from_email,
        from_name=from_name,
        subject=subject,
        body_preview=body_preview,
        attachment_count=1,
        attachments=[_attachment(attachment_filename)],
        source="fixture",
        status="accepted",
        reasons=[],
        created_at=FIXTURE_TIME,
    )


def _roster():
    return load_roster(Path("tests/fixtures/submissions/roster_class_12a.csv"))


def test_attachment_filename_student_id_match_works() -> None:
    result = match_student_for_email(_message(attachment_filename="S0001_vectors_hw1.pdf"), _roster())

    assert result.status == "matched"
    assert result.student_id == "S0001"
    assert "matched_by_attachment_filename" in result.reasons


def test_subject_student_id_match_works() -> None:
    result = match_student_for_email(_message(subject="Submission S0002 vectors"), _roster())

    assert result.status == "matched"
    assert result.student_id == "S0002"
    assert "matched_by_subject" in result.reasons


def test_sender_email_match_works() -> None:
    result = match_student_for_email(_message(from_email="student.three@example.invalid"), _roster())

    assert result.status == "matched"
    assert result.student_id == "S0003"
    assert "matched_by_sender_email" in result.reasons


def test_unknown_student_returns_unknown() -> None:
    result = match_student_for_email(_message(), _roster())

    assert result.status == "unknown"
    assert result.student_id == ""
    assert result.candidates == []
    assert "unknown_student" in result.reasons


def test_multiple_different_matches_return_ambiguous() -> None:
    result = match_student_for_email(
        _message(
            from_email="student.one@example.invalid",
            subject="Submission from S0002",
            attachment_filename="homework.pdf",
        ),
        _roster(),
    )

    assert result.status == "ambiguous"
    assert result.student_id == ""
    assert {candidate.student_id for candidate in result.candidates} == {"S0001", "S0002"}
    assert "ambiguous_student_match" in result.reasons
    assert "multiple_student_ids_found" in result.reasons


def test_display_name_fuzzy_guessing_is_not_used() -> None:
    result = match_student_for_email(
        _message(from_email="unknown@example.invalid", from_name="Student One", subject="homework"),
        _roster(),
    )

    assert result.status == "unknown"


def test_body_student_id_match_is_low_confidence_fallback() -> None:
    result = match_student_for_email(_message(body_preview="My ID is S0001."), _roster())

    assert result.status == "matched"
    assert result.student_id == "S0001"
    assert result.candidates[0].match_source == "body_student_id"
    assert result.candidates[0].confidence == "low"
