from __future__ import annotations

from datetime import datetime, timezone

import pytest

from exam_bank.submissions.email_models import (
    EmailSubmissionProvenance,
    InboundEmailAttachment,
    InboundEmailMessage,
)
from exam_bank.submissions.models import dataclass_to_json_dict


def _attachment(**overrides: object) -> InboundEmailAttachment:
    values = {
        "attachment_id": "msg-1:0",
        "message_id": "msg-1",
        "filename": "S0001_vectors_hw1.pdf",
        "content_type": "application/pdf",
        "size_bytes": 128,
        "sha256": "abc123",
        "stored_path": "",
        "attachment_index": 0,
        "status": "accepted",
        "reasons": [],
    }
    values.update(overrides)
    return InboundEmailAttachment(**values)


def test_inbound_email_message_serializes_safely() -> None:
    created_at = datetime(2026, 6, 22, 10, 0, tzinfo=timezone.utc)
    attachment = _attachment()
    message = InboundEmailMessage(
        message_id="msg-1",
        thread_id="thread-1",
        assignment_id="p3_vectors_hw1",
        received_at=created_at,
        from_email="student.one@example.invalid",
        from_name="Student One",
        subject="p3_vectors_hw1 S0001",
        body_preview="Synthetic fixture body only.",
        attachment_count=1,
        attachments=[attachment],
        source="fixture",
        status="accepted",
        reasons=[],
        created_at=created_at,
    )

    payload = dataclass_to_json_dict(message)

    assert payload["message_id"] == "msg-1"
    assert payload["received_at"] == "2026-06-22T10:00:00+00:00"
    assert payload["from_email"].endswith("@example.invalid")
    assert payload["attachments"][0]["sha256"] == "abc123"


def test_inbound_email_attachment_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="Invalid email attachment status"):
        _attachment(status="sent")


def test_email_submission_provenance_includes_message_and_attachment_hash() -> None:
    provenance = EmailSubmissionProvenance(
        source="fixture",
        message_id="msg-1",
        thread_id="thread-1",
        from_email="student.one@example.invalid",
        received_at=datetime(2026, 6, 22, 10, 0, tzinfo=timezone.utc),
        original_attachment_filename="S0001_vectors_hw1.pdf",
        attachment_id="msg-1:0",
        attachment_sha256="abc123",
    )

    payload = dataclass_to_json_dict(provenance)

    assert payload["message_id"] == "msg-1"
    assert payload["attachment_id"] == "msg-1:0"
    assert payload["attachment_sha256"] == "abc123"
