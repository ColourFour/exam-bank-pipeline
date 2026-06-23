from __future__ import annotations

import json
import shutil
from pathlib import Path

from exam_bank.submissions.email_fixtures import load_email_fixtures


FIXTURES = Path("tests/fixtures/submissions")


def test_email_fixture_loading_computes_attachment_metadata() -> None:
    loaded = load_email_fixtures(FIXTURES / "email_inbox", "p3_vectors_hw1")

    message = next(item.message for item in loaded if item.message.message_id == "fake-message-valid-filename")
    attachment = message.attachments[0]

    assert message.received_at is not None
    assert message.thread_id == "fake-thread-a"
    assert attachment.filename == "S0001_email_quiz.pdf"
    assert attachment.size_bytes > 0
    assert len(attachment.sha256) == 64
    assert attachment.stored_path == "attachments/S0001_email_quiz.pdf"


def test_unsafe_attachment_path_is_rejected(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "email_inbox"
    shutil.copytree(FIXTURES / "email_inbox", fixture_dir)
    (fixture_dir / "message_unsafe.json").write_text(
        json.dumps(
            {
                "message_id": "fake-message-unsafe",
                "thread_id": "fake-thread-unsafe",
                "received_at": "2026-06-23T17:45:00+08:00",
                "from_email": "student.one@example.invalid",
                "from_name": "Student One",
                "subject": "p3_vectors_hw1 S0001",
                "body_preview": "Attached.",
                "attachments": [
                    {
                        "filename": "escaped.pdf",
                        "path": "../escaped.pdf",
                        "content_type": "application/pdf",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    message = next(item.message for item in load_email_fixtures(fixture_dir, "p3_vectors_hw1") if item.message.message_id == "fake-message-unsafe")

    assert message.attachments[0].status == "rejected"
    assert message.attachments[0].reasons == ["unsafe_attachment_path"]
    assert message.attachments[0].sha256 == ""


def test_missing_attachment_file_is_recorded(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "email_inbox"
    shutil.copytree(FIXTURES / "email_inbox", fixture_dir)
    (fixture_dir / "message_missing_file.json").write_text(
        json.dumps(
            {
                "message_id": "fake-message-missing-file",
                "thread_id": "fake-thread-missing",
                "received_at": "2026-06-23T17:50:00+08:00",
                "from_email": "student.one@example.invalid",
                "from_name": "Student One",
                "subject": "p3_vectors_hw1 S0001",
                "body_preview": "Attached.",
                "attachments": [
                    {
                        "filename": "missing.pdf",
                        "path": "attachments/missing.pdf",
                        "content_type": "application/pdf",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    message = next(item.message for item in load_email_fixtures(fixture_dir, "p3_vectors_hw1") if item.message.message_id == "fake-message-missing-file")

    assert message.attachments[0].status == "rejected"
    assert "attachment_missing_file" in message.attachments[0].reasons
