from __future__ import annotations

import json
import shutil
from pathlib import Path

from exam_bank.submissions.email_intake import ingest_email_fixtures


FIXTURES = Path("tests/fixtures/submissions")


def _run_intake(tmp_path: Path, *, dry_run: bool = False, fixture_dir: Path | None = None) -> dict[str, object]:
    return ingest_email_fixtures(
        assignment_path=FIXTURES / "assignment_p3_vectors_hw1.json",
        roster_path=FIXTURES / "roster_class_12a.csv",
        email_fixtures_dir=fixture_dir or FIXTURES / "email_inbox",
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        dry_run=dry_run,
    )


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_fixture_backed_email_intake_stages_valid_messages_and_runs_phase1(tmp_path: Path) -> None:
    result = _run_intake(tmp_path)
    summary = result["summary"]

    assert summary["messages_seen"] == 7
    assert summary["messages_accepted"] == 3
    assert summary["messages_quarantined"] == 4
    assert summary["messages_rejected"] == 0
    assert summary["pdf_attachments_staged"] == 3
    assert summary["phase1_accepted_count"] == 3
    assert summary["phase1_rejected_count"] == 0
    assert Path(str(summary["completion_csv"])).is_file()
    assert Path(str(summary["email_intake_summary"])).is_file()

    staged = sorted((Path(result["staged_dir"])).glob("*.pdf"))
    assert len(staged) == 3
    assert all(str(path).startswith(str(tmp_path / "output" / "submissions")) for path in staged)
    assert any(path.name.startswith("S0001_fake-message-valid-filename_attachment-0") for path in staged)
    assert any(path.name.startswith("S0002_fake-message-valid-sender_attachment-0") for path in staged)
    assert any(path.name.startswith("S0003_fake-message-duplicate_attachment-0") for path in staged)


def test_email_intake_message_classification_reasons(tmp_path: Path) -> None:
    result = _run_intake(tmp_path)
    messages = {message["message_id"]: message for message in result["messages"]}

    assert messages["fake-message-valid-filename"]["status"] == "accepted"
    assert messages["fake-message-valid-filename"]["student_match"]["student_id"] == "S0001"
    assert messages["fake-message-valid-sender"]["status"] == "accepted"
    assert messages["fake-message-valid-sender"]["student_match"]["student_id"] == "S0002"
    assert messages["fake-message-no-pdf"]["status"] == "quarantined"
    assert "no_pdf_attachment" in messages["fake-message-no-pdf"]["reasons"]
    assert messages["fake-message-unknown"]["status"] == "quarantined"
    assert "unknown_student" in messages["fake-message-unknown"]["reasons"]
    assert messages["fake-message-ambiguous"]["status"] == "quarantined"
    assert "ambiguous_student_match" in messages["fake-message-ambiguous"]["reasons"]

    duplicate_messages = [message for message in result["messages"] if message["message_id"] == "fake-message-duplicate"]
    assert [message["status"] for message in duplicate_messages] == ["accepted", "quarantined"]
    assert "duplicate_message_id" in duplicate_messages[1]["reasons"]


def test_non_pdf_only_message_rejects_attachment_and_creates_resend_draft(tmp_path: Path) -> None:
    result = _run_intake(tmp_path)
    no_pdf = next(message for message in result["messages"] if message["message_id"] == "fake-message-no-pdf")
    draft = next(draft for draft in result["drafts"] if draft["message_id"] == "fake-message-no-pdf")

    assert no_pdf["attachments"][0]["status"] == "rejected"
    assert no_pdf["attachments"][0]["reasons"] == ["non_pdf_attachment"]
    assert no_pdf["status"] == "quarantined"
    assert draft["draft_type"] == "email_resend_needed"
    assert draft["send_allowed"] is False


def test_duplicate_attachment_hash_is_not_restaged_twice(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "email_inbox"
    shutil.copytree(FIXTURES / "email_inbox", fixture_dir)
    for path in fixture_dir.glob("message_*.json"):
        path.unlink()
    shared = "attachments/S0001_email_quiz.pdf"
    (fixture_dir / "message_hash_a.json").write_text(
        json.dumps(
            {
                "message_id": "fake-message-hash-a",
                "thread_id": "fake-thread-hash",
                "received_at": "2026-06-23T18:00:00+08:00",
                "from_email": "student.one@example.invalid",
                "from_name": "Student One",
                "subject": "p3_vectors_hw1",
                "body_preview": "Attached.",
                "attachments": [{"filename": "S0001_hash_a.pdf", "path": shared, "content_type": "application/pdf"}],
            }
        ),
        encoding="utf-8",
    )
    (fixture_dir / "message_hash_b.json").write_text(
        json.dumps(
            {
                "message_id": "fake-message-hash-b",
                "thread_id": "fake-thread-hash",
                "received_at": "2026-06-23T18:05:00+08:00",
                "from_email": "student.two@example.invalid",
                "from_name": "Student Two",
                "subject": "p3_vectors_hw1",
                "body_preview": "Attached.",
                "attachments": [{"filename": "S0002_hash_b.pdf", "path": shared, "content_type": "application/pdf"}],
            }
        ),
        encoding="utf-8",
    )

    result = _run_intake(tmp_path, fixture_dir=fixture_dir)
    messages = {message["message_id"]: message for message in result["messages"]}

    assert result["summary"]["pdf_attachments_staged"] == 1
    assert messages["fake-message-hash-b"]["status"] == "quarantined"
    assert "duplicate_attachment_hash" in messages["fake-message-hash-b"]["reasons"]
    assert len(list(Path(result["staged_dir"]).glob("*.pdf"))) == 1


def test_provenance_and_draft_artifacts_are_written(tmp_path: Path) -> None:
    result = _run_intake(tmp_path)
    email_dir = Path(result["email_intake_dir"])
    provenance = _json(email_dir / "provenance.json")
    drafts = _jsonl(email_dir / "drafts" / "email_drafts.jsonl")

    assert len(provenance) == 3
    first = next(item for item in provenance if item["message_id"] == "fake-message-valid-filename")
    assert first["phase1_staged_filename"].startswith("S0001_fake-message-valid-filename_attachment-0")
    assert first["original_attachment_filename"] == "S0001_email_quiz.pdf"
    assert len(first["attachment_sha256"]) == 64
    assert first["student_match_source"] == "attachment_filename_student_id"

    assert len(drafts) == 7
    assert all(draft["send_allowed"] is False for draft in drafts)
    assert {"email_acknowledgement", "email_resend_needed"} <= {draft["draft_type"] for draft in drafts}


def test_phase1_completion_csv_and_manifest_are_written(tmp_path: Path) -> None:
    result = _run_intake(tmp_path)
    phase1_result = result["phase1_result"]

    assert phase1_result is not None
    assert Path(phase1_result["manifest"]).is_file()
    assert Path(phase1_result["completion_report"]).is_file()
    assert {submission.student_id for submission in phase1_result["accepted"]} == {"S0001", "S0002", "S0003"}


def test_audit_log_includes_email_intake_events(tmp_path: Path) -> None:
    result = _run_intake(tmp_path)
    events = _jsonl(Path(result["audit_log"]))
    event_types = {event["event_type"] for event in events}

    assert "email_intake_started" in event_types
    assert "email_fixture_loaded" in event_types
    assert "email_message_seen" in event_types
    assert "email_message_accepted" in event_types
    assert "email_message_quarantined" in event_types
    assert "email_attachment_seen" in event_types
    assert "email_attachment_accepted" in event_types
    assert "email_attachment_rejected" in event_types
    assert "email_pdf_staged_for_phase1" in event_types
    assert "email_phase1_ingest_started" in event_types
    assert "email_phase1_ingest_finished" in event_types
    assert "email_intake_summary_written" in event_types
    assert "email_intake_finished" in event_types
    assert "file_accepted" in event_types


def test_dry_run_does_not_copy_attachments_or_call_phase1(tmp_path: Path) -> None:
    result = _run_intake(tmp_path, dry_run=True)
    summary = result["summary"]

    assert summary["messages_seen"] == 7
    assert summary["pdf_attachments_staged"] == 0
    assert summary["phase1_accepted_count"] == 0
    assert result["phase1_result"] is None
    assert not Path(result["email_intake_dir"]).exists()
    assert all("dry_run_only" in decision["reasons"] for decision in result["dry_run_decisions"])


def test_no_live_credentials_or_sending_behavior_is_required() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/exam_bank/submissions").glob("email_*.py"))

    for forbidden in ["smtplib", "imaplib", "poplib", "outlook", "gmail", "smtp", "password", "client_secret", "sendmail"]:
        assert forbidden not in source.lower()
