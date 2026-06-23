from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from exam_bank.submissions.ingest import ingest_assignment_submissions
from exam_bank.submissions.outgoing_email import (
    build_outgoing_email_queue,
    send_outgoing_email_fake,
    write_outgoing_email_dry_run_report,
)


FIXTURES = Path("tests/fixtures/submissions")
ASSIGNMENT_ID = "p3_vectors_hw1"


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_phase1(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    ingest_assignment_submissions(
        assignment_path=FIXTURES / "assignment_p3_vectors_hw1.json",
        roster_path=FIXTURES / "roster_class_12a.csv",
        submissions_dir=FIXTURES / "inbox",
        output_root=output_root,
        reports_root=reports_root,
    )
    return output_root, reports_root


def _write_approval_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_build_queue_normalizes_drafts_and_writes_approval_template(tmp_path: Path) -> None:
    output_root, reports_root = _run_phase1(tmp_path)

    result = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    template = Path(result["approval_template"])
    blocked = _json(Path(result["blocked_drafts"]))
    queue = _json(Path(result["outgoing_queue"]))
    summary = _json(Path(result["outgoing_email_summary"]))
    audit = _jsonl(Path(result["outgoing_email_audit"]))

    assert template.is_file()
    with template.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["message_type"] for row in rows} == {
        "submission_acknowledgement",
        "resend_request",
        "missing_submission_reminder",
    }
    assert len(blocked) == 4
    assert all("approval_required" in draft["blocked_reasons"] for draft in blocked)
    assert queue == []
    assert summary["drafts_seen"] == 4
    assert summary["approved_count"] == 0
    assert summary["blocked_count"] == 4
    assert summary["queued_count"] == 0
    assert "outgoing_approval_template_written" in {event["event_type"] for event in audit}


def test_approved_safe_draft_becomes_dry_run_ready_queue_item(tmp_path: Path) -> None:
    output_root, reports_root = _run_phase1(tmp_path)
    initial = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
    )
    template_rows = list(csv.DictReader(Path(initial["approval_template"]).open("r", encoding="utf-8", newline="")))
    approved_row = next(row for row in template_rows if row["message_type"] == "submission_acknowledgement")
    approved_row["approval_status"] = "approved"
    approved_row["approved_by"] = "teacher@example.invalid"
    approval_csv = tmp_path / "approval.csv"
    _write_approval_csv(approval_csv, [approved_row])

    result = build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    approved = _json(Path(result["approved_drafts"]))
    queue = _json(Path(result["outgoing_queue"]))
    summary = _json(Path(result["outgoing_email_summary"]))

    assert len(approved) == 1
    assert approved[0]["send_allowed"] is True
    assert approved[0]["approval_status"] == "approved"
    assert len(queue) == 1
    assert queue[0]["delivery_mode"] == "dry_run"
    assert queue[0]["status"] == "dry_run_ready"
    assert queue[0]["send_allowed"] is True
    assert summary["approved_count"] == 1
    assert summary["dry_run_ready_count"] == 1


def test_dry_run_report_is_written_and_does_not_send(tmp_path: Path) -> None:
    output_root, reports_root = _run_phase1(tmp_path)
    initial = build_outgoing_email_queue(assignment_id=ASSIGNMENT_ID, submission_output_root=output_root, reports_root=reports_root)
    template_rows = list(csv.DictReader(Path(initial["approval_template"]).open("r", encoding="utf-8", newline="")))
    approved_row = template_rows[0]
    approved_row["approval_status"] = "approved"
    approved_row["approved_by"] = "teacher@example.invalid"
    approval_csv = tmp_path / "approval.csv"
    _write_approval_csv(approval_csv, [approved_row])
    build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    report = write_outgoing_email_dry_run_report(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    assert Path(report["dry_run_report"]).is_file()
    with Path(report["dry_run_report"]).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["would_send"] == "true"
    assert not (output_root / ASSIGNMENT_ID / "outgoing_email" / "fake_sent_messages.jsonl").exists()


def test_fake_adapter_requires_flag_and_writes_local_jsonl_only(tmp_path: Path) -> None:
    output_root, reports_root = _run_phase1(tmp_path)
    initial = build_outgoing_email_queue(assignment_id=ASSIGNMENT_ID, submission_output_root=output_root, reports_root=reports_root)
    template_rows = list(csv.DictReader(Path(initial["approval_template"]).open("r", encoding="utf-8", newline="")))
    approved_row = template_rows[0]
    approved_row["approval_status"] = "approved"
    approved_row["approved_by"] = "teacher@example.invalid"
    approval_csv = tmp_path / "approval.csv"
    _write_approval_csv(approval_csv, [approved_row])
    build_outgoing_email_queue(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        reports_root=reports_root,
        approval_csv=approval_csv,
    )

    with pytest.raises(ValueError, match="use-fake-adapter"):
        send_outgoing_email_fake(assignment_id=ASSIGNMENT_ID, submission_output_root=output_root)

    result = send_outgoing_email_fake(
        assignment_id=ASSIGNMENT_ID,
        submission_output_root=output_root,
        use_fake_adapter=True,
    )

    assert result["sent_count"] == 1
    records = _jsonl(Path(result["fake_sent_messages"]))
    assert len(records) == 1
    assert records[0]["adapter"] == "fake_adapter"
    assert records[0]["network_sent"] is False
    assert str(result["fake_sent_messages"]).startswith(str(output_root))
