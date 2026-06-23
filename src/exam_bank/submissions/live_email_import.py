from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exam_bank.submissions.email_connectors import EmailConnector, build_configured_connector
from exam_bank.submissions.email_identity import match_student_for_email
from exam_bank.submissions.email_intake import ingest_email_fixtures
from exam_bank.submissions.email_models import EmailIntakeDryRunDecision, InboundEmailMessage
from exam_bank.submissions.email_policy import build_email_dry_run_decision
from exam_bank.submissions.ingest import _require_private_roots, load_assignment, load_roster, parse_datetime
from exam_bank.submissions.models import dataclass_to_json_dict


def load_email_connector_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def import_live_email_submissions(
    *,
    assignment_path: Path,
    roster_path: Path,
    connector_config_path: Path,
    submission_output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
    apply: bool = False,
    connector: EmailConnector | None = None,
) -> dict[str, object]:
    _require_private_roots(submission_output_root, reports_root)
    assignment = load_assignment(assignment_path)
    roster = [student for student in load_roster(roster_path) if student.class_id == assignment.class_id and student.active]
    config = load_email_connector_config(connector_config_path)
    _validate_config(config, assignment.assignment_id)

    dry_run = not apply
    live_dir = submission_output_root / assignment.assignment_id / "live_email_import"
    live_dir.mkdir(parents=True, exist_ok=True)
    audit_path = live_dir / "live_email_import_audit.jsonl"
    audit_events: list[dict[str, object]] = []
    _record(audit_events, "live_email_import_started", assignment.assignment_id, status="dry_run" if dry_run else "apply")

    connector = connector or build_configured_connector(config, assignment_id=assignment.assignment_id)
    since = parse_datetime(str(config["since"])) if str(config.get("since", "")).strip() else None
    limit = int(config.get("limit", 100))
    refs = connector.list_messages(
        mailbox_scope=str(config.get("mailbox_scope", "")).strip(),
        search_query=str(config.get("search_query", "")).strip(),
        since=since,
        limit=limit,
    )
    messages: list[InboundEmailMessage] = []
    attachments_by_position: list[dict[str, Path]] = []
    for ref in refs:
        messages.append(connector.fetch_message(ref.message_id))
        attachments_by_position.append(connector.fetch_attachments(ref.message_id))
    for message_index, message in enumerate(messages):
        _record(audit_events, "live_email_message_seen", assignment.assignment_id, message=message, status="seen")

    if dry_run:
        decisions = _dry_run_decisions(messages=messages, roster=roster, assignment=assignment)
        summary = _summary(
            assignment_id=assignment.assignment_id,
            roster_count=len(roster),
            config=config,
            dry_run=True,
            messages=messages,
            decisions=decisions,
            phase1_result=None,
        )
        dry_run_path = live_dir / "live_email_dry_run.json"
        _write_json(dry_run_path, {"summary": summary, "messages": [_safe_message_payload(message) for message in messages], "decisions": [_payload(decision) for decision in decisions]})
        readiness_path = reports_root / f"{assignment.assignment_id}_first_assignment_readiness.md"
        _write_readiness_report(readiness_path, summary)
        _record(audit_events, "live_email_dry_run_written", assignment.assignment_id, status="written")
        _write_jsonl(audit_path, audit_events)
        return {
            "assignment_id": assignment.assignment_id,
            "dry_run": True,
            "apply": False,
            "live_email_dir": live_dir,
            "dry_run_report": dry_run_path,
            "readiness_report": readiness_path,
            "audit_log": audit_path,
            "summary": summary,
            "messages": messages,
            "decisions": decisions,
            "phase4_result": None,
        }

    fixture_dir = _materialize_connector_fixtures(
        messages=messages,
        attachments_by_position=attachments_by_position,
        target_dir=live_dir / "materialized_fixtures",
    )
    _record(audit_events, "live_email_fixtures_materialized", assignment.assignment_id, status="written")
    phase4_result = ingest_email_fixtures(
        assignment_path=assignment_path,
        roster_path=roster_path,
        email_fixtures_dir=fixture_dir,
        output_root=submission_output_root,
        reports_root=reports_root,
        dry_run=False,
        source_label="live_connector",
    )
    decisions = _dry_run_decisions(messages=messages, roster=roster, assignment=assignment)
    summary = _summary(
        assignment_id=assignment.assignment_id,
        roster_count=len(roster),
        config=config,
        dry_run=False,
        messages=messages,
        decisions=decisions,
        phase1_result=phase4_result.get("phase1_result"),
    )
    summary_path = live_dir / "live_email_import_summary.json"
    readiness_path = reports_root / f"{assignment.assignment_id}_first_assignment_readiness.md"
    _write_json(summary_path, summary)
    _write_readiness_report(readiness_path, summary)
    _record(audit_events, "live_email_apply_finished", assignment.assignment_id, status="finished")
    _write_jsonl(audit_path, audit_events)
    return {
        "assignment_id": assignment.assignment_id,
        "dry_run": False,
        "apply": True,
        "live_email_dir": live_dir,
        "summary_path": summary_path,
        "readiness_report": readiness_path,
        "audit_log": audit_path,
        "summary": summary,
        "messages": messages,
        "decisions": decisions,
        "phase4_result": phase4_result,
    }


def _validate_config(config: dict[str, object], assignment_id: str) -> None:
    if str(config.get("assignment_id", "")).strip() != assignment_id:
        raise ValueError("Email connector config assignment_id must match assignment")
    if not str(config.get("mailbox_scope", "")).strip() and not str(config.get("search_query", "")).strip():
        raise ValueError("Email connector config must set mailbox_scope or search_query")
    if int(config.get("limit", 0)) <= 0:
        raise ValueError("Email connector config limit must be positive")


def _dry_run_decisions(
    *,
    messages: list[InboundEmailMessage],
    roster: list[object],
    assignment: object,
) -> list[EmailIntakeDryRunDecision]:
    seen_message_ids: set[str] = set()
    seen_attachment_hashes: set[str] = set()
    accepted_student_ids: set[str] = set()
    decisions: list[EmailIntakeDryRunDecision] = []
    for message in messages:
        student_match = match_student_for_email(message, roster)
        decision = build_email_dry_run_decision(
            message=message,
            roster=roster,
            assignment=assignment,
            seen_message_ids=seen_message_ids,
            seen_attachment_hashes=seen_attachment_hashes,
            accepted_student_ids=accepted_student_ids,
        )
        decisions.append(decision)
        if message.message_id:
            seen_message_ids.add(message.message_id)
        if decision.status == "accepted" and student_match.student_id:
            accepted_student_ids.add(student_match.student_id)
        for attachment in message.attachments:
            if attachment.sha256:
                seen_attachment_hashes.add(attachment.sha256)
    return decisions


def _materialize_connector_fixtures(
    *,
    messages: list[InboundEmailMessage],
    attachments_by_position: list[dict[str, Path]],
    target_dir: Path,
) -> Path:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    attachments_dir = target_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    for message_index, message in enumerate(messages):
        attachment_rows: list[dict[str, object]] = []
        paths = attachments_by_position[message_index] if message_index < len(attachments_by_position) else {}
        for attachment in message.attachments:
            source = paths.get(attachment.attachment_id)
            relative_path = ""
            if source is not None and source.exists() and source.is_file():
                target = attachments_dir / f"{message_index}_{_safe_token(message.message_id)}_{attachment.attachment_index}_{_safe_token(attachment.filename)}"
                shutil.copy2(source, target)
                relative_path = target.relative_to(target_dir).as_posix()
            attachment_rows.append(
                {
                    "filename": attachment.filename,
                    "path": relative_path,
                    "content_type": attachment.content_type,
                }
            )
        payload = {
            "message_id": message.message_id,
            "thread_id": message.thread_id,
            "received_at": message.received_at.isoformat() if message.received_at is not None else "",
            "from_email": message.from_email,
            "from_name": message.from_name,
            "subject": message.subject,
            "body_preview": message.body_preview,
            "attachments": attachment_rows,
        }
        (target_dir / f"{message_index}_{_safe_token(message.message_id)}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target_dir


def _summary(
    *,
    assignment_id: str,
    roster_count: int,
    config: dict[str, object],
    dry_run: bool,
    messages: list[InboundEmailMessage],
    decisions: list[EmailIntakeDryRunDecision],
    phase1_result: object | None,
) -> dict[str, object]:
    accepted = sum(1 for decision in decisions if decision.status == "accepted")
    quarantined = sum(1 for decision in decisions if decision.status == "quarantined")
    rejected = sum(1 for decision in decisions if decision.status == "rejected")
    warnings: list[str] = []
    if not messages:
        warnings.append("no_messages_found")
    if quarantined or rejected:
        warnings.append("review_quarantine_before_apply")
    if dry_run:
        warnings.append("dry_run_only")
    apply_recommended = bool(messages) and quarantined == 0 and rejected == 0
    summary = {
        "assignment_id": assignment_id,
        "roster_count": roster_count,
        "account_label": str(config.get("account_label", "")),
        "mailbox_scope": str(config.get("mailbox_scope", "")),
        "search_query": str(config.get("search_query", "")),
        "since": str(config.get("since", "")),
        "limit": int(config.get("limit", 0)),
        "dry_run": dry_run,
        "messages_found": len(messages),
        "likely_accepted_count": accepted,
        "likely_quarantined_count": quarantined,
        "likely_rejected_count": rejected,
        "apply_recommended": apply_recommended,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(phase1_result, dict):
        summary["phase1_accepted_count"] = len(phase1_result.get("accepted", []))
        summary["phase1_rejected_count"] = len(phase1_result.get("rejected", []))
    else:
        summary["phase1_accepted_count"] = 0
        summary["phase1_rejected_count"] = 0
    return summary


def _write_readiness_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        f"# First Assignment Readiness: {summary['assignment_id']}",
        "",
        f"- Assignment ID: `{summary['assignment_id']}`",
        f"- Roster count: `{summary['roster_count']}`",
        f"- Email scope: `{summary['mailbox_scope']}`",
        f"- Search query: `{summary['search_query']}`",
        f"- Dry run: `{str(summary['dry_run']).lower()}`",
        f"- Messages found: `{summary['messages_found']}`",
        f"- Likely accepted: `{summary['likely_accepted_count']}`",
        f"- Likely quarantined: `{summary['likely_quarantined_count']}`",
        f"- Likely rejected: `{summary['likely_rejected_count']}`",
        f"- Apply recommended: `{str(summary['apply_recommended']).lower()}`",
        f"- Warnings: `{', '.join(summary['warnings']) if summary['warnings'] else 'none'}`",
        "",
        "Do not apply until the mailbox scope, roster, assignment metadata, and quarantine result have been reviewed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_message_payload(message: InboundEmailMessage) -> dict[str, object]:
    payload = dataclass_to_json_dict(message)
    payload["body_preview"] = message.body_preview[:240]
    return payload


def _payload(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _record(
    events: list[dict[str, object]],
    event_type: str,
    assignment_id: str,
    *,
    message: InboundEmailMessage | None = None,
    status: str = "",
    reasons: list[str] | None = None,
) -> None:
    events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "assignment_id": assignment_id,
            "message_id": message.message_id if message is not None else "",
            "thread_id": message.thread_id if message is not None else "",
            "status": status,
            "reasons": reasons or [],
        }
    )


def _safe_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return token.strip("._") or "unknown"
