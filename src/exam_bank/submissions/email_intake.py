from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import exam_bank.submissions.email_reasons as reasons
from exam_bank.submissions.audit_log import AuditLog
from exam_bank.submissions.email_fixtures import LoadedEmailFixture, load_email_fixtures
from exam_bank.submissions.email_identity import match_student_for_email
from exam_bank.submissions.email_models import (
    EmailIntakeDryRunDecision,
    EmailSubmissionProvenance,
    InboundEmailAttachment,
    InboundEmailMessage,
    StudentMatchResult,
)
from exam_bank.submissions.email_policy import build_email_dry_run_decision
from exam_bank.submissions.ingest import _require_private_roots, ingest_assignment_submissions, load_assignment, load_roster
from exam_bank.submissions.models import Assignment, Student, dataclass_to_json_dict


def ingest_email_fixtures(
    *,
    assignment_path: Path,
    roster_path: Path,
    email_fixtures_dir: Path,
    output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
    dry_run: bool = False,
    source_label: str = "fixture",
) -> dict[str, object]:
    _require_private_roots(output_root, reports_root)
    assignment = load_assignment(assignment_path)
    roster = [student for student in load_roster(roster_path) if student.class_id == assignment.class_id and student.active]
    roster_by_id = {student.student_id: student for student in roster}

    assignment_output = output_root / assignment.assignment_id
    email_output = assignment_output / "email_intake"
    staged_dir = email_output / "staged_for_phase1"
    quarantine_dir = email_output / "quarantine"
    attachments_dir = email_output / "attachments"
    audit_path = assignment_output / "audit.jsonl"
    audit = AuditLog(audit_path, assignment.assignment_id)

    if not dry_run:
        if email_output.exists():
            shutil.rmtree(email_output)
        staged_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        attachments_dir.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            audit_path.unlink()

    audit_events: list[dict[str, object]] = []
    _record(audit_events, "email_intake_started", assignment.assignment_id, status="started", dry_run=dry_run)

    fixtures = load_email_fixtures(email_fixtures_dir, assignment.assignment_id)
    _record(audit_events, "email_fixture_loaded", assignment.assignment_id, status="loaded", fixture_count=len(fixtures))

    seen_message_ids: set[str] = set()
    seen_attachment_hashes: set[str] = set()
    accepted_student_ids: set[str] = set()
    staged_names: set[str] = set()
    processed_messages: list[dict[str, object]] = []
    accepted_messages: list[dict[str, object]] = []
    quarantined_messages: list[dict[str, object]] = []
    rejected_messages: list[dict[str, object]] = []
    dry_run_decisions: list[dict[str, object]] = []
    provenance_records: list[dict[str, object]] = []
    draft_records: list[dict[str, object]] = []
    staged_count = 0

    for fixture in fixtures:
        message = fixture.message
        _record(
            audit_events,
            "email_message_seen",
            assignment.assignment_id,
            message=message,
            status="seen",
            reasons=message.reasons,
        )
        for attachment in message.attachments:
            _record(
                audit_events,
                "email_attachment_seen",
                assignment.assignment_id,
                message=message,
                attachment=attachment,
                status="seen",
                reasons=attachment.reasons,
            )

        student_match = match_student_for_email(message, roster)
        decision = build_email_dry_run_decision(
            message=message,
            roster=roster,
            assignment=assignment,
            seen_message_ids=seen_message_ids,
            seen_attachment_hashes=seen_attachment_hashes,
            accepted_student_ids=accepted_student_ids,
        )
        dry_run_decisions.append(_decision_payload(decision, student_match))

        classified_message, accepted_attachments, quarantined_attachments, rejected_attachments = _classify_message(
            message=message,
            assignment=assignment,
            student_match=student_match,
            seen_message_ids=seen_message_ids,
            seen_attachment_hashes=seen_attachment_hashes,
            accepted_student_ids=accepted_student_ids,
        )

        message_payload = _message_payload(classified_message, student_match)
        processed_messages.append(message_payload)
        if classified_message.status == "accepted":
            accepted_messages.append(message_payload)
        elif classified_message.status == "rejected":
            rejected_messages.append(message_payload)
        else:
            quarantined_messages.append(message_payload)

        if message.message_id:
            seen_message_ids.add(message.message_id)

        if classified_message.status == "accepted":
            accepted_student_ids.add(student_match.student_id)

        for attachment in accepted_attachments:
            if attachment.sha256:
                seen_attachment_hashes.add(attachment.sha256)
            attachment_source = fixture.attachment_paths.get(attachment.attachment_id)
            if attachment_source is None:
                continue
            staged_filename = _stage_filename(student_match.student_id, message.message_id, attachment, staged_names)
            staged_names.add(staged_filename)
            if not dry_run:
                staged_path = staged_dir / staged_filename
                shutil.copy2(attachment_source, staged_path)
                if message.received_at is not None:
                    timestamp = message.received_at.timestamp()
                    os.utime(staged_path, (timestamp, timestamp))
                staged_count += 1
                _record(
                    audit_events,
                    "email_pdf_staged_for_phase1",
                    assignment.assignment_id,
                    message=message,
                    attachment=attachment,
                    student_id=student_match.student_id,
                    status="staged",
                    staged_filename=staged_filename,
                )
            provenance_records.append(
                _provenance_payload(
                    provenance=EmailSubmissionProvenance(
                        source=source_label,
                        message_id=message.message_id,
                        thread_id=message.thread_id,
                        from_email=message.from_email,
                        received_at=message.received_at,
                        original_attachment_filename=attachment.filename,
                        attachment_id=attachment.attachment_id,
                        attachment_sha256=attachment.sha256,
                    ),
                    staged_filename=staged_filename,
                    student_id=student_match.student_id,
                    match_source=_match_source(student_match),
                )
            )

        if not dry_run:
            for attachment in quarantined_attachments + rejected_attachments:
                source = fixture.attachment_paths.get(attachment.attachment_id)
                if source is not None and source.exists() and source.is_file():
                    target = quarantine_dir / _quarantine_filename(message.message_id, attachment)
                    shutil.copy2(source, target)

        draft_records.append(_draft_payload(assignment, roster_by_id, classified_message, student_match))

        _record_message_and_attachment_events(
            audit_events,
            assignment.assignment_id,
            classified_message,
            student_match,
            accepted_attachments,
            quarantined_attachments,
            rejected_attachments,
        )

    phase1_result: dict[str, object] | None = None
    _record(audit_events, "email_phase1_ingest_started", assignment.assignment_id, status="skipped" if dry_run else "started")
    if not dry_run:
        _flush_audit(audit, audit_events)
        audit_events.clear()
        phase1_result = ingest_assignment_submissions(
            assignment_path=assignment_path,
            roster_path=roster_path,
            submissions_dir=staged_dir,
            output_root=output_root,
            reports_root=reports_root,
            reset_audit=False,
        )
        _record(
            audit_events,
            "email_phase1_ingest_finished",
            assignment.assignment_id,
            status="finished",
            phase1_accepted_count=len(phase1_result["accepted"]),
            phase1_rejected_count=len(phase1_result["rejected"]),
        )
    else:
        _record(audit_events, "email_phase1_ingest_finished", assignment.assignment_id, status="skipped")

    summary = {
        "assignment_id": assignment.assignment_id,
        "dry_run": dry_run,
        "messages_seen": len(processed_messages),
        "messages_accepted": len(accepted_messages),
        "messages_quarantined": len(quarantined_messages),
        "messages_rejected": len(rejected_messages),
        "attachments_seen": sum(int(message["attachment_count"]) for message in processed_messages),
        "pdf_attachments_staged": 0 if dry_run else staged_count,
        "phase1_accepted_count": len(phase1_result["accepted"]) if phase1_result else 0,
        "phase1_rejected_count": len(phase1_result["rejected"]) if phase1_result else 0,
        "completion_csv": str(phase1_result["completion_report"]) if phase1_result else "",
        "email_intake_summary": str(email_output / "email_intake_summary.json") if not dry_run else "",
    }

    if not dry_run:
        _write_json(email_output / "messages.json", processed_messages)
        _write_json(email_output / "accepted_messages.json", accepted_messages)
        _write_json(email_output / "quarantined_messages.json", quarantined_messages)
        _write_json(email_output / "rejected_messages.json", rejected_messages)
        _write_json(email_output / "provenance.json", provenance_records)
        _write_json(email_output / "dry_run_decisions.json", dry_run_decisions)
        _write_json(email_output / "email_intake_summary.json", summary)
        _write_jsonl(email_output / "drafts" / "email_drafts.jsonl", draft_records)
        _record(
            audit_events,
            "email_intake_summary_written",
            assignment.assignment_id,
            status="written",
            summary_path=(email_output / "email_intake_summary.json").as_posix(),
        )
    else:
        _record(audit_events, "email_intake_summary_written", assignment.assignment_id, status="skipped")

    _record(audit_events, "email_intake_finished", assignment.assignment_id, status="finished" if not dry_run else "dry_run")
    if not dry_run:
        _flush_audit(audit, audit_events)

    return {
        "assignment_id": assignment.assignment_id,
        "summary": summary,
        "messages": processed_messages,
        "accepted_messages": accepted_messages,
        "quarantined_messages": quarantined_messages,
        "rejected_messages": rejected_messages,
        "dry_run_decisions": dry_run_decisions,
        "provenance": provenance_records,
        "drafts": draft_records,
        "email_intake_dir": email_output,
        "staged_dir": staged_dir,
        "audit_log": audit_path,
        "phase1_result": phase1_result,
    }


def _classify_message(
    *,
    message: InboundEmailMessage,
    assignment: Assignment,
    student_match: StudentMatchResult,
    seen_message_ids: set[str],
    seen_attachment_hashes: set[str],
    accepted_student_ids: set[str],
) -> tuple[InboundEmailMessage, list[InboundEmailAttachment], list[InboundEmailAttachment], list[InboundEmailAttachment]]:
    message_reasons = list(message.reasons)
    if message.message_id and message.message_id in seen_message_ids:
        message_reasons.append(reasons.DUPLICATE_MESSAGE_ID)
    if student_match.status == "unknown":
        message_reasons.extend(student_match.reasons)
    elif student_match.status == "ambiguous":
        message_reasons.extend(student_match.reasons)
    if message.received_at is None:
        message_reasons.append(reasons.MISSING_RECEIVED_AT)

    duplicate_hash_seen = any(attachment.sha256 and attachment.sha256 in seen_attachment_hashes for attachment in message.attachments)
    if duplicate_hash_seen:
        message_reasons.append(reasons.DUPLICATE_ATTACHMENT_HASH)
    if student_match.student_id and student_match.student_id in accepted_student_ids:
        message_reasons.append(reasons.RESUBMISSION_DETECTED)

    pdf_attachments = [attachment for attachment in message.attachments if _is_pdf_attachment(attachment)]
    if not pdf_attachments:
        message_reasons.append(reasons.NO_PDF_ATTACHMENT)
    if assignment.max_files_per_student >= 0 and len(pdf_attachments) > assignment.max_files_per_student:
        message_reasons.append(reasons.TOO_MANY_PDF_ATTACHMENTS)

    accepted_attachments: list[InboundEmailAttachment] = []
    quarantined_attachments: list[InboundEmailAttachment] = []
    rejected_attachments: list[InboundEmailAttachment] = []
    blocking_attachment_reasons: list[str] = []
    blocking_message_reasons = {
        reasons.MISSING_MESSAGE_ID,
        reasons.MISSING_RECEIVED_AT,
        reasons.UNKNOWN_STUDENT,
        reasons.AMBIGUOUS_STUDENT_MATCH,
        reasons.DUPLICATE_MESSAGE_ID,
        reasons.DUPLICATE_ATTACHMENT_HASH,
        reasons.NO_PDF_ATTACHMENT,
        reasons.TOO_MANY_PDF_ATTACHMENTS,
    }

    for attachment in message.attachments:
        attachment_reasons = list(attachment.reasons)
        if not _is_pdf_attachment(attachment):
            attachment_reasons.append(reasons.NON_PDF_ATTACHMENT)
        if attachment.sha256 and attachment.sha256 in seen_attachment_hashes:
            attachment_reasons.append(reasons.DUPLICATE_ATTACHMENT_HASH)

        unique_attachment_reasons = _dedupe(attachment_reasons)
        if unique_attachment_reasons:
            blocking_attachment_reasons.extend(
                reason
                for reason in unique_attachment_reasons
                if reason
                in {
                    reasons.ATTACHMENT_MISSING_FILE,
                    reasons.ATTACHMENT_EMPTY,
                    reasons.ATTACHMENT_TOO_LARGE,
                    reasons.UNSAFE_ATTACHMENT_PATH,
                    reasons.DUPLICATE_ATTACHMENT_HASH,
                }
            )
            updated = replace(attachment, status="rejected", reasons=unique_attachment_reasons)
            rejected_attachments.append(updated)
            continue
        if any(reason in blocking_message_reasons for reason in message_reasons):
            updated = replace(attachment, status="quarantined", reasons=_dedupe(message_reasons))
            quarantined_attachments.append(updated)
            continue
        accepted_attachments.append(replace(attachment, status="accepted", reasons=[]))

    unique_message_reasons = _dedupe([*message_reasons, *blocking_attachment_reasons])
    accepted_pdf_count = sum(1 for attachment in accepted_attachments if _is_pdf_attachment(attachment))
    if not unique_message_reasons and accepted_pdf_count > 0:
        status = "accepted"
    elif reasons.MISSING_MESSAGE_ID in unique_message_reasons or reasons.MISSING_RECEIVED_AT in unique_message_reasons:
        status = "rejected"
    else:
        status = "quarantined"

    classified_attachments = [*accepted_attachments, *quarantined_attachments, *rejected_attachments]
    classified_attachments.sort(key=lambda item: item.attachment_index)
    classified_message = replace(
        message,
        status=status,
        reasons=unique_message_reasons,
        attachments=classified_attachments,
        attachment_count=len(classified_attachments),
    )
    return classified_message, accepted_attachments, quarantined_attachments, rejected_attachments


def _record_message_and_attachment_events(
    audit_events: list[dict[str, object]],
    assignment_id: str,
    message: InboundEmailMessage,
    student_match: StudentMatchResult,
    accepted_attachments: list[InboundEmailAttachment],
    quarantined_attachments: list[InboundEmailAttachment],
    rejected_attachments: list[InboundEmailAttachment],
) -> None:
    event_type = {
        "accepted": "email_message_accepted",
        "quarantined": "email_message_quarantined",
        "rejected": "email_message_rejected",
    }[message.status]
    _record(
        audit_events,
        event_type,
        assignment_id,
        message=message,
        student_id=student_match.student_id,
        status=message.status,
        reasons=message.reasons,
    )
    for attachment in accepted_attachments:
        _record(
            audit_events,
            "email_attachment_accepted",
            assignment_id,
            message=message,
            attachment=attachment,
            student_id=student_match.student_id,
            status="accepted",
            reasons=attachment.reasons,
        )
    for attachment in quarantined_attachments:
        _record(
            audit_events,
            "email_attachment_quarantined",
            assignment_id,
            message=message,
            attachment=attachment,
            student_id=student_match.student_id,
            status="quarantined",
            reasons=attachment.reasons,
        )
    for attachment in rejected_attachments:
        _record(
            audit_events,
            "email_attachment_rejected",
            assignment_id,
            message=message,
            attachment=attachment,
            student_id=student_match.student_id,
            status="rejected",
            reasons=attachment.reasons,
        )


def _message_payload(message: InboundEmailMessage, student_match: StudentMatchResult) -> dict[str, object]:
    payload = dataclass_to_json_dict(message)
    payload["student_match"] = dataclass_to_json_dict(student_match)
    return payload


def _decision_payload(decision: EmailIntakeDryRunDecision, student_match: StudentMatchResult) -> dict[str, object]:
    payload = dataclass_to_json_dict(decision)
    payload["student_match"] = dataclass_to_json_dict(student_match)
    return payload


def _provenance_payload(
    *,
    provenance: EmailSubmissionProvenance,
    staged_filename: str,
    student_id: str,
    match_source: str,
) -> dict[str, object]:
    payload = dataclass_to_json_dict(provenance)
    payload["staged_filename"] = staged_filename
    payload["phase1_staged_filename"] = staged_filename
    payload["student_id"] = student_id
    payload["student_match_source"] = match_source
    return payload


def _draft_payload(
    assignment: Assignment,
    roster_by_id: dict[str, Student],
    message: InboundEmailMessage,
    student_match: StudentMatchResult,
) -> dict[str, object]:
    student = roster_by_id.get(student_match.student_id)
    if message.status == "accepted":
        draft_type = "email_acknowledgement"
        body = f"Draft only. We received your email submission for {assignment.title}."
    else:
        draft_type = "email_resend_needed"
        reason_text = ", ".join(message.reasons) or "quarantine_review_required"
        body = f"Draft only. Your email submission for {assignment.title} needs review or resend. Reason: {reason_text}."
    return {
        "draft_id": f"{assignment.assignment_id}:{message.message_id or 'missing-message-id'}:{draft_type}",
        "assignment_id": assignment.assignment_id,
        "message_id": message.message_id,
        "thread_id": message.thread_id,
        "student_id": student_match.student_id,
        "draft_type": draft_type,
        "recipient_email": student.email if student is not None else message.from_email,
        "subject": f"Draft: {assignment.title}",
        "body_text": body,
        "send_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reasons": message.reasons,
    }


def _stage_filename(
    student_id: str,
    message_id: str,
    attachment: InboundEmailAttachment,
    existing: set[str],
) -> str:
    safe_message_id = _safe_token(message_id or "missing-message-id")
    original_stem = _safe_token(Path(attachment.filename).stem or "attachment")
    suffix = Path(attachment.filename).suffix.lower() or ".pdf"
    base = f"{student_id}_{safe_message_id}_attachment-{attachment.attachment_index}_{original_stem}{suffix}"
    candidate = base
    counter = 1
    while candidate in existing:
        candidate = f"{Path(base).stem}_{counter}{suffix}"
        counter += 1
    return candidate


def _quarantine_filename(message_id: str, attachment: InboundEmailAttachment) -> str:
    safe_message_id = _safe_token(message_id or "missing-message-id")
    return f"{safe_message_id}_attachment-{attachment.attachment_index}_{_safe_token(attachment.filename)}"


def _safe_token(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    token = "".join(allowed).strip("._")
    return token or "unknown"


def _match_source(student_match: StudentMatchResult) -> str:
    if not student_match.candidates:
        return ""
    order = {
        "attachment_filename_student_id": 0,
        "subject_student_id": 1,
        "sender_email": 2,
        "body_student_id": 3,
    }
    return sorted(student_match.candidates, key=lambda item: order.get(item.match_source, 99))[0].match_source


def _is_pdf_attachment(attachment: InboundEmailAttachment) -> bool:
    filename_is_pdf = attachment.filename.lower().endswith(".pdf")
    content_type_is_pdf = attachment.content_type.lower() in {"application/pdf", "application/x-pdf"}
    return filename_is_pdf or content_type_is_pdf


def _record(
    events: list[dict[str, object]],
    event_type: str,
    assignment_id: str,
    *,
    message: InboundEmailMessage | None = None,
    attachment: InboundEmailAttachment | None = None,
    student_id: str = "",
    status: str = "",
    reasons: list[str] | None = None,
    **extra: object,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "assignment_id": assignment_id,
        "message_id": message.message_id if message is not None else "",
        "thread_id": message.thread_id if message is not None else "",
        "student_id": student_id,
        "attachment_id": attachment.attachment_id if attachment is not None else "",
        "status": status,
        "reasons": reasons or [],
    }
    event.update(extra)
    events.append(event)


def _flush_audit(audit: AuditLog, events: list[dict[str, object]]) -> None:
    for event in events:
        event_type = str(event.pop("event_type"))
        audit.write(
            event_type,
            student_id=str(event.pop("student_id", "")),
            status=str(event.pop("status", "")),
            reasons=list(event.pop("reasons", [])),
            **event,
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
