from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from exam_bank.submissions.ingest import _require_private_roots

OUTGOING_APPROVAL_STATUSES = {"draft", "approved", "blocked", "sent", "send_failed"}
OUTGOING_DELIVERY_MODES = {"dry_run", "fake_adapter", "live_adapter_disabled"}
OUTGOING_QUEUE_STATUSES = {"queued", "blocked", "dry_run_ready", "sent", "failed"}

ALLOWED_MESSAGE_TYPES = {
    "submission_acknowledgement",
    "resend_request",
    "missing_submission_reminder",
    "late_submission_notice",
    "teacher_review_notice",
    "generic_teacher_approved_feedback",
}
BLOCKED_MESSAGE_TYPES = {
    "final_grade",
    "ai_generated_score_feedback",
    "draft_auto_grading_feedback",
}

APPROVAL_TEMPLATE_FIELDS = [
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


@dataclass(frozen=True)
class OutgoingEmailDraft:
    draft_id: str
    assignment_id: str
    student_id: str
    recipient_email: str
    subject: str
    body_text: str
    message_type: str
    source_phase: str
    send_allowed: bool
    approval_status: str
    approved_by: str
    approved_at: str
    blocked_reasons: list[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        _require_allowed("outgoing approval status", self.approval_status, OUTGOING_APPROVAL_STATUSES)
        if self.approval_status != "approved" and self.send_allowed:
            raise ValueError("Outgoing email drafts may set send_allowed=true only after approval")


@dataclass(frozen=True)
class OutgoingEmailQueueItem:
    queue_id: str
    draft_id: str
    assignment_id: str
    student_id: str
    recipient_email: str
    subject: str
    body_text: str
    message_type: str
    approval_status: str
    send_allowed: bool
    delivery_mode: str
    status: str
    created_at: str
    sent_at: str
    send_attempts: int
    last_error: str

    def __post_init__(self) -> None:
        _require_allowed("outgoing approval status", self.approval_status, OUTGOING_APPROVAL_STATUSES)
        _require_allowed("outgoing delivery mode", self.delivery_mode, OUTGOING_DELIVERY_MODES)
        _require_allowed("outgoing queue status", self.status, OUTGOING_QUEUE_STATUSES)
        if self.send_attempts < 0:
            raise ValueError("Outgoing email queue send_attempts must be non-negative")
        if self.approval_status != "approved" and self.send_allowed:
            raise ValueError("Outgoing queue items may set send_allowed=true only after approval")


@dataclass(frozen=True)
class OutgoingEmailAuditEvent:
    timestamp: str
    event_type: str
    assignment_id: str
    draft_id: str
    queue_id: str
    student_id: str
    recipient_email: str
    status: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OutgoingEmailSummary:
    assignment_id: str
    drafts_seen: int
    approved_count: int
    blocked_count: int
    queued_count: int
    dry_run_ready_count: int
    sent_count: int
    failed_count: int
    created_at: str


def build_outgoing_email_queue(
    *,
    assignment_id: str,
    submission_output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
    approval_csv: Path | None = None,
) -> dict[str, object]:
    _require_private_roots(submission_output_root, reports_root)
    assignment_output = submission_output_root / assignment_id
    outgoing_dir = assignment_output / "outgoing_email"
    outgoing_dir.mkdir(parents=True, exist_ok=True)

    created_at = _now()
    audit_events: list[OutgoingEmailAuditEvent] = []
    drafts = load_outgoing_email_drafts(assignment_id=assignment_id, submission_output_root=submission_output_root)
    for draft in drafts:
        audit_events.append(_audit("outgoing_draft_normalized", draft=draft, status=draft.approval_status, reasons=draft.blocked_reasons))

    approval_template_path = outgoing_dir / "approval_template.csv"
    write_approval_template(drafts, approval_template_path)
    audit_events.append(_audit("outgoing_approval_template_written", assignment_id=assignment_id, status="written"))

    approval_rows = _read_approval_rows(approval_csv) if approval_csv is not None else {}
    approved_drafts: list[OutgoingEmailDraft] = []
    blocked_drafts: list[OutgoingEmailDraft] = []
    queue: list[OutgoingEmailQueueItem] = []

    known_draft_ids = {draft.draft_id for draft in drafts}
    for unknown_draft_id in sorted(approval_rows.keys() - known_draft_ids):
        row = approval_rows[unknown_draft_id]
        audit_events.append(
            _audit(
                "outgoing_approval_rejected",
                assignment_id=str(row.get("assignment_id") or assignment_id),
                draft_id=unknown_draft_id,
                student_id=str(row.get("student_id", "")),
                recipient_email=str(row.get("recipient_email", "")),
                status="blocked",
                reasons=["unknown_draft_id"],
            )
        )

    for draft in drafts:
        approval_row = approval_rows.get(draft.draft_id)
        approved, reasons = _approval_reasons(draft, approval_row)
        if approved:
            approved_draft = OutgoingEmailDraft(
                draft_id=draft.draft_id,
                assignment_id=draft.assignment_id,
                student_id=draft.student_id,
                recipient_email=draft.recipient_email,
                subject=draft.subject,
                body_text=draft.body_text,
                message_type=draft.message_type,
                source_phase=draft.source_phase,
                send_allowed=True,
                approval_status="approved",
                approved_by=str(approval_row.get("approved_by", "")).strip(),
                approved_at=created_at,
                blocked_reasons=[],
                created_at=draft.created_at,
            )
            approved_drafts.append(approved_draft)
            item = _queue_item(approved_draft, created_at)
            queue.append(item)
            audit_events.append(_audit("outgoing_draft_approved", draft=approved_draft, status="approved"))
            audit_events.append(_audit("outgoing_queue_item_created", draft=approved_draft, queue_id=item.queue_id, status=item.status))
            continue

        blocked = OutgoingEmailDraft(
            draft_id=draft.draft_id,
            assignment_id=draft.assignment_id,
            student_id=draft.student_id,
            recipient_email=draft.recipient_email,
            subject=draft.subject,
            body_text=draft.body_text,
            message_type=draft.message_type,
            source_phase=draft.source_phase,
            send_allowed=False,
            approval_status="blocked",
            approved_by="",
            approved_at="",
            blocked_reasons=_dedupe([*draft.blocked_reasons, *reasons]),
            created_at=draft.created_at,
        )
        blocked_drafts.append(blocked)
        audit_events.append(_audit("outgoing_draft_blocked", draft=blocked, status="blocked", reasons=blocked.blocked_reasons))

    summary = OutgoingEmailSummary(
        assignment_id=assignment_id,
        drafts_seen=len(drafts),
        approved_count=len(approved_drafts),
        blocked_count=len(blocked_drafts),
        queued_count=len(queue),
        dry_run_ready_count=sum(1 for item in queue if item.status == "dry_run_ready"),
        sent_count=sum(1 for item in queue if item.status == "sent"),
        failed_count=sum(1 for item in queue if item.status == "failed"),
        created_at=created_at,
    )

    approved_path = outgoing_dir / "approved_drafts.json"
    blocked_path = outgoing_dir / "blocked_drafts.json"
    queue_path = outgoing_dir / "outgoing_queue.json"
    summary_path = outgoing_dir / "outgoing_email_summary.json"
    audit_path = outgoing_dir / "outgoing_email_audit.jsonl"
    _write_json_records(approved_path, approved_drafts)
    _write_json_records(blocked_path, blocked_drafts)
    _write_json_records(queue_path, queue)
    _write_json(summary_path, summary)
    audit_events.append(_audit("outgoing_email_summary_written", assignment_id=assignment_id, status="written"))
    _write_jsonl(audit_path, audit_events)

    return {
        "assignment_id": assignment_id,
        "approval_template": approval_template_path,
        "approved_drafts": approved_path,
        "blocked_drafts": blocked_path,
        "outgoing_queue": queue_path,
        "outgoing_email_summary": summary_path,
        "outgoing_email_audit": audit_path,
        "drafts": drafts,
        "approved": approved_drafts,
        "blocked": blocked_drafts,
        "queue": queue,
        "summary": summary,
    }


def load_outgoing_email_drafts(*, assignment_id: str, submission_output_root: Path) -> list[OutgoingEmailDraft]:
    assignment_output = submission_output_root / assignment_id
    candidates = [
        *(assignment_output / "drafts").glob("*.jsonl"),
        *(assignment_output / "email_intake" / "drafts").glob("*.jsonl"),
    ]
    drafts: list[OutgoingEmailDraft] = []
    for path in sorted(candidate for candidate in candidates if candidate.is_file()):
        source_phase = _source_phase_for_path(path)
        for payload in _read_jsonl(path):
            if str(payload.get("assignment_id", "")) != assignment_id:
                continue
            drafts.append(normalize_outgoing_email_draft(payload, source_phase=source_phase))
    return sorted(drafts, key=lambda draft: draft.draft_id)


def normalize_outgoing_email_draft(payload: dict[str, object], *, source_phase: str) -> OutgoingEmailDraft:
    draft_type = str(payload.get("message_type") or payload.get("draft_type") or "")
    message_type = _message_type(draft_type)
    reasons: list[str] = []
    if message_type in BLOCKED_MESSAGE_TYPES:
        reasons.append("blocked_message_type")
    if message_type not in ALLOWED_MESSAGE_TYPES:
        reasons.append("unsupported_message_type")
    if _is_phase3_draft_auto(payload, source_phase) and message_type != "generic_teacher_approved_feedback":
        reasons.append("phase3_draft_auto_feedback_blocked")
    if bool(payload.get("send_allowed", False)):
        reasons.append("source_draft_send_allowed_ignored")

    return OutgoingEmailDraft(
        draft_id=str(payload.get("draft_id", "")).strip(),
        assignment_id=str(payload.get("assignment_id", "")).strip(),
        student_id=str(payload.get("student_id", "")).strip(),
        recipient_email=str(payload.get("recipient_email", "")).strip(),
        subject=str(payload.get("subject", "")).strip(),
        body_text=str(payload.get("body_text", "")),
        message_type=message_type,
        source_phase=source_phase,
        send_allowed=False,
        approval_status="blocked" if reasons else "draft",
        approved_by="",
        approved_at="",
        blocked_reasons=_dedupe(reasons),
        created_at=str(payload.get("created_at", "")).strip(),
    )


def write_approval_template(drafts: list[OutgoingEmailDraft], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=APPROVAL_TEMPLATE_FIELDS)
        writer.writeheader()
        for draft in drafts:
            writer.writerow(
                {
                    "draft_id": draft.draft_id,
                    "assignment_id": draft.assignment_id,
                    "student_id": draft.student_id,
                    "recipient_email": draft.recipient_email,
                    "message_type": draft.message_type,
                    "subject": draft.subject,
                    "approval_status": "blocked" if draft.blocked_reasons else "draft",
                    "approved_by": "",
                    "teacher_note": "",
                }
            )


def write_outgoing_email_dry_run_report(
    *,
    assignment_id: str,
    submission_output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
) -> dict[str, object]:
    _require_private_roots(submission_output_root, reports_root)
    queue_path = submission_output_root / assignment_id / "outgoing_email" / "outgoing_queue.json"
    items = _load_queue(queue_path)
    report_path = reports_root / f"{assignment_id}_outgoing_email_dry_run.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "queue_id",
        "draft_id",
        "student_id",
        "recipient_email",
        "subject",
        "message_type",
        "status",
        "would_send",
        "blocked_reasons",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            would_send = item.approval_status == "approved" and item.send_allowed and item.status in {"dry_run_ready", "queued"}
            writer.writerow(
                {
                    "queue_id": item.queue_id,
                    "draft_id": item.draft_id,
                    "student_id": item.student_id,
                    "recipient_email": item.recipient_email,
                    "subject": item.subject,
                    "message_type": item.message_type,
                    "status": item.status,
                    "would_send": str(would_send).lower(),
                    "blocked_reasons": "" if would_send else "not_send_eligible",
                }
            )
    return {"assignment_id": assignment_id, "dry_run_report": report_path, "rows": len(items)}


def send_outgoing_email_fake(
    *,
    assignment_id: str,
    submission_output_root: Path = Path("output/submissions"),
    use_fake_adapter: bool = False,
) -> dict[str, object]:
    if not use_fake_adapter:
        raise ValueError("Fake outgoing email adapter requires --use-fake-adapter")
    _require_submission_output_root(submission_output_root)
    queue_path = submission_output_root / assignment_id / "outgoing_email" / "outgoing_queue.json"
    items = _load_queue(queue_path)
    eligible = [
        item
        for item in items
        if item.approval_status == "approved" and item.send_allowed and item.status in {"dry_run_ready", "queued"}
    ]
    fake_path = submission_output_root / assignment_id / "outgoing_email" / "fake_sent_messages.jsonl"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    with fake_path.open("w", encoding="utf-8") as handle:
        for item in eligible:
            handle.write(
                json.dumps(
                    {
                        "timestamp": _now(),
                        "adapter": "fake_adapter",
                        "queue_id": item.queue_id,
                        "draft_id": item.draft_id,
                        "assignment_id": item.assignment_id,
                        "student_id": item.student_id,
                        "recipient_email": item.recipient_email,
                        "subject": item.subject,
                        "body_text": item.body_text,
                        "message_type": item.message_type,
                        "network_sent": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return {"assignment_id": assignment_id, "fake_sent_messages": fake_path, "sent_count": len(eligible)}


def _approval_reasons(draft: OutgoingEmailDraft, approval_row: dict[str, str] | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if draft.blocked_reasons:
        reasons.extend(draft.blocked_reasons)
    if approval_row is None:
        reasons.append("approval_required")
    elif str(approval_row.get("approval_status", "")).strip().lower() != "approved":
        reasons.append("approval_required")
    elif not str(approval_row.get("approved_by", "")).strip():
        reasons.append("approved_by_required")
    if approval_row is not None:
        if str(approval_row.get("recipient_email", "")).strip() != draft.recipient_email:
            reasons.append("recipient_email_mismatch")
        if str(approval_row.get("assignment_id", "")).strip() != draft.assignment_id:
            reasons.append("assignment_id_mismatch")
        if str(approval_row.get("student_id", "")).strip() != draft.student_id:
            reasons.append("student_id_mismatch")
    if draft.message_type in BLOCKED_MESSAGE_TYPES:
        reasons.append("blocked_message_type")
    if draft.message_type not in ALLOWED_MESSAGE_TYPES:
        reasons.append("unsupported_message_type")
    return not reasons, _dedupe(reasons)


def _queue_item(draft: OutgoingEmailDraft, created_at: str) -> OutgoingEmailQueueItem:
    return OutgoingEmailQueueItem(
        queue_id=f"{draft.draft_id}:outgoing",
        draft_id=draft.draft_id,
        assignment_id=draft.assignment_id,
        student_id=draft.student_id,
        recipient_email=draft.recipient_email,
        subject=draft.subject,
        body_text=draft.body_text,
        message_type=draft.message_type,
        approval_status=draft.approval_status,
        send_allowed=draft.send_allowed,
        delivery_mode="dry_run",
        status="dry_run_ready",
        created_at=created_at,
        sent_at="",
        send_attempts=0,
        last_error="",
    )


def _message_type(draft_type: str) -> str:
    mapping = {
        "acknowledgement": "submission_acknowledgement",
        "email_acknowledgement": "submission_acknowledgement",
        "resend": "resend_request",
        "email_resend_needed": "resend_request",
        "missing_reminder": "missing_submission_reminder",
    }
    return mapping.get(draft_type, draft_type)


def _source_phase_for_path(path: Path) -> str:
    parts = set(path.parts)
    if "email_intake" in parts:
        return "phase4_email_intake"
    if "drafts" in parts:
        return "phase1_submission_tracker"
    return "unknown"


def _is_phase3_draft_auto(payload: dict[str, object], source_phase: str) -> bool:
    return (
        source_phase == "phase3_draft_auto_grading"
        or str(payload.get("source_phase", "")) == "phase3_draft_auto_grading"
        or str(payload.get("grading_mode", "")) == "draft_auto"
        or str(payload.get("draft_type", "")) in {"draft_auto_grading_feedback", "ai_generated_score_feedback"}
    )


def _read_approval_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row.get("draft_id", "")).strip(): row for row in rows if str(row.get("draft_id", "")).strip()}


def _load_queue(path: Path) -> list[OutgoingEmailQueueItem]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [OutgoingEmailQueueItem(**item) for item in payload]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_payload(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_records(path: Path, records: list[object]) -> None:
    _write_json(path, [_json_payload(record) for record in records])


def _write_jsonl(path: Path, records: list[OutgoingEmailAuditEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_payload(record), sort_keys=True) + "\n")


def _json_payload(payload: object) -> object:
    if hasattr(payload, "__dataclass_fields__"):
        return asdict(payload)
    return payload


def _audit(
    event_type: str,
    *,
    assignment_id: str = "",
    draft: OutgoingEmailDraft | None = None,
    draft_id: str = "",
    queue_id: str = "",
    student_id: str = "",
    recipient_email: str = "",
    status: str = "",
    reasons: list[str] | None = None,
) -> OutgoingEmailAuditEvent:
    return OutgoingEmailAuditEvent(
        timestamp=_now(),
        event_type=event_type,
        assignment_id=draft.assignment_id if draft is not None else assignment_id,
        draft_id=draft.draft_id if draft is not None else draft_id,
        queue_id=queue_id,
        student_id=draft.student_id if draft is not None else student_id,
        recipient_email=draft.recipient_email if draft is not None else recipient_email,
        status=status,
        reasons=reasons or [],
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_submission_output_root(output_root: Path) -> None:
    output_parts = output_root.parts
    if len(output_parts) < 2 or output_parts[-2:] != ("output", "submissions"):
        raise ValueError("Submission output_root must end with output/submissions")


def _require_allowed(label: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid {label}: {value}")
