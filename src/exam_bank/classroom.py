from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import fitz

from exam_bank.emailing.mailapp import MailAppEmailProvider
from exam_bank.emailing.providers import EmailProvider
from exam_bank.mupdf_tools import quiet_mupdf
from exam_bank.submissions.ingest import ingest_assignment_submissions, load_assignment, load_roster, parse_datetime

quiet_mupdf(fitz)

DEFAULT_CLASSES_ROOT = Path("data/classes")
DEFAULT_SUBMISSION_OUTPUT_ROOT = Path("output/submissions")
DEFAULT_SUBMISSION_REPORTS_ROOT = Path("reports/submissions")
ROSTER_FIELDS = [
    "student_id",
    "class_id",
    "display_name",
    "email",
    "active",
    "source_file",
    "last_assignment_id",
    "last_assignment_status",
    "last_submitted_at",
    "last_questions_completed",
    "last_notes",
]
SCHEDULE_STATUSES = {"scheduled", "dry_run", "sent", "skipped", "failed"}


@dataclass(frozen=True)
class ClassPaths:
    class_dir: Path
    roster_path: Path
    assignments_dir: Path


def class_paths(class_id: str, *, classes_root: Path = DEFAULT_CLASSES_ROOT) -> ClassPaths:
    class_dir = classes_root / _safe_segment(class_id)
    return ClassPaths(
        class_dir=class_dir,
        roster_path=class_dir / "roster.csv",
        assignments_dir=class_dir / "assignments",
    )


def init_class_workspace(
    *,
    class_id: str,
    roster_source: Path | None = None,
    classes_root: Path = DEFAULT_CLASSES_ROOT,
) -> dict[str, object]:
    paths = class_paths(class_id, classes_root=classes_root)
    paths.assignments_dir.mkdir(parents=True, exist_ok=True)
    if roster_source is not None:
        _copy_normalized_roster(roster_source, paths.roster_path, class_id=class_id)
    elif not paths.roster_path.exists():
        _write_roster_rows(paths.roster_path, [])
    return {
        "class_id": class_id,
        "class_dir": paths.class_dir,
        "roster": paths.roster_path,
        "assignments_dir": paths.assignments_dir,
    }


def add_assignment(
    *,
    class_id: str,
    pdf_path: Path,
    assignment_id: str,
    title: str,
    due_at: datetime,
    send_at: datetime,
    classes_root: Path = DEFAULT_CLASSES_ROOT,
    course_id: str = "local_course",
    timezone_name: str = "Asia/Shanghai",
) -> dict[str, object]:
    paths = class_paths(class_id, classes_root=classes_root)
    if not paths.roster_path.exists():
        init_class_workspace(class_id=class_id, classes_root=classes_root)
    assignment_dir = paths.assignments_dir / _safe_segment(assignment_id)
    inbox_dir = assignment_dir / "inbox"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    inbox_dir.mkdir(parents=True, exist_ok=True)

    stored_pdf = assignment_dir / "assignment.pdf"
    shutil.copy2(pdf_path, stored_pdf)
    assignment_payload = {
        "assignment_id": assignment_id,
        "course_id": course_id,
        "title": title,
        "class_id": class_id,
        "due_at": due_at.isoformat(),
        "timezone": timezone_name,
        "accepted_file_types": ["pdf"],
        "max_files_per_student": 1,
        "max_file_size_mb": 50,
        "allow_late": True,
        "source_question_ids": [],
    }
    assignment_json = assignment_dir / "assignment.json"
    _write_json(assignment_json, assignment_payload)
    schedule_path = build_assignment_schedule(
        class_id=class_id,
        assignment_id=assignment_id,
        classes_root=classes_root,
        send_at=send_at,
    )["schedule"]
    return {
        "class_id": class_id,
        "assignment_id": assignment_id,
        "assignment_dir": assignment_dir,
        "assignment_pdf": stored_pdf,
        "assignment_json": assignment_json,
        "inbox": inbox_dir,
        "schedule": schedule_path,
    }


def build_assignment_schedule(
    *,
    class_id: str,
    assignment_id: str,
    classes_root: Path = DEFAULT_CLASSES_ROOT,
    send_at: datetime | None = None,
) -> dict[str, object]:
    paths = class_paths(class_id, classes_root=classes_root)
    assignment_dir = paths.assignments_dir / _safe_segment(assignment_id)
    assignment_path = assignment_dir / "assignment.json"
    assignment = load_assignment(assignment_path)
    roster = [student for student in load_roster(paths.roster_path) if student.class_id == class_id and student.active]
    effective_send_at = send_at or datetime.now(timezone.utc)
    schedule: list[dict[str, object]] = []
    for student in roster:
        if not student.email:
            schedule.append(_schedule_item(assignment_id, student.student_id, "", "assignment_distribution", effective_send_at, "skipped", "missing_student_email"))
            continue
        schedule.append(
            _schedule_item(
                assignment_id,
                student.student_id,
                student.email,
                "assignment_distribution",
                effective_send_at,
                "scheduled",
                "",
                subject=f"{assignment.title}",
                body_text=(
                    f"Hello {student.display_name},\n\n"
                    f"Please complete the attached assignment: {assignment.title}.\n"
                    f"Due: {assignment.due_at.isoformat() if assignment.due_at else 'not set'}.\n\n"
                    "Reply to this email with your completed PDF attached."
                ),
                attachment_path=(assignment_dir / "assignment.pdf").as_posix(),
            )
        )
        if assignment.due_at is not None:
            for hours in (48, 24, 12):
                scheduled_at = assignment.due_at - timedelta(hours=hours)
                schedule.append(
                    _schedule_item(
                        assignment_id,
                        student.student_id,
                        student.email,
                        f"reminder_{hours}h",
                        scheduled_at,
                        "scheduled",
                        "",
                        subject=f"Reminder: {assignment.title} due in {hours} hours",
                        body_text=(
                            f"Hello {student.display_name},\n\n"
                            f"Our records do not yet show your submission for {assignment.title}. "
                            f"It is due at {assignment.due_at.isoformat()}."
                        ),
                    )
                )
    schedule_path = assignment_dir / "message_schedule.json"
    _write_json(schedule_path, schedule)
    return {"class_id": class_id, "assignment_id": assignment_id, "schedule": schedule_path, "scheduled_count": len(schedule)}


def dispatch_due_messages(
    *,
    class_id: str,
    assignment_id: str,
    now: datetime | None = None,
    send_live: bool = False,
    from_address: str | None = None,
    provider: EmailProvider | None = None,
    classes_root: Path = DEFAULT_CLASSES_ROOT,
) -> dict[str, object]:
    paths = class_paths(class_id, classes_root=classes_root)
    assignment_dir = paths.assignments_dir / _safe_segment(assignment_id)
    schedule_path = assignment_dir / "message_schedule.json"
    schedule = _read_json_list(schedule_path)
    current_time = now or datetime.now(timezone.utc)
    completed_student_ids = _completed_student_ids(assignment_id=assignment_id)
    mail_provider = provider or MailAppEmailProvider(requested_from_address=from_address)
    sent = dry_run = skipped = failed = 0
    audit_events: list[dict[str, object]] = []

    for item in schedule:
        if str(item.get("status")) not in {"scheduled", "failed"}:
            continue
        scheduled_at = parse_datetime(str(item["scheduled_at"]))
        if scheduled_at > current_time:
            continue
        student_id = str(item.get("student_id") or "")
        message_type = str(item.get("message_type") or "")
        if message_type.startswith("reminder_") and student_id in completed_student_ids:
            item["status"] = "skipped"
            item["last_error"] = "student_already_submitted"
            skipped += 1
            continue
        recipient = str(item.get("recipient_email") or "")
        if not recipient:
            item["status"] = "skipped"
            item["last_error"] = "missing_recipient_email"
            skipped += 1
            continue
        attachments = [Path(str(item["attachment_path"]))] if item.get("attachment_path") else None
        if not send_live:
            dry_run += 1
            _append_dispatch_audit(audit_events, item, status="dry_run", error_code="")
            continue
        result = mail_provider.send_message(
            to=recipient,
            subject=str(item.get("subject") or ""),
            body_text=str(item.get("body_text") or ""),
            from_address=from_address,
            attachments=attachments,
        )
        item["status"] = "sent" if result.sent else "failed"
        item["sent_at"] = result.sent_at.isoformat() if result.sent else ""
        item["last_error"] = result.error_code or ""
        if result.sent:
            sent += 1
        else:
            failed += 1
        _append_dispatch_audit(audit_events, item, status=str(item["status"]), error_code=result.error_code or "")

    _write_json(schedule_path, schedule)
    audit_path = assignment_dir / "message_dispatch_audit.jsonl"
    _append_jsonl(audit_path, audit_events)
    return {
        "class_id": class_id,
        "assignment_id": assignment_id,
        "schedule": schedule_path,
        "audit_log": audit_path,
        "sent": sent,
        "dry_run": dry_run,
        "skipped": skipped,
        "failed": failed,
    }


def ingest_class_assignment(
    *,
    class_id: str,
    assignment_id: str,
    classes_root: Path = DEFAULT_CLASSES_ROOT,
    output_root: Path = DEFAULT_SUBMISSION_OUTPUT_ROOT,
    reports_root: Path = DEFAULT_SUBMISSION_REPORTS_ROOT,
    send_acknowledgements: bool = False,
    from_address: str | None = None,
    provider: EmailProvider | None = None,
    import_from_mailapp: bool = False,
    mail_query: str | None = None,
    mail_limit: int = 50,
) -> dict[str, object]:
    paths = class_paths(class_id, classes_root=classes_root)
    assignment_dir = paths.assignments_dir / _safe_segment(assignment_id)
    imported_attachments: list[Path] = []
    if import_from_mailapp:
        mail_provider = provider if isinstance(provider, MailAppEmailProvider) else MailAppEmailProvider(requested_from_address=from_address)
        imported_attachments = mail_provider.export_pdf_attachments(
            query=mail_query or assignment_id,
            target_dir=assignment_dir / "inbox",
            limit=mail_limit,
        )
    result = ingest_assignment_submissions(
        assignment_path=assignment_dir / "assignment.json",
        roster_path=paths.roster_path,
        submissions_dir=assignment_dir / "inbox",
        output_root=output_root,
        reports_root=reports_root,
    )
    summary_path = write_completion_summary(
        assignment_id=assignment_id,
        output_root=output_root,
        reports_root=reports_root,
    )
    update_class_roster_from_completion(
        class_id=class_id,
        assignment_id=assignment_id,
        classes_root=classes_root,
        completion_report=Path(result["completion_report"]),
        completion_summary=summary_path,
    )
    ack_result = {"sent": 0, "failed": 0}
    if send_acknowledgements:
        ack_result = send_submission_acknowledgements(
            class_id=class_id,
            assignment_id=assignment_id,
            completion_report=Path(result["completion_report"]),
            from_address=from_address,
            provider=provider,
        )
    return {
        "class_id": class_id,
        "assignment_id": assignment_id,
        "manifest": result["manifest"],
        "completion_report": result["completion_report"],
        "completion_summary": summary_path,
        "accepted_count": len(result["accepted"]),
        "rejected_count": len(result["rejected"]),
        "imported_attachments": [path.as_posix() for path in imported_attachments],
        "acknowledgements": ack_result,
    }


def write_completion_summary(*, assignment_id: str, output_root: Path, reports_root: Path) -> Path:
    manifest = json.loads((output_root / assignment_id / "manifest.json").read_text(encoding="utf-8"))
    accepted = [item for item in manifest.get("accepted_submissions", []) if isinstance(item, dict)]
    summaries: list[dict[str, object]] = []
    for submission in accepted:
        questions, notes = _completed_questions(Path(str(submission.get("stored_pdf_path") or "")))
        summaries.append(
            {
                "assignment_id": assignment_id,
                "student_id": str(submission.get("student_id") or ""),
                "submission_id": str(submission.get("submission_id") or ""),
                "source_filename": str(submission.get("source_filename") or ""),
                "completed_questions": questions,
                "notes": notes,
                "teacher_review_required": True,
                "student_facing": False,
            }
        )
    path = output_root / assignment_id / "classroom_completion_summary.json"
    _write_json(path, summaries)
    csv_path = reports_root / f"{assignment_id}_classroom_completion_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["assignment_id", "student_id", "submission_id", "source_filename", "completed_questions", "notes"])
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "assignment_id": row["assignment_id"],
                    "student_id": row["student_id"],
                    "submission_id": row["submission_id"],
                    "source_filename": row["source_filename"],
                    "completed_questions": ";".join(row["completed_questions"]),
                    "notes": ";".join(row["notes"]),
                }
            )
    return path


def update_class_roster_from_completion(
    *,
    class_id: str,
    assignment_id: str,
    completion_report: Path,
    completion_summary: Path,
    classes_root: Path = DEFAULT_CLASSES_ROOT,
) -> Path:
    paths = class_paths(class_id, classes_root=classes_root)
    roster_rows = _read_roster_rows(paths.roster_path)
    with completion_report.open("r", encoding="utf-8", newline="") as handle:
        completion_rows = {row["student_id"]: row for row in csv.DictReader(handle)}
    summary_payload = json.loads(completion_summary.read_text(encoding="utf-8")) if completion_summary.exists() else []
    questions_by_student = {
        str(row.get("student_id") or ""): ";".join(str(item) for item in row.get("completed_questions", []))
        for row in summary_payload
        if isinstance(row, dict)
    }
    for row in roster_rows:
        if row.get("class_id") != class_id:
            continue
        completion = completion_rows.get(str(row.get("student_id") or ""))
        if not completion:
            continue
        row["last_assignment_id"] = assignment_id
        row["last_assignment_status"] = completion.get("status", "")
        row["last_submitted_at"] = completion.get("submitted_at", "")
        row["last_questions_completed"] = questions_by_student.get(str(row.get("student_id") or ""), "")
        notes = [completion.get("notes", ""), completion.get("rejection_reasons", "")]
        row["last_notes"] = ";".join(item for item in notes if item)
    _write_roster_rows(paths.roster_path, roster_rows)
    return paths.roster_path


def send_submission_acknowledgements(
    *,
    class_id: str,
    assignment_id: str,
    completion_report: Path,
    from_address: str | None = None,
    provider: EmailProvider | None = None,
) -> dict[str, int]:
    mail_provider = provider or MailAppEmailProvider(requested_from_address=from_address)
    sent = failed = 0
    with completion_report.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = row.get("status", "")
            if status not in {"submitted", "late", "rejected"}:
                continue
            email = row.get("email", "")
            if not email:
                continue
            if status == "rejected":
                body = f"We received your email for {assignment_id}, but the attached work needs review/resubmission. Reason: {row.get('rejection_reasons', 'validation_failed')}."
            else:
                body = f"We received your submission for {assignment_id}: {row.get('source_filename', '')}."
            result = mail_provider.send_message(
                to=email,
                subject=f"Received: {assignment_id}",
                body_text=body,
                from_address=from_address,
            )
            if result.sent:
                sent += 1
            else:
                failed += 1
    return {"sent": sent, "failed": failed}


def parse_cli_datetime(value: str) -> datetime:
    return parse_datetime(value)


def _copy_normalized_roster(source: Path, target: Path, *, class_id: str) -> None:
    rows = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {field: str(row.get(field, "")).strip() for field in ROSTER_FIELDS}
            normalized["class_id"] = normalized["class_id"] or class_id
            normalized["active"] = normalized["active"] or "true"
            rows.append(normalized)
    _write_roster_rows(target, rows)


def _read_roster_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{field: str(row.get(field, "")).strip() for field in ROSTER_FIELDS} for row in csv.DictReader(handle)]


def _write_roster_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROSTER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ROSTER_FIELDS})


def _schedule_item(
    assignment_id: str,
    student_id: str,
    recipient_email: str,
    message_type: str,
    scheduled_at: datetime,
    status: str,
    last_error: str,
    *,
    subject: str = "",
    body_text: str = "",
    attachment_path: str = "",
) -> dict[str, object]:
    if status not in SCHEDULE_STATUSES:
        raise ValueError(f"Invalid schedule status: {status}")
    return {
        "message_id": f"{assignment_id}:{student_id}:{message_type}",
        "assignment_id": assignment_id,
        "student_id": student_id,
        "recipient_email": recipient_email,
        "message_type": message_type,
        "scheduled_at": scheduled_at.isoformat(),
        "subject": subject,
        "body_text": body_text,
        "attachment_path": attachment_path,
        "status": status,
        "sent_at": "",
        "last_error": last_error,
    }


def _completed_student_ids(*, assignment_id: str) -> set[str]:
    report_path = DEFAULT_SUBMISSION_REPORTS_ROOT / f"{assignment_id}_completion.csv"
    if not report_path.exists():
        return set()
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["student_id"] for row in csv.DictReader(handle) if row.get("status") in {"submitted", "late"}}


def _completed_questions(path: Path) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    if not path.exists():
        return [], ["missing_pdf"]
    try:
        with fitz.open(path) as doc:
            text = "\n".join(page.get_text("text") for page in doc)
    except Exception as exc:  # noqa: BLE001
        return [], [f"pdf_text_extract_failed:{exc.__class__.__name__}"]
    normalized = " ".join(text.split())
    if not normalized:
        return [], ["pdf_text_not_extractable"]
    matches = sorted({match.group(1) for match in re.finditer(r"\b(?:q|question)\s*([0-9]{1,2})\b", normalized, flags=re.IGNORECASE)}, key=int)
    if not matches:
        notes.append("no_question_numbers_detected")
    return matches, notes


def _append_dispatch_audit(events: list[dict[str, object]], item: dict[str, object], *, status: str, error_code: str) -> None:
    events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "classroom_message_dispatch",
            "assignment_id": item.get("assignment_id", ""),
            "student_id": item.get("student_id", ""),
            "recipient_domain": str(item.get("recipient_email") or "").rsplit("@", 1)[-1] if "@" in str(item.get("recipient_email") or "") else "",
            "message_type": item.get("message_type", ""),
            "subject": item.get("subject", ""),
            "status": status,
            "error_code": error_code,
        }
    )


def _read_json_list(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON at {path}")
    return [item for item in payload if isinstance(item, dict)]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not normalized:
        raise ValueError("Identifier cannot be empty")
    return normalized
