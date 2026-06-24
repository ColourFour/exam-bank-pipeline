from __future__ import annotations

import csv
import json
import mimetypes
import re
import shutil
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from exam_bank.classroom import (
    DEFAULT_CLASSES_ROOT,
    DEFAULT_SUBMISSION_OUTPUT_ROOT,
    DEFAULT_SUBMISSION_REPORTS_ROOT,
    ROSTER_FIELDS,
    add_assignment,
    class_paths,
    dispatch_due_messages,
    ingest_class_assignment,
    send_submission_acknowledgements,
)
from exam_bank.emailing.mailapp import MailAppEmailProvider
from exam_bank.submissions.ingest import load_assignment, parse_datetime


AUDIT_PATH = Path("reports/classroom_dashboard/classroom_dashboard_audit.jsonl")
STATIC_ROOT = Path(__file__).parent / "static"
PUBLIC_ROSTER_FIELDS = ["student_id", "class_id", "display_name", "email", "active"]
DEFAULT_TEACHER_EMAIL = "brooker@rdfzcygj.cn"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class ClassroomDashboardConfig:
    classes_root: Path = DEFAULT_CLASSES_ROOT
    output_root: Path = DEFAULT_SUBMISSION_OUTPUT_ROOT
    reports_root: Path = DEFAULT_SUBMISSION_REPORTS_ROOT
    audit_path: Path = AUDIT_PATH
    default_teacher_email: str = DEFAULT_TEACHER_EMAIL
    default_timezone: str = "Asia/Shanghai"


@dataclass(frozen=True)
class DashboardResponse:
    status: int
    body: bytes
    headers: dict[str, str]


class ClassroomDashboardApp:
    def __init__(self, config: ClassroomDashboardConfig | None = None) -> None:
        self.config = config or ClassroomDashboardConfig()

    def handle(
        self,
        method: str,
        raw_path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> DashboardResponse:
        parsed = urlparse(raw_path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/" and method == "GET":
                return self._static("index.html")
            if path.startswith("/static/") and method == "GET":
                return self._static(path.removeprefix("/static/"))
            if path == "/api/classes":
                if method == "GET":
                    return self._json(self._list_classes())
                if method == "POST":
                    return self._create_class(_json_body(body))
            if path == "/api/reports/" and method == "GET":
                return self._json({"reports": self._list_reports()})

            parts = [unquote(part) for part in path.strip("/").split("/") if part]
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "classes":
                class_id = parts[2]
                if len(parts) == 3 and method == "GET":
                    return self._json(self._class_detail(class_id))
                if len(parts) == 4 and parts[3] == "delete" and method == "POST":
                    return self._delete_class(class_id, _json_body(body))
                if len(parts) == 4 and parts[3] == "roster":
                    if method == "GET":
                        return self._json({"rows": self._read_roster(class_id)})
                    if method == "POST":
                        return self._save_roster(class_id, _json_body(body))
                if len(parts) == 5 and parts[3] == "roster" and parts[4] == "import" and method == "POST":
                    form = parse_multipart(headers or {}, body)
                    return self._import_roster(class_id, form)
                if len(parts) == 5 and parts[3] == "roster" and parts[4] == "export" and method == "GET":
                    roster = class_paths(class_id, classes_root=self.config.classes_root).roster_path
                    return self._file_response(roster, "text/csv")
                if len(parts) == 4 and parts[3] == "assignments":
                    if method == "GET":
                        return self._json({"assignments": self._list_assignments(class_id)})
                    if method == "POST":
                        form = parse_multipart(headers or {}, body)
                        return self._create_assignment(class_id, form)
                if len(parts) >= 5 and parts[3] == "assignments":
                    assignment_id = parts[4]
                    if len(parts) == 5 and method == "GET":
                        return self._json(self._assignment_detail(class_id, assignment_id))
                    if len(parts) == 6 and parts[5] == "upload-pdf" and method == "POST":
                        form = parse_multipart(headers or {}, body)
                        return self._upload_assignment_pdf(class_id, assignment_id, form)
                    if len(parts) == 6 and parts[5] == "preview-dispatch" and method == "POST":
                        return self._preview_dispatch(class_id, assignment_id)
                    if len(parts) == 6 and parts[5] == "dry-run-dispatch" and method == "POST":
                        return self._dry_run_dispatch(class_id, assignment_id, _json_body(body))
                    if len(parts) == 6 and parts[5] == "send-dispatch" and method == "POST":
                        return self._send_dispatch(class_id, assignment_id, _json_body(body))
                    if len(parts) == 6 and parts[5] == "confirm-send-later" and method == "POST":
                        return self._confirm_send_later(class_id, assignment_id, _json_body(body))
                    if len(parts) == 6 and parts[5] == "confirm-send-date" and method == "POST":
                        return self._confirm_send_later(class_id, assignment_id, _json_body(body))
                    if len(parts) == 6 and parts[5] == "send-now" and method == "POST":
                        return self._send_dispatch(class_id, assignment_id, _json_body(body), mode="send_now")
                    if len(parts) == 6 and parts[5] == "upload-submissions" and method == "POST":
                        form = parse_multipart(headers or {}, body)
                        return self._upload_submissions(class_id, assignment_id, form)
                    if len(parts) == 6 and parts[5] == "ingest-submissions" and method == "POST":
                        return self._ingest_submissions(class_id, assignment_id, _json_body(body))
                    if len(parts) == 6 and parts[5] == "preview-acknowledgements" and method == "POST":
                        return self._preview_acknowledgements(class_id, assignment_id)
                    if len(parts) == 6 and parts[5] == "confirm-acknowledgements-later" and method == "POST":
                        return self._confirm_acknowledgements_later(class_id, assignment_id, _json_body(body))
                    if len(parts) == 6 and parts[5] == "send-acknowledgements" and method == "POST":
                        return self._send_acknowledgements(class_id, assignment_id, _json_body(body))
                if len(parts) >= 5 and parts[3] == "files" and method == "GET":
                    return self._serve_class_file(class_id, parts[4:])
            if path.startswith("/reports/") and method == "GET":
                return self._file_response(Path(path.removeprefix("/")), _guess_type(path))
            return self._json({"ok": False, "error": "not_found"}, status=404)
        except DashboardError as exc:
            return self._json({"ok": False, "error": exc.code, "message": str(exc)}, status=exc.status)
        except Exception as exc:  # noqa: BLE001
            return self._json({"ok": False, "error": "server_error", "message": f"{exc.__class__.__name__}: {exc}"}, status=500)

    def _list_classes(self) -> dict[str, object]:
        self.config.classes_root.mkdir(parents=True, exist_ok=True)
        classes = []
        for class_dir in sorted(path for path in self.config.classes_root.iterdir() if path.is_dir()):
            class_id = class_dir.name
            meta = _read_json(class_dir / "class.json", default={})
            roster_rows = self._read_roster(class_id)
            assignments = self._list_assignments(class_id)
            assignment_counts = _assignment_status_counts(assignments)
            updated_at = max(
                [p.stat().st_mtime for p in [class_dir / "class.json", class_dir / "roster.csv"] if p.exists()]
                + [class_dir.stat().st_mtime]
            )
            classes.append(
                {
                    "class_id": class_id,
                    "display_name": meta.get("display_name") or class_id,
                    "course_id": meta.get("course_id") or "p3",
                    "teacher_email": meta.get("teacher_email") or self.config.default_teacher_email,
                    "timezone": meta.get("timezone") or self.config.default_timezone,
                    "student_count": sum(1 for row in roster_rows if _is_active(row)),
                    "active_assignment_count": assignment_counts["active"],
                    "draft_assignment_count": assignment_counts["draft"],
                    "archived_assignment_count": assignment_counts["archived"],
                    "last_updated": datetime.fromtimestamp(updated_at, tz=timezone.utc).isoformat(),
                }
            )
        return {"classes": classes}

    def _create_class(self, payload: dict[str, object]) -> DashboardResponse:
        class_id = _safe_segment(str(payload.get("class_id") or ""))
        paths = class_paths(class_id, classes_root=self.config.classes_root)
        paths.assignments_dir.mkdir(parents=True, exist_ok=True)
        now = _now()
        meta = {
            "class_id": class_id,
            "display_name": str(payload.get("display_name") or class_id).strip(),
            "course_id": str(payload.get("course_id") or "p3").strip(),
            "teacher_email": str(payload.get("teacher_email") or self.config.default_teacher_email).strip(),
            "timezone": str(payload.get("timezone") or self.config.default_timezone).strip(),
            "created_at": now,
            "updated_at": now,
        }
        _write_json(paths.class_dir / "class.json", meta)
        if not paths.roster_path.exists():
            _write_roster(paths.roster_path, [])
        self._audit("class_created", class_id=class_id, status="created")
        return self._json({"ok": True, "class": meta}, status=201)

    def _delete_class(self, class_id: str, payload: dict[str, object]) -> DashboardResponse:
        class_id = _safe_segment(class_id)
        confirmed = bool(payload.get("confirm")) or bool(payload.get("confirmed"))
        confirmation_text = str(payload.get("confirm_text") or payload.get("confirmation_text") or "")
        if not confirmed:
            raise DashboardError("delete_confirmation_required", "Class deletion requires explicit confirmation", status=400)
        if confirmation_text != class_id:
            raise DashboardError("delete_confirmation_text_required", "Type the class id to confirm deletion", status=400)
        paths = class_paths(class_id, classes_root=self.config.classes_root)
        if not paths.class_dir.exists():
            raise DashboardError("class_not_found", "Class not found", status=404)
        shutil.rmtree(paths.class_dir)
        self._audit("class_deleted", class_id=class_id, status="deleted")
        return self._json({"ok": True, "deleted": True, "class_id": class_id})

    def _class_detail(self, class_id: str) -> dict[str, object]:
        paths = class_paths(class_id, classes_root=self.config.classes_root)
        meta = _read_json(paths.class_dir / "class.json", default={"class_id": class_id, "display_name": class_id})
        roster = self._read_roster(class_id)
        assignments = self._list_assignments(class_id)
        return {
            "class": meta,
            "roster_summary": {
                "students": len(roster),
                "active_students": sum(1 for row in roster if _is_active(row)),
                "with_email": sum(1 for row in roster if _is_active(row) and row.get("email")),
            },
            "assignment_counts": _assignment_status_counts(assignments),
            "assignments": assignments,
        }

    def _read_roster(self, class_id: str) -> list[dict[str, str]]:
        path = class_paths(class_id, classes_root=self.config.classes_root).roster_path
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [{key: str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]

    def _save_roster(self, class_id: str, payload: dict[str, object]) -> DashboardResponse:
        rows_raw = payload.get("rows")
        if not isinstance(rows_raw, list):
            raise DashboardError("invalid_roster", "Roster payload requires rows list", status=400)
        existing = self._read_roster(class_id)
        existing_by_id = {row.get("student_id", ""): row for row in existing if row.get("student_id")}
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        warnings: list[str] = []
        for index, item in enumerate(rows_raw, start=1):
            if not isinstance(item, dict):
                continue
            row = {key: str(value or "").strip() for key, value in item.items()}
            row["class_id"] = class_id
            row["active"] = row.get("active") or "true"
            if not row.get("student_id"):
                row["student_id"] = _student_id_from_name(row.get("display_name", ""), index)
                warnings.append(f"student_id_generated:{row['student_id']}")
            if row["student_id"] in seen:
                raise DashboardError("duplicate_student_id", f"Duplicate student id: {row['student_id']}", status=400)
            seen.add(row["student_id"])
            if row.get("email") and not _valid_email(row["email"]):
                warnings.append(f"invalid_email:{row['student_id']}")
            preserved = existing_by_id.get(row["student_id"], {})
            merged = {**preserved, **row}
            rows.append(merged)
        roster_path = class_paths(class_id, classes_root=self.config.classes_root).roster_path
        backup = _backup_file(roster_path)
        _write_roster(roster_path, rows)
        self._audit("roster_saved", class_id=class_id, status="saved")
        return self._json({"ok": True, "rows": rows, "backup": str(backup) if backup else "", "warnings": warnings})

    def _import_roster(self, class_id: str, form: dict[str, Any]) -> DashboardResponse:
        uploads = form.get("files", [])
        if not uploads:
            raise DashboardError("missing_roster_csv", "No roster CSV uploaded", status=400)
        upload = uploads[0]
        text = upload["content"].decode("utf-8-sig")
        rows = list(csv.DictReader(text.splitlines()))
        return self._save_roster(class_id, {"rows": rows})

    def _list_assignments(self, class_id: str) -> list[dict[str, object]]:
        paths = class_paths(class_id, classes_root=self.config.classes_root)
        if not paths.assignments_dir.exists():
            return []
        assignments = []
        for assignment_dir in sorted(path for path in paths.assignments_dir.iterdir() if path.is_dir()):
            assignment_id = assignment_dir.name
            meta = _read_json(assignment_dir / "assignment.json", default={})
            completion = _completion_counts(assignment_id, reports_root=self.config.reports_root)
            schedule = _read_json(assignment_dir / "message_schedule.json", default=[])
            assignment_status = _assignment_status_bucket(meta, schedule)
            assignments.append(
                {
                    "assignment_id": meta.get("assignment_id") or assignment_id,
                    "title": _assignment_label(meta, assignment_id),
                    "send_at": meta.get("send_at") or _first_schedule_time(schedule, "assignment_distribution"),
                    "due_at": meta.get("due_at") or "",
                    "status": meta.get("status") or "active",
                    "assignment_status": assignment_status,
                    "email_status": meta.get("email_status") or _email_status(schedule),
                    "submitted_count": completion["submitted"],
                    "missing_count": completion["missing"],
                }
            )
        return assignments

    def _create_assignment(self, class_id: str, form: dict[str, Any]) -> DashboardResponse:
        fields = form.get("fields", {})
        files = form.get("files", [])
        pdf_upload = _first_file(files, "pdf") or (files[0] if files else None)
        if pdf_upload is None:
            raise DashboardError("missing_assignment_pdf", "Assignment PDF upload is required", status=400)
        return self._save_assignment_from_upload(class_id, fields, pdf_upload, created=True)

    def _upload_assignment_pdf(self, class_id: str, assignment_id: str, form: dict[str, Any]) -> DashboardResponse:
        files = form.get("files", [])
        pdf_upload = _first_file(files, "pdf") or (files[0] if files else None)
        if pdf_upload is None:
            raise DashboardError("missing_assignment_pdf", "Assignment PDF upload is required", status=400)
        assignment_dir = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / _safe_segment(assignment_id)
        assignment_dir.mkdir(parents=True, exist_ok=True)
        target = assignment_dir / "assignment.pdf"
        _write_pdf_upload(pdf_upload, target)
        meta_path = assignment_dir / "assignment.json"
        meta = _read_json(meta_path, default={"assignment_id": assignment_id, "class_id": class_id})
        meta["assignment_pdf"] = "assignment.pdf"
        meta["updated_at"] = _now()
        _write_json(meta_path, meta)
        self._audit("assignment_pdf_uploaded", class_id=class_id, assignment_id=assignment_id, status="uploaded")
        return self._json({"ok": True, "assignment_pdf": str(target)})

    def _save_assignment_from_upload(self, class_id: str, fields: dict[str, str], pdf_upload: dict[str, Any], *, created: bool) -> DashboardResponse:
        assignment_id = _safe_segment(str(fields.get("assignment_id") or ""))
        title = str(fields.get("title") or assignment_id).strip()
        timezone_name = str(fields.get("timezone") or self._class_timezone(class_id)).strip()
        send_at = parse_dashboard_datetime(str(fields.get("send_at") or ""), timezone_name)
        due_at = parse_dashboard_datetime(str(fields.get("due_at") or ""), timezone_name)
        assignment_dir = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / assignment_id
        assignment_dir.mkdir(parents=True, exist_ok=True)
        temp_pdf = assignment_dir / ".assignment_upload.tmp.pdf"
        _write_pdf_upload(pdf_upload, temp_pdf)
        result = add_assignment(
            class_id=class_id,
            pdf_path=temp_pdf,
            assignment_id=assignment_id,
            title=title,
            due_at=due_at,
            send_at=send_at,
            classes_root=self.config.classes_root,
            course_id=str(fields.get("course_id") or self._class_course_id(class_id) or "p3"),
            timezone_name=timezone_name,
        )
        temp_pdf.unlink(missing_ok=True)
        meta_path = Path(result["assignment_json"])
        existing = _read_json(meta_path, default={})
        now = _now()
        existing.update(
            {
                "assignment_id": assignment_id,
                "class_id": class_id,
                "course_id": str(fields.get("course_id") or self._class_course_id(class_id) or "p3"),
                "title": title,
                "assignment_pdf": "assignment.pdf",
                "send_at": send_at.isoformat(),
                "due_at": due_at.isoformat(),
                "timezone": timezone_name,
                "created_at": existing.get("created_at") or now,
                "updated_at": now,
                "email_status": existing.get("email_status") or "not_sent",
                "accepted_file_types": ["pdf"],
                "max_files_per_student": 1,
                "max_file_size_mb": 50,
                "allow_late": True,
                "source_question_ids": [],
            }
        )
        _write_json(meta_path, existing)
        self._audit("assignment_created", class_id=class_id, assignment_id=assignment_id, status="created" if created else "updated")
        return self._json({"ok": True, "assignment": self._assignment_detail(class_id, assignment_id)}, status=201 if created else 200)

    def _assignment_detail(self, class_id: str, assignment_id: str) -> dict[str, object]:
        assignment_dir = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / _safe_segment(assignment_id)
        meta = _read_json(assignment_dir / "assignment.json", default={"assignment_id": assignment_id, "class_id": class_id})
        schedule = _read_json(assignment_dir / "message_schedule.json", default=[])
        completion = _completion_counts(assignment_id, reports_root=self.config.reports_root)
        pdf = assignment_dir / "assignment.pdf"
        return {
            "assignment": meta,
            "assignment_pdf": f"/api/classes/{class_id}/files/assignments/{assignment_id}/assignment.pdf" if pdf.exists() else "",
            "roster_count": len([row for row in self._read_roster(class_id) if _is_active(row)]),
            "schedule": schedule,
            "dispatch_status": _email_status(schedule),
            "submission_status": completion,
            "reports": _assignment_report_links(assignment_id, reports_root=self.config.reports_root),
            "email_preview": self._email_preview(class_id, assignment_id),
        }

    def _preview_dispatch(self, class_id: str, assignment_id: str) -> DashboardResponse:
        preview = self._email_preview(class_id, assignment_id)
        self._audit("dispatch_previewed", class_id=class_id, assignment_id=assignment_id, status="previewed")
        return self._json({"ok": True, "preview": preview})

    def _dry_run_dispatch(self, class_id: str, assignment_id: str, payload: dict[str, object]) -> DashboardResponse:
        now = parse_datetime(str(payload["now"])) if payload.get("now") else datetime.now(timezone.utc)
        result = dispatch_due_messages(
            class_id=class_id,
            assignment_id=assignment_id,
            now=now,
            send_live=False,
            from_address=str(payload.get("from_address") or self._class_teacher_email(class_id)),
            classes_root=self.config.classes_root,
        )
        self._audit("dispatch_dry_run", class_id=class_id, assignment_id=assignment_id, status="dry_run")
        return self._json({"ok": True, "result": _jsonable(result)})

    def _confirm_send_later(self, class_id: str, assignment_id: str, payload: dict[str, object]) -> DashboardResponse:
        preview = self._email_preview(class_id, assignment_id)
        self._validate_confirmation(payload, recipient_count=int(preview["recipient_count"]), require_send_word=False)
        if preview["invalid_emails"]:
            raise DashboardError("invalid_roster_emails", f"Invalid student emails: {preview['invalid_emails']}", status=400)
        self._mark_email_status(class_id, assignment_id, "scheduled")
        self._audit("dispatch_schedule_confirmed", class_id=class_id, assignment_id=assignment_id, status="scheduled")
        return self._json(
            {
                "ok": True,
                "scheduled": True,
                "sent": 0,
                "preview": preview,
                "message": "Scheduled send confirmed. No email sent.",
            }
        )

    def _send_dispatch(self, class_id: str, assignment_id: str, payload: dict[str, object], *, mode: str = "due") -> DashboardResponse:
        effective_payload = dict(payload)
        resend_sent = mode == "send_now"
        if mode == "confirmed_send_date":
            effective_payload["now"] = self._assignment_send_at(class_id, assignment_id).isoformat()
        elif mode == "send_now" and not effective_payload.get("now"):
            effective_payload["now"] = self._assignment_send_at(class_id, assignment_id).isoformat()
        due_messages = self._due_messages(class_id, assignment_id, effective_payload, include_sent=resend_sent)
        provider = MailAppEmailProvider(requested_from_address=str(payload.get("from_address") or self._class_teacher_email(class_id)))
        self._validate_live_send(class_id, assignment_id, payload, due_messages, provider=provider, require_send_word=len(due_messages) > 1)
        self._audit("dispatch_live_send_attempted", class_id=class_id, assignment_id=assignment_id, status="attempted")
        result = dispatch_due_messages(
            class_id=class_id,
            assignment_id=assignment_id,
            now=parse_datetime(str(effective_payload["now"])) if effective_payload.get("now") else datetime.now(timezone.utc),
            send_live=True,
            from_address=str(payload.get("from_address") or self._class_teacher_email(class_id)),
            provider=provider,
            classes_root=self.config.classes_root,
            resend_sent=resend_sent,
        )
        self._mark_email_status(class_id, assignment_id, "sent" if int(result.get("sent", 0)) else "not_sent")
        self._audit("dispatch_live_send_completed", class_id=class_id, assignment_id=assignment_id, status="completed")
        return self._json({"ok": True, "result": _jsonable(result)})

    def _upload_submissions(self, class_id: str, assignment_id: str, form: dict[str, Any]) -> DashboardResponse:
        files = form.get("files", [])
        saved: list[str] = []
        inbox = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / _safe_segment(assignment_id) / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        for upload in files:
            filename = _safe_filename(upload.get("filename") or "submission.pdf")
            if not filename.lower().endswith(".pdf") or not upload.get("content", b"").startswith(b"%PDF"):
                raise DashboardError("invalid_submission_pdf", "Only PDF submissions are accepted", status=400)
            target = inbox / filename
            target.write_bytes(upload["content"])
            saved.append(str(target))
        self._audit("submissions_uploaded", class_id=class_id, assignment_id=assignment_id, status="uploaded", count=len(saved))
        return self._json({"ok": True, "saved": saved})

    def _ingest_submissions(self, class_id: str, assignment_id: str, payload: dict[str, object]) -> DashboardResponse:
        result = ingest_class_assignment(
            class_id=class_id,
            assignment_id=assignment_id,
            classes_root=self.config.classes_root,
            output_root=self.config.output_root,
            reports_root=self.config.reports_root,
            import_from_mailapp=bool(payload.get("from_mailapp")),
            mail_query=str(payload.get("mail_query") or assignment_id),
            from_address=str(payload.get("from_address") or self._class_teacher_email(class_id)),
        )
        self._audit("submissions_ingested", class_id=class_id, assignment_id=assignment_id, status="ingested")
        return self._json({"ok": True, "result": _jsonable(result), "assignment": self._assignment_detail(class_id, assignment_id)})

    def _preview_acknowledgements(self, class_id: str, assignment_id: str) -> DashboardResponse:
        preview = self._acknowledgement_preview(class_id, assignment_id)
        self._audit("acknowledgements_previewed", class_id=class_id, assignment_id=assignment_id, status="previewed")
        return self._json({"ok": True, **preview})

    def _confirm_acknowledgements_later(self, class_id: str, assignment_id: str, payload: dict[str, object]) -> DashboardResponse:
        preview = self._acknowledgement_preview(class_id, assignment_id)
        self._validate_confirmation(payload, recipient_count=int(preview["count"]), require_send_word=False)
        self._audit("acknowledgements_schedule_confirmed", class_id=class_id, assignment_id=assignment_id, status="scheduled")
        return self._json(
            {
                "ok": True,
                "scheduled": True,
                "sent": 0,
                "preview": preview,
                "message": "Receipt send confirmed for later. No email sent.",
            }
        )

    def _send_acknowledgements(self, class_id: str, assignment_id: str, payload: dict[str, object]) -> DashboardResponse:
        rows = _completion_rows(assignment_id, reports_root=self.config.reports_root)
        recipients = [row for row in rows if row.get("status") in {"submitted", "late"} and row.get("email")]
        self._validate_confirmation(payload, recipient_count=len(recipients), require_send_word=len(recipients) > 1)
        provider = MailAppEmailProvider(requested_from_address=str(payload.get("from_address") or self._class_teacher_email(class_id)))
        status = provider.check_connection()
        if not status.connected:
            raise DashboardError("email_provider_not_configured", "Email provider not configured. Run email smoke test first.", status=400)
        result = send_submission_acknowledgements(
            class_id=class_id,
            assignment_id=assignment_id,
            completion_report=self.config.reports_root / f"{assignment_id}_completion.csv",
            from_address=str(payload.get("from_address") or self._class_teacher_email(class_id)),
            provider=provider,
        )
        self._audit("acknowledgements_sent", class_id=class_id, assignment_id=assignment_id, status="sent")
        return self._json({"ok": True, "result": result})

    def _acknowledgement_preview(self, class_id: str, assignment_id: str) -> dict[str, object]:
        rows = _completion_rows(assignment_id, reports_root=self.config.reports_root)
        recipients = [
            {"student_id": row["student_id"], "email": row["email"], "status": row["status"], "subject": f"Received: {assignment_id}"}
            for row in rows
            if row.get("status") in {"submitted", "late"} and row.get("email")
        ]
        submitted_without_email = [
            row.get("student_id", "")
            for row in rows
            if row.get("status") in {"submitted", "late"} and not row.get("email")
        ]
        return {
            "recipients": recipients,
            "sample_recipients": recipients[:5],
            "count": len(recipients),
            "submitted_without_email": submitted_without_email,
            "subject": f"Received: {assignment_id}",
            "body": "We received your submitted PDF. Scores, feedback, and attachments are not included.",
            "scores_included": False,
            "feedback_included": False,
            "attachments": [],
            "attachments_included": False,
            "missing_students_receive_email": False,
        }

    def _email_preview(self, class_id: str, assignment_id: str) -> dict[str, object]:
        assignment_dir = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / _safe_segment(assignment_id)
        meta = _read_json(assignment_dir / "assignment.json", default={})
        roster = [row for row in self._read_roster(class_id) if _is_active(row)]
        recipients = [row for row in roster if row.get("email")]
        pdf = assignment_dir / "assignment.pdf"
        schedule = _read_json(assignment_dir / "message_schedule.json", default=[])
        sent_recipients = _sent_email_rows(schedule)
        subject = _assignment_label(meta, assignment_id)
        body = (
            f"Please complete the assignment: {subject}.\n"
            f"Due: {meta.get('due_at') or ''}.\n\n"
            "Reply to this email with your completed PDF attached."
        )
        return {
            "recipient_count": len(recipients),
            "roster_total": len(roster),
            "already_sent": bool(sent_recipients),
            "sent_recipient_count": len(sent_recipients),
            "sent_recipients": sent_recipients,
            "missing_email_count": sum(1 for row in roster if not row.get("email")),
            "invalid_email_count": len(_invalid_roster_emails(roster)),
            "sample_recipients": [{"student_id": row.get("student_id", ""), "email": row.get("email", "")} for row in recipients[:5]],
            "subject": subject,
            "body": body,
            "attachments": [str(pdf)] if pdf.exists() else [],
            "pdf_attached": pdf.exists(),
            "invalid_emails": _invalid_roster_emails(roster),
            "send_at": meta.get("send_at") or "",
            "due_at": meta.get("due_at") or "",
            "provider": "Mail.app",
            "scores_included": False,
        }

    def _due_messages(self, class_id: str, assignment_id: str, payload: dict[str, object], *, include_sent: bool = False) -> list[dict[str, object]]:
        assignment_dir = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / _safe_segment(assignment_id)
        schedule = _read_json(assignment_dir / "message_schedule.json", default=[])
        now = parse_datetime(str(payload["now"])) if payload.get("now") else datetime.now(timezone.utc)
        eligible_statuses = {"scheduled", "failed"}
        if include_sent:
            eligible_statuses.add("sent")
        due = []
        for item in schedule:
            if str(item.get("status")) not in eligible_statuses:
                continue
            if parse_datetime(str(item.get("scheduled_at"))) <= now and item.get("recipient_email"):
                due.append(item)
        return due

    def _validate_live_send(
        self,
        class_id: str,
        assignment_id: str,
        payload: dict[str, object],
        due_messages: list[dict[str, object]],
        *,
        provider: MailAppEmailProvider,
        require_send_word: bool,
    ) -> None:
        self._validate_confirmation(payload, recipient_count=len(due_messages), require_send_word=require_send_word)
        if not due_messages:
            raise DashboardError("no_due_messages", "No due messages are ready to send", status=400)
        roster = [row for row in self._read_roster(class_id) if _is_active(row)]
        if not any(row.get("email") for row in roster):
            raise DashboardError("roster_has_no_emails", "Roster has no student emails", status=400)
        invalid = _invalid_roster_emails(roster)
        if invalid:
            raise DashboardError("invalid_roster_emails", f"Invalid student emails: {invalid}", status=400)
        assignment_pdf = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / _safe_segment(assignment_id) / "assignment.pdf"
        if not assignment_pdf.exists():
            raise DashboardError("missing_assignment_pdf", "Assignment is missing PDF", status=400)
        status = provider.check_connection()
        if not status.connected:
            raise DashboardError("email_provider_not_configured", "Email provider not configured. Run email smoke test first.", status=400)

    def _assignment_send_at(self, class_id: str, assignment_id: str) -> datetime:
        assignment_dir = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / _safe_segment(assignment_id)
        meta = _read_json(assignment_dir / "assignment.json", default={})
        if meta.get("send_at"):
            return parse_datetime(str(meta["send_at"]))
        schedule = _read_json(assignment_dir / "message_schedule.json", default=[])
        first = _first_schedule_time(schedule, "assignment_distribution")
        return parse_datetime(first) if first else datetime.now(timezone.utc)

    def _validate_confirmation(self, payload: dict[str, object], *, recipient_count: int, require_send_word: bool) -> None:
        confirmed = bool(payload.get("confirm")) or bool(payload.get("confirmed"))
        confirmation_text = str(payload.get("confirm_text") or payload.get("confirmation_text") or "")
        if not confirmed:
            raise DashboardError("send_confirmation_required", "Live send requires explicit confirmation", status=400)
        if require_send_word and confirmation_text != "SEND":
            raise DashboardError("send_confirmation_text_required", "Type SEND to confirm multi-recipient live send", status=400)
        if recipient_count <= 0:
            raise DashboardError("no_recipients", "No recipients are eligible", status=400)

    def _serve_class_file(self, class_id: str, rel_parts: list[str]) -> DashboardResponse:
        class_dir = class_paths(class_id, classes_root=self.config.classes_root).class_dir.resolve()
        target = (class_dir / Path(*rel_parts)).resolve()
        if class_dir not in target.parents and target != class_dir:
            raise DashboardError("invalid_file_path", "Invalid file path", status=400)
        return self._file_response(target, _guess_type(target.name))

    def _list_reports(self) -> list[dict[str, str]]:
        reports = []
        for root in [self.config.reports_root, self.config.audit_path.parent]:
            if not root.exists():
                continue
            for path in sorted(item for item in root.glob("*") if item.is_file()):
                reports.append({"name": path.name, "path": path.as_posix(), "url": f"/reports/{path.relative_to(Path('.')).as_posix()}" if not path.is_absolute() else ""})
        return reports

    def _class_teacher_email(self, class_id: str) -> str:
        meta = _read_json(class_paths(class_id, classes_root=self.config.classes_root).class_dir / "class.json", default={})
        return str(meta.get("teacher_email") or self.config.default_teacher_email)

    def _class_timezone(self, class_id: str) -> str:
        meta = _read_json(class_paths(class_id, classes_root=self.config.classes_root).class_dir / "class.json", default={})
        return str(meta.get("timezone") or self.config.default_timezone)

    def _class_course_id(self, class_id: str) -> str:
        meta = _read_json(class_paths(class_id, classes_root=self.config.classes_root).class_dir / "class.json", default={})
        return str(meta.get("course_id") or "p3")

    def _mark_email_status(self, class_id: str, assignment_id: str, status: str) -> None:
        path = class_paths(class_id, classes_root=self.config.classes_root).assignments_dir / _safe_segment(assignment_id) / "assignment.json"
        meta = _read_json(path, default={})
        meta["email_status"] = status
        meta["updated_at"] = _now()
        _write_json(path, meta)

    def _audit(self, event_type: str, *, class_id: str = "", assignment_id: str = "", status: str = "", count: int | None = None) -> None:
        event: dict[str, object] = {
            "timestamp": _now(),
            "event_type": event_type,
            "class_id": class_id,
            "assignment_id": assignment_id,
            "status": status,
        }
        if count is not None:
            event["count"] = count
        self.config.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _static(self, filename: str) -> DashboardResponse:
        target = (STATIC_ROOT / filename).resolve()
        static_root = STATIC_ROOT.resolve()
        if static_root not in target.parents and target != static_root:
            return self._json({"ok": False, "error": "not_found"}, status=404)
        if not target.exists() or not target.is_file():
            return self._json({"ok": False, "error": "not_found"}, status=404)
        return self._file_response(target, _guess_type(target.name))

    def _file_response(self, path: Path, content_type: str) -> DashboardResponse:
        if not path.exists() or not path.is_file():
            return self._json({"ok": False, "error": "not_found"}, status=404)
        return DashboardResponse(200, path.read_bytes(), {"Content-Type": content_type})

    def _json(self, payload: object, *, status: int = 200) -> DashboardResponse:
        return DashboardResponse(
            status,
            json.dumps(_jsonable(payload), indent=2, sort_keys=True).encode("utf-8"),
            {"Content-Type": "application/json; charset=utf-8"},
        )


class DashboardError(ValueError):
    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def serve_dashboard(*, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False, config: ClassroomDashboardConfig | None = None) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        print(f"Warning: classroom dashboard should bind only to localhost. Requested host: {host}")
    app = ClassroomDashboardApp(config)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._dispatch()

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch()

        def do_HEAD(self) -> None:  # noqa: N802
            self._dispatch(head_only=True)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _dispatch(self, *, head_only: bool = False) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            method = "GET" if self.command == "HEAD" else self.command
            response = app.handle(method, self.path, headers={key: value for key, value in self.headers.items()}, body=body)
            self.send_response(response.status)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(response.body)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print("Classroom dashboard running at:")
    print(url)
    if open_browser:
        webbrowser.open(url)
    server.serve_forever()


def parse_multipart(headers: dict[str, str], body: bytes) -> dict[str, Any]:
    content_type = headers.get("Content-Type") or headers.get("content-type") or ""
    if not content_type.startswith("multipart/form-data"):
        return {"fields": _json_body_bytes(body), "files": []}
    parser = BytesParser(policy=default)
    message = parser.parsebytes(f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body)
    fields: dict[str, str] = {}
    files: list[dict[str, Any]] = []
    for part in message.iter_parts():
        disposition = part.get_content_disposition()
        if disposition != "form-data":
            continue
        name = part.get_param("name", header="content-disposition") or ""
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files.append({"field": name, "filename": filename, "content": payload, "content_type": part.get_content_type()})
        else:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8")
    return {"fields": fields, "files": files}


def parse_dashboard_datetime(value: str, timezone_name: str) -> datetime:
    if not value:
        raise DashboardError("datetime_required", "send_at and due_at are required", status=400)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DashboardError("invalid_datetime", f"Invalid datetime: {value}", status=400) from exc
    if parsed.tzinfo is not None:
        return parsed
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    return parsed.replace(tzinfo=tz)


def _json_body(body: bytes) -> dict[str, object]:
    return _json_body_bytes(body)


def _json_body_bytes(body: bytes) -> dict[str, object]:
    if not body:
        return {}
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise DashboardError("invalid_json", "Expected JSON object", status=400)
    return payload


def _write_pdf_upload(upload: dict[str, Any], target: Path) -> None:
    filename = str(upload.get("filename") or "")
    content = upload.get("content", b"")
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF"):
        raise DashboardError("invalid_assignment_pdf", "Assignment upload must be a PDF", status=400)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _write_roster(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = _roster_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _roster_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    fields = list(dict.fromkeys([*PUBLIC_ROSTER_FIELDS, *ROSTER_FIELDS]))
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.backup.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{path.suffix}")
    backup.write_bytes(path.read_bytes())
    return backup


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _completion_counts(assignment_id: str, *, reports_root: Path) -> dict[str, int]:
    rows = _completion_rows(assignment_id, reports_root=reports_root)
    return {
        "submitted": sum(1 for row in rows if row.get("status") in {"submitted", "late"}),
        "missing": sum(1 for row in rows if row.get("status") == "missing"),
        "rejected": sum(1 for row in rows if row.get("status") == "rejected"),
    }


def _completion_rows(assignment_id: str, *, reports_root: Path) -> list[dict[str, str]]:
    path = reports_root / f"{assignment_id}_completion.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def _assignment_report_links(assignment_id: str, *, reports_root: Path) -> list[dict[str, str]]:
    links = []
    for path in sorted(reports_root.glob(f"{assignment_id}*")):
        if path.is_file():
            links.append({"name": path.name, "path": path.as_posix(), "url": f"/reports/{path.as_posix()}"})
    return links


def _first_schedule_time(schedule: object, message_type: str) -> str:
    if not isinstance(schedule, list):
        return ""
    for item in schedule:
        if isinstance(item, dict) and item.get("message_type") == message_type:
            return str(item.get("scheduled_at") or "")
    return ""


def _email_status(schedule: object) -> str:
    if not isinstance(schedule, list) or not schedule:
        return "not_scheduled"
    statuses = {str(item.get("status") or "") for item in schedule if isinstance(item, dict)}
    if "sent" in statuses:
        return "sent"
    if "failed" in statuses:
        return "failed"
    return "not_sent"


def _sent_email_rows(schedule: object) -> list[dict[str, str]]:
    if not isinstance(schedule, list):
        return []
    rows = []
    for item in schedule:
        if not isinstance(item, dict):
            continue
        if item.get("message_type") != "assignment_distribution" or item.get("status") != "sent":
            continue
        rows.append(
            {
                "student_id": str(item.get("student_id") or ""),
                "email": str(item.get("recipient_email") or ""),
                "sent_at": str(item.get("sent_at") or ""),
            }
        )
    return rows


def _assignment_status_counts(assignments: list[dict[str, object]]) -> dict[str, int]:
    counts = {"active": 0, "draft": 0, "archived": 0}
    for assignment in assignments:
        status = str(assignment.get("assignment_status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _assignment_status_bucket(meta: dict[str, object], schedule: object, *, now: datetime | None = None) -> str:
    status = str(meta.get("status") or meta.get("assignment_status") or "").strip().lower()
    if bool(meta.get("archived")) or status in {"archived", "archive", "closed"}:
        return "archived"
    due_at = _optional_datetime(str(meta.get("due_at") or ""))
    if due_at is not None and due_at <= (now or datetime.now(timezone.utc)):
        return "archived"

    explicit_email_status = str(meta.get("email_status") or "").strip().lower()
    scheduled_email_status = _email_status(schedule).strip().lower() if isinstance(schedule, list) and schedule else ""
    email_status = explicit_email_status or scheduled_email_status
    if email_status == "sent":
        return "active"
    if email_status in {"not_sent", "not_scheduled", "scheduled", "draft"}:
        return "draft"
    return ""


def _optional_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parse_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _assignment_label(meta: dict[str, object], assignment_id: str) -> str:
    title = str(meta.get("title") or assignment_id).strip()
    if title.lower().startswith("assignment:"):
        return title
    return f"assignment: {title}"


def _first_file(files: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    for item in files:
        if item.get("field") == field:
            return item
    return None


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not normalized:
        raise DashboardError("invalid_identifier", "Identifier cannot be empty", status=400)
    return normalized


def _safe_filename(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def _student_id_from_name(name: str, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").upper()
    return base[:20] if base else f"S{index:04d}"


def _is_active(row: dict[str, str]) -> bool:
    return str(row.get("active", "true")).strip().lower() in {"1", "true", "yes", "y", "active"}


def _valid_email(value: str) -> bool:
    return bool(EMAIL_RE.match(value.strip()))


def _invalid_roster_emails(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"student_id": row.get("student_id", ""), "email": row.get("email", "")}
        for row in rows
        if row.get("email") and not _valid_email(row["email"])
    ]


def _guess_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
