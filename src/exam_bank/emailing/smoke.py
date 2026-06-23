from __future__ import annotations

import html
import json
import secrets
import webbrowser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import EmailMessageSummary, EmailSmokeTestReport
from .providers import EmailProvider

DEFAULT_REPORTS_ROOT = Path("reports/email")
AUDIT_LOG_NAME = "email_audit.jsonl"
SMOKE_SUBJECT_PHRASE = "Smoke Test"


class EmailSmokeSafetyError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_smoke_send_request(
    *,
    to: str,
    subject: str,
    attachments: list[Path] | None = None,
    allow_non_smoke_subject: bool = False,
    assignment_id: str | None = None,
    roster_file: Path | None = None,
    score_file: Path | None = None,
    cc: str | None = None,
    bcc: str | None = None,
) -> None:
    recipient = to.strip()
    if not recipient or "," in recipient or ";" in recipient or len(recipient.split()) != 1:
        raise EmailSmokeSafetyError("multiple_recipients_blocked", "Email smoke tests require exactly one recipient.")
    if "@" not in recipient:
        raise EmailSmokeSafetyError("invalid_recipient", "Email smoke tests require a single email address recipient.")
    if cc or bcc:
        raise EmailSmokeSafetyError("cc_bcc_blocked", "CC and BCC are blocked for email smoke tests.")
    if attachments:
        raise EmailSmokeSafetyError("attachments_blocked", "Attachments are blocked for email smoke tests.")
    if assignment_id or roster_file or score_file:
        raise EmailSmokeSafetyError("student_bulk_options_blocked", "Assignment, roster, and score-file options are blocked for email smoke tests.")
    if not allow_non_smoke_subject and SMOKE_SUBJECT_PHRASE not in subject:
        raise EmailSmokeSafetyError("non_smoke_subject_blocked", 'Smoke-test sends require a subject containing "Smoke Test".')


def make_smoke_token(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d")
    return f"{stamp}-{secrets.token_hex(3)}"


def write_audit_event(
    *,
    event_type: str,
    provider: str,
    subject: str = "",
    recipient: str = "",
    dry_run: bool = False,
    status: str,
    error_code: str | None = None,
    provider_message_id: str | None = None,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
) -> Path:
    reports_root.mkdir(parents=True, exist_ok=True)
    path = reports_root / AUDIT_LOG_NAME
    event = {
        "event_type": event_type,
        "provider": provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "recipient_domain": _recipient_domain(recipient),
        "subject": subject,
        "status": status,
        "error_code": error_code,
    }
    if provider_message_id:
        event["provider_message_id"] = provider_message_id
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return path


def run_email_smoke_test(
    *,
    provider: EmailProvider,
    to: str,
    from_address: str | None = None,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    open_report: bool = False,
    body_text: str = "This is a controlled test email from the exam-bank system.",
) -> EmailSmokeTestReport:
    started_at = datetime.now(timezone.utc)
    token = make_smoke_token(started_at)
    subject = f"Exam Bank Email Smoke Test [{token}]"
    body = f"{body_text}\n\nSmoke token: {token}"
    validate_smoke_send_request(to=to, subject=subject)

    connection = provider.check_connection()
    warnings: list[str] = []
    if not connection.smtp_ok:
        warnings.append(connection.error_code or "smtp_not_connected")
    if not connection.imap_ok and (connection.error_code or "imap_not_connected") not in warnings:
        warnings.append(connection.error_code or "imap_not_connected")

    send_result = None
    if connection.smtp_ok:
        send_result = provider.send_message(to=to, subject=subject, body_text=body, from_address=from_address, attachments=None, dry_run=False)
        write_audit_event(
            event_type="email_smoke_send",
            provider=provider.provider_name,
            subject=subject,
            recipient=to,
            dry_run=False,
            status="sent" if send_result.sent else "failed",
            error_code=send_result.error_code,
            provider_message_id=send_result.provider_message_id,
            reports_root=reports_root,
        )
        if send_result.error_code:
            warnings.append(send_result.error_code)
    else:
        write_audit_event(
            event_type="email_smoke_send",
            provider=provider.provider_name,
            subject=subject,
            recipient=to,
            dry_run=False,
            status="skipped",
            error_code=connection.error_code,
            reports_root=reports_root,
        )

    matched: list[EmailMessageSummary] = []
    receive_error: str | None = None
    receive_supported = connection.receive_supported if connection.receive_supported is not None else True
    if connection.imap_ok:
        try:
            matched = provider.search_messages(query=token, limit=5)
        except RuntimeError as exc:
            receive_error = str(exc) or "imap_login_failed"
            if receive_error == "mailapp_search_unsupported":
                receive_supported = False
        write_audit_event(
            event_type="email_smoke_receive_search",
            provider=provider.provider_name,
            subject=subject,
            recipient=to,
            dry_run=False,
            status="found" if matched else "not_found" if receive_error is None else "failed",
            error_code=receive_error,
            provider_message_id=send_result.provider_message_id if send_result else None,
            reports_root=reports_root,
        )
    else:
        write_audit_event(
            event_type="email_smoke_receive_search",
            provider=provider.provider_name,
            subject=subject,
            recipient=to,
            dry_run=False,
            status="skipped",
            error_code=connection.error_code,
            reports_root=reports_root,
        )
    if receive_error and receive_error not in warnings:
        warnings.append(receive_error)

    completed_at = datetime.now(timezone.utc)
    reports_root.mkdir(parents=True, exist_ok=True)
    report_stem = f"email_smoke_test_{completed_at.strftime('%Y%m%dT%H%M%SZ')}"
    html_path = reports_root / f"{report_stem}.html"
    json_path = reports_root / f"{report_stem}.json"
    report = EmailSmokeTestReport(
        provider=provider.provider_name,
        account_email=connection.account_email,
        started_at=started_at,
        completed_at=completed_at,
        connection_ok=connection.connected,
        smtp_ok=connection.smtp_ok,
        imap_ok=connection.imap_ok,
        send_ok=bool(send_result and send_result.sent),
        receive_ok=bool(matched),
        sent_message_id=send_result.provider_message_id if send_result else None,
        matched_received_message_ids=[message.message_id for message in matched],
        report_path=html_path,
        json_report_path=json_path,
        smoke_token=token,
        warnings=warnings,
        requested_from_address=from_address,
        mailapp_available=connection.mailapp_available,
        account_verified=connection.account_verified,
        receive_supported=receive_supported,
    )
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(render_smoke_report_html(report), encoding="utf-8")
    if open_report:
        webbrowser.open(html_path.resolve().as_uri())
    return report


def render_smoke_report_html(report: EmailSmokeTestReport) -> str:
    rows = [
        ("provider", report.provider),
        ("account", report.account_email or ""),
        ("requested from account", report.requested_from_address or ""),
        ("Mail.app available", "" if report.mailapp_available is None else str(report.mailapp_available).lower()),
        ("account verified", "" if report.account_verified is None else str(report.account_verified).lower()),
        ("SMTP status", "ok" if report.smtp_ok else "failed"),
        ("IMAP status", "ok" if report.imap_ok else "failed"),
        ("send status", "sent" if report.send_ok else "not sent"),
        ("receive/search status", "unsupported" if report.receive_supported is False else "found" if report.receive_ok else "not found"),
        ("sent message id", report.sent_message_id or ""),
        ("matched received message ids", ", ".join(report.matched_received_message_ids)),
        ("smoke token", report.smoke_token),
        ("warnings", ", ".join(report.warnings)),
        ("credentials", "not included"),
        ("student data", "not included"),
        ("scores sent", str(report.scores_sent).lower()),
        ("attachments sent", str(report.attachments_sent).lower()),
    ]
    body = "\n".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>" for label, value in rows)
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\"><title>Exam Bank Email Smoke Test</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.4}"
        "table{border-collapse:collapse;min-width:36rem}th,td{border:1px solid #ccc;padding:.45rem .6rem;text-align:left}"
        "th{background:#f5f5f5;width:16rem}</style></head>"
        f"<body><h1>Exam Bank Email Smoke Test</h1><table>{body}</table></body></html>\n"
    )


def safe_summary_dict(message: EmailMessageSummary) -> dict[str, object]:
    payload = asdict(message)
    if message.date is not None:
        payload["date"] = message.date.isoformat()
    return payload


def _recipient_domain(recipient: str) -> str:
    if "@" not in recipient:
        return ""
    return recipient.rsplit("@", 1)[-1].lower()
