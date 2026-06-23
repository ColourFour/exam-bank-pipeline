from __future__ import annotations

import email
import imaplib
import os
import smtplib
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import getaddresses, make_msgid, parsedate_to_datetime
from pathlib import Path

from .models import EmailConnectionStatus, EmailMessageSummary, EmailSendResult


OUTLOOK_CN_PROVIDER = "outlook-cn"
DEFAULT_IMAP_HOST = "partner.outlook.cn"
DEFAULT_IMAP_PORT = 993
DEFAULT_SMTP_HOST = "smtp.partner.outlook.cn"
DEFAULT_SMTP_PORT = 587


@dataclass(frozen=True)
class OutlookCnConfig:
    account_email: str | None
    password: str | None
    imap_host: str = DEFAULT_IMAP_HOST
    imap_port: int = DEFAULT_IMAP_PORT
    smtp_host: str = DEFAULT_SMTP_HOST
    smtp_port: int = DEFAULT_SMTP_PORT


class OutlookCnEmailProvider:
    provider_name = OUTLOOK_CN_PROVIDER

    def __init__(self, config: OutlookCnConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls, env: dict[str, str] | os._Environ[str] | None = None) -> "OutlookCnEmailProvider":
        source = env or os.environ
        return cls(
            OutlookCnConfig(
                account_email=_clean(source.get("EXAM_BANK_EMAIL_ADDRESS")),
                password=source.get("EXAM_BANK_EMAIL_PASSWORD"),
                imap_host=_clean(source.get("EXAM_BANK_IMAP_HOST")) or DEFAULT_IMAP_HOST,
                imap_port=_int(source.get("EXAM_BANK_IMAP_PORT"), DEFAULT_IMAP_PORT),
                smtp_host=_clean(source.get("EXAM_BANK_SMTP_HOST")) or DEFAULT_SMTP_HOST,
                smtp_port=_int(source.get("EXAM_BANK_SMTP_PORT"), DEFAULT_SMTP_PORT),
            )
        )

    def check_connection(self) -> EmailConnectionStatus:
        checked_at = datetime.now(timezone.utc)
        if not self.config.account_email or not self.config.password:
            return EmailConnectionStatus(
                provider=self.provider_name,
                connected=False,
                account_email=self.config.account_email,
                smtp_ok=False,
                imap_ok=False,
                checked_at=checked_at,
                error_code="bad_credentials_or_app_password_required",
                error_message_safe="Email address and password/app password are required in environment variables.",
            )

        smtp_ok, smtp_code = self._check_smtp()
        imap_ok, imap_code = self._check_imap()
        code = None
        if not smtp_ok and not imap_ok:
            code = "outlook_cn_auth_failed" if smtp_code in _AUTH_CODES or imap_code in _AUTH_CODES else smtp_code or imap_code
        elif not smtp_ok:
            code = smtp_code
        elif not imap_ok:
            code = imap_code
        return EmailConnectionStatus(
            provider=self.provider_name,
            connected=smtp_ok and imap_ok,
            account_email=self.config.account_email,
            smtp_ok=smtp_ok,
            imap_ok=imap_ok,
            checked_at=checked_at,
            error_code=code,
            error_message_safe=_safe_message(code),
        )

    def send_message(
        self,
        *,
        to: str,
        subject: str,
        body_text: str,
        from_address: str | None = None,
        attachments: list[Path] | None = None,
        dry_run: bool = False,
    ) -> EmailSendResult:
        sent_at = datetime.now(timezone.utc)
        if attachments:
            return self._send_error(to, subject, sent_at, dry_run, "attachments_blocked", from_address=from_address)
        if not self.config.account_email or not self.config.password:
            return self._send_error(to, subject, sent_at, dry_run, "bad_credentials_or_app_password_required", from_address=from_address)

        message_id = make_msgid(domain=(self.config.account_email.split("@", 1)[-1] or None))
        if dry_run:
            return EmailSendResult(
                provider=self.provider_name,
                sent=False,
                dry_run=True,
                to=to,
                subject=subject,
                provider_message_id=message_id,
                sent_at=sent_at,
                from_address=from_address or self.config.account_email,
            )

        message = EmailMessage()
        message["From"] = self.config.account_email
        message["To"] = to
        message["Subject"] = subject
        message["Message-ID"] = message_id
        message.set_content(body_text)
        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=30) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(self.config.account_email, self.config.password)
                smtp.send_message(message)
        except Exception as exc:  # noqa: BLE001 - map provider errors to safe public codes.
            code = _classify_smtp_error(exc)
            return self._send_error(to, subject, sent_at, dry_run, code, from_address=from_address)
        return EmailSendResult(
            provider=self.provider_name,
            sent=True,
            dry_run=False,
            to=to,
            subject=subject,
            provider_message_id=message_id,
            sent_at=sent_at,
            from_address=from_address or self.config.account_email,
        )

    def search_messages(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> list[EmailMessageSummary]:
        if not self.config.account_email or not self.config.password:
            raise OutlookCnProviderError("bad_credentials_or_app_password_required")
        try:
            with imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port, timeout=30) as imap:
                imap.login(self.config.account_email, self.config.password)
                imap.select("INBOX", readonly=True)
                status, data = imap.search(None, "ALL")
                if status != "OK":
                    raise OutlookCnProviderError("imap_disabled_or_blocked")
                ids = (data[0] or b"").split()
                matches: list[EmailMessageSummary] = []
                for message_id in reversed(ids[-100:]):
                    if len(matches) >= limit:
                        break
                    fetched_status, fetched = imap.fetch(message_id, "(RFC822)")
                    if fetched_status != "OK" or not fetched:
                        continue
                    raw = _raw_message_bytes(fetched)
                    if not raw:
                        continue
                    summary = _summarize_message(raw, provider=self.provider_name, fallback_id=message_id.decode("ascii", "ignore"))
                    haystack = f"{summary.subject}\n{summary.snippet}".lower()
                    if query.lower() in haystack:
                        matches.append(summary)
                return matches
        except OutlookCnProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - map provider errors to safe public codes.
            raise OutlookCnProviderError(_classify_imap_error(exc)) from None

    def _check_smtp(self) -> tuple[bool, str | None]:
        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=30) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(self.config.account_email, self.config.password)
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, _classify_smtp_error(exc)

    def _check_imap(self) -> tuple[bool, str | None]:
        try:
            with imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port, timeout=30) as imap:
                imap.login(self.config.account_email, self.config.password)
                imap.logout()
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, _classify_imap_error(exc)

    def _send_error(
        self,
        to: str,
        subject: str,
        sent_at: datetime,
        dry_run: bool,
        code: str,
        from_address: str | None = None,
    ) -> EmailSendResult:
        return EmailSendResult(
            provider=self.provider_name,
            sent=False,
            dry_run=dry_run,
            to=to,
            subject=subject,
            provider_message_id=None,
            sent_at=sent_at,
            error_code=code,
            error_message_safe=_safe_message(code),
            from_address=from_address or self.config.account_email,
        )


class OutlookCnProviderError(RuntimeError):
    pass


_AUTH_CODES = {"smtp_login_failed", "imap_login_failed", "bad_credentials_or_app_password_required", "tenant_requires_oauth"}


def _classify_smtp_error(exc: Exception) -> str:
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "bad_credentials_or_app_password_required"
    if isinstance(exc, smtplib.SMTPNotSupportedError):
        return "smtp_disabled_or_blocked"
    if isinstance(exc, (socket.timeout, TimeoutError, OSError)):
        return "network_blocked"
    text = str(exc).lower()
    if "auth" in text and ("disabled" in text or "enable" in text):
        return "smtp_disabled_or_blocked"
    if "oauth" in text or "modern authentication" in text:
        return "tenant_requires_oauth"
    return "smtp_login_failed"


def _classify_imap_error(exc: Exception) -> str:
    if isinstance(exc, imaplib.IMAP4.error):
        text = str(exc).lower()
        if "oauth" in text or "modern authentication" in text:
            return "tenant_requires_oauth"
        if "disabled" in text or "enable" in text:
            return "imap_disabled_or_blocked"
        return "bad_credentials_or_app_password_required"
    if isinstance(exc, (socket.timeout, TimeoutError, OSError)):
        return "network_blocked"
    return "imap_login_failed"


def _safe_message(code: str | None) -> str | None:
    if code is None:
        return None
    messages = {
        "attachments_blocked": "Attachments are blocked for email smoke tests.",
        "bad_credentials_or_app_password_required": "Authentication failed or an app password is required.",
        "imap_disabled_or_blocked": "IMAP appears disabled or blocked for this tenant/account.",
        "imap_login_failed": "IMAP login failed.",
        "network_blocked": "Network connection to the email provider failed.",
        "outlook_cn_auth_failed": "Outlook China authentication failed for SMTP and IMAP.",
        "smtp_disabled_or_blocked": "SMTP AUTH appears disabled or blocked for this tenant/account.",
        "smtp_login_failed": "SMTP login failed.",
        "tenant_requires_oauth": "Tenant appears to require OAuth/Graph or admin enablement.",
    }
    return messages.get(code, "Email provider operation failed.")


def _summarize_message(raw: bytes, *, provider: str, fallback_id: str) -> EmailMessageSummary:
    message = email.message_from_bytes(raw)
    parsed_date = _parse_date(message.get("Date"))
    to_addresses = [address for _, address in getaddresses(message.get_all("To", [])) if address]
    message_id = (message.get("Message-ID") or fallback_id).strip()
    return EmailMessageSummary(
        provider=provider,
        message_id=message_id,
        from_address=message.get("From", ""),
        to_addresses=to_addresses,
        subject=message.get("Subject", ""),
        date=parsed_date,
        snippet=_text_snippet(message),
        has_attachments=_has_attachments(message),
    )


def _text_snippet(message: email.message.Message) -> str:
    chunks: list[str] = []
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "multipart" or part.get_filename():
                continue
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                chunks.append(payload.decode(charset, errors="replace"))
                break
    else:
        payload = message.get_payload(decode=True)
        if isinstance(payload, bytes):
            chunks.append(payload.decode(message.get_content_charset() or "utf-8", errors="replace"))
        elif isinstance(message.get_payload(), str):
            chunks.append(str(message.get_payload()))
    return " ".join(" ".join(chunks).split())[:240]


def _has_attachments(message: email.message.Message) -> bool:
    return any(bool(part.get_filename()) for part in message.walk()) if message.is_multipart() else False


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _raw_message_bytes(fetched: list[bytes | tuple[bytes, bytes]]) -> bytes | None:
    for item in fetched:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], bytes):
            return item[1]
    return None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default
