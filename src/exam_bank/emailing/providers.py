from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .models import EmailConnectionStatus, EmailMessageSummary, EmailSendResult


class EmailProvider(Protocol):
    provider_name: str

    def check_connection(self) -> EmailConnectionStatus:
        ...

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
        ...

    def search_messages(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> list[EmailMessageSummary]:
        ...


class FakeEmailProvider:
    provider_name = "fake"

    def __init__(
        self,
        *,
        account_email: str = "teacher@example.invalid",
        smtp_ok: bool = True,
        imap_ok: bool = True,
        messages: list[EmailMessageSummary] | None = None,
    ) -> None:
        self.account_email = account_email
        self.smtp_ok = smtp_ok
        self.imap_ok = imap_ok
        self.sent_messages: list[EmailSendResult] = []
        self.messages = list(messages or [])

    def check_connection(self) -> EmailConnectionStatus:
        code = None if self.smtp_ok and self.imap_ok else "smtp_login_failed" if not self.smtp_ok else "imap_login_failed"
        return EmailConnectionStatus(
            provider=self.provider_name,
            connected=self.smtp_ok and self.imap_ok,
            account_email=self.account_email,
            smtp_ok=self.smtp_ok,
            imap_ok=self.imap_ok,
            checked_at=datetime.now(timezone.utc),
            error_code=code,
            error_message_safe=None if code is None else "Fake provider configured partial failure.",
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
        now = datetime.now(timezone.utc)
        if attachments:
            return EmailSendResult(
                provider=self.provider_name,
                sent=False,
                dry_run=dry_run,
                to=to,
                subject=subject,
                provider_message_id=None,
                sent_at=now,
                error_code="attachments_blocked",
                error_message_safe="Attachments are blocked for email smoke tests.",
                from_address=from_address,
            )
        if not self.smtp_ok:
            return EmailSendResult(
                provider=self.provider_name,
                sent=False,
                dry_run=dry_run,
                to=to,
                subject=subject,
                provider_message_id=None,
                sent_at=now,
                error_code="smtp_login_failed",
                error_message_safe="SMTP login failed.",
                from_address=from_address,
            )
        message_id = f"fake-message-{len(self.sent_messages) + 1}"
        result = EmailSendResult(
            provider=self.provider_name,
            sent=not dry_run,
            dry_run=dry_run,
            to=to,
            subject=subject,
            provider_message_id=message_id,
            sent_at=now,
            from_address=from_address or self.account_email,
        )
        self.sent_messages.append(result)
        if not dry_run:
            self.messages.append(
                EmailMessageSummary(
                    provider=self.provider_name,
                    message_id=message_id,
                    from_address=self.account_email,
                    to_addresses=[to],
                    subject=subject,
                    date=now,
                    snippet=body_text[:240],
                    has_attachments=False,
                )
            )
        return result

    def search_messages(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> list[EmailMessageSummary]:
        if not self.imap_ok:
            raise RuntimeError("imap_login_failed")
        normalized = query.strip().lower()
        matches = [
            message
            for message in self.messages
            if not normalized or normalized in message.subject.lower() or normalized in message.snippet.lower()
        ]
        return matches[:limit]


def build_email_provider(provider: str | None = None, *, env: dict[str, str] | None = None) -> EmailProvider:
    selected = (provider or (env or os.environ).get("EXAM_BANK_EMAIL_PROVIDER") or "").strip().lower()
    if selected == "outlook-cn":
        from .outlook_cn import OutlookCnEmailProvider

        return OutlookCnEmailProvider.from_env(env or os.environ)
    if selected == "mailapp":
        from .mailapp import MailAppEmailProvider

        return MailAppEmailProvider()
    if selected == "fake":
        return FakeEmailProvider()
    raise ValueError(f"Unsupported email provider: {selected or '<empty>'}")
