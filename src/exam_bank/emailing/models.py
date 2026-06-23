from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class EmailConnectionStatus:
    provider: str
    connected: bool
    account_email: str | None
    smtp_ok: bool
    imap_ok: bool
    checked_at: datetime
    error_code: str | None = None
    error_message_safe: str | None = None
    mailapp_available: bool | None = None
    account_verified: bool | None = None
    receive_supported: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return _dataclass_dict(self)


@dataclass(frozen=True)
class EmailSendResult:
    provider: str
    sent: bool
    dry_run: bool
    to: str
    subject: str
    provider_message_id: str | None
    sent_at: datetime
    error_code: str | None = None
    error_message_safe: str | None = None
    from_address: str | None = None

    def to_dict(self) -> dict[str, object]:
        return _dataclass_dict(self)


@dataclass(frozen=True)
class EmailMessageSummary:
    provider: str
    message_id: str
    from_address: str
    to_addresses: list[str]
    subject: str
    date: datetime | None
    snippet: str
    has_attachments: bool

    def to_dict(self) -> dict[str, object]:
        return _dataclass_dict(self)


@dataclass(frozen=True)
class EmailSmokeTestReport:
    provider: str
    account_email: str | None
    started_at: datetime
    completed_at: datetime
    connection_ok: bool
    smtp_ok: bool
    imap_ok: bool
    send_ok: bool
    receive_ok: bool
    sent_message_id: str | None
    matched_received_message_ids: list[str]
    report_path: Path
    json_report_path: Path
    smoke_token: str
    warnings: list[str] = field(default_factory=list)
    student_email_block_enforced: bool = True
    scores_sent: bool = False
    attachments_sent: bool = False
    requested_from_address: str | None = None
    mailapp_available: bool | None = None
    account_verified: bool | None = None
    receive_supported: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return _dataclass_dict(self)


def _dataclass_dict(value: object) -> dict[str, object]:
    payload = asdict(value)
    for key, item in list(payload.items()):
        if isinstance(item, datetime):
            payload[key] = item.isoformat()
        elif isinstance(item, Path):
            payload[key] = item.as_posix()
        elif isinstance(item, list):
            payload[key] = [entry.isoformat() if isinstance(entry, datetime) else entry for entry in item]
    return payload
