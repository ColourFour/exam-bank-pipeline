from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from exam_bank.submissions.email_fixtures import LoadedEmailFixture, load_email_fixtures
from exam_bank.submissions.email_models import InboundEmailAttachment, InboundEmailMessage


@dataclass(frozen=True)
class EmailMessageRef:
    message_id: str
    thread_id: str
    received_at: datetime | None


class EmailConnector(Protocol):
    def list_messages(
        self,
        *,
        mailbox_scope: str,
        search_query: str,
        since: datetime | None,
        limit: int,
    ) -> list[EmailMessageRef]:
        ...

    def fetch_message(self, message_id: str) -> InboundEmailMessage:
        ...

    def fetch_attachments(self, message_id: str) -> dict[str, Path]:
        ...


class FakeEmailConnector:
    def __init__(self, fixtures: list[LoadedEmailFixture]) -> None:
        self._fixtures = {f"{index}:{item.message.message_id}": item for index, item in enumerate(fixtures)}

    def list_messages(
        self,
        *,
        mailbox_scope: str,
        search_query: str,
        since: datetime | None,
        limit: int,
    ) -> list[EmailMessageRef]:
        refs = [
            EmailMessageRef(
                message_id=key,
                thread_id=item.message.thread_id,
                received_at=item.message.received_at,
            )
            for key, item in self._fixtures.items()
            if _matches_since(item.message, since)
        ]
        return sorted(refs, key=lambda ref: (ref.received_at or _MIN_TIME, ref.message_id))[:limit]

    def fetch_message(self, message_id: str) -> InboundEmailMessage:
        return _with_source(self._fixtures[message_id].message, "fake_connector")

    def fetch_attachments(self, message_id: str) -> dict[str, Path]:
        return dict(self._fixtures[message_id].attachment_paths)


class LocalExportEmailConnector:
    def __init__(self, *, export_dir: Path, assignment_id: str) -> None:
        self.export_dir = export_dir
        self.assignment_id = assignment_id
        loaded = load_email_fixtures(export_dir, assignment_id)
        self._fixtures = {f"{index}:{item.message.message_id}": item for index, item in enumerate(loaded)}

    def list_messages(
        self,
        *,
        mailbox_scope: str,
        search_query: str,
        since: datetime | None,
        limit: int,
    ) -> list[EmailMessageRef]:
        refs = [
            EmailMessageRef(
                message_id=key,
                thread_id=item.message.thread_id,
                received_at=item.message.received_at,
            )
            for key, item in self._fixtures.items()
            if _matches_since(item.message, since) and _matches_query(item.message, search_query)
        ]
        return sorted(refs, key=lambda ref: (ref.received_at or _MIN_TIME, ref.message_id))[:limit]

    def fetch_message(self, message_id: str) -> InboundEmailMessage:
        return _with_source(self._fixtures[message_id].message, "local_export_connector")

    def fetch_attachments(self, message_id: str) -> dict[str, Path]:
        return dict(self._fixtures[message_id].attachment_paths)


def build_configured_connector(config: dict[str, object], *, assignment_id: str) -> EmailConnector:
    provider = str(config.get("provider", "")).strip()
    if provider == "local_export":
        export_dir = Path(str(config.get("export_dir", "")).strip())
        if not export_dir:
            raise ValueError("local_export connector requires export_dir")
        return LocalExportEmailConnector(export_dir=export_dir, assignment_id=assignment_id)
    if provider == "fake":
        return FakeEmailConnector([_fake_fixture(assignment_id)])
    raise ValueError(f"Unsupported email connector provider: {provider}")


def _matches_since(message: InboundEmailMessage, since: datetime | None) -> bool:
    if since is None or message.received_at is None:
        return True
    return message.received_at >= since


def _matches_query(message: InboundEmailMessage, search_query: str) -> bool:
    query = search_query.strip()
    if not query:
        return True
    return query in message.subject or query in message.body_preview


def _with_source(message: InboundEmailMessage, source: str) -> InboundEmailMessage:
    attachments: list[InboundEmailAttachment] = [replace(attachment, stored_path=attachment.stored_path) for attachment in message.attachments]
    return replace(message, source=source, attachments=attachments, attachment_count=len(attachments))


def _fake_fixture(assignment_id: str) -> LoadedEmailFixture:
    created_at = datetime.now(timezone.utc)
    message = InboundEmailMessage(
        message_id="fake-live-message-1",
        thread_id="fake-live-thread-1",
        assignment_id=assignment_id,
        received_at=created_at,
        from_email="student.one@example.invalid",
        from_name="Student One",
        subject=f"{assignment_id} S0001",
        body_preview="Synthetic connector dry-run message.",
        attachment_count=0,
        attachments=[],
        source="fake_connector",
        status="accepted",
        reasons=[],
        created_at=created_at,
    )
    return LoadedEmailFixture(message=message, fixture_path=Path("fake_connector"), attachment_paths={})


_MIN_TIME = datetime.min.replace(tzinfo=timezone.utc)
