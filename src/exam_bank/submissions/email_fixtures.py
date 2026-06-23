from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import exam_bank.submissions.email_reasons as reasons
from exam_bank.submissions.email_models import InboundEmailAttachment, InboundEmailMessage
from exam_bank.submissions.ingest import parse_datetime
from exam_bank.submissions.validation import sha256_file


@dataclass(frozen=True)
class LoadedEmailFixture:
    message: InboundEmailMessage
    fixture_path: Path
    attachment_paths: dict[str, Path]


def load_email_fixtures(fixtures_dir: Path, assignment_id: str) -> list[LoadedEmailFixture]:
    fixture_root = fixtures_dir.resolve()
    if not fixture_root.is_dir():
        raise ValueError(f"Email fixtures directory does not exist: {fixtures_dir}")

    loaded: list[LoadedEmailFixture] = []
    for path in sorted(fixture_root.glob("*.json")):
        loaded.append(load_email_fixture(path, fixture_root, assignment_id))
    return loaded


def load_email_fixture(path: Path, fixtures_dir: Path, assignment_id: str) -> LoadedEmailFixture:
    fixture_root = fixtures_dir.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    message_id = str(data.get("message_id", "")).strip()
    received_at = _parse_optional_datetime(data.get("received_at"))
    created_at = datetime.now(timezone.utc)

    attachments: list[InboundEmailAttachment] = []
    attachment_paths: dict[str, Path] = {}
    for index, item in enumerate(data.get("attachments", [])):
        filename = str(item.get("filename", "")).strip()
        raw_path = str(item.get("path", "")).strip()
        content_type = str(item.get("content_type", "")).strip()
        attachment_id = f"{message_id or path.stem}:attachment-{index}"
        attachment_path, path_reasons = _resolve_attachment_path(fixture_root, raw_path)
        size_bytes = 0
        digest = ""
        status = "accepted"
        attachment_reasons = list(path_reasons)
        stored_path = raw_path

        if attachment_path is not None:
            attachment_paths[attachment_id] = attachment_path
            if attachment_path.exists() and attachment_path.is_file():
                size_bytes = attachment_path.stat().st_size
                if size_bytes == 0:
                    attachment_reasons.append(reasons.ATTACHMENT_EMPTY)
                digest = sha256_file(attachment_path)
            else:
                attachment_reasons.append(reasons.ATTACHMENT_MISSING_FILE)
        else:
            stored_path = ""

        if attachment_reasons:
            status = "rejected"

        attachments.append(
            InboundEmailAttachment(
                attachment_id=attachment_id,
                message_id=message_id,
                filename=filename or Path(raw_path).name,
                content_type=content_type,
                size_bytes=size_bytes,
                sha256=digest,
                stored_path=stored_path,
                attachment_index=index,
                status=status,
                reasons=_dedupe(attachment_reasons),
            )
        )

    message_reasons: list[str] = []
    if not message_id:
        message_reasons.append(reasons.MISSING_MESSAGE_ID)
    if received_at is None:
        message_reasons.append(reasons.MISSING_RECEIVED_AT)

    message = InboundEmailMessage(
        message_id=message_id,
        thread_id=str(data.get("thread_id", "")).strip(),
        assignment_id=assignment_id,
        received_at=received_at,
        from_email=str(data.get("from_email", "")).strip(),
        from_name=str(data.get("from_name", "")).strip(),
        subject=str(data.get("subject", "")).strip(),
        body_preview=str(data.get("body_preview", "")).strip(),
        attachment_count=len(attachments),
        attachments=attachments,
        source="fixture",
        status="rejected" if message_reasons else "accepted",
        reasons=message_reasons,
        created_at=created_at,
    )
    return LoadedEmailFixture(message=message, fixture_path=path, attachment_paths=attachment_paths)


def _parse_optional_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    return parse_datetime(str(value))


def _resolve_attachment_path(fixtures_dir: Path, raw_path: str) -> tuple[Path | None, list[str]]:
    if not raw_path:
        return None, [reasons.ATTACHMENT_MISSING_FILE]
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return None, [reasons.UNSAFE_ATTACHMENT_PATH]
    resolved = (fixtures_dir / candidate).resolve()
    if not resolved.is_relative_to(fixtures_dir):
        return None, [reasons.UNSAFE_ATTACHMENT_PATH]
    return resolved, []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
