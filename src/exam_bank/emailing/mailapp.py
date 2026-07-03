from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .models import EmailConnectionStatus, EmailMessageSummary, EmailSendResult


MAILAPP_PROVIDER = "mailapp"
MAILAPP_PERMISSION_NOTE = (
    "macOS blocked automation access to Mail.app. Open System Settings -> Privacy & Security -> "
    "Automation and allow your terminal/Python process to control Mail."
)


@dataclass(frozen=True)
class MailAppCommandResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[str], MailAppCommandResult]


class MailAppEmailProvider:
    provider_name = MAILAPP_PROVIDER

    def __init__(
        self,
        *,
        requested_from_address: str | None = None,
        allow_default_mail_account: bool = False,
        runner: Runner | None = None,
    ) -> None:
        self.requested_from_address = requested_from_address
        self.allow_default_mail_account = allow_default_mail_account
        self._runner = runner or run_osascript

    def check_connection(self) -> EmailConnectionStatus:
        checked_at = datetime.now(timezone.utc)
        available = self._runner(build_mailapp_availability_script())
        if available.returncode != 0:
            code = _classify_mailapp_error(available.stderr)
            return EmailConnectionStatus(
                provider=self.provider_name,
                connected=False,
                account_email=self.requested_from_address,
                smtp_ok=False,
                imap_ok=False,
                checked_at=checked_at,
                error_code=code,
                error_message_safe=_safe_mailapp_message(code),
                mailapp_available=False,
                account_verified=False,
                receive_supported=True,
            )

        account_verified: bool | None = None
        account_error: str | None = None
        if self.requested_from_address:
            accounts = self._runner(build_mailapp_account_list_script())
            if accounts.returncode == 0:
                addresses = {line.strip().lower() for line in accounts.stdout.splitlines() if "@" in line}
                account_verified = self.requested_from_address.lower() in addresses
                if not account_verified:
                    account_error = "mailapp_account_not_verified"
            else:
                account_verified = False
                account_error = "mailapp_account_verification_unavailable"

        return EmailConnectionStatus(
            provider=self.provider_name,
            connected=True,
            account_email=self.requested_from_address,
            smtp_ok=True,
            imap_ok=True,
            checked_at=checked_at,
            error_code=account_error,
            error_message_safe=_safe_mailapp_message(account_error),
            mailapp_available=True,
            account_verified=account_verified,
            receive_supported=True,
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
        requested_from = from_address or self.requested_from_address
        missing_attachments = [path for path in attachments or [] if not path.exists()]
        if missing_attachments:
            return _send_error(to, subject, sent_at, "attachment_missing", from_address=requested_from, dry_run=dry_run)
        resolved_attachments = [path.expanduser().resolve() for path in attachments or []]
        if dry_run:
            return EmailSendResult(
                provider=self.provider_name,
                sent=False,
                dry_run=True,
                to=to,
                subject=subject,
                provider_message_id="mailapp-dry-run",
                sent_at=sent_at,
                from_address=requested_from,
            )

        script = build_mailapp_send_script(
            to=to,
            subject=subject,
            body_text=body_text,
            from_address=requested_from,
            attachments=resolved_attachments,
            allow_default_mail_account=self.allow_default_mail_account,
        )
        result = self._runner(script)
        if result.returncode != 0:
            code = _classify_mailapp_error(result.stderr)
            return _send_error(to, subject, sent_at, code, from_address=requested_from, dry_run=dry_run)
        message_id = result.stdout.strip() or None
        return EmailSendResult(
            provider=self.provider_name,
            sent=True,
            dry_run=False,
            to=to,
            subject=subject,
            provider_message_id=message_id,
            sent_at=sent_at,
            from_address=requested_from,
        )

    def search_messages(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> list[EmailMessageSummary]:
        result = self._runner(build_mailapp_search_script(query=query, limit=limit))
        if result.returncode != 0:
            code = _classify_mailapp_error(result.stderr)
            if code == "mailapp_search_unsupported":
                raise MailAppProviderError("mailapp_search_unsupported")
            raise MailAppProviderError(code)
        messages: list[EmailMessageSummary] = []
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) < 5:
                continue
            message_id, sender, subject, date_text = fields[:4]
            if len(fields) > 5 and fields[4].strip().isdigit():
                attachment_count = _parse_int(fields[4])
                snippet = fields[5]
            else:
                snippet = fields[4]
                attachment_count = _parse_int(fields[5]) if len(fields) > 5 else 0
            messages.append(
                EmailMessageSummary(
                    provider=self.provider_name,
                    message_id=message_id or subject,
                    from_address=sender,
                    to_addresses=[],
                    subject=subject,
                    date=_parse_mailapp_date(date_text),
                    snippet=snippet[:240],
                    has_attachments=attachment_count > 0,
                )
            )
        return messages[:limit]

    def export_pdf_attachments(self, *, query: str, target_dir: Path, limit: int = 50) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        result = self._runner(build_mailapp_export_pdf_attachments_script(query=query, target_dir=target_dir, limit=limit))
        if result.returncode != 0:
            raise MailAppProviderError(_classify_mailapp_error(result.stderr))
        return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]

    def export_student_pdf_attachments(
        self,
        *,
        query: str,
        target_dir: Path,
        roster_email_to_student_id: dict[str, str],
        teacher_email: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        roster_lookup = {email.strip().lower(): student_id for email, student_id in roster_email_to_student_id.items() if email.strip() and student_id.strip()}
        if not roster_lookup:
            return []
        scratch_dir = target_dir / ".mailapp_import_tmp"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        result = self._runner(
            build_mailapp_export_roster_pdf_attachment_records_script(
                sender_emails=sorted(roster_lookup),
                target_dir=scratch_dir,
                limit=limit,
            )
        )
        if result.returncode != 0:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            raise MailAppProviderError(_classify_mailapp_error(result.stderr))

        teacher = (teacher_email or self.requested_from_address or "").strip().lower()
        imported: list[Path] = []
        try:
            for line in result.stdout.splitlines():
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                sender, saved_path = fields[:2]
                date_text = fields[4] if len(fields) > 4 else ""
                received_at = _parse_mailapp_date(date_text)
                if since is not None and received_at is not None and received_at < since:
                    continue
                sender_email = _extract_email_address(sender)
                if not sender_email or (teacher and sender_email == teacher):
                    continue
                student_id = roster_lookup.get(sender_email)
                if not student_id:
                    continue
                source = Path(saved_path)
                if not source.is_file():
                    continue
                target = _unique_path(target_dir / f"{_safe_filename_token(student_id)}__{_safe_filename_token(source.name)}")
                shutil.move(source.as_posix(), target.as_posix())
                if received_at is not None:
                    timestamp = received_at.timestamp()
                    os.utime(target, (timestamp, timestamp))
                imported.append(target)
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        return imported


class MailAppProviderError(RuntimeError):
    pass


def run_osascript(script: str) -> MailAppCommandResult:
    completed = subprocess.run(
        ["osascript"],
        check=False,
        capture_output=True,
        input=script,
        text=True,
        timeout=60,
    )
    return MailAppCommandResult(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def build_mailapp_availability_script() -> str:
    return 'tell application "Finder" to return exists application file id "com.apple.mail"'


def build_mailapp_account_list_script() -> str:
    return """
tell application "Mail"
    set oldDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to linefeed
    set accountLines to {}
    repeat with eachAccount in accounts
        try
            repeat with eachAddress in email addresses of eachAccount
                set end of accountLines to (eachAddress as text)
            end repeat
        on error
            set end of accountLines to (name of eachAccount as text)
        end try
    end repeat
    set outputText to accountLines as text
    set AppleScript's text item delimiters to oldDelimiters
    return outputText
end tell
""".strip()


def build_mailapp_send_script(
    *,
    to: str,
    subject: str,
    body_text: str,
    from_address: str | None = None,
    attachments: list[Path] | None = None,
    allow_default_mail_account: bool = False,
) -> str:
    sender_block = ""
    if from_address:
        if allow_default_mail_account:
            sender_block = f"""
            try
                set sender to {_as_applescript_string(from_address)}
            on error
                -- Continue with Mail.app default sender only when explicitly allowed.
            end try
""".rstrip()
        else:
            sender_block = f"""
            try
                set sender to {_as_applescript_string(from_address)}
            on error
                error "mailapp_sender_refused"
            end try
""".rstrip()
    attachment_block = ""
    for index, attachment in enumerate(attachments or [], start=1):
        attachment_block += f"""
        set attachmentFile{index} to POSIX file {_as_applescript_string(str(attachment))}
        tell content
            make new attachment with properties {{file name:attachmentFile{index}}} at after last paragraph
        end tell
""".rstrip()
    attachment_delay = "\n        delay 2" if attachments else ""
    return f"""
tell application "Mail"
    set newMessage to make new outgoing message with properties {{subject:{_as_applescript_string(subject)}, content:{_as_applescript_string(body_text)}, visible:false}}
    tell newMessage
        make new to recipient at end of to recipients with properties {{address:{_as_applescript_string(to)}}}
{sender_block}
{attachment_block}
{attachment_delay}
        send
        try
            return message id
        on error
            return ""
        end try
    end tell
end tell
""".strip()


def build_mailapp_search_script(*, query: str, limit: int = 10) -> str:
    safe_limit = max(1, min(int(limit), 50))
    return f"""
tell application "Mail"
    set oldDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to tab
    set matchedLines to {{}}
    set queryText to {_as_applescript_string(query)}
    repeat with eachAccount in accounts
        repeat with eachMailbox in mailboxes of eachAccount
            try
                set foundMessages to messages of eachMailbox whose subject contains queryText
                repeat with eachMessage in foundMessages
                    if (count of matchedLines) is greater than or equal to {safe_limit} then exit repeat
                    set attachmentCount to 0
                    repeat with eachAttachment in mail attachments of eachMessage
                        set attachmentCount to attachmentCount + 1
                    end repeat
                    set lineFields to {{message id of eachMessage as text, sender of eachMessage as text, subject of eachMessage as text, date received of eachMessage as text, attachmentCount as text, content of eachMessage as text}}
                    set end of matchedLines to lineFields as text
                end repeat
            on error
            end try
            if (count of matchedLines) is greater than or equal to {safe_limit} then exit repeat
        end repeat
        if (count of matchedLines) is greater than or equal to {safe_limit} then exit repeat
    end repeat
    set AppleScript's text item delimiters to linefeed
    set outputText to matchedLines as text
    set AppleScript's text item delimiters to oldDelimiters
    return outputText
end tell
""".strip()


def build_mailapp_export_pdf_attachments_script(*, query: str, target_dir: Path, limit: int = 50) -> str:
    safe_limit = max(1, min(int(limit), 200))
    target = str(target_dir.resolve()) + "/"
    return f"""
tell application "Mail"
    set oldDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to linefeed
    set queryText to {_as_applescript_string(query)}
    set targetFolder to POSIX file {_as_applescript_string(target)} as alias
    set savedPaths to {{}}
    repeat with eachAccount in accounts
        repeat with eachMailbox in mailboxes of eachAccount
            try
                set foundMessages to messages of eachMailbox whose subject contains queryText
                repeat with eachMessage in foundMessages
                    if (count of savedPaths) is greater than or equal to {safe_limit} then exit repeat
                    repeat with eachAttachment in mail attachments of eachMessage
                        try
                            set attachmentName to name of eachAttachment as text
                            if attachmentName ends with ".pdf" or attachmentName ends with ".PDF" then
                                set saveFile to ((targetFolder as text) & attachmentName)
                                save eachAttachment in file saveFile
                                set end of savedPaths to POSIX path of saveFile
                            end if
                        end try
                    end repeat
                end repeat
            on error
            end try
            if (count of savedPaths) is greater than or equal to {safe_limit} then exit repeat
        end repeat
        if (count of savedPaths) is greater than or equal to {safe_limit} then exit repeat
    end repeat
    set outputText to savedPaths as text
    set AppleScript's text item delimiters to oldDelimiters
    return outputText
end tell
""".strip()


def build_mailapp_export_pdf_attachment_records_script(*, query: str, target_dir: Path, limit: int = 50) -> str:
    safe_limit = max(1, min(int(limit), 200))
    target = str(target_dir.resolve()) + "/"
    return f"""
tell application "Mail"
    set oldDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to linefeed
    set queryText to {_as_applescript_string(query)}
    set targetFolder to POSIX file {_as_applescript_string(target)} as alias
    set savedLines to {{}}
    set savedCount to 0
    repeat with eachAccount in accounts
        repeat with eachMailbox in mailboxes of eachAccount
            try
                set foundMessages to messages of eachMailbox whose subject contains queryText
                repeat with eachMessage in foundMessages
                    if savedCount is greater than or equal to {safe_limit} then exit repeat
                    repeat with eachAttachment in mail attachments of eachMessage
                        try
                            set attachmentName to name of eachAttachment as text
                            if attachmentName ends with ".pdf" or attachmentName ends with ".PDF" then
                                set savedCount to savedCount + 1
                                set saveFile to ((targetFolder as text) & (savedCount as text) & "_" & attachmentName)
                                save eachAttachment in file saveFile
                                set AppleScript's text item delimiters to tab
                                set lineFields to {{sender of eachMessage as text, POSIX path of saveFile, subject of eachMessage as text, message id of eachMessage as text}}
                                set end of savedLines to lineFields as text
                                set AppleScript's text item delimiters to linefeed
                            end if
                        end try
                        if savedCount is greater than or equal to {safe_limit} then exit repeat
                    end repeat
                end repeat
            on error
            end try
            if savedCount is greater than or equal to {safe_limit} then exit repeat
        end repeat
        if savedCount is greater than or equal to {safe_limit} then exit repeat
    end repeat
    set outputText to savedLines as text
    set AppleScript's text item delimiters to oldDelimiters
    return outputText
end tell
""".strip()


def build_mailapp_export_roster_pdf_attachment_records_script(*, sender_emails: list[str], target_dir: Path, limit: int = 50) -> str:
    safe_limit = max(1, min(int(limit), 500))
    target = str(target_dir.resolve()) + "/"
    sender_list = _as_applescript_list(sender_emails)
    return f"""
tell application "Mail"
    set oldDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to linefeed
    set senderQueries to {sender_list}
    set targetFolder to POSIX file {_as_applescript_string(target)} as alias
    set savedLines to {{}}
    set savedCount to 0
    repeat with eachAccount in accounts
        repeat with eachMailbox in mailboxes of eachAccount
            try
                repeat with eachSenderQuery in senderQueries
                    if savedCount is greater than or equal to {safe_limit} then exit repeat
                    set foundMessages to messages of eachMailbox whose sender contains (eachSenderQuery as text)
                    repeat with eachMessage in foundMessages
                        if savedCount is greater than or equal to {safe_limit} then exit repeat
                        repeat with eachAttachment in mail attachments of eachMessage
                            try
                                set attachmentName to name of eachAttachment as text
                                if attachmentName ends with ".pdf" or attachmentName ends with ".PDF" then
                                    set savedCount to savedCount + 1
                                    set saveFile to ((targetFolder as text) & (savedCount as text) & "_" & attachmentName)
                                    save eachAttachment in file saveFile
                                    set AppleScript's text item delimiters to tab
                                    set lineFields to {{sender of eachMessage as text, POSIX path of saveFile, subject of eachMessage as text, message id of eachMessage as text, date received of eachMessage as text}}
                                    set end of savedLines to lineFields as text
                                    set AppleScript's text item delimiters to linefeed
                                end if
                            end try
                            if savedCount is greater than or equal to {safe_limit} then exit repeat
                        end repeat
                    end repeat
                end repeat
            on error
            end try
            if savedCount is greater than or equal to {safe_limit} then exit repeat
        end repeat
        if savedCount is greater than or equal to {safe_limit} then exit repeat
    end repeat
    set outputText to savedLines as text
    set AppleScript's text item delimiters to oldDelimiters
    return outputText
end tell
""".strip()


def _as_applescript_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n")
    return f'"{escaped}"'


def _as_applescript_list(values: list[str]) -> str:
    return "{" + ", ".join(_as_applescript_string(value) for value in values) + "}"


def _extract_email_address(value: str) -> str:
    match = re.search(r"<([^<>@\s]+@[^<>\s]+)>", value)
    if match:
        return match.group(1).strip().lower()
    match = re.search(r"([A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", value)
    return match.group(1).strip().lower() if match else ""


def _safe_filename_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return cleaned or "attachment"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not choose unique path for {path}")


def _classify_mailapp_error(stderr: str) -> str:
    text = stderr.lower()
    if "not allowed" in text or "not authorized" in text or "privacy" in text or "-1743" in text:
        return "mailapp_automation_permission_blocked"
    if "mailapp_sender_refused" in text:
        return "mailapp_sender_refused"
    if "can't get application" in text or "application isn't running" in text or "application file" in text:
        return "mailapp_unavailable"
    if "whose" in text or "messages of" in text:
        return "mailapp_search_unsupported"
    return "mailapp_osascript_failed"


def _safe_mailapp_message(code: str | None) -> str | None:
    if code is None:
        return None
    messages = {
        "attachments_blocked": "Attachments are blocked for email smoke tests.",
        "attachment_missing": "One or more requested attachments do not exist.",
        "mailapp_account_not_verified": "Mail.app is available, but the requested account address was not found.",
        "mailapp_account_verification_unavailable": "Mail.app is available, but account address could not be verified automatically.",
        "mailapp_automation_permission_blocked": MAILAPP_PERMISSION_NOTE,
        "mailapp_osascript_failed": "Mail.app AppleScript operation failed.",
        "mailapp_search_unsupported": "Mail.app local receive/search is not reliable through this provider.",
        "mailapp_sender_refused": "Mail.app refused the requested sender. Pass --allow-default-mail-account to use Mail.app's default account.",
        "mailapp_unavailable": "Mail.app is not available on this system.",
    }
    return messages.get(code, "Mail.app provider operation failed.")


def _send_error(
    to: str,
    subject: str,
    sent_at: datetime,
    code: str,
    *,
    from_address: str | None,
    dry_run: bool,
) -> EmailSendResult:
    return EmailSendResult(
        provider=MAILAPP_PROVIDER,
        sent=False,
        dry_run=dry_run,
        to=to,
        subject=subject,
        provider_message_id=None,
        sent_at=sent_at,
        error_code=code,
        error_message_safe=_safe_mailapp_message(code),
        from_address=from_address,
    )


def _parse_mailapp_date(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%A, %B %d, %Y at %I:%M:%S %p", "%A, %d %B %Y at %H:%M:%S", "%m/%d/%y, %I:%M:%S %p"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0
