from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


EMAIL_MESSAGE_STATUSES = {"accepted", "quarantined", "rejected"}
EMAIL_ATTACHMENT_STATUSES = {"accepted", "quarantined", "rejected"}
STUDENT_MATCH_SOURCES = {
    "attachment_filename_student_id",
    "subject_student_id",
    "sender_email",
    "body_student_id",
}
STUDENT_MATCH_CONFIDENCE = {"low", "medium", "high"}
STUDENT_MATCH_STATUSES = {"matched", "ambiguous", "unknown"}
EMAIL_DRY_RUN_STATUSES = {"accepted", "quarantined", "rejected"}


@dataclass(frozen=True)
class InboundEmailAttachment:
    attachment_id: str
    message_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    stored_path: str
    attachment_index: int
    status: str
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_allowed("email attachment status", self.status, EMAIL_ATTACHMENT_STATUSES)
        if self.size_bytes < 0:
            raise ValueError("Email attachment size_bytes must be non-negative")
        if self.attachment_index < 0:
            raise ValueError("Email attachment attachment_index must be non-negative")


@dataclass(frozen=True)
class InboundEmailMessage:
    message_id: str
    thread_id: str
    assignment_id: str
    received_at: datetime | None
    from_email: str
    from_name: str
    subject: str
    body_preview: str
    attachment_count: int
    attachments: list[InboundEmailAttachment]
    source: str
    status: str
    reasons: list[str]
    created_at: datetime

    def __post_init__(self) -> None:
        _require_allowed("email message status", self.status, EMAIL_MESSAGE_STATUSES)
        if self.attachment_count < 0:
            raise ValueError("Email message attachment_count must be non-negative")
        if self.attachment_count != len(self.attachments):
            raise ValueError("Email message attachment_count must match attachments length")


@dataclass(frozen=True)
class StudentMatchCandidate:
    student_id: str
    match_source: str
    confidence: str
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_allowed("student match source", self.match_source, STUDENT_MATCH_SOURCES)
        _require_allowed("student match confidence", self.confidence, STUDENT_MATCH_CONFIDENCE)


@dataclass(frozen=True)
class StudentMatchResult:
    status: str
    student_id: str
    candidates: list[StudentMatchCandidate]
    reasons: list[str]

    def __post_init__(self) -> None:
        _require_allowed("student match status", self.status, STUDENT_MATCH_STATUSES)
        if self.status == "matched" and not self.student_id:
            raise ValueError("Matched student result must include student_id")
        if self.status != "matched" and self.student_id:
            raise ValueError("Only matched student results may include student_id")


@dataclass(frozen=True)
class EmailIntakeDryRunDecision:
    message_id: str
    status: str
    assignment_id: str
    student_match_status: str
    student_id: str
    accepted_attachments: list[InboundEmailAttachment]
    quarantined_attachments: list[InboundEmailAttachment]
    rejected_attachments: list[InboundEmailAttachment]
    reasons: list[str]
    would_stage_pdf: bool
    would_create_submission: bool
    would_create_draft: bool

    def __post_init__(self) -> None:
        _require_allowed("email dry-run status", self.status, EMAIL_DRY_RUN_STATUSES)
        _require_allowed("student match status", self.student_match_status, STUDENT_MATCH_STATUSES)


@dataclass(frozen=True)
class EmailSubmissionProvenance:
    source: str
    message_id: str
    thread_id: str
    from_email: str
    received_at: datetime | None
    original_attachment_filename: str
    attachment_id: str
    attachment_sha256: str


def _require_allowed(label: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid {label}: {value}")
