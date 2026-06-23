from __future__ import annotations

from dataclasses import replace

import exam_bank.submissions.email_reasons as reasons
from exam_bank.submissions.email_identity import match_student_for_email
from exam_bank.submissions.email_models import (
    EmailIntakeDryRunDecision,
    InboundEmailAttachment,
    InboundEmailMessage,
)
from exam_bank.submissions.models import Assignment, Student


def duplicate_or_resend_reasons(
    *,
    message: InboundEmailMessage,
    student_id: str,
    seen_message_ids: set[str] | None = None,
    seen_attachment_hashes: set[str] | None = None,
    accepted_student_ids: set[str] | None = None,
) -> list[str]:
    result: list[str] = []
    seen_message_ids = seen_message_ids or set()
    seen_attachment_hashes = seen_attachment_hashes or set()
    accepted_student_ids = accepted_student_ids or set()

    if message.message_id in seen_message_ids:
        result.append(reasons.DUPLICATE_MESSAGE_ID)
    if any(attachment.sha256 and attachment.sha256 in seen_attachment_hashes for attachment in message.attachments):
        result.append(reasons.DUPLICATE_ATTACHMENT_HASH)
    if student_id and student_id in accepted_student_ids:
        result.append(reasons.RESUBMISSION_DETECTED)
    return result


def build_email_dry_run_decision(
    *,
    message: InboundEmailMessage,
    roster: list[Student],
    assignment: Assignment,
    seen_message_ids: set[str] | None = None,
    seen_attachment_hashes: set[str] | None = None,
    accepted_student_ids: set[str] | None = None,
) -> EmailIntakeDryRunDecision:
    student_match = match_student_for_email(message, roster)
    accepted_attachments: list[InboundEmailAttachment] = []
    quarantined_attachments: list[InboundEmailAttachment] = []
    rejected_attachments: list[InboundEmailAttachment] = []
    decision_reasons = [reasons.DRY_RUN_ONLY]

    if not message.message_id:
        decision_reasons.append(reasons.MISSING_MESSAGE_ID)
    if message.received_at is None:
        decision_reasons.append(reasons.MISSING_RECEIVED_AT)
    if message.assignment_id != assignment.assignment_id:
        decision_reasons.append(reasons.ASSIGNMENT_MISMATCH)
    if student_match.status != "matched":
        decision_reasons.extend(student_match.reasons)

    pdf_attachments = [attachment for attachment in message.attachments if _is_pdf_attachment(attachment)]
    if not pdf_attachments:
        decision_reasons.append(reasons.NO_PDF_ATTACHMENT)
    if assignment.max_files_per_student >= 0 and len(pdf_attachments) > assignment.max_files_per_student:
        decision_reasons.append(reasons.TOO_MANY_PDF_ATTACHMENTS)

    duplicate_reasons = duplicate_or_resend_reasons(
        message=message,
        student_id=student_match.student_id,
        seen_message_ids=seen_message_ids,
        seen_attachment_hashes=seen_attachment_hashes,
        accepted_student_ids=accepted_student_ids,
    )
    decision_reasons.extend(duplicate_reasons)

    for attachment in message.attachments:
        attachment_reasons = _attachment_reasons(attachment, assignment, seen_attachment_hashes or set())
        if reasons.DUPLICATE_MESSAGE_ID in duplicate_reasons or reasons.TOO_MANY_PDF_ATTACHMENTS in decision_reasons:
            attachment_reasons.append(reasons.DRY_RUN_ONLY)
        if attachment_reasons:
            status = "rejected" if reasons.NON_PDF_ATTACHMENT in attachment_reasons else "quarantined"
            updated = replace(attachment, status=status, reasons=_dedupe([*attachment.reasons, *attachment_reasons]))
            if status == "rejected":
                rejected_attachments.append(updated)
            else:
                quarantined_attachments.append(updated)
        else:
            accepted_attachments.append(replace(attachment, status="accepted", reasons=[]))

    status = "accepted"
    if (
        student_match.status != "matched"
        or reasons.MISSING_MESSAGE_ID in decision_reasons
        or reasons.MISSING_RECEIVED_AT in decision_reasons
        or reasons.ASSIGNMENT_MISMATCH in decision_reasons
        or reasons.DUPLICATE_MESSAGE_ID in decision_reasons
        or reasons.DUPLICATE_ATTACHMENT_HASH in decision_reasons
        or reasons.TOO_MANY_PDF_ATTACHMENTS in decision_reasons
        or reasons.NO_PDF_ATTACHMENT in decision_reasons
        or quarantined_attachments
    ):
        status = "quarantined"
    if (
        rejected_attachments
        and not accepted_attachments
        and not quarantined_attachments
        and reasons.NO_PDF_ATTACHMENT not in decision_reasons
    ):
        status = "rejected"

    return EmailIntakeDryRunDecision(
        message_id=message.message_id,
        status=status,
        assignment_id=assignment.assignment_id,
        student_match_status=student_match.status,
        student_id=student_match.student_id,
        accepted_attachments=accepted_attachments,
        quarantined_attachments=quarantined_attachments,
        rejected_attachments=rejected_attachments,
        reasons=_dedupe(decision_reasons),
        would_stage_pdf=False,
        would_create_submission=False,
        would_create_draft=student_match.status == "matched",
    )


def _attachment_reasons(attachment: InboundEmailAttachment, assignment: Assignment, seen_hashes: set[str]) -> list[str]:
    result: list[str] = []
    if not _is_pdf_attachment(attachment):
        result.append(reasons.NON_PDF_ATTACHMENT)
    if attachment.size_bytes == 0:
        result.append(reasons.ATTACHMENT_EMPTY)
    max_bytes = assignment.max_file_size_mb * 1024 * 1024
    if max_bytes >= 0 and attachment.size_bytes > max_bytes:
        result.append(reasons.ATTACHMENT_TOO_LARGE)
    if attachment.stored_path and (".." in attachment.stored_path.split("/") or attachment.stored_path.startswith("/")):
        result.append(reasons.UNSAFE_ATTACHMENT_PATH)
    if attachment.sha256 and attachment.sha256 in seen_hashes:
        result.append(reasons.DUPLICATE_ATTACHMENT_HASH)
    return result


def _is_pdf_attachment(attachment: InboundEmailAttachment) -> bool:
    filename_is_pdf = attachment.filename.lower().endswith(".pdf")
    content_type_is_pdf = attachment.content_type.lower() in {"application/pdf", "application/x-pdf"}
    return filename_is_pdf or content_type_is_pdf


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
