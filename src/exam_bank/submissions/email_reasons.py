"""Machine-readable reasons for email intake readiness decisions."""

UNKNOWN_STUDENT = "unknown_student"
AMBIGUOUS_STUDENT_MATCH = "ambiguous_student_match"
MISSING_MESSAGE_ID = "missing_message_id"
DUPLICATE_MESSAGE_ID = "duplicate_message_id"
MISSING_RECEIVED_AT = "missing_received_at"
NO_PDF_ATTACHMENT = "no_pdf_attachment"
NON_PDF_ATTACHMENT = "non_pdf_attachment"
TOO_MANY_PDF_ATTACHMENTS = "too_many_pdf_attachments"
ATTACHMENT_MISSING_FILE = "attachment_missing_file"
ATTACHMENT_EMPTY = "attachment_empty"
ATTACHMENT_TOO_LARGE = "attachment_too_large"
UNSAFE_ATTACHMENT_PATH = "unsafe_attachment_path"
PDF_VALIDATION_FAILED = "pdf_validation_failed"
ASSIGNMENT_MISMATCH = "assignment_mismatch"
DUPLICATE_ATTACHMENT_HASH = "duplicate_attachment_hash"
RESUBMISSION_DETECTED = "resubmission_detected"
DRY_RUN_ONLY = "dry_run_only"
MATCHED_BY_ATTACHMENT_FILENAME = "matched_by_attachment_filename"
MATCHED_BY_SUBJECT = "matched_by_subject"
MATCHED_BY_SENDER_EMAIL = "matched_by_sender_email"
MATCHED_BY_BODY = "matched_by_body"
MULTIPLE_STUDENT_IDS_FOUND = "multiple_student_ids_found"
NO_ROSTER_EMAIL_MATCH = "no_roster_email_match"

EMAIL_INTAKE_REASONS = {
    UNKNOWN_STUDENT,
    AMBIGUOUS_STUDENT_MATCH,
    MISSING_MESSAGE_ID,
    DUPLICATE_MESSAGE_ID,
    MISSING_RECEIVED_AT,
    NO_PDF_ATTACHMENT,
    NON_PDF_ATTACHMENT,
    TOO_MANY_PDF_ATTACHMENTS,
    ATTACHMENT_MISSING_FILE,
    ATTACHMENT_EMPTY,
    ATTACHMENT_TOO_LARGE,
    UNSAFE_ATTACHMENT_PATH,
    PDF_VALIDATION_FAILED,
    ASSIGNMENT_MISMATCH,
    DUPLICATE_ATTACHMENT_HASH,
    RESUBMISSION_DETECTED,
    DRY_RUN_ONLY,
    MATCHED_BY_ATTACHMENT_FILENAME,
    MATCHED_BY_SUBJECT,
    MATCHED_BY_SENDER_EMAIL,
    MATCHED_BY_BODY,
    MULTIPLE_STUDENT_IDS_FOUND,
    NO_ROSTER_EMAIL_MATCH,
}
