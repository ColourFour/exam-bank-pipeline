"""Local-first assignment submission tracking."""

from exam_bank.submissions.draft_grading import build_submission_draft_grades, create_draft_grading_result
from exam_bank.submissions.email_identity import match_student_for_email
from exam_bank.submissions.email_models import (
    EmailIntakeDryRunDecision,
    EmailSubmissionProvenance,
    InboundEmailAttachment,
    InboundEmailMessage,
    StudentMatchCandidate,
    StudentMatchResult,
)
from exam_bank.submissions.email_policy import build_email_dry_run_decision, duplicate_or_resend_reasons
from exam_bank.submissions.email_connectors import EmailConnector, EmailMessageRef, FakeEmailConnector, LocalExportEmailConnector
from exam_bank.submissions.extraction import extract_submission_pdf
from exam_bank.submissions.ingest import ingest_assignment_submissions
from exam_bank.submissions.live_email_import import import_live_email_submissions
from exam_bank.submissions.models import (
    Assignment,
    CompletionRow,
    DraftGradingResult,
    DraftGradingSummary,
    DraftQuestionResult,
    FeedbackDraft,
    Student,
    Submission,
    SubmissionExtractionResult,
)
from exam_bank.submissions.outgoing_email import (
    OutgoingEmailAuditEvent,
    OutgoingEmailDraft,
    OutgoingEmailQueueItem,
    OutgoingEmailSummary,
    build_outgoing_email_queue,
    send_outgoing_email_fake,
    write_outgoing_email_dry_run_report,
)
from exam_bank.submissions.review_queue import (
    ManualGradingPrepRecord,
    ReviewQueueSummary,
    SubmissionReviewRecord,
    build_submission_review_queue,
)

__all__ = [
    "Assignment",
    "CompletionRow",
    "DraftGradingResult",
    "DraftGradingSummary",
    "DraftQuestionResult",
    "FeedbackDraft",
    "Student",
    "Submission",
    "SubmissionExtractionResult",
    "SubmissionReviewRecord",
    "ManualGradingPrepRecord",
    "ReviewQueueSummary",
    "InboundEmailAttachment",
    "InboundEmailMessage",
    "StudentMatchCandidate",
    "StudentMatchResult",
    "EmailIntakeDryRunDecision",
    "EmailSubmissionProvenance",
    "EmailConnector",
    "EmailMessageRef",
    "FakeEmailConnector",
    "LocalExportEmailConnector",
    "OutgoingEmailDraft",
    "OutgoingEmailQueueItem",
    "OutgoingEmailAuditEvent",
    "OutgoingEmailSummary",
    "ingest_assignment_submissions",
    "build_submission_review_queue",
    "extract_submission_pdf",
    "create_draft_grading_result",
    "build_submission_draft_grades",
    "match_student_for_email",
    "duplicate_or_resend_reasons",
    "build_email_dry_run_decision",
    "build_outgoing_email_queue",
    "write_outgoing_email_dry_run_report",
    "send_outgoing_email_fake",
    "import_live_email_submissions",
]
