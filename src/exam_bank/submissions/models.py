from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

PHASE3_GRADING_MODE = "draft_auto"
PHASE3_TEACHER_REVIEW_REQUIRED = True
PHASE3_STUDENT_FACING = False

EXTRACTION_STATUSES = {"not_attempted", "extracted", "partial", "failed"}
DRAFT_GRADING_STATUSES = {"not_attempted", "draft_created", "needs_review", "failed"}
DRAFT_GRADING_CONFIDENCE = {"none", "low", "medium", "high"}


@dataclass(frozen=True)
class Assignment:
    assignment_id: str
    course_id: str
    title: str
    class_id: str
    due_at: datetime | None
    timezone: str
    accepted_file_types: list[str]
    max_files_per_student: int
    max_file_size_mb: int
    allow_late: bool
    source_question_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Student:
    student_id: str
    class_id: str
    display_name: str
    email: str = ""
    active: bool = True
    source_file: str = ""


@dataclass(frozen=True)
class Submission:
    submission_id: str
    assignment_id: str
    student_id: str
    source_filename: str
    stored_pdf_path: str
    sha256: str
    received_at: datetime
    submitted_via: str
    status: str
    late: bool
    validation_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompletionRow:
    assignment_id: str
    assignment_title: str
    class_id: str
    student_id: str
    display_name: str
    email: str
    status: str
    submitted_at: str
    late: bool
    source_filename: str
    stored_pdf_path: str
    rejection_reasons: str
    notes: str


@dataclass(frozen=True)
class FeedbackDraft:
    draft_id: str
    assignment_id: str
    student_id: str
    draft_type: str
    recipient_email: str
    subject: str
    body_text: str
    send_allowed: bool
    created_at: datetime


@dataclass(frozen=True)
class SubmissionExtractionResult:
    extraction_id: str
    assignment_id: str
    student_id: str
    submission_id: str
    stored_pdf_path: str
    status: str
    page_count: int
    text_extractable: bool
    extracted_text_preview: str
    extraction_warnings: list[str]
    created_at: datetime
    grading_mode: str = PHASE3_GRADING_MODE
    student_facing: bool = PHASE3_STUDENT_FACING
    teacher_review_required: bool = PHASE3_TEACHER_REVIEW_REQUIRED

    def __post_init__(self) -> None:
        _require_allowed("extraction status", self.status, EXTRACTION_STATUSES)
        _require_phase3_flags(self.grading_mode, self.student_facing, self.teacher_review_required)


@dataclass(frozen=True)
class DraftQuestionResult:
    question_id: str
    question_label: str
    draft_score: float | None
    draft_max_score: float | None
    confidence: str
    evidence: list[str]
    notes: list[str]
    review_required: bool
    grading_mode: str = PHASE3_GRADING_MODE
    student_facing: bool = PHASE3_STUDENT_FACING
    teacher_review_required: bool = PHASE3_TEACHER_REVIEW_REQUIRED

    def __post_init__(self) -> None:
        _require_allowed("draft question confidence", self.confidence, DRAFT_GRADING_CONFIDENCE)
        _require_phase3_flags(self.grading_mode, self.student_facing, self.teacher_review_required)
        if self.review_required is not True:
            raise ValueError("Draft question results must require review in Phase 3")


@dataclass(frozen=True)
class DraftGradingResult:
    draft_grading_id: str
    grading_result_id: str
    assignment_id: str
    student_id: str
    submission_id: str
    grading_mode: str
    status: str
    draft_score: float | None
    draft_max_score: float | None
    confidence: str
    confidence_reasons: list[str]
    teacher_review_required: bool
    student_facing: bool
    question_results: list[DraftQuestionResult]
    overall_notes: list[str]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_allowed("draft grading status", self.status, DRAFT_GRADING_STATUSES)
        _require_allowed("draft grading confidence", self.confidence, DRAFT_GRADING_CONFIDENCE)
        _require_phase3_flags(self.grading_mode, self.student_facing, self.teacher_review_required)


@dataclass(frozen=True)
class DraftGradingSummary:
    assignment_id: str
    submissions_attempted: int
    drafts_created: int
    failed_count: int
    low_confidence_count: int
    medium_confidence_count: int
    high_confidence_count: int
    teacher_review_required_count: int
    student_facing_count: int
    created_at: datetime
    grading_mode: str = PHASE3_GRADING_MODE
    student_facing: bool = PHASE3_STUDENT_FACING
    teacher_review_required: bool = PHASE3_TEACHER_REVIEW_REQUIRED

    def __post_init__(self) -> None:
        _require_phase3_flags(self.grading_mode, self.student_facing, self.teacher_review_required)
        if self.student_facing_count != 0:
            raise ValueError("Phase 3 draft grading summary student_facing_count must be 0")


def dataclass_to_json_dict(value: object) -> dict[str, object]:
    payload = asdict(value)
    for key, item in list(payload.items()):
        if isinstance(item, datetime):
            payload[key] = item.isoformat()
    return payload


def _require_allowed(label: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid {label}: {value}")


def _require_phase3_flags(grading_mode: str, student_facing: bool, teacher_review_required: bool) -> None:
    if grading_mode != PHASE3_GRADING_MODE:
        raise ValueError("Phase 3 grading outputs must use grading_mode='draft_auto'")
    if student_facing is not PHASE3_STUDENT_FACING:
        raise ValueError("Phase 3 grading outputs must use student_facing=false")
    if teacher_review_required is not PHASE3_TEACHER_REVIEW_REQUIRED:
        raise ValueError("Phase 3 grading outputs must use teacher_review_required=true")
