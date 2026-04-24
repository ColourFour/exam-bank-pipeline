from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .trust import Confidence, CurationStatus, QuestionTextRole, QuestionTextTrust, ValidationStatus


@dataclass(frozen=True)
class BoundingBox:
    """PDF-space rectangle in points."""

    x0: float
    y0: float
    x1: float
    y1: float

    def padded(self, amount: float, width: float, height: float) -> "BoundingBox":
        return BoundingBox(
            max(0, self.x0 - amount),
            max(0, self.y0 - amount),
            min(width, self.x1 + amount),
            min(height, self.y1 + amount),
        )


@dataclass(frozen=True)
class TextBlock:
    page_number: int
    text: str
    bbox: BoundingBox
    source: str = "pdf"
    confidence: float | None = None
    font_size: float | None = None
    font_name: str | None = None
    is_bold: bool = False

    @property
    def first_line(self) -> str:
        for line in self.text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""


@dataclass(frozen=True)
class PageLayout:
    page_number: int
    width: float
    height: float
    blocks: list[TextBlock]
    graphics: list[BoundingBox] = field(default_factory=list)
    text_source: str = "pdf"
    extraction_warning: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text.strip())


@dataclass(frozen=True)
class QuestionStart:
    question_number: str
    page_number: int
    y0: float
    x0: float
    label: str
    block_index: int
    bbox: BoundingBox | None = None
    font_size: float | None = None
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class QuestionSpan:
    source_pdf: Path
    paper_name: str
    question_number: str
    start_page: int
    start_y: float
    end_page: int
    end_y: float
    page_numbers: list[int]
    blocks: list[TextBlock]
    full_question_label: str
    review_flags: list[str] = field(default_factory=list)
    anchor: QuestionStart | None = None
    validation_status: str = ValidationStatus.PASS
    validation_flags: list[str] = field(default_factory=list)
    recovery_attempted: bool = False
    recovery_result: str = ""
    structure_detected: dict[str, Any] = field(default_factory=dict)
    question_total_detected: int | None = None
    format_profile: str = "legacy"

    @property
    def combined_text(self) -> str:
        return "\n".join(_clean_model_text(block.text) for block in self.blocks if _clean_model_text(block.text)).strip()


@dataclass
class ClassificationResult:
    paper_family: str
    source_paper_code: str
    source_paper_family: str
    inferred_paper_family: str
    paper_family_confidence: str
    topic: str
    subtopic: str
    difficulty: str
    difficulty_confidence: str
    difficulty_evidence: str
    difficulty_uncertain: bool
    confidence: float
    review_flags: list[str] = field(default_factory=list)
    topic_confidence: str = Confidence.LOW
    topic_evidence: str = ""
    topic_evidence_details: dict[str, Any] = field(default_factory=dict)
    secondary_topics: list[str] = field(default_factory=list)
    topic_uncertain: bool = False
    alternative_topics: list[str] = field(default_factory=list)


@dataclass
class RenderResult:
    screenshot_path: Path
    review_flags: list[str] = field(default_factory=list)
    crop_uncertain: bool = False
    debug_paths: list[str] = field(default_factory=list)
    extracted_text: str = ""
    crop_diagnostics: dict[str, Any] = field(default_factory=dict)
    ocr_ran: bool = False
    ocr_engine: str = ""
    ocr_text: str = ""
    ocr_text_trust: str = QuestionTextTrust.UNUSABLE
    ocr_failure_reason: str = "disabled"
    ocr_text_role: str = QuestionTextRole.MISSING


@dataclass(frozen=True)
class QuestionExtractionState:
    source_pdf: str
    paper_name: str
    question_number: str
    full_question_label: str
    combined_question_text: str
    body_text_raw: str
    body_text_normalized: str
    math_lines: list[str]
    diagram_text: list[str]
    extraction_quality_score: float
    extraction_quality_flags: list[str]
    part_texts: list[dict[str, Any]]
    page_numbers: list[int]
    question_subparts: list[str]
    question_marks_total: int | None
    question_structure_detected: dict[str, Any]
    question_total_detected: int | None
    question_format_profile: str
    recovery_attempted: bool
    recovery_result: str
    ocr_ran: bool
    ocr_engine: str
    ocr_text: str
    ocr_text_trust: str
    ocr_failure_reason: str
    ocr_text_role: str


@dataclass(frozen=True)
class QuestionClassificationState:
    paper_family: str
    source_paper_family: str
    inferred_paper_family: str
    paper_family_confidence: str
    question_level_paper_family: str
    topic: str
    subtopic: str
    question_level_topic: str
    question_level_subtopic: str
    part_level_topics: list[dict[str, Any]]
    topic_confidence: str
    topic_evidence: str
    topic_evidence_details: dict[str, Any]
    examiner_report_evidence: dict[str, Any]
    secondary_topics: list[str]
    topic_uncertain: bool
    topic_alternatives: list[str]
    difficulty: str
    difficulty_confidence: str
    difficulty_evidence: str
    difficulty_uncertain: bool
    confidence: float


@dataclass(frozen=True)
class QuestionImageState:
    screenshot_path: str
    crop_uncertain: bool
    question_crop_confidence: str
    crop_debug_paths: list[str]
    question_crop_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MarkSchemeState:
    source_pdf: str
    answer_text: str
    image_path: str
    pages: list[int]
    question_number: str
    crop_confidence: str
    mapping_method: str
    table_detected: bool
    table_header_detected: list[str]
    nearby_anchors: list[str]
    debug_paths: list[str]
    table_header_ok: bool
    continuation_rows_included: bool
    question_subparts: list[str]
    markscheme_subparts: list[str]
    question_marks_total: int | None
    markscheme_marks_total: int | None
    mapping_status: str
    failure_reason: str
    structure_detected: dict[str, Any]
    total_detected: int | None


@dataclass(frozen=True)
class ValidationTrustState:
    review_flags: list[str]
    validation_status: str
    validation_flags: list[str]
    scope_quality_status: str
    text_source_profile: str
    text_fidelity_status: str
    text_fidelity_flags: list[str]
    question_text_role: str
    question_text_trust: str
    visual_required: bool
    visual_reason_flags: list[str]
    visual_curation_status: str
    text_only_status: str
    topic_trust_status: str


@dataclass(frozen=True)
class PaperMetadataState:
    source_paper_code: str
    syllabus_code: str
    session: str
    year: str
    document_type: str
    component: str
    document_key: str
    metadata_source: str


@dataclass(frozen=True)
class PaperTotalState:
    expected: int | None
    detected: int | None
    status: str
    rescan_triggered: bool
    rescan_result: str
    before_rescan: int | None
    after_rescan: int | None
    focus_questions: list[str]
    focus_pages: list[int]
    focus_reason: str


@dataclass
class QuestionRecord:
    source_pdf: str
    paper_name: str
    question_number: str
    full_question_label: str
    screenshot_path: str
    combined_question_text: str
    body_text_raw: str
    body_text_normalized: str
    math_lines: list[str]
    diagram_text: list[str]
    extraction_quality_score: float
    extraction_quality_flags: list[str]
    part_texts: list[dict[str, Any]]
    answer_text: str
    paper_family: str
    source_paper_family: str
    inferred_paper_family: str
    paper_family_confidence: str
    topic: str
    subtopic: str
    topic_confidence: str
    topic_evidence: str
    secondary_topics: list[str]
    topic_uncertain: bool
    difficulty: str
    difficulty_confidence: str
    difficulty_evidence: str
    difficulty_uncertain: bool
    marks: int | None
    marks_if_available: int | None
    page_numbers: list[int]
    review_flags: list[str]
    confidence: float
    source_paper_code: str = ""
    syllabus_code: str = ""
    session: str = ""
    year: str = ""
    document_type: str = ""
    component: str = ""
    document_key: str = ""
    metadata_source: str = ""
    mark_scheme_source_pdf: str = ""
    crop_uncertain: bool = False
    question_crop_confidence: str = ""
    crop_debug_paths: list[str] = field(default_factory=list)
    question_crop_diagnostics: dict[str, Any] = field(default_factory=dict)
    topic_alternatives: list[str] = field(default_factory=list)
    topic_evidence_details: dict[str, Any] = field(default_factory=dict)
    examiner_report_evidence: dict[str, Any] = field(default_factory=dict)
    question_level_paper_family: str = ""
    question_level_topic: str = ""
    question_level_subtopic: str = ""
    part_level_topics: list[dict[str, Any]] = field(default_factory=list)
    markscheme_image: str = ""
    markscheme_pages: list[int] = field(default_factory=list)
    markscheme_question_number: str = ""
    markscheme_crop_confidence: str = ""
    markscheme_mapping_method: str = ""
    markscheme_table_detected: bool = False
    markscheme_table_header_detected: list[str] = field(default_factory=list)
    markscheme_nearby_anchors: list[str] = field(default_factory=list)
    markscheme_debug_paths: list[str] = field(default_factory=list)
    markscheme_table_header_ok: bool = False
    markscheme_continuation_rows_included: bool = False
    question_subparts: list[str] = field(default_factory=list)
    markscheme_subparts: list[str] = field(default_factory=list)
    question_marks_total: int | None = None
    markscheme_marks_total: int | None = None
    markscheme_mapping_status: str = ""
    markscheme_failure_reason: str = ""
    validation_status: str = ""
    validation_flags: list[str] = field(default_factory=list)
    scope_quality_status: str = ""
    text_source_profile: str = ""
    text_fidelity_status: str = ""
    text_fidelity_flags: list[str] = field(default_factory=list)
    question_text_role: str = QuestionTextRole.READABLE_TEXT
    question_text_trust: str = QuestionTextTrust.HIGH
    visual_required: bool = False
    visual_reason_flags: list[str] = field(default_factory=list)
    visual_curation_status: str = CurationStatus.READY
    text_only_status: str = CurationStatus.READY
    topic_trust_status: str = ""
    recovery_attempted: bool = False
    recovery_result: str = ""
    ocr_ran: bool = False
    ocr_engine: str = ""
    ocr_text: str = ""
    ocr_text_trust: str = QuestionTextTrust.UNUSABLE
    ocr_failure_reason: str = "disabled"
    ocr_text_role: str = QuestionTextRole.MISSING
    question_structure_detected: dict[str, Any] = field(default_factory=dict)
    mark_scheme_structure_detected: dict[str, Any] = field(default_factory=dict)
    question_total_detected: int | None = None
    mark_scheme_total_detected: int | None = None
    question_format_profile: str = ""
    reconciliation_changed_topic: bool = False
    reconciliation_reason: str = ""
    reconciliation_note: str = ""
    paper_repair_considered: bool = False
    paper_repair_changed_topic: bool = False
    paper_repair_reason: str = ""
    paper_repair_note: str = ""
    paper_repair_from_topic: str = ""
    paper_repair_to_topic: str = ""
    paper_repair_candidates: list[str] = field(default_factory=list)
    paper_repair_missing_topics: list[str] = field(default_factory=list)
    paper_repair_supporting_evidence: dict[str, Any] = field(default_factory=dict)
    paper_total_expected: int | None = None
    paper_total_detected: int | None = None
    paper_total_status: str = ""
    rescan_triggered: bool = False
    rescan_result: str = ""
    paper_total_before_rescan: int | None = None
    paper_total_after_rescan: int | None = None
    paper_total_focus_questions: list[str] = field(default_factory=list)
    paper_total_focus_pages: list[int] = field(default_factory=list)
    paper_total_focus_reason: str = ""

    @property
    def extraction(self) -> QuestionExtractionState:
        return QuestionExtractionState(
            source_pdf=self.source_pdf,
            paper_name=self.paper_name,
            question_number=self.question_number,
            full_question_label=self.full_question_label,
            combined_question_text=self.combined_question_text,
            body_text_raw=self.body_text_raw,
            body_text_normalized=self.body_text_normalized,
            math_lines=list(self.math_lines),
            diagram_text=list(self.diagram_text),
            extraction_quality_score=self.extraction_quality_score,
            extraction_quality_flags=list(self.extraction_quality_flags),
            part_texts=list(self.part_texts),
            page_numbers=list(self.page_numbers),
            question_subparts=list(self.question_subparts),
            question_marks_total=self.question_marks_total,
            question_structure_detected=dict(self.question_structure_detected),
            question_total_detected=self.question_total_detected,
            question_format_profile=self.question_format_profile,
            recovery_attempted=self.recovery_attempted,
            recovery_result=self.recovery_result,
            ocr_ran=self.ocr_ran,
            ocr_engine=self.ocr_engine,
            ocr_text=self.ocr_text,
            ocr_text_trust=self.ocr_text_trust,
            ocr_failure_reason=self.ocr_failure_reason,
            ocr_text_role=self.ocr_text_role,
        )

    @property
    def classification(self) -> QuestionClassificationState:
        return QuestionClassificationState(
            paper_family=self.paper_family,
            source_paper_family=self.source_paper_family,
            inferred_paper_family=self.inferred_paper_family,
            paper_family_confidence=self.paper_family_confidence,
            question_level_paper_family=self.question_level_paper_family,
            topic=self.topic,
            subtopic=self.subtopic,
            question_level_topic=self.question_level_topic,
            question_level_subtopic=self.question_level_subtopic,
            part_level_topics=list(self.part_level_topics),
            topic_confidence=self.topic_confidence,
            topic_evidence=self.topic_evidence,
            topic_evidence_details=dict(self.topic_evidence_details),
            examiner_report_evidence=dict(self.examiner_report_evidence),
            secondary_topics=list(self.secondary_topics),
            topic_uncertain=self.topic_uncertain,
            topic_alternatives=list(self.topic_alternatives),
            difficulty=self.difficulty,
            difficulty_confidence=self.difficulty_confidence,
            difficulty_evidence=self.difficulty_evidence,
            difficulty_uncertain=self.difficulty_uncertain,
            confidence=self.confidence,
        )

    @property
    def images(self) -> QuestionImageState:
        return QuestionImageState(
            screenshot_path=self.screenshot_path,
            crop_uncertain=self.crop_uncertain,
            question_crop_confidence=self.question_crop_confidence,
            crop_debug_paths=list(self.crop_debug_paths),
            question_crop_diagnostics=dict(self.question_crop_diagnostics),
        )

    @property
    def mark_scheme(self) -> MarkSchemeState:
        return MarkSchemeState(
            source_pdf=self.mark_scheme_source_pdf,
            answer_text=self.answer_text,
            image_path=self.markscheme_image,
            pages=list(self.markscheme_pages),
            question_number=self.markscheme_question_number,
            crop_confidence=self.markscheme_crop_confidence,
            mapping_method=self.markscheme_mapping_method,
            table_detected=self.markscheme_table_detected,
            table_header_detected=list(self.markscheme_table_header_detected),
            nearby_anchors=list(self.markscheme_nearby_anchors),
            debug_paths=list(self.markscheme_debug_paths),
            table_header_ok=self.markscheme_table_header_ok,
            continuation_rows_included=self.markscheme_continuation_rows_included,
            question_subparts=list(self.question_subparts),
            markscheme_subparts=list(self.markscheme_subparts),
            question_marks_total=self.question_marks_total,
            markscheme_marks_total=self.markscheme_marks_total,
            mapping_status=self.markscheme_mapping_status,
            failure_reason=self.markscheme_failure_reason,
            structure_detected=dict(self.mark_scheme_structure_detected),
            total_detected=self.mark_scheme_total_detected,
        )

    @property
    def validation(self) -> ValidationTrustState:
        return ValidationTrustState(
            review_flags=list(self.review_flags),
            validation_status=self.validation_status,
            validation_flags=list(self.validation_flags),
            scope_quality_status=self.scope_quality_status,
            text_source_profile=self.text_source_profile,
            text_fidelity_status=self.text_fidelity_status,
            text_fidelity_flags=list(self.text_fidelity_flags),
            question_text_role=self.question_text_role,
            question_text_trust=self.question_text_trust,
            visual_required=self.visual_required,
            visual_reason_flags=list(self.visual_reason_flags),
            visual_curation_status=self.visual_curation_status,
            text_only_status=self.text_only_status,
            topic_trust_status=self.topic_trust_status,
        )

    @property
    def paper_metadata(self) -> PaperMetadataState:
        return PaperMetadataState(
            source_paper_code=self.source_paper_code,
            syllabus_code=self.syllabus_code,
            session=self.session,
            year=self.year,
            document_type=self.document_type,
            component=self.component,
            document_key=self.document_key,
            metadata_source=self.metadata_source,
        )

    @property
    def paper_total(self) -> PaperTotalState:
        return PaperTotalState(
            expected=self.paper_total_expected,
            detected=self.paper_total_detected,
            status=self.paper_total_status,
            rescan_triggered=self.rescan_triggered,
            rescan_result=self.rescan_result,
            before_rescan=self.paper_total_before_rescan,
            after_rescan=self.paper_total_after_rescan,
            focus_questions=list(self.paper_total_focus_questions),
            focus_pages=list(self.paper_total_focus_pages),
            focus_reason=self.paper_total_focus_reason,
        )


def _clean_model_text(text: str) -> str:
    stripped_controls = "".join(char if ord(char) >= 32 or char in "\n\t\r" else " " for char in text)
    return " ".join(stripped_controls.replace("\u00a0", " ").split())
