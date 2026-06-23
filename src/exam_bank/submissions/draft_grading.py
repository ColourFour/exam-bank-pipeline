from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.auto_grade.constants import DEFAULT_REVIEWED_RUBRICS_PATH
from exam_bank.auto_grade.schemas import ReviewedRubric, load_reviewed_rubrics
from exam_bank.submissions.audit_log import AuditLog
from exam_bank.submissions.extraction import extract_submission_pdf
from exam_bank.submissions.grading_packets import (
    write_draft_grading_summary_csv,
    write_teacher_grading_review_packet,
)
from exam_bank.submissions.models import (
    DraftGradingResult,
    DraftGradingSummary,
    DraftQuestionResult,
    PHASE3_GRADING_MODE,
    PHASE3_STUDENT_FACING,
    PHASE3_TEACHER_REVIEW_REQUIRED,
    SubmissionExtractionResult,
    dataclass_to_json_dict,
)


DEFAULT_MARK_EVENTS_PATH = Path("output/json/question_bank.mark_events.v1.json")


@dataclass(frozen=True)
class ReviewedRubricContext:
    assignment_question_ids: list[str]
    approved_rubrics_by_question_id: dict[str, ReviewedRubric]
    rubric_warnings: list[str]
    advisory_mark_event_question_ids: set[str]


def build_submission_draft_grades(
    *,
    assignment_id: str,
    submission_output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
    reviewed_rubrics_path: Path | None = Path(DEFAULT_REVIEWED_RUBRICS_PATH),
    mark_events_path: Path | None = DEFAULT_MARK_EVENTS_PATH,
) -> dict[str, object]:
    _require_private_roots(submission_output_root, reports_root)
    assignment_output = submission_output_root / assignment_id
    manifest_path = assignment_output / "manifest.json"
    review_queue_path = assignment_output / "review" / "review_queue.json"
    grading_prep_path = assignment_output / "review" / "grading_prep.json"

    manifest = _load_json(manifest_path)
    review_queue = _load_json_list(review_queue_path)
    grading_prep = _load_json_list(grading_prep_path)
    assignment = manifest.get("assignment") if isinstance(manifest.get("assignment"), dict) else {}
    audit = AuditLog(assignment_output / "audit.jsonl", assignment_id)
    audit.write("draft_grading_started", status="started")

    context = _reviewed_rubric_context(
        assignment=assignment,
        reviewed_rubrics_path=reviewed_rubrics_path,
        mark_events_path=mark_events_path,
    )
    audit.write(
        "legacy_grading_salvage_audited",
        status="completed",
        reasons=context.rubric_warnings,
        reviewed_rubric_count=len(context.approved_rubrics_by_question_id),
        advisory_mark_event_question_count=len(context.advisory_mark_event_question_ids),
    )

    accepted_submission_ids = {
        str(submission.get("submission_id") or "")
        for submission in manifest.get("accepted_submissions", [])
        if isinstance(submission, dict)
    }
    grading_by_submission_id = {
        str(record.get("submission_id") or ""): record
        for record in grading_prep
        if isinstance(record, dict) and record.get("submission_id")
    }
    accepted_reviews = [
        record
        for record in review_queue
        if isinstance(record, dict)
        and str(record.get("submission_id") or "") in accepted_submission_ids
        and str(record.get("status") or "") != "excluded"
    ]

    created_at = datetime.now(timezone.utc)
    extraction_results: list[SubmissionExtractionResult] = []
    draft_results: list[DraftGradingResult] = []

    for review in accepted_reviews:
        student_id = str(review.get("student_id") or "")
        submission_id = str(review.get("submission_id") or "")
        audit.write("submission_extraction_started", student_id=student_id, submission_id=submission_id, status="started")
        extraction = extract_submission_pdf(
            assignment_id=assignment_id,
            student_id=student_id,
            submission_id=submission_id,
            stored_pdf_path=str(review.get("stored_pdf_path") or ""),
            created_at=created_at,
        )
        extraction_results.append(extraction)
        audit.write(
            "submission_extraction_finished",
            student_id=student_id,
            submission_id=submission_id,
            status=extraction.status,
            reasons=extraction.extraction_warnings,
            page_count=extraction.page_count,
            text_extractable=extraction.text_extractable,
        )

        grading = grading_by_submission_id.get(submission_id, {})
        draft = create_draft_grading_result(
            review_record=review,
            grading_prep_record=grading,
            extraction=extraction,
            rubric_context=context,
            created_at=created_at,
        )
        draft_results.append(draft)
        audit.write(
            "draft_grading_created",
            student_id=draft.student_id,
            submission_id=draft.submission_id,
            status=draft.status,
            reasons=draft.confidence_reasons,
            confidence=draft.confidence,
            draft_score=draft.draft_score,
        )
        if draft.draft_score is None:
            audit.write(
                "draft_grading_skipped",
                student_id=draft.student_id,
                submission_id=draft.submission_id,
                status="score_not_assigned",
                reasons=draft.confidence_reasons,
            )

    if not accepted_reviews:
        audit.write("draft_grading_skipped", status="no_accepted_phase2_submissions")

    summary = summarize_draft_grading_results(assignment_id=assignment_id, draft_results=draft_results, created_at=created_at)

    draft_dir = assignment_output / "draft_grading"
    extraction_results_path = draft_dir / "extraction_results.json"
    draft_results_path = draft_dir / "draft_grading_results.json"
    summary_path = draft_dir / "draft_grading_summary.json"
    packet_path = draft_dir / "teacher_grading_review_packet.md"
    summary_csv_path = reports_root / f"{assignment_id}_draft_grading_summary.csv"

    _write_json_records(extraction_results_path, extraction_results)
    _write_json_records(draft_results_path, draft_results)
    _write_json_record(summary_path, summary)
    write_draft_grading_summary_csv(path=summary_csv_path, extraction_results=extraction_results, draft_results=draft_results)
    audit.write(
        "draft_grading_summary_written",
        status="written",
        path=summary_path.as_posix(),
        csv_path=summary_csv_path.as_posix(),
    )
    write_teacher_grading_review_packet(
        path=packet_path,
        assignment=assignment,
        extraction_results=extraction_results,
        draft_results=draft_results,
        summary=summary,
    )
    audit.write("teacher_grading_review_packet_written", status="written", path=packet_path.as_posix())
    audit.write("draft_grading_finished", status="finished", reasons=context.rubric_warnings)

    return {
        "assignment_id": assignment_id,
        "extraction_results": extraction_results_path,
        "draft_grading_results": draft_results_path,
        "draft_grading_summary": summary_path,
        "teacher_grading_review_packet": packet_path,
        "draft_grading_summary_csv": summary_csv_path,
        "summary": summary,
        "extractions": extraction_results,
        "drafts": draft_results,
        "draft_scores_assigned": sum(1 for result in draft_results if result.draft_score is not None),
    }


def create_draft_grading_result(
    *,
    review_record: dict[str, Any],
    grading_prep_record: dict[str, Any],
    extraction: SubmissionExtractionResult,
    rubric_context: ReviewedRubricContext,
    created_at: datetime | None = None,
) -> DraftGradingResult:
    timestamp = created_at or datetime.now(timezone.utc)
    assignment_id = str(review_record.get("assignment_id") or extraction.assignment_id)
    student_id = str(review_record.get("student_id") or extraction.student_id)
    submission_id = str(review_record.get("submission_id") or extraction.submission_id)
    grading_result_id = str(grading_prep_record.get("grading_result_id") or f"{assignment_id}:{student_id}:{submission_id}:manual_placeholder")

    reasons: list[str] = []
    question_results: list[DraftQuestionResult] = []
    draft_max_score: float | None = None

    if extraction.status != "extracted" or extraction.text_extractable is not True:
        confidence = "none"
        status = "failed"
        reasons.extend(["native_text_unavailable", *extraction.extraction_warnings])
    else:
        question_ids = rubric_context.assignment_question_ids
        if not question_ids:
            confidence = "low"
            status = "needs_review"
            reasons.append("missing_assignment_question_mapping")
        else:
            missing_rubrics = [
                question_id
                for question_id in question_ids
                if question_id not in rubric_context.approved_rubrics_by_question_id
            ]
            advisory_only = bool(set(question_ids) & rubric_context.advisory_mark_event_question_ids) and bool(missing_rubrics)
            if missing_rubrics:
                confidence = "low"
                status = "needs_review"
                reasons.append("missing_reviewed_rubric_mapping")
                reasons.extend(f"missing_reviewed_rubric:{question_id}" for question_id in missing_rubrics)
                if advisory_only:
                    reasons.append("advisory_mark_events_not_scoring_contract")
            else:
                confidence = "medium"
                status = "needs_review"
                reasons.extend(["deterministic_student_answer_mapping_missing", "teacher_review_required"])
                rubric_totals = [
                    rubric_context.approved_rubrics_by_question_id[question_id].total_marks
                    for question_id in question_ids
                ]
                if all(total is not None for total in rubric_totals):
                    draft_max_score = float(sum(int(total or 0) for total in rubric_totals))

        question_results = _draft_question_results(
            question_ids=rubric_context.assignment_question_ids,
            rubrics=rubric_context.approved_rubrics_by_question_id,
            confidence=confidence,
            reasons=reasons,
            text_extractable=extraction.text_extractable,
        )

    return DraftGradingResult(
        draft_grading_id=f"{grading_result_id}:draft_auto",
        grading_result_id=grading_result_id,
        assignment_id=assignment_id,
        student_id=student_id,
        submission_id=submission_id,
        grading_mode=PHASE3_GRADING_MODE,
        status=status,
        draft_score=None,
        draft_max_score=draft_max_score,
        confidence=confidence,
        confidence_reasons=_dedupe(reasons),
        teacher_review_required=PHASE3_TEACHER_REVIEW_REQUIRED,
        student_facing=PHASE3_STUDENT_FACING,
        question_results=question_results,
        overall_notes=[
            "Phase 3 draft only; teacher review is required.",
            "No automated result is a final grade.",
        ],
        created_at=timestamp,
        updated_at=timestamp,
    )


def summarize_draft_grading_results(
    *,
    assignment_id: str,
    draft_results: list[DraftGradingResult],
    created_at: datetime | None = None,
) -> DraftGradingSummary:
    timestamp = created_at or datetime.now(timezone.utc)
    return DraftGradingSummary(
        assignment_id=assignment_id,
        submissions_attempted=len(draft_results),
        drafts_created=len(draft_results),
        failed_count=sum(1 for result in draft_results if result.status == "failed"),
        low_confidence_count=sum(1 for result in draft_results if result.confidence == "low"),
        medium_confidence_count=sum(1 for result in draft_results if result.confidence == "medium"),
        high_confidence_count=sum(1 for result in draft_results if result.confidence == "high"),
        teacher_review_required_count=sum(1 for result in draft_results if result.teacher_review_required is True),
        student_facing_count=sum(1 for result in draft_results if result.student_facing is True),
        created_at=timestamp,
    )


def _draft_question_results(
    *,
    question_ids: list[str],
    rubrics: dict[str, ReviewedRubric],
    confidence: str,
    reasons: list[str],
    text_extractable: bool,
) -> list[DraftQuestionResult]:
    results: list[DraftQuestionResult] = []
    for index, question_id in enumerate(question_ids, start=1):
        rubric = rubrics.get(question_id)
        evidence = ["native_pdf_text_preview_available"] if text_extractable else []
        question_confidence = "medium" if rubric and confidence == "medium" else confidence
        results.append(
            DraftQuestionResult(
                question_id=question_id,
                question_label=f"Question {index}",
                draft_score=None,
                draft_max_score=float(rubric.total_marks) if rubric and rubric.total_marks is not None else None,
                confidence=question_confidence,
                evidence=evidence,
                notes=_dedupe(reasons),
                review_required=True,
            )
        )
    return results


def _reviewed_rubric_context(
    *,
    assignment: dict[str, object],
    reviewed_rubrics_path: Path | None,
    mark_events_path: Path | None,
) -> ReviewedRubricContext:
    question_ids = [str(item) for item in assignment.get("source_question_ids", []) if str(item).strip()]
    rubrics: dict[str, ReviewedRubric] = {}
    warnings: list[str] = []
    rubric_path = Path(reviewed_rubrics_path) if reviewed_rubrics_path is not None else None
    if rubric_path is None:
        warnings.append("reviewed_rubrics_not_configured")
    elif not rubric_path.exists():
        warnings.append("reviewed_rubrics_file_missing")
    else:
        payload = _load_json(rubric_path)
        loaded, errors = load_reviewed_rubrics(payload)
        warnings.extend(errors)
        rubrics = {
            question_id: rubric
            for question_id, rubric in loaded.items()
            if rubric.is_approved
        }

    advisory_mark_event_question_ids = _mark_event_question_ids(Path(mark_events_path) if mark_events_path is not None else None)
    return ReviewedRubricContext(
        assignment_question_ids=question_ids,
        approved_rubrics_by_question_id=rubrics,
        rubric_warnings=_dedupe(warnings),
        advisory_mark_event_question_ids=advisory_mark_event_question_ids,
    )


def _mark_event_question_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = _load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return set()
    return {
        str(record.get("question_id") or "")
        for record in records
        if isinstance(record, dict) and str(record.get("question_id") or "").strip()
    }


def _require_private_roots(submission_output_root: Path, reports_root: Path) -> None:
    output_parts = submission_output_root.parts
    reports_parts = reports_root.parts
    if len(output_parts) < 2 or output_parts[-2:] != ("output", "submissions"):
        raise ValueError("submission_output_root must end with output/submissions")
    if len(reports_parts) < 2 or reports_parts[-2:] != ("reports", "submissions"):
        raise ValueError("reports_root must end with reports/submissions")


def _write_json_records(path: Path, records: list[object]) -> None:
    write_atomic_json([dataclass_to_json_dict(record) for record in records], path, sort_keys=True)


def _write_json_record(path: Path, record: object) -> None:
    write_atomic_json(dataclass_to_json_dict(record), path, sort_keys=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [record for record in payload if isinstance(record, dict)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
