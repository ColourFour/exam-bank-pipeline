from __future__ import annotations

import csv
from pathlib import Path

from exam_bank.submissions.models import DraftGradingResult, DraftGradingSummary, SubmissionExtractionResult


DRAFT_GRADING_CSV_FIELDS = [
    "assignment_id",
    "student_id",
    "submission_id",
    "stored_pdf_path",
    "extraction_status",
    "page_count",
    "text_extractable",
    "draft_grading_status",
    "grading_mode",
    "draft_score",
    "draft_max_score",
    "confidence",
    "teacher_review_required",
    "student_facing",
    "warnings",
    "confidence_reasons",
]


def write_draft_grading_summary_csv(
    *,
    path: Path,
    extraction_results: list[SubmissionExtractionResult],
    draft_results: list[DraftGradingResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extraction_by_submission = {item.submission_id: item for item in extraction_results}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DRAFT_GRADING_CSV_FIELDS)
        writer.writeheader()
        for result in draft_results:
            extraction = extraction_by_submission.get(result.submission_id)
            writer.writerow(
                {
                    "assignment_id": result.assignment_id,
                    "student_id": result.student_id,
                    "submission_id": result.submission_id,
                    "stored_pdf_path": extraction.stored_pdf_path if extraction else "",
                    "extraction_status": extraction.status if extraction else "",
                    "page_count": extraction.page_count if extraction else "",
                    "text_extractable": extraction.text_extractable if extraction else "",
                    "draft_grading_status": result.status,
                    "grading_mode": result.grading_mode,
                    "draft_score": "" if result.draft_score is None else result.draft_score,
                    "draft_max_score": "" if result.draft_max_score is None else result.draft_max_score,
                    "confidence": result.confidence,
                    "teacher_review_required": result.teacher_review_required,
                    "student_facing": result.student_facing,
                    "warnings": ";".join(extraction.extraction_warnings if extraction else []),
                    "confidence_reasons": ";".join(result.confidence_reasons),
                }
            )


def write_teacher_grading_review_packet(
    *,
    path: Path,
    assignment: dict[str, object],
    extraction_results: list[SubmissionExtractionResult],
    draft_results: list[DraftGradingResult],
    summary: DraftGradingSummary,
) -> str:
    text = render_teacher_grading_review_packet(
        assignment=assignment,
        extraction_results=extraction_results,
        draft_results=draft_results,
        summary=summary,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def render_teacher_grading_review_packet(
    *,
    assignment: dict[str, object],
    extraction_results: list[SubmissionExtractionResult],
    draft_results: list[DraftGradingResult],
    summary: DraftGradingSummary,
) -> str:
    title = str(assignment.get("title") or "")
    assignment_id = summary.assignment_id
    extraction_by_submission = {item.submission_id: item for item in extraction_results}
    lines = [
        "# Teacher Draft Grading Review Packet",
        "",
        "Teacher-facing draft only. No student-facing feedback or final grades are produced by Phase 3.",
        "",
        "## Assignment",
        "",
        f"- Assignment ID: `{assignment_id}`",
        f"- Title: `{title or 'unknown'}`",
        f"- Grading mode: `{summary.grading_mode}`",
        f"- Student facing: `{str(summary.student_facing).lower()}`",
        f"- Teacher review required: `{str(summary.teacher_review_required).lower()}`",
        "",
        "## Summary",
        "",
        f"- Submissions attempted: {summary.submissions_attempted}",
        f"- Draft records created: {summary.drafts_created}",
        f"- Failed draft records: {summary.failed_count}",
        f"- Low confidence: {summary.low_confidence_count}",
        f"- Medium confidence: {summary.medium_confidence_count}",
        f"- High confidence: {summary.high_confidence_count}",
        f"- Teacher review required: {summary.teacher_review_required_count}",
        f"- Student-facing records: {summary.student_facing_count}",
        "",
        "## Submission Review Items",
        "",
    ]
    if not draft_results:
        lines.append("- none")
    for result in draft_results:
        extraction = extraction_by_submission.get(result.submission_id)
        warnings = extraction.extraction_warnings if extraction else []
        action = _teacher_action(result, extraction)
        lines.extend(
            [
                f"### `{result.student_id}` / `{result.submission_id}`",
                "",
                f"- Stored PDF path: `{extraction.stored_pdf_path if extraction else ''}`",
                f"- Extraction status: `{extraction.status if extraction else 'missing'}`",
                f"- Page count: `{extraction.page_count if extraction else ''}`",
                f"- Text extractable: `{str(extraction.text_extractable).lower() if extraction else 'false'}`",
                f"- Draft grading status: `{result.status}`",
                f"- Confidence: `{result.confidence}`",
                f"- Draft score: `{'' if result.draft_score is None else result.draft_score}`",
                f"- Draft max score: `{'' if result.draft_max_score is None else result.draft_max_score}`",
                f"- Warnings: `{'; '.join(warnings) if warnings else 'none'}`",
                f"- Reasons: `{'; '.join(result.confidence_reasons) if result.confidence_reasons else 'none'}`",
                f"- Teacher action needed: {action}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _teacher_action(result: DraftGradingResult, extraction: SubmissionExtractionResult | None) -> str:
    if extraction is None or extraction.status != "extracted":
        return "Inspect the PDF manually; native text was unavailable or incomplete."
    if result.draft_score is None:
        return "Review the submission manually; Phase 3 did not assign a draft score."
    return "Review every draft mark before any manual grade is recorded."
