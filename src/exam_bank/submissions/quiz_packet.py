from __future__ import annotations

import csv
import html
import json
import os
import re
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

from exam_bank.mupdf_tools import quiet_mupdf
from exam_bank.submissions.extraction import extract_submission_pdf
from exam_bank.submissions.ingest import ingest_assignment_submissions
from exam_bank.submissions.review_queue import build_submission_review_queue

quiet_mupdf(fitz)


DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_QUESTION_BANK_PATH = Path("output/json/question_bank.json")
CONFIDENT_TEXT_MATCH = 0.72
PARTIAL_TEXT_MATCH = 0.5
CONFIDENT_MARK_SCHEME_SCORE = 0.8


@dataclass(frozen=True)
class ParsedScan:
    source_file: str
    display_name: str
    student_id: str
    warnings: list[str]


@dataclass(frozen=True)
class AssignmentQuestion:
    assignment_question_index: int
    question_number: str
    paper_code: str
    page_numbers: list[int]
    text: str


def run_quiz_packet(
    *,
    quiz_dir: Path,
    course_id: str,
    assignment_pdf: Path | None = None,
    assignment_id: str | None = None,
    class_id: str = "local_class",
    timezone_name: str | None = None,
    output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
    open_report: bool = False,
    no_grade: bool = False,
    force: bool = False,
) -> dict[str, object]:
    quiz_dir = quiz_dir.expanduser()
    assignment_pdf_path = _resolve_assignment_pdf(quiz_dir, assignment_pdf)
    scans_dir = quiz_dir / "scans"
    if not scans_dir.is_dir():
        raise FileNotFoundError(f"Missing scans directory: {scans_dir}")

    generated_warnings: list[str] = []
    assignment_path, assignment_payload, assignment_created = ensure_assignment_json(
        quiz_dir=quiz_dir,
        assignment_pdf=assignment_pdf_path,
        assignment_id=assignment_id,
        course_id=course_id,
        class_id=class_id,
        timezone_name=timezone_name or os.environ.get("TZ") or DEFAULT_TIMEZONE,
        force=force,
    )
    actual_assignment_id = str(assignment_payload["assignment_id"])
    actual_class_id = str(assignment_payload["class_id"])
    roster_path, roster_result = ensure_roster_csv(
        quiz_dir=quiz_dir,
        scans_dir=scans_dir,
        assignment_id=actual_assignment_id,
        class_id=actual_class_id,
    )
    generated_warnings.extend(roster_result["warnings"])

    ingest_result = ingest_assignment_submissions(
        assignment_path=assignment_path,
        roster_path=roster_path,
        submissions_dir=scans_dir,
        output_root=output_root,
        reports_root=reports_root,
    )
    review_result = build_submission_review_queue(
        assignment_id=actual_assignment_id,
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    matching_result = match_assignment_pdf(
        assignment_id=actual_assignment_id,
        assignment_pdf=assignment_pdf_path,
        output_root=output_root,
        question_bank_path=DEFAULT_QUESTION_BANK_PATH,
    )
    generated_warnings.extend(matching_result["warnings"])

    if no_grade:
        grading_result = write_no_grade_artifact(
            assignment_id=actual_assignment_id,
            output_root=output_root,
            reports_root=reports_root,
        )
    else:
        grading_result = write_packet_draft_grading(
            assignment_id=actual_assignment_id,
            output_root=output_root,
            reports_root=reports_root,
            manifest_path=Path(ingest_result["manifest"]),
            matched_questions=matching_result["question_records"],
        )
    generated_warnings.extend(grading_result["warnings"])

    teacher_report_path = write_teacher_report(
        assignment_id=actual_assignment_id,
        quiz_dir=quiz_dir,
        assignment_path=assignment_path,
        roster_path=roster_path,
        output_root=output_root,
        reports_root=reports_root,
        ingest_result=ingest_result,
        review_result=review_result,
        matching_result=matching_result,
        grading_result=grading_result,
        warnings=generated_warnings,
    )
    update_manifest_quiz_packet_section(
        manifest_path=Path(ingest_result["manifest"]),
        assignment_path=assignment_path,
        roster_path=roster_path,
        teacher_report_path=teacher_report_path,
        matching_result=matching_result,
        grading_result=grading_result,
    )

    report_opened = False
    if open_report:
        report_opened = _open_path(teacher_report_path)

    return {
        "assignment_id": actual_assignment_id,
        "assignment_json": assignment_path,
        "assignment_created": assignment_created,
        "roster_csv": roster_path,
        "roster_rows": roster_result["row_count"],
        "manifest": ingest_result["manifest"],
        "audit_log": ingest_result["audit_log"],
        "completion_report": ingest_result["completion_report"],
        "review_queue": review_result["review_queue"],
        "matched_questions": matching_result["matched_questions"],
        "matched_mark_scheme": matching_result["matched_mark_scheme"],
        "draft_grading": grading_result["draft_grading"],
        "draft_grading_summary": grading_result["draft_grading_summary"],
        "teacher_report": teacher_report_path,
        "accepted_count": len(ingest_result["accepted"]),
        "rejected_count": len(ingest_result["rejected"]),
        "missing_count": sum(1 for row in ingest_result["completion_rows"] if row.status == "missing"),
        "automatic_grading_status": grading_result["automatic_grading_status"],
        "email_sent": False,
        "warnings": _dedupe(generated_warnings),
        "report_opened": report_opened,
    }


def ensure_assignment_json(
    *,
    quiz_dir: Path,
    assignment_pdf: Path,
    assignment_id: str | None,
    course_id: str,
    class_id: str,
    timezone_name: str,
    force: bool,
) -> tuple[Path, dict[str, Any], bool]:
    assignment_path = quiz_dir / "assignment.json"
    if assignment_path.exists() and not force:
        payload = json.loads(assignment_path.read_text(encoding="utf-8"))
        return assignment_path, payload, False

    actual_assignment_id = assignment_id or quiz_dir.name
    payload: dict[str, Any] = {
        "assignment_id": actual_assignment_id,
        "course_id": course_id,
        "title": _title_from_assignment_id(actual_assignment_id),
        "class_id": class_id,
        "due_at": None,
        "timezone": timezone_name,
        "accepted_file_types": ["pdf"],
        "max_files_per_student": 1,
        "max_file_size_mb": 25,
        "allow_late": True,
        "source_question_ids": [],
        "assignment_pdf": _relative_path(assignment_pdf, quiz_dir),
        "generated_by": "quiz-packet",
        "generated_at": _utc_now(),
    }
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return assignment_path, payload, True


def ensure_roster_csv(
    *,
    quiz_dir: Path,
    scans_dir: Path,
    assignment_id: str,
    class_id: str,
) -> tuple[Path, dict[str, object]]:
    roster_path = quiz_dir / "roster.csv"
    scan_paths = sorted(path for path in scans_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    parsed_scans = parse_scan_filenames(scan_paths, assignment_id=assignment_id)
    warnings = [warning for scan in parsed_scans for warning in scan.warnings]

    existing_rows: list[dict[str, str]] = []
    existing_fields: list[str] = []
    if roster_path.exists():
        with roster_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = [{key: value for key, value in row.items() if key is not None} for row in reader]

    required_fields = ["student_id", "class_id", "display_name", "email", "submitted", "source_file"]
    fieldnames = list(existing_fields) if existing_fields else list(required_fields)
    for field in required_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    rows = [_normalize_roster_row(row, fieldnames, class_id=class_id) for row in existing_rows]
    rows_by_source = {
        row.get("source_file", ""): row
        for row in rows
        if row.get("source_file", "")
    }
    rows_by_student = {
        row.get("student_id", ""): row
        for row in rows
        if row.get("student_id", "")
    }
    seen_student_ids: set[str] = set()
    seen_source_files = {scan.source_file for scan in parsed_scans}

    for scan in parsed_scans:
        row = rows_by_source.get(scan.source_file) or rows_by_student.get(scan.student_id)
        if row is None:
            row = {field: "" for field in fieldnames}
            rows.append(row)
        row["student_id"] = row.get("student_id") or scan.student_id
        row["class_id"] = row.get("class_id") or class_id
        row["display_name"] = row.get("display_name") or scan.display_name
        row["email"] = row.get("email", "")
        row["submitted"] = "true"
        row["source_file"] = scan.source_file
        rows_by_source[scan.source_file] = row
        rows_by_student[row["student_id"]] = row
        seen_student_ids.add(row["student_id"])

    for row in rows:
        if row.get("source_file", "") not in seen_source_files and row.get("student_id", "") not in seen_student_ids:
            row["submitted"] = "false"

    roster_path.parent.mkdir(parents=True, exist_ok=True)
    with roster_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return roster_path, {"row_count": len(rows), "warnings": _dedupe(warnings)}


def parse_scan_filenames(scan_paths: list[Path], *, assignment_id: str) -> list[ParsedScan]:
    parsed: list[tuple[Path, str, str, list[str]]] = []
    for path in scan_paths:
        display_name, warnings = _display_name_from_scan(path.stem, assignment_id=assignment_id)
        base_student_id = _safe_student_id(display_name) or "student"
        parsed.append((path, display_name, base_student_id, warnings))

    base_counts: dict[str, int] = {}
    for _, _, base_student_id, _ in parsed:
        base_counts[base_student_id] = base_counts.get(base_student_id, 0) + 1

    assigned_counts: dict[str, int] = {}
    result: list[ParsedScan] = []
    for path, display_name, base_student_id, warnings in parsed:
        assigned_counts[base_student_id] = assigned_counts.get(base_student_id, 0) + 1
        student_id = base_student_id
        if base_counts[base_student_id] > 1:
            ordinal = assigned_counts[base_student_id]
            student_id = base_student_id if ordinal == 1 else f"{base_student_id}_{ordinal}"
            warnings = [
                *warnings,
                f"duplicate_student_id:{base_student_id}:{path.name}:assigned:{student_id}",
            ]
        result.append(
            ParsedScan(
                source_file=path.name,
                display_name=display_name,
                student_id=student_id,
                warnings=warnings,
            )
        )
    return result


def match_assignment_pdf(
    *,
    assignment_id: str,
    assignment_pdf: Path,
    output_root: Path,
    question_bank_path: Path,
) -> dict[str, object]:
    assignment_output = output_root / assignment_id
    matching_dir = assignment_output / "assignment_matching"
    matching_dir.mkdir(parents=True, exist_ok=True)
    matched_questions_path = matching_dir / "matched_questions.json"
    matched_mark_scheme_path = matching_dir / "matched_mark_scheme.json"

    warnings: list[str] = []
    extracted = extract_assignment_pdf_questions(assignment_pdf)
    warnings.extend(extracted["warnings"])
    questions = extracted["questions"]

    question_bank = load_question_bank(question_bank_path)
    if not question_bank:
        warnings.append(f"question_bank_missing_or_empty:{question_bank_path.as_posix()}")

    records: list[dict[str, object]] = []
    for question in questions:
        match = match_assignment_question(question, question_bank)
        records.append(match)

    if not records and extracted["text_status"] != "extracted":
        records.append(_blocked_unreadable_record())

    matched_questions_payload = {
        "assignment_id": assignment_id,
        "assignment_pdf": assignment_pdf.as_posix(),
        "generated_at": _utc_now(),
        "assignment_text_status": extracted["text_status"],
        "detected_question_count": len(records),
        "question_bank_path": question_bank_path.as_posix(),
        "records": records,
    }
    matched_mark_scheme_payload = {
        "assignment_id": assignment_id,
        "generated_at": _utc_now(),
        "records": [
            {
                "assignment_question_index": record["assignment_question_index"],
                "matched_question_id": record["matched_question_id"],
                "matched_mark_scheme_id": record["matched_mark_scheme_id"],
                "mark_scheme_status": record["mark_scheme_status"],
                "teacher_review_required": record["teacher_review_required"],
                "evidence": record["evidence"],
            }
            for record in records
        ],
    }
    matched_questions_path.write_text(json.dumps(matched_questions_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matched_mark_scheme_path.write_text(json.dumps(matched_mark_scheme_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    match_statuses = {str(record["match_status"]) for record in records}
    mark_statuses = {str(record["mark_scheme_status"]) for record in records}
    if match_statuses - {"matched_confident"}:
        warnings.append("weak_or_missing_assignment_question_matches")
    if mark_statuses - {"matched_confident"}:
        warnings.append("weak_or_missing_mark_scheme_matches")

    return {
        "matched_questions": matched_questions_path,
        "matched_mark_scheme": matched_mark_scheme_path,
        "question_records": records,
        "assignment_text_status": extracted["text_status"],
        "warnings": _dedupe(warnings),
    }


def extract_assignment_pdf_questions(path: Path) -> dict[str, object]:
    warnings: list[str] = []
    grouped: list[dict[str, object]] = []
    all_text: list[str] = []
    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")
                all_text.append(raw_text)
                paper_code = _paper_code_from_text(raw_text)
                question_number = _question_number_from_page_text(raw_text)
                page_clean_text = _clean_assignment_page_text(raw_text)
                if not page_clean_text:
                    continue
                if (
                    grouped
                    and paper_code
                    and grouped[-1].get("paper_code") == paper_code
                    and (not question_number or grouped[-1].get("question_number") == question_number)
                ):
                    grouped[-1]["page_numbers"].append(page_index)
                    grouped[-1]["text"] = f"{grouped[-1]['text']}\n{page_clean_text}"
                    continue
                grouped.append(
                    {
                        "question_number": question_number,
                        "paper_code": paper_code,
                        "page_numbers": [page_index],
                        "text": page_clean_text,
                    }
                )
    except Exception as exc:
        return {
            "text_status": "blocked_assignment_text_unreadable",
            "questions": [],
            "warnings": [f"assignment_pdf_text_extraction_failed:{exc.__class__.__name__}"],
        }

    normalized_all_text = _normalize_text(" ".join(all_text))
    if not normalized_all_text:
        return {
            "text_status": "blocked_assignment_text_unreadable",
            "questions": [],
            "warnings": ["assignment_pdf_native_text_empty"],
        }

    questions = [
        AssignmentQuestion(
            assignment_question_index=index,
            question_number=str(group.get("question_number") or ""),
            paper_code=str(group.get("paper_code") or ""),
            page_numbers=[int(item) for item in group.get("page_numbers", [])],
            text=str(group.get("text") or ""),
        )
        for index, group in enumerate(grouped, start=1)
    ]
    if not questions:
        warnings.append("assignment_questions_not_detected")
    return {"text_status": "extracted", "questions": questions, "warnings": warnings}


def load_question_bank(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def match_assignment_question(question: AssignmentQuestion, question_bank: list[dict[str, Any]]) -> dict[str, object]:
    strong_candidates = _strong_identifier_candidates(question, question_bank)
    evidence: list[str] = [
        f"assignment_pages:{','.join(str(page) for page in question.page_numbers)}",
    ]
    if question.paper_code:
        evidence.append(f"assignment_paper_code:{question.paper_code}")
    if question.question_number:
        evidence.append(f"assignment_question_number:{question.question_number}")

    if len(strong_candidates) == 1:
        record = strong_candidates[0]
        similarity = _question_text_similarity(question.text, str(record.get("question_text") or ""))
        evidence.extend(
            [
                "strong_identifier_match",
                f"text_similarity:{similarity:.3f}",
            ]
        )
        return _matched_record(
            question=question,
            record=record,
            status="matched_confident",
            confidence=max(0.92, min(0.99, 0.9 + similarity * 0.1)),
            evidence=evidence,
        )
    if len(strong_candidates) > 1:
        evidence.append(f"ambiguous_strong_identifier_candidates:{len(strong_candidates)}")
        return _unmatched_record(
            question=question,
            status="ambiguous_match",
            confidence=0.0,
            evidence=evidence,
        )

    text_candidates = _text_candidates(question, question_bank)
    if text_candidates:
        best_score, best_record = text_candidates[0]
        second_score = text_candidates[1][0] if len(text_candidates) > 1 else 0.0
        evidence.extend(
            [
                f"text_similarity:{best_score:.3f}",
                f"second_text_similarity:{second_score:.3f}",
            ]
        )
        if best_score >= CONFIDENT_TEXT_MATCH and best_score - second_score >= 0.08:
            evidence.append("near_exact_text_match")
            return _matched_record(
                question=question,
                record=best_record,
                status="matched_confident",
                confidence=min(0.9, best_score),
                evidence=evidence,
            )
        if best_score >= PARTIAL_TEXT_MATCH:
            evidence.append("partial_text_match_below_confident_threshold")
            return _matched_record(
                question=question,
                record=best_record,
                status="matched_partial",
                confidence=best_score,
                evidence=evidence,
            )

    return _unmatched_record(
        question=question,
        status="no_match",
        confidence=0.0,
        evidence=[*evidence, "no_confident_identifier_or_text_match"],
    )


def write_packet_draft_grading(
    *,
    assignment_id: str,
    output_root: Path,
    reports_root: Path,
    manifest_path: Path,
    matched_questions: list[dict[str, object]],
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_submissions = [
        submission
        for submission in manifest.get("accepted_submissions", [])
        if isinstance(submission, dict)
    ]
    assignment_output = output_root / assignment_id
    draft_dir = assignment_output / "draft_grading"
    draft_dir.mkdir(parents=True, exist_ok=True)
    draft_path = draft_dir / "quiz_packet_draft_grading.json"
    summary_path = draft_dir / "quiz_packet_draft_grading_summary.json"
    summary_csv_path = reports_root / f"{assignment_id}_quiz_packet_draft_grading_summary.csv"

    safe_questions = [
        record
        for record in matched_questions
        if record.get("match_status") == "matched_confident"
        and record.get("mark_scheme_status") == "matched_confident"
        and not record.get("structural_uncertainty")
    ]
    blocked_questions = [
        record
        for record in matched_questions
        if record not in safe_questions
    ]
    automatic_grading_status = _automatic_grading_status(
        total_questions=len(matched_questions),
        safe_questions=len(safe_questions),
        accepted_submissions=len(accepted_submissions),
    )

    created_at = _utc_now()
    records: list[dict[str, object]] = []
    extraction_records: list[dict[str, object]] = []
    warnings: list[str] = []
    for submission in accepted_submissions:
        extraction = extract_submission_pdf(
            assignment_id=assignment_id,
            student_id=str(submission.get("student_id") or ""),
            submission_id=str(submission.get("submission_id") or ""),
            stored_pdf_path=str(submission.get("stored_pdf_path") or ""),
        )
        extraction_payload = {
            "student_id": extraction.student_id,
            "submission_id": extraction.submission_id,
            "status": extraction.status,
            "text_extractable": extraction.text_extractable,
            "page_count": extraction.page_count,
            "warnings": extraction.extraction_warnings,
        }
        extraction_records.append(extraction_payload)
        if extraction.status != "extracted" or not extraction.text_extractable:
            warnings.append(f"submission_text_unavailable:{extraction.student_id}")
        reason = _draft_block_reason(blocked_questions=blocked_questions, extraction=extraction_payload)
        records.append(
            {
                "student_id": str(submission.get("student_id") or ""),
                "submission_id": str(submission.get("submission_id") or ""),
                "source_filename": str(submission.get("source_filename") or ""),
                "status": "blocked_needs_teacher_review",
                "score": None,
                "max_score": None,
                "student_facing": False,
                "teacher_review_required": True,
                "reason": reason,
                "safe_assignment_question_count": len(safe_questions),
                "blocked_assignment_question_count": len(blocked_questions),
                "extraction": extraction_payload,
                "created_at": created_at,
            }
        )

    if not accepted_submissions:
        warnings.append("no_accepted_submissions_for_draft_grading")

    summary = {
        "assignment_id": assignment_id,
        "generated_at": created_at,
        "automatic_grading_status": automatic_grading_status,
        "accepted_submission_count": len(accepted_submissions),
        "draft_record_count": len(records),
        "safe_assignment_question_count": len(safe_questions),
        "blocked_assignment_question_count": len(blocked_questions),
        "scores_assigned": 0,
        "student_facing_count": 0,
        "teacher_review_required_count": len(records),
        "reason": "No scores assigned. Quiz-packet grading is draft-only and fail-closed.",
    }
    draft_path.write_text(
        json.dumps({"assignment_id": assignment_id, "records": records, "extractions": extraction_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_draft_summary_csv(summary_csv_path, records)
    return {
        "draft_grading": draft_path,
        "draft_grading_summary": summary_path,
        "draft_grading_summary_csv": summary_csv_path,
        "automatic_grading_status": automatic_grading_status,
        "warnings": _dedupe(warnings),
        "summary": summary,
    }


def write_no_grade_artifact(*, assignment_id: str, output_root: Path, reports_root: Path) -> dict[str, object]:
    draft_dir = output_root / assignment_id / "draft_grading"
    draft_dir.mkdir(parents=True, exist_ok=True)
    draft_path = draft_dir / "quiz_packet_draft_grading.json"
    summary_path = draft_dir / "quiz_packet_draft_grading_summary.json"
    summary_csv_path = reports_root / f"{assignment_id}_quiz_packet_draft_grading_summary.csv"
    summary = {
        "assignment_id": assignment_id,
        "generated_at": _utc_now(),
        "automatic_grading_status": "disabled",
        "scores_assigned": 0,
        "student_facing_count": 0,
        "teacher_review_required_count": 0,
        "reason": "--no-grade was passed.",
    }
    draft_path.write_text(json.dumps({"assignment_id": assignment_id, "records": []}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_draft_summary_csv(summary_csv_path, [])
    return {
        "draft_grading": draft_path,
        "draft_grading_summary": summary_path,
        "draft_grading_summary_csv": summary_csv_path,
        "automatic_grading_status": "disabled",
        "warnings": [],
        "summary": summary,
    }


def write_teacher_report(
    *,
    assignment_id: str,
    quiz_dir: Path,
    assignment_path: Path,
    roster_path: Path,
    output_root: Path,
    reports_root: Path,
    ingest_result: dict[str, object],
    review_result: dict[str, object],
    matching_result: dict[str, object],
    grading_result: dict[str, object],
    warnings: list[str],
) -> Path:
    report_path = reports_root / f"{assignment_id}_teacher_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    completion_rows = list(ingest_result["completion_rows"])
    accepted = list(ingest_result["accepted"])
    rejected = list(ingest_result["rejected"])
    matched_questions = list(matching_result["question_records"])
    grading_summary = grading_result.get("summary", {})

    body = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        f"<title>{_e(assignment_id)} Teacher Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;line-height:1.45;margin:32px;color:#202124}",
        "h1,h2{line-height:1.15} table{border-collapse:collapse;width:100%;margin:12px 0 24px}",
        "th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left;vertical-align:top}",
        "th{background:#f6f8fa} code{background:#f6f8fa;padding:1px 4px;border-radius:3px}",
        ".warn{background:#fff8c5;border:1px solid #f0d060;padding:10px 12px;margin:12px 0}",
        ".ok{background:#dafbe1;border:1px solid #8ddb8c;padding:10px 12px;margin:12px 0}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{_e(assignment_id)} Teacher Report</h1>",
        "<h2>Assignment Summary</h2>",
        "<table>",
        _row("Quiz folder", quiz_dir.as_posix()),
        _row("Assignment JSON", _link(report_path, assignment_path)),
        _row("Roster CSV", _link(report_path, roster_path)),
        _row("Manifest", _link(report_path, Path(ingest_result["manifest"]))),
        _row("Audit log", _link(report_path, Path(ingest_result["audit_log"]))),
        "</table>",
        "<h2>Roster Summary</h2>",
        "<table>",
        _row("Roster rows", str(len(completion_rows))),
        _row("Submitted", str(sum(1 for row in completion_rows if row.status in {'submitted', 'late'}))),
        _row("Missing", str(sum(1 for row in completion_rows if row.status == 'missing'))),
        _row("Rejected", str(len(rejected))),
        _row("Completion CSV", _link(report_path, Path(ingest_result["completion_report"]))),
        _row("Review queue", _link(report_path, Path(review_result["review_queue"]))),
        "</table>",
        "<h2>Accepted PDFs</h2>",
        _submission_table(report_path, accepted),
        "<h2>Rejected PDFs</h2>",
        _submission_table(report_path, rejected),
        "<h2>Assignment Matching</h2>",
        "<table>",
        _row("Matched questions JSON", _link(report_path, Path(matching_result["matched_questions"]))),
        _row("Matched mark scheme JSON", _link(report_path, Path(matching_result["matched_mark_scheme"]))),
        _row("Assignment text status", str(matching_result["assignment_text_status"])),
        "</table>",
        _matching_table(matched_questions),
        "<h2>Draft Grading Status</h2>",
        "<table>",
        _row("Automatic grading status", str(grading_result["automatic_grading_status"])),
        _row("Draft grading JSON", _link(report_path, Path(grading_result["draft_grading"]))),
        _row("Draft grading summary", _link(report_path, Path(grading_result["draft_grading_summary"]))),
        _row("Draft grading CSV", _link(report_path, Path(grading_result["draft_grading_summary_csv"]))),
        _row("Scores assigned", str(grading_summary.get("scores_assigned", 0)) if isinstance(grading_summary, dict) else "0"),
        _row("Student-facing feedback", "0"),
        _row("Email sent", "No"),
        "</table>",
        "<h2>Warnings</h2>",
        _warnings_block(warnings),
        "<h2>Next Step Checklist</h2>",
        "<ul>",
        "<li>Review weak question or mark scheme matches before relying on any draft grading artifact.</li>",
        "<li>Open accepted PDFs from this report and confirm the roster names are correct.</li>",
        "<li>Edit roster.csv if a filename was ambiguous or an email should be added later.</li>",
        "<li>Keep all draft feedback teacher-only until a teacher approves final marks.</li>",
        "</ul>",
        "</body>",
        "</html>",
    ]
    report_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return report_path


def update_manifest_quiz_packet_section(
    *,
    manifest_path: Path,
    assignment_path: Path,
    roster_path: Path,
    teacher_report_path: Path,
    matching_result: dict[str, object],
    grading_result: dict[str, object],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quiz_packet"] = {
        "assignment_json": assignment_path.as_posix(),
        "roster_csv": roster_path.as_posix(),
        "teacher_report": teacher_report_path.as_posix(),
        "matched_questions": Path(matching_result["matched_questions"]).as_posix(),
        "matched_mark_scheme": Path(matching_result["matched_mark_scheme"]).as_posix(),
        "draft_grading": Path(grading_result["draft_grading"]).as_posix(),
        "draft_grading_summary": Path(grading_result["draft_grading_summary"]).as_posix(),
        "automatic_grading_status": grading_result["automatic_grading_status"],
        "email_sent": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_assignment_pdf(quiz_dir: Path, assignment_pdf: Path | None) -> Path:
    path = assignment_pdf.expanduser() if assignment_pdf is not None else quiz_dir / "assignment.pdf"
    if not path.exists():
        raise FileNotFoundError(f"Missing assignment PDF: {path}")
    return path


def _title_from_assignment_id(assignment_id: str) -> str:
    return " ".join(part.upper() if re.fullmatch(r"[a-z]\d+", part) else part.capitalize() for part in re.split(r"[_-]+", assignment_id) if part)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _display_name_from_scan(stem: str, *, assignment_id: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    name = stem
    suffixes = [
        f"_{assignment_id}",
        f"-{assignment_id}",
        f" {assignment_id}",
        f"_{assignment_id.replace('-', '_')}",
    ]
    for suffix in suffixes:
        if name.lower().endswith(suffix.lower()):
            name = name[: -len(suffix)]
            break
    if name == stem and "_" in stem:
        warnings.append(f"assignment_suffix_not_detected:{stem}.pdf")
        name = stem.split("_", 1)[0]
    display = re.sub(r"[_-]+", " ", name).strip()
    display = " ".join(part.capitalize() if part.islower() else part for part in display.split())
    if not display:
        display = "Student"
        warnings.append(f"empty_student_name:{stem}.pdf")
    return display, warnings


def _safe_student_id(display_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", display_name.strip().lower()).strip("_")


def _normalize_roster_row(row: dict[str, str], fieldnames: list[str], *, class_id: str) -> dict[str, str]:
    normalized = {field: str(row.get(field, "") or "") for field in fieldnames}
    normalized["class_id"] = normalized.get("class_id") or class_id
    normalized["email"] = normalized.get("email", "")
    normalized["submitted"] = normalized.get("submitted", "false") or "false"
    normalized["source_file"] = normalized.get("source_file", "")
    return normalized


def _paper_code_from_text(text: str) -> str:
    match = re.search(r"9709/\d{2}/[A-Z]/[A-Z]/\d{2}", text)
    return match.group(0) if match else ""


def _question_number_from_page_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[1:12]:
        if re.fullmatch(r"9709/\d{2}/[A-Z]/[A-Z]/\d{2}", line) or line.startswith("© UCLES") or line == "[Turn over":
            continue
        if re.fullmatch(r"\d{1,2}", line):
            return line
        if re.search(r"[A-Za-z]", line) or re.match(r"^\([a-z]\)", line, flags=re.IGNORECASE):
            break
    return ""


def _clean_assignment_page_text(text: str) -> str:
    cleaned: list[str] = []
    for index, line in enumerate(line.strip() for line in text.splitlines() if line.strip()):
        if index == 0 and re.fullmatch(r"\d{1,2}", line):
            continue
        if re.fullmatch(r"9709/\d{2}/[A-Z]/[A-Z]/\d{2}", line):
            continue
        if line.startswith("© UCLES") or line == "[Turn over":
            continue
        if set(line) <= {"."}:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _strong_identifier_candidates(question: AssignmentQuestion, question_bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not question.paper_code or not question.question_number:
        return []
    paper_id, source_stem = _paper_identifiers(question.paper_code)
    question_number = str(int(question.question_number)) if question.question_number.isdigit() else question.question_number
    candidates: list[dict[str, Any]] = []
    for record in question_bank:
        record_question_number = str(record.get("question_number") or "")
        if record_question_number != question_number:
            continue
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        source_pdf = str(notes.get("source_pdf") or "")
        if str(record.get("canonical_paper_id") or "") == paper_id or source_stem in source_pdf:
            candidates.append(record)
    return candidates


def _paper_identifiers(paper_code: str) -> tuple[str, str]:
    match = re.fullmatch(r"9709/(\d{2})/([A-Z])/([A-Z])/(\d{2})", paper_code)
    if not match:
        return "", ""
    component, season_left, season_right, year = match.groups()
    season_code = "m"
    canonical_season = "summer"
    if (season_left, season_right) == ("O", "N"):
        season_code = "w"
        canonical_season = "winter"
    elif (season_left, season_right) == ("F", "M"):
        season_code = "s"
        canonical_season = "spring"
    return f"{component}{canonical_season}{year}", f"9709_{season_code}{year}_qp_{component}"


def _text_candidates(question: AssignmentQuestion, question_bank: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in question_bank:
        score = _question_text_similarity(question.text, str(record.get("question_text") or ""))
        if score >= PARTIAL_TEXT_MATCH:
            scored.append((score, record))
    return sorted(scored, key=lambda item: item[0], reverse=True)[:5]


def _question_text_similarity(left: str, right: str) -> float:
    left_tokens = _fingerprint_tokens(left)
    right_tokens = _fingerprint_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    jaccard = len(intersection) / len(union)
    containment = len(intersection) / min(len(left_tokens), len(right_tokens))
    return max(jaccard, containment * 0.85)


def _fingerprint_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", _normalize_text(text)))
    return {token for token in tokens if len(token) > 1 and token not in {"the", "and", "for", "with", "where", "hence"}}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ").lower()).strip()


def _matched_record(
    *,
    question: AssignmentQuestion,
    record: dict[str, Any],
    status: str,
    confidence: float,
    evidence: list[str],
) -> dict[str, object]:
    mark_scheme_status, mark_scheme_id, mark_scheme_evidence = _mark_scheme_status(record)
    structural_uncertainty = _structural_uncertainty(record)
    return {
        "assignment_question_index": question.assignment_question_index,
        "assignment_question_number": question.question_number,
        "assignment_paper_code": question.paper_code,
        "assignment_pages": question.page_numbers,
        "matched_question_id": str(record.get("question_id") or ""),
        "match_status": status,
        "match_confidence": round(confidence, 3),
        "evidence": [*evidence, *mark_scheme_evidence],
        "matched_mark_scheme_id": mark_scheme_id,
        "mark_scheme_status": mark_scheme_status,
        "teacher_review_required": status != "matched_confident" or mark_scheme_status != "matched_confident" or bool(structural_uncertainty),
        "structural_uncertainty": structural_uncertainty,
    }


def _unmatched_record(
    *,
    question: AssignmentQuestion,
    status: str,
    confidence: float,
    evidence: list[str],
) -> dict[str, object]:
    return {
        "assignment_question_index": question.assignment_question_index,
        "assignment_question_number": question.question_number,
        "assignment_paper_code": question.paper_code,
        "assignment_pages": question.page_numbers,
        "matched_question_id": None,
        "match_status": status,
        "match_confidence": confidence,
        "evidence": evidence,
        "matched_mark_scheme_id": None,
        "mark_scheme_status": "blocked_no_mark_scheme",
        "teacher_review_required": True,
        "structural_uncertainty": ["question_not_confidently_matched"],
    }


def _blocked_unreadable_record() -> dict[str, object]:
    return {
        "assignment_question_index": 1,
        "matched_question_id": None,
        "match_status": "blocked_assignment_text_unreadable",
        "match_confidence": 0.0,
        "evidence": ["assignment_pdf_native_text_unavailable"],
        "matched_mark_scheme_id": None,
        "mark_scheme_status": "blocked_no_mark_scheme",
        "teacher_review_required": True,
    }


def _mark_scheme_status(record: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    block_ids = [str(item) for item in record.get("mark_scheme_block_ids", []) if str(item).strip()]
    artifact = str(record.get("canonical_mark_scheme_artifact") or record.get("mark_scheme_image_path") or "")
    text = str(record.get("mark_scheme_text") or "")
    raw_score = record.get("mark_scheme_confidence_score")
    score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
    mark_scheme_id = block_ids[0] if block_ids else artifact or None
    evidence = [f"mark_scheme_confidence:{score:.3f}"]
    if artifact:
        evidence.append(f"mark_scheme_artifact:{artifact}")
    if not (block_ids or artifact or text):
        return "blocked_no_mark_scheme", None, evidence
    if score >= CONFIDENT_MARK_SCHEME_SCORE and (artifact or block_ids):
        return "matched_confident", mark_scheme_id, evidence
    return "matched_partial", mark_scheme_id, evidence


def _structural_uncertainty(record: dict[str, Any]) -> list[str]:
    uncertainties: list[str] = []
    if str(record.get("text_only_status") or "") not in {"ready", ""}:
        uncertainties.append(f"text_only_status:{record.get('text_only_status')}")
    if str(record.get("visual_curation_status") or "") in {"fail"}:
        uncertainties.append(f"visual_curation_status:{record.get('visual_curation_status')}")
    if str(record.get("question_text_trust") or "") in {"low", "failed"}:
        uncertainties.append(f"question_text_trust:{record.get('question_text_trust')}")
    return uncertainties


def _automatic_grading_status(*, total_questions: int, safe_questions: int, accepted_submissions: int) -> str:
    if accepted_submissions == 0:
        return "blocked"
    if total_questions == 0 or safe_questions == 0:
        return "blocked"
    if safe_questions < total_questions:
        return "partial"
    return "available"


def _draft_block_reason(*, blocked_questions: list[dict[str, object]], extraction: dict[str, object]) -> str:
    if extraction.get("status") != "extracted" or extraction.get("text_extractable") is not True:
        return "submission page/text extraction was unavailable"
    if blocked_questions:
        return "assignment question or mark scheme was not confidently matched"
    return "deterministic student answer mapping is not available in quiz-packet"


def _write_draft_summary_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["student_id", "submission_id", "status", "score", "max_score", "student_facing", "teacher_review_required", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def _submission_table(report_path: Path, submissions: list[object]) -> str:
    if not submissions:
        return "<p>None.</p>"
    rows = ["<table><tr><th>Student</th><th>Source file</th><th>Status</th><th>PDF</th><th>Reasons</th></tr>"]
    for submission in submissions:
        student_id = getattr(submission, "student_id", "")
        source_filename = getattr(submission, "source_filename", "")
        status = getattr(submission, "status", "")
        stored_pdf_path = Path(getattr(submission, "stored_pdf_path", ""))
        reasons = "; ".join(getattr(submission, "validation_reasons", []))
        rows.append(
            "<tr>"
            f"<td>{_e(student_id)}</td>"
            f"<td>{_e(source_filename)}</td>"
            f"<td>{_e(status)}</td>"
            f"<td>{_link(report_path, stored_pdf_path) if stored_pdf_path.as_posix() else ''}</td>"
            f"<td>{_e(reasons)}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _matching_table(records: list[object]) -> str:
    if not records:
        return "<p>No assignment questions detected.</p>"
    rows = ["<table><tr><th>#</th><th>Paper</th><th>Question</th><th>Match</th><th>Confidence</th><th>Mark scheme</th><th>Review?</th></tr>"]
    for record in records:
        if not isinstance(record, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_e(record.get('assignment_question_index', ''))}</td>"
            f"<td>{_e(record.get('assignment_paper_code', ''))}</td>"
            f"<td>{_e(record.get('assignment_question_number', ''))}</td>"
            f"<td>{_e(record.get('match_status', ''))}: {_e(record.get('matched_question_id', ''))}</td>"
            f"<td>{_e(record.get('match_confidence', ''))}</td>"
            f"<td>{_e(record.get('mark_scheme_status', ''))}: {_e(record.get('matched_mark_scheme_id', ''))}</td>"
            f"<td>{_e(record.get('teacher_review_required', ''))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _warnings_block(warnings: list[str]) -> str:
    unique = _dedupe(warnings)
    if not unique:
        return "<div class=\"ok\">No packet warnings.</div>"
    items = "".join(f"<li>{_e(warning)}</li>" for warning in unique)
    return f"<div class=\"warn\"><ul>{items}</ul></div>"


def _row(left: str, right: str) -> str:
    return f"<tr><th>{_e(left)}</th><td>{right}</td></tr>"


def _link(report_path: Path, target: Path) -> str:
    href = os.path.relpath(target, start=report_path.parent)
    return f"<a href=\"{html.escape(href, quote=True)}\">{_e(target.as_posix())}</a>"


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _open_path(path: Path) -> bool:
    try:
        return bool(webbrowser.open(path.resolve().as_uri()))
    except Exception:
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
