from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

from exam_bank.mupdf_tools import quiet_mupdf

quiet_mupdf(fitz)


ANSWER_CHECK_DIRNAME = "answer_check"
ANSWER_CHECK_FILENAME = "answer_check_results.json"
ANSWER_CHECK_CSV_FIELDS = [
    "assignment_id",
    "student_id",
    "submission_id",
    "source_filename",
    "status",
    "total_answered",
    "total_expected",
    "question_id",
    "question_label",
    "question_status",
    "score",
    "max_score",
    "notes",
]


def build_submission_answer_check(
    *,
    assignment_id: str,
    assignment_path: Path,
    submission_output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
) -> dict[str, object]:
    """Run the temporary per-question answer-presence check for accepted PDFs."""

    assignment = _read_json(assignment_path, default={})
    manifest_path = submission_output_root / assignment_id / "manifest.json"
    manifest = _read_json(manifest_path, default={})
    questions, question_source = _expected_questions(assignment, assignment_path.parent / str(assignment.get("assignment_pdf") or "assignment.pdf"))
    question_set_missing = not questions

    students: list[dict[str, object]] = []
    for submission in _accepted_submissions(manifest):
        students.append(_check_submission(assignment_id, submission, questions, question_set_missing))

    summary = {
        "students_checked": len(students),
        "question_count": len(questions),
        "question_source": question_source,
        "question_set_missing": question_set_missing,
        "answered_count": sum(int(student.get("total_answered") or 0) for student in students),
        "review_needed_count": sum(1 for student in students if student.get("status") == "review_needed"),
    }
    payload: dict[str, object] = {
        "assignment_id": assignment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question_source": question_source,
        "question_set_missing": question_set_missing,
        "questions": questions,
        "students": students,
        "summary": summary,
        "teacher_review_required": True,
        "student_facing": False,
    }

    answer_check_dir = submission_output_root / assignment_id / ANSWER_CHECK_DIRNAME
    answer_check_dir.mkdir(parents=True, exist_ok=True)
    result_path = answer_check_dir / ANSWER_CHECK_FILENAME
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = reports_root / f"{assignment_id}_answer_check.csv"
    _write_answer_check_csv(csv_path, assignment_id, students)
    payload["result_path"] = result_path.as_posix()
    payload["csv_path"] = csv_path.as_posix()
    return payload


def load_answer_check_results(
    assignment_id: str,
    *,
    submission_output_root: Path = Path("output/submissions"),
) -> dict[str, object]:
    return _read_json(submission_output_root / assignment_id / ANSWER_CHECK_DIRNAME / ANSWER_CHECK_FILENAME, default={})


def _expected_questions(assignment: dict[str, object], assignment_pdf: Path) -> tuple[list[dict[str, str]], str]:
    source_ids = [str(item).strip() for item in assignment.get("source_question_ids", []) if str(item).strip()]
    if source_ids:
        return [_question_payload(item) for item in source_ids], "source_question_ids"

    text, notes, page_count = _extract_pdf_text(assignment_pdf)
    if text:
        detected = _extract_question_labels(text)
        if detected:
            return [_question_payload(label) for label in detected], "assignment_pdf"
    if assignment_pdf.exists() and not notes and page_count > 0:
        return [_question_payload(f"Q{index}") for index in range(1, page_count + 1)], "page_fallback"
    return [], "missing"


def _question_payload(value: str) -> dict[str, str]:
    label = _display_question_label(value)
    return {
        "question_id": _question_id(label),
        "question_label": label,
        "display_label": label,
    }


def _display_question_label(value: str) -> str:
    cleaned = str(value).strip()
    match = re.fullmatch(r"(?:q|question)?\s*([0-9]{1,3}[a-z]?)", cleaned, flags=re.IGNORECASE)
    if match:
        return f"Q{match.group(1).upper()}"
    embedded = re.search(r"(?:^|[^a-z0-9])(?:q|question)[_\-\s]*([0-9]{1,3}[a-z]?)(?:$|[^a-z0-9])", cleaned, flags=re.IGNORECASE)
    if embedded:
        return f"Q{embedded.group(1).upper()}"
    return cleaned


def _question_id(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _extract_question_labels(text: str) -> list[str]:
    labels: set[str] = set()
    for match in re.finditer(r"\b(?:q|question)\s*([0-9]{1,3}[a-z]?)\b", text, flags=re.IGNORECASE):
        labels.add(f"Q{match.group(1).upper()}")
    for match in re.finditer(r"(?m)^\s*([0-9]{1,3}[a-z]?)\s*[\).:]", text, flags=re.IGNORECASE):
        labels.add(f"Q{match.group(1).upper()}")
    return sorted(labels, key=_question_sort_key)


def _question_sort_key(label: str) -> tuple[int, str]:
    match = re.search(r"([0-9]+)([A-Z]?)", label.upper())
    if not match:
        return (9999, label)
    return (int(match.group(1)), match.group(2))


def _accepted_submissions(manifest: dict[str, object]) -> list[dict[str, object]]:
    submissions = manifest.get("accepted_submissions", [])
    if not isinstance(submissions, list):
        return []
    return [item for item in submissions if isinstance(item, dict)]


def _check_submission(
    assignment_id: str,
    submission: dict[str, object],
    questions: list[dict[str, str]],
    question_set_missing: bool,
) -> dict[str, object]:
    notes: list[str] = []
    path = Path(str(submission.get("stored_pdf_path") or ""))
    text, extraction_notes, _page_count = _extract_pdf_text(path)
    notes.extend(extraction_notes)

    if question_set_missing:
        notes.append("question_set_missing")
        question_results: list[dict[str, object]] = []
        status = "review_needed"
    elif not text:
        question_results = [_question_result(question, "review_needed", 0, ["native_text_unavailable"]) for question in questions]
        status = "review_needed"
        if not notes:
            notes.append("pdf_text_not_extractable")
    else:
        question_results = []
        for question in questions:
            answered = _has_answer_evidence(text, question["question_label"], single_question=len(questions) == 1)
            question_results.append(_question_result(question, "answered" if answered else "missing", 1 if answered else 0, []))
        status = "checked"

    total_answered = sum(1 for item in question_results if item.get("status") == "answered")
    return {
        "assignment_id": assignment_id,
        "student_id": str(submission.get("student_id") or ""),
        "submission_id": str(submission.get("submission_id") or ""),
        "source_filename": str(submission.get("source_filename") or ""),
        "stored_pdf_path": str(submission.get("stored_pdf_path") or ""),
        "submitted_at": str(submission.get("received_at") or ""),
        "status": status,
        "total_answered": total_answered,
        "total_expected": len(questions),
        "teacher_review_required": True,
        "student_facing": False,
        "notes": notes,
        "questions": question_results,
    }


def _question_result(question: dict[str, str], status: str, score: int, notes: list[str]) -> dict[str, object]:
    return {
        "question_id": question["question_id"],
        "question_label": question["question_label"],
        "display_label": question["display_label"],
        "status": status,
        "score": score,
        "max_score": 1,
        "teacher_review_required": True,
        "student_facing": False,
        "notes": notes,
    }


def _has_answer_evidence(text: str, question_label: str, *, single_question: bool) -> bool:
    normalized = " ".join(text.split())
    if not normalized:
        return False
    match = re.search(r"([0-9]{1,3})([A-Z]?)", question_label.upper())
    if not match:
        return question_label.lower() in normalized.lower()
    number = match.group(1)
    suffix = match.group(2)
    suffix_pattern = re.escape(suffix) if suffix else "[a-z]?"
    patterns = [
        rf"\bq\s*{number}{suffix_pattern}\b",
        rf"\bquestion\s*{number}{suffix_pattern}\b",
        rf"(?m)^\s*{number}{suffix_pattern}\s*[\).:]",
    ]
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
        return True
    return single_question and len(normalized) >= 12


def _extract_pdf_text(path: Path) -> tuple[str, list[str], int]:
    if not path.exists():
        return "", ["missing_pdf"], 0
    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc), [], doc.page_count
    except Exception as exc:  # noqa: BLE001
        return "", [f"pdf_text_extract_failed:{exc.__class__.__name__}"], 0


def _write_answer_check_csv(path: Path, assignment_id: str, students: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANSWER_CHECK_CSV_FIELDS)
        writer.writeheader()
        for student in students:
            questions = student.get("questions")
            if isinstance(questions, list) and questions:
                for question in questions:
                    if isinstance(question, dict):
                        writer.writerow(_answer_check_csv_row(assignment_id, student, question))
            else:
                writer.writerow(_answer_check_csv_row(assignment_id, student, {}))


def _answer_check_csv_row(assignment_id: str, student: dict[str, object], question: dict[str, object]) -> dict[str, object]:
    notes = student.get("notes")
    question_notes = question.get("notes") if isinstance(question, dict) else []
    note_values = [*(notes if isinstance(notes, list) else []), *(question_notes if isinstance(question_notes, list) else [])]
    return {
        "assignment_id": assignment_id,
        "student_id": student.get("student_id", ""),
        "submission_id": student.get("submission_id", ""),
        "source_filename": student.get("source_filename", ""),
        "status": student.get("status", ""),
        "total_answered": student.get("total_answered", 0),
        "total_expected": student.get("total_expected", 0),
        "question_id": question.get("question_id", "") if isinstance(question, dict) else "",
        "question_label": question.get("question_label", "") if isinstance(question, dict) else "",
        "question_status": question.get("status", "") if isinstance(question, dict) else "",
        "score": question.get("score", 0) if isinstance(question, dict) else 0,
        "max_score": question.get("max_score", 1) if isinstance(question, dict) else 0,
        "notes": ";".join(str(item) for item in note_values if str(item)),
    }


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
