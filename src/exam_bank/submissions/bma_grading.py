from __future__ import annotations

import csv
import html
import json
import os
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GRADE_STATUSES = {
    "graded_confident",
    "graded_needs_review",
    "blocked_unreadable_submission",
    "blocked_missing_question_crop",
    "blocked_missing_mark_scheme",
    "blocked_no_visual_grader",
    "failed_suspicious_full_marks",
}

DETAILED_FIELDS = [
    "student_id",
    "display_name",
    "question_id",
    "question_label",
    "mark_id",
    "mark_type",
    "max_mark",
    "awarded",
    "confidence",
    "evidence_text",
    "evidence_image",
    "reason",
    "teacher_review_required",
    "student_facing",
    "grading_status",
    "mark_scheme_text",
]

BASE_MATRIX_FIELDS = ["display_name", "student_id"]
TOTAL_MATRIX_FIELDS = [
    "total_awarded",
    "total_assessable_max",
    "total_full_max",
    "teacher_review_count",
    "grading_status",
]


def build_bma_grading_matrix(
    *,
    assignment_id: str,
    submission_output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
    mode: str = "visual-first",
    allow_suspicious_full_marks: bool = False,
    open_report: bool = False,
) -> dict[str, object]:
    if mode != "visual-first":
        raise ValueError(f"Unsupported B/M/A grading mode: {mode}")

    assignment_output = submission_output_root / assignment_id
    grading_root = assignment_output / "grading_review"
    reports_root.mkdir(parents=True, exist_ok=True)
    grading_root.mkdir(parents=True, exist_ok=True)

    manifest_path = assignment_output / "manifest.json"
    rubric_path = grading_root / "grading_rubric_bma.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing submission manifest: {manifest_path}")
    if not rubric_path.exists():
        raise FileNotFoundError(f"Missing B/M/A rubric: {rubric_path}")

    manifest = _load_json(manifest_path)
    rubric = _load_json(rubric_path)
    completion_rows = _load_completion_rows(Path(str(manifest.get("completion_report") or reports_root / f"{assignment_id}_completion.csv")))
    accepted = [
        item
        for item in manifest.get("accepted_submissions", [])
        if isinstance(item, dict) and str(item.get("student_id") or "").strip()
    ]
    questions = _normalize_questions(rubric)
    visual_grader_available = False

    decisions: list[dict[str, object]] = []
    for submission in sorted(accepted, key=lambda item: str(item.get("student_id") or "")):
        student_id = str(submission.get("student_id") or "")
        completion = completion_rows.get(student_id, {})
        display_name = str(completion.get("display_name") or student_id)
        student_page_dir = grading_root / "upright_quiz_pages" / student_id
        student_pdf = Path(str(submission.get("stored_pdf_path") or ""))

        for question in questions:
            evidence_images = _question_evidence_images(student_page_dir, question)
            question_status = _question_status(
                visual_grader_available=visual_grader_available,
                student_page_dir=student_page_dir,
                evidence_images=evidence_images,
                question=question,
            )
            for mark in question["marks"]:
                decision = _blocked_decision(
                    student_id=student_id,
                    display_name=display_name,
                    source_filename=str(submission.get("source_filename") or ""),
                    stored_pdf_path=student_pdf.as_posix(),
                    question=question,
                    mark=mark,
                    evidence_images=evidence_images,
                    grading_status=question_status,
                )
                decisions.append(decision)

    student_questions, matrix_rows = _aggregate_matrix(decisions, questions)
    audit = _audit_scores(matrix_rows, allow_suspicious_full_marks=allow_suspicious_full_marks)
    if audit["failed"]:
        for row in matrix_rows:
            row["grading_status"] = "failed_suspicious_full_marks"

    summary = _summary(
        assignment_id=assignment_id,
        questions=questions,
        decisions=decisions,
        matrix_rows=matrix_rows,
        audit=audit,
        mode=mode,
    )

    detailed_json_path = grading_root / "bma_mark_decisions.json"
    detailed_csv_path = grading_root / "bma_mark_decisions.csv"
    matrix_json_path = grading_root / "bma_matrix.json"
    matrix_csv_path = grading_root / "bma_matrix.csv"
    report_path = reports_root / f"{assignment_id}_bma_matrix_report.html"

    _write_json(
        detailed_json_path,
        {
            "summary": summary,
            "records": decisions,
        },
    )
    _write_decisions_csv(detailed_csv_path, decisions)
    _write_json(
        matrix_json_path,
        {
            "summary": summary,
            "students": matrix_rows,
            "student_questions": student_questions,
        },
    )
    _write_matrix_csv(matrix_csv_path, matrix_rows, questions)
    _write_matrix_report(
        path=report_path,
        assignment_id=assignment_id,
        summary=summary,
        matrix_rows=matrix_rows,
        student_questions=student_questions,
        questions=questions,
        manifest_path=manifest_path,
        original_teacher_report=reports_root / f"{assignment_id}_teacher_report.html",
        rubric_path=rubric_path,
        detailed_json_path=detailed_json_path,
        detailed_csv_path=detailed_csv_path,
        matrix_json_path=matrix_json_path,
        matrix_csv_path=matrix_csv_path,
    )
    report_opened = False
    if open_report:
        report_opened = _open_path(report_path)

    return {
        "assignment_id": assignment_id,
        "status": "failed_suspicious_full_marks" if audit["failed"] else "blocked_no_visual_grader",
        "summary": summary,
        "bma_mark_decisions_json": detailed_json_path,
        "bma_mark_decisions_csv": detailed_csv_path,
        "bma_matrix_json": matrix_json_path,
        "bma_matrix_csv": matrix_csv_path,
        "bma_matrix_report": report_path,
        "report_opened": report_opened,
        "exit_code": 2 if audit["failed"] else 0,
    }


def _normalize_questions(rubric: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for index, raw_question in enumerate(rubric.get("questions", []), start=1):
        if not isinstance(raw_question, dict):
            continue
        qkey = f"q{index}"
        marks: list[dict[str, Any]] = []
        for raw_mark in raw_question.get("marks", []):
            if not isinstance(raw_mark, dict):
                continue
            mark_type = str(raw_mark.get("type") or raw_mark.get("mark_type") or "").upper()
            marks.append(
                {
                    "mark_id": str(raw_mark.get("mark_id") or f"{qkey}_{mark_type}{len(marks) + 1}"),
                    "mark_type": mark_type,
                    "max_mark": int(raw_mark.get("points") or raw_mark.get("max_mark") or 1),
                    "mark_scheme_text": str(raw_mark.get("description") or raw_mark.get("mark_scheme_text") or ""),
                }
            )
        questions.append(
            {
                "question_id": str(raw_question.get("question_id") or qkey),
                "question_key": qkey,
                "question_label": f"Q{index}",
                "title": str(raw_question.get("label") or f"Q{index}"),
                "full_max": sum(int(mark["max_mark"]) for mark in marks),
                "evidence_pages": [str(item) for item in raw_question.get("evidence_pages", [])],
                "marks": marks,
            }
        )
    return questions


def _question_evidence_images(student_page_dir: Path, question: dict[str, Any]) -> list[str]:
    images: list[str] = []
    for page_name in question.get("evidence_pages", []):
        if not str(page_name).lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = student_page_dir / str(page_name)
        if path.exists():
            images.append(path.as_posix())
    return images


def _question_status(
    *,
    visual_grader_available: bool,
    student_page_dir: Path,
    evidence_images: list[str],
    question: dict[str, Any],
) -> str:
    if not visual_grader_available:
        return "blocked_no_visual_grader"
    if not student_page_dir.exists():
        return "blocked_unreadable_submission"
    if question.get("marks") == []:
        return "blocked_missing_mark_scheme"
    if not evidence_images:
        return "blocked_missing_question_crop"
    return "graded_needs_review"


def _blocked_decision(
    *,
    student_id: str,
    display_name: str,
    source_filename: str,
    stored_pdf_path: str,
    question: dict[str, Any],
    mark: dict[str, Any],
    evidence_images: list[str],
    grading_status: str,
) -> dict[str, object]:
    evidence_image = evidence_images[0] if evidence_images else ""
    reason_by_status = {
        "blocked_no_visual_grader": "No visual grader is configured, so handwritten work cannot be assessed safely.",
        "blocked_unreadable_submission": "Rendered student pages are missing or unreadable.",
        "blocked_missing_question_crop": "No question evidence page or crop was found for this mark.",
        "blocked_missing_mark_scheme": "No B/M/A mark scheme item was available for this question.",
    }
    reason = reason_by_status.get(grading_status, "Teacher review required before this mark can be trusted.")
    return {
        "student_id": student_id,
        "display_name": display_name,
        "source_filename": source_filename,
        "stored_pdf_path": stored_pdf_path,
        "question_id": question["question_id"],
        "question_label": question["question_label"],
        "question_title": question["title"],
        "mark_id": mark["mark_id"],
        "mark_type": mark["mark_type"],
        "max_mark": mark["max_mark"],
        "awarded": None,
        "confidence": 0.0,
        "evidence_text": "",
        "evidence_image": evidence_image,
        "evidence_images": evidence_images,
        "reason": reason,
        "teacher_review_required": True,
        "student_facing": False,
        "grading_status": grading_status,
        "mark_scheme_text": mark["mark_scheme_text"],
    }


def _aggregate_matrix(
    decisions: list[dict[str, object]],
    questions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, object]]], list[dict[str, object]]]:
    by_student: dict[str, list[dict[str, object]]] = {}
    for decision in decisions:
        by_student.setdefault(str(decision["student_id"]), []).append(decision)

    student_questions: dict[str, dict[str, dict[str, object]]] = {}
    matrix_rows: list[dict[str, object]] = []
    for student_id, student_decisions in sorted(by_student.items()):
        display_name = str(student_decisions[0].get("display_name") or student_id)
        row: dict[str, object] = {
            "display_name": display_name,
            "student_id": student_id,
            "total_awarded": 0,
            "total_assessable_max": 0,
            "total_full_max": 0,
            "teacher_review_count": 0,
            "grading_status": "blocked_no_visual_grader",
            "suspicious_full_mark_low_evidence": False,
        }
        student_questions[student_id] = {}
        for question in questions:
            qkey = question["question_key"]
            qdecisions = [item for item in student_decisions if item["question_id"] == question["question_id"]]
            awarded_total = sum(int(item["awarded"]) for item in qdecisions if item.get("awarded") in {0, 1})
            assessable_max = sum(int(item["max_mark"]) for item in qdecisions if item.get("awarded") in {0, 1})
            full_max = sum(int(item["max_mark"]) for item in qdecisions)
            review_count = sum(1 for item in qdecisions if item.get("teacher_review_required") is True)
            notes = _question_notes(qdecisions)
            qstatus = _aggregate_status(qdecisions)
            student_questions[student_id][qkey] = {
                "question_id": question["question_id"],
                "question_label": question["question_label"],
                "awarded": awarded_total,
                "assessable_max": assessable_max,
                "full_max": full_max,
                "review_count": review_count,
                "notes": notes,
                "grading_status": qstatus,
                "decisions": qdecisions,
            }
            row[f"{qkey}_awarded"] = awarded_total
            row[f"{qkey}_max"] = full_max
            row[f"{qkey}_review_count"] = review_count
            row[f"{qkey}_notes"] = notes
            row["total_awarded"] = int(row["total_awarded"]) + awarded_total
            row["total_assessable_max"] = int(row["total_assessable_max"]) + assessable_max
            row["total_full_max"] = int(row["total_full_max"]) + full_max
            row["teacher_review_count"] = int(row["teacher_review_count"]) + review_count
        low_conf_awarded = [
            item
            for item in student_decisions
            if item.get("awarded") == 1 and float(item.get("confidence") or 0.0) < 0.85
        ]
        awarded = [item for item in student_decisions if item.get("awarded") == 1]
        if int(row["total_awarded"]) == int(row["total_full_max"]) and awarded:
            row["suspicious_full_mark_low_evidence"] = len(low_conf_awarded) / max(1, len(awarded)) > 0.25
        matrix_rows.append(row)
    return student_questions, matrix_rows


def _question_notes(decisions: list[dict[str, object]]) -> str:
    notes: list[str] = []
    per_type_counter: Counter[str] = Counter()
    for decision in decisions:
        mark_type = str(decision.get("mark_type") or "?")
        per_type_counter[mark_type] += 1
        label = f"{mark_type}{per_type_counter[mark_type]}"
        awarded = "?" if decision.get("awarded") is None else str(decision.get("awarded"))
        max_mark = int(decision.get("max_mark") or 1)
        confidence = float(decision.get("confidence") or 0.0)
        reason = str(decision.get("reason") or "")
        notes.append(f"{label}: {awarded}/{max_mark} conf {confidence:.2f} - {reason}")
    return " | ".join(notes)


def _aggregate_status(decisions: list[dict[str, object]]) -> str:
    statuses = {str(item.get("grading_status") or "") for item in decisions}
    for status in [
        "blocked_no_visual_grader",
        "blocked_unreadable_submission",
        "blocked_missing_question_crop",
        "blocked_missing_mark_scheme",
        "graded_needs_review",
        "graded_confident",
    ]:
        if status in statuses:
            return status
    return "graded_needs_review"


def _audit_scores(matrix_rows: list[dict[str, object]], *, allow_suspicious_full_marks: bool) -> dict[str, object]:
    suspicious_full_rows = sum(1 for row in matrix_rows if row.get("suspicious_full_mark_low_evidence") is True)
    scores = [int(row.get("total_awarded") or 0) for row in matrix_rows]
    full_scores = [int(row.get("total_full_max") or 0) for row in matrix_rows]
    suspicious_uniform_scores = bool(scores) and len(set(scores)) == 1
    every_student_full_marks = bool(scores) and all(score == full for score, full in zip(scores, full_scores))
    failed = every_student_full_marks and not allow_suspicious_full_marks
    return {
        "suspicious_full_mark_low_evidence_rows": suspicious_full_rows,
        "suspicious_uniform_scores": suspicious_uniform_scores,
        "every_student_full_marks": every_student_full_marks,
        "failed": failed,
    }


def _summary(
    *,
    assignment_id: str,
    questions: list[dict[str, Any]],
    decisions: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
    audit: dict[str, object],
    mode: str,
) -> dict[str, object]:
    confident_awarded = sum(1 for item in decisions if item.get("awarded") == 1 and float(item.get("confidence") or 0.0) >= 0.85)
    low_conf_awarded = sum(1 for item in decisions if item.get("awarded") == 1 and float(item.get("confidence") or 0.0) < 0.85)
    confident_denied = sum(1 for item in decisions if item.get("awarded") == 0 and float(item.get("confidence") or 0.0) >= 0.85)
    unassessable = sum(1 for item in decisions if item.get("awarded") is None)
    review_required = sum(1 for item in decisions if item.get("teacher_review_required") is True)
    return {
        "assignment_id": assignment_id,
        "generated_at": _utc_now(),
        "mode": mode,
        "status": "failed_suspicious_full_marks" if audit["failed"] else "blocked_no_visual_grader",
        "students": len(matrix_rows),
        "questions": len(questions),
        "total_mark_decisions": len(decisions),
        "marks_awarded_confidently": confident_awarded,
        "marks_awarded_low_confidence": low_conf_awarded,
        "marks_denied_confidently": confident_denied,
        "unassessable_marks": unassessable,
        "teacher_review_required": review_required,
        "suspicious_full_mark_rows": audit["suspicious_full_mark_low_evidence_rows"],
        "suspicious_uniform_scores": audit["suspicious_uniform_scores"],
        "every_student_full_marks": audit["every_student_full_marks"],
        "student_facing": False,
        "email_sent": False,
    }


def _write_decisions_csv(path: Path, decisions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAILED_FIELDS)
        writer.writeheader()
        for decision in decisions:
            row = {field: decision.get(field, "") for field in DETAILED_FIELDS}
            row["awarded"] = "" if decision.get("awarded") is None else decision.get("awarded")
            writer.writerow(row)


def _write_matrix_csv(path: Path, matrix_rows: list[dict[str, object]], questions: list[dict[str, Any]]) -> None:
    fields = list(BASE_MATRIX_FIELDS)
    for question in questions:
        qkey = question["question_key"]
        fields.extend([f"{qkey}_awarded", f"{qkey}_max", f"{qkey}_review_count", f"{qkey}_notes"])
    fields.extend(TOTAL_MATRIX_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in matrix_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_matrix_report(
    *,
    path: Path,
    assignment_id: str,
    summary: dict[str, object],
    matrix_rows: list[dict[str, object]],
    student_questions: dict[str, dict[str, dict[str, object]]],
    questions: list[dict[str, Any]],
    manifest_path: Path,
    original_teacher_report: Path,
    rubric_path: Path,
    detailed_json_path: Path,
    detailed_csv_path: Path,
    matrix_json_path: Path,
    matrix_csv_path: Path,
) -> None:
    headers = "".join(f"<th>{_e(question['question_label'])}</th>" for question in questions)
    rows: list[str] = []
    for row in matrix_rows:
        student_id = str(row["student_id"])
        student_decisions = [
            decision
            for qagg in student_questions.get(student_id, {}).values()
            for decision in qagg.get("decisions", [])
            if isinstance(decision, dict)
        ]
        stored_pdf = ""
        if student_decisions:
            stored_pdf = str(student_decisions[0].get("stored_pdf_path") or "")
        question_cells = []
        for question in questions:
            qkey = question["question_key"]
            qagg = student_questions.get(student_id, {}).get(qkey, {})
            links = _evidence_links(path, qagg.get("decisions", []))
            question_cells.append(
                "<td>"
                f"<div><strong>{_e(qagg.get('awarded', 0))} / {_e(qagg.get('assessable_max', 0))} assessed</strong></div>"
                f"<div>{_e(qagg.get('awarded', 0))} / {_e(qagg.get('full_max', 0))} full</div>"
                f"<div>Review: {'yes' if int(qagg.get('review_count', 0) or 0) else 'no'}</div>"
                f"<div class=\"notes\">{_e(qagg.get('notes', ''))}</div>"
                f"<div>{links}</div>"
                "</td>"
            )
        rows.append(
            "<tr>"
            f"<th>{_e(row.get('display_name', ''))}<br><code>{_e(student_id)}</code><br>{_link(path, Path(stored_pdf)) if stored_pdf else ''}</th>"
            + "".join(question_cells)
            + (
                f"<td>{_e(row['total_awarded'])} / {_e(row['total_assessable_max'])} assessed<br>"
                f"{_e(row['total_awarded'])} / {_e(row['total_full_max'])} full<br>"
                f"review {_e(row['teacher_review_count'])} marks</td>"
                f"<td>{_e(row['grading_status'])}</td>"
            )
            + "</tr>"
        )

    summary_items = "".join(
        f"<tr><th>{_e(label)}</th><td>{_e(value)}</td></tr>"
        for label, value in [
            ("Students", summary["students"]),
            ("Questions", summary["questions"]),
            ("Total mark decisions", summary["total_mark_decisions"]),
            ("Marks awarded confidently", summary["marks_awarded_confidently"]),
            ("Marks awarded low confidence", summary["marks_awarded_low_confidence"]),
            ("Marks denied confidently", summary["marks_denied_confidently"]),
            ("Unassessable marks", summary["unassessable_marks"]),
            ("Teacher review required", summary["teacher_review_required"]),
            ("Suspicious full-mark rows", summary["suspicious_full_mark_rows"]),
            ("Suspicious uniform scores", summary["suspicious_uniform_scores"]),
            ("Every student full marks", summary["every_student_full_marks"]),
            ("Student-facing", summary["student_facing"]),
            ("Email sent", summary["email_sent"]),
        ]
    )
    artifact_links = "".join(
        f"<li>{label}: {_link(path, target)}</li>"
        for label, target in [
            ("Manifest", manifest_path),
            ("Rubric JSON", rubric_path),
            ("Mark decisions JSON", detailed_json_path),
            ("Mark decisions CSV", detailed_csv_path),
            ("Matrix JSON", matrix_json_path),
            ("Matrix CSV", matrix_csv_path),
            ("Original teacher report", original_teacher_report),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang=\"en\"><head><meta charset=\"utf-8\">",
                f"<title>{_e(assignment_id)} B/M/A Matrix</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;line-height:1.45;margin:28px;color:#202124}",
                "table{border-collapse:collapse;width:100%;margin:12px 0 24px}",
                "th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left;vertical-align:top}",
                "th{background:#f6f8fa}.notes{font-size:12px;max-height:10em;overflow:auto}",
                ".warn{background:#fff8c5;border:1px solid #f0d060;padding:10px 12px;margin:12px 0}",
                "code{font-size:12px}",
                "</style></head><body>",
                f"<h1>{_e(assignment_id)} B/M/A Matrix Report</h1>",
                "<div class=\"warn\">No visual grading model is configured. This run created evidence-linked mark rows and failed closed instead of awarding guessed marks.</div>",
                "<h2>Summary</h2>",
                f"<table>{summary_items}</table>",
                "<h2>Artifacts</h2>",
                f"<ul>{artifact_links}</ul>",
                "<h2>Matrix</h2>",
                f"<table><tr><th>Student</th>{headers}<th>Total</th><th>Review status</th></tr>{''.join(rows)}</table>",
                "</body></html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _evidence_links(report_path: Path, decisions: object) -> str:
    if not isinstance(decisions, list):
        return ""
    seen: set[str] = set()
    links: list[str] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        evidence_values = []
        if isinstance(decision.get("evidence_images"), list):
            evidence_values.extend(str(item) for item in decision["evidence_images"])
        evidence_values.append(str(decision.get("evidence_image") or ""))
        for evidence in evidence_values:
            if not evidence or evidence in seen:
                continue
            seen.add(evidence)
            links.append(_link(report_path, Path(evidence)))
    return " ".join(links)


def _load_completion_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["student_id"]: row for row in csv.DictReader(handle) if row.get("student_id")}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _link(report_path: Path, target: Path) -> str:
    href = os.path.relpath(target, start=report_path.parent)
    return f"<a href=\"{html.escape(href, quote=True)}\">{_e(target.as_posix())}</a>"


def _open_path(path: Path) -> bool:
    try:
        return bool(webbrowser.open(path.resolve().as_uri()))
    except Exception:
        return False


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
