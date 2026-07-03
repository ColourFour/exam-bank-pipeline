from __future__ import annotations

import json
from pathlib import Path

import fitz

from exam_bank.submissions.answer_check import build_submission_answer_check


def _write_pdf(path: Path, text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _write_assignment(path: Path, *, source_question_ids: list[str] | None = None, pdf_text: str | None = "Question 1\nQuestion 2") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_text is not None:
        _write_pdf(path.parent / "assignment.pdf", pdf_text)
    path.write_text(
        json.dumps(
            {
                "assignment_id": "hw1",
                "course_id": "p3",
                "title": "Homework 1",
                "class_id": "class_12a",
                "timezone": "UTC",
                "accepted_file_types": ["pdf"],
                "max_files_per_student": 1,
                "max_file_size_mb": 50,
                "allow_late": True,
                "source_question_ids": source_question_ids or [],
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(output_root: Path, pdf_path: Path) -> None:
    manifest = {
        "accepted_submissions": [
            {
                "assignment_id": "hw1",
                "student_id": "S0001",
                "submission_id": "hw1:S0001.pdf:abc123",
                "source_filename": pdf_path.name,
                "stored_pdf_path": pdf_path.as_posix(),
                "received_at": "2026-06-28T10:00:00+00:00",
            }
        ]
    }
    target = output_root / "hw1" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest), encoding="utf-8")


def test_answer_check_marks_all_native_text_questions_answered(tmp_path: Path) -> None:
    assignment_path = tmp_path / "classes" / "assignment.json"
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    pdf_path = output_root / "hw1" / "accepted_pdfs" / "S0001.pdf"
    _write_assignment(assignment_path, source_question_ids=["paper_a_q1", "paper_a_q2"])
    _write_pdf(pdf_path, "Question 1 complete\nQuestion 2 complete")
    _write_manifest(output_root, pdf_path)

    result = build_submission_answer_check(
        assignment_id="hw1",
        assignment_path=assignment_path,
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    student = result["students"][0]
    assert student["total_answered"] == 2
    assert [question["status"] for question in student["questions"]] == ["answered", "answered"]
    assert (output_root / "hw1" / "answer_check" / "answer_check_results.json").is_file()
    assert (reports_root / "hw1_answer_check.csv").is_file()


def test_answer_check_marks_missing_questions_zero(tmp_path: Path) -> None:
    assignment_path = tmp_path / "classes" / "assignment.json"
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    pdf_path = output_root / "hw1" / "accepted_pdfs" / "S0001.pdf"
    _write_assignment(assignment_path, source_question_ids=["Q1", "Q2"])
    _write_pdf(pdf_path, "Q1 complete")
    _write_manifest(output_root, pdf_path)

    result = build_submission_answer_check(
        assignment_id="hw1",
        assignment_path=assignment_path,
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    student = result["students"][0]
    assert student["total_answered"] == 1
    assert [(question["status"], question["score"]) for question in student["questions"]] == [("answered", 1), ("missing", 0)]


def test_answer_check_requires_review_for_blank_native_text(tmp_path: Path) -> None:
    assignment_path = tmp_path / "classes" / "assignment.json"
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    pdf_path = output_root / "hw1" / "accepted_pdfs" / "S0001.pdf"
    _write_assignment(assignment_path, source_question_ids=["Q1"])
    _write_pdf(pdf_path, "")
    _write_manifest(output_root, pdf_path)

    result = build_submission_answer_check(
        assignment_id="hw1",
        assignment_path=assignment_path,
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    student = result["students"][0]
    assert student["status"] == "review_needed"
    assert student["questions"][0]["status"] == "review_needed"
    assert student["questions"][0]["score"] == 0


def test_answer_check_reports_missing_question_set(tmp_path: Path) -> None:
    assignment_path = tmp_path / "classes" / "assignment.json"
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    pdf_path = output_root / "hw1" / "accepted_pdfs" / "S0001.pdf"
    _write_assignment(assignment_path, source_question_ids=[], pdf_text=None)
    _write_pdf(pdf_path, "Question 1 complete")
    _write_manifest(output_root, pdf_path)

    result = build_submission_answer_check(
        assignment_id="hw1",
        assignment_path=assignment_path,
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    student = result["students"][0]
    assert result["question_set_missing"] is True
    assert result["question_source"] == "missing"
    assert student["status"] == "review_needed"
    assert student["questions"] == []
    assert "question_set_missing" in student["notes"]
