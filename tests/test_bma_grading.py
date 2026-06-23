from __future__ import annotations

import csv
import json
from pathlib import Path

from exam_bank.submissions.bma_grading import build_bma_grading_matrix


def _write_fixture_packet(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "output" / "submissions"
    reports_root = tmp_path / "reports" / "submissions"
    assignment_id = "quiz1"
    assignment_root = output_root / assignment_id
    grading_root = assignment_root / "grading_review"
    accepted_pdf = assignment_root / "accepted_pdfs" / "Alice.pdf"
    accepted_pdf.parent.mkdir(parents=True, exist_ok=True)
    accepted_pdf.write_bytes(b"%PDF-1.4\n% fake test fixture\n")
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / f"{assignment_id}_completion.csv").write_text(
        "assignment_id,assignment_title,class_id,student_id,display_name,email,status,submitted_at,late,source_filename,stored_pdf_path,rejection_reasons,notes\n"
        f"{assignment_id},Quiz,local_class,alice,Alice,,submitted,,False,Alice.pdf,{accepted_pdf.as_posix()},,\n",
        encoding="utf-8",
    )
    (assignment_root / "manifest.json").write_text(
        json.dumps(
            {
                "completion_report": (reports_root / f"{assignment_id}_completion.csv").as_posix(),
                "accepted_submissions": [
                    {
                        "student_id": "alice",
                        "source_filename": "Alice.pdf",
                        "stored_pdf_path": accepted_pdf.as_posix(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    page = grading_root / "upright_quiz_pages" / "alice" / "qpage_01.jpg"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_bytes(b"not actually an image; path existence is enough for this unit test")
    (grading_root / "grading_rubric_bma.json").write_text(
        json.dumps(
            {
                "assignment_id": assignment_id,
                "questions": [
                    {
                        "question_id": "q_bank_1",
                        "label": "Q1 test",
                        "evidence_pages": ["qpage_01.jpg"],
                        "marks": [
                            {
                                "mark_id": "Q1_M1",
                                "type": "M",
                                "points": 1,
                                "description": "Valid method seen.",
                            },
                            {
                                "mark_id": "Q1_A1",
                                "type": "A",
                                "points": 1,
                                "description": "Correct answer seen.",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return output_root, reports_root


def test_bma_grading_matrix_fails_closed_without_visual_grader(tmp_path: Path) -> None:
    output_root, reports_root = _write_fixture_packet(tmp_path)

    result = build_bma_grading_matrix(
        assignment_id="quiz1",
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    assert result["exit_code"] == 0
    assert result["status"] == "blocked_no_visual_grader"

    decisions = json.loads(Path(result["bma_mark_decisions_json"]).read_text(encoding="utf-8"))
    assert decisions["summary"]["total_mark_decisions"] == 2
    assert decisions["summary"]["unassessable_marks"] == 2
    assert decisions["summary"]["marks_awarded_confidently"] == 0
    assert decisions["summary"]["student_facing"] is False
    assert decisions["summary"]["email_sent"] is False
    for record in decisions["records"]:
        assert record["awarded"] is None
        assert record["confidence"] == 0.0
        assert record["teacher_review_required"] is True
        assert record["student_facing"] is False
        assert record["grading_status"] == "blocked_no_visual_grader"
        assert record["mark_scheme_text"]
        assert record["evidence_image"].endswith("qpage_01.jpg")


def test_bma_matrix_csv_and_report_are_written(tmp_path: Path) -> None:
    output_root, reports_root = _write_fixture_packet(tmp_path)

    result = build_bma_grading_matrix(
        assignment_id="quiz1",
        submission_output_root=output_root,
        reports_root=reports_root,
    )

    matrix_path = Path(result["bma_matrix_csv"])
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "display_name": "Alice",
            "student_id": "alice",
            "q1_awarded": "0",
            "q1_max": "2",
            "q1_review_count": "2",
            "q1_notes": rows[0]["q1_notes"],
            "total_awarded": "0",
            "total_assessable_max": "0",
            "total_full_max": "2",
            "teacher_review_count": "2",
            "grading_status": "blocked_no_visual_grader",
        }
    ]
    assert "No visual grader is configured" in rows[0]["q1_notes"]

    matrix_json = json.loads(Path(result["bma_matrix_json"]).read_text(encoding="utf-8"))
    assert matrix_json["summary"]["suspicious_uniform_scores"] is True
    assert matrix_json["summary"]["every_student_full_marks"] is False

    report_text = Path(result["bma_matrix_report"]).read_text(encoding="utf-8")
    assert "B/M/A Matrix Report" in report_text
    assert "Mark decisions JSON" in report_text
    assert "qpage_01.jpg" in report_text
    assert "Alice.pdf" in report_text
