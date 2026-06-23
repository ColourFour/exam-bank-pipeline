from __future__ import annotations

import csv
import json
from pathlib import Path

import fitz

from exam_bank.submissions.quiz_packet import parse_scan_filenames, run_quiz_packet


def _write_pdf(path: Path, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _write_question_bank(root: Path) -> None:
    path = root / "output" / "json" / "question_bank.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": "31summer22_q02",
                        "question_number": "2",
                        "canonical_paper_id": "31summer22",
                        "question_text": "2 (a) Expand (2 - x^2)^-2 in ascending powers of x up to x^4. (b) State the set of values of x.",
                        "canonical_mark_scheme_artifact": "pm3/pm3_2022_m22_31_ms_q02_markscheme.png",
                        "mark_scheme_block_ids": ["9709_m22_ms_31:q02"],
                        "mark_scheme_confidence_score": 0.95,
                        "mark_scheme_text": "2(a) expansion marks. 2(b) validity marks.",
                        "text_only_status": "ready",
                        "visual_curation_status": "review",
                        "question_text_trust": "high",
                        "notes": {
                            "source_pdf": "input/pastpapers/9709/2022/question_papers/9709_m22_qp_31.pdf",
                            "source_paper_code": "31",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_quiz_packet_creates_local_files_and_teacher_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    quiz_dir = tmp_path / "data" / "submissions" / "p3_quiz_2026_06_23"
    scans_dir = quiz_dir / "scans"
    assignment_id = "p3_quiz_2026_06_23"
    _write_pdf(
        quiz_dir / "assignment.pdf",
        [
            "4\n2\n(a)\nExpand (2 - x^2)^-2 in ascending powers of x, up to x^4.\n(b)\nState the set of values of x.\n9709/31/M/J/22",
            "5\n9709/31/M/J/22\n© UCLES 2022\n[Turn over\n(b) Hence solve the equation\nx\n5\n2\n[4]",
        ],
    )
    _write_pdf(scans_dir / f"Bill_{assignment_id}.pdf", ["Bill answer text"])
    _write_pdf(scans_dir / f"Sally_{assignment_id}.pdf", ["Sally answer text"])
    _write_question_bank(tmp_path)

    result = run_quiz_packet(
        quiz_dir=quiz_dir,
        course_id="p3",
        output_root=Path("output/submissions"),
        reports_root=Path("reports/submissions"),
    )

    assignment = json.loads((quiz_dir / "assignment.json").read_text(encoding="utf-8"))
    assert assignment["assignment_id"] == assignment_id
    assert assignment["course_id"] == "p3"
    assert assignment["due_at"] is None
    assert assignment["assignment_pdf"] == "assignment.pdf"

    with (quiz_dir / "roster.csv").open("r", encoding="utf-8", newline="") as handle:
        roster_rows = list(csv.DictReader(handle))
    assert roster_rows == [
        {
            "student_id": "bill",
            "class_id": "local_class",
            "display_name": "Bill",
            "email": "",
            "submitted": "true",
            "source_file": f"Bill_{assignment_id}.pdf",
        },
        {
            "student_id": "sally",
            "class_id": "local_class",
            "display_name": "Sally",
            "email": "",
            "submitted": "true",
            "source_file": f"Sally_{assignment_id}.pdf",
        },
    ]

    assert Path(result["manifest"]).is_file()
    assert Path(result["completion_report"]).is_file()
    assert Path(result["teacher_report"]).is_file()
    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 0
    assert result["email_sent"] is False

    matched = json.loads(Path(result["matched_questions"]).read_text(encoding="utf-8"))
    assert matched["detected_question_count"] == 1
    assert matched["records"][0]["match_status"] == "matched_confident"
    assert matched["records"][0]["matched_question_id"] == "31summer22_q02"
    assert matched["records"][0]["mark_scheme_status"] == "matched_confident"

    drafts = json.loads(Path(result["draft_grading"]).read_text(encoding="utf-8"))
    assert len(drafts["records"]) == 2
    assert {record["status"] for record in drafts["records"]} == {"blocked_needs_teacher_review"}
    assert all(record["score"] is None for record in drafts["records"])
    assert all(record["student_facing"] is False for record in drafts["records"])

    report_text = Path(result["teacher_report"]).read_text(encoding="utf-8")
    assert "Email sent" in report_text
    assert "Matched questions JSON" in report_text


def test_existing_roster_preserves_extra_columns_and_missing_students(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assignment_id = "p3_quiz_2026_06_23"
    quiz_dir = tmp_path / "data" / "submissions" / assignment_id
    scans_dir = quiz_dir / "scans"
    _write_pdf(quiz_dir / "assignment.pdf", ["4\n2\nQuestion text\n9709/31/M/J/22"])
    _write_pdf(scans_dir / f"Bill_{assignment_id}.pdf", ["Bill answer text"])
    (quiz_dir / "roster.csv").write_text(
        "student_id,class_id,display_name,email,submitted,source_file,notes\n"
        "bill,local_class,Bill,bill@example.invalid,false,,keep me\n"
        "missing,local_class,Missing Student,missing@example.invalid,true,old.pdf,still here\n",
        encoding="utf-8",
    )
    _write_question_bank(tmp_path)

    run_quiz_packet(
        quiz_dir=quiz_dir,
        course_id="p3",
        output_root=Path("output/submissions"),
        reports_root=Path("reports/submissions"),
    )

    with (quiz_dir / "roster.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["email"] == "bill@example.invalid"
    assert rows[0]["notes"] == "keep me"
    assert rows[0]["submitted"] == "true"
    assert rows[1]["student_id"] == "missing"
    assert rows[1]["submitted"] == "false"
    assert rows[1]["notes"] == "still here"


def test_duplicate_scan_names_get_stable_generated_ids() -> None:
    assignment_id = "p3_quiz_2026_06_23"
    parsed = parse_scan_filenames(
        [
            Path(f"Bill_{assignment_id}.pdf"),
            Path(f"Bill-{assignment_id}.pdf"),
        ],
        assignment_id=assignment_id,
    )

    assert [scan.student_id for scan in parsed] == ["bill", "bill_2"]
    assert any("duplicate_student_id:bill" in warning for scan in parsed for warning in scan.warnings)
