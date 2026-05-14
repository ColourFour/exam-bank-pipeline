from __future__ import annotations

import json
from pathlib import Path

import pytest

from exam_bank.advisory_evidence.extraction import extract_native_pdf_text
from exam_bank.advisory_evidence.inventory import write_all_inventories
from exam_bank.advisory_evidence.linking import build_all_links
from exam_bank.advisory_evidence.parsing import parse_examiner_report_text, parse_grade_threshold_text
from exam_bank.advisory_evidence.reports import build_review_reports
from exam_bank.advisory_evidence.sidecar import build_final_sidecar
from exam_bank.advisory_evidence.signals import build_examiner_difficulty, build_grade_threshold_context, build_topic_evidence
from exam_bank.advisory_evidence.validation import validate_advisory_evidence
from exam_bank.document_metadata import parse_filename_metadata


def _write_pdf(path: Path, text: str) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_metadata_classifies_grade_thresholds() -> None:
    metadata = parse_filename_metadata("9709 Mathematics June 2024 Grade Thresholds.pdf")

    assert metadata.document_type == "grade_thresholds"
    assert metadata.syllabus == "9709"
    assert metadata.year == "2024"
    assert metadata.session == "MayJune"


def test_inventory_flags_duplicate_identities_and_dry_run_does_not_write(tmp_path: Path) -> None:
    reports = tmp_path / "examiner_reports"
    thresholds = tmp_path / "grade_thresholds"
    reports.mkdir()
    thresholds.mkdir()
    _write_pdf(reports / "9709 Mathematics March 2023 Examiner Report.pdf", "Paper 9709/12\nComments on specific questions\nQuestion 1\nOK")
    _write_pdf(reports / "9709 Mathematics March 2023 Examiner Report (1).pdf", "Paper 9709/12\nComments on specific questions\nQuestion 1\nOK")
    _write_pdf(thresholds / "9709 Mathematics June 2024 Grade Thresholds.pdf", "Component 31\n75\n48\n39\n31\n23\n14")

    summary = write_all_inventories(
        output_root=tmp_path / "advisory",
        examiner_reports_dir=reports,
        grade_thresholds_dir=thresholds,
        dry_run=True,
    )

    assert not (tmp_path / "advisory").exists()
    examiner_docs = summary["inventories"]["examiner_report"]["documents"]
    assert len(examiner_docs) == 2
    assert all(doc["duplicate_identity"] for doc in examiner_docs)
    assert summary["inventories"]["grade_thresholds"]["documents"][0]["document_type"] == "grade_thresholds"


def test_native_extraction_preserves_pages_and_method(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "9709 Mathematics June 2025 Examiner Report.pdf", "Paper 9709/31\nQuestion 1\nComplex numbers")
    payload = extract_native_pdf_text(
        {
            "source_path": str(pdf),
            "syllabus": "9709",
            "year": "2025",
            "session": "MayJune",
            "document_type": "examiner_report",
        }
    )

    assert payload["extraction_method"] == "native_pdf_text"
    assert payload["page_count"] == 1
    assert payload["page_text"][0]["text_length"] > 0
    assert "Paper 9709/31" in payload["raw_text"]


def test_parsers_capture_examiner_comments_and_threshold_rows() -> None:
    examiner = parse_examiner_report_text(
        {
            "source_path": "report.pdf",
            "syllabus": "9709",
            "year": "2025",
            "session": "MayJune",
            "session_key": "9709_2025_MayJune",
            "raw_text": """
Paper 9709/31
Pure Mathematics 3

Key messages
Use exact values.

General comments
The paper proved challenging.

Comments on specific questions

Question 1
Complex numbers in polar form were well answered.

Question 2
There were too few candidates for a meaningful report.
""",
        }
    )
    thresholds = parse_grade_threshold_text(
        {
            "source_path": "thresholds.pdf",
            "syllabus": "9709",
            "year": "2024",
            "session": "MayJune",
            "raw_text": """
Component 31
75
48
39
31
23
14
Option
CX
250
11, 31, 51, 61
201
175
149
117
85
53
""",
        }
    )

    assert examiner["components"][0]["component"] == "31"
    assert examiner["components"][0]["questions"][0]["question_number"] == 1
    assert examiner["components"][0]["questions"][1]["evidence_level"] == "none"
    assert thresholds["components"][0]["thresholds"]["A"] == 48
    assert thresholds["options"][0]["option"] == "CX"
    assert thresholds["options"][0]["thresholds"]["A*"] == 201


def test_linking_topic_difficulty_context_validation_reports_and_sidecar(tmp_path: Path) -> None:
    question_bank = _write_json(
        tmp_path / "question_bank.json",
        {
            "questions": [
                {
                    "question_id": "31summer25_q01",
                    "question_number": "1",
                    "notes": {
                        "source_pdf": "input/question_papers/9709 Mathematics June 2025 Question Paper  31.pdf",
                        "source_paper_code": "31",
                    },
                }
            ]
        },
    )
    parsed_root = tmp_path / "advisory" / "parsed"
    _write_json(
        parsed_root / "examiner_reports" / "report.json",
        {
            "schema": "exam_bank.advisory_evidence.examiner_report_parsed.v1",
            "source_path": "input/examiner_reports/9709 Mathematics June 2025 Examiner Report.pdf",
            "syllabus": "9709",
            "year": "2025",
            "session": "MayJune",
            "session_key": "9709_2025_MayJune",
            "components": [
                {
                    "component": "31",
                    "questions": [
                        {
                            "question_number": 1,
                            "comment_text": "Complex numbers in polar form proved difficult and few correct solutions were seen.",
                            "evidence_level": "normal",
                            "warnings": [],
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        parsed_root / "grade_thresholds" / "thresholds.json",
        {
            "schema": "exam_bank.advisory_evidence.grade_thresholds_parsed.v1",
            "source_path": "input/grade_thresholds/9709 Mathematics June 2025 Grade Thresholds.pdf",
            "syllabus": "9709",
            "year": "2025",
            "session": "MayJune",
            "session_key": "9709_2025_MayJune",
            "components": [{"component": "31", "max_raw_mark": 75, "thresholds": {"A": 48, "B": 39, "C": 31, "D": 23, "E": 14}, "warnings": []}],
            "options": [],
        },
    )

    links = build_all_links(question_bank_path=question_bank, parsed_root=parsed_root, output_root=tmp_path / "advisory")
    topic = build_topic_evidence(
        parsed_dir=parsed_root / "examiner_reports",
        links_path=tmp_path / "advisory/linking/examiner_report_question_links.json",
        output_path=tmp_path / "advisory/predictions/advisory_topic_evidence.v1.json",
    )
    difficulty = build_examiner_difficulty(
        parsed_dir=parsed_root / "examiner_reports",
        links_path=tmp_path / "advisory/linking/examiner_report_question_links.json",
        output_path=tmp_path / "advisory/predictions/advisory_examiner_report_difficulty.v1.json",
    )
    context = build_grade_threshold_context(
        parsed_dir=parsed_root / "grade_thresholds",
        output_path=tmp_path / "advisory/predictions/advisory_grade_threshold_context.v1.json",
    )
    sidecar = build_final_sidecar(advisory_root=tmp_path / "advisory", question_bank_path=question_bank, output_path=tmp_path / "advisory/question_bank.advisory_evidence.v1.json")
    validation = validate_advisory_evidence(advisory_root=tmp_path / "advisory", question_bank_path=question_bank)
    reports = build_review_reports(advisory_root=tmp_path / "advisory", output_dir=tmp_path / "advisory/reports")

    assert links["examiner_report"]["links"][0]["match_status"] == "linked"
    assert topic["records"][0]["topic_evidence"]["predicted_topic_ids"] == ["complex_numbers"]
    assert difficulty["records"][0]["examiner_report_difficulty"]["item_signal"] == "hard"
    assert context["contexts"][0]["component_context_label"] in {"paper_context_typical", "paper_context_unknown"}
    assert sidecar["records"][0]["advisory_evidence"]["examiner_report"]["available"] is True
    assert validation["ok"] is True
    assert len(reports["outputs"]) == 6


def test_validation_fails_invalid_topic_and_threshold_only_item_difficulty(tmp_path: Path) -> None:
    question_bank = _write_json(tmp_path / "question_bank.json", {"questions": [{"question_id": "q1"}]})
    root = tmp_path / "advisory"
    _write_json(
        root / "predictions" / "advisory_topic_evidence.v1.json",
        {
            "schema": "exam_bank.advisory_evidence.topic_evidence.v1",
            "records": [{"question_id": "q1", "topic_evidence": {"predicted_topic_ids": ["invented"], "confidence": "high"}}],
        },
    )
    _write_json(
        root / "predictions" / "advisory_examiner_report_difficulty.v1.json",
        {
            "schema": "exam_bank.advisory_evidence.examiner_report_difficulty.v1",
            "records": [
                {
                    "question_id": "q1",
                    "examiner_report_difficulty": {
                        "item_signal": "hard",
                        "confidence": "medium",
                        "evidence_level": "normal",
                        "evidence_sources": ["grade_threshold_context"],
                    },
                }
            ],
        },
    )

    report = validate_advisory_evidence(advisory_root=root, question_bank_path=question_bank)

    assert report["ok"] is False
    assert any("invalid_topic_id" in error for error in report["errors"])
    assert any("threshold_only_item_difficulty" in error for error in report["errors"])

