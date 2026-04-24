from pathlib import Path
import json

from exam_bank.config import AppConfig
from exam_bank.exporters import QUESTION_BANK_SCHEMA_NAME, QUESTION_BANK_SCHEMA_VERSION, export_records
from exam_bank.models import QuestionRecord
from exam_bank.output_layout import mark_scheme_image_output_path, paper_instance_id, question_image_output_path


def _record() -> QuestionRecord:
    return QuestionRecord(
        source_pdf="input/question_papers/9709 Mathematics March 2021 Question paper  12.pdf",
        paper_name="9709_Mathematics_March_2021_Question_paper_12",
        question_number="1",
        full_question_label="1(a)-(b)",
        screenshot_path="output/p1/12spring21/questions/q01.png",
        combined_question_text="Find x.",
        body_text_raw="Find x.",
        body_text_normalized="Find x.",
        math_lines=[],
        diagram_text=[],
        extraction_quality_score=0.95,
        extraction_quality_flags=[],
        part_texts=[],
        answer_text="x = 2",
        paper_family="P1",
        source_paper_family="P1",
        inferred_paper_family="P1",
        paper_family_confidence="high",
        topic="binomial_expansion",
        subtopic="general",
        topic_confidence="high",
        topic_evidence="fixture",
        secondary_topics=[],
        topic_uncertain=False,
        topic_trust_status="normal",
        difficulty="easy",
        difficulty_confidence="high",
        difficulty_evidence="fixture",
        difficulty_uncertain=False,
        marks=3,
        marks_if_available=3,
        page_numbers=[3, 4],
        review_flags=["markscheme_parent_label_match"],
        confidence=0.8,
        session="March",
        year="2021",
        component="12",
        source_paper_code="12",
        markscheme_image="output/p1/12spring21/mark_scheme/q01.png",
        markscheme_pages=[6],
        markscheme_mapping_status="pass",
        question_marks_total=3,
        markscheme_marks_total=3,
        question_subparts=["a", "b"],
        scope_quality_status="clean",
        text_source_profile="native_pdf",
        text_fidelity_status="clean",
        text_fidelity_flags=[],
        mark_scheme_source_pdf="input/mark_schemes/9709 Mathematics March 2021 Mark Scheme  12.pdf",
    )


def test_paper_first_image_paths_follow_family_paper_questions_and_mark_scheme_layout(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")

    qp_path = question_image_output_path(
        "input/question_papers/9709 Mathematics March 2021 Question paper  12.pdf",
        "1",
        config,
    )
    ms_path = mark_scheme_image_output_path(
        "input/mark_schemes/9709 Mathematics March 2021 Mark Scheme  12.pdf",
        "1",
        config,
    )

    assert paper_instance_id("12", "March", "2021") == "12spring21"
    assert qp_path == tmp_path / "output" / "p1" / "12spring21" / "questions" / "q01.png"
    assert ms_path == tmp_path / "output" / "p1" / "12spring21" / "mark_scheme" / "q01.png"


def test_export_records_writes_json_under_output_json_only(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    record = _record()
    record.screenshot_path = str(tmp_path / "output" / "p1" / "12spring21" / "questions" / "q01.png")
    record.markscheme_image = str(tmp_path / "output" / "p1" / "12spring21" / "mark_scheme" / "q01.png")

    json_path = export_records([record], config)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path == tmp_path / "output" / "json" / "question_bank.json"
    assert json_path.exists()
    assert payload["schema_name"] == QUESTION_BANK_SCHEMA_NAME
    assert payload["schema_version"] == QUESTION_BANK_SCHEMA_VERSION
    assert payload["record_count"] == 1
    question = payload["questions"][0]
    assert question["question_image_paths"] == ["p1/12spring21/questions/q01.png"]
    assert question["mark_scheme_image_paths"] == ["p1/12spring21/mark_scheme/q01.png"]
    assert question["notes"]["topic_confidence"] == "high"
    assert question["notes"]["topic_trust_status"] == "normal"
    assert question["notes"]["scope_quality_status"] == "clean"
    assert question["notes"]["text_source_profile"] == "native_pdf"
    assert question["notes"]["text_fidelity_status"] == "clean"
    assert question["notes"]["text_fidelity_flags"] == []
    assert "validation_status" in question["notes"]
    assert "question_structure_detected" in question["notes"]
    assert "paper_total_status" in question["notes"]
    assert "rescan_result" in question["notes"]
    assert "paper_total_before_rescan" in question["notes"]
    assert "paper_total_focus_questions" in question["notes"]
    assert not (tmp_path / "output" / "csv").exists()
    assert not (tmp_path / "output" / "review").exists()


def test_question_record_exposes_grouped_internal_state_without_changing_flat_fields() -> None:
    record = _record()

    assert record.extraction.question_number == record.question_number
    assert record.extraction.combined_question_text == record.combined_question_text
    assert record.classification.topic == record.topic
    assert record.classification.question_level_topic == record.question_level_topic
    assert record.images.screenshot_path == record.screenshot_path
    assert record.mark_scheme.image_path == record.markscheme_image
    assert record.mark_scheme.mapping_status == record.markscheme_mapping_status
    assert record.validation.topic_trust_status == record.topic_trust_status
    assert record.paper_metadata.component == record.component


def test_question_bank_export_contract_includes_required_metadata_and_question_fields(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    record = _record()
    record.screenshot_path = str(tmp_path / "output" / "p1" / "12spring21" / "questions" / "q01.png")
    record.markscheme_image = str(tmp_path / "output" / "p1" / "12spring21" / "mark_scheme" / "q01.png")

    json_path = export_records([record], config)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert set(payload) == {"schema_name", "schema_version", "record_count", "questions"}
    assert payload["schema_name"] == QUESTION_BANK_SCHEMA_NAME
    assert payload["schema_version"] == QUESTION_BANK_SCHEMA_VERSION
    assert payload["record_count"] == len(payload["questions"]) == 1

    question = payload["questions"][0]
    assert {
        "question_id",
        "paper",
        "paper_family",
        "question_number",
        "question_text",
        "mark_scheme_text",
        "question_solution_marks",
        "subparts",
        "subparts_solution_marks",
        "question_image_paths",
        "mark_scheme_image_paths",
        "page_refs",
        "topic",
        "notes",
    }.issubset(question)
    assert question["question_id"] == "12spring21_q01"
    assert question["paper"] == "12spring21"
    assert question["paper_family"] == "p1"
    assert question["question_image_paths"] == ["p1/12spring21/questions/q01.png"]
    assert question["mark_scheme_image_paths"] == ["p1/12spring21/mark_scheme/q01.png"]
    assert set(question["page_refs"]) == {"question", "mark_scheme"}

    assert {
        "source_pdf",
        "mark_scheme_source_pdf",
        "source_paper_code",
        "full_question_label",
        "topic_confidence",
        "topic_uncertain",
        "topic_trust_status",
        "mapping_status",
        "mapping_failure_reason",
        "scope_quality_status",
        "question_crop_confidence",
        "text_source_profile",
        "text_fidelity_status",
        "text_fidelity_flags",
        "mark_scheme_crop_confidence",
        "review_flags",
        "extraction_quality_score",
        "extraction_quality_flags",
        "validation_status",
        "validation_flags",
        "question_structure_detected",
        "mark_scheme_structure_detected",
        "paper_total_status",
        "rescan_result",
        "paper_total_before_rescan",
        "paper_total_focus_questions",
    }.issubset(question["notes"])
