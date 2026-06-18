from datetime import datetime
import json
from pathlib import Path
import re

import pytest

from exam_bank import __version__
from exam_bank.config import AppConfig
from exam_bank.exporters import QUESTION_BANK_SCHEMA_NAME, QUESTION_BANK_SCHEMA_VERSION, export_records
from exam_bank.models import QuestionRecord
from exam_bank.output_layout import mark_scheme_image_output_path, paper_instance_id, question_image_output_path
from exam_bank.output_layout import (
    CANONICAL_LAYOUT_PROFILE,
    OCR_CANDIDATE_LAYOUT_PROFILE,
    default_asterion_export_path,
    default_triage_comparison_path,
    generated_output_contract,
    output_profile_for_root,
)


def _record() -> QuestionRecord:
    return QuestionRecord(
        source_pdf="input/question_papers/9709 Mathematics March 2021 Question paper  12.pdf",
        paper_name="9709_Mathematics_March_2021_Question_paper_12",
        question_number="1",
        full_question_label="1(a)-(b)",
        screenshot_path="output/p1/12summer21/questions/q01.png",
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
        difficulty_score=18,
        difficulty_band="easy",
        difficulty_score_scale="0-100",
        difficulty_features={"marks": {"marks": 3, "contribution": -0.8}},
        difficulty_review_flags=[],
        difficulty_model_version="local-difficulty-v1",
        marks=3,
        marks_if_available=3,
        page_numbers=[3, 4],
        review_flags=["markscheme_parent_label_match"],
        confidence=0.8,
        session="summer21",
        year="2021",
        component="12",
        source_paper_code="12",
        markscheme_image="output/p1/12summer21/mark_scheme/q01.png",
        markscheme_pages=[6],
        markscheme_mapping_status="pass",
        question_marks_total=3,
        markscheme_marks_total=3,
        question_subparts=["a", "b"],
        scope_quality_status="clean",
        text_source_profile="native_pdf",
        text_fidelity_status="clean",
        text_fidelity_flags=[],
        question_text_role="readable_text",
        question_text_trust="high",
        visual_required=False,
        visual_reason_flags=[],
        visual_curation_status="ready",
        text_only_status="ready",
        validation_status="pass",
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

    assert paper_instance_id("12", "March", "2021") == "12summer21"
    assert qp_path == tmp_path / "output" / "pm1" / "pm1_2021_m21_qp_q01_question.png"
    assert ms_path == tmp_path / "output" / "pm1" / "pm1_2021_m21_ms_q01_markscheme.png"


def test_export_records_writes_json_under_output_json_only(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    record = _record()
    record.screenshot_path = str(tmp_path / "output" / "p1" / "12summer21" / "questions" / "q01.png")
    record.markscheme_image = str(tmp_path / "output" / "p1" / "12summer21" / "mark_scheme" / "q01.png")

    json_path = export_records([record], config)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path == tmp_path / "output" / "json" / "question_bank.json"
    assert json_path.exists()
    assert payload["schema_name"] == QUESTION_BANK_SCHEMA_NAME
    assert payload["schema_version"] == QUESTION_BANK_SCHEMA_VERSION
    assert payload["record_count"] == 1
    question = payload["questions"][0]
    assert question["paper"] == "12summer21"
    assert question["canonical_paper_id"] == "12summer21"
    assert question["canonical_session"] == "summer21"
    assert question["canonical_year_folder"] == "2021"
    assert question["question_image_paths"] == ["pm1/pm1_2021_s21_qp_q01_question.png"]
    assert question["mark_scheme_image_paths"] == ["pm1/pm1_2021_s21_ms_q01_markscheme.png"]
    assert question["canonical_question_artifact"] == "pm1/pm1_2021_s21_qp_q01_question.png"
    assert question["question_image_path"] == "pm1/pm1_2021_s21_qp_q01_question.png"
    assert question["mark_scheme_image_path"] == "pm1/pm1_2021_s21_ms_q01_markscheme.png"
    assert question["question_text_role"] == "readable_text"
    assert question["question_text_trust"] == "high"
    assert question["difficulty"] == "easy"
    assert question["difficulty_score"] == 18
    assert question["difficulty_band"] == "easy"
    assert question["visual_required"] is False
    assert question["visual_reason_flags"] == []
    assert question["visual_curation_status"] == "ready"
    assert question["text_only_status"] == "ready"
    assert question["notes"]["topic_confidence"] == "high"
    assert question["notes"]["topic_trust_status"] == "normal"
    assert question["notes"]["difficulty_confidence"] == "high"
    assert question["notes"]["difficulty_score"] == 18
    assert question["notes"]["difficulty_score_scale"] == "0-100"
    assert question["notes"]["difficulty_features"]["marks"]["marks"] == 3
    assert question["notes"]["difficulty_review_flags"] == []
    assert question["notes"]["difficulty_model_version"] == "local-difficulty-v1"
    assert question["notes"]["scope_quality_status"] == "clean"
    assert question["notes"]["question_crop_diagnostics"] == {}
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


def test_export_records_fails_when_artifact_path_session_disagrees_with_metadata(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    record = _record()
    record.session = "summer21"
    record.year = "2021"
    record.screenshot_path = str(tmp_path / "output" / "pm1" / "pm1_2021_w21_qp_q01_question.png")
    record.markscheme_image = str(tmp_path / "output" / "pm1" / "pm1_2021_s21_ms_q01_markscheme.png")

    with pytest.raises(ValueError, match="artifact path session mismatch"):
        export_records([record], config)


def test_question_record_exposes_grouped_internal_state_without_changing_flat_fields() -> None:
    record = _record()

    assert record.extraction.question_number == record.question_number
    assert record.extraction.combined_question_text == record.combined_question_text
    assert record.classification.topic == record.topic
    assert record.classification.question_level_topic == record.question_level_topic
    assert record.classification.difficulty_score == record.difficulty_score
    assert record.classification.difficulty_features == record.difficulty_features
    assert record.images.screenshot_path == record.screenshot_path
    assert record.mark_scheme.image_path == record.markscheme_image
    assert record.mark_scheme.mapping_status == record.markscheme_mapping_status
    assert record.validation.topic_trust_status == record.topic_trust_status
    assert record.paper_metadata.component == record.component


def test_question_bank_export_contract_includes_required_metadata_and_question_fields(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    record = _record()
    record.screenshot_path = str(tmp_path / "output" / "p1" / "12summer21" / "questions" / "q01.png")
    record.markscheme_image = str(tmp_path / "output" / "p1" / "12summer21" / "mark_scheme" / "q01.png")

    json_path = export_records([record], config)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert set(payload) == {"schema_name", "schema_version", "record_count", "run_manifest", "questions"}
    assert payload["schema_name"] == QUESTION_BANK_SCHEMA_NAME
    assert payload["schema_version"] == QUESTION_BANK_SCHEMA_VERSION
    assert payload["record_count"] == len(payload["questions"]) == 1
    manifest = payload["run_manifest"]
    assert {
        "schema_version",
        "generated_at",
        "run_id",
        "pipeline_version",
        "git_commit",
        "model_versions",
        "ocr_engine_version",
        "input_manifest_sha256",
        "artifact_root",
        "output_layout",
        "qa_summary",
    } == set(manifest)
    assert manifest["schema_version"] == 1
    assert datetime.fromisoformat(manifest["generated_at"])
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{12}", manifest["run_id"])
    assert manifest["pipeline_version"] == __version__
    assert isinstance(manifest["git_commit"], str)
    assert manifest["model_versions"] == {
        "topic_classifier": "local-topic-heuristics-v1",
        "difficulty_classifier": ["local-difficulty-v1"],
        "openai_classifier": "",
    }
    assert manifest["ocr_engine_version"] == ""
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["input_manifest_sha256"])
    assert manifest["artifact_root"] == str(tmp_path / "output")
    assert manifest["output_layout"] == {"version": 1, "profile": CANONICAL_LAYOUT_PROFILE}
    assert manifest["qa_summary"]["record_count"] == 1
    assert manifest["qa_summary"]["paper_family_counts"] == {"pm1": 1}
    assert manifest["qa_summary"]["validation_status_counts"] == {"pass": 1}
    assert manifest["qa_summary"]["mapping_status_counts"] == {"pass": 1}
    assert manifest["qa_summary"]["scope_quality_status_counts"] == {"clean": 1}
    assert manifest["qa_summary"]["text_fidelity_status_counts"] == {"clean": 1}
    assert manifest["qa_summary"]["visual_curation_status_counts"] == {"ready": 1}
    assert manifest["qa_summary"]["text_only_status_counts"] == {"ready": 1}
    assert manifest["qa_summary"]["question_crop_confidence_counts"] == {"high": 1}
    assert manifest["qa_summary"]["ocr_summary"] == {
        "ran_count": 0,
        "selected_count": 0,
        "engine_counts": {},
    }
    assert manifest["qa_summary"]["artifact_path_counts"] == {
        "missing_question_image_path": 0,
        "missing_mark_scheme_image_path": 0,
    }

    question = payload["questions"][0]
    assert {
        "question_id",
        "paper",
        "canonical_paper_id",
        "canonical_session",
        "canonical_year_folder",
        "paper_family",
        "question_number",
        "canonical_question_artifact",
        "question_image_path",
        "mark_scheme_image_path",
        "question_text",
        "question_text_role",
        "question_text_trust",
        "ocr_ran",
        "ocr_engine",
        "ocr_text",
        "ocr_text_trust",
        "ocr_failure_reason",
        "visual_required",
        "visual_reason_flags",
        "visual_curation_status",
        "text_only_status",
        "mark_scheme_text",
        "question_solution_marks",
        "difficulty",
        "difficulty_score",
        "difficulty_band",
        "subparts",
        "subparts_solution_marks",
        "question_image_paths",
        "mark_scheme_image_paths",
        "page_refs",
        "topic",
        "notes",
    }.issubset(question)
    assert question["question_id"] == "12summer21_q01"
    assert question["paper"] == "12summer21"
    assert question["canonical_paper_id"] == "12summer21"
    assert question["canonical_session"] == "summer21"
    assert question["canonical_year_folder"] == "2021"
    assert question["paper_family"] == "pm1"
    assert question["question_image_paths"] == ["pm1/pm1_2021_s21_qp_q01_question.png"]
    assert question["mark_scheme_image_paths"] == ["pm1/pm1_2021_s21_ms_q01_markscheme.png"]
    assert set(question["page_refs"]) == {"question", "mark_scheme"}

    assert {
        "source_pdf",
        "mark_scheme_source_pdf",
        "source_paper_code",
        "full_question_label",
        "topic_confidence",
        "topic_uncertain",
        "topic_trust_status",
        "difficulty",
        "difficulty_confidence",
        "difficulty_evidence",
        "difficulty_uncertain",
        "difficulty_score",
        "difficulty_score_scale",
        "difficulty_features",
        "difficulty_review_flags",
        "difficulty_model_version",
        "mapping_status",
        "mapping_failure_reason",
        "scope_quality_status",
        "question_crop_confidence",
        "text_source_profile",
        "text_fidelity_status",
        "text_fidelity_flags",
        "question_text_role",
        "question_text_trust",
        "ocr_ran",
        "ocr_engine",
        "ocr_text_trust",
        "ocr_failure_reason",
        "ocr_text_role",
        "text_candidate_source",
        "native_text_score",
        "ocr_text_score",
        "selected_text_score",
        "text_candidate_decision",
        "text_candidate_decision_reasons",
        "ocr_selected",
        "ocr_rejected_reasons",
        "visual_required",
        "visual_reason_flags",
        "visual_curation_status",
        "text_only_status",
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


def test_generated_output_contract_paths_are_single_source_of_truth(tmp_path: Path) -> None:
    contract = generated_output_contract(tmp_path / "output")

    assert contract.run_root("20260513T010000Z-standard-abcd1234") == (
        tmp_path / "output" / "runs" / "20260513T010000Z-standard-abcd1234"
    )
    assert contract.canonical_current_dir() == tmp_path / "output" / "current"
    assert contract.ocr_candidate_root("iteration_005") == tmp_path / "output" / "candidates" / "ocr" / "iteration_005"
    assert contract.triage_iteration_dir("iteration_005") == tmp_path / "output" / "triage" / "iteration_005"
    assert default_triage_comparison_path(
        tmp_path / "output" / "triage" / "iteration_005",
        "comparison.auto-iteration-005.json",
    ) == tmp_path / "output" / "triage" / "iteration_005" / "comparisons" / "comparison.auto-iteration-005.json"
    assert default_asterion_export_path(
        tmp_path / "output" / "json" / "question_bank.json",
        "asterion_question_bank_v1.json",
    ) == tmp_path / "output" / "asterion" / "exports" / "latest" / "asterion_question_bank_v1.json"


def test_output_profile_detection_preserves_legacy_ocr_candidate_root() -> None:
    assert output_profile_for_root("output") == CANONICAL_LAYOUT_PROFILE
    assert output_profile_for_root("output_ocr_candidate") == OCR_CANDIDATE_LAYOUT_PROFILE
    assert output_profile_for_root("output/candidates/ocr/latest") == OCR_CANDIDATE_LAYOUT_PROFILE
