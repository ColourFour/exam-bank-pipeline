from __future__ import annotations

import json
from pathlib import Path

import pytest

from exam_bank.config import AppConfig
from exam_bank.exporters import export_records
from exam_bank.models import QuestionRecord
from exam_bank.ocr import disabled_ocr_result, score_text_candidate, select_text_candidate, run_question_crop_ocr
from exam_bank.trust import derive_question_text_semantics, derive_text_only_status, visual_reason_flags


def _record(tmp_path: Path, *, ocr_ran: bool = False, ocr_text: str = "", ocr_text_trust: str = "unusable", ocr_failure_reason: str = "disabled") -> QuestionRecord:
    output_root = tmp_path / "output"
    return QuestionRecord(
        source_pdf="input/question_papers/9709 Mathematics March 2021 Question paper  12.pdf",
        paper_name="9709_Mathematics_March_2021_Question_paper_12",
        question_number="1",
        full_question_label="1",
        screenshot_path=str(output_root / "p1" / "12spring21" / "questions" / "q01.png"),
        combined_question_text="Solve the inequality 5 3 x x - - 1 3 6.",
        body_text_raw="Solve the inequality 5 3 x x - - 1 3 6.",
        body_text_normalized="Solve the inequality 5 3 x x - - 1 3 6.",
        math_lines=[],
        diagram_text=[],
        extraction_quality_score=0.75,
        extraction_quality_flags=[],
        part_texts=[],
        answer_text="",
        paper_family="P1",
        source_paper_family="P1",
        inferred_paper_family="P1",
        paper_family_confidence="high",
        topic="algebra",
        subtopic="general",
        topic_confidence="medium",
        topic_evidence="fixture",
        secondary_topics=[],
        topic_uncertain=False,
        difficulty="average",
        difficulty_confidence="medium",
        difficulty_evidence="fixture",
        difficulty_uncertain=False,
        marks=4,
        marks_if_available=4,
        page_numbers=[1],
        review_flags=[],
        confidence=0.7,
        session="March",
        year="2021",
        component="12",
        source_paper_code="12",
        markscheme_image=str(output_root / "p1" / "12spring21" / "mark_scheme" / "q01.png"),
        markscheme_mapping_status="pass",
        question_text_role="untrusted_math_text",
        question_text_trust="low",
        visual_required=True,
        visual_reason_flags=["contains_inequality_or_region_prompt", "contains_math_text_corruption"],
        visual_curation_status="ready",
        text_only_status="fail",
        ocr_ran=ocr_ran,
        ocr_engine="tesseract" if ocr_ran else "",
        ocr_text=ocr_text,
        ocr_text_trust=ocr_text_trust,
        ocr_failure_reason=ocr_failure_reason,
    )


def test_ocr_disabled_by_default_and_not_attempted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = AppConfig()
    called = False

    def fake_ocr(_image_path: Path, _config: AppConfig) -> str:
        nonlocal called
        called = True
        return "should not run"

    monkeypatch.setattr("exam_bank.ocr._tesseract_image_to_string", fake_ocr)
    result = run_question_crop_ocr(tmp_path / "q01.png", config)

    assert config.ocr.enabled is False
    assert called is False
    assert result == disabled_ocr_result()


def test_ocr_enabled_success_returns_low_trust_for_math_heavy_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = AppConfig()
    config.ocr.enabled = True
    monkeypatch.setattr(
        "exam_bank.ocr._tesseract_image_to_string",
        lambda _image_path, _config: "Solve the inequality 5 3 x x - - 1 3 6",
    )

    result = run_question_crop_ocr(tmp_path / "q01.png", config)

    assert result.ocr_ran is True
    assert "tesseract" in result.ocr_engine
    assert result.ocr_text == "Solve the inequality 5 3 x x - - 1 3 6"
    assert result.ocr_text_trust in {"low", "unusable"}
    assert result.ocr_text_role == "untrusted_math_text"
    assert result.ocr_failure_reason == ""


def test_ocr_enabled_failure_is_captured_without_crashing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = AppConfig()
    config.ocr.enabled = True

    def fail(_image_path: Path, _config: AppConfig) -> str:
        raise RuntimeError("missing tesseract")

    monkeypatch.setattr("exam_bank.ocr._tesseract_image_to_string", fail)

    result = run_question_crop_ocr(tmp_path / "q01.png", config)

    assert result.ocr_ran is True
    assert result.ocr_text == ""
    assert result.ocr_text_trust == "unusable"
    assert "missing tesseract" in result.ocr_failure_reason


def test_exported_ocr_fields_do_not_override_canonical_question_artifact(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    record = _record(
        tmp_path,
        ocr_ran=True,
        ocr_text="x x 4 d 2 3 7 12 x -3 y",
        ocr_text_trust="low",
        ocr_failure_reason="",
    )

    json_path = export_records([record], config)
    question = json.loads(json_path.read_text(encoding="utf-8"))["questions"][0]

    assert question["canonical_question_artifact"] == "p1/12spring21/questions/q01.png"
    assert question["question_image_path"] == "p1/12spring21/questions/q01.png"
    assert question["question_text"] == record.combined_question_text
    assert question["ocr_ran"] is True
    assert question["ocr_engine"] == "tesseract"
    assert question["ocr_text"] == "x x 4 d 2 3 7 12 x -3 y"
    assert question["ocr_text_trust"] == "low"
    assert question["notes"]["ocr_failure_reason"] == ""


def test_candidate_selection_selects_ocr_when_native_has_merged_prose_and_control_garbage() -> None:
    native = "7 Sharma\x03knowsthatshehas3tinsofcarrots,2tinsofpeasand2tinsofsweetcorn. (a) Find the probability. [3]"
    ocr = (
        "7 Sharma knows that she has 3 tins of carrots, 2 tins of peas and 2 tins of sweetcorn. "
        "(a) Find the probability that she opens a tin of carrots. [3]"
    )

    decision = select_text_candidate(
        native_text=native,
        ocr_text=ocr,
        expected_question_number="7",
        expected_subparts=["a"],
        scope_quality_status="clean",
    )

    assert decision.ocr_selected is True
    assert decision.text_candidate_source == "ocr"
    assert decision.selected_text == ocr
    assert decision.ocr_text_score > decision.native_text_score
    assert "ocr_score_clear_margin" in decision.text_candidate_decision_reasons


def test_candidate_selection_retains_native_when_ocr_has_symbol_garbage() -> None:
    native = "2 Find the value of x for which 3(2 - x) = 12. Give your answer in exact form. [4]"
    ocr = "2 Find the value @@@ x ??? 3(2 - x) == ||||. Give answer?? [4]"

    decision = select_text_candidate(
        native_text=native,
        ocr_text=ocr,
        expected_question_number="2",
        expected_subparts=[],
        scope_quality_status="clean",
    )

    assert decision.ocr_selected is False
    assert decision.text_candidate_source == "native"
    assert "ocr_not_clearly_better" in decision.ocr_rejected_reasons


def test_candidate_selection_rejects_page_furniture_or_barcode_text() -> None:
    native = "4 Find the area of the shaded region. [5]"
    ocr = "PUTT RT TT TR TT Cambridge UCLES 4 Find the area of the shaded region. [5]"

    decision = select_text_candidate(
        native_text=native,
        ocr_text=ocr,
        expected_question_number="4",
        expected_subparts=[],
        scope_quality_status="clean",
    )

    assert decision.ocr_selected is False
    assert "page_furniture_or_header_text" in decision.ocr_rejected_reasons


def test_candidate_selection_rejects_next_question_contamination() -> None:
    native = "5 Find the probability that more than 9 students play at least one instrument. [3]"
    ocr = (
        "5 Find the probability that more than 9 students play at least one instrument. [3] "
        "6 Given that X is normally distributed, find the mean. [4]"
    )

    decision = select_text_candidate(
        native_text=native,
        ocr_text=ocr,
        expected_question_number="5",
        expected_subparts=[],
        scope_quality_status="clean",
    )

    assert decision.ocr_selected is False
    assert "next_question_contamination" in decision.ocr_rejected_reasons


def test_candidate_selection_rejects_ocr_that_loses_expected_mark_brackets() -> None:
    native = "6 (a) Find the power developed by the cyclist. [3] (b) Find the distance travelled. [4]"
    ocr = "6 (a) Find the power developed by the cyclist. (b) Find the distance travelled."

    decision = select_text_candidate(
        native_text=native,
        ocr_text=ocr,
        expected_question_number="6",
        expected_subparts=["a", "b"],
        scope_quality_status="clean",
    )

    assert decision.ocr_selected is False
    assert "ocr_lost_mark_brackets" in decision.ocr_rejected_reasons


def test_selected_ocr_math_text_remains_text_only_gated() -> None:
    native = "6 \x03≤ iveyouranswersintheformFindthecomplexnumberswxwhich+iywherexandyarereal."
    ocr = (
        "6 Find the complex numbers w which satisfy the equation w* + 2iw* = |. "
        "Give your answers in the form x + iy, where x and y are real. [6]"
    )

    decision = select_text_candidate(
        native_text=native,
        ocr_text=ocr,
        expected_question_number="6",
        expected_subparts=[],
        scope_quality_status="clean",
    )

    assert decision.ocr_selected is True
    visual_flags = visual_reason_flags(
        question_text=decision.selected_text,
        extraction_quality_flags=["flattened_display_math", "heavy_math_density"],
        review_flags=["ocr_question_text"],
        question_structure_detected={},
        text_source_profile="ocr",
    )
    role, trust, _visual_required = derive_question_text_semantics(
        question_text=decision.selected_text,
        text_fidelity_status="degraded",
        visual_reason_flags=visual_flags,
    )

    assert derive_text_only_status(
        validation_status="pass",
        scope_quality_status="clean",
        question_text_role=role,
        question_text_trust=trust,
    ) != "ready"


def test_candidate_score_exposes_reasons_and_rejections() -> None:
    score = score_text_candidate(
        "PUTT RT TT Cambridge 4 Find x. [2]",
        source="ocr",
        expected_question_number="4",
        expected_subparts=[],
        expected_mark_count=1,
    )

    assert score.score < 0
    assert "page_furniture_or_header_text" in score.reasons
    assert "page_furniture_or_header_text" in score.rejection_reasons


def test_exported_notes_include_text_candidate_decision_metadata(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    record = _record(
        tmp_path,
        ocr_ran=True,
        ocr_text="1 Find x. [3]",
        ocr_text_trust="medium",
        ocr_failure_reason="",
    )
    record.combined_question_text = "1 Find x. [3]"
    record.text_candidate_source = "ocr"
    record.native_text_score = 12
    record.ocr_text_score = 42
    record.selected_text_score = 42
    record.text_candidate_decision = "ocr_selected"
    record.text_candidate_decision_reasons = ["ocr_score_clear_margin", "readable_prose_spacing"]
    record.ocr_selected = True
    record.ocr_rejected_reasons = []

    json_path = export_records([record], config)
    question = json.loads(json_path.read_text(encoding="utf-8"))["questions"][0]

    assert question["question_text"] == "1 Find x. [3]"
    assert question["notes"]["text_candidate_source"] == "ocr"
    assert question["notes"]["native_text_score"] == 12
    assert question["notes"]["ocr_text_score"] == 42
    assert question["notes"]["selected_text_score"] == 42
    assert question["notes"]["text_candidate_decision"] == "ocr_selected"
    assert question["notes"]["ocr_selected"] is True
    assert question["notes"]["ocr_rejected_reasons"] == []
