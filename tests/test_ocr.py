from __future__ import annotations

import json
from pathlib import Path

import pytest

from exam_bank.config import AppConfig
from exam_bank.exporters import export_records
from exam_bank.models import QuestionRecord
from exam_bank.ocr import disabled_ocr_result, run_question_crop_ocr


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
