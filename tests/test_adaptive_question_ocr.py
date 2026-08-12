from pathlib import Path

import pytest

from exam_bank.config import AppConfig
from exam_bank.image_rendering import _native_text_needs_ocr, _run_adaptive_question_crop_ocr
from exam_bank.models import QuestionSpan


def _span(*, status: str = "pass") -> QuestionSpan:
    return QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="2",
        start_page=1,
        start_y=80,
        end_page=1,
        end_y=220,
        page_numbers=[1],
        blocks=[],
        full_question_label="2",
        validation_status=status,
        structure_detected={"mark_values_detected": [4], "subparts": []},
    )


def test_adaptive_ocr_skips_a_high_quality_native_prompt() -> None:
    config = AppConfig()
    text = "2 Find the value of x, showing all necessary working and give your answer exactly. [4]"

    assert _native_text_needs_ocr(_span(), text, config) is False


def test_adaptive_ocr_runs_for_sparse_or_failed_native_extraction() -> None:
    config = AppConfig()

    assert _native_text_needs_ocr(_span(), "x", config) is True
    assert _native_text_needs_ocr(_span(status="fail"), "2 Find the value of x with complete working. [4]", config) is True


def test_adaptive_and_force_all_strategies_control_tesseract(monkeypatch) -> None:
    config = AppConfig()
    config.ocr.enabled = True
    calls: list[Path] = []
    sentinel = object()
    monkeypatch.setattr(
        "exam_bank.image_rendering.run_question_crop_ocr",
        lambda path, seen_config: calls.append(Path(path)) or sentinel,
    )
    strong = "2 Find the value of x, showing all necessary working and give your answer exactly. [4]"

    skipped = _run_adaptive_question_crop_ocr(Path("q02.png"), config, _span(), strong)
    assert skipped.ocr_ran is False
    assert calls == []

    config.ocr.strategy = "always"
    assert _run_adaptive_question_crop_ocr(Path("q02.png"), config, _span(), strong) is sentinel
    assert calls == [Path("q02.png")]


def test_unknown_ocr_strategy_fails_closed() -> None:
    config = AppConfig()
    config.ocr.enabled = True
    config.ocr.strategy = "sometimes"

    with pytest.raises(ValueError, match="Unsupported OCR strategy"):
        _run_adaptive_question_crop_ocr(Path("q02.png"), config, _span(), "x")
