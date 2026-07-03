from __future__ import annotations

from pathlib import Path

from exam_bank.config import AppConfig
from exam_bank.mark_scheme_models import MarkSchemeWord
from exam_bank.mark_schemes import (
    _build_formulaic_mark_scheme_blocks,
    _detect_formulaic_left_margin_mark_scheme_anchors,
)
from exam_bank.models import BoundingBox, PageLayout, TextBlock


def _word(page_number: int, text: str, x0: float, y0: float, x1: float | None = None, y1: float | None = None) -> MarkSchemeWord:
    return MarkSchemeWord(
        page_number=page_number,
        text=text,
        bbox=BoundingBox(x0, y0, x1 or x0 + max(8, len(text) * 5), y1 or y0 + 10),
    )


def test_formulaic_anchor_detection_ignores_notes_marks_tokens_and_totals() -> None:
    config = AppConfig()
    layouts = [
        PageLayout(
            page_number=1,
            width=595,
            height=842,
            blocks=[TextBlock(1, "Mark Scheme Notes\nAbbreviations", BoundingBox(45, 80, 320, 140))],
        ),
        PageLayout(
            page_number=2,
            width=595,
            height=842,
            blocks=[
                TextBlock(2, "1", BoundingBox(45, 120, 54, 130)),
                TextBlock(2, "M1", BoundingBox(460, 130, 475, 140)),
                TextBlock(2, "[4]", BoundingBox(490, 145, 508, 155)),
                TextBlock(2, "2", BoundingBox(45, 260, 54, 270)),
            ],
        ),
    ]
    words = {
        1: [_word(1, "1", 45, 120)],
        2: [
            _word(2, "1", 45, 120),
            _word(2, "M1", 460, 130),
            _word(2, "[4]", 490, 145),
            _word(2, "2", 45, 260),
        ],
    }

    anchors = _detect_formulaic_left_margin_mark_scheme_anchors(layouts, words, config)

    assert [anchor.question_number for anchor in anchors] == ["1", "2"]
    assert [anchor.page_number for anchor in anchors] == [2, 2]


def test_formulaic_block_regions_stop_at_next_top_level_anchor() -> None:
    config = AppConfig()
    layouts = [
        PageLayout(
            page_number=1,
            width=595,
            height=842,
            blocks=[
                TextBlock(1, "1", BoundingBox(45, 120, 54, 130)),
                TextBlock(1, "answer line", BoundingBox(90, 126, 260, 140)),
                TextBlock(1, "2", BoundingBox(45, 260, 54, 270)),
                TextBlock(1, "next answer", BoundingBox(90, 266, 260, 280)),
            ],
        )
    ]
    words = {
        1: [
            _word(1, "1", 45, 120),
            _word(1, "answer", 90, 126),
            _word(1, "line", 140, 126),
            _word(1, "2", 45, 260),
            _word(1, "next", 90, 266),
        ]
    }

    blocks = _build_formulaic_mark_scheme_blocks(
        Path("9709_s15_ms_31.pdf"),
        layouts,
        words,
        config,
        ["1"],
        question_marks={"1": None},
        question_subparts={"1": []},
    )

    block = blocks["1"]
    assert block.method == "formulaic_left_margin_anchor_fallback"
    assert len(block.regions) == 1
    assert block.regions[0].bbox.y0 <= 120
    assert block.regions[0].bbox.y1 < 260
