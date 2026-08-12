import json
from pathlib import Path

import pytest

from exam_bank.config import AppConfig
from exam_bank.models import BoundingBox, TextBlock
from exam_bank.pdf_extract import (
    _dense_grid_graphic_box,
    _dense_grid_graphic_boxes,
    _dense_non_text_cluster,
    _extract_text_blocks,
    _group_spans_into_visual_lines,
    _is_legacy_pdf,
    _line_text_from_spans,
    _ocr_hint_graphics,
    _serialize_table_visual_lines,
    _should_run_sparse_lower_ocr,
    _sparse_lower_ocr_clip,
    _stacked_fraction_line_text,
)


def span(text: str, x0: float, y0: float, x1: float, y1: float, size: float = 10) -> dict:
    return {
        "text": text,
        "bbox": [x0, y0, x1, y1],
        "size": size,
        "font": "TestFont",
    }


def serialized_visual_text(lines: list[list[dict]]) -> list[str]:
    return [
        text_override
        if text_override is not None
        else _line_text_from_spans(line_spans)
        for line_spans, text_override in _serialize_table_visual_lines(lines)
    ]


def test_spans_are_grouped_by_visual_y_then_x_not_raw_order() -> None:
    raw_order = [
        span("second", 50, 40, 85, 50),
        span("[3]", 120, 10, 135, 20),
        span("Find", 50, 10, 72, 20),
        span("x", 76, 10, 82, 20),
        span("2", 83, 5, 88, 12, size=7),
        span(".", 89, 10, 92, 20),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)
    text_lines = [_line_text_from_spans(line) for line in lines]

    assert text_lines == ["Find x^{2}. [3]", "second"]


def test_nearby_y_offsets_stay_on_same_visual_line() -> None:
    raw_order = [
        span("+", 84, 11, 90, 21),
        span("1", 94, 13, 100, 23),
        span("x", 76, 9, 82, 19),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "x + 1"


def test_explicit_pdf_space_span_preserves_touching_word_boundary() -> None:
    raw_order = [
        span("The", 50, 10, 68, 20),
        span(" ", 68, 10, 71, 20),
        span("diagram", 71, 10, 108, 20),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "The diagram"


def test_overlapping_span_chain_does_not_merge_successive_printed_lines() -> None:
    raw_order = [
        span("first line", 50, 10, 120, 22),
        span("tall math", 125, 8, 165, 30),
        span("second line", 50, 24, 130, 36),
        span("third line", 50, 38, 120, 50),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert [_line_text_from_spans(line) for line in lines] == [
        "first line tall math",
        "second line",
        "third line",
    ]


def test_small_scripts_are_assigned_after_prose_baselines_are_established() -> None:
    raw_order = [
        span("The diagram shows a tractor.", 72.84, 267.80, 441.42, 279.18, 11.38),
        span(" ", 441.42, 267.80, 445.68, 279.18, 11.38),
        span("The graph consists of", 445.68, 267.80, 545.36, 279.18, 11.38),
        span("four straight line segments.", 72.84, 281.48, 200.10, 292.86, 11.38),
        span(" ", 200.10, 281.48, 204.36, 292.86, 11.38),
        span("The tractor passes a point", 204.36, 281.48, 323.55, 292.86, 11.38),
        span(" O", 323.55, 280.79, 335.00, 292.68, 11.89),
        span(" at time", 335.00, 281.48, 369.00, 292.86, 11.38),
        span(" t", 369.00, 280.79, 376.00, 292.68, 11.89),
        span(" =", 376.00, 279.33, 386.00, 290.33, 11.00),
        span(" 0 with speed", 386.00, 281.48, 447.10, 292.86, 11.38),
        span(" U", 447.10, 280.79, 458.80, 292.68, 11.89),
        span(" m s", 458.80, 281.48, 476.70, 292.86, 11.38),
        span("−", 476.64, 277.80, 481.68, 285.83, 8.03),
        span("1", 481.68, 279.37, 486.48, 287.68, 8.31),
        span(".", 486.48, 281.48, 490.00, 292.86, 11.38),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert [_line_text_from_spans(line) for line in lines] == [
        "The diagram shows a tractor. The graph consists of",
        "four straight line segments. The tractor passes a point O at time "
        "t = 0 with speed U m s^{-1}.",
    ]


@pytest.mark.parametrize("formula_is_upper", [True, False])
def test_standalone_relation_row_does_not_merge_with_prose(
    formula_is_upper: bool,
) -> None:
    upper_y, lower_y = (10, 24)
    formula_y = upper_y if formula_is_upper else lower_y
    prose_y = lower_y if formula_is_upper else upper_y
    raw_order = [
        span("ln(1 + x) = 2 ln x,", 220, formula_y, 350, formula_y + 20, 11.4),
        span(
            "giving your answer correct to 3 significant figures.",
            72,
            prose_y,
            310,
            prose_y + 20,
            11.4,
        ),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    expected = [
        "ln(1 + x) = 2 ln x,",
        "giving your answer correct to 3 significant figures.",
    ]
    if not formula_is_upper:
        expected.reverse()
    assert [_line_text_from_spans(line) for line in lines] == expected


def test_geometry_aligned_table_rows_gain_canonical_delimiters() -> None:
    lines = [
        [span("The results are shown in the table.", 50, 10, 230, 20, 11.4)],
        [
            span("Class interval", 100, 30, 165, 40, 11.4),
            span("1 − 5", 200, 30, 225, 40, 11.4),
            span("6 − 10", 250, 30, 285, 40, 11.4),
            span("11 − 20", 300, 30, 345, 40, 11.4),
        ],
        [
            span("Frequency", 100, 47, 150, 57, 11.4),
            span("4", 210, 47, 216, 57, 11.4),
            span("7", 264, 47, 270, 57, 11.4),
            span("9", 320, 47, 326, 57, 11.4),
        ],
    ]

    assert serialized_visual_text(lines) == [
        "The results are shown in the table.",
        "Class interval: 1-5, 6-10, 11-20.",
        "Frequency: 4, 7, 9.",
    ]


def test_wrapped_table_headers_are_joined_by_column_before_flattening() -> None:
    lines = [
        [span("The information is recorded in the table below.", 50, 10, 260, 20, 11.4)],
        [
            span("Number of", 200, 30, 250, 40, 11.4),
            span("Mean price", 300, 30, 350, 40, 11.4),
            span("Standard", 400, 30, 450, 40, 11.4),
        ],
        [
            span("items", 205, 42, 245, 52, 11.4),
            span("($)", 315, 42, 335, 52, 11.4),
            span("deviation ($)", 390, 42, 450, 52, 11.4),
        ],
        [
            span("Shop A", 100, 59, 150, 69, 11.4),
            span("30", 215, 59, 225, 69, 11.4),
            span("1500", 315, 59, 335, 69, 11.4),
            span("230", 415, 59, 430, 69, 11.4),
        ],
        [
            span("Shop B", 100, 76, 150, 86, 11.4),
            span("21", 215, 76, 225, 86, 11.4),
            span("2400", 315, 76, 335, 86, 11.4),
            span("160", 415, 76, 430, 86, 11.4),
        ],
    ]

    assert serialized_visual_text(lines) == [
        "The information is recorded in the table below.",
        "Number of items Mean price ($) Standard deviation ($)",
        "Shop A 30 1500 230",
        "Shop B 21 2400 160",
    ]


def test_table_cell_geometry_restores_stacked_digit_fractions() -> None:
    lines = [
        [span("Complete the probability distribution table.", 50, 10, 250, 20, 11.4)],
        [
            span("x", 200, 30, 206, 40, 11.4),
            span("1", 250, 30, 256, 40, 11.4),
            span("2", 300, 30, 306, 40, 11.4),
            span("3", 350, 30, 356, 40, 11.4),
        ],
        [
            span("P(X = x)", 180, 52, 225, 64, 11.4),
            span("7", 300, 48, 306, 58, 11.4),
            span("19", 350, 48, 362, 58, 11.4),
        ],
        [
            span("64", 297, 61, 309, 71, 11.4),
            span("64", 350, 61, 362, 71, 11.4),
        ],
    ]

    assert serialized_visual_text(lines) == [
        "Complete the probability distribution table.",
        "x: 1, 2, 3.",
        "P(X = x): (7)/(64), (19)/(64).",
    ]


def test_numeric_display_rows_are_kept_with_their_shown_below_prompt() -> None:
    lines = [
        [span("The results are shown below.", 50, 10, 200, 20, 11.4)],
        [
            span("23", 200, 30, 212, 40, 11.4),
            span("19", 240, 30, 252, 40, 11.4),
            span("32", 280, 30, 292, 40, 11.4),
        ],
        [
            span("14", 200, 45, 212, 55, 11.4),
            span("25", 240, 45, 252, 55, 11.4),
            span("22", 280, 45, 292, 55, 11.4),
        ],
    ]

    assert serialized_visual_text(lines) == [
        "The results are shown below. 23 19 32 14 25 22",
    ]


def test_rectangle_class_interval_is_compacted_without_global_subtraction_repair() -> None:
    assert _line_text_from_spans(
        [span("The 1 − 10 rectangle has height 3 cm.", 50, 10, 240, 20)]
    ) == "The 1-10 rectangle has height 3 cm."
    assert _line_text_from_spans(
        [span("The value of 10 − 3 is required.", 50, 10, 220, 20)]
    ) == "The value of 10 − 3 is required."


@pytest.mark.parametrize("label", ["F", "L"])
def test_sigma_power_with_stacked_label_is_repaired_as_a_subscript(label: str) -> None:
    assert _line_text_from_spans(
        [span(f"Σ(x^{{2}})/({label})", 50, 10, 130, 20)]
    ) == f"Σx_{{{label}}}^{{2}}"


def test_overlapping_displayed_equation_remains_a_separate_reading_line() -> None:
    raw_order = [
        span("6 The straight line has equation", 50, 10, 240, 30),
        span("r = 7i + j", 72, 24, 145, 44),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert [_line_text_from_spans(line) for line in lines] == [
        "6 The straight line has equation",
        "r = 7i + j",
    ]


def test_encoded_digit_normalization_uses_page_number_glyphs() -> None:
    from exam_bank.models import BoundingBox, PageLayout, TextBlock
    from exam_bank.pdf_extract import _normalize_encoded_digit_text

    def text_block(page: int, text: str, x: float, y: float) -> TextBlock:
        return TextBlock(page_number=page, text=text, bbox=BoundingBox(x, y, x + 30, y + 12))

    layouts = [
        PageLayout(page_number=2, width=612, height=792, blocks=[text_block(2, "t", 303, 36)]),
        PageLayout(page_number=3, width=612, height=792, blocks=[text_block(3, "\x90", 303, 36)]),
        PageLayout(page_number=10, width=612, height=792, blocks=[text_block(10, "\x99\xf6", 300, 36)]),
        PageLayout(
            page_number=11,
            width=612,
            height=792,
            blocks=[
                text_block(11, "\x99\x99", 300, 36),
                text_block(11, "\x99~First question prompt", 72, 60),
                text_block(11, "t~Second question prompt", 72, 120),
            ],
        ),
    ]

    normalized = _normalize_encoded_digit_text(layouts)

    assert normalized[3].blocks[1].text == "1 First question prompt"
    assert normalized[3].blocks[2].text == "2 Second question prompt"


def test_legacy_pdf_detection_switches_on_pre_2017_names() -> None:
    assert _is_legacy_pdf(Path("9709_s16_qp_12.pdf")) is True
    assert _is_legacy_pdf(Path("9709_s17_qp_12.pdf")) is False


def test_font_encoding_differences_repair_semantic_math_aliases() -> None:
    math_span = span("10ÅÓ", 50, 10, 80, 20)
    math_span["font"] = "NewMathSymb"
    digit_span = span("101", 50, 40, 68, 50)
    digit_span["font"] = "xRoman"
    page = FakePage(
        [math_span, digit_span],
        fonts=[
            (10, "cff", "Type1", "ABCDEF+NewMathSymb", "R10", "WinAnsiEncoding", 0),
            (11, "cff", "Type1", "UVWXYZ+xRoman", "R11", "WinAnsiEncoding", 0),
        ],
        encodings={
            10: "<< /Differences [48 /pislant /thetaslant 197 /degrees 211 /textint] >>",
            11: "<< /Differences [49 /one] >>",
        },
    )

    blocks = _extract_text_blocks(page, 1, AppConfig())

    assert [block.text for block in blocks] == ["θπ°∫", "101"]


def test_font_encoding_differences_leave_ambiguous_same_font_code_untouched() -> None:
    ambiguous_span = span("1", 50, 10, 56, 20)
    ambiguous_span["font"] = "NewMathSymb"
    page = FakePage(
        [ambiguous_span],
        fonts=[
            (10, "cff", "Type1", "ABCDEF+NewMathSymb", "R10", "WinAnsiEncoding", 0),
            (11, "cff", "Type1", "UVWXYZ+NewMathSymb", "R11", "WinAnsiEncoding", 0),
        ],
        encodings={
            10: "<< /Differences [49 /thetaslant] >>",
            11: "<< /Differences [49 /pislant] >>",
        },
    )

    blocks = _extract_text_blocks(page, 1, AppConfig())

    assert [block.text for block in blocks] == ["1"]


def test_legacy_dense_non_text_cluster_recovers_fragmented_figure() -> None:
    boxes = [
        BoundingBox(100, 120, 108, 128),
        BoundingBox(112, 130, 126, 144),
        BoundingBox(130, 122, 150, 142),
        BoundingBox(160, 135, 175, 155),
    ]

    cluster = _dense_non_text_cluster(boxes, page_width=595, page_height=842)

    assert cluster == BoundingBox(100, 120, 175, 155)


def test_dense_grid_graphic_recovers_zero_area_vector_strokes() -> None:
    vertical_rules = [BoundingBox(x, 200, x, 320) for x in (100, 150, 200, 250)]
    horizontal_rules = [BoundingBox(100, y, 250, y) for y in (200, 240, 280, 320)]

    grid = _dense_grid_graphic_box(
        [*vertical_rules, *horizontal_rules],
        page_width=595,
        page_height=842,
    )

    assert grid == BoundingBox(100, 200, 250, 320)


def test_dense_grid_graphic_rejects_an_incomplete_axis_cluster() -> None:
    vertical_rules = [BoundingBox(x, 200, x, 320) for x in (100, 150, 200, 250)]
    horizontal_rules = [BoundingBox(100, y, 250, y) for y in (200, 320)]

    assert (
        _dense_grid_graphic_box(
            [*vertical_rules, *horizontal_rules],
            page_width=595,
            page_height=842,
        )
        is None
    )


def test_dense_grid_graphics_keep_stacked_grids_separate() -> None:
    upper = [
        *[BoundingBox(x, 100, x, 220) for x in (100, 150, 200, 250)],
        *[BoundingBox(100, y, 250, y) for y in (100, 140, 180, 220)],
    ]
    lower = [
        *[BoundingBox(x, 300, x, 420) for x in (100, 150, 200, 250)],
        *[BoundingBox(100, y, 250, y) for y in (300, 340, 380, 420)],
    ]

    assert _dense_grid_graphic_boxes(
        [*upper, *lower],
        page_width=595,
        page_height=842,
    ) == [
        BoundingBox(100, 100, 250, 220),
        BoundingBox(100, 300, 250, 420),
    ]


def test_sparse_lower_ocr_clip_excludes_page_margins_and_footer_blocks() -> None:
    class FakeFitz:
        @staticmethod
        def Rect(*coordinates: float) -> tuple[float, ...]:
            return coordinates

    config = AppConfig()
    blocks = [
        TextBlock(
            page_number=1,
            text=(
                "2 Find the exact value of x, showing all necessary working "
                "and giving your answer in simplified form. [4]"
            ),
            bbox=BoundingBox(72, 100, 500, 120),
        ),
        TextBlock(
            page_number=1,
            text="9709/12/M/J/26",
            bbox=BoundingBox(245, 810, 350, 822),
        ),
    ]

    clip = _sparse_lower_ocr_clip(FakePage([]), blocks, config, FakeFitz)

    assert clip == (
        config.detection.crop_left_margin,
        136.0,
        FakePage.rect.width - config.detection.crop_right_margin,
        FakePage.rect.height - config.detection.crop_bottom_margin,
    )


def test_adaptive_sparse_lower_ocr_skips_complete_native_page_tail() -> None:
    config = AppConfig()
    blocks = [
        TextBlock(
            page_number=1,
            text=(
                "2 Find the exact value of x, showing all necessary working "
                "and giving your answer in simplified form. [4]"
            ),
            bbox=BoundingBox(72, 100, 500, 120),
        )
    ]

    assert _should_run_sparse_lower_ocr(blocks, FakePage.rect.height, config) is False


def test_adaptive_sparse_lower_ocr_runs_when_terminal_mark_is_missing() -> None:
    config = AppConfig()
    blocks = [
        TextBlock(
            page_number=1,
            text=(
                "2 Find the exact value of x, showing all necessary working "
                "and giving your answer in simplified form."
            ),
            bbox=BoundingBox(72, 100, 500, 120),
        )
    ]

    assert _should_run_sparse_lower_ocr(blocks, FakePage.rect.height, config) is True


@pytest.mark.parametrize(
    "native_text",
    [
        "Question Paper 2 Solve the equation, showing all necessary working. [4]",
        "2 Solve the equa\ufffd\ufffdtion, showing all necessary working. [4]",
    ],
)
def test_adaptive_sparse_lower_ocr_runs_for_rejected_or_corrupt_native_text(native_text: str) -> None:
    config = AppConfig()
    blocks = [
        TextBlock(
            page_number=1,
            text=native_text,
            bbox=BoundingBox(72, 100, 500, 120),
        )
    ]

    assert _should_run_sparse_lower_ocr(blocks, FakePage.rect.height, config) is True


def test_always_strategy_preserves_sparse_lower_ocr_behavior() -> None:
    config = AppConfig()
    config.ocr.strategy = "always"
    blocks = [
        TextBlock(
            page_number=1,
            text=(
                "2 Find the exact value of x, showing all necessary working "
                "and giving your answer in simplified form. [4]"
            ),
            bbox=BoundingBox(72, 100, 500, 120),
        )
    ]

    assert _should_run_sparse_lower_ocr(blocks, FakePage.rect.height, config) is True


def test_ocr_hint_signals_expand_nearby_graphics_deterministically() -> None:
    hint = TextBlock(page_number=1, text="The diagram shows a sector.", bbox=BoundingBox(80, 220, 250, 238))
    graphic = BoundingBox(120, 250, 300, 360)

    first = _ocr_hint_graphics([hint], [graphic], page_width=595, page_height=842, legacy_fallback=False)
    second = _ocr_hint_graphics([hint], [graphic], page_width=595, page_height=842, legacy_fallback=False)

    assert first == second
    assert first[0].y0 <= hint.bbox.y0
    assert first[0].y1 >= graphic.y1


def test_question_number_stem_and_mark_stay_on_one_visual_line() -> None:
    raw_order = [
        span("The coefficient", 32, 10, 112, 20),
        span("1", 10, 10, 16, 20),
        span("[3]", 220, 10, 235, 20),
        span("of", 116, 10, 126, 20),
        span("x", 130, 10, 136, 20),
        span("3", 137, 5, 142, 12, size=7),
        span(".", 143, 10, 146, 20),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "1 The coefficient of x^{3}. [3]"


def test_mark_token_is_not_treated_as_script() -> None:
    raw_order = [
        span("Find", 50, 10, 72, 20),
        span("x", 76, 10, 82, 20),
        span("[3]", 120, 6, 135, 18, size=7),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "Find x [3]"


def test_question_number_is_not_treated_as_subscript_or_split_line() -> None:
    raw_order = [
        span("8", 10, 11, 16, 21, size=7),
        span("Express", 32, 10, 72, 20),
        span("x", 78, 10, 84, 20),
        span("2", 85, 5, 90, 12, size=7),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "8 Express x^{2}"


def test_line_text_repairs_spacing_before_math_functions() -> None:
    raw_order = [
        span("y", 50, 10, 56, 20),
        span("=", 60, 10, 66, 20),
        span("e", 72, 10, 78, 20),
        span("2", 79, 5, 84, 12, size=7),
        span("x", 85, 10, 91, 20),
        span("sin", 91, 10, 108, 20),
        span("2x", 108, 10, 122, 20),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "y = e^{2}x sin 2x"


def test_negative_exponent_repair_captures_digits_without_following_word() -> None:
    raw_order = [
        span("s", 50, 10, 56, 20),
        span("−", 56, 5, 60, 12, size=7),
        span("1", 60, 6, 64, 13, size=7),
        span("when", 64, 10, 88, 20),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert _line_text_from_spans(lines[0]) == "s^{-1} when"


def test_adjacent_small_star_after_variable_is_a_superscript() -> None:
    raw_order = [
        span("where", 50, 10, 75, 20, 11.38),
        span(" z", 75, 9.5, 83.3, 21.5, 11.89),
        span("*", 83.35, 10, 88, 19, 9.10),
        span(" ", 88, 8, 91, 20, 11.38),
        span("denotes", 91, 10, 130, 20, 11.38),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert _line_text_from_spans(lines[0]) == "where z^{*} denotes"


def test_full_size_star_between_variables_remains_an_operator() -> None:
    raw_order = [
        span("x", 50, 10, 56, 20),
        span(" ", 56, 10, 60, 20),
        span("*", 60, 10, 66, 20),
        span(" ", 66, 10, 70, 20),
        span("y", 70, 10, 76, 20),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert _line_text_from_spans(lines[0]) == "x * y"


def test_line_text_repairs_joined_pdf_words_with_capital_boundary() -> None:
    raw_order = [
        span("value", 50, 10, 75, 20),
        span("of", 75, 10, 85, 20),
        span("Express", 85, 10, 125, 20),
        span("R", 132, 10, 140, 20),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "valueof Express R"


def test_line_text_attaches_exponent_after_pdf_parenthesis_glyph() -> None:
    raw_order = [
        span("(", 50, 10, 54, 28),
        span("3", 54, 10, 60, 20),
        span("-", 61, 10, 66, 20),
        span("2x", 68, 10, 80, 20),
        span(")", 80, 10, 84, 28),
        span("5", 84, 7, 89, 14, size=7),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "(3 - 2x)^{5}"


def test_line_text_renders_raised_ring_glyph_as_degree_sign() -> None:
    raw_order = [
        span("360", 50, 10, 68, 20),
        span("◦", 68, 5, 72, 12, size=7),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert _line_text_from_spans(lines[0]) == "360°"


def test_line_text_repairs_pdf_control_parenthesis_before_exponent_detection() -> None:
    raw_order = [
        span("\x00", 50, 10, 54, 28),
        span("3", 54, 10, 60, 20),
        span("-", 61, 10, 66, 20),
        span("2x", 68, 10, 80, 20),
        span("\x01", 80, 10, 84, 28),
        span("5", 84, 7, 89, 14, size=7),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=6)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "(3 - 2x)^{5}"


def test_line_text_reconstructs_stacked_trig_fraction() -> None:
    raw_order = [
        span("tan", 50, 10, 66, 20),
        span("x", 66, 10, 72, 20),
        span("+", 73, 10, 78, 20),
        span("sin", 80, 10, 96, 20),
        span("x", 96, 10, 102, 20),
        span("tan", 50, 26, 66, 36),
        span("x", 66, 26, 72, 36),
        span("-", 73, 26, 78, 36),
        span("sin", 80, 26, 96, 36),
        span("x", 96, 26, 102, 36),
        span("=", 110, 18, 116, 28),
        span("k", 120, 18, 126, 28),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "(tan x + sin x)/(tan x - sin x) = k"


def test_line_text_keeps_tall_math_operator_inside_stacked_fraction() -> None:
    raw_order = [
        span("6", 62, 6, 68, 16),
        span("2x", 50, 22, 62, 32),
        span(" −", 62, 16, 72, 38, size=10),
        span("3", 74, 22, 80, 32),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == "(6)/(2x − 3)"


def test_line_text_keeps_prose_span_out_of_stacked_fraction() -> None:
    raw_order = [
        span("7 ", 49.6, 73.2, 58.3, 84.7),
        span("(a) Prove the identity tan", 72.3, 73.2, 195.3, 93.8),
        span("tan", 183.4, 67.1, 197.4, 78.6),
        span("i", 199.0, 67.9, 204.9, 78.4),
        span("+", 206.5, 67.1, 214.3, 78.6),
        span("7", 215.9, 67.1, 221.6, 78.6),
        span("i", 201.1, 83.1, 207.0, 93.6),
        span("-", 208.6, 82.3, 216.4, 93.8),
        span("3", 218.0, 82.3, 223.7, 93.8),
        span(".", 330.9, 73.2, 336.7, 84.7),
        span("[3]", 532.3, 73.2, 548.6, 84.7),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)
    text = _line_text_from_spans(lines[0])

    assert "(tani + 7)/(i - 3)" in text
    assert "/((a)" not in text


def test_line_text_does_not_reconstruct_derivative_prompt_as_fraction() -> None:
    raw_order = [
        span("(a) Express x", 72.3, 95.4, 146.8, 113.4),
        span("d", 135.9, 87.5, 141.7, 99.0),
        span("y", 141.7, 87.5, 146.8, 99.0),
        span("d", 135.9, 101.9, 141.7, 113.4),
        span("as a simplified fraction", 148.4, 95.4, 260.0, 106.9),
        span("[4]", 532.3, 95.4, 545.7, 106.9),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)
    text = _line_text_from_spans(lines[0])

    assert _stacked_fraction_line_text(lines[0]) is None
    assert "/" not in text
    assert "(dy)/((a)" not in text


@pytest.mark.parametrize(
    ("numerator_variable", "denominator_variable"),
    [("y", "x"), ("x", "y"), ("x", "t"), ("x", "θ")],
)
def test_stacked_derivative_fraction_preserves_variable_orientation(
    numerator_variable: str,
    denominator_variable: str,
) -> None:
    raw_order = [
        span("Show that", 70, 99, 112, 110),
        span("d", 145, 92, 151, 103),
        span(numerator_variable, 151, 92, 157, 103),
        span("d", 145, 107, 151, 118),
        span(denominator_variable, 151, 107, 157, 118),
        span("=", 164, 94, 170, 117),
        span("k", 178, 99, 184, 110),
    ]

    assert _stacked_fraction_line_text(raw_order) == (
        f"Show that (d{numerator_variable})/(d{denominator_variable}) = k"
    )


@pytest.mark.parametrize("raw_text", ["ddxy", "ddyx", "xddy", "ddθx"])
def test_flat_derivative_glyph_run_is_not_guessed_without_geometry(raw_text: str) -> None:
    assert _line_text_from_spans([span(raw_text, 70, 99, 100, 110)]) == raw_text


def test_adjacent_derivative_equations_do_not_merge_as_a_fraction() -> None:
    raw_order = [
        span("Show that", 70, 90, 112, 110),
        span("d", 130, 90, 136, 110),
        span("y", 136, 90, 142, 110),
        span("=", 148, 90, 154, 110),
        span("2x", 160, 90, 174, 110),
        span("d", 130, 107, 136, 127),
        span("x", 136, 107, 142, 127),
        span("=", 148, 107, 154, 127),
        span("3t", 160, 107, 174, 127),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)

    assert [_line_text_from_spans(line) for line in lines] == [
        "Show that dy = 2x",
        "dx = 3t",
    ]


def test_split_derivative_rows_must_be_horizontally_aligned() -> None:
    raw_order = [
        span("Show that", 70, 90, 112, 110),
        span("d", 130, 90, 136, 110),
        span("y", 136, 90, 142, 110),
        span("=", 148, 90, 154, 110),
        span("x^2 + y^2", 160, 90, 260, 110),
        span("d", 210, 107, 216, 127),
        span("x", 216, 107, 222, 127),
        span("y + 2xy", 230, 107, 300, 127),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)

    assert [_line_text_from_spans(line) for line in lines] == [
        "Show that dy = x^2 + y^2",
        "dx y + 2xy",
    ]


def test_split_derivative_equation_rows_merge_before_serialization() -> None:
    raw_order = [
        span("(a)", 72.8, 99.3, 86.6, 110.7, 11.4),
        span("Show that", 96.4, 99.4, 143.2, 110.8, 11.4),
        span(" ", 143.2, 99.4, 147.6, 110.8, 11.4),
        span("d", 147.6, 92.0, 153.5, 103.4, 11.4),
        span("y", 153.5, 91.6, 158.9, 103.4, 11.9),
        span("d", 147.6, 107.0, 153.5, 118.4, 11.4),
        span("x", 153.5, 106.6, 158.9, 118.4, 11.9),
        span(" ", 158.9, 93.5, 163.6, 124.7, 11.0),
        span("=", 163.6, 93.5, 170.4, 117.2, 11.0),
        span(" ", 170.4, 99.0, 178.2, 110.9, 11.9),
        span("x", 178.2, 91.6, 183.7, 103.4, 11.9),
        span("2", 183.7, 89.9, 188.0, 98.2, 8.3),
        span(" +", 188.0, 81.6, 197.8, 109.7, 11.0),
        span(" y", 197.8, 91.6, 205.7, 103.4, 11.9),
        span("2", 205.7, 89.9, 210.0, 98.2, 8.3),
        span("y", 175.0, 107.5, 180.4, 119.4, 11.9),
        span("2", 180.4, 106.1, 184.7, 114.4, 8.3),
        span(" ", 184.7, 97.8, 187.7, 125.7, 11.0),
        span("−", 187.7, 102.0, 194.5, 125.7, 11.0),
        span("2", 196.9, 108.0, 202.8, 119.3, 11.4),
        span("xy", 202.8, 107.5, 213.8, 119.4, 11.9),
        span(".", 215.3, 99.4, 218.2, 110.8, 11.4),
        span("[4]", 531.7, 99.4, 545.4, 110.8, 11.4),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)

    assert len(lines) == 1
    assert _line_text_from_spans(lines[0]) == (
        "(a) Show that (dy)/(dx) = (x^{2} + y^{2})/(y^{2} − 2xy). [4]"
    )


def test_completed_display_equation_does_not_bridge_into_following_prose() -> None:
    raw_order = [
        span("t", 265.2, 309.3, 268.6, 321.2, 11.9),
        span("d", 270.3, 302.3, 276.2, 313.7, 11.4),
        span("x", 276.2, 301.8, 281.6, 313.7, 11.9),
        span("d", 271.3, 317.3, 277.1, 328.7, 11.4),
        span("t", 277.1, 316.8, 280.6, 328.7, 11.9),
        span("=", 286.2, 300.9, 293.1, 332.4, 11.0),
        span("k", 297.7, 301.8, 303.1, 313.7, 11.9),
        span(" −", 303.1, 293.5, 312.6, 325.0, 11.0),
        span("x", 315.1, 301.8, 320.5, 313.7, 11.9),
        span("3", 320.5, 300.4, 324.6, 308.1, 7.8),
        span("2", 303.4, 318.2, 309.3, 329.6, 11.4),
        span("x", 309.3, 317.8, 314.8, 329.6, 11.9),
        span("2", 314.8, 316.6, 318.8, 324.3, 7.8),
        span(",", 326.6, 309.8, 329.5, 321.1, 11.4),
        span("for t", 72.8, 336.7, 92.9, 348.1, 11.4),
        span(" >", 92.9, 327.9, 103.0, 359.4, 11.0),
        span(
            " 0, where k is a constant. When t = 1, x = 1.",
            103.0,
            336.7,
            414.5,
            348.1,
            11.4,
        ),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)

    assert [_line_text_from_spans(line) for line in lines] == [
        "t(dx)/(dt) = (k − x^{3})/(2x^{2}),",
        "for t > 0, where k is a constant. When t = 1, x = 1.",
    ]


def test_line_text_does_not_reconstruct_superscripts_as_fraction() -> None:
    raw_order = [
        span("1 ", 49.6, 72.1, 58.3, 83.6),
        span("(a) Expand ", 72.3, 72.1, 133.0, 83.6),
        span("b", 133.4, 71.9, 137.9, 83.7),
        span("2", 137.5, 72.1, 143.2, 83.6),
        span("-", 144.8, 72.1, 152.6, 83.6),
        span("1", 155.5, 69.5, 159.6, 77.6, size=8.1),
        span("2", 155.5, 78.8, 159.6, 86.8, size=8.1),
        span("x", 161.3, 72.1, 166.4, 83.6),
        span("l", 166.1, 71.9, 170.6, 83.7),
        span("6", 170.0, 64.7, 174.1, 72.8, size=8.1),
        span(" in ascending powers of x", 175.2, 72.1, 290.0, 83.6),
        span("[3]", 532.3, 72.1, 545.7, 83.6),
    ]

    lines = _group_spans_into_visual_lines(raw_order, y_tolerance=16)

    assert "/" not in _line_text_from_spans(lines[0])


def test_extract_text_blocks_drops_tall_margin_furniture_before_grouping() -> None:
    page = FakePage(
        [
            span("DO NOT WRITE IN THIS MARGIN " * 5, 6, 32, 14, 812),
            span("(c) Make two comparisons between the times for the two teams.", 72, 505, 377, 517),
        ]
    )

    blocks = _extract_text_blocks(page, 1, AppConfig())

    assert len(blocks) == 1
    assert blocks[0].text == "(c) Make two comparisons between the times for the two teams."
    assert blocks[0].bbox.y0 == 505


def test_extract_text_blocks_drops_control_artifact_runs_before_parenthesis_repair() -> None:
    page = FakePage(
        [
            span("1 Find E(X). [2]", 72, 90, 180, 102),
            span(",\x01\x01\x01\x01\x01\x01\x01\x01\x05,", 210, 92, 280, 104),
            span("\x00", 72, 130, 76, 148),
            span("x", 76, 130, 82, 142),
            span("\x01", 82, 130, 86, 148),
            span("2", 86, 127, 90, 135, size=7),
        ]
    )

    blocks = _extract_text_blocks(page, 1, AppConfig())
    text = "\n".join(block.text for block in blocks)

    assert "))))" not in text
    assert "(x)^{2}" in text


def _require_corpus_pdf(relative_path: str) -> Path:
    path = Path(__file__).resolve().parents[1] / relative_path
    if not path.exists():
        pytest.skip(f"repository corpus PDF is unavailable: {relative_path}")
    return path


def _structured_corpus_question_text(relative_path: str, question_number: str) -> str:
    from exam_bank.extraction_structure import build_structured_question_text
    from exam_bank.pdf_extract import extract_pdf_layout
    from exam_bank.question_detection import detect_question_spans

    pdf_path = _require_corpus_pdf(relative_path)
    config = AppConfig()
    config.ocr.enabled = False
    layouts = extract_pdf_layout(pdf_path, config, use_ocr=False)
    span = next(
        item
        for item in detect_question_spans(layouts, pdf_path, config)
        if item.question_number == question_number
    )
    return " ".join(build_structured_question_text(span, layouts, config).combined_question_text.split())


def _canonical_gold_question_text(question_id: str) -> str:
    path = Path(__file__).resolve().parents[1] / (
        "data/review/canonical/text_fidelity/question_text_gold.v1.json"
    )
    if not path.exists():
        pytest.skip("canonical question-text gold registry is unavailable")
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = next(
        row for row in payload["records"] if row["question_id"] == question_id
    )
    return " ".join(record["question_text"].split())


@pytest.mark.integration
@pytest.mark.parametrize(
    ("question_id", "relative_path", "question_number"),
    [
        ("51winter23_q05", "input/pastpapers/9709/2023/question_papers/9709_w23_qp_51.pdf", "5"),
        ("53winter20_q07", "input/pastpapers/9709/2020/question_papers/9709_w20_qp_53.pdf", "7"),
        ("53winter22_q04", "input/pastpapers/9709/2022/question_papers/9709_w22_qp_53.pdf", "4"),
        ("61summer16_q07", "input/pastpapers/9709/2016/question_papers/9709_s16_qp_61.pdf", "7"),
        ("61winter10_q04", "input/pastpapers/9709/2010/question_papers/9709_w10_qp_61.pdf", "4"),
        ("61winter18_q07", "input/pastpapers/9709/2018/question_papers/9709_w18_qp_61.pdf", "7"),
        ("62spring16_q04", "input/pastpapers/9709/2016/question_papers/9709_m16_qp_62.pdf", "4"),
        ("63summer15_q02", "input/pastpapers/9709/2015/question_papers/9709_s15_qp_63.pdf", "2"),
        ("52winter25_q04", "input/pastpapers/9709/2025/question_papers/9709_w25_qp_52.pdf", "4"),
        ("62winter09_q06", "input/pastpapers/9709/2009/question_papers/9709_w09_qp_62.pdf", "6"),
        ("63summer18_q04", "input/pastpapers/9709/2018/question_papers/9709_s18_qp_63.pdf", "4"),
        ("51spring21_q05", "input/pastpapers/9709/2021/question_papers/9709_m21_qp_51.pdf", "5"),
        ("52winter23_q04", "input/pastpapers/9709/2023/question_papers/9709_w23_qp_52.pdf", "4"),
        ("61winter16_q07", "input/pastpapers/9709/2016/question_papers/9709_w16_qp_61.pdf", "7"),
        ("63winter14_q04", "input/pastpapers/9709/2014/question_papers/9709_w14_qp_63.pdf", "4"),
        ("53spring23_q04", "input/pastpapers/9709/2023/question_papers/9709_m23_qp_53.pdf", "4"),
    ],
)
def test_caie_geometry_serialized_tables_match_canonical_gold(
    question_id: str,
    relative_path: str,
    question_number: str,
) -> None:
    assert _structured_corpus_question_text(
        relative_path,
        question_number,
    ) == _canonical_gold_question_text(question_id)


@pytest.mark.integration
def test_caie_newmathsymb_repairs_theta_and_pi_from_2013_subset() -> None:
    fitz = pytest.importorskip("fitz")
    pdf_path = _require_corpus_pdf(
        "input/pastpapers/9709/2013/question_papers/9709_s13_qp_13.pdf"
    )

    with fitz.open(pdf_path) as document:
        blocks = _extract_text_blocks(document[1], 2, AppConfig())

    text = "\n".join(block.text for block in blocks)
    assert "2 cos^{2} θ = tan^{2} θ" in text
    assert "0 ≤ θ ≤ π" in text
    assert "solutions in terms of π" in text
    assert "0 ≤ x ≤ 2π" in text


@pytest.mark.integration
def test_caie_stacked_trig_fractions_are_serialized_in_reading_order() -> None:
    text = _structured_corpus_question_text(
        "input/pastpapers/9709/2015/question_papers/9709_w15_qp_12.pdf",
        "4",
    )

    assert text == (
        "4 (i) Prove the identity ((1)/(sin x) - (1)/(tan x))^{2} ≡ "
        "(1 - cos x)/(1 + cos x). [4] (ii) Hence solve the equation "
        "((1)/(sin x) - (1)/(tan x))^{2} = (2)/(5) for 0 ≤ x ≤ 2π. [3]"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("relative_path", "question_number", "expected"),
    [
        (
            "input/pastpapers/9709/2020/question_papers/9709_m20_qp_32.pdf",
            "7",
            "7 The equation of a curve is x^{3} + 3xy^{2} - y^{3} = 5. "
            "(a) Show that (dy)/(dx) = (x^{2} + y^{2})/(y^{2} - 2xy). [4] "
            "(b) Find the coordinates of the points on the curve where the tangent is "
            "parallel to the y-axis. [5]",
        ),
        (
            "input/pastpapers/9709/2013/question_papers/9709_s13_qp_33.pdf",
            "8",
            "8 The variables x and t satisfy the differential equation "
            "t(dx)/(dt) = (k - x^{3})/(2x^{2}), for t > 0, where k is a constant. "
            "When t = 1, x = 1 and when t = 4, x = 2. "
            "(i) Solve the differential equation, finding the value of k and obtaining "
            "an expression for x in terms of t. [9] "
            "(ii) State what happens to the value of x as t becomes large. [1]",
        ),
        (
            "input/pastpapers/9709/2018/question_papers/9709_s18_qp_33.pdf",
            "8",
            "8 The equation of a curve is 2x^{3} - y^{3} - 3xy^{2} = 2a^{3}, "
            "where a is a non-zero constant. "
            "(i) Show that (dy)/(dx) = (2x^{2} - y^{2})/(y^{2} + 2xy). [4] "
            "(ii) Find the coordinates of the two points on the curve at which the "
            "tangent is parallel to the y-axis. [5]",
        ),
    ],
)
def test_caie_stacked_derivative_equations_are_serialized_exactly(
    relative_path: str,
    question_number: str,
    expected: str,
) -> None:
    assert _structured_corpus_question_text(relative_path, question_number) == expected


@pytest.mark.integration
def test_caie_integral_bounds_and_opaque_integral_glyph_are_serialized() -> None:
    older_text = _structured_corpus_question_text(
        "input/pastpapers/9709/2019/question_papers/9709_s19_qp_33.pdf",
        "3",
    )
    recent_text = _structured_corpus_question_text(
        "input/pastpapers/9709/2025/question_papers/9709_w25_qp_33.pdf",
        "6",
    )

    assert older_text == (
        "3 Let f(θ) = (1 - cos 2θ + sin 2θ)/(1 + cos 2θ + sin 2θ). "
        "(i) Show that f(θ) = tan θ. [3] "
        "(ii) Hence show that ∫_{(1)/(6)π}^{(1)/(4)π} f(θ) dθ = "
        "(1)/(2) ln((3)/(2)). [4]"
    )
    assert recent_text == "6 Find the exact value of ∫_{0}^{(1)/(6)π} x^{2} sin 2x dx. [6]"


@pytest.mark.integration
def test_caie_newmathsymb_repairs_theta_from_2021_subset() -> None:
    fitz = pytest.importorskip("fitz")
    pdf_path = _require_corpus_pdf(
        "input/pastpapers/9709/2021/question_papers/9709_w21_qp_33.pdf"
    )

    with fitz.open(pdf_path) as document:
        blocks = _extract_text_blocks(document[5], 6, AppConfig())

    text = "\n".join(block.text for block in blocks)
    assert "sin θ = 3 cos 2θ + 2" in text
    assert "0° ≤θ ≤ 360°" in text


class FakeRect:
    width = 595
    height = 842


class FakeDocument:
    def __init__(self, encodings: dict[int, str]) -> None:
        self.encodings = encodings

    def xref_get_key(self, xref: int, key: str) -> tuple[str, str]:
        assert key == "Encoding"
        encoding = self.encodings.get(xref)
        if encoding is None:
            return ("null", "null")
        return ("dict", encoding)


class FakePage:
    rect = FakeRect()
    rotation = 0

    def __init__(
        self,
        spans: list[dict],
        *,
        fonts: list[tuple] | None = None,
        encodings: dict[int, str] | None = None,
    ) -> None:
        self.spans = spans
        self.fonts = fonts or []
        self.parent = FakeDocument(encodings or {})

    def get_fonts(self, full: bool = False) -> list[tuple]:
        assert full is True
        return self.fonts

    def get_text(self, kind: str) -> dict:
        assert kind == "dict"
        return {
            "blocks": [
                {
                    "type": 0,
                    "lines": [{"spans": [span]} for span in self.spans],
                }
            ]
        }
