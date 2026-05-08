from exam_bank.config import AppConfig
from exam_bank.pdf_extract import _extract_text_blocks, _group_spans_into_visual_lines, _line_text_from_spans


def span(text: str, x0: float, y0: float, x1: float, y1: float, size: float = 10) -> dict:
    return {
        "text": text,
        "bbox": [x0, y0, x1, y1],
        "size": size,
        "font": "TestFont",
    }


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


from exam_bank.pdf_extract import _group_spans_into_visual_lines, _line_text_from_spans


def span(text: str, x0: float, y0: float, x1: float, y1: float, size: float = 10) -> dict:
    return {
        "text": text,
        "bbox": [x0, y0, x1, y1],
        "size": size,
        "font": "TestFont",
    }


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

    assert "/" not in text
    assert "(dy)/((a)" not in text


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


class FakeRect:
    width = 595
    height = 842


class FakePage:
    rect = FakeRect()
    rotation = 0

    def __init__(self, spans: list[dict]) -> None:
        self.spans = spans

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
