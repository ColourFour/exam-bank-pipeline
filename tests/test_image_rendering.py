from pathlib import Path

import pytest

from exam_bank.config import AppConfig
from exam_bank.image_rendering import (
    CropRegion,
    _crop_diagnostics,
    _dedupe_crop_regions,
    _detect_prompt_regions,
    _graphics_for_segment,
    _same_page_diagram_union_regions,
    _single_page_union_regions,
    _trim_vertical_furniture_from_regions,
)
from exam_bank.core.asset_paths import AssetPath
from exam_bank.core.paper_identity import PaperIdentity
from exam_bank.models import BoundingBox, PageLayout, QuestionSpan, TextBlock


pytestmark = pytest.mark.rendering


def text_block(text: str, y: float, x: float = 50, width: float = 450) -> TextBlock:
    return TextBlock(page_number=1, text=text, bbox=BoundingBox(x, y, x + width, y + 14))


def span() -> QuestionSpan:
    return QuestionSpan(
        source_pdf=Path("9709_m24_qp_12.pdf"),
        paper_name="9709_m24_qp_12",
        question_number="10",
        start_page=1,
        start_y=60,
        end_page=1,
        end_y=500,
        page_numbers=[1],
        blocks=[],
        full_question_label="10",
    )


def test_formula_rule_graphics_do_not_create_diagram_regions() -> None:
    config = AppConfig()
    text_box = BoundingBox(50, 280, 545, 315)
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[],
        graphics=[
            BoundingBox(150, 300, 235, 300.5),
            BoundingBox(250, 330, 350, 440),
            BoundingBox(145, 80, 230, 80.5),
        ],
    )

    graphics, excluded = _graphics_for_segment(text_box, layout, config)

    assert BoundingBox(150, 300, 235, 300.5) not in graphics
    assert BoundingBox(250, 330, 350, 440) in graphics
    assert {"label": "barcode", "bbox": {"x0": 145, "y0": 80, "x1": 230, "y1": 80.5}} in excluded


def test_page_diagram_union_does_not_cross_large_answer_gap() -> None:
    config = AppConfig()
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    regions = [
        CropRegion(page_number=1, bbox=BoundingBox(40, 55, 300, 90), text_blocks=[text_block("10", 65)], region_kind="text"),
        CropRegion(
            page_number=1,
            bbox=BoundingBox(80, 90, 360, 250),
            graphics=[BoundingBox(100, 100, 340, 230)],
            region_kind="figure",
        ),
        CropRegion(page_number=1, bbox=BoundingBox(70, 255, 555, 360), text_blocks=[text_block("(a) Find the tangent. [3]", 265)], region_kind="text"),
        CropRegion(page_number=1, bbox=BoundingBox(70, 560, 555, 600), text_blocks=[text_block("(b) Find the circle. [2]", 570)], region_kind="text"),
    ]

    merged, flags = _same_page_diagram_union_regions(regions, span(), [layout], config)

    assert [region.region_kind for region in merged] == ["page_diagram_union", "text"]
    assert merged[0].bbox.y1 < 380
    assert merged[1].bbox.y0 == 560
    assert "page_diagram_union_used" in flags


def test_crop_region_dedupe_removes_stale_duplicate_fragment() -> None:
    regions = [
        CropRegion(
            page_number=1,
            bbox=BoundingBox(60, 120, 520, 360),
            text_blocks=[text_block("10 (a) Full prompt", 130), text_block("(b) Continuation", 310)],
            region_kind="page_diagram_union",
        ),
        CropRegion(
            page_number=1,
            bbox=BoundingBox(65, 125, 500, 210),
            text_blocks=[text_block("10 (a) Full prompt", 130)],
            region_kind="text",
        ),
    ]

    deduped, flags = _dedupe_crop_regions(regions)

    assert len(deduped) == 1
    assert deduped[0].region_kind == "page_diagram_union"
    assert "stale_crop_fragment_removed" in flags


def test_single_page_union_skips_disjoint_text_tail_and_allows_page_union() -> None:
    config = AppConfig()
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    regions = [
        CropRegion(page_number=1, bbox=BoundingBox(40, 55, 300, 90), text_blocks=[text_block("2", 65)], region_kind="text"),
        CropRegion(
            page_number=1,
            bbox=BoundingBox(80, 90, 520, 250),
            graphics=[BoundingBox(100, 100, 500, 230)],
            text_blocks=[text_block("axis", 95)],
            region_kind="figure",
            excluded_regions=[
                {"label": "barcode", "bbox": {"x0": 140, "y0": 80, "x1": 200, "y1": 110}},
                {"label": "barcode", "bbox": {"x0": 140, "y0": 80, "x1": 200, "y1": 110}},
            ],
        ),
        CropRegion(page_number=1, bbox=BoundingBox(60, 260, 555, 360), text_blocks=[text_block("(a) Describe the graph. [4]", 270)], region_kind="text"),
        CropRegion(page_number=1, bbox=BoundingBox(60, 550, 555, 585), text_blocks=[text_block("(b) Find f(x). [2]", 560)], region_kind="text"),
    ]

    union_regions, union_flags = _single_page_union_regions(regions, span(), [layout], config)
    page_regions, page_flags = _same_page_diagram_union_regions(regions, span(), [layout], config)

    assert union_regions is None
    assert "single_page_union_skipped_disjoint_tail" in union_flags
    assert [region.region_kind for region in page_regions] == ["page_diagram_union", "text"]
    assert page_regions[0].bbox.y1 < 400
    assert page_regions[1].bbox.y0 == 550
    assert len(page_regions[0].excluded_regions) == 1
    assert "page_diagram_union_used" in page_flags


def test_prompt_regions_drop_trailing_foreign_question_after_missed_anchor() -> None:
    config = AppConfig()
    current_question = [
        TextBlock(page_number=1, text="1 Solve the equation. [2]", bbox=BoundingBox(50, 82, 310, 96)),
        TextBlock(page_number=1, text="Show your working clearly.", bbox=BoundingBox(72, 118, 330, 132)),
    ]
    foreign_number = TextBlock(page_number=1, text="2", bbox=BoundingBox(50, 255, 60, 269))
    foreign_prompt = TextBlock(page_number=1, text="Find the next answer. [4]", bbox=BoundingBox(72, 256, 340, 270))
    test_span = QuestionSpan(
        source_pdf=Path("9709_s21_qp_12.pdf"),
        paper_name="9709_s21_qp_12",
        question_number="1",
        start_page=1,
        start_y=82,
        end_page=1,
        end_y=330,
        page_numbers=[1],
        blocks=[*current_question, foreign_number, foreign_prompt],
        full_question_label="1",
    )
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[*current_question, foreign_number, foreign_prompt],
    )

    regions, flags = _detect_prompt_regions(test_span, [layout], config)
    rendered_text = "\n".join(block.text for region in regions for block in region.text_blocks)

    assert "foreign_question_region_removed" in flags
    assert all(region.bbox.y1 < foreign_number.bbox.y0 for region in regions)
    assert "Find the next answer" not in rendered_text


def test_vertical_furniture_trim_removes_centered_header_without_cutting_top_diagram() -> None:
    config = AppConfig()
    diagram = BoundingBox(150, 72, 430, 260)
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[
            TextBlock(page_number=1, text="10", bbox=BoundingBox(292, 47, 304, 60)),
            text_block("The diagram shows a sector. [5]", 285, x=72),
        ],
        graphics=[diagram],
    )
    region = CropRegion(
        page_number=1,
        bbox=BoundingBox(35, 45, 560, 340),
        text_blocks=[text_block("The diagram shows a sector. [5]", 285, x=72)],
        graphics=[diagram],
        region_kind="page_diagram_union",
        figure_bbox=diagram,
    )

    trimmed, flags = _trim_vertical_furniture_from_regions([region], [layout], config)

    assert trimmed[0].bbox.y0 > 60
    assert trimmed[0].bbox.y0 < diagram.y0
    assert trimmed[0].bbox.y1 == 340
    assert "centered_page_number_trimmed" in flags
    assert "crop_header_footer_trimmed" in flags


def test_diagram_union_trims_page_number_after_union_padding() -> None:
    config = AppConfig()
    diagram = BoundingBox(150, 72, 430, 260)
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[
            TextBlock(page_number=1, text="10", bbox=BoundingBox(292, 47, 304, 60)),
            text_block("10", 65, x=50, width=20),
            text_block("The diagram shows a sector. [5]", 285, x=72),
        ],
        graphics=[diagram],
    )
    regions = [
        CropRegion(page_number=1, bbox=BoundingBox(40, 62, 90, 90), text_blocks=[text_block("10", 65, x=50, width=20)], region_kind="text"),
        CropRegion(page_number=1, bbox=BoundingBox(140, 72, 440, 260), graphics=[diagram], region_kind="figure"),
        CropRegion(page_number=1, bbox=BoundingBox(62, 280, 555, 330), text_blocks=[text_block("The diagram shows a sector. [5]", 285, x=72)], region_kind="text"),
    ]

    merged, flags = _same_page_diagram_union_regions(regions, span(), [layout], config)

    assert len(merged) == 1
    assert merged[0].region_kind == "page_diagram_union"
    assert merged[0].bbox.y0 > 60
    assert merged[0].bbox.y0 < diagram.y0
    assert "centered_page_number_trimmed" in flags
    assert "crop_header_footer_trimmed" in flags


def test_vertical_trim_removes_safe_page_number_even_when_lower_furniture_is_unsafe() -> None:
    config = AppConfig()
    diagram = BoundingBox(150, 140, 430, 260)
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[
            TextBlock(page_number=1, text="10", bbox=BoundingBox(292, 44, 304, 51)),
            text_block("12", 65, x=50, width=20),
            text_block("The diagram shows a sector. [5]", 285, x=72),
        ],
        graphics=[
            diagram,
            BoundingBox(245, 92, 360, 155),
        ],
    )
    region = CropRegion(
        page_number=1,
        bbox=BoundingBox(35, 45, 560, 340),
        text_blocks=[
            text_block("12", 65, x=50, width=20),
            text_block("The diagram shows a sector. [5]", 285, x=72),
        ],
        graphics=[diagram],
        region_kind="page_diagram_union",
        figure_bbox=diagram,
    )

    trimmed, flags = _trim_vertical_furniture_from_regions([region], [layout], config)

    assert trimmed[0].bbox.y0 > 51
    assert trimmed[0].bbox.y0 < 65
    assert trimmed[0].bbox.y0 < diagram.y0
    assert "centered_page_number_trimmed" in flags
    assert "crop_header_footer_trimmed" in flags


def test_vertical_furniture_trim_handles_top_and_bottom_page_numbers_across_regions() -> None:
    config = AppConfig()
    layout1 = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[
            TextBlock(page_number=1, text="8", bbox=BoundingBox(292, 47, 304, 60)),
            TextBlock(page_number=1, text="First page prompt. [3]", bbox=BoundingBox(72, 95, 420, 112)),
        ],
    )
    layout2 = PageLayout(
        page_number=2,
        width=595,
        height=842,
        blocks=[
            TextBlock(page_number=2, text="12", bbox=BoundingBox(292, 780, 308, 795)),
            TextBlock(page_number=2, text="Continuation prompt. [2]", bbox=BoundingBox(72, 690, 420, 707)),
        ],
    )
    regions = [
        CropRegion(
            page_number=1,
            bbox=BoundingBox(35, 45, 560, 160),
            text_blocks=[TextBlock(page_number=1, text="First page prompt. [3]", bbox=BoundingBox(72, 95, 420, 112))],
            region_kind="text",
        ),
        CropRegion(
            page_number=2,
            bbox=BoundingBox(35, 650, 560, 800),
            text_blocks=[TextBlock(page_number=2, text="Continuation prompt. [2]", bbox=BoundingBox(72, 690, 420, 707))],
            region_kind="text",
        ),
    ]

    trimmed, flags = _trim_vertical_furniture_from_regions(regions, [layout1, layout2], config)

    assert trimmed[0].bbox.y0 > 60
    assert trimmed[1].bbox.y1 < 780
    assert flags.count("centered_page_number_trimmed") <= 1
    assert "centered_page_number_trimmed" in flags


def test_diagram_union_keeps_graph_labels_with_figure_and_removes_stale_fragment() -> None:
    config = AppConfig()
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    graph = BoundingBox(120, 80, 420, 300)
    label = text_block("8 y = a cos(bx) + c", 145, x=195, width=180)
    regions = [
        CropRegion(page_number=1, bbox=BoundingBox(40, 55, 300, 90), text_blocks=[text_block("5", 65)], region_kind="text"),
        CropRegion(
            page_number=1,
            bbox=BoundingBox(100, 70, 440, 320),
            graphics=[graph],
            text_blocks=[label],
            region_kind="figure",
        ),
        CropRegion(page_number=1, bbox=BoundingBox(60, 330, 555, 390), text_blocks=[text_block("(a) Find a, b and c. [3]", 340)], region_kind="text"),
        CropRegion(
            page_number=1,
            bbox=BoundingBox(110, 90, 430, 280),
            graphics=[graph],
            text_blocks=[label],
            region_kind="figure",
        ),
    ]

    merged, flags = _same_page_diagram_union_regions(regions, span(), [layout], config)

    assert len([region for region in merged if region.region_kind == "page_diagram_union"]) == 1
    assert "page_diagram_union_used" in flags
    assert any("8 y = a cos" in block.text for region in merged for block in region.text_blocks)


def test_question_context_infers_figure_below_diagram_prompt() -> None:
    config = AppConfig()
    prompt = text_block("3 The diagram shows a sector. [5]", 180, x=60, width=250)
    diagram = BoundingBox(120, 90, 430, 150)
    test_span = QuestionSpan(
        source_pdf=Path("9709_s16_qp_12.pdf"),
        paper_name="9709_s16_qp_12",
        question_number="3",
        start_page=1,
        start_y=170,
        end_page=1,
        end_y=380,
        page_numbers=[1],
        blocks=[prompt],
        full_question_label="3",
    )
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[prompt], graphics=[diagram])

    regions, flags = _detect_prompt_regions(test_span, [layout], config)

    assert any(region.region_kind == "context_inferred_figure" for region in regions)
    assert "question_context_figure_inference_used" in flags


def test_missing_figure_prompt_is_marked_detection_failure() -> None:
    config = AppConfig()
    prompt = text_block("4 The diagram shows a sector. [5]", 110, x=60, width=250)
    test_span = QuestionSpan(
        source_pdf=Path("9709_s16_qp_12.pdf"),
        paper_name="9709_s16_qp_12",
        question_number="4",
        start_page=1,
        start_y=100,
        end_page=1,
        end_y=240,
        page_numbers=[1],
        blocks=[prompt],
        full_question_label="4",
    )
    identity = PaperIdentity(
        syllabus="9709",
        subject_family="pm1",
        year=2016,
        session_code="s16",
        canonical_session="summer16",
        component="12",
        paper_id="12summer16",
        question_id="12summer16_q04",
    )
    asset = AssetPath(
        kind="question_image",
        paper_id="12summer16",
        question_id="12summer16_q04",
        component="12",
        canonical_path="pm1/pm1_2016_s16_12_qp_q04_question.png",
        absolute_path=Path("/tmp/12summer16_q04.png"),
    )
    regions, flags = _detect_prompt_regions(
        test_span,
        [PageLayout(page_number=1, width=595, height=842, blocks=[prompt], graphics=[])],
        config,
    )

    diagnostics = _crop_diagnostics(Path("9709_s16_qp_12.pdf"), test_span, regions, flags, identity=identity, asset=asset)

    assert diagnostics["detected_figure_count"] == 0
    assert diagnostics["missing_image_reason"] == "detection_failure"
