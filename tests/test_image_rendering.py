from pathlib import Path

import pytest

from exam_bank.config import AppConfig
from exam_bank.image_rendering import (
    CropRegion,
    _anchor_is_current_question_diagram_label,
    _crop_diagnostics,
    _dedupe_crop_regions,
    _detect_prompt_regions,
    _graphics_for_segment,
    _is_answer_space_text,
    _is_figure_label_or_current_anchor_block,
    _is_prompt_text_block,
    _is_unit_diagram_label_block,
    _page_furniture_box_label,
    _same_page_diagram_union_regions,
    _single_page_union_regions,
    _trim_content_top_padding_from_regions,
    _trim_text_bottom_padding_from_regions,
    _trim_text_only_bottom_padding,
    _trim_text_only_top_padding,
    _trim_union_trailing_answer_rule_padding,
    _trim_permission_footer_from_regions,
    _trim_regions_at_foreign_question_boundaries,
    _trim_vertical_furniture_from_regions,
    _watermark_box_looks_like_current_question_diagram,
)
from exam_bank.core.asset_paths import AssetPath
from exam_bank.core.paper_identity import PaperIdentity
from exam_bank.models import BoundingBox, PageLayout, QuestionSpan, QuestionStart, TextBlock


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


def test_top_right_diagonal_papacambridge_banner_is_watermark_furniture() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    banner = BoundingBox(394, 0, 595, 200)

    assert _page_furniture_box_label(banner, layout, AppConfig(), []) == "watermark"


def test_right_edge_diagonal_papacambridge_fragment_is_watermark_furniture() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    fragment = BoundingBox(498, 325, 595, 430)

    assert _page_furniture_box_label(fragment, layout, AppConfig(), []) == "watermark"


def test_top_left_watermark_fragment_is_watermark_furniture() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    fragment = BoundingBox(0, 10.9, 250, 132.4)

    assert _page_furniture_box_label(fragment, layout, AppConfig(), []) == "watermark"


def test_bottom_edge_footer_graphic_is_furniture() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    footer = BoundingBox(2, 782, 592, 842)

    assert _page_furniture_box_label(footer, layout, AppConfig(), []) == "header_footer"


def test_broad_top_watermark_over_plain_text_is_not_current_diagram() -> None:
    config = AppConfig()
    blocks = [
        text_block("10 The complex number w is given by w = -1/2 + i sqrt(3)/2.", 55, x=49, width=496),
        text_block("(i) Find the modulus and argument of w.", 102, x=72, width=300),
        text_block("(iii) Hence explain why, in an Argand diagram, the points represent z, wz and z/w.", 145, x=72, width=470),
    ]
    test_span = QuestionSpan(
        source_pdf=Path("9709_w08_qp_3.pdf"),
        paper_name="9709_w08_qp_3",
        question_number="10",
        start_page=1,
        start_y=45,
        end_page=1,
        end_y=284,
        page_numbers=[1],
        blocks=blocks,
        full_question_label="10",
    )
    layout = PageLayout(page_number=1, width=595, height=842, blocks=blocks, graphics=[BoundingBox(64, 0, 595, 208)])

    assert not _watermark_box_looks_like_current_question_diagram(
        BoundingBox(64, 0, 595, 208),
        test_span,
        blocks,
        layout,
        config,
    )


def test_inset_full_page_graphic_is_background_furniture() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    page_box = BoundingBox(41.58, 0, 595.28, 841.89)

    assert _page_furniture_box_label(page_box, layout, AppConfig(), []) == "page_background"


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


def test_non_visual_prompt_ignores_broad_nearby_graphic() -> None:
    config = AppConfig()
    prompt = text_block("4 Find the possible values of alpha and beta. [6]", 396, x=49, width=496)
    broad_artifact = BoundingBox(20, 377, 595, 416)
    test_span = span()
    test_span.question_number = "4"
    test_span.blocks = [prompt]
    test_span.start_y = 396
    test_span.end_y = 465
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[prompt], graphics=[broad_artifact])

    regions, _flags = _detect_prompt_regions(test_span, [layout], config)

    assert len(regions) == 1
    assert regions[0].region_kind == "text"
    assert not regions[0].graphics
    assert regions[0].bbox.y0 > broad_artifact.y0


def test_scan_junk_after_dotted_answer_line_is_answer_space() -> None:
    assert _is_answer_space_text("." * 120 + "ĬÕĊ®Ġ´íÈõÏĪ°Ċàù·þ×")


def test_top_page_text_only_crop_trims_padding_that_can_contain_barcode() -> None:
    config = AppConfig()
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    crop = BoundingBox(40, 55, 555, 89)
    text = BoundingBox(50, 65, 545, 79)

    trimmed = _trim_text_only_top_padding(crop, text, layout, config)

    assert trimmed.y0 == pytest.approx(63)
    assert trimmed.y1 == crop.y1


def test_text_only_crop_trims_bottom_padding_that_can_contain_answer_rule() -> None:
    config = AppConfig()
    crop = BoundingBox(40, 62, 555, 90)
    text = BoundingBox(50, 65, 545, 79)

    trimmed = _trim_text_only_bottom_padding(crop, text, config)

    assert trimmed.y1 == pytest.approx(81)
    assert trimmed.y0 == crop.y0


def test_text_region_postpass_trims_bottom_spillover_padding() -> None:
    config = AppConfig()
    question = text_block("(iii) Find the values of t. [4]", 570, x=72)
    spillover = text_block("9 The next question starts here.", 620, x=62)
    region = CropRegion(
        page_number=1,
        bbox=BoundingBox(62, 434, 592, 627),
        text_blocks=[question],
        text_bbox=BoundingBox(72, 570, 545, 595),
        region_kind="text",
    )
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[question, spillover])

    trimmed, flags = _trim_text_bottom_padding_from_regions([region], [layout], config)

    assert "text_bottom_padding_trimmed" in flags
    assert trimmed[0].bbox.y1 < 600
    assert trimmed[0].bbox.y1 > region.text_bbox.y1


def test_union_crop_trims_trailing_answer_rule_padding() -> None:
    config = AppConfig()
    dotted_rule = TextBlock(page_number=1, text="................................................", bbox=BoundingBox(80, 276, 530, 277))
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[dotted_rule],
        graphics=[],
    )
    crop = BoundingBox(35, 55, 560, 286)
    content = BoundingBox(49, 65, 545, 263)

    trimmed, flags = _trim_union_trailing_answer_rule_padding(crop, content, layout, config)

    assert "trailing_answer_rule_trimmed" in flags
    assert trimmed.y1 < 276
    assert trimmed.y1 >= content.y1


def test_union_crop_trims_trailing_graphic_answer_rule_padding() -> None:
    config = AppConfig()
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[],
        graphics=[BoundingBox(80, 276, 530, 277)],
    )
    crop = BoundingBox(35, 55, 560, 286)
    content = BoundingBox(49, 65, 545, 263)

    trimmed, flags = _trim_union_trailing_answer_rule_padding(crop, content, layout, config)

    assert "trailing_answer_rule_trimmed" in flags
    assert trimmed.y1 < 276


def test_text_only_prompt_splits_across_answer_rule_band() -> None:
    config = AppConfig()
    blocks = [
        text_block("10 (a) Find the quotient and remainder. [2]", 80, x=50, width=500),
        text_block("(b) Find the exact value of the integral. [6]", 150, x=50, width=500),
    ]
    test_span = span()
    test_span.blocks = blocks
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=blocks,
        graphics=[BoundingBox(55, 122, 540, 122.5)],
    )

    regions, flags = _detect_prompt_regions(test_span, [layout], config)

    assert len(regions) == 2
    assert regions[0].bbox.y1 < 122
    assert regions[1].bbox.y0 > 122
    assert "crop_split_prompt_regions" in flags


def test_text_only_prompt_splits_across_dotted_answer_space_text() -> None:
    config = AppConfig()
    blocks = [
        text_block("(c) Prove the stated result. [2]", 80, x=50, width=500),
        text_block("(d) Hence find the exact value. [3]", 150, x=50, width=500),
    ]
    dotted_rule = TextBlock(
        page_number=1,
        text="................................................",
        bbox=BoundingBox(60, 122, 540, 123),
    )
    test_span = span()
    test_span.blocks = blocks
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[*blocks, dotted_rule], graphics=[])

    regions, flags = _detect_prompt_regions(test_span, [layout], config)

    assert len(regions) == 2
    assert regions[0].bbox.y1 < dotted_rule.bbox.y0
    assert regions[1].bbox.y0 > dotted_rule.bbox.y1
    assert "crop_split_prompt_regions" in flags


def test_figure_separated_text_does_not_restore_answer_rule_top_padding() -> None:
    config = AppConfig()
    dotted_rule = TextBlock(
        page_number=1,
        text="................................................................................",
        bbox=BoundingBox(95, 308, 545, 323),
    )
    part = TextBlock(
        page_number=1,
        text="(d) Using your answers to part (b), prove the identity. [3]",
        bbox=BoundingBox(72, 327, 545, 357),
    )
    formula_graphic = BoundingBox(330, 329, 360, 355)
    test_span = span()
    test_span.blocks = [part]
    test_span.start_y = 300
    test_span.end_y = 370
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[dotted_rule, part], graphics=[formula_graphic])

    regions, _flags = _detect_prompt_regions(test_span, [layout], config)

    text_regions = [region for region in regions if region.region_kind == "text"]
    assert text_regions
    assert min(region.bbox.y0 for region in text_regions) > dotted_rule.bbox.y1


def test_text_crop_trims_top_padding_when_answer_space_text_sits_above_part() -> None:
    config = AppConfig()
    part_a = text_block("10 (a) Find the quotient and remainder. [2]", 64, x=50, width=495)
    answer_line = text_block("." * 120, 546, x=95, width=450)
    part_b = text_block("(b) Find the exact value of the integral. [6]", 554, x=72, width=470)
    test_span = span()
    test_span.blocks = [part_a, answer_line, part_b]
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[part_a, answer_line, part_b], graphics=[])

    regions, flags = _detect_prompt_regions(test_span, [layout], config)

    assert len(regions) == 2
    assert regions[1].bbox.y0 >= part_b.bbox.y0 - 3
    assert "crop_split_prompt_regions" in flags


def test_full_sentence_question_start_is_not_a_figure_label() -> None:
    config = AppConfig()
    block = text_block("8 The diagram shows the graph of y = sec x. [3]", 265, x=72, width=300)
    test_span = span()
    test_span.question_number = "8"

    assert not _is_figure_label_or_current_anchor_block(block, test_span, config)


def test_unit_bearing_angle_label_is_figure_label() -> None:
    config = AppConfig()
    block = text_block("x rad", 154, x=320, width=24)
    test_span = span()

    assert _is_unit_diagram_label_block(block, test_span, config)


def test_oversized_graphic_is_trimmed_before_repeated_prompt_prose() -> None:
    config = AppConfig()
    label = text_block("8 y", 65, x=50, width=190)
    axis = text_block("O 1_{2}0 x", 225, x=240, width=130)
    prose = text_block("The diagram shows the graph of y = sec x for 0 <= x < 1/2 pi.", 264, x=72, width=270)
    part = text_block("(i) Use the trapezium rule with 2 intervals. [3]", 300, x=80, width=460)
    test_span = span()
    test_span.question_number = "8"
    test_span.blocks = [label, axis, prose, part]
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[label, axis, prose, part],
        graphics=[BoundingBox(65, 94, 348, 296)],
    )
    text_box = BoundingBox(49, 65, 545, 326)

    graphics, _excluded = _graphics_for_segment(text_box, layout, config, span=test_span, segment=test_span.blocks)

    assert graphics
    assert graphics[0].y1 < prose.bbox.y0
    assert graphics[0].y1 > axis.bbox.y1


def test_duplicate_axis_label_is_removed_from_following_text_region() -> None:
    config = AppConfig()
    label = text_block("8 y", 65, x=50, width=190)
    axis = text_block("O 1_{2}0 x", 225, x=240, width=130)
    prose = text_block("The diagram shows the graph of y = sec x for 0 <= x < 1/2 pi.", 264, x=72, width=270)
    part = text_block("(i) Use the trapezium rule with 2 intervals. [3]", 300, x=80, width=460)
    test_span = span()
    test_span.question_number = "8"
    test_span.start_y = 65
    test_span.end_y = 340
    test_span.blocks = [label, axis, prose, part]
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[label, axis, prose, part],
        graphics=[BoundingBox(65, 94, 348, 296)],
    )

    regions, flags = _detect_prompt_regions(test_span, [layout], config)
    text_blocks = [block.text for region in regions if region.region_kind == "text" for block in region.text_blocks]
    figure_regions = [region for region in regions if region.graphics]

    assert "duplicate_figure_label_block_excluded" in flags
    assert figure_regions[0].bbox.y1 < prose.bbox.y0
    assert "O 1_{2}0 x" not in text_blocks
    assert any("The diagram shows" in text for text in text_blocks)


def test_mark_bearing_text_below_formula_is_not_removed_after_figure_trim() -> None:
    config = AppConfig()
    intro = text_block("7 (a) Use the substitution u = x^2 - 3 to show that", 65, x=50, width=250)
    formula = text_block("integral expression equals transformed integral,", 98, x=220, width=160)
    where = text_block("where a and b are values to be found. [4]", 145, x=95, width=450)
    test_span = span()
    test_span.question_number = "7"
    test_span.start_y = 64
    test_span.end_y = 170
    test_span.blocks = [intro, formula, where]
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[intro, formula, where],
        graphics=[BoundingBox(225, 98, 378, 129), BoundingBox(337, 114, 348, 125)],
    )

    regions, flags = _detect_prompt_regions(test_span, [layout], config)
    rendered_text = "\n".join(block.text for region in regions for block in region.text_blocks)

    assert "where a and b" in rendered_text
    assert "text_region_removed_after_figure_trim" not in flags


def test_numeric_leading_body_continuation_is_not_foreign_question_start() -> None:
    config = AppConfig()
    continuation = TextBlock(
        page_number=1,
        text="2 decimal places. Give the result of each iteration to 4 decimal places. [3]",
        bbox=BoundingBox(96.36, 290, 545, 302),
    )
    test_span = span()
    test_span.question_number = "3"
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[continuation])

    assert _is_prompt_text_block(continuation, test_span, layout, config)


def test_wide_text_crop_expands_to_preserve_right_edge_glyphs() -> None:
    config = AppConfig()
    prompt = TextBlock(
        page_number=1,
        text="1 Use logarithms to solve the equation, giving your answer correct to 3 significant figures.",
        bbox=BoundingBox(49.32, 60.3, 545.41, 90.32),
    )
    test_span = span()
    test_span.question_number = "1"
    test_span.start_y = 60
    test_span.end_y = 102
    test_span.blocks = [prompt]
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[prompt], graphics=[])

    regions, _flags = _detect_prompt_regions(test_span, [layout], config)

    assert len(regions) == 1
    assert regions[0].bbox.x1 > 590


def test_wide_text_crop_does_not_expand_into_right_side_panel() -> None:
    config = AppConfig()
    prompt = TextBlock(
        page_number=1,
        text="1 Solve the inequality |3x + 2| < 3|2x - 1|.",
        bbox=BoundingBox(49.64, 66.66, 545.73, 78.17),
    )
    side_panel = BoundingBox(574.5, -9.92, 595.5, 854.65)
    test_span = span()
    test_span.question_number = "1"
    test_span.start_y = 60
    test_span.end_y = 90
    test_span.blocks = [prompt]
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[prompt], graphics=[side_panel])

    regions, flags = _detect_prompt_regions(test_span, [layout], config)

    assert "side_panel_excluded" in flags
    assert len(regions) == 1
    assert regions[0].bbox.x1 <= 560


def test_source_pagination_note_is_not_rendered_as_question_text() -> None:
    config = AppConfig()
    question = text_block("(iii) Find the value of k. [4]", 690, x=72)
    note = TextBlock(
        page_number=1,
        text="[Question 10 is printed on the next page.]",
        bbox=BoundingBox(190, 755, 410, 768),
    )
    test_span = span()
    test_span.blocks = [question, note]
    test_span.start_y = 680
    test_span.end_y = 780
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[question, note], graphics=[])

    regions, _flags = _detect_prompt_regions(test_span, [layout], config)
    rendered_text = "\n".join(block.text for region in regions for block in region.text_blocks)

    assert "printed on the next page" not in rendered_text
    assert regions[0].bbox.y1 < note.bbox.y0


def test_full_height_page_edge_graphic_is_treated_as_furniture() -> None:
    config = AppConfig()
    text_box = BoundingBox(50, 70, 545, 130)
    page_edge_furniture = BoundingBox(0, 0, 294.6, 841.89)
    diagram = BoundingBox(330, 95, 430, 190)
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[],
        graphics=[page_edge_furniture, diagram],
    )

    graphics, excluded = _graphics_for_segment(text_box, layout, config)

    assert page_edge_furniture not in graphics
    assert diagram in graphics
    assert {"label": "page_edge_furniture", "bbox": {"x0": 0, "y0": 0, "x1": 294.6, "y1": 841.89}} in excluded


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


def test_page_diagram_union_trims_padding_after_page_edge_furniture_exclusion() -> None:
    config = AppConfig()
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    page_edge_furniture = {"label": "page_edge_furniture", "bbox": {"x0": 0, "y0": 0, "x1": 170, "y1": 842}}
    diagram = BoundingBox(205, 95, 395, 265)
    regions = [
        CropRegion(
            page_number=1,
            bbox=BoundingBox(40, 58, 560, 275),
            text_blocks=[text_block("2", 73, x=50, width=20)],
            graphics=[diagram],
            region_kind="figure",
            excluded_regions=[page_edge_furniture],
        ),
        CropRegion(
            page_number=1,
            bbox=BoundingBox(60, 280, 555, 400),
            text_blocks=[text_block("A particle is attached to a string. [3]", 285, x=72)],
            region_kind="text",
            excluded_regions=[page_edge_furniture],
        ),
    ]

    merged, flags = _same_page_diagram_union_regions(regions, span(), [layout], config)

    assert len(merged) == 1
    assert merged[0].region_kind == "page_diagram_union"
    assert merged[0].bbox.y0 >= 71
    assert merged[0].figure_bbox == diagram
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


def test_crop_region_dedupe_removes_lower_overlapping_figure_fragment() -> None:
    current = CropRegion(
        page_number=1,
        bbox=BoundingBox(39, 199, 438, 434),
        graphics=[BoundingBox(39, 199, 438, 434)],
        region_kind="figure",
    )
    stale_answer_axes = CropRegion(
        page_number=1,
        bbox=BoundingBox(35, 309, 560, 434),
        graphics=[BoundingBox(35, 309, 560, 434)],
        text_blocks=[text_block("O", 319, x=304, width=12), text_block("x", 319, x=438, width=12)],
        region_kind="figure",
    )

    deduped, flags = _dedupe_crop_regions([current, stale_answer_axes])

    assert deduped == [current]
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


def test_single_page_union_trims_accidental_later_question_tail() -> None:
    config = AppConfig()
    current_question = [
        TextBlock(page_number=1, text="8 The function is defined for x.", bbox=BoundingBox(50, 120, 360, 138)),
        TextBlock(page_number=1, text="(i) Find the inverse. [3]", bbox=BoundingBox(72, 180, 340, 196)),
    ]
    foreign_number = TextBlock(page_number=1, text="9", bbox=BoundingBox(50, 260, 60, 276))
    foreign_prompt = TextBlock(page_number=1, text="The equation of a curve is y = x^3. [3]", bbox=BoundingBox(72, 260, 430, 276))
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[*current_question, foreign_number, foreign_prompt],
    )
    test_span = QuestionSpan(
        source_pdf=Path("9709_s15_qp_11.pdf"),
        paper_name="9709_s15_qp_11",
        question_number="8",
        start_page=1,
        start_y=120,
        end_page=1,
        end_y=300,
        page_numbers=[1],
        blocks=[*current_question, foreign_number, foreign_prompt],
        full_question_label="8",
    )
    regions = [
        CropRegion(
            page_number=1,
            bbox=BoundingBox(40, 112, 450, 280),
            text_blocks=[*current_question, foreign_number, foreign_prompt],
            region_kind="text",
        ),
        CropRegion(
            page_number=1,
            bbox=BoundingBox(90, 150, 260, 170),
            graphics=[BoundingBox(90, 150, 260, 170)],
            region_kind="figure",
        )
    ]

    union_regions, flags = _single_page_union_regions(regions, test_span, [layout], config)

    assert union_regions is not None
    assert "foreign_question_boundary_trimmed" in flags
    assert union_regions[0].bbox.y1 < foreign_number.bbox.y0


def test_region_boundary_trim_uses_later_question_anchor_below_span_end() -> None:
    config = AppConfig()
    current_question = [
        TextBlock(page_number=1, text="8 The function is defined for x.", bbox=BoundingBox(50, 120, 360, 138)),
        TextBlock(page_number=1, text="(i) Find the inverse. [3]", bbox=BoundingBox(72, 180, 340, 196)),
    ]
    foreign_number = TextBlock(page_number=1, text="9", bbox=BoundingBox(50, 260, 60, 276))
    foreign_prompt = TextBlock(page_number=1, text="The equation of a curve is y = x^3. [3]", bbox=BoundingBox(72, 260, 430, 276))
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[*current_question, foreign_number, foreign_prompt],
    )
    test_span = QuestionSpan(
        source_pdf=Path("9709_s15_qp_11.pdf"),
        paper_name="9709_s15_qp_11",
        question_number="8",
        start_page=1,
        start_y=120,
        end_page=1,
        end_y=220,
        page_numbers=[1],
        blocks=[*current_question, foreign_number, foreign_prompt],
        full_question_label="8",
    )
    region = CropRegion(
        page_number=1,
        bbox=BoundingBox(40, 112, 450, 280),
        text_blocks=[*current_question, foreign_number, foreign_prompt],
        graphics=[BoundingBox(90, 150, 260, 170)],
        region_kind="figure",
    )

    trimmed, flags = _trim_regions_at_foreign_question_boundaries([region], test_span, [layout], config)

    assert "foreign_question_boundary_trimmed" in flags
    assert trimmed[0].bbox.y1 < foreign_number.bbox.y0


def test_later_question_sentence_near_graphic_is_not_diagram_label() -> None:
    config = AppConfig()
    graphic = BoundingBox(35, 418, 560, 585)
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[], graphics=[graphic])
    test_span = QuestionSpan(
        source_pdf=Path("9709_s15_qp_11.pdf"),
        paper_name="9709_s15_qp_11",
        question_number="8",
        start_page=1,
        start_y=424,
        end_page=1,
        end_y=550,
        page_numbers=[1],
        blocks=[],
        full_question_label="8",
    )
    foreign_anchor = QuestionStart(
        question_number="9",
        page_number=1,
        y0=575,
        x0=49,
        label="The equation of a curve is y = x^3 + px^2, where p is a positive constant.",
        block_index=0,
        bbox=BoundingBox(49, 575, 402, 589),
        confidence=0.9,
    )

    assert not _anchor_is_current_question_diagram_label(foreign_anchor, test_span, layout, config)


def test_later_bare_question_number_near_graphic_is_not_diagram_label() -> None:
    config = AppConfig()
    graphic = BoundingBox(35, 418, 560, 585)
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[], graphics=[graphic])
    test_span = QuestionSpan(
        source_pdf=Path("9709_s15_qp_11.pdf"),
        paper_name="9709_s15_qp_11",
        question_number="8",
        start_page=1,
        start_y=424,
        end_page=1,
        end_y=550,
        page_numbers=[1],
        blocks=[],
        full_question_label="8",
    )
    foreign_anchor = QuestionStart(
        question_number="9",
        page_number=1,
        y0=575,
        x0=49,
        label="9",
        block_index=0,
        bbox=BoundingBox(49, 575, 402, 589),
        confidence=0.9,
    )

    assert not _anchor_is_current_question_diagram_label(foreign_anchor, test_span, layout, config)


def test_later_numeric_graph_label_outside_anchor_column_is_diagram_label() -> None:
    config = AppConfig()
    graphic = BoundingBox(54, 96, 560, 277)
    prompt = text_block("The diagram shows the velocity-time graph for a train.", 247, x=72, width=470)
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[], graphics=[graphic])
    test_span = QuestionSpan(
        source_pdf=Path("9709_s18_qp_43.pdf"),
        paper_name="9709_s18_qp_43",
        question_number="1",
        start_page=1,
        start_y=65,
        end_page=1,
        end_y=635,
        page_numbers=[1],
        blocks=[prompt],
        full_question_label="1",
    )
    graph_label_anchor = QuestionStart(
        question_number="16",
        page_number=1,
        y0=110,
        x0=124,
        label="16",
        block_index=0,
        bbox=BoundingBox(124, 110, 136, 121),
        confidence=0.76,
    )

    assert _anchor_is_current_question_diagram_label(graph_label_anchor, test_span, layout, config)


def test_single_page_union_trims_accidental_previous_question_head() -> None:
    config = AppConfig()
    previous_tail = TextBlock(page_number=1, text="(iv) Find the value of k. [2]", bbox=BoundingBox(70, 92, 360, 108))
    current_question = [
        TextBlock(page_number=1, text="10 Functions f and g are defined by", bbox=BoundingBox(50, 160, 360, 176)),
        TextBlock(page_number=1, text="(i) Evaluate fg(2). [2]", bbox=BoundingBox(72, 245, 330, 261)),
    ]
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[previous_tail, *current_question],
    )
    test_span = QuestionSpan(
        source_pdf=Path("9709_s11_qp_13.pdf"),
        paper_name="9709_s11_qp_13",
        question_number="10",
        start_page=1,
        start_y=160,
        end_page=1,
        end_y=320,
        page_numbers=[1],
        blocks=[previous_tail, *current_question],
        full_question_label="10",
    )
    regions = [
        CropRegion(
            page_number=1,
            bbox=BoundingBox(40, 82, 450, 300),
            text_blocks=[previous_tail, *current_question],
            region_kind="text",
        ),
        CropRegion(
            page_number=1,
            bbox=BoundingBox(90, 190, 260, 220),
            graphics=[BoundingBox(90, 190, 260, 220)],
            region_kind="figure",
        )
    ]

    union_regions, flags = _single_page_union_regions(regions, test_span, [layout], config)

    assert union_regions is not None
    assert "crop_header_padding_trimmed" in flags
    assert union_regions[0].bbox.y0 >= test_span.start_y - 24.0


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


def test_question_context_does_not_infer_tiny_square_furniture_as_figure() -> None:
    config = AppConfig()
    prompt = text_block("6 On an Argand diagram shade the region. [5]", 80, x=60, width=450)
    continuation = text_block("(b) Calculate the greatest value of arg z. [2]", 420, x=72, width=430)
    square = BoundingBox(544, 40, 560, 56)
    test_span = QuestionSpan(
        source_pdf=Path("9709_m24_qp_33.pdf"),
        paper_name="9709_m24_qp_33",
        question_number="6",
        start_page=1,
        start_y=70,
        end_page=2,
        end_y=450,
        page_numbers=[1, 2],
        blocks=[prompt, continuation],
        full_question_label="6",
    )
    layout_1 = PageLayout(page_number=1, width=595, height=842, blocks=[prompt], graphics=[])
    layout_2 = PageLayout(page_number=2, width=595, height=842, blocks=[continuation], graphics=[square])

    regions, flags = _detect_prompt_regions(test_span, [layout_1, layout_2], config)

    assert not any(region.region_kind == "context_inferred_figure" for region in regions)
    assert "question_context_figure_inference_used" not in flags


def test_text_only_top_padding_trims_sparse_fragments_above_prompt() -> None:
    config = AppConfig()
    crop_box = BoundingBox(35, 45, 560, 220)
    text_box = BoundingBox(50, 112, 545, 190)
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[], graphics=[])

    trimmed = _trim_text_only_top_padding(crop_box, text_box, layout, config)

    assert trimmed.y0 > 100
    assert trimmed.y1 == crop_box.y1


def test_region_content_top_padding_trim_handles_union_crops() -> None:
    config = AppConfig()
    region = CropRegion(
        page_number=1,
        bbox=BoundingBox(35, 45, 560, 260),
        text_blocks=[text_block("4 The complex number u is defined by", 120)],
        graphics=[BoundingBox(220, 170, 360, 190)],
        region_kind="single_page_union",
        text_bbox=BoundingBox(50, 120, 545, 210),
        figure_bbox=BoundingBox(220, 170, 360, 190),
    )

    trimmed, flags = _trim_content_top_padding_from_regions([region], config)

    assert trimmed[0].bbox.y0 > 100
    assert trimmed[0].bbox.y1 == region.bbox.y1
    assert "crop_top_padding_trimmed" in flags


def test_region_content_top_padding_preserves_diagram_above_text() -> None:
    config = AppConfig()
    region = CropRegion(
        page_number=1,
        bbox=BoundingBox(35, 45, 560, 260),
        text_blocks=[text_block("The diagram shows a curve.", 180)],
        graphics=[BoundingBox(180, 58, 420, 145)],
        region_kind="page_diagram_union",
        text_bbox=BoundingBox(50, 180, 545, 220),
        figure_bbox=BoundingBox(180, 58, 420, 145),
    )

    trimmed, flags = _trim_content_top_padding_from_regions([region], config)

    assert trimmed[0].bbox.y0 < region.figure_bbox.y0
    assert trimmed[0].bbox.y1 == region.bbox.y1


def test_text_only_graph_axis_labels_are_rendered_as_single_diagram_region() -> None:
    config = AppConfig()
    prompt = text_block("5 Hence sketch a displacement-time graph for the race. [6]", 120, x=72)
    label_top = text_block("displacement (m)", 410, x=125, width=90)
    label_value = text_block("200", 442, x=145, width=18)
    label_axis = text_block("0 time (s)", 725, x=155, width=360)
    label_ticks = text_block("0 20", 738, x=165, width=295)
    continuation = text_block("(ii) Find the value of V. [2]", 790, x=72)
    test_span = QuestionSpan(
        source_pdf=Path("9709_s18_qp_41.pdf"),
        paper_name="9709_s18_qp_41",
        question_number="5",
        start_page=1,
        start_y=100,
        end_page=1,
        end_y=820,
        page_numbers=[1],
        blocks=[prompt, label_top, label_value, label_axis, label_ticks, continuation],
        full_question_label="5(i)-(ii)",
    )
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[prompt, label_top, label_value, label_axis, label_ticks, continuation],
        graphics=[],
    )

    regions, flags = _detect_prompt_regions(test_span, [layout], config)

    diagram_regions = [region for region in regions if region.region_kind == "text_diagram_union"]
    assert len(diagram_regions) == 1
    assert "text_only_diagram_union_used" in flags
    assert diagram_regions[0].bbox.y0 <= label_top.bbox.y0
    assert diagram_regions[0].bbox.y1 >= label_ticks.bbox.y1
    rendered_text = "\n".join(block.text for region in regions if region.region_kind != "text_diagram_union" for block in region.text_blocks)
    assert "displacement (m)" not in rendered_text
    assert "0 time (s)" not in rendered_text
    assert "(ii) Find the value of V. [2]" in rendered_text


def test_text_only_diagram_region_keeps_nearby_barcode_shaped_graphic() -> None:
    config = AppConfig()
    prompt = text_block("9 The diagram shows the graph of y = f(x). [2]", 250, x=72)
    label_top = text_block("9 y", 70, x=50, width=130)
    label_curve = text_block("y = f(x)", 125, x=380, width=45)
    label_axis = text_block("O 1_{2}0 0 x", 210, x=175, width=270)
    graph = BoundingBox(187, 114, 410, 171)
    test_span = QuestionSpan(
        source_pdf=Path("9709_s19_qp_13.pdf"),
        paper_name="9709_s19_qp_13",
        question_number="9",
        start_page=1,
        start_y=65,
        end_page=1,
        end_y=340,
        page_numbers=[1],
        blocks=[label_top, label_curve, label_axis, prompt],
        full_question_label="9",
    )
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[label_top, label_curve, label_axis, prompt],
        graphics=[graph],
    )

    regions, flags = _detect_prompt_regions(test_span, [layout], config)

    diagram_regions = [region for region in regions if region.region_kind == "text_diagram_union"]
    assert len(diagram_regions) == 1
    assert diagram_regions[0].graphics == [graph]
    assert diagram_regions[0].bbox.y0 <= label_top.bbox.y0
    assert diagram_regions[0].bbox.y1 >= label_axis.bbox.y1
    assert "missing_image_detection_failure" not in flags


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

    diagnostics = _crop_diagnostics(
        Path("9709_s16_qp_12.pdf"),
        test_span,
        regions,
        flags,
        identity=identity,
        asset=asset,
        config=config,
    )

    assert diagnostics["detected_figure_count"] == 0
    assert diagnostics["missing_image_reason"] == "detection_failure"


def test_student_generated_sketch_prompt_is_not_marked_detection_failure() -> None:
    config = AppConfig()
    prompt = text_block("4 Sketch, on a single diagram, the graphs of y = cos x and y = 1/2. [3]", 110, x=60, width=420)
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

    diagnostics = _crop_diagnostics(
        Path("9709_s16_qp_12.pdf"),
        test_span,
        regions,
        flags,
        identity=identity,
        asset=asset,
        config=config,
    )

    assert "missing_image_detection_failure" not in flags
    assert diagnostics["detected_figure_count"] == 0
    assert diagnostics["missing_image_reason"] == ""


def test_permission_footer_phrase_is_trimmed_from_final_question_crop() -> None:
    config = AppConfig()
    question = text_block("(b) Hence find x. [3]", 690, x=72)
    footer = TextBlock(
        page_number=1,
        text="Permission to reproduce items where third-party owned material protected by copyright is included has been sought.",
        bbox=BoundingBox(45, 780, 550, 788),
        font_size=6,
    )
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[question, footer])
    region = CropRegion(page_number=1, bbox=BoundingBox(35, 620, 560, 802), text_blocks=[question], region_kind="text")

    trimmed, flags = _trim_permission_footer_from_regions([region], [layout], config)

    assert "permission_footer_trimmed" in flags
    assert trimmed[0].bbox.y1 < footer.bbox.y0
    assert trimmed[0].bbox.y1 > question.bbox.y1
    assert trimmed[0].footer_cutoff["reason"] == "footer_phrase"
    assert trimmed[0].footer_cutoff["signals"] == ["footer_phrase"]


def test_permission_footer_detector_leaves_normal_bottom_question_unchanged() -> None:
    config = AppConfig()
    question = text_block("(c) Find the area of the shaded region. [5]", 735, x=72)
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[question])
    region = CropRegion(page_number=1, bbox=BoundingBox(35, 640, 560, 790), text_blocks=[question], region_kind="text")

    trimmed, flags = _trim_permission_footer_from_regions([region], [layout], config)

    assert flags == []
    assert trimmed[0].bbox == region.bbox
    assert trimmed[0].footer_cutoff["reason"] == "not_detected"
    assert trimmed[0].footer_cutoff["final_bottom"] == 790


def test_permission_footer_detector_does_not_overtrim_question_content_near_bottom() -> None:
    config = AppConfig()
    final_line = text_block("State the exact value of k. [2]", 768, x=72)
    footer = TextBlock(
        page_number=1,
        text="UCLES 2008",
        bbox=BoundingBox(45, 782, 105, 790),
        font_size=6,
    )
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[final_line, footer])
    region = CropRegion(page_number=1, bbox=BoundingBox(35, 700, 560, 804), text_blocks=[final_line], region_kind="text")

    trimmed, flags = _trim_permission_footer_from_regions([region], [layout], config)

    assert "permission_footer_trimmed" not in flags
    assert "permission_footer_trim_skipped_protected_content" in flags
    assert trimmed[0].bbox == region.bbox
    assert trimmed[0].footer_cutoff["reason"] == "skipped_protected_content"


def test_permission_footer_rule_plus_cambridge_text_is_trimmed_above_rule() -> None:
    config = AppConfig()
    question = text_block("(iii) Complete the proof. [4]", 650, x=72)
    rule = BoundingBox(35, 742, 560, 744)
    footer_1 = TextBlock(
        page_number=1,
        text="University of Cambridge International Examinations",
        bbox=BoundingBox(45, 758, 305, 766),
        font_size=6,
    )
    footer_2 = TextBlock(
        page_number=1,
        text="Cambridge Assessment is the brand name of the University of Cambridge Local Examinations Syndicate",
        bbox=BoundingBox(45, 770, 550, 778),
        font_size=6,
    )
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[question, footer_1, footer_2], graphics=[rule])
    region = CropRegion(page_number=1, bbox=BoundingBox(35, 600, 560, 805), text_blocks=[question], region_kind="text")

    trimmed, flags = _trim_permission_footer_from_regions([region], [layout], config)

    assert "permission_footer_trimmed" in flags
    assert trimmed[0].bbox.y1 < rule.y0
    assert trimmed[0].bbox.y1 > question.bbox.y1
    assert trimmed[0].footer_cutoff["reason"] == "horizontal_rule_with_footer_phrase"
    assert trimmed[0].footer_cutoff["signals"] == ["horizontal_rule", "footer_phrase"]


def test_permission_footer_only_region_is_removed() -> None:
    config = AppConfig()
    question = text_block("Find the perimeter of ABCD. [3]", 555, x=72)
    footer = TextBlock(
        page_number=1,
        text="University of Cambridge International Examinations is part of the Cambridge Assessment Group.",
        bbox=BoundingBox(45, 771, 520, 778),
        font_size=7,
    )
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[question, footer])
    question_region = CropRegion(page_number=1, bbox=BoundingBox(35, 520, 560, 580), text_blocks=[question], region_kind="text")
    footer_region = CropRegion(page_number=1, bbox=BoundingBox(35, 762, 560, 798), text_blocks=[footer], region_kind="text")

    trimmed, flags = _trim_permission_footer_from_regions([question_region, footer_region], [layout], config)

    assert "permission_footer_region_removed" in flags
    assert len(trimmed) == 1
    assert trimmed[0].text_blocks == [question]
