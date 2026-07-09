from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable
import json
import re

from .config import AppConfig
from .core.asset_paths import AssetPath, AssetPathResolver
from .core.paper_identity import IdentityError, PaperIdentity, paper_identity_from_parts, session_for_source_path
from .document_metadata import parse_filename_metadata
from .image_limits import cap_image_pixels, clean_rendered_crop_image, render_pdf_area
from .models import BoundingBox, PageLayout, QuestionSpan, QuestionStart, RenderResult, TextBlock
from .mupdf_tools import quiet_mupdf
from .ocr import run_question_crop_ocr
from .question_detection_layout import looks_like_diagram_axis_or_label_text as _looks_like_diagram_axis_or_label_text
from .question_detection import detect_question_anchor_candidates, extract_text_from_blocks, parse_question_start
from .trust import references_source_visual


@dataclass
class CropRegion:
    page_number: int
    bbox: BoundingBox
    text_blocks: list[TextBlock] = field(default_factory=list)
    graphics: list[BoundingBox] = field(default_factory=list)
    duplicate_graphics_removed: int = 0
    original_bbox: BoundingBox | None = None
    excluded_regions: list[dict[str, object]] = field(default_factory=list)
    region_kind: str = "combined"
    text_bbox: BoundingBox | None = None
    figure_bbox: BoundingBox | None = None
    text_figure_overlap_area: float = 0.0
    text_trimmed_for_figure: bool = False
    footer_cutoff: dict[str, object] | None = None


def render_question_image(
    pdf_path: str | Path,
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
    *,
    identity: PaperIdentity | None = None,
) -> RenderResult:
    """Render original PDF pixels cropped tightly to prompt content."""

    identity = identity or _question_identity_from_span(span)
    if identity is None:
        return _missing_identity_render_result(span)
    if config.detection.output_mode == "full_region":
        return _render_full_region_image(pdf_path, span, layouts, config, identity=identity)
    return _render_prompt_crop_image(pdf_path, span, layouts, config, identity=identity)


def _render_prompt_crop_image(
    pdf_path: str | Path,
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
    *,
    identity: PaperIdentity,
) -> RenderResult:
    try:
        import fitz
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("PyMuPDF and Pillow are required for rendering screenshots.") from exc
    quiet_mupdf(fitz)

    asset = AssetPathResolver(config.output.root_dir()).question_image(identity)
    output_path = asset.absolute_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regions, flags = _detect_prompt_regions(span, layouts, config)
    union_regions, union_flags = _single_page_union_regions(regions, span, layouts, config)
    if union_regions is not None:
        regions = union_regions
        flags.extend(union_flags)
    else:
        page_union_regions, page_union_flags = _same_page_diagram_union_regions(regions, span, layouts, config)
        regions = page_union_regions
        flags.extend(page_union_flags)
    crop_uncertain = False

    if not regions:
        regions = _fallback_regions(span, layouts, config)
        flags.extend(["crop_fallback_used", "crop_uncertain"])
        crop_uncertain = True

    if any(flag == "ocr_question_text" or flag.startswith("ocr_") for flag in span.review_flags):
        flags.append("crop_uncertain")
        crop_uncertain = True

    crops = []
    debug_paths: list[str] = []

    with fitz.open(pdf_path) as doc:
        rendered_pages = {}
        for region in regions:
            page = doc[region.page_number - 1]
            rect = fitz.Rect(region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1)
            crop, used_zoom, render_flags = _render_pdf_area_for_crop_region(
                doc,
                region,
                fitz,
                dpi=config.detection.render_dpi,
                source_file=pdf_path,
                page_number=region.page_number,
                context=f"question_crop:{span.question_number}",
                clip=rect,
            )
            crops.append(crop)
            flags.extend(render_flags)

            if config.debug.enabled and region.page_number not in rendered_pages:
                page_image, page_zoom = render_pdf_area(
                    page,
                    fitz,
                    dpi=config.detection.render_dpi,
                    source_file=pdf_path,
                    page_number=region.page_number,
                    context=f"question_debug_page:{span.question_number}",
                )
                rendered_pages[region.page_number] = (page_image, page_zoom)
                if config.debug.save_rendered_pages:
                    debug_paths.append(_save_debug_image(page_image, span, region.page_number, "rendered", config))

            layout = _layout_by_number(layouts, region.page_number)
            if _box_height(region.bbox) > layout.height * config.detection.max_crop_height_ratio:
                flags.extend(["crop_reaches_page_margin", "crop_uncertain"])
                crop_uncertain = True

            if used_zoom * 72 < config.detection.render_dpi * 0.8:
                flags.append("render_dpi_capped")

        if config.debug.enabled:
            debug_paths.extend(_write_debug_overlays(rendered_pages, span, layouts, regions, config))

    if not crops:
        raise RuntimeError(f"No crops could be rendered for {span.paper_name} question {span.question_number}.")

    stitched = cap_image_pixels(
        _stitch_images(crops, config.detection.stitch_gap_px),
        source_file=pdf_path,
        context=f"question_output:{span.question_number}",
    )
    stitched = clean_rendered_crop_image(stitched)
    stitched.save(output_path)
    ocr_result = run_question_crop_ocr(output_path, config)
    if ocr_result.ocr_ran and ocr_result.ocr_failure_reason:
        flags.append("ocr_question_crop_failed")

    if config.debug.enabled:
        debug_paths.append(_write_crop_metadata(span, regions, flags, config))

    crop_uncertain = crop_uncertain or "crop_uncertain" in flags
    extracted_text = _text_from_regions(regions) or span.combined_text
    flags = sorted(set(flags))
    crop_diagnostics = _crop_diagnostics(pdf_path, span, regions, flags, identity=identity, asset=asset)
    return RenderResult(
        screenshot_path=output_path,
        review_flags=flags,
        crop_uncertain=crop_uncertain,
        debug_paths=debug_paths,
        extracted_text=extracted_text,
        crop_diagnostics=crop_diagnostics,
        question_id=identity.question_id,
        paper_id=identity.paper_id,
        component=identity.component,
        canonical_path=asset.canonical_path,
        ocr_ran=ocr_result.ocr_ran,
        ocr_engine=ocr_result.ocr_engine,
        ocr_text=ocr_result.ocr_text,
        ocr_text_trust=ocr_result.ocr_text_trust,
        ocr_failure_reason=ocr_result.ocr_failure_reason,
        ocr_text_role=ocr_result.ocr_text_role,
    )


def _render_pdf_area_for_crop_region(
    doc: object,
    region: CropRegion,
    fitz: object,
    *,
    dpi: int,
    source_file: str | Path,
    page_number: int,
    context: str,
    clip: object,
) -> tuple[object, float, list[str]]:
    page = doc[page_number - 1]
    if not _region_has_suppressed_watermark_image(region):
        image, zoom = render_pdf_area(
            page,
            fitz,
            dpi=dpi,
            source_file=source_file,
            page_number=page_number,
            context=context,
            clip=clip,
        )
        return image, zoom, []

    with fitz.open() as temp_doc:
        temp_doc.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
        temp_page = temp_doc[0]
        if not _delete_excluded_watermark_images(temp_page, region):
            image, zoom = render_pdf_area(
                page,
                fitz,
                dpi=dpi,
                source_file=source_file,
                page_number=page_number,
                context=context,
                clip=clip,
            )
            return image, zoom, []
        image, zoom = render_pdf_area(
            temp_page,
            fitz,
            dpi=dpi,
            source_file=source_file,
            page_number=page_number,
            context=context,
            clip=clip,
        )
        return image, zoom, ["source_watermark_image_suppressed"]


def _region_has_suppressed_watermark_image(region: CropRegion) -> bool:
    return any(excluded.get("label") in {"watermark", "page_background"} for excluded in region.excluded_regions)


def _delete_excluded_watermark_images(page: object, region: CropRegion) -> bool:
    excluded_boxes = [
        box
        for excluded in region.excluded_regions
        if excluded.get("label") in {"watermark", "page_background"}
        if (box := _excluded_region_box(excluded)) is not None
    ]
    if not excluded_boxes:
        return False

    removed = False
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        if not any(_image_rect_matches_excluded_watermark(rect, excluded_boxes, region.bbox) for rect in rects):
            continue
        page.delete_image(xref)
        removed = True
    return removed


def _image_rect_matches_excluded_watermark(rect: object, excluded_boxes: list[BoundingBox], crop_box: BoundingBox) -> bool:
    image_box = BoundingBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    if not _boxes_intersect(image_box, crop_box):
        return False
    for excluded_box in excluded_boxes:
        overlap = _intersection_area(image_box, excluded_box)
        if overlap / max(1.0, min(_box_area(image_box), _box_area(excluded_box))) >= 0.25:
            return True
    return False


def _excluded_region_box(excluded: dict[str, object]) -> BoundingBox | None:
    payload = excluded.get("bbox")
    if not isinstance(payload, dict):
        return None
    try:
        return BoundingBox(
            float(payload["x0"]),
            float(payload["y0"]),
            float(payload["x1"]),
            float(payload["y1"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _detect_prompt_regions(
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    regions: list[CropRegion] = []
    flags: list[str] = []
    seen_graphics: dict[int, list[BoundingBox]] = {}

    for page_number in span.page_numbers:
        layout = _layout_by_number(layouts, page_number)
        blocks = [
            block
            for block in span.blocks
            if block.page_number == page_number and _is_prompt_text_block(block, span, layout, config)
        ]
        if not blocks:
            continue

        segments = _split_prompt_segments(blocks, layout, config)
        if len(segments) > 1:
            flags.append("crop_split_prompt_regions")

        for segment in segments:
            text_box = _union_boxes([block.bbox for block in segment])
            raw_graphics, excluded_regions = _graphics_for_segment(text_box, layout, config, span=span, segment=segment)
            for excluded in excluded_regions:
                reason = str(excluded.get("label") or "")
                if reason:
                    flags.append(f"{reason}_excluded")
            graphics, duplicate_count = _dedupe_graphics(raw_graphics, seen_graphics.setdefault(page_number, []))
            if duplicate_count:
                flags.append("duplicate_visual_regions_removed")
                flags.append("duplicate_visual_fragment_excluded")
            if graphics:
                separated, separation_flags = _separate_text_and_figure_regions(
                    page_number,
                    segment,
                    text_box,
                    graphics,
                    duplicate_count,
                    excluded_regions,
                    layout,
                    config,
                    span,
                )
                flags.extend(separation_flags)
                regions.extend(separated)
                continue
            if duplicate_count and _segment_is_figure_label_only(segment, span, config):
                flags.append("duplicate_figure_label_segment_excluded")
                continue

            original_box = text_box.padded(config.detection.crop_padding, layout.width, layout.height)
            crop_box = _clamp_crop_to_prompt_area(original_box, layout, config)
            crop_box = _trim_crop_furniture_edges(crop_box, layout, config)
            crop_box = _trim_padding_for_page_edge_furniture(crop_box, text_box, excluded_regions, config)
            crop_box = _expand_text_crop_for_wide_prompt(crop_box, segment, layout, config, excluded_regions=excluded_regions)
            crop_box = _trim_text_only_top_padding(crop_box, text_box, layout, config)
            crop_box = _trim_text_top_padding_after_answer_rule(crop_box, text_box, layout, config)
            crop_box = _trim_text_bottom_padding_after_answer_rule(crop_box, text_box, layout, config)
            if _box_height(crop_box) < config.detection.min_crop_height:
                flags.append("crop_uncertain")
            regions.append(
                CropRegion(
                    page_number=page_number,
                    bbox=crop_box,
                    text_blocks=segment,
                    duplicate_graphics_removed=duplicate_count,
                    original_bbox=original_box,
                    excluded_regions=excluded_regions,
                    region_kind="text",
                    text_bbox=text_box,
                )
            )

    context_regions, context_flags = _question_context_figure_regions(regions, span, layouts, config)
    if context_regions:
        regions.extend(context_regions)
        flags.extend(context_flags)

    regions, overlap_flags = _remove_meaningful_region_overlaps(regions, config)
    regions, dedupe_flags = _dedupe_crop_regions(regions)
    regions, duplicate_label_flags = _remove_duplicate_figure_labels_from_text_regions(regions, span, layouts, config)
    regions, text_diagram_flags = _separate_text_only_diagram_label_regions(regions, span, layouts, config)
    regions, furniture_flags = _trim_vertical_furniture_from_regions(regions, layouts, config)
    regions, content_top_flags = _trim_content_top_padding_from_regions(regions, config)
    flags.extend(overlap_flags)
    flags.extend(dedupe_flags)
    flags.extend(duplicate_label_flags)
    flags.extend(text_diagram_flags)
    flags.extend(furniture_flags)
    flags.extend(content_top_flags)
    if _span_references_source_visual(span) and not any(region.graphics for region in regions):
        flags.extend(["missing_image_detection_failure", "crop_uncertain"])
    regions, foreign_flags = _trim_regions_at_foreign_question_boundaries(regions, span, layouts, config)
    flags.extend(foreign_flags)
    regions, footer_flags = _trim_permission_footer_from_regions(regions, layouts, config)
    flags.extend(footer_flags)
    regions, text_bottom_flags = _trim_text_bottom_padding_from_regions(regions, layouts, config)
    flags.extend(text_bottom_flags)
    return regions, sorted(set(flags))


def _trim_padding_for_page_edge_furniture(
    crop_box: BoundingBox,
    content_box: BoundingBox,
    excluded_regions: list[dict[str, object]],
    config: AppConfig,
) -> BoundingBox:
    if not any(excluded.get("label") == "page_edge_furniture" for excluded in excluded_regions):
        return crop_box
    top = max(crop_box.y0, content_box.y0 - max(2.0, config.detection.crop_padding * 0.2))
    if top >= crop_box.y1 - config.detection.min_crop_height:
        return crop_box
    return BoundingBox(crop_box.x0, top, crop_box.x1, crop_box.y1)


def _trim_text_only_top_padding(
    crop_box: BoundingBox,
    text_box: BoundingBox,
    layout: PageLayout,
    config: AppConfig,
) -> BoundingBox:
    top = max(crop_box.y0, text_box.y0 - max(2.0, config.detection.crop_padding * 0.2))
    if top >= crop_box.y1 - config.detection.min_crop_height:
        return crop_box
    return BoundingBox(crop_box.x0, top, crop_box.x1, crop_box.y1)


def _trim_text_top_padding_after_answer_rule(
    crop_box: BoundingBox,
    text_box: BoundingBox,
    layout: PageLayout,
    config: AppConfig,
) -> BoundingBox:
    if text_box.y0 <= crop_box.y0 + 2.0:
        return crop_box
    has_answer_rule_graphic = any(crop_box.y0 <= band <= text_box.y0 for band in _answer_rule_y_bands(layout))
    has_answer_rule_text = any(
        block.bbox.y1 >= crop_box.y0
        and block.bbox.y0 <= text_box.y0
        and _is_answer_space_text(_clean_text_line(block.text))
        for block in layout.blocks
    )
    if not has_answer_rule_graphic and not has_answer_rule_text:
        return crop_box
    top = max(crop_box.y0, text_box.y0 - max(1.5, config.detection.crop_padding * 0.2))
    if top >= crop_box.y1 - config.detection.min_crop_height * 0.5:
        return crop_box
    return BoundingBox(crop_box.x0, top, crop_box.x1, crop_box.y1)


def _trim_union_trailing_answer_rule_padding(
    crop_box: BoundingBox,
    content_box: BoundingBox,
    layout: PageLayout,
    config: AppConfig,
) -> tuple[BoundingBox, list[str]]:
    trailing_rule_tops = [
        band
        for band in _answer_rule_y_bands(layout)
        if content_box.y1 + 1.0 <= band <= crop_box.y1 - 1.0
    ]
    trailing_rule_tops.extend(
        block.bbox.y0
        for block in layout.blocks
        if content_box.y1 + 1.0 <= block.bbox.y0 <= crop_box.y1 - 1.0
        and _is_answer_space_text(_clean_text_line(block.text))
    )
    if not trailing_rule_tops:
        return crop_box, []
    bottom = min(trailing_rule_tops) - 2.0
    if bottom <= crop_box.y0 + config.detection.min_crop_height * 0.5:
        return crop_box, []
    if bottom < content_box.y1:
        return crop_box, []
    return BoundingBox(crop_box.x0, crop_box.y0, crop_box.x1, bottom), ["trailing_answer_rule_trimmed"]


def _trim_text_only_bottom_padding(
    crop_box: BoundingBox,
    text_box: BoundingBox,
    config: AppConfig,
) -> BoundingBox:
    bottom = min(crop_box.y1, text_box.y1 + max(2.0, config.detection.crop_padding * 0.2))
    min_preserved_height = max(8.0, config.detection.min_crop_height * 0.5)
    if bottom <= crop_box.y0 + min_preserved_height:
        return crop_box
    return BoundingBox(crop_box.x0, crop_box.y0, crop_box.x1, bottom)


def _trim_text_bottom_padding_after_answer_rule(
    crop_box: BoundingBox,
    text_box: BoundingBox,
    layout: PageLayout,
    config: AppConfig,
) -> BoundingBox:
    lookahead_bottom = crop_box.y1 + max(18.0, config.detection.crop_padding * 2.0)
    has_answer_rule_graphic = any(text_box.y1 <= band <= lookahead_bottom for band in _answer_rule_y_bands(layout))
    has_answer_rule_text = any(
        block.bbox.y0 >= text_box.y1
        and block.bbox.y0 <= lookahead_bottom
        and _is_answer_space_text(_clean_text_line(block.text))
        for block in layout.blocks
    )
    if not has_answer_rule_graphic and not has_answer_rule_text:
        return crop_box
    bottom = min(crop_box.y1, text_box.y1 - 1.0)
    if bottom <= crop_box.y0 + max(8.0, config.detection.min_crop_height * 0.5):
        return crop_box
    return BoundingBox(crop_box.x0, crop_box.y0, crop_box.x1, bottom)


def _trim_text_bottom_padding_from_regions(
    regions: list[CropRegion],
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    trimmed: list[CropRegion] = []
    flags: list[str] = []
    for region in regions:
        if region.region_kind != "text" or region.text_bbox is None:
            trimmed.append(region)
            continue
        layout = _layout_by_number(layouts, region.page_number)
        region_block_keys = {_block_identity_key(block) for block in region.text_blocks}
        trailing_blocks = [
            block
            for block in layout.blocks
            if region.text_bbox.y1 + 1.0 <= block.bbox.y0 <= region.bbox.y1
            and _block_identity_key(block) not in region_block_keys
            and _trailing_text_block_should_trim_bottom_padding(block, layout, config)
        ]
        if not trailing_blocks:
            trimmed.append(region)
            continue
        bbox = _trim_text_only_bottom_padding(region.bbox, region.text_bbox, config)
        if bbox != region.bbox:
            flags.append("text_bottom_padding_trimmed")
            trimmed.append(replace(region, bbox=bbox))
        else:
            trimmed.append(region)
    return trimmed, sorted(set(flags))


def _trailing_text_block_should_trim_bottom_padding(block: TextBlock, layout: PageLayout, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if not text:
        return False
    if _is_answer_space_text(text) or _is_source_pagination_note_text(text):
        return True
    if _is_footer_or_header_box(block.bbox, layout, config) or _is_centered_page_number_block(block, layout, config):
        return False
    if _is_boilerplate_text(text) or _is_margin_furniture_text(block, layout, config) or _is_control_artifact_text(text):
        return False
    return True


def _trim_content_top_padding_from_regions(
    regions: list[CropRegion],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    trimmed: list[CropRegion] = []
    flags: list[str] = []
    for region in regions:
        updated = _trim_region_top_padding_to_content(region, config)
        if updated.bbox.y0 > region.bbox.y0 + 0.5:
            flags.append("crop_top_padding_trimmed")
        trimmed.append(updated)
    return trimmed, sorted(set(flags))


def _trim_region_top_padding_to_content(region: CropRegion, config: AppConfig) -> CropRegion:
    content_boxes: list[BoundingBox] = []
    if region.text_bbox is not None:
        content_boxes.append(region.text_bbox)
    if region.figure_bbox is not None:
        content_boxes.append(region.figure_bbox)
    content_boxes.extend(region.graphics)
    if not content_boxes:
        return region

    content_top = min(box.y0 for box in content_boxes)
    top_padding = max(2.0, config.detection.crop_padding * 0.2)
    top = max(region.bbox.y0, content_top - top_padding)
    if top <= region.bbox.y0 + 0.5:
        return region
    if top >= region.bbox.y1 - max(8.0, config.detection.min_crop_height * 0.5):
        return region
    return replace(
        region,
        bbox=BoundingBox(region.bbox.x0, top, region.bbox.x1, region.bbox.y1),
        original_bbox=region.original_bbox or region.bbox,
    )


def _question_context_figure_regions(
    regions: list[CropRegion],
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    if any(region.graphics for region in regions):
        return [], []
    if not _span_has_figure_prompt(span):
        return [], []

    inferred: list[CropRegion] = []
    flags: list[str] = []
    for page_number in span.page_numbers:
        layout = _layout_by_number(layouts, page_number)
        span_blocks = [block for block in span.blocks if block.page_number == page_number]
        text_box = _union_boxes([block.bbox for block in span_blocks]) if span_blocks else None
        candidates = _context_graphics_for_question(text_box, span, layout, config)
        if not candidates:
            continue
        figure_box = _trim_crop_furniture_edges(
            _clamp_crop_to_prompt_area(
                _union_boxes(candidates).padded(config.detection.crop_padding, layout.width, layout.height),
                layout,
                config,
            ),
            layout,
            config,
        )
        inferred.append(
            CropRegion(
                page_number=page_number,
                bbox=figure_box,
                text_blocks=[
                    block
                    for block in span_blocks
                    if _block_belongs_to_figure(block, figure_box, config) or _is_diagram_label_only_block(block, span, config)
                ],
                graphics=candidates,
                original_bbox=_union_boxes(candidates),
                region_kind="context_inferred_figure",
                figure_bbox=figure_box,
                text_bbox=text_box,
            )
        )
        flags.append("question_context_figure_inference_used")
    return inferred, sorted(set(flags))


def _context_graphics_for_question(
    text_box: BoundingBox | None,
    span: QuestionSpan,
    layout: PageLayout,
    config: AppConfig,
) -> list[BoundingBox]:
    answer_rule_bands = _answer_rule_y_bands(layout)
    top = span.start_y if layout.page_number == span.start_page else config.detection.crop_top_margin
    bottom = span.end_y if layout.page_number == span.end_page else layout.height - config.detection.crop_bottom_margin
    search_radius = max(config.detection.prompt_graphic_lookahead * 1.35, 240.0)
    if text_box is not None:
        start_page_floor = (
            max(config.detection.crop_top_margin, span.start_y - max(90.0, config.detection.crop_padding * 6.0))
            if layout.page_number == span.start_page
            else config.detection.crop_top_margin
        )
        top = max(start_page_floor, min(top, text_box.y0 - search_radius))
        bottom = min(layout.height - config.detection.crop_bottom_margin, max(bottom, text_box.y1 + search_radius))
        left = max(config.detection.crop_left_margin, text_box.x0 - search_radius * 0.6)
        right = min(layout.width - config.detection.crop_right_margin, text_box.x1 + search_radius * 0.6)
    else:
        left = config.detection.crop_left_margin
        right = layout.width - config.detection.crop_right_margin

    candidates: list[BoundingBox] = []
    for graphic in layout.graphics:
        furniture_label = _page_furniture_box_label(graphic, layout, config, answer_rule_bands)
        if furniture_label:
            continue
        if _is_tiny_context_furniture_graphic(graphic, layout):
            continue
        if _is_formula_rule_box(graphic, layout):
            continue
        center_y = (graphic.y0 + graphic.y1) / 2
        center_x = (graphic.x0 + graphic.x1) / 2
        in_vertical_search = top <= center_y <= bottom
        adjacent_margin = text_box is not None and graphic.y1 >= text_box.y0 - search_radius and graphic.y0 <= text_box.y1 + search_radius
        in_horizontal_search = left <= center_x <= right or adjacent_margin
        if in_vertical_search and in_horizontal_search:
            candidates.append(graphic)
    return _dominant_graphic_cluster(candidates)


def _is_tiny_context_furniture_graphic(graphic: BoundingBox, layout: PageLayout) -> bool:
    max_tiny_width = max(24.0, layout.width * 0.04)
    max_tiny_height = max(24.0, layout.height * 0.04)
    return _box_width(graphic) <= max_tiny_width and _box_height(graphic) <= max_tiny_height


def _span_has_figure_prompt(span: QuestionSpan) -> bool:
    text = _clean_text_line(span.combined_text).lower()
    if not text:
        return False
    if re.search(r"\b(?:figure|fig\.?|diagram|graph|sketch|draw|shown|sector|circle|histogram|scatter diagram|box plot|table|shaded)\b", text):
        return True
    compact_text = re.sub(r"\s+", "", text)
    if any(token in compact_text for token in ("thediagram", "diagramshows", "graphof", "graphthe", "graphshown")):
        return True
    return bool(re.search(r"\bcurve\b", text) and re.search(r"\b(?:shown|sketch|diagram|graph|shaded|area under|bounded)\b", text))


def _span_references_source_visual(span: QuestionSpan) -> bool:
    return references_source_visual(_clean_text_line(span.combined_text))


def _single_page_union_regions(
    regions: list[CropRegion],
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]] | tuple[None, list[str]]:
    if len(regions) < 2:
        return None, []
    page_numbers = {region.page_number for region in regions}
    if len(page_numbers) != 1:
        return None, ["single_page_union_skipped_multi_page"]
    if not any(region.graphics for region in regions):
        return None, ["single_page_union_skipped_no_graphics"]
    grouped_regions = _nearby_region_groups(regions, config)
    if len(grouped_regions) > 1 and _has_disjoint_text_only_tail(grouped_regions):
        return None, ["single_page_union_skipped_disjoint_tail"]

    page_number = next(iter(page_numbers))
    layout = _layout_by_number(layouts, page_number)
    if _has_top_page_edge_graphic(regions, layout, config):
        return None, ["single_page_union_skipped_page_edge_diagram"]
    text_blocks = [block for region in regions for block in region.text_blocks]
    labelled_graphics = [graphic for region in regions if region.text_blocks for graphic in region.graphics]
    graphics = _dominant_graphic_cluster(labelled_graphics or [graphic for region in regions for graphic in region.graphics])
    union_source_regions = [
        region
        for region in regions
        if not region.graphics or any(_boxes_intersect(graphic, dominant) for graphic in region.graphics for dominant in graphics)
    ]
    union_box = _union_boxes([region.bbox for region in union_source_regions] or [region.bbox for region in regions])
    content_boxes = [block.bbox for block in text_blocks] + graphics
    if not content_boxes:
        return None, ["single_page_union_skipped_no_content"]

    content_box = _union_boxes(content_boxes)
    padded = _trim_crop_furniture_edges(
        _clamp_crop_to_prompt_area(
            _union_boxes([union_box, content_box]).padded(config.detection.crop_padding, layout.width, layout.height),
            layout,
            config,
        ),
        layout,
        config,
    )
    padded, top_flags = _trim_crop_top_to_current_anchor(padded, content_box, span, layout)
    padded, boundary_flags = _trim_crop_at_next_question_anchor(padded, content_box, span, layout, config)
    trimmed_content_box = _content_box_within_crop(content_box, padded)
    if _box_height(padded) > layout.height * config.detection.max_crop_height_ratio:
        return None, ["single_page_union_skipped_too_tall"]
    if _contains_other_question_start(trimmed_content_box, span, layout, config):
        return None, ["single_page_union_skipped_neighbor_question"]

    content_area = sum(_box_area(box) for box in content_boxes)
    sparse_ratio = _box_area(padded) / max(1.0, content_area)
    if sparse_ratio > 7.5 and _box_height(padded) > layout.height * 0.42:
        return None, ["single_page_union_skipped_sparse"]
    padded, trailing_answer_rule_flags = _trim_union_trailing_answer_rule_padding(padded, content_box, layout, config)

    union_region = CropRegion(
        page_number=page_number,
        bbox=padded,
        text_blocks=sorted(text_blocks, key=lambda block: (block.bbox.y0, block.bbox.x0)),
        graphics=graphics,
        duplicate_graphics_removed=sum(region.duplicate_graphics_removed for region in regions),
        original_bbox=union_box,
        excluded_regions=_dedupe_excluded_regions([excluded for region in regions for excluded in region.excluded_regions]),
        region_kind="single_page_union",
        text_bbox=_union_boxes([block.bbox for block in text_blocks]) if text_blocks else None,
        figure_bbox=_union_boxes(graphics) if graphics else None,
    )
    trimmed, trim_flags = _trim_vertical_furniture_from_regions([union_region], layouts, config)
    trimmed, content_top_flags = _trim_content_top_padding_from_regions(trimmed, config)
    return trimmed, [
        "single_page_union_crop_used",
        f"single_page_union_fragments:{len(regions)}",
        *top_flags,
        *boundary_flags,
        *trailing_answer_rule_flags,
        *trim_flags,
        *content_top_flags,
    ]


def _same_page_diagram_union_regions(
    regions: list[CropRegion],
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    flags: list[str] = []
    output: list[CropRegion] = []
    by_page: dict[int, list[CropRegion]] = {}
    for region in regions:
        by_page.setdefault(region.page_number, []).append(region)

    for page_number in sorted(by_page):
        page_regions = by_page[page_number]
        layout = _layout_by_number(layouts, page_number)
        for group in _nearby_region_groups(page_regions, config):
            if len(group) < 2 or not any(region.graphics for region in group):
                output.extend(group)
                continue
            union_region, reason_flags = _union_regions_for_page(group, span, layout, config, "page_diagram_union")
            if union_region is None:
                flags.extend(reason_flags)
                output.extend(group)
                continue
            output.append(union_region)
            flags.extend([*reason_flags, f"page_diagram_union_fragments:{len(group)}"])

    deduped, dedupe_flags = _dedupe_crop_regions(output)
    flags.extend(dedupe_flags)
    trimmed, trim_flags = _trim_vertical_furniture_from_regions(deduped, layouts, config)
    flags.extend(trim_flags)
    trimmed, content_top_flags = _trim_content_top_padding_from_regions(trimmed, config)
    flags.extend(content_top_flags)
    return sorted(trimmed, key=lambda region: (region.page_number, region.bbox.y0, region.bbox.x0)), flags


def _nearby_region_groups(regions: list[CropRegion], config: AppConfig) -> list[list[CropRegion]]:
    sorted_regions = sorted(regions, key=lambda region: (region.bbox.y0, region.bbox.x0))
    if not sorted_regions:
        return []
    groups: list[list[CropRegion]] = [[sorted_regions[0]]]
    previous = sorted_regions[0]
    for region in sorted_regions[1:]:
        if region.bbox.y0 - previous.bbox.y1 > config.detection.prompt_region_max_gap:
            groups.append([region])
        else:
            groups[-1].append(region)
        previous = region
    return groups


def _union_regions_for_page(
    regions: list[CropRegion],
    span: QuestionSpan,
    layout: PageLayout,
    config: AppConfig,
    kind: str,
) -> tuple[CropRegion | None, list[str]]:
    if _has_top_page_edge_graphic(regions, layout, config):
        return None, [f"{kind}_skipped_page_edge_diagram"]

    graphics = _dominant_graphic_cluster([graphic for region in regions for graphic in region.graphics])
    text_blocks = _text_blocks_for_dominant_diagram_union(
        [block for region in regions for block in region.text_blocks],
        graphics,
        span,
        config,
    )
    union_source_regions = [
        region
        for region in regions
        if region.graphics or any(block in text_blocks for block in region.text_blocks)
    ]
    union_box = _union_boxes([region.bbox for region in union_source_regions] or [region.bbox for region in regions])
    content_boxes = [block.bbox for block in text_blocks] + graphics
    if not content_boxes:
        return None, [f"{kind}_skipped_no_content"]

    content_box = _union_boxes(content_boxes)
    padded = _trim_crop_furniture_edges(
        _clamp_crop_to_prompt_area(
            _union_boxes([union_box, content_box]).padded(config.detection.crop_padding, layout.width, layout.height),
            layout,
            config,
        ),
        layout,
        config,
    )
    padded, _top_flags = _trim_crop_top_to_current_anchor(padded, content_box, span, layout)
    padded, _boundary_flags = _trim_crop_at_next_question_anchor(padded, content_box, span, layout, config)
    trimmed_content_box = _content_box_within_crop(content_box, padded)
    if _box_height(padded) > layout.height * config.detection.max_crop_height_ratio:
        return None, [f"{kind}_skipped_too_tall"]
    if _contains_other_question_start(trimmed_content_box, span, layout, config):
        return None, [f"{kind}_skipped_neighbor_question"]

    content_area = sum(_box_area(box) for box in content_boxes)
    sparse_ratio = _box_area(padded) / max(1.0, content_area)
    if sparse_ratio > 7.5 and _box_height(padded) > layout.height * 0.42:
        return None, [f"{kind}_skipped_sparse"]

    excluded_regions = _dedupe_excluded_regions([excluded for region in regions for excluded in region.excluded_regions])
    padded = _trim_padding_for_page_edge_furniture(padded, content_box, excluded_regions, config)
    padded, trailing_answer_rule_flags = _trim_union_trailing_answer_rule_padding(padded, content_box, layout, config)

    return CropRegion(
        page_number=layout.page_number,
        bbox=padded,
        text_blocks=sorted(text_blocks, key=lambda block: (block.bbox.y0, block.bbox.x0)),
        graphics=graphics,
        duplicate_graphics_removed=sum(region.duplicate_graphics_removed for region in regions),
        original_bbox=union_box,
        excluded_regions=excluded_regions,
        region_kind=kind,
        text_bbox=_union_boxes([block.bbox for block in text_blocks]) if text_blocks else None,
        figure_bbox=_union_boxes(graphics) if graphics else None,
    ), [f"{kind}_used", *trailing_answer_rule_flags]


def _content_box_within_crop(content_box: BoundingBox, crop_box: BoundingBox) -> BoundingBox:
    return BoundingBox(
        content_box.x0,
        max(content_box.y0, crop_box.y0),
        content_box.x1,
        min(content_box.y1, crop_box.y1),
    )


def _trim_crop_top_to_current_anchor(
    crop_box: BoundingBox,
    content_box: BoundingBox,
    span: QuestionSpan,
    layout: PageLayout,
) -> tuple[BoundingBox, list[str]]:
    if layout.page_number != span.start_page:
        return crop_box, []
    safe_top = max(crop_box.y0, span.start_y - 24.0)
    if safe_top <= crop_box.y0 + 1.0:
        return crop_box, []
    if safe_top >= min(crop_box.y1 - 1.0, content_box.y1 - 1.0):
        return crop_box, []
    return BoundingBox(crop_box.x0, safe_top, crop_box.x1, crop_box.y1), ["crop_header_padding_trimmed"]


def _trim_crop_at_next_question_anchor(
    crop_box: BoundingBox,
    content_box: BoundingBox,
    span: QuestionSpan,
    layout: PageLayout,
    config: AppConfig,
) -> tuple[BoundingBox, list[str]]:
    boundary_y = _next_foreign_question_anchor_y(span, layout, config)
    if boundary_y is None or crop_box.y1 <= boundary_y:
        return crop_box, []

    safe_bottom = boundary_y - 1.0
    if safe_bottom <= crop_box.y0 + config.detection.min_crop_height:
        return crop_box, ["foreign_question_boundary_trim_skipped_protected_content", "crop_uncertain"]

    return BoundingBox(crop_box.x0, crop_box.y0, crop_box.x1, safe_bottom), ["foreign_question_boundary_trimmed"]


def _next_foreign_question_anchor_y(span: QuestionSpan, layout: PageLayout, config: AppConfig) -> float | None:
    anchors: list[float] = []
    for anchor in detect_question_anchor_candidates([layout], config):
        if anchor.bbox is None:
            continue
        if not _anchor_is_later_foreign_question(anchor.question_number, span.question_number):
            continue
        if _anchor_is_current_question_diagram_label(anchor, span, layout, config):
            continue
        if layout.page_number == span.start_page and anchor.y0 <= span.start_y + config.detection.anchor_y_tolerance:
            continue
        if anchor.confidence < max(0.52, config.detection.anchor_min_confidence - 0.08):
            continue
        anchors.append(anchor.y0)
    return min(anchors) if anchors else None


def _has_disjoint_text_only_tail(groups: list[list[CropRegion]]) -> bool:
    if len(groups) < 2:
        return False
    trailing_groups = groups[1:]
    return any(not any(region.graphics for region in group) for group in trailing_groups)


def _has_top_page_edge_graphic(regions: list[CropRegion], layout: PageLayout, config: AppConfig) -> bool:
    if not any(region.text_blocks and not region.graphics for region in regions):
        return False
    return any(
        layout.page_number == region.page_number and graphic.y0 <= config.detection.crop_top_margin
        for region in regions
        for graphic in region.graphics
    )


def _dominant_graphic_cluster(graphics: list[BoundingBox]) -> list[BoundingBox]:
    if len(graphics) <= 1:
        return graphics
    clusters: list[list[BoundingBox]] = []
    for graphic in sorted(graphics, key=lambda box: (_box_area(box), box.y0), reverse=True):
        match: list[BoundingBox] | None = None
        for cluster in clusters:
            if any(_graphics_same_cluster(graphic, other) for other in cluster):
                match = cluster
                break
        if match is None:
            clusters.append([graphic])
        else:
            match.append(graphic)
    return max(clusters, key=lambda cluster: (_box_area(_union_boxes(cluster)), len(cluster)))


def _graphics_same_cluster(a: BoundingBox, b: BoundingBox) -> bool:
    if _intersection_area(a, b) > 0:
        return True
    horizontal_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    vertical_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    common_size = max(12.0, min(max(_box_width(a), _box_height(a)), max(_box_width(b), _box_height(b))))
    return horizontal_gap <= common_size * 0.55 and vertical_gap <= common_size * 0.55


def _text_blocks_for_dominant_diagram_union(
    blocks: list[TextBlock],
    graphics: list[BoundingBox],
    span: QuestionSpan,
    config: AppConfig,
) -> list[TextBlock]:
    if not graphics:
        return blocks
    graphic_box = _union_boxes(graphics)
    kept: list[TextBlock] = []
    for block in blocks:
        if not _is_diagram_label_only_block(block, span, config):
            kept.append(block)
            continue
        if _block_belongs_to_figure(block, graphic_box, config) or _distance_between_boxes(block.bbox, graphic_box) <= 28:
            kept.append(block)
    return kept or blocks


def _is_diagram_label_only_block(block: TextBlock, span: QuestionSpan, config: AppConfig) -> bool:
    return _segment_is_figure_label_only([block], span, config)


def _distance_between_boxes(a: BoundingBox, b: BoundingBox) -> float:
    horizontal_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    vertical_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return (horizontal_gap**2 + vertical_gap**2) ** 0.5


def _contains_other_question_start(box: BoundingBox, span: QuestionSpan, layout: PageLayout, config: AppConfig) -> bool:
    for anchor in _foreign_question_anchors_for_span_page(span, layout, config):
        if anchor.bbox is not None and _boxes_intersect(box, anchor.bbox):
            return True

    for block in layout.blocks:
        if block.bbox.y0 < box.y0 or block.bbox.y0 > box.y1:
            continue
        if _block_is_current_question_diagram_label(block, span, layout, config):
            continue
        if re.fullmatch(r"[\d\s]+", _clean_text_line(block.first_line)):
            continue
        parsed = parse_question_start(block.first_line, config)
        if parsed and parsed[0] != span.question_number:
            return True
    return False


def _trim_regions_at_foreign_question_boundaries(
    regions: list[CropRegion],
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    if not regions:
        return regions, []

    boundary_by_page: dict[int, float] = {}
    for page_number in {region.page_number for region in regions}:
        layout = _layout_by_number(layouts, page_number)
        anchors = _foreign_question_anchors_for_span_page(span, layout, config)
        if anchors:
            boundary_by_page[page_number] = min(anchor.y0 for anchor in anchors)
    if not boundary_by_page:
        return regions, []

    trimmed: list[CropRegion] = []
    flags: list[str] = []
    for region in regions:
        boundary_y = boundary_by_page.get(region.page_number)
        if boundary_y is None or region.bbox.y1 < boundary_y:
            trimmed.append(region)
            continue

        safe_bottom = boundary_y - 1.0
        if region.bbox.y0 >= safe_bottom or safe_bottom <= region.bbox.y0 + config.detection.min_crop_height * 0.5:
            flags.append("foreign_question_region_removed")
            continue

        kept_text = [block for block in region.text_blocks if block.bbox.y1 <= safe_bottom]
        kept_graphics = [graphic for graphic in region.graphics if graphic.y1 <= safe_bottom]
        if not kept_text and not kept_graphics:
            flags.append("foreign_question_region_removed")
            continue

        flags.append("foreign_question_boundary_trimmed")
        trimmed.append(
            replace(
                region,
                bbox=BoundingBox(region.bbox.x0, region.bbox.y0, region.bbox.x1, safe_bottom),
                text_blocks=kept_text,
                graphics=kept_graphics,
                original_bbox=region.original_bbox or region.bbox,
                text_bbox=_union_boxes([block.bbox for block in kept_text]) if kept_text else None,
                figure_bbox=_union_boxes(kept_graphics) if kept_graphics else None,
            )
        )
    return trimmed, sorted(set(flags))


def _foreign_question_anchors_for_span_page(
    span: QuestionSpan,
    layout: PageLayout,
    config: AppConfig,
) -> list[QuestionStart]:
    anchors: list[QuestionStart] = []
    for anchor in detect_question_anchor_candidates([layout], config):
        if anchor.bbox is None:
            continue
        if not _anchor_is_later_foreign_question(anchor.question_number, span.question_number):
            continue
        if _anchor_is_current_question_diagram_label(anchor, span, layout, config):
            continue
        if layout.page_number == span.start_page and anchor.y0 <= span.start_y + config.detection.anchor_y_tolerance:
            continue
        if anchor.confidence < max(0.52, config.detection.anchor_min_confidence - 0.08):
            continue
        anchors.append(anchor)
    return anchors


def _anchor_is_current_question_diagram_label(
    anchor: QuestionStart,
    span: QuestionSpan,
    layout: PageLayout,
    config: AppConfig,
) -> bool:
    if not _span_has_figure_prompt(span) or anchor.bbox is None:
        return False
    if anchor.question_number != span.question_number:
        label = _clean_text_line(anchor.label)
        if label == anchor.question_number and anchor.x0 <= config.detection.question_start_max_x:
            return False
        if not _looks_like_diagram_axis_or_label_text(label):
            return False
    answer_rule_bands = _answer_rule_y_bands(layout)
    for graphic in layout.graphics:
        if _page_furniture_box_label(graphic, layout, config, answer_rule_bands):
            continue
        if _distance_between_boxes(anchor.bbox, graphic) <= max(8.0, config.detection.crop_padding * 2.5):
            return True
    return False


def _block_is_current_question_diagram_label(
    block: TextBlock,
    span: QuestionSpan,
    layout: PageLayout,
    config: AppConfig,
) -> bool:
    if not _span_has_figure_prompt(span):
        return False
    text = _clean_text_line(block.text)
    if not text or re.search(r"\[\d{1,2}\]", text):
        return False
    if not _is_diagram_label_only_block(block, span, config):
        tail = _current_question_axis_label(text, span, config)
        if tail is None:
            return False
    answer_rule_bands = _answer_rule_y_bands(layout)
    for graphic in layout.graphics:
        if _page_furniture_box_label(graphic, layout, config, answer_rule_bands):
            continue
        if _distance_between_boxes(block.bbox, graphic) <= max(16.0, config.detection.crop_padding * 3.2):
            return True
    return False


def _anchor_is_later_foreign_question(candidate_number: str, current_number: str) -> bool:
    try:
        return int(candidate_number) > int(current_number)
    except ValueError:
        return candidate_number != current_number


def _split_prompt_segments(blocks: list[TextBlock], layout: PageLayout, config: AppConfig) -> list[list[TextBlock]]:
    sorted_blocks = sorted(blocks, key=lambda item: (item.bbox.y0, item.bbox.x0))
    if not sorted_blocks:
        return []

    answer_rule_bands = _answer_rule_y_bands(layout)
    answer_rule_bands.extend(
        (block.bbox.y0 + block.bbox.y1) / 2
        for block in layout.blocks
        if _is_answer_space_text(_clean_text_line(block.text))
    )
    segments: list[list[TextBlock]] = [[sorted_blocks[0]]]
    previous = sorted_blocks[0]
    for block in sorted_blocks[1:]:
        gap = block.bbox.y0 - previous.bbox.y1
        if gap > config.detection.prompt_region_max_gap or _gap_contains_answer_rule_band(previous, block, answer_rule_bands):
            segments.append([block])
        else:
            segments[-1].append(block)
        previous = block
    return segments


def _gap_contains_answer_rule_band(previous: TextBlock, block: TextBlock, bands: list[float]) -> bool:
    if not bands:
        return False
    top = previous.bbox.y1 + 1.0
    bottom = block.bbox.y0 - 1.0
    if bottom <= top:
        return False
    return any(top <= band <= bottom for band in bands)


def _graphics_for_segment(
    text_box: BoundingBox,
    layout: PageLayout,
    config: AppConfig,
    *,
    span: QuestionSpan | None = None,
    segment: list[TextBlock] | None = None,
) -> tuple[list[BoundingBox], list[dict[str, object]]]:
    graphics: list[BoundingBox] = []
    excluded_regions: list[dict[str, object]] = []
    top = text_box.y0 - config.detection.prompt_graphic_overlap_padding
    bottom = text_box.y1 + config.detection.prompt_graphic_lookahead
    answer_rule_bands = _answer_rule_y_bands(layout)
    segment_text = _clean_text_line(" ".join(block.text for block in segment or []))
    expects_source_visual = bool(
        span is not None
        and (_span_has_figure_prompt(span) or (segment_text and references_source_visual(segment_text)))
    )
    for graphic in layout.graphics:
        furniture_label = _page_furniture_box_label(graphic, layout, config, answer_rule_bands)
        allow_furniture_graphic = (
            furniture_label == "watermark"
            and span is not None
            and _watermark_box_looks_like_current_question_diagram(graphic, span, segment or [], layout, config)
        )
        if furniture_label and not allow_furniture_graphic:
            excluded_regions.append(_excluded_region(furniture_label, graphic))
            continue
        if _is_formula_rule_box(graphic, layout):
            continue
        if span is not None and not expects_source_visual and _is_broad_shallow_non_visual_artifact(graphic, layout):
            continue
        if graphic.y1 < text_box.y0 and text_box.y0 - graphic.y1 > 6:
            continue
        overlaps_vertically = graphic.y1 >= top and graphic.y0 <= bottom
        overlaps_horizontally = graphic.x1 >= text_box.x0 - 30 and graphic.x0 <= text_box.x1 + 30
        graphic_width = graphic.x1 - graphic.x0
        graphic_height = graphic.y1 - graphic.y0
        significant_nearby_graphic = graphic_width >= 20 and graphic_height >= 20
        if overlaps_vertically and (overlaps_horizontally or significant_nearby_graphic):
            candidate = graphic
            if allow_furniture_graphic and span is not None:
                candidate = _trim_top_page_watermark_diagram_graphic(candidate, span, layout, config)
            if candidate is not None and span is not None:
                candidate = _trim_graphic_at_previous_question_content(candidate, span, layout, config)
            if candidate is not None and span is not None:
                candidate = _trim_graphic_at_next_question_boundary(candidate, span, layout, config)
            if candidate is not None and span is not None and segment:
                candidate = _trim_graphic_at_segment_prose_boundary(candidate, segment, span, config)
            if candidate is not None:
                graphics.append(candidate)
    return graphics, excluded_regions


def _is_broad_shallow_non_visual_artifact(graphic: BoundingBox, layout: PageLayout) -> bool:
    return _box_width(graphic) >= layout.width * 0.75 and _box_height(graphic) <= layout.height * 0.08


def _trim_top_page_watermark_diagram_graphic(
    graphic: BoundingBox,
    span: QuestionSpan,
    layout: PageLayout,
    config: AppConfig,
) -> BoundingBox | None:
    page_span_blocks = [block for block in span.blocks if block.page_number == layout.page_number]
    diagram_blocks = [
        block
        for block in page_span_blocks
        if _is_figure_label_or_current_anchor_block(block, span, config) and block.bbox.y0 <= graphic.y1 + config.detection.crop_padding
    ]
    if not diagram_blocks:
        return graphic

    diagram_label_box = _union_boxes([block.bbox for block in diagram_blocks])
    anchor_blocks = [
        block
        for block in page_span_blocks
        if (parsed := parse_question_start(block.first_line, config)) is not None and parsed[0] == span.question_number
    ]
    anchor_box = _union_boxes([block.bbox for block in anchor_blocks]) if anchor_blocks else None
    x_padding = max(55.0, min(70.0, config.detection.prompt_graphic_lookahead * 0.38))
    left = max(config.detection.crop_left_margin, diagram_label_box.x0 - x_padding)
    if anchor_box is not None:
        left = min(left, max(config.detection.crop_left_margin, anchor_box.x0 - config.detection.crop_padding))
    right = min(layout.width - config.detection.crop_right_margin, diagram_label_box.x1 + x_padding)

    prose_tops = [
        block.bbox.y0
        for block in page_span_blocks
        if block not in diagram_blocks
        and block not in anchor_blocks
        and block.bbox.y0 > diagram_label_box.y1
        and len(_clean_text_line(block.text)) > 18
    ]
    bottom = graphic.y1
    if prose_tops:
        safe_bottom = min(
            min(prose_tops) - config.detection.crop_padding - 1.0,
            diagram_label_box.y1 + config.detection.crop_padding,
        )
        if safe_bottom > diagram_label_box.y1 + 2.0:
            bottom = min(bottom, safe_bottom)

    top = graphic.y0
    if right <= left + 12 or bottom <= top + config.detection.min_crop_height:
        return None
    return BoundingBox(left, top, right, bottom)


def _watermark_box_looks_like_current_question_diagram(
    graphic: BoundingBox,
    span: QuestionSpan,
    segment: list[TextBlock],
    layout: PageLayout,
    config: AppConfig,
) -> bool:
    if not _span_has_figure_prompt(span):
        return False
    if layout.page_number != span.start_page:
        return False
    if span.start_y > config.detection.crop_top_margin + 55:
        return False
    if graphic.y0 > config.detection.crop_top_margin:
        return False
    if graphic.y1 > span.end_y + max(45.0, config.detection.crop_padding * 4):
        return False

    blocks = segment or [block for block in span.blocks if block.page_number == layout.page_number]
    evidence = [
        block
        for block in blocks
        if _block_center_inside_box(block, graphic) or _intersection_area(block.bbox, graphic) / max(1.0, _box_area(block.bbox)) >= 0.35
    ]
    broad_edge_watermark = _box_width(graphic) >= layout.width * 0.75 and _box_height(graphic) >= layout.height * 0.16
    if broad_edge_watermark and not any(_is_figure_label_or_current_anchor_block(block, span, config) for block in evidence):
        return False
    evidence_text = _clean_text_line(" ".join(block.text for block in evidence)).lower()
    if not re.search(r"\b(?:diagram|circle|arc|tangent|sector|shaded|graph)\b", evidence_text):
        return False
    return len(evidence) >= 2


def _trim_graphic_at_previous_question_content(
    graphic: BoundingBox,
    span: QuestionSpan | None,
    layout: PageLayout,
    config: AppConfig,
) -> BoundingBox | None:
    if span is None or layout.page_number != span.start_page:
        return graphic
    if graphic.y0 >= span.start_y - config.detection.crop_padding:
        return graphic

    span_block_keys = {_block_identity_key(block) for block in span.blocks}
    foreign_blocks_above = [
        block
        for block in layout.blocks
        if _block_identity_key(block) not in span_block_keys
        and block.bbox.y1 <= span.start_y - config.detection.anchor_y_tolerance
        and _boxes_intersect(block.bbox, graphic)
        and _graphic_trim_foreign_content_block(block, layout, config)
    ]
    if not foreign_blocks_above:
        return graphic

    safe_top = max(block.bbox.y1 for block in foreign_blocks_above) + max(2.0, config.detection.crop_padding * 3.5)
    if safe_top <= graphic.y0 + 1.0:
        return graphic
    if safe_top >= graphic.y1 - max(8.0, config.detection.min_crop_height * 0.5):
        return None
    return BoundingBox(graphic.x0, safe_top, graphic.x1, graphic.y1)


def _trim_graphic_at_next_question_boundary(
    graphic: BoundingBox,
    span: QuestionSpan,
    layout: PageLayout,
    config: AppConfig,
) -> BoundingBox | None:
    if layout.page_number != span.end_page:
        return graphic
    if graphic.y0 >= span.end_y - config.detection.anchor_y_tolerance:
        return None
    if graphic.y1 <= span.end_y + config.detection.crop_padding:
        return graphic

    safe_bottom = span.end_y - 1.0
    if safe_bottom <= graphic.y0 + max(8.0, config.detection.min_crop_height * 0.5):
        return None
    return BoundingBox(graphic.x0, graphic.y0, graphic.x1, safe_bottom)


def _trim_graphic_at_segment_prose_boundary(
    graphic: BoundingBox,
    segment: list[TextBlock],
    span: QuestionSpan,
    config: AppConfig,
) -> BoundingBox | None:
    graphic_height = _box_height(graphic)
    lower_zone_start = graphic.y0 + max(24.0, min(90.0, graphic_height * 0.32))
    blocks = _segment_with_page_span_blocks(segment, span)
    boundaries = [
        block.bbox.y0
        for block in blocks
        if lower_zone_start <= block.bbox.y0 < graphic.y1 - 1.0
        and _horizontal_overlap_ratio(block.bbox, graphic) >= 0.45
        and _is_question_prose_boundary_block(block, span, config)
    ]
    if not boundaries:
        return graphic

    safe_bottom = min(boundaries) - max(2.0, config.detection.crop_padding * 0.25)
    if safe_bottom <= graphic.y0 + max(12.0, config.detection.min_crop_height * 0.5):
        return graphic
    return BoundingBox(graphic.x0, graphic.y0, graphic.x1, safe_bottom)


def _segment_with_page_span_blocks(segment: list[TextBlock], span: QuestionSpan) -> list[TextBlock]:
    page_numbers = {block.page_number for block in segment}
    keys: set[tuple[int, str, float, float, float, float]] = set()
    blocks: list[TextBlock] = []
    for block in [*segment, *[item for item in span.blocks if item.page_number in page_numbers]]:
        key = _block_identity_key(block)
        if key in keys:
            continue
        keys.add(key)
        blocks.append(block)
    return blocks


def _is_question_prose_boundary_block(block: TextBlock, span: QuestionSpan, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if not text:
        return False
    if _is_answer_space_text(text) or _is_source_pagination_note_text(text):
        return False
    if _is_figure_label_or_current_anchor_block(block, span, config) or _is_diagram_label_only_block(block, span, config):
        return False
    parsed = parse_question_start(block.first_line, config)
    if parsed and parsed[0] == span.question_number:
        text = re.sub(rf"^\s*{re.escape(span.question_number)}\s*", "", text, count=1).strip()
    if len(text) < 24:
        return False
    if _looks_like_diagram_axis_or_label_text(text):
        return False
    words = [word for word in re.split(r"\s+", text) if word]
    if len(words) >= 6:
        return True
    return bool(re.search(r"\b(?:the|find|show|calculate|use|explain|given|hence|diagram|graph|curve)\b", text, re.IGNORECASE))


def _block_identity_key(block: TextBlock) -> tuple[int, str, float, float, float, float]:
    return (
        block.page_number,
        _clean_text_line(block.text),
        round(block.bbox.x0, 2),
        round(block.bbox.y0, 2),
        round(block.bbox.x1, 2),
        round(block.bbox.y1, 2),
    )


def _block_center_inside_box(block: TextBlock, box: BoundingBox) -> bool:
    center_x = (block.bbox.x0 + block.bbox.x1) / 2
    center_y = (block.bbox.y0 + block.bbox.y1) / 2
    return box.x0 <= center_x <= box.x1 and box.y0 <= center_y <= box.y1


def _graphic_trim_foreign_content_block(block: TextBlock, layout: PageLayout, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if not text:
        return False
    if _is_footer_or_header_box(block.bbox, layout, config):
        return False
    if _is_centered_page_number_block(block, layout, config):
        return False
    if _is_boilerplate_text(text):
        return False
    if _is_answer_space_text(text):
        return False
    if _is_margin_furniture_text(block, layout, config):
        return False
    if _is_control_artifact_text(text):
        return False
    return True


def _separate_text_and_figure_regions(
    page_number: int,
    segment: list[TextBlock],
    text_box: BoundingBox,
    graphics: list[BoundingBox],
    duplicate_count: int,
    excluded_regions: list[dict[str, object]],
    layout: PageLayout,
    config: AppConfig,
    span: QuestionSpan,
) -> tuple[list[CropRegion], list[str]]:
    flags = ["figure_region_separated"]
    figure_box = _figure_box_for_segment(segment, graphics, layout, config, span)
    figure_crop = _trim_crop_furniture_edges(_clamp_crop_to_prompt_area(figure_box, layout, config), layout, config)
    figure_label_blocks = [
        block
        for block in segment
        if (
            _is_figure_label_or_current_anchor_block(block, span, config)
            or _is_unit_diagram_label_block(block, span, config)
        )
        and _block_belongs_to_figure(block, figure_crop, config)
    ]
    if figure_label_blocks:
        label_box = _union_boxes([block.bbox for block in figure_label_blocks])
        if not _box_contains(figure_crop, label_box, tolerance=1.0):
            figure_crop = _trim_crop_furniture_edges(
                _clamp_crop_to_prompt_area(
                    _union_boxes([figure_crop, label_box]).padded(config.detection.crop_padding, layout.width, layout.height),
                    layout,
                    config,
                ),
                layout,
                config,
            )
            flags.append("figure_label_edge_safety_applied")
    trimmed_figure_crop = _trim_graphic_at_segment_prose_boundary(figure_crop, segment, span, config)
    if trimmed_figure_crop is not None and trimmed_figure_crop != figure_crop:
        figure_crop = trimmed_figure_crop
        flags.append("figure_crop_prose_boundary_trimmed")
    figure_label_ids = {id(block) for block in figure_label_blocks}
    text_blocks = [block for block in segment if id(block) not in figure_label_ids]
    text_segments = _split_prompt_segments(text_blocks, layout, config)

    regions: list[CropRegion] = []
    overlap_area = _intersection_area(text_box, figure_crop)
    if overlap_area > 1:
        flags.extend(["text_figure_overlap_trimmed", "question_text_figure_overlap_prevented"])

    for text_segment in text_segments:
        text_region_box = _union_boxes([block.bbox for block in text_segment])
        original_text_crop = text_region_box.padded(config.detection.crop_padding, layout.width, layout.height)
        crop_box, trimmed = _trim_box_to_exclude_figure(
            _trim_crop_furniture_edges(_clamp_crop_to_prompt_area(original_text_crop, layout, config), layout, config),
            figure_crop,
        )
        crop_box = _expand_text_crop_for_wide_prompt(crop_box, text_segment, layout, config, excluded_regions=excluded_regions)
        crop_box = _trim_text_top_padding_after_answer_rule(crop_box, text_region_box, layout, config)
        crop_box = _trim_text_bottom_padding_after_answer_rule(crop_box, text_region_box, layout, config)
        if trimmed:
            safe_crop_box = _ensure_crop_contains_text(crop_box, text_segment, original_text_crop, layout)
            if safe_crop_box != crop_box:
                crop_box = safe_crop_box
                trimmed = False
                flags.append("text_crop_edge_safety_applied")
            else:
                flags.extend(["text_figure_overlap_trimmed", "question_text_figure_overlap_prevented"])
            crop_box = _trim_text_top_padding_after_answer_rule(crop_box, text_region_box, layout, config)
            crop_box = _trim_text_bottom_padding_after_answer_rule(crop_box, text_region_box, layout, config)
        if _box_height(crop_box) < config.detection.min_crop_height or _box_width(crop_box) < 8:
            restored_crop = _trim_crop_furniture_edges(_clamp_crop_to_prompt_area(original_text_crop, layout, config), layout, config)
            if _box_height(restored_crop) >= config.detection.min_crop_height * 0.5 and _box_contains(
                restored_crop,
                text_region_box,
                tolerance=1.0,
            ):
                crop_box = restored_crop
                flags.append("text_crop_restored_to_preserve_content")
            else:
                flags.append("text_region_removed_after_figure_trim")
                continue
        regions.append(
            CropRegion(
                page_number=page_number,
                bbox=crop_box,
                text_blocks=text_segment,
                original_bbox=original_text_crop,
                excluded_regions=excluded_regions if not regions else [],
                region_kind="text",
                text_bbox=text_region_box,
                figure_bbox=figure_crop,
                text_figure_overlap_area=_intersection_area(original_text_crop, figure_crop),
                text_trimmed_for_figure=trimmed,
            )
        )

    regions.append(
        CropRegion(
            page_number=page_number,
            bbox=figure_crop,
            text_blocks=figure_label_blocks,
            graphics=graphics,
            duplicate_graphics_removed=duplicate_count,
            original_bbox=figure_box,
            excluded_regions=[] if regions else excluded_regions,
            region_kind="figure",
            figure_bbox=figure_crop,
            text_bbox=_union_boxes([block.bbox for block in figure_label_blocks]) if figure_label_blocks else None,
            text_figure_overlap_area=overlap_area,
        )
    )

    regions = sorted(regions, key=lambda region: (region.bbox.y0, region.bbox.x0, 0 if region.region_kind == "text" else 1))
    return regions, sorted(set(flags))


def _figure_box_for_segment(
    segment: list[TextBlock],
    graphics: list[BoundingBox],
    layout: PageLayout,
    config: AppConfig,
    span: QuestionSpan,
) -> BoundingBox:
    graphic_box = _union_boxes(_merge_graphics_into_figures(graphics))
    label_boxes = [
        block.bbox
        for block in segment
        if _is_figure_label_or_current_anchor_block(block, span, config) and _block_belongs_to_figure(block, graphic_box, config)
    ]
    return _union_boxes([graphic_box] + label_boxes).padded(config.detection.crop_padding, layout.width, layout.height)


def _expand_text_crop_for_wide_prompt(
    crop_box: BoundingBox,
    text_blocks: list[TextBlock],
    layout: PageLayout,
    config: AppConfig,
    *,
    excluded_regions: list[dict[str, object]] | None = None,
) -> BoundingBox:
    if not text_blocks:
        return crop_box
    if _has_right_side_panel_exclusion(excluded_regions or []):
        return crop_box
    right_limit = layout.width - max(2.0, config.detection.crop_right_margin * 0.08)
    if crop_box.x1 >= right_limit - 1.0:
        return crop_box

    usable_right = layout.width - config.detection.crop_right_margin
    near_right_edge = any(block.bbox.x1 >= usable_right - 18.0 for block in text_blocks)
    long_line = any(_box_width(block.bbox) >= layout.width * 0.70 and len(_clean_text_line(block.text)) >= 70 for block in text_blocks)
    if not near_right_edge and not long_line:
        return crop_box
    if crop_box.x1 < usable_right - 8.0:
        return crop_box

    return BoundingBox(crop_box.x0, crop_box.y0, min(layout.width, right_limit), crop_box.y1)


def _has_right_side_panel_exclusion(excluded_regions: list[dict[str, object]]) -> bool:
    for excluded in excluded_regions:
        if excluded.get("label") != "side_panel":
            continue
        box = _excluded_region_box(excluded)
        if box is not None and box.x0 > 0:
            return True
    return False


def _merge_graphics_into_figures(graphics: list[BoundingBox]) -> list[BoundingBox]:
    if not graphics:
        return []
    # Treat graphics found for a single prompt segment as one figure source.
    # Cambridge diagrams are often decomposed into many PDF drawing primitives;
    # keeping a single union avoids re-rendering graph fragments separately.
    return [_union_boxes(graphics)]


def _block_belongs_to_figure(block: TextBlock, figure_box: BoundingBox, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if len(text) > 36 and not _looks_like_diagram_axis_or_label_text(text):
        return False

    block_area = _box_area(block.bbox)
    if block_area <= 0:
        return False
    padding = max(2.0, config.detection.crop_padding * 0.5)
    padded_figure = BoundingBox(
        max(0.0, figure_box.x0 - padding),
        max(0.0, figure_box.y0 - padding),
        figure_box.x1 + padding,
        figure_box.y1 + padding,
    )
    overlap_ratio = _intersection_area(block.bbox, padded_figure) / block_area
    if overlap_ratio >= 0.35:
        return True
    center_x = (block.bbox.x0 + block.bbox.x1) / 2
    center_y = (block.bbox.y0 + block.bbox.y1) / 2
    return padded_figure.x0 <= center_x <= padded_figure.x1 and padded_figure.y0 <= center_y <= padded_figure.y1


def _is_figure_label_or_current_anchor_block(block: TextBlock, span: QuestionSpan, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    parsed = parse_question_start(block.first_line, config)
    if parsed and parsed[0] == span.question_number:
        tail = re.sub(rf"^\s*{re.escape(span.question_number)}\s*", "", text, count=1).strip()
        if not tail:
            return True
        return _looks_like_diagram_axis_or_label_text(tail) and len(tail) <= 24
    return _is_diagram_label_only_block(block, span, config)


def _segment_is_figure_label_only(segment: list[TextBlock], span: QuestionSpan, config: AppConfig) -> bool:
    text = _clean_text_line(" ".join(block.text for block in segment))
    if not text:
        return False
    parsed = parse_question_start(text, config)
    if parsed and parsed[0] == span.question_number:
        return False
    if re.search(r"\[\d{1,2}\]", text):
        return False
    if _looks_like_diagram_axis_or_label_text(text):
        return True
    tokens = [token for token in re.split(r"\s+", text) if token]
    if not tokens or len(tokens) > 8 or len(text) > 24:
        return False
    return all(re.fullmatch(r"[A-Za-z]|\d{1,2}|[()+\-−=]", token) for token in tokens)


def _trim_box_to_exclude_figure(box: BoundingBox, figure_box: BoundingBox) -> tuple[BoundingBox, bool]:
    if _intersection_area(box, figure_box) <= 1:
        return box, False

    candidates: list[BoundingBox] = []
    if box.y0 < figure_box.y0:
        candidates.append(BoundingBox(box.x0, box.y0, box.x1, min(box.y1, figure_box.y0 - 1)))
    if box.y1 > figure_box.y1:
        candidates.append(BoundingBox(box.x0, max(box.y0, figure_box.y1 + 1), box.x1, box.y1))
    if box.x0 < figure_box.x0:
        candidates.append(BoundingBox(box.x0, box.y0, min(box.x1, figure_box.x0 - 1), box.y1))
    if box.x1 > figure_box.x1:
        candidates.append(BoundingBox(max(box.x0, figure_box.x1 + 1), box.y0, box.x1, box.y1))
    candidates = [candidate for candidate in candidates if _box_width(candidate) >= 8 and _box_height(candidate) >= 4]
    if not candidates:
        return box, False
    return max(candidates, key=_box_area), True


def _ensure_crop_contains_text(
    crop_box: BoundingBox,
    text_blocks: list[TextBlock],
    original_box: BoundingBox,
    layout: PageLayout,
) -> BoundingBox:
    if not text_blocks:
        return crop_box
    text_box = _union_boxes([block.bbox for block in text_blocks])
    if _box_contains(crop_box, text_box, tolerance=1.0):
        return crop_box
    return BoundingBox(
        max(0.0, min(crop_box.x0, original_box.x0)),
        max(0.0, min(crop_box.y0, original_box.y0)),
        min(layout.width, max(crop_box.x1, original_box.x1)),
        min(layout.height, max(crop_box.y1, original_box.y1)),
    )


def _box_contains(outer: BoundingBox, inner: BoundingBox, tolerance: float = 0.0) -> bool:
    return (
        outer.x0 <= inner.x0 + tolerance
        and outer.y0 <= inner.y0 + tolerance
        and outer.x1 >= inner.x1 - tolerance
        and outer.y1 >= inner.y1 - tolerance
    )


def _remove_meaningful_region_overlaps(regions: list[CropRegion], config: AppConfig) -> tuple[list[CropRegion], list[str]]:
    flags: list[str] = []
    cleaned: list[CropRegion] = []
    for region in sorted(regions, key=lambda item: (item.page_number, item.bbox.y0, item.bbox.x0)):
        current = region
        for previous in [item for item in cleaned if item.page_number == region.page_number]:
            overlap = _intersection_area(current.bbox, previous.bbox)
            if overlap <= 1 or _horizontal_overlap_ratio(current.bbox, previous.bbox) < 0.08:
                continue
            if current.graphics:
                flags.append("figure_overlap_preserved")
                continue
            trimmed_box, trimmed = _trim_box_to_exclude_figure(current.bbox, previous.bbox)
            if trimmed and _box_area(trimmed_box) < _box_area(current.bbox):
                if current.text_blocks:
                    text_box = _union_boxes([block.bbox for block in current.text_blocks])
                    if not _box_contains(trimmed_box, text_box, tolerance=1.0):
                        flags.append("text_crop_edge_safety_applied")
                        continue
                flags.append("overlapping_crop_region_trimmed")
                current = CropRegion(
                    page_number=current.page_number,
                    bbox=trimmed_box,
                    text_blocks=current.text_blocks,
                    graphics=current.graphics,
                    duplicate_graphics_removed=current.duplicate_graphics_removed,
                    original_bbox=current.original_bbox,
                    excluded_regions=current.excluded_regions,
                    region_kind=current.region_kind,
                    text_bbox=current.text_bbox,
                    figure_bbox=current.figure_bbox,
                    text_figure_overlap_area=max(current.text_figure_overlap_area, overlap),
                    text_trimmed_for_figure=True,
                    footer_cutoff=current.footer_cutoff,
                )
            elif overlap > max(12.0, _box_area(current.bbox) * 0.05):
                flags.append("text_figure_overlap_unresolved")
        if _box_width(current.bbox) >= 8 and _box_height(current.bbox) >= config.detection.min_crop_height * 0.5:
            cleaned.append(current)
    return cleaned, sorted(set(flags))


def _remove_duplicate_figure_labels_from_text_regions(
    regions: list[CropRegion],
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    figure_regions_by_page: dict[int, list[CropRegion]] = {}
    for region in regions:
        if region.graphics:
            figure_regions_by_page.setdefault(region.page_number, []).append(region)
    if not figure_regions_by_page:
        return regions, []

    output: list[CropRegion] = []
    flags: list[str] = []
    for region in regions:
        figure_regions = figure_regions_by_page.get(region.page_number, [])
        if region.graphics or not region.text_blocks or not figure_regions:
            output.append(region)
            continue

        kept_blocks: list[TextBlock] = []
        removed_blocks: list[TextBlock] = []
        for block in region.text_blocks:
            if _text_block_is_duplicate_figure_label(block, figure_regions, span, config):
                removed_blocks.append(block)
            else:
                kept_blocks.append(block)

        if not removed_blocks:
            output.append(region)
            continue

        flags.append("duplicate_figure_label_block_excluded")
        if not kept_blocks:
            flags.append("duplicate_figure_label_region_removed")
            continue

        layout = _layout_by_number(layouts, region.page_number)
        text_bbox = _union_boxes([block.bbox for block in kept_blocks])
        original_box = text_bbox.padded(config.detection.crop_padding, layout.width, layout.height)
        crop_box = _trim_crop_furniture_edges(_clamp_crop_to_prompt_area(original_box, layout, config), layout, config)
        for figure_region in figure_regions:
            figure_box = figure_region.figure_bbox or _union_boxes(figure_region.graphics) if figure_region.graphics else figure_region.bbox
            trimmed_box, trimmed = _trim_box_to_exclude_figure(crop_box, figure_box)
            if trimmed and _box_contains(trimmed_box, text_bbox, tolerance=1.0):
                crop_box = trimmed_box
        crop_box = _expand_text_crop_for_wide_prompt(crop_box, kept_blocks, layout, config, excluded_regions=region.excluded_regions)
        crop_box = _trim_text_only_top_padding(crop_box, text_bbox, layout, config)
        crop_box = _trim_text_top_padding_after_answer_rule(crop_box, text_bbox, layout, config)
        crop_box = _trim_text_bottom_padding_after_answer_rule(crop_box, text_bbox, layout, config)
        if _box_height(crop_box) < config.detection.min_crop_height * 0.5 or _box_width(crop_box) < 8:
            flags.append("duplicate_figure_label_region_removed")
            continue

        output.append(
            replace(
                region,
                bbox=crop_box,
                text_blocks=kept_blocks,
                original_bbox=region.original_bbox or region.bbox,
                text_bbox=text_bbox,
            )
        )

    return sorted(output, key=lambda item: (item.page_number, item.bbox.y0, item.bbox.x0)), sorted(set(flags))


def _text_block_is_duplicate_figure_label(
    block: TextBlock,
    figure_regions: list[CropRegion],
    span: QuestionSpan,
    config: AppConfig,
) -> bool:
    text = _clean_text_line(block.text)
    if not text or re.search(r"\[\d{1,2}\]", text):
        return False
    label_text = _current_question_axis_label(text, span, config) or text
    if not (_is_diagram_label_only_block(block, span, config) or _looks_like_diagram_axis_or_label_text(label_text)):
        return False
    for figure_region in figure_regions:
        figure_box = figure_region.figure_bbox or (_union_boxes(figure_region.graphics) if figure_region.graphics else figure_region.bbox)
        if _block_belongs_to_figure(block, figure_box, config):
            return True
        if _distance_between_boxes(block.bbox, figure_box) <= max(32.0, config.detection.crop_padding * 4.0):
            return True
    return False


def _separate_text_only_diagram_label_regions(
    regions: list[CropRegion],
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    if not regions or not _span_has_figure_prompt(span):
        return regions, []

    pages_with_detected_graphics = {region.page_number for region in regions if region.graphics}
    label_blocks_by_page: dict[int, list[TextBlock]] = {}
    for region in regions:
        if region.graphics:
            continue
        layout = _layout_by_number(layouts, region.page_number)
        if region.page_number in pages_with_detected_graphics:
            continue
        for block in region.text_blocks:
            if _is_text_only_diagram_label_block(block, span, config):
                label_blocks_by_page.setdefault(region.page_number, []).append(block)

    accepted: set[tuple[int, str, float, float, float, float]] = set()
    diagram_regions: list[CropRegion] = []
    flags: list[str] = []
    max_gap = max(config.detection.prompt_graphic_lookahead * 1.7, config.detection.prompt_region_max_gap * 4.0)
    for page_number, blocks in label_blocks_by_page.items():
        layout = _layout_by_number(layouts, page_number)
        for cluster in _text_only_diagram_label_clusters(blocks, max_gap=max_gap):
            if not _text_only_diagram_label_cluster_is_strong(cluster):
                continue
            cluster_keys = {_block_identity_key(block) for block in cluster}
            accepted.update(cluster_keys)
            label_box = _union_boxes([block.bbox for block in cluster])
            graphics = _graphics_for_text_only_diagram_cluster(label_box, layout, config)
            figure_box = _union_boxes([label_box] + graphics)
            crop_box = _trim_crop_furniture_edges(
                _clamp_crop_to_prompt_area(figure_box.padded(config.detection.crop_padding, layout.width, layout.height), layout, config),
                layout,
                config,
            )
            diagram_regions.append(
                CropRegion(
                    page_number=page_number,
                    bbox=crop_box,
                    text_blocks=sorted(cluster, key=lambda block: (block.bbox.y0, block.bbox.x0)),
                    graphics=graphics,
                    region_kind="text_diagram_union",
                    text_bbox=label_box,
                    figure_bbox=figure_box,
                )
            )
            flags.append("text_only_diagram_union_used")

    if not accepted:
        return regions, []

    output: list[CropRegion] = []
    for region in regions:
        kept_blocks = [block for block in region.text_blocks if _block_identity_key(block) not in accepted]
        if len(kept_blocks) == len(region.text_blocks):
            output.append(region)
            continue
        if not kept_blocks and not region.graphics:
            flags.append("diagram_label_text_region_removed")
            continue
        layout = _layout_by_number(layouts, region.page_number)
        text_bbox = _union_boxes([block.bbox for block in kept_blocks]) if kept_blocks else None
        if text_bbox is None:
            output.append(replace(region, text_blocks=kept_blocks, text_bbox=None, original_bbox=region.original_bbox or region.bbox))
            continue
        crop_box = _trim_crop_furniture_edges(
            _clamp_crop_to_prompt_area(text_bbox.padded(config.detection.crop_padding, layout.width, layout.height), layout, config),
            layout,
            config,
        )
        output.append(
            replace(
                region,
                bbox=crop_box,
                text_blocks=kept_blocks,
                text_bbox=text_bbox,
                original_bbox=region.original_bbox or region.bbox,
            )
        )

    output.extend(diagram_regions)
    deduped, dedupe_flags = _dedupe_crop_regions(output)
    flags.extend(dedupe_flags)
    return sorted(deduped, key=lambda region: (region.page_number, region.bbox.y0, region.bbox.x0)), sorted(set(flags))


def _text_only_diagram_label_clusters(blocks: list[TextBlock], *, max_gap: float) -> list[list[TextBlock]]:
    sorted_blocks = sorted(blocks, key=lambda block: (block.bbox.y0, block.bbox.x0))
    if not sorted_blocks:
        return []
    clusters: list[list[TextBlock]] = [[sorted_blocks[0]]]
    previous = sorted_blocks[0]
    for block in sorted_blocks[1:]:
        if block.bbox.y0 - previous.bbox.y1 > max_gap:
            clusters.append([block])
        else:
            clusters[-1].append(block)
        previous = block
    return clusters


def _graphics_for_text_only_diagram_cluster(
    label_box: BoundingBox,
    layout: PageLayout,
    config: AppConfig,
) -> list[BoundingBox]:
    if not layout.graphics:
        return []
    answer_rule_bands = _answer_rule_y_bands(layout)
    search_box = label_box.padded(max(18.0, config.detection.crop_padding * 4.0), layout.width, layout.height)
    graphics: list[BoundingBox] = []
    for graphic in layout.graphics:
        furniture_label = _page_furniture_box_label(graphic, layout, config, answer_rule_bands)
        if furniture_label and not (furniture_label == "barcode" and _boxes_intersect(search_box, graphic)):
            continue
        if _is_formula_rule_box(graphic, layout):
            continue
        if _boxes_intersect(search_box, graphic) or _distance_between_boxes(label_box, graphic) <= max(32.0, config.detection.crop_padding * 5.0):
            graphics.append(graphic)
    return _dominant_graphic_cluster(graphics)


def _text_only_diagram_label_cluster_is_strong(blocks: list[TextBlock]) -> bool:
    if len(blocks) < 2:
        return False
    cleaned = [_clean_text_line(block.text) for block in blocks]
    has_axis_label = any("(" in text and ")" in text for text in cleaned)
    has_numeric_or_origin_label = any(re.search(r"(?:^|\s)(?:O|-?\d+(?:\.\d+)?)(?:\s|$)", text) for text in cleaned)
    return has_axis_label or has_numeric_or_origin_label


def _is_text_only_diagram_label_block(block: TextBlock, span: QuestionSpan, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if not text:
        return False
    if re.search(r"\[\d{1,2}\]", text):
        return False
    if parse_question_start(text, config) and not _current_question_axis_label(text, span, config):
        if text == span.question_number and block.bbox.x0 > config.detection.question_start_max_x + 20:
            return True
        return False
    if _current_question_axis_label(text, span, config):
        text = _current_question_axis_label(text, span, config) or text
    sentence_like = bool(
        re.search(
            r"\b(?:the|find|show|calculate|solve|given|diagram|graph|sketch|draw|hence|for|from|with)\b",
            text,
            re.IGNORECASE,
        )
    )
    if sentence_like and not _looks_like_diagram_axis_or_label_text(text):
        return False
    if _looks_like_diagram_axis_or_label_text(text):
        return True
    return len(text) <= 28 and len(text.split()) <= 4 and not sentence_like and bool(re.search(r"[A-Z0-9()°]", text))


def _current_question_axis_label(text: str, span: QuestionSpan, config: AppConfig) -> str | None:
    match = re.match(rf"^\s*{re.escape(span.question_number)}\s+(.+?)\s*$", text)
    if not match:
        return None
    tail = match.group(1)
    return tail if _looks_like_diagram_axis_or_label_text(tail) else None


def _dedupe_crop_regions(regions: list[CropRegion]) -> tuple[list[CropRegion], list[str]]:
    kept: list[CropRegion] = []
    flags: list[str] = []
    for region in sorted(regions, key=lambda item: (item.page_number, _box_area(item.bbox), item.bbox.y0, item.bbox.x0), reverse=True):
        if any(region.page_number == other.page_number and _region_is_stale_fragment(region, other) for other in kept):
            flags.append("stale_crop_fragment_removed")
            continue
        if any(region.page_number == other.page_number and _boxes_duplicate(region.bbox, other.bbox) for other in kept):
            flags.append("duplicate_crop_region_removed")
            continue
        kept.append(region)
    return sorted(kept, key=lambda item: (item.page_number, item.bbox.y0, item.bbox.x0)), sorted(set(flags))


def _dedupe_excluded_regions(excluded_regions: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[object, ...]] = set()
    deduped: list[dict[str, object]] = []
    for item in excluded_regions:
        bbox = item.get("bbox") or {}
        key = (
            item.get("label"),
            bbox.get("x0"),
            bbox.get("y0"),
            bbox.get("x1"),
            bbox.get("y1"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _region_is_stale_fragment(candidate: CropRegion, current: CropRegion) -> bool:
    if _intersection_area(candidate.bbox, current.bbox) <= 1:
        return False
    if _box_area(candidate.bbox) >= _box_area(current.bbox):
        return False
    if _is_lower_overlapping_figure_fragment(candidate, current):
        return True
    overlap_ratio = _intersection_area(candidate.bbox, current.bbox) / max(1.0, _box_area(candidate.bbox))
    if overlap_ratio < 0.9:
        return False
    if candidate.region_kind == current.region_kind == "text":
        candidate_text = {_clean_text_line(block.text) for block in candidate.text_blocks if _clean_text_line(block.text)}
        current_text = {_clean_text_line(block.text) for block in current.text_blocks if _clean_text_line(block.text)}
        return bool(candidate_text) and candidate_text <= current_text
    return True


def _is_lower_overlapping_figure_fragment(candidate: CropRegion, current: CropRegion) -> bool:
    if candidate.region_kind != "figure" or current.region_kind != "figure":
        return False
    if not candidate.graphics or not current.graphics:
        return False
    if candidate.bbox.y0 <= current.bbox.y0 + 24.0:
        return False
    candidate_height = _box_height(candidate.bbox)
    if candidate_height <= 0:
        return False
    vertical_overlap = max(0.0, min(candidate.bbox.y1, current.bbox.y1) - max(candidate.bbox.y0, current.bbox.y0))
    if vertical_overlap / candidate_height < 0.65:
        return False
    if abs(candidate.bbox.y1 - current.bbox.y1) > 8.0:
        return False
    if _horizontal_overlap_ratio(candidate.bbox, current.bbox) < 0.55:
        return False
    label_texts = [_clean_text_line(block.text) for block in candidate.text_blocks if _clean_text_line(block.text)]
    if not label_texts:
        return True
    return all(len(text) <= 24 and _looks_like_diagram_axis_or_label_text(text) for text in label_texts)


def _is_prompt_text_block(block: TextBlock, span: QuestionSpan, layout: PageLayout, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if not text:
        return False
    if _is_source_pagination_note_text(text):
        return False
    if _is_footer_or_header_box(block.bbox, layout, config):
        return False
    if _is_centered_page_number_block(block, layout, config):
        return False
    if _is_boilerplate_text(text):
        return False
    if _is_answer_space_text(text):
        return False
    if _is_margin_furniture_text(block, layout, config):
        return False
    if _is_control_artifact_text(text):
        return False

    parsed = parse_question_start(text, config)
    if parsed and parsed[0] != span.question_number and not _is_unit_diagram_label_block(block, span, config):
        if _is_numeric_leading_continuation_text(text, parsed[1]):
            return True
        if block.bbox.x0 > config.detection.question_start_max_x:
            return True
        return False

    # Lone page numbers and administrative codes should not set crop bounds.
    if text.isdigit() and (block.bbox.y0 < config.detection.crop_top_margin or block.bbox.y1 > layout.height - config.detection.bottom_margin):
        return False
    return True


def _is_numeric_leading_continuation_text(text: str, label: str) -> bool:
    if not re.fullmatch(r"\d{1,2}", label):
        return False
    tail = re.sub(rf"^\s*{re.escape(label)}\s*", "", text, count=1).strip()
    return bool(tail and tail[0].islower())


def _is_unit_diagram_label_block(block: TextBlock, span: QuestionSpan, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if not re.search(r"\b(?:cm|mm|m|km|kg|g|s|ms|rad|N)\b", text):
        return False
    if _is_diagram_label_only_block(block, span, config):
        return True
    tokens = [token for token in re.split(r"\s+", text) if token]
    return len(tokens) <= 3 and len(text) <= 24 and not re.search(r"\[\d{1,2}\]", text)


def _trim_vertical_furniture_from_regions(
    regions: list[CropRegion],
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    trimmed: list[CropRegion] = []
    flags: list[str] = []
    for region in regions:
        layout = _layout_by_number(layouts, region.page_number)
        updated, region_flags = _trim_vertical_furniture_from_region(region, layout, config)
        trimmed.append(updated)
        flags.extend(region_flags)
    return trimmed, sorted(set(flags))


def _trim_vertical_furniture_from_region(
    region: CropRegion,
    layout: PageLayout,
    config: AppConfig,
) -> tuple[CropRegion, list[str]]:
    flags: list[str] = []
    top_candidates: list[tuple[BoundingBox, str]] = []
    bottom_candidates: list[tuple[BoundingBox, str]] = []
    edge_band = max(48.0, config.detection.crop_padding * 4)

    for block in layout.blocks:
        label = _vertical_text_furniture_label(block, layout, config)
        if label is None or not _boxes_intersect(region.bbox, block.bbox):
            continue
        if block.bbox.y0 <= region.bbox.y0 + edge_band:
            top_candidates.append((block.bbox, label))
        if block.bbox.y1 >= region.bbox.y1 - edge_band:
            bottom_candidates.append((block.bbox, label))

    answer_rule_bands = _answer_rule_y_bands(layout)
    for graphic in layout.graphics:
        label = _vertical_graphic_furniture_label(graphic, layout, config, answer_rule_bands)
        if label is None or not _boxes_intersect(region.bbox, graphic):
            continue
        if graphic.y0 <= region.bbox.y0 + edge_band:
            top_candidates.append((graphic, label))
        if graphic.y1 >= region.bbox.y1 - edge_band:
            bottom_candidates.append((graphic, label))

    top = region.bbox.y0
    bottom = region.bbox.y1
    protected = _protected_region_boxes(region)

    if top_candidates:
        candidate_top, labels = _safe_top_furniture_trim(top_candidates, protected, top=top, bottom=bottom, config=config)
        if candidate_top is not None:
            top = candidate_top
            flags.extend(_trim_flags_for_labels(labels))
        elif max(box.y1 for box, _label in top_candidates) + 1.0 > region.bbox.y0 + 1.0:
            flags.extend(["crop_furniture_trim_skipped_protected_content", "crop_uncertain"])

    if bottom_candidates:
        candidate_bottom, labels = _safe_bottom_furniture_trim(bottom_candidates, protected, top=top, bottom=bottom, config=config)
        if candidate_bottom is not None:
            bottom = candidate_bottom
            flags.extend(_trim_flags_for_labels(labels))
        elif min(box.y0 for box, _label in bottom_candidates) - 1.0 < region.bbox.y1 - 1.0:
            flags.extend(["crop_furniture_trim_skipped_protected_content", "crop_uncertain"])

    if top == region.bbox.y0 and bottom == region.bbox.y1:
        return region, sorted(set(flags))

    return replace(
        region,
        bbox=BoundingBox(region.bbox.x0, top, region.bbox.x1, bottom),
        original_bbox=region.original_bbox or region.bbox,
    ), sorted(set(flags))


def _trim_permission_footer_from_regions(
    regions: list[CropRegion],
    layouts: list[PageLayout],
    config: AppConfig,
) -> tuple[list[CropRegion], list[str]]:
    trimmed: list[CropRegion] = []
    flags: list[str] = []
    for region in regions:
        layout = _layout_by_number(layouts, region.page_number)
        updated, region_flags = _trim_permission_footer_from_region(region, layout, config)
        if updated is not None:
            trimmed.append(updated)
        flags.extend(region_flags)
    return trimmed, sorted(set(flags))


def _trim_permission_footer_from_region(
    region: CropRegion,
    layout: PageLayout,
    config: AppConfig,
) -> tuple[CropRegion | None, list[str]]:
    original_bottom = region.bbox.y1
    decision = _detect_permission_footer_cutoff(region, layout, config)
    if decision is None:
        return replace(
            region,
            footer_cutoff={
                "original_bottom": round(original_bottom, 2),
                "detected_footer_cutoff_y": None,
                "reason": "not_detected",
                "signals": [],
                "final_bottom": round(region.bbox.y1, 2),
            },
        ), []

    cutoff_y, signals, reason = decision
    padding = max(8.0, min(20.0, config.detection.crop_padding))
    final_bottom = max(region.bbox.y0, cutoff_y - padding)
    min_preserved_height = max(config.detection.min_crop_height * 2.0, 60.0)
    if final_bottom <= region.bbox.y0 + min_preserved_height:
        if _region_is_permission_footer_only(region, layout):
            return None, ["permission_footer_region_removed", "permission_footer_trimmed"]
        return replace(
            region,
            footer_cutoff={
                "original_bottom": round(original_bottom, 2),
                "detected_footer_cutoff_y": round(cutoff_y, 2),
                "reason": "skipped_min_preserved_height",
                "signals": signals,
                "final_bottom": round(region.bbox.y1, 2),
            },
        ), ["permission_footer_trim_skipped_min_height", "crop_uncertain"]

    if not _footer_trim_preserves_region_content(region, final_bottom):
        return replace(
            region,
            footer_cutoff={
                "original_bottom": round(original_bottom, 2),
                "detected_footer_cutoff_y": round(cutoff_y, 2),
                "reason": "skipped_protected_content",
                "signals": signals,
                "final_bottom": round(region.bbox.y1, 2),
            },
        ), ["permission_footer_trim_skipped_protected_content", "crop_uncertain"]

    if final_bottom >= region.bbox.y1 - 1.0:
        return replace(
            region,
            footer_cutoff={
                "original_bottom": round(original_bottom, 2),
                "detected_footer_cutoff_y": round(cutoff_y, 2),
                "reason": "detected_below_crop",
                "signals": signals,
                "final_bottom": round(region.bbox.y1, 2),
            },
        ), []

    return replace(
        region,
        bbox=BoundingBox(region.bbox.x0, region.bbox.y0, region.bbox.x1, final_bottom),
        original_bbox=region.original_bbox or region.bbox,
        footer_cutoff={
            "original_bottom": round(original_bottom, 2),
            "detected_footer_cutoff_y": round(cutoff_y, 2),
            "reason": reason,
            "signals": signals,
            "final_bottom": round(final_bottom, 2),
        },
    ), ["permission_footer_trimmed"]


def _detect_permission_footer_cutoff(
    region: CropRegion,
    layout: PageLayout,
    config: AppConfig,
) -> tuple[float, list[str], str] | None:
    if region.bbox.y1 < layout.height * 0.72:
        return None

    lower_y = layout.height * 0.70
    phrase_blocks = [
        block
        for block in layout.blocks
        if block.bbox.y0 >= lower_y
        and _boxes_intersect(region.bbox, block.bbox)
        and _is_permission_footer_phrase(_clean_text_line(block.text))
    ]
    rule = _footer_horizontal_rule(region, layout)
    dense_blocks = _footer_like_dense_small_text_blocks(region, layout, lower_y)

    if rule is not None:
        blocks_below_rule = [block for block in phrase_blocks + dense_blocks if block.bbox.y0 >= rule.y0 - 4.0]
        if phrase_blocks and blocks_below_rule:
            return rule.y0, ["horizontal_rule", "footer_phrase"], "horizontal_rule_with_footer_phrase"
        if dense_blocks and blocks_below_rule and _dense_footer_text_is_strong(dense_blocks):
            return rule.y0, ["horizontal_rule", "dense_small_footer_text"], "horizontal_rule_with_dense_footer_text"

    if phrase_blocks:
        first_phrase = min(block.bbox.y0 for block in phrase_blocks)
        return first_phrase, ["footer_phrase"], "footer_phrase"

    return None


def _footer_horizontal_rule(region: CropRegion, layout: PageLayout) -> BoundingBox | None:
    candidates = []
    for graphic in layout.graphics:
        if not _boxes_intersect(region.bbox, graphic):
            continue
        width = _box_width(graphic)
        height = _box_height(graphic)
        if graphic.y0 < layout.height * 0.74:
            continue
        if height <= 3.5 and width >= layout.width * 0.45:
            candidates.append(graphic)
    if not candidates:
        return None
    return min(candidates, key=lambda box: box.y0)


def _footer_like_dense_small_text_blocks(region: CropRegion, layout: PageLayout, lower_y: float) -> list[TextBlock]:
    blocks = []
    for block in layout.blocks:
        if block.bbox.y0 < lower_y or not _boxes_intersect(region.bbox, block.bbox):
            continue
        text = _clean_text_line(block.text)
        if not text or _is_permission_footer_phrase(text):
            blocks.append(block)
            continue
        height = _box_height(block.bbox)
        small_font = block.font_size is not None and block.font_size <= 7.5
        short_line_height = height <= 10.0
        if (small_font or short_line_height) and len(text) >= 8:
            blocks.append(block)
    return blocks


def _dense_footer_text_is_strong(blocks: list[TextBlock]) -> bool:
    cleaned = [_clean_text_line(block.text) for block in blocks if _clean_text_line(block.text)]
    if not cleaned:
        return False
    return len(cleaned) >= 2 and sum(len(text) for text in cleaned) >= 45


def _is_permission_footer_phrase(text: str) -> bool:
    patterns = [
        r"\bPermission to reproduce\b",
        r"\bUCLES\b",
        r"\bCambridge Assessment\b",
        r"\bcopyright\b",
        r"\bLocal Examinations Syndicate\b",
        r"\bUniversity of Cambridge International Examinations\b",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _footer_trim_preserves_region_content(region: CropRegion, final_bottom: float) -> bool:
    protected_boxes = [block.bbox for block in region.text_blocks if not _is_permission_footer_phrase(_clean_text_line(block.text))]
    protected_boxes.extend(graphic for graphic in region.graphics if graphic.y0 < final_bottom + 1.0)
    return all(box.y1 <= final_bottom + 1.0 for box in protected_boxes)


def _region_is_permission_footer_only(region: CropRegion, layout: PageLayout) -> bool:
    if region.graphics:
        return False
    if not region.text_blocks:
        return False
    lower_y = layout.height * 0.70
    for block in region.text_blocks:
        text = _clean_text_line(block.text)
        if block.bbox.y0 < lower_y:
            return False
        if _is_permission_footer_phrase(text):
            continue
        if block.font_size is not None and block.font_size <= 7.5 and len(text) >= 8:
            continue
        if _box_height(block.bbox) <= 10.0 and len(text) >= 8:
            continue
        return False
    return True


def _safe_top_furniture_trim(
    candidates: list[tuple[BoundingBox, str]],
    protected: list[BoundingBox],
    *,
    top: float,
    bottom: float,
    config: AppConfig,
) -> tuple[float | None, list[str]]:
    candidate_tops = sorted({box.y1 + 1.0 for box, _label in candidates if box.y1 + 1.0 > top + 1.0}, reverse=True)
    for candidate_top in candidate_tops:
        if candidate_top >= bottom - config.detection.min_crop_height:
            continue
        if not _trim_preserves_protected_boxes(protected, top=candidate_top, bottom=bottom):
            continue
        labels = [label for box, label in candidates if box.y1 + 1.0 <= candidate_top + 0.5]
        return candidate_top, labels
    return None, []


def _safe_bottom_furniture_trim(
    candidates: list[tuple[BoundingBox, str]],
    protected: list[BoundingBox],
    *,
    top: float,
    bottom: float,
    config: AppConfig,
) -> tuple[float | None, list[str]]:
    candidate_bottoms = sorted({box.y0 - 1.0 for box, _label in candidates if box.y0 - 1.0 < bottom - 1.0})
    for candidate_bottom in candidate_bottoms:
        if candidate_bottom <= top + config.detection.min_crop_height:
            continue
        if not _trim_preserves_protected_boxes(protected, top=top, bottom=candidate_bottom):
            continue
        labels = [label for box, label in candidates if box.y0 - 1.0 >= candidate_bottom - 0.5]
        return candidate_bottom, labels
    return None, []


def _vertical_text_furniture_label(block: TextBlock, layout: PageLayout, config: AppConfig) -> str | None:
    text = _clean_text_line(block.text)
    if not text:
        return None
    if _is_centered_page_number_block(block, layout, config):
        return "centered_page_number"
    if _is_footer_or_header_box(block.bbox, layout, config) or _is_boilerplate_text(text):
        return "header_footer"
    if _is_control_artifact_text(text):
        return "control_artifact"
    return None


def _vertical_graphic_furniture_label(
    box: BoundingBox,
    layout: PageLayout,
    config: AppConfig,
    answer_rule_bands: list[float],
) -> str | None:
    label = _page_furniture_box_label(box, layout, config, answer_rule_bands)
    if label in {"header_footer", "barcode", "scan_edge"}:
        return label
    return None


def _is_centered_page_number_block(block: TextBlock, layout: PageLayout, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if not re.fullmatch(r"\d{1,3}", text):
        return False
    center_x = (block.bbox.x0 + block.bbox.x1) / 2
    if not (layout.width * 0.35 <= center_x <= layout.width * 0.65):
        return False
    near_top = block.bbox.y0 <= config.detection.crop_top_margin + 18
    near_bottom = block.bbox.y1 >= layout.height - config.detection.crop_bottom_margin - 18
    return near_top or near_bottom


def _protected_region_boxes(region: CropRegion) -> list[BoundingBox]:
    boxes = [block.bbox for block in region.text_blocks]
    boxes.extend(region.graphics)
    if region.text_bbox is not None:
        boxes.append(region.text_bbox)
    if region.figure_bbox is not None:
        boxes.append(region.figure_bbox)
    return boxes


def _trim_preserves_protected_boxes(protected_boxes: list[BoundingBox], *, top: float, bottom: float) -> bool:
    for box in protected_boxes:
        if box.y0 < top - 1.0 or box.y1 > bottom + 1.0:
            return False
    return True


def _trim_flags_for_labels(labels: Iterable[str]) -> list[str]:
    flags = ["crop_header_footer_trimmed"]
    for label in labels:
        if label == "centered_page_number":
            flags.append("centered_page_number_trimmed")
        elif label == "barcode":
            flags.append("barcode_trimmed")
        elif label == "scan_edge":
            flags.append("scan_edge_trimmed")
        elif label == "control_artifact":
            flags.append("control_artifact_trimmed")
    return flags


def _fallback_regions(span: QuestionSpan, layouts: list[PageLayout], config: AppConfig) -> list[CropRegion]:
    regions: list[CropRegion] = []
    for page_number in span.page_numbers:
        layout = _layout_by_number(layouts, page_number)
        top = span.start_y if page_number == span.start_page else config.detection.crop_top_margin
        bottom = span.end_y if page_number == span.end_page else layout.height - config.detection.crop_bottom_margin
        bbox = BoundingBox(
            config.detection.crop_left_margin,
            max(config.detection.crop_top_margin, top),
            layout.width - config.detection.crop_right_margin,
            min(layout.height - config.detection.crop_bottom_margin, bottom),
        )
        if bbox.y1 > bbox.y0:
            regions.append(CropRegion(page_number=page_number, bbox=bbox))
    return regions


def _render_full_region_image(
    pdf_path: str | Path,
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
    *,
    identity: PaperIdentity,
) -> RenderResult:
    """Render the full exam question region for debugging."""

    try:
        import fitz
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PyMuPDF and Pillow are required for rendering screenshots.") from exc
    quiet_mupdf(fitz)

    asset = AssetPathResolver(config.output.root_dir()).question_image(identity)
    output_path = asset.absolute_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []
    regions: list[CropRegion] = []
    debug_paths: list[str] = []

    with fitz.open(pdf_path) as doc:
        rendered_pages = {}
        for page_number in span.page_numbers:
            layout = _layout_by_number(layouts, page_number)
            crop = _full_region_crop_for_page(layout, span, config)
            if crop is None:
                continue
            regions.append(CropRegion(page_number=page_number, bbox=crop, text_blocks=[block for block in span.blocks if block.page_number == page_number]))
            page = doc[page_number - 1]
            rect = fitz.Rect(crop.x0, crop.y0, crop.x1, crop.y1)
            image, used_zoom = render_pdf_area(
                page,
                fitz,
                dpi=config.detection.render_dpi,
                source_file=pdf_path,
                page_number=page_number,
                context=f"question_full_region:{span.question_number}",
                clip=rect,
            )
            images.append(image)
            if config.debug.enabled and page_number not in rendered_pages:
                page_image, page_zoom = render_pdf_area(
                    page,
                    fitz,
                    dpi=config.detection.render_dpi,
                    source_file=pdf_path,
                    page_number=page_number,
                    context=f"question_debug_page:{span.question_number}",
                )
                rendered_pages[page_number] = (page_image, page_zoom)
                if config.debug.save_rendered_pages:
                    debug_paths.append(_save_debug_image(page_image, span, page_number, "rendered", config))

        if config.debug.enabled:
            debug_paths.extend(_write_debug_overlays(rendered_pages, span, layouts, regions, config))

    if not images:
        return RenderResult(
            None,
            ["crop_fallback_failed", "crop_uncertain"],
            crop_uncertain=True,
            crop_diagnostics=_crop_diagnostics(pdf_path, span, regions, ["crop_fallback_failed", "crop_uncertain"], identity=identity, asset=asset),
        )

    stitched = cap_image_pixels(
        _stitch_images(images, config.detection.stitch_gap_px),
        source_file=pdf_path,
        context=f"question_full_region_output:{span.question_number}",
    )
    stitched = clean_rendered_crop_image(stitched)
    stitched.save(output_path)
    ocr_result = run_question_crop_ocr(output_path, config)
    flags = ["full_region_mode"]
    if ocr_result.ocr_ran and ocr_result.ocr_failure_reason:
        flags.append("ocr_question_crop_failed")
    if config.debug.enabled:
        debug_paths.append(_write_crop_metadata(span, regions, flags, config))
    return RenderResult(
        output_path,
        review_flags=flags,
        debug_paths=debug_paths,
        extracted_text=span.combined_text,
        crop_diagnostics=_crop_diagnostics(pdf_path, span, regions, flags, identity=identity, asset=asset),
        question_id=identity.question_id,
        paper_id=identity.paper_id,
        component=identity.component,
        canonical_path=asset.canonical_path,
        ocr_ran=ocr_result.ocr_ran,
        ocr_engine=ocr_result.ocr_engine,
        ocr_text=ocr_result.ocr_text,
        ocr_text_trust=ocr_result.ocr_text_trust,
        ocr_failure_reason=ocr_result.ocr_failure_reason,
        ocr_text_role=ocr_result.ocr_text_role,
    )


def _full_region_crop_for_page(layout: PageLayout, span: QuestionSpan, config: AppConfig) -> BoundingBox | None:
    padding = config.detection.crop_padding
    top = span.start_y - padding if layout.page_number == span.start_page else config.detection.crop_top_margin
    bottom = span.end_y + padding if layout.page_number == span.end_page else layout.height - config.detection.crop_bottom_margin
    top = max(config.detection.crop_top_margin, top)
    bottom = min(layout.height - config.detection.crop_bottom_margin, bottom)
    if bottom <= top + 4:
        return None
    return BoundingBox(
        config.detection.crop_left_margin,
        top,
        max(config.detection.crop_left_margin + 20, layout.width - config.detection.crop_right_margin),
        bottom,
    )


def _write_debug_overlays(
    rendered_pages: dict[int, tuple["Image.Image", float]],
    span: QuestionSpan,
    layouts: list[PageLayout],
    regions: list[CropRegion],
    config: AppConfig,
) -> list[str]:
    from PIL import ImageDraw

    paths: list[str] = []
    for page_number, (page_image, zoom) in rendered_pages.items():
        layout = _layout_by_number(layouts, page_number)
        anchors = [
            anchor
            for anchor in detect_question_anchor_candidates([layout], config)
            if anchor.bbox is not None
        ]
        proposed = _proposed_region_for_page(layout, span, config)

        if config.debug.save_anchor_candidates:
            image = page_image.copy()
            draw = ImageDraw.Draw(image)
            for anchor in anchors:
                draw.rectangle(_pdf_box_to_pixel_box(anchor.bbox, zoom, image.size), outline="orange", width=4)
            paths.append(_save_debug_image(image, span, page_number, "anchor_candidates", config))

        if config.debug.save_text_boxes:
            image = page_image.copy()
            draw = ImageDraw.Draw(image)
            included = {
                (block.page_number, round(block.bbox.x0, 2), round(block.bbox.y0, 2), round(block.bbox.x1, 2), round(block.bbox.y1, 2))
                for region in regions
                if region.page_number == page_number
                for block in region.text_blocks
            }
            for block in layout.blocks:
                key = (block.page_number, round(block.bbox.x0, 2), round(block.bbox.y0, 2), round(block.bbox.x1, 2), round(block.bbox.y1, 2))
                color = "lime" if key in included else "dodgerblue"
                draw.rectangle(_pdf_box_to_pixel_box(block.bbox, zoom, image.size), outline=color, width=3 if key in included else 1)
            for anchor in anchors:
                draw.rectangle(_pdf_box_to_pixel_box(anchor.bbox, zoom, image.size), outline="orange", width=4)
            paths.append(_save_debug_image(image, span, page_number, "text_boxes", config))

        if config.debug.save_proposed_boxes and proposed is not None:
            image = page_image.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle(_pdf_box_to_pixel_box(proposed, zoom, image.size), outline="cyan", width=5)
            for anchor in anchors:
                draw.rectangle(_pdf_box_to_pixel_box(anchor.bbox, zoom, image.size), outline="orange", width=4)
            paths.append(_save_debug_image(image, span, page_number, "proposed_boxes", config))

        if config.debug.save_crop_boxes:
            image = page_image.copy()
            draw = ImageDraw.Draw(image)
            for region in [region for region in regions if region.page_number == page_number]:
                draw.rectangle(_pdf_box_to_pixel_box(region.bbox, zoom, image.size), outline="magenta", width=5)
                if region.text_bbox is not None:
                    draw.rectangle(_pdf_box_to_pixel_box(region.text_bbox, zoom, image.size), outline="lime", width=3)
                if region.figure_bbox is not None:
                    draw.rectangle(_pdf_box_to_pixel_box(region.figure_bbox, zoom, image.size), outline="red", width=3)
            for anchor in anchors:
                draw.rectangle(_pdf_box_to_pixel_box(anchor.bbox, zoom, image.size), outline="orange", width=4)
            paths.append(_save_debug_image(image, span, page_number, "crop_boxes", config))
    return paths


def _write_crop_metadata(span: QuestionSpan, regions: list[CropRegion], flags: list[str], config: AppConfig) -> str:
    path = _debug_path(span, "crop_boxes", config, suffix=".json")
    payload = {
        "paper_name": span.paper_name,
        "question_number": span.question_number,
        "flags": sorted(set(flags)),
        "regions": [
            {
                "page_number": region.page_number,
                "region_kind": region.region_kind,
                "bbox_pdf_points": {
                    "x0": round(region.bbox.x0, 2),
                    "y0": round(region.bbox.y0, 2),
                    "x1": round(region.bbox.x1, 2),
                    "y1": round(region.bbox.y1, 2),
                },
                "original_bbox_pdf_points": _box_payload(region.original_bbox or region.bbox),
                "text_bbox_pdf_points": _box_payload(region.text_bbox) if region.text_bbox else None,
                "figure_bbox_pdf_points": _box_payload(region.figure_bbox) if region.figure_bbox else None,
                "text_figure_overlap_area": round(region.text_figure_overlap_area, 2),
                "text_trimmed_for_figure": region.text_trimmed_for_figure,
                "footer_cutoff": _footer_cutoff_payload(region),
                "text_blocks": [block.text for block in region.text_blocks],
                "merged_blocks": len(region.text_blocks),
                "graphics_count": len(region.graphics),
                "duplicate_graphics_removed": region.duplicate_graphics_removed,
                "excluded_regions": region.excluded_regions,
            }
            for region in regions
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _display_path(path)


def _save_debug_image(image: "Image.Image", span: QuestionSpan, page_number: int, kind: str, config: AppConfig) -> str:
    path = _debug_path(span, f"p{page_number:02d}_{kind}", config)
    image.save(path)
    return _display_path(path)


def _debug_path(span: QuestionSpan, kind: str, config: AppConfig, suffix: str = ".png") -> Path:
    config.output.debug_dir.mkdir(parents=True, exist_ok=True)
    if span.question_number.isdigit():
        qid = f"q{int(span.question_number):02d}"
    else:
        qid = f"q{span.question_number}"
    return config.output.debug_dir / f"{span.paper_name}_{qid}_{kind}{suffix}"


def _proposed_region_for_page(layout: PageLayout, span: QuestionSpan, config: AppConfig) -> BoundingBox | None:
    if layout.page_number not in span.page_numbers:
        return None
    top = span.start_y if layout.page_number == span.start_page else config.detection.crop_top_margin
    bottom = span.end_y if layout.page_number == span.end_page else layout.height - config.detection.crop_bottom_margin
    if bottom <= top:
        return None
    return BoundingBox(
        config.detection.crop_left_margin,
        max(config.detection.crop_top_margin, top),
        layout.width - config.detection.crop_right_margin,
        min(layout.height - config.detection.bottom_margin, bottom),
    )


def _pdf_box_to_pixel_box(box: BoundingBox, zoom: float, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, min(width - 1, int(box.x0 * zoom)))
    top = max(0, min(height - 1, int(box.y0 * zoom)))
    right = max(left + 1, min(width, int(box.x1 * zoom)))
    bottom = max(top + 1, min(height, int(box.y1 * zoom)))
    return (left, top, right, bottom)


def _clamp_crop_to_prompt_area(box: BoundingBox, layout: PageLayout, config: AppConfig) -> BoundingBox:
    return BoundingBox(
        max(0, box.x0),
        max(config.detection.crop_top_margin, box.y0),
        min(layout.width, box.x1),
        min(layout.height - config.detection.bottom_margin, box.y1),
    )


def _trim_crop_furniture_edges(box: BoundingBox, layout: PageLayout, config: AppConfig) -> BoundingBox:
    return BoundingBox(
        max(box.x0, config.detection.crop_left_margin),
        box.y0,
        min(box.x1, layout.width - config.detection.crop_right_margin),
        box.y1,
    )


def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _dedupe_graphics(boxes: list[BoundingBox], seen: list[BoundingBox]) -> tuple[list[BoundingBox], int]:
    kept: list[BoundingBox] = []
    removed = 0
    for box in sorted(boxes, key=lambda item: (_box_area(item), item.y0, item.x0), reverse=True):
        if any(_boxes_duplicate(box, other) for other in kept) or any(_boxes_duplicate(box, other) for other in seen):
            removed += 1
            continue
        kept.append(box)
        seen.append(box)
    return sorted(kept, key=lambda item: (item.y0, item.x0)), removed


def _boxes_duplicate(a: BoundingBox, b: BoundingBox) -> bool:
    if _intersection_area(a, b) / max(1.0, min(_box_area(a), _box_area(b))) >= 0.88:
        return True
    smaller = min(_box_area(a), _box_area(b))
    larger = max(_box_area(a), _box_area(b))
    if smaller > 0 and smaller <= larger * 0.35 and _intersection_area(a, b) / smaller >= 0.65:
        return True
    return (
        abs(a.x0 - b.x0) <= 3
        and abs(a.y0 - b.y0) <= 3
        and abs(a.x1 - b.x1) <= 3
        and abs(a.y1 - b.y1) <= 3
    )


def _boxes_intersect(a: BoundingBox, b: BoundingBox) -> bool:
    return _intersection_area(a, b) > 1.0


def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    width = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    height = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return width * height


def _box_area(box: BoundingBox) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def _box_width(box: BoundingBox) -> float:
    return max(0.0, box.x1 - box.x0)


def _horizontal_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    return overlap / max(1.0, min(_box_width(a), _box_width(b)))


def _is_footer_or_header_box(box: BoundingBox, layout: PageLayout, config: AppConfig) -> bool:
    return box.y1 < config.detection.crop_top_margin or box.y0 > layout.height - config.detection.bottom_margin


def _page_furniture_box_label(
    box: BoundingBox,
    layout: PageLayout,
    config: AppConfig,
    answer_rule_bands: list[float],
) -> str | None:
    if _is_full_page_background_graphic(box, layout):
        return "page_background"
    if _is_watermark_like_box(box, layout):
        return "watermark"
    if _is_right_edge_watermark_fragment(box, layout):
        return "watermark"
    if _is_top_left_watermark_fragment(box, layout):
        return "watermark"
    if _is_bottom_edge_footer_graphic(box, layout):
        return "header_footer"
    if _is_full_height_page_edge_furniture_box(box, layout):
        return "page_edge_furniture"
    if _is_footer_or_header_box(box, layout, config):
        return "header_footer"
    if _is_answer_rule_like(box, layout) or _is_in_answer_rule_band(box, answer_rule_bands):
        return "answer_lines"
    if _is_side_panel_box(box, layout, config):
        return "side_panel"
    if _is_barcode_like_box(box, layout, config):
        return "barcode"
    if _is_scan_edge_box(box, layout):
        return "scan_edge"
    return None


def _is_side_panel_box(box: BoundingBox, layout: PageLayout, config: AppConfig) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    near_left = box.x0 <= config.detection.crop_left_margin * 0.8
    near_right = box.x1 >= layout.width - config.detection.crop_right_margin * 0.8
    return width <= 55 and height >= layout.height * 0.16 and (near_left or near_right)


def _is_barcode_like_box(box: BoundingBox, layout: PageLayout, config: AppConfig) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    return box.y0 <= config.detection.crop_top_margin + 70 and height <= 90 and 20 <= width <= layout.width * 0.45


def _is_scan_edge_box(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    near_edge = box.x0 <= 4 or box.x1 >= layout.width - 4 or box.y0 <= 4 or box.y1 >= layout.height - 4
    return near_edge and (width <= 8 or height <= 8)


def _is_full_page_background_graphic(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    touches_page_edge = box.x0 <= 4 or box.x1 >= layout.width - 4 or box.y0 <= 4 or box.y1 >= layout.height - 4
    return touches_page_edge and width >= layout.width * 0.82 and height >= layout.height * 0.75


def _is_watermark_like_box(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    touches_page_edge = box.x0 <= 4 or box.x1 >= layout.width - 4 or box.y0 <= 4 or box.y1 >= layout.height - 4
    if (
        touches_page_edge
        and box.x0 >= layout.width * 0.55
        and box.y0 <= layout.height * 0.08
        and width >= layout.width * 0.20
        and height >= layout.height * 0.16
    ):
        return True
    if width < layout.width * 0.82 or height < layout.height * 0.12:
        return False
    if height >= layout.height * 0.75:
        return False
    if not touches_page_edge:
        return False
    near_top_or_bottom = box.y0 <= layout.height * 0.18 or box.y1 >= layout.height * 0.82
    return near_top_or_bottom


def _is_right_edge_watermark_fragment(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    touches_right_edge = box.x1 >= layout.width - 4
    return (
        touches_right_edge
        and box.x0 >= layout.width * 0.62
        and layout.width * 0.08 <= width <= layout.width * 0.38
        and layout.height * 0.035 <= height <= layout.height * 0.24
    )


def _is_top_left_watermark_fragment(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    touches_top_left = box.x0 <= 4 and box.y0 <= layout.height * 0.03
    return (
        touches_top_left
        and layout.width * 0.20 <= width <= layout.width * 0.55
        and layout.height * 0.07 <= height <= layout.height * 0.20
    )


def _is_bottom_edge_footer_graphic(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    touches_bottom = box.y1 >= layout.height - 4
    return touches_bottom and box.y0 >= layout.height * 0.9 and width >= layout.width * 0.75 and height <= layout.height * 0.12


def _is_full_height_page_edge_furniture_box(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    touches_page_side = box.x0 <= 4 or box.x1 >= layout.width - 4
    spans_page_height = box.y0 <= 4 and box.y1 >= layout.height - 4
    return (
        touches_page_side
        and spans_page_height
        and height >= layout.height * 0.82
        and layout.width * 0.12 <= width <= layout.width * 0.65
    )


def _is_answer_rule_like(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    return height <= 2.5 and width >= layout.width * 0.28


def _is_formula_rule_box(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    return height <= 1.5 and 12 <= width < layout.width * 0.22


def _answer_rule_y_bands(layout: PageLayout) -> list[float]:
    rows: dict[int, list[BoundingBox]] = {}
    for graphic in layout.graphics:
        width = max(0.0, graphic.x1 - graphic.x0)
        height = max(0.0, graphic.y1 - graphic.y0)
        if height > 2.5 or width <= 1:
            continue
        y_key = round(((graphic.y0 + graphic.y1) / 2) / 2)
        rows.setdefault(y_key, []).append(graphic)

    bands: list[float] = []
    for y_key, boxes in rows.items():
        total_width = sum(box.x1 - box.x0 for box in boxes)
        if total_width >= layout.width * 0.25 or len(boxes) >= 5:
            bands.append(y_key * 2)
    return bands


def _is_in_answer_rule_band(box: BoundingBox, bands: list[float]) -> bool:
    if not bands:
        return False
    y_mid = (box.y0 + box.y1) / 2
    return any(abs(y_mid - band) <= 2.5 for band in bands)


def _is_boilerplate_text(text: str) -> bool:
    patterns = [
        r"^Additional Page\b",
        r"If you use the following lined page",
        r"write the question number",
        r"^©\s*UCLES\b",
        r"^UCLES\b",
        r"^\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}$",
        r"^9709[/_ -]",
        r"^Cambridge International",
        r"DO NOT WRITE IN THIS MARGIN",
        r"^This document consists of",
        r"^BLANK PAGE$",
        r"^Question Paper$",
        r"^Mark Scheme$",
        r"^Turn over$",
    ]
    return _is_source_pagination_note_text(text) or any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _is_source_pagination_note_text(text: str) -> bool:
    normalized = _clean_text_line(text)
    return bool(
        re.fullmatch(
            r"\[\s*Questions?\s+\d+(?:\([a-zivx]+\))?(?:\s*(?:and|,)\s*\d+(?:\([a-zivx]+\))?)*\s+"
            r"(?:is|are)\s+printed\s+on\s+the\s+next\s+page\.?\s*\]",
            normalized,
            re.IGNORECASE,
        )
    )


def _is_margin_furniture_text(block: TextBlock, layout: PageLayout, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if re.search(r"DO NOT WRITE IN THIS MARGIN", text, re.IGNORECASE):
        return True
    narrow_edge = (block.bbox.x1 - block.bbox.x0) <= 70 and (
        block.bbox.x0 <= config.detection.crop_left_margin or block.bbox.x1 >= layout.width - config.detection.crop_right_margin
    )
    tall = (block.bbox.y1 - block.bbox.y0) >= 80
    return narrow_edge and tall


def _is_control_artifact_text(text: str) -> bool:
    if not text:
        return False
    control_count = sum(1 for char in text if ord(char) < 32 and char not in "\n\t\r")
    if control_count == 0:
        return False
    cleaned = _strip_control_chars(text).strip()
    visible_count = sum(1 for char in cleaned if not char.isspace())
    if visible_count <= 3:
        return True
    return control_count >= max(4, visible_count)


def _is_answer_space_text(text: str) -> bool:
    if re.fullmatch(r"[._\-–—\s]{6,}", text):
        return True
    if re.fullmatch(r"(?:\.\s*){6,}", text):
        return True
    rule_chars = sum(1 for char in text if char in "._-–—")
    visible_chars = sum(1 for char in text if not char.isspace())
    if rule_chars >= 12 and rule_chars / max(1, visible_chars) >= 0.55:
        return True
    return bool(re.search(r"\bAnswer\b\s*[._\-–—]{6,}", text, re.IGNORECASE))


def _text_from_regions(regions: list[CropRegion]) -> str:
    blocks: list[TextBlock] = []
    for region in regions:
        if region.region_kind in {"figure", "context_inferred_figure", "text_diagram_union"}:
            continue
        blocks.extend(region.text_blocks)
    return extract_text_from_blocks(blocks)


def _stitch_images(images: list["Image.Image"], gap_px: int) -> "Image.Image":
    from PIL import Image

    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap_px * max(0, len(images) - 1)
    stitched = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        stitched.paste(image, (0, y))
        y += image.height + gap_px
    return stitched


def _question_identity_from_span(span: QuestionSpan) -> PaperIdentity | None:
    try:
        metadata = parse_filename_metadata(span.source_pdf)
        return paper_identity_from_parts(
            syllabus=metadata.syllabus or "9709",
            subject_family=metadata.paper_family,
            year=metadata.year,
            session=session_for_source_path(
                span.source_pdf,
                year=metadata.year,
                fallback_session=metadata.normalized_session_key or metadata.session,
            ),
            component=metadata.component,
            question_number=span.question_number,
        )
    except IdentityError:
        return None


def _missing_identity_render_result(span: QuestionSpan) -> RenderResult:
    return RenderResult(
        screenshot_path=None,
        review_flags=["identity_unresolved", "question_asset_not_emitted"],
        crop_uncertain=True,
        extracted_text=span.combined_text,
        crop_diagnostics={
            "source_file": str(span.source_pdf),
            "question_number": span.question_number,
            "flags": ["identity_unresolved", "question_asset_not_emitted"],
            "regions": [],
        },
    )


def _crop_diagnostics(
    pdf_path: str | Path,
    span: QuestionSpan,
    regions: list[CropRegion],
    flags: list[str],
    *,
    identity: PaperIdentity,
    asset: AssetPath,
) -> dict[str, object]:
    detected_figure_regions = [region for region in regions if region.graphics or region.figure_bbox is not None or "figure" in region.region_kind]
    missing_image_reason = "detection_failure" if _span_references_source_visual(span) and not detected_figure_regions else ""
    return {
        "source_file": str(pdf_path),
        "question_number": span.question_number,
        "question_id": identity.question_id,
        "paper_id": identity.paper_id,
        "component": identity.component,
        "canonical_path": asset.canonical_path,
        "flags": sorted(set(flags)),
        "merged_blocks": sum(len(region.text_blocks) for region in regions),
        "duplicate_visual_blocks_removed": sum(region.duplicate_graphics_removed for region in regions),
        "detected_figure_count": len(detected_figure_regions),
        "missing_image_reason": missing_image_reason,
        "missing_image_failure_metadata": (
            {
                "reason": missing_image_reason,
                "question_pages": list(span.page_numbers),
                "detection_methods_attempted": [
                    "embedded_image_objects",
                    "vector_graphic_regions",
                    "bbox_detected_diagrams",
                    "ocr_hint_signals",
                    "question_context_inference",
                ],
            }
            if missing_image_reason
            else {}
        ),
        "excluded_boilerplate_reasons": sorted(flag.replace("excluded_boilerplate_", "") for flag in flags if flag.startswith("excluded_boilerplate_")),
        "regions": [
            {
                "page_number": region.page_number,
                "region_kind": region.region_kind,
                "original_crop_bbox": _box_payload(region.original_bbox or region.bbox),
                "final_crop_bbox": {
                    "x0": round(region.bbox.x0, 2),
                    "y0": round(region.bbox.y0, 2),
                    "x1": round(region.bbox.x1, 2),
                    "y1": round(region.bbox.y1, 2),
                },
                "text_bbox": _box_payload(region.text_bbox) if region.text_bbox else None,
                "figure_bbox": _box_payload(region.figure_bbox) if region.figure_bbox else None,
                "text_figure_overlap_area": round(region.text_figure_overlap_area, 2),
                "text_trimmed_for_figure": region.text_trimmed_for_figure,
                "footer_cutoff": _footer_cutoff_payload(region),
                "merged_blocks": len(region.text_blocks),
                "graphics_count": len(region.graphics),
                "duplicate_visual_blocks_removed": region.duplicate_graphics_removed,
                "excluded_regions": region.excluded_regions,
            }
            for region in regions
        ],
    }


def _excluded_region(label: str, box: BoundingBox) -> dict[str, object]:
    return {"label": label, "bbox": _box_payload(box)}


def _footer_cutoff_payload(region: CropRegion) -> dict[str, object]:
    if region.footer_cutoff is not None:
        return region.footer_cutoff
    original_bottom = (region.original_bbox or region.bbox).y1
    return {
        "original_bottom": round(original_bottom, 2),
        "detected_footer_cutoff_y": None,
        "reason": "not_evaluated",
        "signals": [],
        "final_bottom": round(region.bbox.y1, 2),
    }


def _box_payload(box: BoundingBox) -> dict[str, float]:
    return {
        "x0": round(box.x0, 2),
        "y0": round(box.y0, 2),
        "x1": round(box.x1, 2),
        "y1": round(box.y1, 2),
    }


def _layout_by_number(layouts: list[PageLayout], page_number: int) -> PageLayout:
    for layout in layouts:
        if layout.page_number == page_number:
            return layout
    raise ValueError(f"No layout for page {page_number}")


def _box_height(box: BoundingBox) -> float:
    return max(0.0, box.y1 - box.y0)


def _clean_text_line(text: str) -> str:
    return " ".join(_strip_control_chars(text).replace("\u00a0", " ").split())


def _strip_control_chars(text: str) -> str:
    return "".join(char if ord(char) >= 32 or char in "\n\t\r" else " " for char in text)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
