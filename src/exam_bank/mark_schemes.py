from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path

from .config import AppConfig
from .core.asset_paths import AssetPath, AssetPathResolver
from .core.paper_identity import IdentityError, PaperIdentity, paper_identity_from_parts, session_for_source_path
from .document_metadata import parse_filename_metadata
from .image_limits import cap_image_pixels, clean_rendered_crop_image, render_pdf_area
from .identifiers import normalize_question_id, parent_question_id
from .mark_scheme_models import (
    HeaderGeometry,
    MarkSchemeAnchor,
    MarkSchemeBlock,
    MarkSchemeCropRegion,
    MarkSchemeImageResult,
    MarkSchemeRow,
    MarkSchemeTable,
    MarkSchemeWord,
)
from .mark_scheme_pairing import find_mark_scheme
from .models import BoundingBox, PageLayout, QuestionStart, TextBlock
from .mupdf_tools import quiet_mupdf
from .pdf_extract import _visual_box_from_rect
from .pdf_extract import extract_pdf_layout
from .question_detection import parse_question_start
from .trust import CropConfidence, MappingStatus


@dataclass(frozen=True)
class _LegacyTableGrid:
    page_number: int
    bbox: BoundingBox
    question_col_right: float
    horizontal_rules: list[float]
    vertical_rules: list[float]


@dataclass(frozen=True)
class _LegacyRowBand:
    page_number: int
    y0: float
    y1: float
    x0: float
    x1: float
    question_col_right: float
    question_label: str | None


@dataclass(frozen=True)
class _CropValidation:
    passed: bool
    detected_labels: list[str]
    rejected_reason: str = ""


def extract_mark_scheme_answers(
    mark_scheme_pdf: str | Path,
    config: AppConfig,
    expected_numbers: list[str] | None = None,
) -> dict[str, str]:
    mark_scheme_pdf = Path(mark_scheme_pdf)
    layouts = extract_pdf_layout(mark_scheme_pdf, config)
    words = _extract_mark_scheme_words(mark_scheme_pdf)
    tables = _detect_mark_scheme_tables(layouts, config, words)
    anchors = _detect_table_question_anchors(layouts, tables, config, None, words)
    if not anchors:
        return {
            number: block.text
            for number, block in _build_legacy_mark_scheme_blocks(
                mark_scheme_pdf,
                layouts,
                config,
                expected_numbers,
                question_marks={},
                question_subparts={},
            ).items()
            if block.text
        }
    answers: dict[str, str] = {}
    ordered = sorted(anchors, key=lambda item: (item.page_number, item.y0))
    for number in expected_numbers or [anchor.question_number for anchor in ordered]:
        canonical_number = normalize_question_id(number)
        anchor_index, anchor = _anchor_for_question(ordered, canonical_number)
        if anchor is None or anchor_index is None:
            continue
        next_anchor = _next_boundary_anchor(ordered, anchor_index, canonical_number)
        table = anchor.table
        if table is None:
            continue
        blocks = _blocks_for_table_anchor_bounds(layouts, tables, anchor, next_anchor, config)
        text = "\n".join(block.text for block in blocks).strip()
        if text:
            answers[canonical_number] = text
    return answers


def render_mark_scheme_images(
    mark_scheme_pdf: str | Path,
    config: AppConfig,
    expected_numbers: list[str] | None = None,
    question_marks: dict[str, int | None] | None = None,
    question_subparts: dict[str, list[str]] | None = None,
    question_validation_flags: dict[str, list[str]] | None = None,
    question_identities: dict[str, PaperIdentity] | None = None,
    clear_stale: bool = True,
) -> dict[str, MarkSchemeImageResult]:
    """Crop rendered mark-scheme answer regions by top-level question number.

    This keeps mathematical notation exactly as it appears in the source PDF.
    Text extraction is used only to locate the answer boundaries; the exported
    artifact is a crop of the original rendered mark-scheme page.
    """

    if not expected_numbers:
        return {}
    question_marks = question_marks or {}
    question_subparts = question_subparts or {}
    question_validation_flags = question_validation_flags or {}
    identities = _mark_scheme_question_identities(mark_scheme_pdf, expected_numbers, question_identities)

    try:
        import fitz
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PyMuPDF and Pillow are required for mark-scheme image export.") from exc
    quiet_mupdf(fitz)

    mark_scheme_pdf = Path(mark_scheme_pdf)
    words = _extract_mark_scheme_words(mark_scheme_pdf)
    layouts = extract_pdf_layout(mark_scheme_pdf, config)
    tables = _detect_mark_scheme_tables(layouts, config, words)
    anchors = _detect_table_question_anchors(layouts, tables, config, None, words)
    if not tables or not anchors:
        return _render_legacy_mark_scheme_images(
            mark_scheme_pdf,
            config,
            expected_numbers,
            question_marks=question_marks,
            question_subparts=question_subparts,
            identities=identities,
            no_blocks_failure_reason="segmentation_failure",
            clear_stale=clear_stale,
        )

    output: dict[str, MarkSchemeImageResult] = {}
    if clear_stale:
        _clear_stale_mark_scheme_images(mark_scheme_pdf, expected_numbers, config, identities)
    legacy_fallback_blocks = _build_legacy_mark_scheme_blocks(
        mark_scheme_pdf,
        layouts,
        config,
        expected_numbers,
        question_marks=question_marks,
        question_subparts=question_subparts,
    )
    legacy_fallback_anchors = [
        block.anchor
        for block in sorted(legacy_fallback_blocks.values(), key=lambda item: (item.anchor.page_number, item.anchor.y0))
    ]
    with fitz.open(mark_scheme_pdf) as doc:
        rendered_pages = {}
        ordered_anchors = sorted(anchors, key=lambda item: (item.page_number, item.y0))
        for number in expected_numbers:
            canonical_number = normalize_question_id(number)
            identity = identities.get(canonical_number)
            identity_fields = _mark_scheme_identity_fields(identity, config)
            question_subpart_values = question_subparts.get(canonical_number, [])
            question_marks_total = question_marks.get(canonical_number)
            if identity is None:
                output[number] = MarkSchemeImageResult(
                    question_number=number,
                    crop_confidence=CropConfidence.LOW,
                    mapping_method="table_row_block",
                    table_detected=bool(tables),
                    question_subparts=question_subpart_values,
                    question_marks_total=question_marks_total,
                    mapping_status=MappingStatus.FAIL,
                    failure_reason="identity_unresolved",
                    missing_mark_scheme_reason="identity_unresolved",
                    review_flags=["markscheme_image_missing", "identity_unresolved", "markscheme_asset_not_emitted"],
                )
                continue
            anchor_index, anchor = _anchor_for_question(ordered_anchors, canonical_number)
            if anchor is not None:
                next_anchor = _next_boundary_anchor(ordered_anchors, anchor_index, canonical_number)
                regions, flags = _table_regions_for_anchor(layouts, tables, anchor, next_anchor, config)
                if anchor.question_number != canonical_number and parent_question_id(anchor.question_number) != canonical_number:
                    flags.append("markscheme_parent_label_match")
                mapping_method = "table_row_block"
                table_detected = True
                nearby_anchors = _nearby_anchor_labels(ordered_anchors, anchor)
            else:
                legacy_block = legacy_fallback_blocks.get(canonical_number)
                if legacy_block is not None:
                    flags = list(legacy_block.review_flags)
                    flags.append("markscheme_table_anchor_missing_legacy_fallback")
                    output_path, debug_paths = _render_mark_scheme_crops(
                        doc,
                        fitz,
                        mark_scheme_pdf,
                        canonical_number,
                        identity,
                        legacy_block.regions,
                        flags,
                        rendered_pages,
                        layouts,
                        {},
                        legacy_fallback_anchors,
                        config,
                        words_by_page=words,
                        crop_method=legacy_block.method,
                    )
                    if output_path is None:
                        output[number] = MarkSchemeImageResult(
                            question_number=number,
                            **identity_fields,
                            markscheme_question_number=canonical_number,
                            crop_confidence=CropConfidence.LOW,
                            mapping_method=legacy_block.method,
                            table_detected=bool(tables),
                            detected_anchor_pages=[legacy_block.anchor.page_number],
                            nearby_anchors=_nearby_anchor_labels(legacy_fallback_anchors, legacy_block.anchor),
                            debug_paths=debug_paths,
                            review_flags=sorted(set(flags + ["markscheme_image_missing"])),
                            question_subparts=question_subpart_values,
                            markscheme_subparts=legacy_block.subparts,
                            question_marks_total=question_marks_total,
                            markscheme_marks_total=legacy_block.mark_total,
                            mapping_status=MappingStatus.FAIL,
                            failure_reason="segmentation_failure",
                            block_ids=[legacy_block.block_id],
                            confidence_score=legacy_block.confidence_score,
                            missing_mark_scheme_reason="segmentation_failure",
                        )
                        continue

                    output[number] = MarkSchemeImageResult(
                        question_number=number,
                        image_path=output_path,
                        **identity_fields,
                        page_numbers=[region.page_number for region in legacy_block.regions],
                        markscheme_question_number=canonical_number,
                        crop_confidence=_mark_scheme_crop_confidence(legacy_block.regions, layouts, flags),
                        mapping_method=legacy_block.method,
                        table_detected=bool(tables),
                        detected_anchor_pages=[legacy_block.anchor.page_number],
                        nearby_anchors=_nearby_anchor_labels(legacy_fallback_anchors, legacy_block.anchor),
                        debug_paths=debug_paths,
                        review_flags=sorted(set(flags)),
                        table_header_ok=False,
                        continuation_rows_included=any(region.continuation_rows_included for region in legacy_block.regions),
                        question_subparts=question_subpart_values,
                        markscheme_subparts=legacy_block.subparts,
                        question_marks_total=question_marks_total,
                        markscheme_marks_total=legacy_block.mark_total,
                        mapping_status=MappingStatus.PASS,
                        block_ids=[legacy_block.block_id],
                        confidence_score=legacy_block.confidence_score,
                    )
                    continue

                output[number] = MarkSchemeImageResult(
                    question_number=number,
                    **identity_fields,
                    crop_confidence=CropConfidence.LOW,
                    mapping_method="table_row_block",
                    table_detected=bool(tables),
                    question_subparts=question_subpart_values,
                    question_marks_total=question_marks_total,
                    mapping_status=MappingStatus.FAIL,
                    failure_reason="segmentation_failure",
                    missing_mark_scheme_reason="segmentation_failure",
                    review_flags=["markscheme_image_missing", "markscheme_no_row_for_question"],
                    nearby_anchors=[item.question_number for item in ordered_anchors[:8]],
                )
                continue

            markscheme_subpart_values = (
                _detected_subparts_for_question(ordered_anchors, anchor_index, canonical_number)
                if anchor_index is not None
                else []
            )
            markscheme_marks_total = _mark_total_for_question_block(
                layouts,
                anchor,
                next_anchor,
                tables,
                words,
                question_marks_total,
            )
            validation_flags, failure_reason = _validate_mark_scheme_mapping(
                canonical_number,
                question_subpart_values,
                markscheme_subpart_values,
                question_marks_total,
                markscheme_marks_total,
                anchor,
                next_anchor,
                regions,
                flags,
                question_validation_flags=question_validation_flags.get(canonical_number, []),
            )
            flags.extend(validation_flags)
            if not regions:
                output[number] = MarkSchemeImageResult(
                    question_number=number,
                    **identity_fields,
                    markscheme_question_number=canonical_number if anchor else "",
                    crop_confidence=CropConfidence.LOW,
                    mapping_method=mapping_method,
                    table_detected=table_detected,
                    table_header_detected=anchor.table.header_detected if anchor and anchor.table else [],
                    nearby_anchors=nearby_anchors,
                    review_flags=sorted(set(flags + ["markscheme_image_missing"])),
                    table_header_ok=bool(anchor and anchor.table and _table_header_ok(anchor.table)),
                    question_subparts=question_subpart_values,
                    markscheme_subparts=markscheme_subpart_values,
                    question_marks_total=question_marks_total,
                    markscheme_marks_total=markscheme_marks_total,
                    mapping_status=MappingStatus.FAIL,
                    failure_reason=failure_reason or "partial_question_block",
                    block_ids=[_legacy_mark_scheme_block_id(mark_scheme_pdf, canonical_number)] if anchor else [],
                    confidence_score=0.35,
                    missing_mark_scheme_reason="segmentation_failure",
                )
                continue

            output_path, debug_paths = _render_mark_scheme_crops(
                doc,
                fitz,
                mark_scheme_pdf,
                number,
                identity,
                regions,
                flags,
                rendered_pages,
                layouts,
                tables,
                ordered_anchors,
                config,
                words_by_page=words,
                crop_method="table_grid",
            )
            if output_path is None:
                output[number] = MarkSchemeImageResult(
                    question_number=number,
                    **identity_fields,
                    markscheme_question_number=anchor.question_number if anchor else "",
                    crop_confidence=CropConfidence.LOW,
                    mapping_method=mapping_method,
                    table_detected=table_detected,
                    table_header_detected=anchor.table.header_detected if anchor and anchor.table else [],
                    nearby_anchors=nearby_anchors,
                    review_flags=sorted(set(flags + ["markscheme_image_missing"])),
                    table_header_ok=bool(anchor and anchor.table and _table_header_ok(anchor.table)),
                    question_subparts=question_subpart_values,
                    markscheme_subparts=markscheme_subpart_values,
                    question_marks_total=question_marks_total,
                    markscheme_marks_total=markscheme_marks_total,
                    mapping_status=MappingStatus.FAIL,
                    failure_reason=failure_reason or "partial_question_block",
                    block_ids=[_legacy_mark_scheme_block_id(mark_scheme_pdf, canonical_number)] if anchor else [],
                    confidence_score=0.45,
                    missing_mark_scheme_reason="segmentation_failure",
                )
                continue

            confidence = _mark_scheme_crop_confidence(regions, layouts, flags)
            if failure_reason:
                output[number] = MarkSchemeImageResult(
                    question_number=number,
                    image_path=output_path,
                    **identity_fields,
                    page_numbers=[region.page_number for region in regions],
                    markscheme_question_number=canonical_number if anchor else "",
                    crop_confidence=confidence,
                    mapping_method=mapping_method,
                    table_detected=table_detected,
                    table_header_detected=anchor.table.header_detected if anchor and anchor.table else [],
                    detected_anchor_pages=[anchor.page_number] if anchor else [],
                    nearby_anchors=nearby_anchors,
                    debug_paths=debug_paths,
                    review_flags=sorted(set(flags)),
                    table_header_ok=bool(anchor and anchor.table and _table_header_ok(anchor.table)),
                    continuation_rows_included=any(region.continuation_rows_included for region in regions),
                    question_subparts=question_subpart_values,
                    markscheme_subparts=markscheme_subpart_values,
                    question_marks_total=question_marks_total,
                    markscheme_marks_total=markscheme_marks_total,
                    mapping_status=MappingStatus.FAIL,
                    failure_reason=failure_reason,
                    block_ids=[_legacy_mark_scheme_block_id(mark_scheme_pdf, canonical_number)],
                    confidence_score=_table_mark_scheme_confidence_score(confidence, failure_reason),
                )
                continue

            output[number] = MarkSchemeImageResult(
                question_number=number,
                image_path=output_path,
                **identity_fields,
                page_numbers=[region.page_number for region in regions],
                markscheme_question_number=canonical_number if anchor else "",
                crop_confidence=confidence,
                mapping_method=mapping_method,
                table_detected=table_detected,
                table_header_detected=anchor.table.header_detected if anchor and anchor.table else [],
                detected_anchor_pages=[anchor.page_number] if anchor else [],
                nearby_anchors=nearby_anchors,
                debug_paths=debug_paths,
                review_flags=sorted(set(flags)),
                table_header_ok=bool(anchor and anchor.table and _table_header_ok(anchor.table)),
                continuation_rows_included=any(region.continuation_rows_included for region in regions),
                question_subparts=question_subpart_values,
                markscheme_subparts=markscheme_subpart_values,
                question_marks_total=question_marks_total,
                markscheme_marks_total=markscheme_marks_total,
                mapping_status=MappingStatus.PASS,
                block_ids=[_legacy_mark_scheme_block_id(mark_scheme_pdf, canonical_number)],
                confidence_score=_table_mark_scheme_confidence_score(confidence, ""),
            )

    found = set(output)
    for number in expected_numbers:
        if number not in found:
            output[number] = MarkSchemeImageResult(
                question_number=number,
                **_mark_scheme_identity_fields(identities.get(normalize_question_id(number)), config),
                crop_confidence=CropConfidence.LOW,
                mapping_method="table_row_block",
                table_detected=bool(tables),
                question_subparts=question_subparts.get(number, []),
                question_marks_total=question_marks.get(number),
                mapping_status=MappingStatus.FAIL,
                failure_reason="segmentation_failure",
                missing_mark_scheme_reason="segmentation_failure",
                review_flags=["markscheme_image_missing"],
            )
    return output


def _render_mark_scheme_crops(
    doc,
    fitz,
    mark_scheme_pdf: Path,
    question_number: str,
    identity: PaperIdentity,
    regions: list[MarkSchemeCropRegion],
    flags: list[str],
    rendered_pages: dict[int, tuple["Image.Image", float]],
    layouts: list[PageLayout],
    tables: dict[int, MarkSchemeTable],
    ordered_anchors: list[MarkSchemeAnchor],
    config: AppConfig,
    *,
    words_by_page: dict[int, list[MarkSchemeWord]] | None = None,
    crop_method: str = "fallback",
) -> tuple[Path | None, list[str]]:
    crops = []
    debug_paths: list[str] = []
    validation = _validate_mark_scheme_crop_before_save(
        question_number,
        mark_scheme_pdf,
        identity,
        regions,
        layouts,
        config,
        words_by_page or {},
        ordered_anchors,
    )
    debug_paths.append(
        _write_mark_scheme_crop_debug_record(
            question_number=question_number,
            identity=identity,
            mark_scheme_pdf=mark_scheme_pdf,
            regions=regions,
            crop_method=crop_method,
            validation=validation,
            config=config,
        )
    )
    if not validation.passed:
        flags.append("markscheme_validation_failed")
        flags.append(f"markscheme_validation_failed:{validation.rejected_reason}")
        return None, debug_paths

    for region in regions:
        page_number = region.page_number
        box = region.bbox
        page = doc[page_number - 1]
        rect = fitz.Rect(box.x0, box.y0, box.x1, box.y1)
        crop, used_zoom = render_pdf_area(
            page,
            fitz,
            dpi=config.detection.render_dpi,
            source_file=mark_scheme_pdf,
            page_number=page_number,
            context=f"markscheme_crop:{question_number}",
            clip=rect,
        )
        crops.append(crop)
        if used_zoom * 72 < config.detection.render_dpi * 0.8:
            flags.append("markscheme_render_dpi_capped")

        if config.debug.enabled and page_number not in rendered_pages:
            page_image, page_zoom = render_pdf_area(
                page,
                fitz,
                dpi=config.detection.render_dpi,
                source_file=mark_scheme_pdf,
                page_number=page_number,
                context=f"markscheme_debug_page:{question_number}",
            )
            rendered_pages[page_number] = (page_image, page_zoom)
            if config.debug.save_rendered_pages:
                debug_paths.append(
                    _save_mark_scheme_debug_image(page_image, mark_scheme_pdf, question_number, page_number, "rendered", config)
                )

    if not crops:
        return None, debug_paths

    if config.debug.enabled:
        debug_paths.extend(
            _write_mark_scheme_debug_overlays(rendered_pages, mark_scheme_pdf, question_number, layouts, tables, ordered_anchors, regions, config)
        )
        debug_paths.append(_write_mark_scheme_debug_metadata(mark_scheme_pdf, question_number, tables, ordered_anchors, regions, config))

    output_path = AssetPathResolver(config.output.root_dir()).mark_scheme_image(identity).absolute_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched = cap_image_pixels(
        _stitch_images(crops, config.detection.stitch_gap_px),
        source_file=mark_scheme_pdf,
        context=f"markscheme_output:{question_number}",
    )
    stitched = clean_rendered_crop_image(stitched)
    stitched.save(output_path)
    return output_path, debug_paths


def _validate_mark_scheme_crop_before_save(
    question_number: str,
    mark_scheme_pdf: Path,
    identity: PaperIdentity,
    regions: list[MarkSchemeCropRegion],
    layouts: list[PageLayout],
    config: AppConfig,
    words_by_page: dict[int, list[MarkSchemeWord]],
    anchors: list[MarkSchemeAnchor] | None = None,
) -> _CropValidation:
    del mark_scheme_pdf, identity
    canonical_number = parent_question_id(question_number)
    if not regions:
        return _CropValidation(False, [], "empty_crop")

    word_labels = _left_column_labels_in_regions(regions, words_by_page)
    anchor_labels = _left_column_anchor_labels_in_regions(regions, anchors or [])
    labels = _ordered_unique(anchor_labels or word_labels)
    parent_labels = [parent_question_id(label) for label in labels if parent_question_id(label)]
    unique_parents = _ordered_unique(parent_labels)
    if canonical_number not in unique_parents:
        return _CropValidation(False, labels, "target_question_missing")
    if any(label != canonical_number for label in unique_parents):
        return _CropValidation(False, labels, "adjacent_primary_question_in_left_column")

    for region in regions:
        layout = _layout_by_number(layouts, region.page_number)
        if region.bbox.y1 <= region.bbox.y0 + 4 or region.bbox.x1 <= region.bbox.x0 + 4:
            return _CropValidation(False, labels, "empty_crop_box")
        if region.bbox.y0 < config.detection.crop_top_margin - 1:
            return _CropValidation(False, labels, "crop_starts_above_body")
        if _crop_contains_page_header_or_footer(region, layout, config):
            return _CropValidation(False, labels, "page_header_footer_text")
        if _crop_cuts_through_text_line(region, layout):
            return _CropValidation(False, labels, "cuts_through_text_line")
        if len(regions) == 1 and len(labels) <= 1 and region.bbox.y1 - region.bbox.y0 > layout.height * 0.78:
            return _CropValidation(False, labels, "suspicious_full_page_crop")
        if len(regions) == 1 and region.bbox.y1 - region.bbox.y0 > layout.height * 0.92:
            return _CropValidation(False, labels, "suspicious_full_page_crop")

    return _CropValidation(True, labels, "")


def _left_column_labels_in_regions(
    regions: list[MarkSchemeCropRegion],
    words_by_page: dict[int, list[MarkSchemeWord]],
) -> list[str]:
    labels: list[str] = []
    for region in regions:
        left_zone_right = min(_left_question_column_right(region.bbox), region.bbox.x0 + 38.0)
        words = [
            word
            for word in words_by_page.get(region.page_number, [])
            if word.bbox.y0 >= region.bbox.y0 - 2
            and word.bbox.y1 <= region.bbox.y1 + 2
            and word.bbox.x0 >= region.bbox.x0 - 2
            and word.bbox.x0 <= left_zone_right
        ]
        for row in _group_words_by_row(words, tolerance=5.0):
            label = _row_question_label_from_words(row)
            if label and label not in labels:
                labels.append(label)
    return labels


def _left_column_anchor_labels_in_regions(
    regions: list[MarkSchemeCropRegion],
    anchors: list[MarkSchemeAnchor],
) -> list[str]:
    labels: list[str] = []
    for region in regions:
        left_zone_right = _left_question_column_right(region.bbox)
        for anchor in anchors:
            if anchor.page_number != region.page_number:
                continue
            if anchor.y0 < region.bbox.y0 - 2 or anchor.y0 >= region.bbox.y1 - 2:
                continue
            if anchor.x0 < region.bbox.x0 - 12 or anchor.x0 > left_zone_right:
                continue
            label = anchor.question_number
            if label and label not in labels:
                labels.append(label)
    return labels


def _left_question_column_right(box: BoundingBox) -> float:
    width = max(1.0, box.x1 - box.x0)
    return min(box.x1, box.x0 + max(72.0, width * 0.18))


def _crop_contains_page_header_or_footer(
    region: MarkSchemeCropRegion,
    layout: PageLayout,
    config: AppConfig,
) -> bool:
    for block in layout.blocks:
        if not _boxes_overlap(region.bbox, block.bbox):
            continue
        if _is_footer_or_header_box(block.bbox, layout, config):
            return True
        if _is_mark_scheme_boilerplate(block.text):
            return True
        if _is_mark_scheme_page_header_text(block.text):
            return True
    return False


def _is_mark_scheme_page_header_text(text: str) -> bool:
    cleaned = " ".join(text.replace("\u00a0", " ").split())
    if not cleaned:
        return False
    header_terms = [
        r"\bPage\s+\d+\b",
        r"\bMark Scheme\b",
        r"\bSyllabus\b",
        r"\bPaper\b",
        r"\bGCE\s+A/?AS\s+LEVEL\b",
        r"\bGCE\s+AS/?A\s+LEVEL\b",
        r"\bMay/June\s+\d{4}\b",
        r"\b9709\b",
    ]
    hits = sum(1 for pattern in header_terms if re.search(pattern, cleaned, re.IGNORECASE))
    return hits >= 2


def _crop_cuts_through_text_line(region: MarkSchemeCropRegion, layout: PageLayout) -> bool:
    tolerance = 1.5
    for block in layout.blocks:
        if block.bbox.x1 < region.bbox.x0 or block.bbox.x0 > region.bbox.x1:
            continue
        top_inside = block.bbox.y0 + tolerance < region.bbox.y0 < block.bbox.y1 - tolerance
        bottom_inside = block.bbox.y0 + tolerance < region.bbox.y1 < block.bbox.y1 - tolerance
        if top_inside or bottom_inside:
            return True
    return False


def _write_mark_scheme_crop_debug_record(
    *,
    question_number: str,
    identity: PaperIdentity,
    mark_scheme_pdf: Path,
    regions: list[MarkSchemeCropRegion],
    crop_method: str,
    validation: _CropValidation,
    config: AppConfig,
) -> str:
    config.output.debug_dir.mkdir(parents=True, exist_ok=True)
    path = config.output.debug_dir / "mark_scheme_crop_debug.jsonl"
    payload = {
        "question_id": identity.question_id or question_number,
        "source_pdf": str(mark_scheme_pdf),
        "page_number": regions[0].page_number if regions else None,
        "page_numbers": [region.page_number for region in regions],
        "crop_box": [_box_payload(region.bbox) for region in regions],
        "crop_method": crop_method,
        "detected_primary_questions_in_left_column": validation.detected_labels,
        "rejected_reason": validation.rejected_reason,
        "validation_passed": validation.passed,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return _display_path(path)


def _boxes_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    return a.x0 < b.x1 and a.x1 > b.x0 and a.y0 < b.y1 and a.y1 > b.y0


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _detect_mark_scheme_starts(
    layouts: list[PageLayout],
    config: AppConfig,
    expected_numbers: list[str] | None,
) -> list[QuestionStart]:
    expected = set(expected_numbers or [])
    starts: list[QuestionStart] = []
    seen: set[str] = set()
    index = 0
    for page in layouts:
        for block in sorted(page.blocks, key=lambda item: (item.bbox.y0, item.bbox.x0)):
            parsed = parse_question_start(block.first_line, config)
            if parsed:
                number, label = parsed
                if expected and number not in expected:
                    index += 1
                    continue
                if number not in seen and block.bbox.x0 <= max(config.detection.question_start_max_x, 180):
                    starts.append(
                        QuestionStart(
                            question_number=number,
                            page_number=page.page_number,
                            y0=block.bbox.y0,
                            x0=block.bbox.x0,
                            label=label,
                            block_index=index,
                        )
                    )
                    seen.add(number)
            index += 1
    return starts


def _render_legacy_mark_scheme_images(
    mark_scheme_pdf: str | Path,
    config: AppConfig,
    expected_numbers: list[str],
    *,
    question_marks: dict[str, int | None],
    question_subparts: dict[str, list[str]],
    identities: dict[str, PaperIdentity],
    no_blocks_failure_reason: str,
    clear_stale: bool = True,
) -> dict[str, MarkSchemeImageResult]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for legacy mark-scheme image export.") from exc
    quiet_mupdf(fitz)

    mark_scheme_pdf = Path(mark_scheme_pdf)
    layouts = extract_pdf_layout(mark_scheme_pdf, config)
    words = _extract_mark_scheme_words(mark_scheme_pdf)
    blocks = _build_legacy_mark_scheme_blocks(
        mark_scheme_pdf,
        layouts,
        config,
        expected_numbers,
        question_marks=question_marks,
        question_subparts=question_subparts,
    )
    if blocks and clear_stale:
        _clear_stale_mark_scheme_images(mark_scheme_pdf, expected_numbers, config, identities)

    ordered_anchors = [block.anchor for block in sorted(blocks.values(), key=lambda item: (item.anchor.page_number, item.anchor.y0))]
    output: dict[str, MarkSchemeImageResult] = {}
    with fitz.open(mark_scheme_pdf) as doc:
        rendered_pages = {}
        for number in expected_numbers:
            canonical_number = normalize_question_id(number)
            identity = identities.get(canonical_number)
            identity_fields = _mark_scheme_identity_fields(identity, config)
            question_subpart_values = question_subparts.get(canonical_number, [])
            question_marks_total = question_marks.get(canonical_number)
            block = blocks.get(canonical_number)
            if identity is None:
                output[number] = MarkSchemeImageResult(
                    question_number=number,
                    crop_confidence=CropConfidence.LOW,
                    mapping_method="legacy_question_block",
                    table_detected=False,
                    question_subparts=question_subpart_values,
                    question_marks_total=question_marks_total,
                    mapping_status=MappingStatus.FAIL,
                    failure_reason="identity_unresolved",
                    missing_mark_scheme_reason="identity_unresolved",
                    review_flags=["markscheme_image_missing", "identity_unresolved", "markscheme_asset_not_emitted"],
                )
                continue
            if block is None:
                output[number] = MarkSchemeImageResult(
                    question_number=number,
                    **identity_fields,
                    crop_confidence=CropConfidence.LOW,
                    mapping_method="legacy_question_block",
                    table_detected=False,
                    question_subparts=question_subpart_values,
                    question_marks_total=question_marks_total,
                    mapping_status=MappingStatus.FAIL,
                    failure_reason=no_blocks_failure_reason,
                    missing_mark_scheme_reason="segmentation_failure",
                    review_flags=[
                        "markscheme_image_missing",
                        "markscheme_no_legacy_question_block",
                        "markscheme_segmentation_failure",
                    ],
                )
                continue

            flags = list(block.review_flags)
            output_path, debug_paths = _render_mark_scheme_crops(
                doc,
                fitz,
                mark_scheme_pdf,
                canonical_number,
                identity,
                block.regions,
                flags,
                rendered_pages,
                layouts,
                {},
                ordered_anchors,
                config,
                words_by_page=words,
                crop_method=block.method,
            )
            if output_path is None:
                output[number] = MarkSchemeImageResult(
                    question_number=number,
                    **identity_fields,
                    markscheme_question_number=canonical_number,
                    crop_confidence=CropConfidence.LOW,
                    mapping_method=block.method,
                    table_detected=False,
                    detected_anchor_pages=[block.anchor.page_number],
                    nearby_anchors=_nearby_anchor_labels(ordered_anchors, block.anchor),
                    debug_paths=debug_paths,
                    review_flags=sorted(set(flags + ["markscheme_image_missing"])),
                    question_subparts=question_subpart_values,
                    markscheme_subparts=block.subparts,
                    question_marks_total=question_marks_total,
                    markscheme_marks_total=block.mark_total,
                    mapping_status=MappingStatus.FAIL,
                    failure_reason=no_blocks_failure_reason,
                    block_ids=[block.block_id],
                    confidence_score=block.confidence_score,
                    missing_mark_scheme_reason="segmentation_failure",
                )
                continue

            output[number] = MarkSchemeImageResult(
                question_number=number,
                image_path=output_path,
                **identity_fields,
                page_numbers=[region.page_number for region in block.regions],
                markscheme_question_number=canonical_number,
                crop_confidence=_mark_scheme_crop_confidence(block.regions, layouts, flags),
                mapping_method=block.method,
                table_detected=False,
                detected_anchor_pages=[block.anchor.page_number],
                nearby_anchors=_nearby_anchor_labels(ordered_anchors, block.anchor),
                debug_paths=debug_paths,
                review_flags=sorted(set(flags)),
                table_header_ok=False,
                continuation_rows_included=any(region.continuation_rows_included for region in block.regions),
                question_subparts=question_subpart_values,
                markscheme_subparts=block.subparts,
                question_marks_total=question_marks_total,
                markscheme_marks_total=block.mark_total,
                mapping_status=MappingStatus.PASS,
                block_ids=[block.block_id],
                confidence_score=block.confidence_score,
            )
    return output


def _build_legacy_mark_scheme_blocks(
    mark_scheme_pdf: str | Path,
    layouts: list[PageLayout],
    config: AppConfig,
    expected_numbers: list[str] | None,
    *,
    question_marks: dict[str, int | None],
    question_subparts: dict[str, list[str]],
) -> dict[str, MarkSchemeBlock]:
    expected = [normalize_question_id(number) for number in (expected_numbers or []) if normalize_question_id(number)]
    mark_scheme_pdf = Path(mark_scheme_pdf)
    if mark_scheme_pdf.exists():
        words_by_page = _extract_mark_scheme_words(mark_scheme_pdf)
        row_bands = _legacy_table_grid_row_bands(mark_scheme_pdf, layouts, config, words_by_page)
    else:
        words_by_page = {}
        row_bands = []
    anchors = _detect_legacy_mark_scheme_anchors(layouts, config, expected)
    blocks: dict[str, MarkSchemeBlock] = {}
    for index, anchor in enumerate(anchors):
        canonical_number = parent_question_id(anchor.question_number)
        if expected and canonical_number not in expected:
            continue
        next_anchor = anchors[index + 1] if index + 1 < len(anchors) else None
        start = _question_start_from_mark_scheme_anchor(anchor, index)
        next_start = _question_start_from_mark_scheme_anchor(next_anchor, index + 1) if next_anchor else None
        method = "fallback"
        regions, flags = _legacy_table_grid_regions_for_question(
            row_bands,
            canonical_number,
            layouts,
            config,
            anchors=anchors,
        )
        if regions:
            method = "table_grid"
        else:
            regions, flags = _legacy_left_column_regions_for_question(
                layouts,
                words_by_page,
                canonical_number,
                config,
            )
            if regions:
                method = "ocr_left_column"
            else:
                regions, flags = _mark_scheme_regions_for_start(layouts, start, next_start, config)
        text_blocks = (
            _blocks_for_crop_regions(layouts, regions, config)
            if regions
            else _blocks_for_legacy_anchor_bounds(layouts, anchor, next_anchor, config)
        )
        text = "\n".join(block.text for block in text_blocks if block.text.strip()).strip()
        mark_total = _legacy_mark_total_from_text(text, question_marks.get(canonical_number))
        subparts = _legacy_subparts_from_text(text)
        if question_marks.get(canonical_number) is not None and mark_total is not None and mark_total != question_marks.get(canonical_number):
            flags.append("legacy_mark_total_mismatch_review")
        if question_subparts.get(canonical_number) and subparts and any(
            part not in subparts for part in question_subparts.get(canonical_number, [])
        ):
            flags.append("legacy_subpart_mismatch_review")
        if not regions:
            flags.append("markscheme_image_missing")
        blocks[canonical_number] = MarkSchemeBlock(
            question_number=canonical_number,
            block_id=_legacy_mark_scheme_block_id(mark_scheme_pdf, canonical_number),
            anchor=anchor,
            next_anchor=next_anchor,
            text=text,
            regions=regions,
            mark_total=mark_total,
            subparts=subparts,
            confidence_score=_legacy_confidence_score(
                regions=regions,
                text=text,
                mark_total=mark_total,
                expected_total=question_marks.get(canonical_number),
                subparts=subparts,
                expected_subparts=question_subparts.get(canonical_number, []),
                flags=flags,
            ),
            method=method,
            review_flags=sorted(set(["legacy_markscheme_segmentation", "markscheme_relaxed_anchor_detection", *flags])),
        )
    return blocks


def _legacy_table_grid_row_bands(
    mark_scheme_pdf: Path,
    layouts: list[PageLayout],
    config: AppConfig,
    words_by_page: dict[int, list[MarkSchemeWord]],
) -> list[_LegacyRowBand]:
    grids = _legacy_table_grids_from_drawings(mark_scheme_pdf, layouts, config)
    bands: list[_LegacyRowBand] = []
    for layout in layouts:
        grid = grids.get(layout.page_number)
        if grid is None:
            continue
        for top, bottom in zip(grid.horizontal_rules, grid.horizontal_rules[1:]):
            if bottom <= top + max(18.0, config.detection.min_crop_height * 0.7):
                continue
            band_words = [
                word
                for word in words_by_page.get(layout.page_number, [])
                if word.bbox.y1 >= top - 2
                and word.bbox.y0 <= bottom + 2
                and grid.bbox.x0 - 3 <= word.bbox.x0 <= grid.bbox.x1 + 3
            ]
            if not band_words:
                continue
            label_zone_right = min(grid.question_col_right, grid.bbox.x0 + 38.0)
            label_words = [word for word in band_words if word.bbox.x0 <= label_zone_right]
            label = None
            for row in _group_words_by_row(label_words, tolerance=5.0):
                label = _row_question_label_from_words(row)
                if label:
                    break
            bands.append(
                _LegacyRowBand(
                    page_number=layout.page_number,
                    y0=top,
                    y1=bottom,
                    x0=grid.bbox.x0,
                    x1=grid.bbox.x1,
                    question_col_right=grid.question_col_right,
                    question_label=label,
                )
            )
    return sorted(bands, key=lambda item: (item.page_number, item.y0))


def _legacy_table_grids_from_drawings(
    mark_scheme_pdf: Path,
    layouts: list[PageLayout],
    config: AppConfig,
) -> dict[int, _LegacyTableGrid]:
    try:
        import fitz
    except ImportError:
        return {}
    quiet_mupdf(fitz)

    by_page_layout = {layout.page_number: layout for layout in layouts}
    grids: dict[int, _LegacyTableGrid] = {}
    with fitz.open(mark_scheme_pdf) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            layout = by_page_layout.get(page_number)
            if layout is None:
                continue
            horizontal: list[BoundingBox] = []
            vertical: list[BoundingBox] = []
            for drawing in page.get_drawings():
                rect = drawing.get("rect")
                if rect is None or not rect.is_valid or rect.is_empty:
                    continue
                box = _visual_box_from_rect(page, rect)
                width = box.x1 - box.x0
                height = box.y1 - box.y0
                if _is_candidate_table_horizontal_rule(box, width, height, layout, config):
                    horizontal.append(box)
                if _is_candidate_table_vertical_rule(box, width, height, layout, config):
                    vertical.append(box)
            grid = _legacy_table_grid_from_rule_boxes(layout, horizontal, vertical)
            if grid is not None:
                grids[page_number] = grid
    return grids


def _is_candidate_table_horizontal_rule(
    box: BoundingBox,
    width: float,
    height: float,
    layout: PageLayout,
    config: AppConfig,
) -> bool:
    return (
        height <= 2.5
        and width >= max(100.0, layout.width * 0.18)
        and box.y0 >= max(config.detection.min_question_start_y, config.detection.crop_top_margin + 15)
        and box.y1 <= layout.height - config.detection.bottom_margin
    )


def _is_candidate_table_vertical_rule(
    box: BoundingBox,
    width: float,
    height: float,
    layout: PageLayout,
    config: AppConfig,
) -> bool:
    return (
        width <= 2.5
        and height >= max(35.0, config.detection.min_crop_height)
        and box.y0 >= max(config.detection.min_question_start_y, config.detection.crop_top_margin + 15)
        and box.y0 <= layout.height - config.detection.bottom_margin
    )


def _legacy_table_grid_from_rule_boxes(
    layout: PageLayout,
    horizontal: list[BoundingBox],
    vertical: list[BoundingBox],
) -> _LegacyTableGrid | None:
    if len(horizontal) < 2:
        return None
    vertical_groups = _group_rule_boxes_by_position(vertical, axis="x", tolerance=2.0)
    xs = [
        position
        for position, boxes in vertical_groups
        if len(boxes) >= 2 or sum(box.y1 - box.y0 for box in boxes) >= 80
    ]
    if len(xs) >= 2:
        left = min(xs)
        right = max(xs)
    else:
        left = min(box.x0 for box in horizontal)
        right = max(box.x1 for box in horizontal)
    if right <= left + layout.width * 0.45:
        return None

    horizontal_groups = _group_rule_boxes_by_position(horizontal, axis="y", tolerance=2.0)
    rules: list[float] = []
    for y, boxes in horizontal_groups:
        intervals = [(max(left, box.x0), min(right, box.x1)) for box in boxes if box.x1 > left and box.x0 < right]
        coverage = _interval_coverage(intervals)
        if coverage >= (right - left) * 0.55:
            rules.append(y)
    rules = _dedupe_positions(rules, tolerance=2.0)
    if len(rules) < 2:
        return None

    top = min(rules)
    bottom = max(rules)
    if bottom <= top + 40:
        return None
    bbox = BoundingBox(left, top, right, bottom)
    return _LegacyTableGrid(
        page_number=layout.page_number,
        bbox=bbox,
        question_col_right=_left_question_column_right(bbox),
        horizontal_rules=rules,
        vertical_rules=xs,
    )


def _group_rule_boxes_by_position(
    boxes: list[BoundingBox],
    *,
    axis: str,
    tolerance: float,
) -> list[tuple[float, list[BoundingBox]]]:
    if not boxes:
        return []
    if axis == "x":
        ordered = sorted(boxes, key=lambda box: ((box.x0 + box.x1) / 2, box.y0))
        center = lambda box: (box.x0 + box.x1) / 2
    else:
        ordered = sorted(boxes, key=lambda box: ((box.y0 + box.y1) / 2, box.x0))
        center = lambda box: (box.y0 + box.y1) / 2
    groups: list[list[BoundingBox]] = []
    for box in ordered:
        value = center(box)
        if not groups:
            groups.append([box])
            continue
        group_value = sum(center(item) for item in groups[-1]) / len(groups[-1])
        if abs(value - group_value) <= tolerance:
            groups[-1].append(box)
        else:
            groups.append([box])
    return [(sum(center(item) for item in group) / len(group), group) for group in groups]


def _interval_coverage(intervals: list[tuple[float, float]]) -> float:
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0.0
    merged: list[tuple[float, float]] = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 3:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _legacy_table_grid_regions_for_question(
    row_bands: list[_LegacyRowBand],
    canonical_number: str,
    layouts: list[PageLayout],
    config: AppConfig,
    anchors: list[MarkSchemeAnchor] | None = None,
) -> tuple[list[MarkSchemeCropRegion], list[str]]:
    del config
    anchors = anchors or []
    start_index = next(
        (
            index
            for index, band in enumerate(row_bands)
            if _legacy_row_band_question_label(band, anchors)
            and parent_question_id(_legacy_row_band_question_label(band, anchors) or "") == canonical_number
        ),
        None,
    )
    if start_index is None:
        return [], ["legacy_table_grid_target_missing"]

    selected: list[_LegacyRowBand] = []
    for band in row_bands[start_index:]:
        label = _legacy_row_band_question_label(band, anchors)
        if label and parent_question_id(label) != canonical_number:
            break
        selected.append(band)
    if not selected:
        return [], ["legacy_table_grid_target_missing"]

    regions: list[MarkSchemeCropRegion] = []
    for page_number in _ordered_unique([str(band.page_number) for band in selected]):
        page_bands = [band for band in selected if str(band.page_number) == page_number]
        if not page_bands:
            continue
        layout = _layout_by_number(layouts, int(page_number))
        box = BoundingBox(
            min(band.x0 for band in page_bands),
            min(band.y0 for band in page_bands),
            max(band.x1 for band in page_bands),
            max(band.y1 for band in page_bands),
        )
        box = BoundingBox(
            max(0, box.x0),
            max(0, box.y0),
            min(layout.width, box.x1),
            min(layout.height, box.y1),
        )
        if box.y1 <= box.y0 + 4:
            continue
        continuation_rows_included = any(not _legacy_row_band_question_label(band, anchors) for band in page_bands[1:])
        regions.append(MarkSchemeCropRegion(int(page_number), box, table_detected=True, continuation_rows_included=continuation_rows_included))

    flags = ["legacy_markscheme_table_grid_crop"]
    if len(regions) > 1:
        flags.append("markscheme_image_stitched")
    return regions, flags


def _legacy_row_band_question_label(band: _LegacyRowBand, anchors: list[MarkSchemeAnchor]) -> str | None:
    anchor = _legacy_anchor_for_row_band(band, anchors)
    if anchor is not None:
        return anchor.question_number
    return band.question_label


def _legacy_anchor_for_row_band(band: _LegacyRowBand, anchors: list[MarkSchemeAnchor]) -> MarkSchemeAnchor | None:
    candidates = [
        anchor
        for anchor in anchors
        if anchor.page_number == band.page_number
        and band.y0 - 2 <= anchor.y0 < band.y1 - 2
        and anchor.x0 <= band.question_col_right + 12
    ]
    return min(candidates, key=lambda item: (item.y0, item.x0), default=None)


def _legacy_left_column_regions_for_question(
    layouts: list[PageLayout],
    words_by_page: dict[int, list[MarkSchemeWord]],
    canonical_number: str,
    config: AppConfig,
) -> tuple[list[MarkSchemeCropRegion], list[str]]:
    rows: list[tuple[int, float, float, BoundingBox, str | None]] = []
    for layout in layouts:
        body_words = [
            word
            for word in words_by_page.get(layout.page_number, [])
            if config.detection.crop_top_margin <= word.bbox.y0 <= layout.height - config.detection.bottom_margin
            and not _is_mark_scheme_boilerplate(word.text)
        ]
        if not body_words:
            continue
        body_box = _union_boxes([word.bbox for word in body_words])
        left_zone_right = min(body_box.x1, body_box.x0 + 38.0)
        for row in _group_words_by_row(body_words, tolerance=5.0):
            row_box = _union_boxes([word.bbox for word in row])
            if row_box.x1 < body_box.x0 or row_box.x0 > body_box.x1:
                continue
            label_words = [word for word in row if word.bbox.x0 <= left_zone_right]
            label = _row_question_label_from_words(label_words) if label_words else None
            rows.append((layout.page_number, row_box.y0, row_box.y1, body_box, label))
    rows.sort(key=lambda item: (item[0], item[1]))
    start_index = next(
        (
            index
            for index, (_page, _y0, _y1, _box, label) in enumerate(rows)
            if label and parent_question_id(label) == canonical_number
        ),
        None,
    )
    if start_index is None:
        return [], ["legacy_left_column_target_missing"]

    selected: list[tuple[int, float, float, BoundingBox, str | None]] = []
    for row in rows[start_index:]:
        label = row[4]
        if label and parent_question_id(label) != canonical_number:
            break
        selected.append(row)
    if not selected:
        return [], ["legacy_left_column_target_missing"]

    regions: list[MarkSchemeCropRegion] = []
    for page_number in _ordered_unique([str(row[0]) for row in selected]):
        page_rows = [row for row in selected if str(row[0]) == page_number]
        layout = _layout_by_number(layouts, int(page_number))
        body_box = _union_boxes([row[3] for row in page_rows])
        top = max(config.detection.crop_top_margin, min(row[1] for row in page_rows) - 4)
        bottom = min(layout.height - config.detection.bottom_margin, max(row[2] for row in page_rows) + 4)
        box = BoundingBox(
            max(config.detection.crop_left_margin, body_box.x0 - 4),
            top,
            min(layout.width - config.detection.crop_right_margin, body_box.x1 + 4),
            bottom,
        )
        if box.y1 > box.y0 + 4:
            regions.append(MarkSchemeCropRegion(int(page_number), box, table_detected=False))
    flags = ["legacy_markscheme_left_column_crop"]
    if len(regions) > 1:
        flags.append("markscheme_image_stitched")
    return regions, flags


def _blocks_for_crop_regions(
    layouts: list[PageLayout],
    regions: list[MarkSchemeCropRegion],
    config: AppConfig,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for region in regions:
        layout = _layout_by_number(layouts, region.page_number)
        for block in layout.blocks:
            if not _boxes_overlap(region.bbox, block.bbox):
                continue
            if _is_footer_or_header_box(block.bbox, layout, config) or _is_mark_scheme_boilerplate(block.text):
                continue
            blocks.append(block)
    return sorted(blocks, key=lambda block: (block.page_number, block.bbox.y0, block.bbox.x0))


def _layout_by_number(layouts: list[PageLayout], page_number: int) -> PageLayout:
    for layout in layouts:
        if layout.page_number == page_number:
            return layout
    raise ValueError(f"Page {page_number} not present in extracted layout.")


def _question_start_from_mark_scheme_anchor(anchor: MarkSchemeAnchor | None, index: int) -> QuestionStart | None:
    if anchor is None:
        return None
    return QuestionStart(
        question_number=parent_question_id(anchor.question_number),
        page_number=anchor.page_number,
        y0=anchor.y0,
        x0=anchor.x0,
        label=anchor.text,
        block_index=index,
        bbox=None,
        confidence=0.75,
        reasons=["legacy_markscheme_anchor"],
    )


def _detect_legacy_mark_scheme_anchors(
    layouts: list[PageLayout],
    config: AppConfig,
    expected_numbers: list[str],
) -> list[MarkSchemeAnchor]:
    expected = {normalize_question_id(number) for number in expected_numbers if normalize_question_id(number)}
    candidates: list[MarkSchemeAnchor] = []
    seen: set[tuple[str, int, int]] = set()
    for layout in layouts:
        if _is_preliminary_mark_scheme_notes_page(layout):
            continue
        for block in sorted(layout.blocks, key=lambda item: (item.bbox.y0, item.bbox.x0)):
            number = _legacy_question_anchor_number(block, layout, config, expected)
            if not number:
                continue
            key = (number, layout.page_number, round(block.bbox.y0))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                MarkSchemeAnchor(
                    question_number=number,
                    page_number=layout.page_number,
                    y0=block.bbox.y0,
                    y1=block.bbox.y1,
                    x0=block.bbox.x0,
                    text=block.first_line.strip(),
                    table=None,
                )
            )
    return _filter_legacy_mark_scheme_anchor_sequence(candidates, expected_numbers)


def _is_preliminary_mark_scheme_notes_page(layout: PageLayout) -> bool:
    if not _layout_has_mark_scheme_notes(layout):
        return False
    return _legacy_answer_table_header_y(layout) is None


def _legacy_question_anchor_number(
    block: TextBlock,
    layout: PageLayout,
    config: AppConfig,
    expected: set[str],
) -> str | None:
    if block.bbox.x0 > min(config.detection.question_start_max_x, 100):
        return None
    if _is_footer_or_header_box(block.bbox, layout, config) or _is_mark_scheme_boilerplate(block.text):
        return None
    if _is_preliminary_notes_block_before_answer_header(block, layout):
        return None
    line = " ".join(block.first_line.replace("\u00a0", " ").split())
    if not line:
        return None
    if re.match(r"^\d{1,2}(?:\.\d|/\d)", line):
        return None
    standalone_match = re.fullmatch(r"(?:Q\s*)?(?P<number>\d{1,2})", line, re.IGNORECASE)
    if standalone_match:
        number = normalize_question_id(standalone_match.group("number"))
        return number if _legacy_standalone_question_anchor(block, layout, number, expected) else None
    match = re.match(r"^(?:Q\s*)?(?P<number>\d{1,2})(?P<rest>.*)$", line, re.IGNORECASE)
    if not match:
        return None
    raw_rest = match.group("rest")
    if raw_rest and not raw_rest[0].isspace() and not _legacy_compact_part_label_rest(raw_rest):
        return None
    rest = raw_rest.strip()
    if not rest or rest.startswith((".", "/")):
        return None
    number = normalize_question_id(match.group("number"))
    if block.bbox.x0 > min(80.0, layout.width * 0.15) and _legacy_offset_continuation_rest(rest):
        return None
    if _legacy_numeric_data_row_rest(rest):
        return None
    if re.fullmatch(r"\d{1,2}(?:\s+\d{1,2})?", rest) and not _legacy_terminal_mark_total_anchor(block, layout, number, rest, expected):
        return None
    if expected and number not in expected:
        return None
    return number


def _legacy_compact_part_label_rest(rest: str) -> bool:
    return bool(
        re.match(
            r"^\((?:a|b|c|d|e|f|g|h|viii|vii|vi|iv|ix|iii|ii|i|v|x)\)",
            rest,
            re.IGNORECASE,
        )
    )


def _legacy_numeric_data_row_rest(rest: str) -> bool:
    values = re.findall(r"\d{1,3}", rest)
    if len(values) < 3 and not any(int(value) >= 100 for value in values):
        return False
    return not bool(re.search(r"[A-Za-z=+\-−*/^()πθ]", rest))


def _legacy_offset_continuation_rest(rest: str) -> bool:
    return bool(re.match(r"^[=+\-−*/×÷]", rest))


def _is_preliminary_notes_block_before_answer_header(block: TextBlock, layout: PageLayout) -> bool:
    if not _layout_has_mark_scheme_notes(layout):
        return False
    header_y = _legacy_answer_table_header_y(layout)
    if header_y is None:
        return True
    return block.bbox.y0 < header_y


def _layout_has_mark_scheme_notes(layout: PageLayout) -> bool:
    text = "\n".join(" ".join(block.text.replace("\u00a0", " ").split()) for block in layout.blocks)
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "mark scheme notes",
            "specific marking principles",
            "types of mark",
            "abbreviations",
            "general marking principles",
        )
    )


def _legacy_answer_table_header_y(layout: PageLayout) -> float | None:
    rows: dict[int, list[TextBlock]] = {}
    for block in layout.blocks:
        rows.setdefault(round(block.bbox.y0 / 8), []).append(block)
    for row in sorted(rows.values(), key=lambda items: min(block.bbox.y0 for block in items)):
        terms: set[str] = set()
        for block in row:
            if _is_mark_scheme_header_text(block.text):
                return block.bbox.y0
            terms.update(_clean_cell_text(block.text).split())
        if {"question", "answer", "marks"} <= terms:
            return min(block.bbox.y0 for block in row)
    return None


def _legacy_standalone_question_anchor(
    block: TextBlock,
    layout: PageLayout,
    number: str,
    expected: set[str],
) -> bool:
    if not expected or number not in expected:
        return False
    return block.bbox.x0 <= min(70, layout.width * 0.15)


def _legacy_terminal_mark_total_anchor(
    block: TextBlock,
    layout: PageLayout,
    number: str,
    rest: str,
    expected: set[str],
) -> bool:
    if not expected or number not in expected:
        return False
    if block.bbox.x0 > min(70, layout.width * 0.15):
        return False
    values = [int(value) for value in re.findall(r"\d{1,2}", rest)]
    return len(values) == 1 and 1 <= values[0] <= 20


def _filter_legacy_mark_scheme_anchor_sequence(
    candidates: list[MarkSchemeAnchor],
    expected_numbers: list[str],
) -> list[MarkSchemeAnchor]:
    ordered_candidates = sorted(candidates, key=lambda item: (item.page_number, item.y0, item.x0))
    expected = [normalize_question_id(number) for number in expected_numbers if normalize_question_id(number)]
    if not expected:
        return _filter_out_of_sequence_mark_scheme_anchors(ordered_candidates)
    accepted: list[MarkSchemeAnchor] = []
    last_position: tuple[int, float, float] = (0, -1.0, -1.0)
    for expected_index, expected_number in enumerate(expected):
        matched: MarkSchemeAnchor | None = None
        matched_position: tuple[int, float, float] | None = None
        for candidate in ordered_candidates:
            position = _legacy_anchor_position(candidate)
            if position <= last_position:
                continue
            if parent_question_id(candidate.question_number) != expected_number:
                continue
            matched = candidate
            matched_position = position
            break
        if matched is None or matched_position is None:
            continue
        later_expected = set(expected[expected_index + 1 :])
        if _has_earlier_later_expected_legacy_anchor(
            ordered_candidates,
            after_position=last_position,
            before_position=matched_position,
            later_expected=later_expected,
        ):
            continue
        accepted.append(matched)
        last_position = matched_position
    return accepted


def _legacy_anchor_position(anchor: MarkSchemeAnchor) -> tuple[int, float, float]:
    return (anchor.page_number, anchor.y0, anchor.x0)


def _has_earlier_later_expected_legacy_anchor(
    candidates: list[MarkSchemeAnchor],
    *,
    after_position: tuple[int, float, float],
    before_position: tuple[int, float, float],
    later_expected: set[str],
) -> bool:
    if not later_expected:
        return False
    for candidate in candidates:
        position = _legacy_anchor_position(candidate)
        if not after_position < position < before_position:
            continue
        if parent_question_id(candidate.question_number) in later_expected:
            if not _legacy_anchor_blocks_expected_sequence(candidate):
                continue
            return True
    return False


def _legacy_anchor_blocks_expected_sequence(candidate: MarkSchemeAnchor) -> bool:
    if candidate.x0 <= 60:
        return True
    match = re.match(r"^(?:Q\s*)?\d{1,2}(?P<rest>.*)$", " ".join(candidate.text.replace("\u00a0", " ").split()), re.IGNORECASE)
    if not match:
        return False
    return _legacy_compact_part_label_rest(match.group("rest").lstrip())


def _blocks_for_legacy_anchor_bounds(
    layouts: list[PageLayout],
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
    config: AppConfig,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    end_page = next_anchor.page_number if next_anchor else layouts[-1].page_number
    for layout in layouts:
        if not anchor.page_number <= layout.page_number <= end_page:
            continue
        top = anchor.y0 if layout.page_number == anchor.page_number else config.detection.crop_top_margin
        bottom = next_anchor.y0 if next_anchor and layout.page_number == next_anchor.page_number else layout.height - config.detection.bottom_margin
        boundary_y = _legacy_foreign_mark_scheme_boundary_y(layout, anchor.question_number, top, bottom, config)
        if boundary_y is not None:
            bottom = min(bottom, boundary_y)
        if _next_anchor_page_has_no_legacy_continuation(
            layout.page_number,
            anchor.page_number,
            next_anchor.page_number if next_anchor else None,
            top,
            bottom,
            config,
        ):
            continue
        for block in layout.blocks:
            if block.bbox.y1 < top or block.bbox.y0 >= bottom:
                continue
            if _is_footer_or_header_box(block.bbox, layout, config) or _is_mark_scheme_boilerplate(block.text):
                continue
            blocks.append(block)
    return sorted(blocks, key=lambda block: (block.page_number, block.bbox.y0, block.bbox.x0))


def _legacy_mark_total_from_text(text: str, expected_total: int | None) -> int | None:
    values = [int(value) for value in re.findall(r"\[(\d{1,2})\]", text) if 0 < int(value) <= 20]
    if not values:
        return None
    total = sum(values)
    if expected_total is not None and total == expected_total:
        return expected_total
    if expected_total is not None and expected_total in values and len(values) == 1:
        return expected_total
    return total


def _legacy_subparts_from_text(text: str) -> list[str]:
    labels: list[str] = []
    label_order = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
    for line in text.splitlines():
        match = re.match(
            r"^\s*(?:\d{1,2}\s*)?\((a|b|c|d|e|f|g|h|viii|vii|vi|iv|ix|iii|ii|i|v|x)\)",
            line,
            re.IGNORECASE,
        )
        if match:
            label = match.group(1).lower()
            if label not in labels:
                labels.append(label)
    return sorted(labels, key=lambda label: label_order.index(label) if label in label_order else 999)


def _legacy_confidence_score(
    *,
    regions: list[MarkSchemeCropRegion],
    text: str,
    mark_total: int | None,
    expected_total: int | None,
    subparts: list[str],
    expected_subparts: list[str],
    flags: list[str],
) -> float:
    score = 0.62
    if regions:
        score += 0.12
    if len(text) >= 40:
        score += 0.08
    if mark_total is not None:
        score += 0.06
    if expected_total is not None and mark_total == expected_total:
        score += 0.08
    elif expected_total is not None and mark_total is not None:
        score -= 0.04
    if expected_subparts:
        score += 0.05 if subparts and all(part in subparts for part in expected_subparts) else -0.03
    if any(flag in flags for flag in {"markscheme_image_uncertain", "legacy_mark_total_mismatch_review", "legacy_subpart_mismatch_review"}):
        score -= 0.04
    if len(regions) > 2:
        score -= 0.03
    return round(max(0.45, min(0.92, score)), 3)


def _table_mark_scheme_confidence_score(crop_confidence: str, failure_reason: str) -> float:
    if failure_reason:
        return 0.55 if crop_confidence == CropConfidence.HIGH else 0.45
    if crop_confidence == CropConfidence.HIGH:
        return 0.95
    if crop_confidence == CropConfidence.MEDIUM:
        return 0.8
    return 0.65


def _legacy_mark_scheme_block_id(mark_scheme_pdf: str | Path, question_number: str) -> str:
    qid = f"q{int(question_number):02d}" if str(question_number).isdigit() else f"q{_safe_basename(str(question_number))}"
    return f"{Path(mark_scheme_pdf).stem}:{qid}"


def _extract_mark_scheme_words(mark_scheme_pdf: Path) -> dict[int, list[MarkSchemeWord]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            'PyMuPDF is required for mark-scheme word extraction. Install the project with `pip install -e ".[dev]"`.'
        ) from exc
    quiet_mupdf(fitz)

    words_by_page: dict[int, list[MarkSchemeWord]] = {}
    with fitz.open(mark_scheme_pdf) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            words: list[MarkSchemeWord] = []
            for raw_word in page.get_text("words"):
                x0, y0, x1, y1, text, *_ = raw_word
                if str(text).strip():
                    bbox = _visual_box_from_rect(page, (x0, y0, x1, y1))
                    words.append(
                        MarkSchemeWord(
                            page_number=page_number,
                            text=str(text),
                            bbox=bbox,
                        )
                    )
            words_by_page[page_number] = words
    return words_by_page


def _detect_mark_scheme_tables(
    layouts: list[PageLayout],
    config: AppConfig,
    words_by_page: dict[int, list[MarkSchemeWord]] | None = None,
) -> dict[int, MarkSchemeTable]:
    tables: dict[int, MarkSchemeTable] = {}
    for layout in layouts:
        header_geometry = _mark_scheme_header_geometry(words_by_page.get(layout.page_number, []) if words_by_page else [])
        header_blocks = _mark_scheme_header_blocks(layout) if header_geometry is None else []
        header_detected = header_geometry.header_detected if header_geometry else _header_terms(header_blocks)
        if header_detected != ["Question", "Answer", "Marks", "Guidance"]:
            continue

        header_box = header_geometry.header_box if header_geometry else _union_boxes([block.bbox for block in header_blocks])
        content_blocks = [
            block
            for block in layout.blocks
            if block.bbox.y1 >= header_box.y0 - 6
            and block.bbox.y0 <= layout.height - config.detection.bottom_margin
            and not _is_footer_or_header_box(block.bbox, layout, config)
            and not _is_mark_scheme_boilerplate(block.text)
        ]
        if not content_blocks:
            continue

        content_box = _union_boxes([block.bbox for block in content_blocks])
        graphic_box = _table_graphic_bounds(layout, content_box)
        bbox = _union_boxes([content_box, graphic_box]) if graphic_box else content_box
        bbox = BoundingBox(
            max(0, bbox.x0 - 4),
            max(config.detection.crop_top_margin, header_box.y0 - 4),
            min(layout.width, bbox.x1 + 4),
            min(layout.height - config.detection.bottom_margin, bbox.y1 + 4),
        )
        if header_geometry:
            question_col_right = min(
                (header_geometry.question_header.x1 + header_geometry.answer_header.x0) / 2,
                header_geometry.question_header.x1 + 24,
            )
            marks_col_left = max(
                header_geometry.answer_header.x1 + 20,
                header_geometry.marks_header.x0 - 55,
            )
            marks_col_right = min(
                (header_geometry.marks_header.x1 + header_geometry.guidance_header.x0) / 2,
                header_geometry.marks_header.x1 + 18,
            )
        else:
            question_header = _best_header_block(header_blocks, "question")
            answer_header = _best_header_block(header_blocks, "answer")
            marks_header = _best_header_block(header_blocks, "marks")
            guidance_header = _best_header_block(header_blocks, "guidance")
            if question_header and answer_header and question_header is not answer_header:
                question_col_right = min((question_header.bbox.x1 + answer_header.bbox.x0) / 2, question_header.bbox.x1 + 24)
            else:
                question_col_right = min(layout.width * 0.22, bbox.x0 + 110)
            if answer_header and marks_header:
                marks_col_left = max(answer_header.bbox.x1 + 20, marks_header.bbox.x0 - 55)
            else:
                marks_col_left = bbox.x0 + (bbox.x1 - bbox.x0) * 0.58
            if marks_header and guidance_header:
                marks_col_right = min((marks_header.bbox.x1 + guidance_header.bbox.x0) / 2, marks_header.bbox.x1 + 18)
            else:
                marks_col_right = bbox.x0 + (bbox.x1 - bbox.x0) * 0.72
        tables[layout.page_number] = MarkSchemeTable(
            page_number=layout.page_number,
            bbox=bbox,
            question_col_right=question_col_right,
            marks_col_left=marks_col_left,
            marks_col_right=marks_col_right,
            header_bottom=header_box.y1,
            confidence="high",
            header_detected=header_detected,
        )
    return tables


def _mark_scheme_header_blocks(layout: PageLayout) -> list[TextBlock]:
    header_words = {"question", "answer", "marks", "guidance"}
    blocks: list[TextBlock] = []
    for block in layout.blocks:
        cleaned = _clean_cell_text(block.text)
        if cleaned in header_words or any(word in cleaned.split() for word in header_words):
            blocks.append(block)
    if not blocks:
        return []

    row_groups: dict[int, list[TextBlock]] = {}
    for block in blocks:
        row_key = round(block.bbox.y0 / 8)
        row_groups.setdefault(row_key, []).append(block)
    best = max(row_groups.values(), key=lambda group: len(_header_terms(group)))
    if _header_terms(best) != ["Question", "Answer", "Marks", "Guidance"]:
        return []
    return sorted(best, key=lambda block: block.bbox.x0)


def _mark_scheme_header_geometry(words: list[MarkSchemeWord]) -> HeaderGeometry | None:
    if not words:
        return None
    rows = _group_words_by_row(words, tolerance=4.0)
    for row in rows:
        by_word: dict[str, MarkSchemeWord] = {}
        for word in row:
            cleaned = _clean_cell_text(word.text)
            if cleaned in {"question", "answer", "marks", "guidance"} and cleaned not in by_word:
                by_word[cleaned] = word
        if set(by_word) != {"question", "answer", "marks", "guidance"}:
            continue
        ordered = [by_word["question"], by_word["answer"], by_word["marks"], by_word["guidance"]]
        if ordered != sorted(ordered, key=lambda item: item.bbox.x0):
            continue
        return HeaderGeometry(
            header_box=_union_boxes([word.bbox for word in ordered]),
            question_header=by_word["question"].bbox,
            answer_header=by_word["answer"].bbox,
            marks_header=by_word["marks"].bbox,
            guidance_header=by_word["guidance"].bbox,
            header_detected=["Question", "Answer", "Marks", "Guidance"],
        )
    return None


def _group_words_by_row(words: list[MarkSchemeWord], tolerance: float = 5.0) -> list[list[MarkSchemeWord]]:
    rows: list[list[MarkSchemeWord]] = []
    for word in sorted(words, key=lambda item: (_box_center_y(item.bbox), item.bbox.x0)):
        center = _box_center_y(word.bbox)
        best_index: int | None = None
        best_distance: float | None = None
        for index, row in enumerate(rows):
            row_center = sum(_box_center_y(item.bbox) for item in row) / len(row)
            distance = abs(center - row_center)
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best_index = index
                best_distance = distance
        if best_index is None:
            rows.append([word])
        else:
            rows[best_index].append(word)
    return [sorted(row, key=lambda item: item.bbox.x0) for row in rows]


def _box_center_x(box: BoundingBox) -> float:
    return (box.x0 + box.x1) / 2


def _box_center_y(box: BoundingBox) -> float:
    return (box.y0 + box.y1) / 2


def _best_header_block(blocks: list[TextBlock], word: str) -> TextBlock | None:
    return next((block for block in blocks if word in _clean_cell_text(block.text).split()), None)


def _header_terms(blocks: list[TextBlock]) -> list[str]:
    canonical = {
        "question": "Question",
        "answer": "Answer",
        "marks": "Marks",
        "guidance": "Guidance",
    }
    found: set[str] = set()
    for block in blocks:
        words = set(_clean_cell_text(block.text).split())
        found.update(label for word, label in canonical.items() if word in words)
    ordered = ["Question", "Answer", "Marks", "Guidance"]
    return [label for label in ordered if label in found]


def _table_graphic_bounds(layout: PageLayout, content_box: BoundingBox) -> BoundingBox | None:
    graphics = [
        box
        for box in layout.graphics
        if box.y1 >= content_box.y0 - 20
        and box.y0 <= content_box.y1 + 20
        and box.x1 >= content_box.x0 - 40
        and box.x0 <= content_box.x1 + 40
    ]
    if not graphics:
        return None
    return _union_boxes(graphics)


def _detect_table_question_anchors(
    layouts: list[PageLayout],
    tables: dict[int, MarkSchemeTable],
    config: AppConfig,
    expected_numbers: list[str] | None,
    words_by_page: dict[int, list[MarkSchemeWord]] | None = None,
) -> list[MarkSchemeAnchor]:
    expected = {normalize_question_id(number) for number in (expected_numbers or [])}
    expected_parents = {parent_question_id(number) for number in expected}
    anchors: list[MarkSchemeAnchor] = []
    seen: set[tuple[str, int, int]] = set()
    for layout in layouts:
        table = tables.get(layout.page_number)
        if not table:
            continue
        word_anchors = _detect_table_question_anchors_from_words(
            words_by_page.get(layout.page_number, []) if words_by_page else [],
            table,
            expected,
            expected_parents,
        )
        if word_anchors:
            for anchor in word_anchors:
                key = (anchor.question_number, anchor.page_number, round(anchor.y0))
                if key not in seen:
                    seen.add(key)
                    anchors.append(anchor)
            continue
        q_col_right = table.question_col_right
        table_top = table.header_bottom
        table_bottom = table.bbox.y1
        for block in sorted(layout.blocks, key=lambda item: (item.bbox.y0, item.bbox.x0)):
            if block.bbox.y1 < table_top or block.bbox.y0 > table_bottom:
                continue
            if block.bbox.x0 > q_col_right:
                continue
            if not _is_plausible_question_anchor_bbox(block.bbox, table):
                continue
            number = _parse_mark_scheme_question_cell(block.text, expected, expected_parents)
            if not number:
                continue
            key = (number, layout.page_number, round(block.bbox.y0))
            if key in seen:
                continue
            seen.add(key)
            anchors.append(
                MarkSchemeAnchor(
                    question_number=number,
                    page_number=layout.page_number,
                    y0=block.bbox.y0,
                    y1=block.bbox.y1,
                    x0=block.bbox.x0,
                    text=block.text.strip(),
                    table=table,
                )
            )
    ordered = sorted(anchors, key=lambda item: (item.page_number, item.y0, item.x0))
    return _filter_out_of_sequence_mark_scheme_anchors(ordered)


def _filter_out_of_sequence_mark_scheme_anchors(anchors: list[MarkSchemeAnchor]) -> list[MarkSchemeAnchor]:
    accepted: list[MarkSchemeAnchor] = []
    highest_parent = 0
    for index, anchor in enumerate(anchors):
        parent = _anchor_parent_number(anchor)
        if parent is None:
            continue
        if not accepted:
            accepted.append(anchor)
            highest_parent = parent
            continue
        if parent < highest_parent:
            continue
        if parent > highest_parent + 1 and _future_parent_exists(anchors, index, highest_parent + 1):
            continue
        accepted.append(anchor)
        highest_parent = max(highest_parent, parent)
    return accepted


def _anchor_parent_number(anchor: MarkSchemeAnchor) -> int | None:
    match = re.match(r"\d{1,2}", parent_question_id(anchor.question_number))
    return int(match.group(0)) if match else None


def _future_parent_exists(anchors: list[MarkSchemeAnchor], after_index: int, parent_number: int) -> bool:
    return any(_anchor_parent_number(anchor) == parent_number for anchor in anchors[after_index + 1 :])


def _detect_table_question_anchors_from_words(
    words: list[MarkSchemeWord],
    table: MarkSchemeTable,
    expected: set[str],
    expected_parents: set[str],
) -> list[MarkSchemeAnchor]:
    anchors: list[MarkSchemeAnchor] = []
    if not words:
        return anchors
    rows = _group_words_by_row(
        [
            word
            for word in words
            if word.bbox.y1 >= table.header_bottom
            and word.bbox.y0 <= table.bbox.y1
            and table.bbox.x0 - 4 <= word.bbox.x0 <= table.bbox.x1 + 4
        ],
        tolerance=5.0,
    )
    for row in rows:
        label_zone_right = _question_label_zone_right(table)
        question_words = [
            word
            for word in row
            if _box_center_x(word.bbox) <= table.question_col_right
            and word.bbox.x0 <= label_zone_right
        ]
        if not question_words:
            continue
        bbox = _union_boxes([word.bbox for word in question_words])
        if not _is_plausible_question_anchor_bbox(bbox, table):
            continue
        text = _leading_question_label_text(question_words)
        number = _parse_mark_scheme_question_cell(text, expected, expected_parents)
        if not number:
            continue
        anchors.append(
            MarkSchemeAnchor(
                question_number=number,
                page_number=table.page_number,
                y0=bbox.y0,
                y1=bbox.y1,
                x0=bbox.x0,
                text=text,
                table=table,
            )
        )
    return anchors


def _is_plausible_question_anchor_bbox(bbox: BoundingBox, table: MarkSchemeTable) -> bool:
    question_column_width = max(20.0, table.question_col_right - table.bbox.x0)
    max_anchor_x0 = table.bbox.x0 + min(42.0, question_column_width * 0.6)
    if table.bbox.x0 < 20:
        max_anchor_x0 = max(max_anchor_x0, _question_label_zone_right(table))
    return bbox.x0 <= max_anchor_x0


def _question_label_zone_right(table: MarkSchemeTable) -> float:
    zone_right = table.bbox.x0 + 70
    if table.bbox.x0 < 20:
        zone_right = max(zone_right, table.question_col_right - 20)
    return min(table.question_col_right, zone_right)


def _leading_question_label_text(words: list[MarkSchemeWord]) -> str:
    if not words:
        return ""
    ordered = sorted(words, key=lambda item: item.bbox.x0)
    candidates = [" ".join(word.text for word in ordered[:count]) for count in range(1, min(3, len(ordered)) + 1)]
    for candidate in candidates:
        if _is_plausible_mark_scheme_label(candidate):
            return _clean_question_label_candidate(candidate)
    return _clean_question_label_candidate(ordered[0].text)


def _clean_question_label_candidate(text: str) -> str:
    cleaned = _clean_cell_text(text)
    cleaned = cleaned.replace("^{", "").replace("_{", "").replace("}", "")
    cleaned = re.sub(r"[^0-9a-z()]+", "", cleaned)
    cleaned = re.sub(r"^q(?=\d)", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _parse_mark_scheme_question_cell(text: str, expected: set[str], expected_parents: set[str] | None = None) -> str | None:
    cleaned = _clean_cell_text(text)
    if not cleaned:
        return None
    if re.search(r"9709|page|mark|answer|guidance|scheme|paper", cleaned, re.IGNORECASE):
        return None
    if not _is_plausible_mark_scheme_label(cleaned):
        return None
    number = normalize_question_id(_clean_question_label_candidate(cleaned))
    if expected and number not in expected and parent_question_id(number) not in expected and number not in (expected_parents or set()):
        return None
    return number


def _is_plausible_mark_scheme_label(text: str) -> bool:
    cleaned = _clean_cell_text(text)
    if not cleaned or cleaned.startswith("("):
        return False
    if re.fullmatch(r"\d{1,2}[a-h]", cleaned, re.IGNORECASE):
        return False
    label = _clean_question_label_candidate(cleaned)
    return bool(re.fullmatch(r"\d{1,2}(?:[a-h]|\([a-h]\))?(?:\((?:i{1,3}|iv|v|vi{0,3}|ix|x)\))?", label, re.IGNORECASE))


def _anchors_to_question_starts(anchors: list[MarkSchemeAnchor]) -> list[QuestionStart]:
    return [
        QuestionStart(
            question_number=anchor.question_number,
            page_number=anchor.page_number,
            y0=anchor.y0,
            x0=anchor.x0,
            label=anchor.text,
            block_index=index,
        )
        for index, anchor in enumerate(anchors)
    ]


def _anchor_for_question(anchors: list[MarkSchemeAnchor], canonical_number: str) -> tuple[int | None, MarkSchemeAnchor | None]:
    if canonical_number == parent_question_id(canonical_number):
        for index, anchor in enumerate(anchors):
            if parent_question_id(anchor.question_number) == canonical_number and anchor.question_number != canonical_number:
                return index, anchor
    for index, anchor in enumerate(anchors):
        if anchor.question_number == canonical_number:
            return index, anchor
        if (
            parent_question_id(anchor.question_number) == canonical_number
            and canonical_number == parent_question_id(canonical_number)
        ):
            return index, anchor
    return None, None


def _next_boundary_anchor(
    anchors: list[MarkSchemeAnchor],
    anchor_index: int,
    canonical_number: str,
) -> MarkSchemeAnchor | None:
    if canonical_number == parent_question_id(canonical_number):
        for index, item in enumerate(anchors[anchor_index + 1 :], start=anchor_index + 1):
            boundary_parent = parent_question_id(item.question_number)
            if boundary_parent == canonical_number:
                continue
            if item.question_number == boundary_parent and any(
                parent_question_id(later.question_number) == boundary_parent and later.question_number != boundary_parent
                for later in anchors[index + 1 :]
            ):
                continue
            return item
        return None
    return anchors[anchor_index + 1] if anchor_index + 1 < len(anchors) else None


def _table_regions_for_anchor(
    layouts: list[PageLayout],
    tables: dict[int, MarkSchemeTable],
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
    config: AppConfig,
) -> tuple[list[MarkSchemeCropRegion], list[str]]:
    flags: list[str] = []
    if not anchor.table:
        flags.append("markscheme_table_detection_failed")
        return [], flags

    end_page = next_anchor.page_number if next_anchor else layouts[-1].page_number
    regions: list[MarkSchemeCropRegion] = []
    for layout in layouts:
        if not anchor.page_number <= layout.page_number <= end_page:
            continue
        table = tables.get(layout.page_number)
        if not table:
            flags.append("partial_question_block")
            flags.append("invalid_table_header")
            continue

        if layout.page_number == anchor.page_number:
            top = anchor.y0
        else:
            top = table.header_bottom
        # CAIE mark schemes often show the question number once, then leave the
        # question-number cells blank on continuation rows. Those rows belong to
        # the current question until the next visible question-number anchor.
        bottom = next_anchor.y0 if next_anchor and layout.page_number == next_anchor.page_number else table.bbox.y1
        boundary_y = _legacy_foreign_mark_scheme_boundary_y(layout, anchor.question_number, top, bottom, config)
        if boundary_y is not None:
            bottom = min(bottom, boundary_y)
            flags.append("markscheme_foreign_question_boundary_trimmed")
        if (
            next_anchor
            and layout.page_number == next_anchor.page_number
            and layout.page_number != anchor.page_number
            and bottom <= table.header_bottom + 20
        ):
            continue
        bottom = _tighten_table_bottom_from_content(layout, table, top, bottom, config)
        box = BoundingBox(
            table.bbox.x0,
            max(config.detection.crop_top_margin, top - config.detection.crop_padding),
            table.bbox.x1,
            min(layout.height - config.detection.bottom_margin, bottom + config.detection.crop_padding),
        )
        line_box = _line_based_table_crop(layout, table, box, next_anchor, config)
        if line_box:
            box = line_box
        if next_anchor and next_anchor.page_number == layout.page_number:
            box = BoundingBox(box.x0, box.y0, box.x1, min(box.y1, next_anchor.y0))
        if box.y1 <= box.y0 + 4:
            flags.append("markscheme_image_uncertain")
            continue
        continuation_rows_included = _has_continuation_rows(layout, table, anchor, next_anchor, box)
        regions.append(
            MarkSchemeCropRegion(
                layout.page_number,
                box,
                table_detected=table.confidence != "low",
                continuation_rows_included=continuation_rows_included,
            )
        )

    if len(regions) > 1:
        flags.append("markscheme_image_stitched")
    if anchor.table.confidence != "high":
        flags.append("markscheme_image_uncertain")
    return regions, flags


def _has_continuation_rows(
    layout: PageLayout,
    table: MarkSchemeTable,
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
    crop_box: BoundingBox,
) -> bool:
    bottom = next_anchor.y0 if next_anchor and next_anchor.page_number == layout.page_number else crop_box.y1
    for block in layout.blocks:
        if block.bbox.y0 <= anchor.y1 + 3 or block.bbox.y0 >= bottom:
            continue
        if block.bbox.x0 <= table.question_col_right:
            continue
        if crop_box.x0 - 5 <= block.bbox.x0 <= crop_box.x1 + 5:
            return True
    return False


def _tighten_table_bottom_from_content(
    layout: PageLayout,
    table: MarkSchemeTable,
    top: float,
    proposed_bottom: float,
    config: AppConfig,
) -> float:
    blocks = [
        block
        for block in layout.blocks
        if block.bbox.y1 >= top
        and block.bbox.y0 < proposed_bottom
        and table.bbox.x0 - 5 <= block.bbox.x0 <= table.bbox.x1 + 5
        and not _is_footer_or_header_box(block.bbox, layout, config)
        and not _is_mark_scheme_boilerplate(block.text)
    ]
    graphics = [
        graphic
        for graphic in layout.graphics
        if graphic.y1 >= top
        and graphic.y0 < proposed_bottom
        and table.bbox.x0 - 5 <= graphic.x0 <= table.bbox.x1 + 5
    ]
    boxes = [block.bbox for block in blocks] + graphics
    if not boxes:
        return proposed_bottom
    return min(proposed_bottom, max(box.y1 for box in boxes))


def _line_based_table_crop(
    layout: PageLayout,
    table: MarkSchemeTable,
    fallback_box: BoundingBox,
    next_anchor: MarkSchemeAnchor | None,
    config: AppConfig,
) -> BoundingBox | None:
    horizontal = _table_horizontal_rules(layout, table)
    vertical = _table_vertical_rules(layout, table)
    if len(horizontal) < 2:
        return None

    left = min(vertical) if len(vertical) >= 2 else table.bbox.x0
    right = max(vertical) if len(vertical) >= 2 else table.bbox.x1
    top = min((y for y in horizontal if y <= table.header_bottom + 2), default=table.bbox.y0)

    if next_anchor and next_anchor.page_number == layout.page_number:
        bottom_candidates = [y for y in horizontal if fallback_box.y0 < y < next_anchor.y0 - 1]
        bottom = max(bottom_candidates, default=fallback_box.y1)
        bottom = min(bottom, next_anchor.y0)
    else:
        bottom_candidates = [y for y in horizontal if y >= fallback_box.y1 - config.detection.crop_padding - 2]
        bottom = min(bottom_candidates, default=max(horizontal))

    box = BoundingBox(
        max(config.detection.crop_left_margin, left),
        max(config.detection.crop_top_margin, top),
        min(layout.width - config.detection.crop_right_margin, right),
        min(layout.height - config.detection.bottom_margin, bottom),
    )
    if box.y1 <= box.y0 + 4 or box.x1 <= box.x0 + 4:
        return None
    return box


def _table_horizontal_rules(layout: PageLayout, table: MarkSchemeTable) -> list[float]:
    ys: list[float] = []
    min_width = max(80, (table.bbox.x1 - table.bbox.x0) * 0.45)
    for graphic in layout.graphics:
        width = graphic.x1 - graphic.x0
        height = graphic.y1 - graphic.y0
        if height > 4 or width < min_width:
            continue
        if graphic.x1 < table.bbox.x0 - 8 or graphic.x0 > table.bbox.x1 + 8:
            continue
        if graphic.y0 < table.bbox.y0 - 8 or graphic.y1 > table.bbox.y1 + 8:
            continue
        ys.extend([graphic.y0, graphic.y1])
    return _dedupe_positions(ys)


def _table_vertical_rules(layout: PageLayout, table: MarkSchemeTable) -> list[float]:
    xs: list[float] = []
    min_height = max(60, (table.bbox.y1 - table.bbox.y0) * 0.35)
    for graphic in layout.graphics:
        width = graphic.x1 - graphic.x0
        height = graphic.y1 - graphic.y0
        if width > 4 or height < min_height:
            continue
        if graphic.x0 < table.bbox.x0 - 8 or graphic.x1 > table.bbox.x1 + 8:
            continue
        if graphic.y1 < table.bbox.y0 - 8 or graphic.y0 > table.bbox.y1 + 8:
            continue
        xs.extend([graphic.x0, graphic.x1])
    return _dedupe_positions(xs)


def _dedupe_positions(values: list[float], tolerance: float = 2.0) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    deduped = [ordered[0]]
    for value in ordered[1:]:
        if abs(value - deduped[-1]) > tolerance:
            deduped.append(value)
    return deduped


def _nearby_anchor_labels(anchors: list[MarkSchemeAnchor], anchor: MarkSchemeAnchor) -> list[str]:
    try:
        index = anchors.index(anchor)
    except ValueError:
        return [item.question_number for item in anchors[:8]]
    return [item.question_number for item in anchors[max(0, index - 2) : index + 3]]


def _table_header_ok(table: MarkSchemeTable | None) -> bool:
    return bool(table and table.header_detected == ["Question", "Answer", "Marks", "Guidance"])


def _detected_subparts_for_question(anchors: list[MarkSchemeAnchor], anchor_index: int, parent: str) -> list[str]:
    subparts: list[str] = []
    for item in anchors[anchor_index:]:
        if parent_question_id(item.question_number) != parent:
            break
        match = re.search(r"\((a|b|c|d|e|f|g|h|viii|vii|vi|iv|ix|iii|ii|i|v|x)\)", item.question_number)
        if match and match.group(1) not in subparts:
            subparts.append(match.group(1))
    return subparts


def _mark_total_for_question_block(
    layouts: list[PageLayout],
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
    tables: dict[int, MarkSchemeTable],
    words_by_page: dict[int, list[MarkSchemeWord]] | None = None,
    expected_total: int | None = None,
) -> int | None:
    if words_by_page:
        rows = _table_rows_for_question_block(layouts, anchor, next_anchor, tables, words_by_page)
        if not rows:
            return None
        return _mark_total_from_rows(rows, parent_question_id(anchor.question_number), expected_total)

    total = 0
    found = False
    for layout in layouts:
        if layout.page_number < anchor.page_number:
            continue
        if next_anchor and layout.page_number > next_anchor.page_number:
            continue
        table = tables.get(layout.page_number)
        if not table:
            continue
        top = anchor.y0 if layout.page_number == anchor.page_number else table.header_bottom
        top_tolerance = 4.0 if layout.page_number == anchor.page_number else 0.0
        bottom = next_anchor.y0 if next_anchor and layout.page_number == next_anchor.page_number else table.bbox.y1
        for block in layout.blocks:
            if block.bbox.y1 < top or block.bbox.y0 >= bottom:
                continue
            center_x = (block.bbox.x0 + block.bbox.x1) / 2
            if table.marks_col_left <= center_x <= table.marks_col_right:
                marks = _marks_from_marks_cell(block.text)
                if marks:
                    total += sum(marks)
                    found = True
    return total if found else None


def _is_standalone_total_row(row: list[MarkSchemeWord], table: MarkSchemeTable, marks_cell: str) -> bool:
    if not re.fullmatch(r"\s*\d{1,2}\s*", marks_cell):
        return False
    if _is_terminal_total_guidance_row(" ".join(word.text for word in row), marks_cell):
        return True
    if len(row) > 2:
        return False
    if any(not re.fullmatch(r"\d{1,2}", word.text.strip()) for word in row):
        return False
    return True


def _is_terminal_total_guidance_row(row_text: str, marks_cell: str) -> bool:
    value = marks_cell.strip()
    cleaned = " ".join(row_text.replace("\u00a0", " ").split())
    if not cleaned.startswith(value):
        return False
    remainder = cleaned[len(value) :].strip(" .:-–—")
    if not remainder:
        return False
    return bool(
        re.match(
            r"(?i)^(?:N\.?B\.?|SC\d*|Special\s+case|See\s+diagram|Ignore|Allow|Accept|Condone|Max(?:imum)?|No\s+evidence)\b",
            remainder,
        )
    )


def _standalone_total_row_value(row: list[MarkSchemeWord], table: MarkSchemeTable, marks_cell: str) -> int | None:
    if not _is_standalone_total_row(row, table, marks_cell):
        return None
    try:
        return int(marks_cell.strip())
    except ValueError:
        return None


def _sum_terminal_standalone_mark_rows(entries: list[tuple[int, float, bool, int, str]]) -> int | None:
    standalone_total = 0
    found = False
    ordered = sorted(entries, key=lambda item: (item[0], item[1]))
    for index, (page_number, _y0, is_standalone, value, _text) in enumerate(ordered):
        if not is_standalone:
            continue
        next_entry = ordered[index + 1] if index + 1 < len(ordered) else None
        if next_entry and next_entry[0] == page_number and not next_entry[2] and not _starts_mark_scheme_question_label(next_entry[4]):
            continue
        standalone_total += value
        found = True
    return standalone_total if found else None


def _starts_mark_scheme_question_label(text: str) -> bool:
    return bool(re.match(r"\s*\d{1,2}(?:\([a-h]\)|\((?:viii|vii|vi|iv|ix|iii|ii|i|v|x)\))", text, re.IGNORECASE))


def _table_rows_for_question_block(
    layouts: list[PageLayout],
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
    tables: dict[int, MarkSchemeTable],
    words_by_page: dict[int, list[MarkSchemeWord]],
) -> list[MarkSchemeRow]:
    rows_out: list[MarkSchemeRow] = []
    for layout in layouts:
        if layout.page_number < anchor.page_number:
            continue
        if next_anchor and layout.page_number > next_anchor.page_number:
            continue
        table = tables.get(layout.page_number)
        if not table:
            continue
        top = anchor.y0 if layout.page_number == anchor.page_number else table.header_bottom
        top_tolerance = 4.0 if layout.page_number == anchor.page_number else 0.0
        bottom = next_anchor.y0 if next_anchor and layout.page_number == next_anchor.page_number else table.bbox.y1
        if (
            next_anchor
            and layout.page_number == next_anchor.page_number
            and layout.page_number != anchor.page_number
            and bottom <= table.header_bottom + 20
        ):
            continue
        rows = _group_words_by_row(
            [
                word
                for word in words_by_page.get(layout.page_number, [])
                if word.bbox.y0 >= top - top_tolerance
                and word.bbox.y1 <= bottom
                and table.bbox.x0 - 4 <= word.bbox.x0 <= table.bbox.x1 + 4
            ],
            tolerance=5.0,
        )
        for row in rows:
            full_text = " ".join(word.text for word in row)
            if _is_mark_scheme_boilerplate(full_text):
                continue
            if _is_mark_scheme_header_text(full_text):
                continue
            question_words = [
                word
                for word in row
                if _box_center_x(word.bbox) <= table.question_col_right
                and word.bbox.x0 <= _question_label_zone_right(table)
            ]
            marks_cell = " ".join(
                word.text
                for word in row
                if table.marks_col_left <= _box_center_x(word.bbox) <= table.marks_col_right
            )
            rows_out.append(
                MarkSchemeRow(
                    page_number=layout.page_number,
                    y0=min(word.bbox.y0 for word in row),
                    text=full_text,
                    marks_cell=marks_cell,
                    mark_values=tuple(_marks_from_marks_cell(marks_cell)),
                    standalone_total=_standalone_total_row_value(row, table, marks_cell),
                    question_label=_row_question_label_from_words(question_words),
                )
            )
    return rows_out


def _mark_total_from_rows(rows: list[MarkSchemeRow], parent_number: str, expected_total: int | None = None) -> int | None:
    subpart_groups = _group_rows_by_subpart(rows, parent_number)
    if subpart_groups:
        candidate_groups = [_subpart_total_candidates(group_rows) for _label, group_rows in subpart_groups]
        return _select_total_from_candidates(candidate_groups, expected_total)
    candidate_groups = [_subpart_total_candidates(rows)]
    return _select_total_from_candidates(candidate_groups, expected_total)


def _group_rows_by_subpart(rows: list[MarkSchemeRow], parent_number: str) -> list[tuple[str, list[MarkSchemeRow]]]:
    groups: list[tuple[str, list[MarkSchemeRow]]] = []
    current_index: int | None = None
    for row in rows:
        label = row.question_label
        if label and parent_question_id(label) == parent_number and label != parent_number:
            current_index = next((index for index, (name, _rows) in enumerate(groups) if name == label), None)
            if current_index is None:
                groups.append((label, []))
                current_index = len(groups) - 1
        if current_index is not None:
            groups[current_index][1].append(row)
    return [(label, subpart_rows) for label, subpart_rows in groups if subpart_rows]


def _subpart_total_candidates(rows: list[MarkSchemeRow]) -> list[int]:
    branch_totals = [0]
    explicit_totals: list[tuple[int, int]] = []
    current_branch = 0
    for index, row in enumerate(rows):
        if row.standalone_total is not None:
            explicit_totals.append((index, row.standalone_total))
            continue
        if _is_alternative_method_row(row.text):
            if branch_totals[current_branch] > 0 or explicit_totals:
                branch_totals.append(0)
                current_branch = len(branch_totals) - 1
            continue
        if row.mark_values:
            branch_totals[current_branch] += sum(row.mark_values)
    terminal_totals = [
        value
        for index, value in explicit_totals
        if _standalone_total_is_terminal_or_before_alternative(rows, index)
    ]
    candidates = sorted(set(terminal_totals + [total for total in branch_totals if total > 0]))
    if not candidates:
        return []
    return sorted(set(candidates))


def _standalone_total_is_terminal_or_before_alternative(rows: list[MarkSchemeRow], index: int) -> bool:
    for later in rows[index + 1 :]:
        if _is_alternative_method_row(later.text):
            return True
        if later.mark_values:
            return False
    return True


def _select_total_from_candidates(candidate_groups: list[list[int]], expected_total: int | None) -> int | None:
    if not candidate_groups:
        return None
    if any(not candidates for candidates in candidate_groups):
        return None
    sums = [0]
    for candidates in candidate_groups:
        next_sums: set[int] = set()
        for subtotal in sums:
            for candidate in candidates:
                next_sums.add(subtotal + candidate)
        sums = sorted(next_sums)
    if not sums:
        return None
    if expected_total is not None and expected_total in sums:
        return expected_total
    return max(sums)


def _row_question_label_from_words(words: list[MarkSchemeWord]) -> str | None:
    if not words:
        return None
    text = _leading_question_label_text(words)
    if not _is_plausible_mark_scheme_label(text):
        return None
    return normalize_question_id(_clean_question_label_candidate(text))


def _is_alternative_method_row(text: str) -> bool:
    cleaned = " ".join(text.replace("\u00a0", " ").split())
    cleaned = re.sub(
        r"^\s*\d{1,2}(?:\([a-h]\)|\((?:viii|vii|vi|iv|ix|iii|ii|i|v|x)\))?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return bool(
        re.match(
            r"^(?:alternative(?:\s+method(?:\s+for\s+question\b.*)?)?|method\s*[23]\b|special\s+case\b|SC\d*:?\b|either\b)",
            cleaned,
            re.IGNORECASE,
        )
    )


def _marks_from_marks_cell(text: str) -> list[int]:
    cleaned = " ".join(text.replace("\u00a0", " ").split())
    if re.search(r"marks|guidance|answer|question", cleaned, re.IGNORECASE):
        return []
    compact = re.sub(r"\s+", "", cleaned)
    rubric_match = re.fullmatch(r"[MBACE]?\d{1,2}(?:,\d{1,2})+(?:,0)?", compact, re.IGNORECASE)
    if rubric_match:
        values = [int(value) for value in re.findall(r"\d{1,2}", compact) if 0 < int(value) <= 15]
        return [max(values)] if values else []
    coded_values = [
        int(match.group(1))
        for match in re.finditer(r"(?<![A-Za-z])(?:[MBACE]|D?M|D?B|D?A|FT)\s*(\d{1,2})(?!\d)", cleaned)
        if 0 < int(match.group(1)) <= 15
    ]
    if coded_values:
        return coded_values
    return []


def _validate_mark_scheme_mapping(
    canonical_number: str,
    question_subparts: list[str],
    markscheme_subparts: list[str],
    question_marks_total: int | None,
    markscheme_marks_total: int | None,
    anchor: MarkSchemeAnchor | None,
    next_anchor: MarkSchemeAnchor | None,
    regions: list[MarkSchemeCropRegion],
    flags: list[str],
    question_validation_flags: list[str] | None = None,
) -> tuple[list[str], str]:
    validation_flags: list[str] = []
    question_validation_flags = question_validation_flags or []
    if not anchor or not _table_header_ok(anchor.table):
        validation_flags.append("invalid_table_header")
        return validation_flags, "invalid_table_header"
    if "invalid_table_header" in flags:
        validation_flags.append("invalid_table_header")
        return validation_flags, "invalid_table_header"
    if not regions:
        validation_flags.append("partial_question_block")
        return validation_flags, "partial_question_block"
    if next_anchor and parent_question_id(next_anchor.question_number) == canonical_number:
        validation_flags.append("partial_question_block")
        return validation_flags, "partial_question_block"
    candidate_reasons = _mapping_failure_candidates(
        canonical_number=canonical_number,
        question_subparts=question_subparts,
        markscheme_subparts=markscheme_subparts,
        question_marks_total=question_marks_total,
        markscheme_marks_total=markscheme_marks_total,
        anchor=anchor,
        next_anchor=next_anchor,
        regions=regions,
        question_validation_flags=question_validation_flags,
    )
    failure_reason = _select_mapping_failure_reason(candidate_reasons)
    if failure_reason:
        validation_flags.append(failure_reason)
        return validation_flags, failure_reason
    return validation_flags, ""


def _mapping_failure_candidates(
    *,
    canonical_number: str,
    question_subparts: list[str],
    markscheme_subparts: list[str],
    question_marks_total: int | None,
    markscheme_marks_total: int | None,
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
    regions: list[MarkSchemeCropRegion],
    question_validation_flags: list[str],
) -> list[str]:
    candidates: list[str] = []
    if "question_scope_contaminated" in question_validation_flags:
        candidates.append("question_scope_contaminated")
    if any(part not in markscheme_subparts for part in question_subparts):
        candidates.append("mark_scheme_part_structure_mismatch")
    if any(part not in question_subparts for part in markscheme_subparts):
        candidates.append("question_subparts_incomplete")
    if "missing_terminal_mark_total" in question_validation_flags and (markscheme_subparts or question_subparts):
        candidates.append("missing_terminal_mark_total")
    if question_marks_total is None and markscheme_marks_total is not None:
        candidates.append("question_mark_total_missing")
    if question_marks_total is not None and markscheme_marks_total is not None and question_marks_total != markscheme_marks_total:
        candidates.append("question_mark_total_mismatch")
    if "likely_truncated_question_crop" in question_validation_flags:
        candidates.append("likely_truncated_question_crop")
    if "weak_question_anchor" in question_validation_flags:
        candidates.append("weak_question_anchor")
    if _block_contains_adjacent_question(canonical_number, regions, anchor, next_anchor):
        candidates.append("adjacent_question_block_selected")
    return candidates


def _select_mapping_failure_reason(candidates: list[str]) -> str:
    priority = [
        "question_scope_contaminated",
        "mark_scheme_part_structure_mismatch",
        "question_subparts_incomplete",
        "missing_terminal_mark_total",
        "question_mark_total_missing",
        "question_mark_total_mismatch",
        "likely_truncated_question_crop",
        "weak_question_anchor",
        "adjacent_question_block_selected",
    ]
    for reason in priority:
        if reason in candidates:
            return reason
    return ""


def _block_contains_adjacent_question(
    canonical_number: str,
    regions: list[MarkSchemeCropRegion],
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
) -> bool:
    if next_anchor is None:
        return False
    for region in regions:
        if region.page_number != next_anchor.page_number:
            continue
        if region.bbox.y1 > next_anchor.y0 + 2 and parent_question_id(next_anchor.question_number) != canonical_number:
            return True
    return False


def _blocks_between(
    layouts: list[PageLayout],
    start_page: int,
    start_y: float,
    end_page: int,
    end_y: float,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for page in layouts:
        if not start_page <= page.page_number <= end_page:
            continue
        top = start_y if page.page_number == start_page else 0
        bottom = end_y if page.page_number == end_page else page.height
        for block in page.blocks:
            if block.bbox.y1 >= top and block.bbox.y0 < bottom:
                blocks.append(block)
    return sorted(blocks, key=lambda block: (block.page_number, block.bbox.y0, block.bbox.x0))


def _blocks_for_table_anchor_bounds(
    layouts: list[PageLayout],
    tables: dict[int, MarkSchemeTable],
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
    config: AppConfig,
) -> list[TextBlock]:
    end_page = next_anchor.page_number if next_anchor else layouts[-1].page_number
    blocks: list[TextBlock] = []
    for layout in layouts:
        if not anchor.page_number <= layout.page_number <= end_page:
            continue
        table = tables.get(layout.page_number)
        if table is None:
            continue
        top = anchor.y0 if layout.page_number == anchor.page_number else table.header_bottom
        top_tolerance = 4.0 if layout.page_number == anchor.page_number else 0.0
        bottom = next_anchor.y0 if next_anchor and layout.page_number == next_anchor.page_number else table.bbox.y1
        boundary_y = _legacy_foreign_mark_scheme_boundary_y(layout, anchor.question_number, top, bottom, config)
        if boundary_y is not None:
            bottom = min(bottom, boundary_y)
        if (
            next_anchor
            and layout.page_number == next_anchor.page_number
            and layout.page_number != anchor.page_number
            and bottom <= table.header_bottom + 20
        ):
            continue
        for block in layout.blocks:
            if block.bbox.y0 < top - top_tolerance or block.bbox.y1 > bottom:
                continue
            if not (table.bbox.x0 - 5 <= block.bbox.x0 <= table.bbox.x1 + 5):
                continue
            if _is_mark_scheme_header_text(block.text):
                continue
            if _is_footer_or_header_box(block.bbox, layout, config) or _is_mark_scheme_boilerplate(block.text):
                continue
            blocks.append(block)
    return sorted(blocks, key=lambda block: (block.page_number, block.bbox.y0, block.bbox.x0))


def _is_mark_scheme_header_text(text: str) -> bool:
    cleaned = " ".join(text.replace("\u00a0", " ").split())
    return bool(re.search(r"\bQuestion\b", cleaned, re.IGNORECASE)) and bool(
        re.search(r"\bAnswer\b", cleaned, re.IGNORECASE)
    ) and bool(re.search(r"\bMarks\b", cleaned, re.IGNORECASE)) and bool(
        re.search(r"\bGuidance\b", cleaned, re.IGNORECASE)
    )


def _mark_scheme_regions_for_start(
    layouts: list[PageLayout],
    start: QuestionStart,
    next_start: QuestionStart | None,
    config: AppConfig,
) -> tuple[list[MarkSchemeCropRegion], list[str]]:
    flags: list[str] = []
    end_page = next_start.page_number if next_start else layouts[-1].page_number
    end_y = next_start.y0 if next_start else layouts[-1].height - config.detection.bottom_margin
    regions: list[MarkSchemeCropRegion] = []

    for layout in layouts:
        if not start.page_number <= layout.page_number <= end_page:
            continue
        top = start.y0 if layout.page_number == start.page_number else config.detection.crop_top_margin
        bottom = end_y if layout.page_number == end_page else layout.height - config.detection.bottom_margin
        top = max(config.detection.crop_top_margin, top)
        bottom = min(layout.height - config.detection.bottom_margin, bottom)
        boundary_y = _legacy_foreign_mark_scheme_boundary_y(layout, start.question_number, top, bottom, config)
        if boundary_y is not None:
            bottom = min(bottom, boundary_y)
            flags.append("markscheme_foreign_question_boundary_trimmed")
        if _next_anchor_page_has_no_legacy_continuation(
            layout.page_number,
            start.page_number,
            next_start.page_number if next_start else None,
            top,
            bottom,
            config,
        ):
            continue
        if bottom <= top:
            continue

        text_blocks = [
            block
            for block in layout.blocks
            if block.bbox.y1 >= top
            and block.bbox.y0 < bottom
            and not _is_footer_or_header_box(block.bbox, layout, config)
            and not _is_mark_scheme_boilerplate(block.text)
        ]
        graphics = [
            graphic
            for graphic in layout.graphics
            if graphic.y1 >= top
            and graphic.y0 < bottom
            and not _is_footer_or_header_box(graphic, layout, config)
        ]
        boxes = [block.bbox for block in text_blocks] + graphics
        if not boxes:
            continue

        crop_box = _union_boxes(boxes).padded(config.detection.crop_padding, layout.width, layout.height)
        crop_box = BoundingBox(
            max(config.detection.crop_left_margin, crop_box.x0),
            max(config.detection.crop_top_margin, crop_box.y0),
            min(layout.width - config.detection.crop_right_margin, crop_box.x1),
            min(layout.height - config.detection.bottom_margin, crop_box.y1, bottom),
        )
        if crop_box.y1 <= crop_box.y0 or crop_box.x1 <= crop_box.x0:
            flags.append("markscheme_image_uncertain")
            continue
        if crop_box.y1 - crop_box.y0 > layout.height * 0.75:
            flags.append("markscheme_image_uncertain")
        regions.append(MarkSchemeCropRegion(layout.page_number, crop_box, table_detected=False))

    if len(regions) > 1:
        flags.append("markscheme_image_stitched")
    return regions, flags


def _next_anchor_page_has_no_legacy_continuation(
    page_number: int,
    start_page: int,
    next_anchor_page: int | None,
    top: float,
    bottom: float,
    config: AppConfig,
) -> bool:
    if next_anchor_page is None or page_number != next_anchor_page or page_number == start_page:
        return False
    return bottom <= top + max(45.0, config.detection.min_crop_height * 1.5)


def _legacy_foreign_mark_scheme_boundary_y(
    layout: PageLayout,
    current_question_number: str,
    top: float,
    bottom: float,
    config: AppConfig,
) -> float | None:
    for block in sorted(layout.blocks, key=lambda item: (item.bbox.y0, item.bbox.x0)):
        if block.bbox.y0 <= top + config.detection.anchor_y_tolerance or block.bbox.y0 >= bottom:
            continue
        if _is_footer_or_header_box(block.bbox, layout, config) or _is_mark_scheme_boilerplate(block.text):
            continue
        if _is_mark_scheme_header_text(block.text):
            continue
        parsed = parse_question_start(block.first_line, config)
        if not parsed:
            continue
        candidate_number = parent_question_id(parsed[0])
        if not _is_later_mark_scheme_question(candidate_number, parent_question_id(current_question_number)):
            continue
        cleaned = _clean_cell_text(block.first_line)
        if re.fullmatch(r"\d{1,2}", cleaned):
            continue
        if block.bbox.x0 > max(config.detection.question_start_max_x, 120):
            continue
        return block.bbox.y0
    return None


def _is_later_mark_scheme_question(candidate_number: str, current_number: str) -> bool:
    try:
        return int(candidate_number) > int(current_number)
    except ValueError:
        return candidate_number != current_number


def _mark_scheme_crop_confidence(
    regions: list[MarkSchemeCropRegion],
    layouts: list[PageLayout],
    flags: list[str],
) -> str:
    if not regions:
        return CropConfidence.LOW
    if any(flag in flags for flag in {"markscheme_image_missing", "markscheme_image_no_boundaries"}):
        return CropConfidence.LOW
    for region in regions:
        layout = next((layout for layout in layouts if layout.page_number == region.page_number), None)
        if layout and region.bbox.y1 - region.bbox.y0 > layout.height * 0.75:
            return CropConfidence.MEDIUM
    if "markscheme_image_uncertain" in flags:
        return CropConfidence.MEDIUM
    return CropConfidence.HIGH


def _mark_scheme_question_identities(
    mark_scheme_pdf: str | Path,
    expected_numbers: list[str],
    question_identities: dict[str, PaperIdentity] | None,
) -> dict[str, PaperIdentity]:
    supplied = {
        normalize_question_id(key): value
        for key, value in (question_identities or {}).items()
        if normalize_question_id(key)
    }
    identities: dict[str, PaperIdentity] = {}
    metadata = parse_filename_metadata(mark_scheme_pdf)
    for number in expected_numbers:
        canonical_number = normalize_question_id(number)
        if not canonical_number:
            continue
        if canonical_number in supplied:
            identities[canonical_number] = supplied[canonical_number]
            continue
        try:
            identities[canonical_number] = paper_identity_from_parts(
                syllabus=metadata.syllabus or "9709",
                subject_family=metadata.paper_family,
                year=metadata.year,
                session=session_for_source_path(
                    mark_scheme_pdf,
                    year=metadata.year,
                    fallback_session=metadata.normalized_session_key or metadata.session,
                ),
                component=metadata.component,
                question_number=canonical_number,
            )
        except IdentityError:
            continue
    return identities


def _mark_scheme_identity_fields(identity: PaperIdentity | None, config: AppConfig) -> dict[str, str]:
    if identity is None:
        return {}
    try:
        asset = AssetPathResolver(config.output.root_dir()).mark_scheme_image(identity)
    except IdentityError:
        return {}
    return {
        "question_id": asset.question_id,
        "paper_id": asset.paper_id,
        "component": asset.component,
        "canonical_path": asset.canonical_path,
    }


def _clear_stale_mark_scheme_images(
    mark_scheme_pdf: Path,
    expected_numbers: list[str],
    config: AppConfig,
    identities: dict[str, PaperIdentity],
) -> None:
    del mark_scheme_pdf
    resolver = AssetPathResolver(config.output.root_dir())
    expected_paths = {
        resolver.mark_scheme_image(identity).absolute_path
        for key, identity in identities.items()
        if key in {normalize_question_id(number) for number in expected_numbers}
    }
    mark_scheme_dir = next(iter(expected_paths)).parent if expected_paths else None
    if mark_scheme_dir is None or not mark_scheme_dir.exists():
        return
    prefixes = {
        f"{identity.subject_family}_{identity.year}_{identity.session_code}_{identity.component}_ms_q"
        for identity in identities.values()
    }
    for path in mark_scheme_dir.glob("*_ms_q*_markscheme*.png"):
        if not any(path.name.startswith(prefix) for prefix in prefixes):
            continue
        if path not in expected_paths:
            path.unlink(missing_ok=True)


def _pdf_box_to_pixel_box(box: BoundingBox, zoom: float, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, min(width - 1, int(box.x0 * zoom)))
    top = max(0, min(height - 1, int(box.y0 * zoom)))
    right = max(left + 1, min(width, int(box.x1 * zoom)))
    bottom = max(top + 1, min(height, int(box.y1 * zoom)))
    return (left, top, right, bottom)


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


def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _is_footer_or_header_box(box: BoundingBox, layout: PageLayout, config: AppConfig) -> bool:
    return box.y1 < config.detection.crop_top_margin or box.y0 > layout.height - config.detection.bottom_margin


def _is_mark_scheme_boilerplate(text: str) -> bool:
    cleaned = " ".join(text.split())
    patterns = [
        r"^©\s*UCLES\b",
        r"^UCLES\b",
        r"^Cambridge International",
        r"^Cambridge International AS/A Level",
        r"^This document consists of",
        r"^BLANK PAGE$",
        r"^Mark Scheme$",
        r"^Question Paper$",
        r"^GCE A/?AS LEVEL\b",
        r"^GCE AS/?A LEVEL\b",
        r"^9709[/_ -]",
        r"^Page\s+\d+",
    ]
    return any(re.search(pattern, cleaned, re.IGNORECASE) for pattern in patterns)


def _clean_cell_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip().lower()


def _write_mark_scheme_debug_overlays(
    rendered_pages: dict[int, tuple["Image.Image", float]],
    mark_scheme_pdf: Path,
    question_number: str,
    layouts: list[PageLayout],
    tables: dict[int, MarkSchemeTable],
    anchors: list[MarkSchemeAnchor],
    regions: list[MarkSchemeCropRegion],
    config: AppConfig,
) -> list[str]:
    from PIL import ImageDraw

    paths: list[str] = []
    for page_number, (image, zoom) in rendered_pages.items():
        if not any(region.page_number == page_number for region in regions):
            continue
        page_image = image.copy()
        draw = ImageDraw.Draw(page_image)
        table = tables.get(page_number)
        if table:
            draw.rectangle(_pdf_box_to_pixel_box(table.bbox, zoom, page_image.size), outline="cyan", width=4)
            x = int(table.question_col_right * zoom)
            draw.line((x, 0, x, page_image.height), fill="orange", width=3)
        for anchor in [item for item in anchors if item.page_number == page_number]:
            y0 = int(anchor.y0 * zoom)
            y1 = int(anchor.y1 * zoom)
            draw.rectangle((0, y0, page_image.width, y1), outline="yellow", width=2)
        for region in [item for item in regions if item.page_number == page_number]:
            draw.rectangle(_pdf_box_to_pixel_box(region.bbox, zoom, page_image.size), outline="magenta", width=5)
        paths.append(_save_mark_scheme_debug_image(page_image, mark_scheme_pdf, question_number, page_number, "table_crop", config))
    return paths


def _write_mark_scheme_debug_metadata(
    mark_scheme_pdf: Path,
    question_number: str,
    tables: dict[int, MarkSchemeTable],
    anchors: list[MarkSchemeAnchor],
    regions: list[MarkSchemeCropRegion],
    config: AppConfig,
) -> str:
    config.output.debug_dir.mkdir(parents=True, exist_ok=True)
    paper_name = _safe_basename(mark_scheme_pdf.stem)
    qid = f"q{int(question_number):02d}" if question_number.isdigit() else f"q{_safe_basename(question_number)}"
    path = config.output.debug_dir / f"{paper_name}_ms_{qid}_crop_metadata.json"
    payload = {
        "question_number": question_number,
        "tables": [
            {
                "page_number": table.page_number,
                "bbox": _box_payload(table.bbox),
                "header_detected": table.header_detected,
                "header_bottom": table.header_bottom,
                "question_col_right": table.question_col_right,
                "confidence": table.confidence,
            }
            for table in tables.values()
        ],
        "question_number_anchors": [
            {
                "question_number": anchor.question_number,
                "page_number": anchor.page_number,
                "y0": anchor.y0,
                "y1": anchor.y1,
                "x0": anchor.x0,
                "text": anchor.text,
                "in_answer_table": bool(anchor.table),
            }
            for anchor in anchors
        ],
        "selected_row_range": [
            {
                "page_number": region.page_number,
                "crop_box": _box_payload(region.bbox),
                "table_detected": region.table_detected,
            }
            for region in regions
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _display_path(path)


def _box_payload(box: BoundingBox) -> dict[str, float]:
    return {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1}


def _save_mark_scheme_debug_image(
    image: "Image.Image",
    mark_scheme_pdf: Path,
    question_number: str,
    page_number: int,
    kind: str,
    config: AppConfig,
) -> str:
    config.output.debug_dir.mkdir(parents=True, exist_ok=True)
    paper_name = _safe_basename(mark_scheme_pdf.stem)
    qid = f"q{int(question_number):02d}" if question_number.isdigit() else f"q{_safe_basename(question_number)}"
    path = config.output.debug_dir / f"{paper_name}_ms_{qid}_p{page_number:02d}_{kind}.png"
    image.save(path)
    return _display_path(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _safe_basename(stem: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in stem).strip("_") or "paper"


def _fallback_regex_answers(layouts: list[PageLayout], expected_numbers: list[str] | None) -> dict[str, str]:
    if not expected_numbers:
        return {}
    full_text = "\n".join(layout.text for layout in layouts)
    answers: dict[str, str] = {}
    for position, number in enumerate(expected_numbers):
        next_number = expected_numbers[position + 1] if position + 1 < len(expected_numbers) else None
        pattern = rf"(?ms)^\s*{re.escape(number)}\b(?P<body>.*?)(?=^\s*{re.escape(next_number)}\b|\Z)" if next_number else rf"(?ms)^\s*{re.escape(number)}\b(?P<body>.*)\Z"
        match = re.search(pattern, full_text)
        if match:
            answers[number] = match.group(0).strip()
    return answers
