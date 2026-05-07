from __future__ import annotations

from collections import defaultdict
from statistics import median
from pathlib import Path
import re
from typing import Any

from .config import AppConfig
from .image_limits import render_pdf_area
from .models import BoundingBox, PageLayout, TextBlock
from .mupdf_tools import quiet_mupdf


def extract_pdf_layout(pdf_path: str | Path, config: AppConfig, use_ocr: bool | None = None) -> list[PageLayout]:
    """Extract ordered text lines and graphic/image boxes from a PDF.

    PyMuPDF is intentionally imported lazily so preflight can report missing
    dependencies without the entire package failing to import.
    """

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError('PyMuPDF is required for PDF extraction. Install the project with `pip install -e ".[dev]"`.') from exc
    quiet_mupdf(fitz)

    pdf_path = Path(pdf_path)
    layouts: list[PageLayout] = []
    ocr_enabled = config.ocr.enabled if use_ocr is None else use_ocr

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            blocks = _extract_text_blocks(page, page_number, config)
            graphics = _extract_graphics(page)
            text_len = len(" ".join(block.text for block in blocks).strip())
            warning: str | None = None
            source = "pdf"

            if text_len < config.detection.min_text_chars_per_page:
                if ocr_enabled:
                    try:
                        ocr_blocks = _ocr_page(page, page_number, config)
                    except Exception as exc:  # pragma: no cover - depends on local OCR install
                        ocr_blocks = []
                        warning = f"ocr_failed:{exc.__class__.__name__}"
                    if ocr_blocks:
                        merged_blocks = _merge_pdf_and_ocr_blocks(blocks, ocr_blocks)
                        if merged_blocks:
                            blocks = merged_blocks
                            source = "pdf+ocr" if any(block.source == "ocr" for block in merged_blocks) else "pdf"
                            warning = "ocr_merged_low_pdf_text"
                        elif warning is None:
                            warning = "weak_text_no_ocr_words"
                    elif warning is None:
                        warning = "weak_text_no_ocr_words"
                else:
                    warning = "weak_text_ocr_disabled"
            elif ocr_enabled:
                try:
                    supplemental_ocr_blocks = _supplemental_sparse_lower_ocr_blocks(page, page_number, blocks, config, fitz)
                except Exception as exc:  # pragma: no cover - depends on local OCR install
                    supplemental_ocr_blocks = []
                    if warning is None:
                        warning = f"ocr_failed:{exc.__class__.__name__}"
                if supplemental_ocr_blocks:
                    merged_blocks = _merge_pdf_and_ocr_blocks(blocks, supplemental_ocr_blocks)
                    if len(merged_blocks) > len(blocks):
                        blocks = merged_blocks
                        source = "pdf+ocr"
                        warning = "ocr_merged_sparse_lower_region"

            layouts.append(
                PageLayout(
                    page_number=page_number,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    blocks=sorted(blocks, key=lambda block: (block.bbox.y0, block.bbox.x0)),
                    graphics=graphics,
                    text_source=source,
                    extraction_warning=warning,
                )
            )

    return layouts


def _extract_text_blocks(page: Any, page_number: int, config: AppConfig) -> list[TextBlock]:
    text_dict = page.get_text("dict")
    spans: list[dict[str, Any]] = []
    for raw_block in text_dict.get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        for raw_line in raw_block.get("lines", []):
            for span in raw_line.get("spans", []):
                if str(span.get("text", "")).strip():
                    normalized_span = dict(span)
                    normalized_span["bbox"] = _visual_bbox(page, span.get("bbox", [0, 0, 0, 0]))
                    if _is_control_artifact_span(normalized_span):
                        continue
                    if _is_margin_furniture_span(page, normalized_span, config):
                        continue
                    spans.append(normalized_span)

    visual_lines = _group_spans_into_visual_lines(spans, config.detection.span_line_y_tolerance)
    blocks: list[TextBlock] = []
    for line_spans in visual_lines:
        sorted_spans = sorted(line_spans, key=lambda span: (float(span.get("bbox", [0, 0, 0, 0])[0]), float(span.get("bbox", [0, 0, 0, 0])[1])))
        text = _line_text_from_spans(sorted_spans).strip()
        if not text:
            continue
        x0, y0, x1, y1 = _line_bbox_from_spans(sorted_spans)
        font_sizes = [float(span.get("size", 0)) for span in sorted_spans if span.get("text", "").strip()]
        font_names = [str(span.get("font", "")) for span in sorted_spans if span.get("text", "").strip()]
        font_size = sum(font_sizes) / len(font_sizes) if font_sizes else None
        font_name = font_names[0] if font_names else None
        blocks.append(
            TextBlock(
                page_number=page_number,
                text=text,
                bbox=BoundingBox(float(x0), float(y0), float(x1), float(y1)),
                source="pdf",
                font_size=font_size,
                font_name=font_name,
                is_bold=any("bold" in font.lower() for font in font_names),
            )
        )
    return blocks


def _group_spans_into_visual_lines(spans: list[dict[str, Any]], y_tolerance: float) -> list[list[dict[str, Any]]]:
    """Rebuild visual text lines from span boxes using spatial order.

    PDF content streams are often not ordered the way a student reads the page,
    especially around formulas. This function ignores raw parser order: it sorts
    all spans by y/x, groups nearby y positions into a visual line, and lets the
    caller sort within each line by x.
    """

    sorted_spans = sorted(
        [span for span in spans if str(span.get("text", "")).strip()],
        key=lambda span: (_span_center_y(span), _span_x0(span)),
    )
    lines: list[list[dict[str, Any]]] = []
    for span in sorted_spans:
        target_index = _matching_line_index(span, lines, y_tolerance)
        if target_index is None:
            lines.append([span])
        else:
            lines[target_index].append(span)

    normalized_lines: list[list[dict[str, Any]]] = []
    for line in lines:
        normalized_lines.append(sorted(line, key=lambda span: (_span_x0(span), _span_center_y(span))))

    return sorted(normalized_lines, key=lambda line: (_line_center_y(line), min(_span_x0(span) for span in line)))


def _matching_line_index(span: dict[str, Any], lines: list[list[dict[str, Any]]], y_tolerance: float) -> int | None:
    best_index: int | None = None
    best_score: float | None = None
    span_size = max(1.0, float(span.get("size", 0)))
    span_center = _span_center_y(span)

    for index, line in enumerate(lines):
        line_center = _line_center_y(line)
        line_size = max(1.0, _line_median_font_size(line))
        distance = abs(span_center - line_center)
        overlap_ratio = _vertical_overlap_ratio(span, line)
        tolerance = max(y_tolerance, line_size * 0.8, span_size * 0.8)

        if distance > tolerance and overlap_ratio < 0.35:
            continue

        score = distance - overlap_ratio * line_size
        if best_score is None or score < best_score:
            best_index = index
            best_score = score
    return best_index


def _is_mark_token(text: str) -> bool:
    text = text.strip()
    return len(text) >= 3 and text[0] == "[" and text[-1] == "]" and text[1:-1].isdigit()


def _is_question_number_token(text: str) -> bool:
    return text.strip().isdigit()


def _line_text_from_spans(spans: list[dict[str, Any]]) -> str:
    stacked_text = _stacked_fraction_line_text(spans)
    if stacked_text is not None:
        return stacked_text
    return _flat_line_text_from_spans(spans)


def _flat_line_text_from_spans(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return ""
    spans = sorted(spans, key=lambda span: (_span_x0(span), _span_center_y(span)))
    font_sizes = [float(span.get("size", 0)) for span in spans if span.get("text", "").strip()]
    max_size = max(font_sizes) if font_sizes else 0
    median_size = median(font_sizes) if font_sizes else 0
    line_bbox = _line_bbox_from_spans(spans)
    line_mid = (line_bbox[1] + line_bbox[3]) / 2
    pieces: list[str] = []
    previous_span: dict[str, Any] | None = None
    previous_x1: float | None = None
    previous_text = ""

    for span in spans:
        text = _span_text(span)
        if not text:
            continue
        x0, y0, x1, y1 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
        gap = x0 - previous_x1 if previous_x1 is not None else 0.0
        operator_gap = _needs_operator_spacing(previous_text, text) and gap > 0.5
        threshold = max(2.0, float(span.get("size", max_size or 1)) * 0.35)
        if previous_x1 is not None and (operator_gap or gap > threshold):
            pieces.append(" ")

        normalized = text.strip()
        size = float(span.get("size", max_size or 0))
        span_mid = (y0 + y1) / 2
        vertical_shift = abs(span_mid - line_mid)
        previous_bbox = previous_span.get("bbox", [0, 0, 0, 0]) if previous_span is not None else None
        previous_gap = x0 - float(previous_bbox[2]) if previous_bbox is not None else float("inf")
        previous_mid = (
            (float(previous_bbox[1]) + float(previous_bbox[3])) / 2 if previous_bbox is not None else line_mid
        )
        baseline_shift_from_previous = abs(span_mid - previous_mid)
        small_math_token = 0 < len(normalized) <= 2 and normalized not in {",", ".", ":", ";"}
        attached_to_previous = previous_span is not None and previous_gap <= max(2.0, median_size * 0.35)
        previous_text_normalized = _span_text(previous_span).strip() if previous_span is not None else ""
        previous_supports_script = any(ch.isalnum() or ch in ")]" for ch in previous_text_normalized)
        is_script_candidate = (
            bool(normalized)
            and not _is_mark_token(normalized)
            and not (_is_question_number_token(normalized) and not attached_to_previous)
            and bool(max_size)
            and bool(median_size)
            and small_math_token
            and size <= max_size * 0.82
            and attached_to_previous
            and previous_supports_script
            and (
                vertical_shift >= max(1.0, median_size * 0.08)
                or baseline_shift_from_previous >= max(1.0, median_size * 0.15)
            )
        )
        if is_script_candidate:
            script_threshold = max(0.6, median_size * 0.04)
            if span_mid < previous_mid - script_threshold:
                pieces.append(f"^{{{text}}}")
                previous_span = span
                previous_x1 = x1
                previous_text = text
                continue
            if span_mid > previous_mid + script_threshold:
                pieces.append(f"_{{{text}}}")
                previous_span = span
                previous_x1 = x1
                previous_text = text
                continue

        pieces.append(text)
        previous_span = span
        previous_x1 = x1
        previous_text = text

    return _repair_line_spacing("".join(pieces))


def _stacked_fraction_line_text(spans: list[dict[str, Any]]) -> str | None:
    spans = [span for span in spans if _span_text(span).strip()]
    if len(spans) < 4:
        return None

    font_sizes = [float(span.get("size", 0)) for span in spans if _span_text(span).strip()]
    median_size = median(font_sizes) if font_sizes else 0
    if median_size <= 0:
        return None

    centers = [_span_center_y(span) for span in spans]
    top_center = min(centers)
    bottom_center = max(centers)
    if bottom_center - top_center < max(7.0, median_size * 0.65):
        return None

    top_cutoff = top_center + median_size * 0.45
    bottom_cutoff = bottom_center - median_size * 0.45
    if top_cutoff >= bottom_cutoff:
        return None
    top_bottom_spans = [
        span
        for span in spans
        if (_span_center_y(span) <= top_cutoff or _span_center_y(span) >= bottom_cutoff)
        and _is_fraction_row_span_candidate(span, median_size)
    ]
    components = _stacked_fraction_components(top_bottom_spans, median_size, top_cutoff, bottom_cutoff)
    if not components:
        return None

    consumed_ids = {id(span) for component in components for span in component}
    segments: list[dict[str, Any]] = []
    for component in components:
        top_spans = [span for span in component if _span_center_y(span) <= top_cutoff]
        bottom_spans = [span for span in component if _span_center_y(span) >= bottom_cutoff]
        top_text = _flat_line_text_from_spans(top_spans).strip()
        bottom_text = _flat_line_text_from_spans(bottom_spans).strip()
        if not top_text or not bottom_text:
            continue
        x0, y0, x1, y1 = _line_bbox_from_spans(component)
        segments.append(
            {
                "text": f"({top_text})/({bottom_text})",
                "bbox": [x0, y0, x1, y1],
                "size": median_size,
                "font": "",
            }
        )

    if not segments:
        return None
    segments.extend(span for span in spans if id(span) not in consumed_ids)
    return _flat_line_text_from_spans(segments)


def _stacked_fraction_components(
    spans: list[dict[str, Any]],
    median_size: float,
    top_cutoff: float,
    bottom_cutoff: float,
) -> list[list[dict[str, Any]]]:
    components: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_x1: float | None = None
    max_gap = max(3.0, median_size * 0.75)

    for span in sorted(spans, key=lambda item: (_span_x0(item), _span_center_y(item))):
        x0, _y0, x1, _y1 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
        if current and current_x1 is not None and x0 > current_x1 + max_gap:
            if _is_stacked_fraction_component(current, median_size, top_cutoff, bottom_cutoff):
                components.append(current)
            current = []
            current_x1 = None
        current.append(span)
        current_x1 = max(current_x1 if current_x1 is not None else x1, x1)

    if current and _is_stacked_fraction_component(current, median_size, top_cutoff, bottom_cutoff):
        components.append(current)
    return components


def _is_stacked_fraction_component(
    spans: list[dict[str, Any]],
    median_size: float,
    top_cutoff: float,
    bottom_cutoff: float,
) -> bool:
    top_spans = [span for span in spans if _span_center_y(span) <= top_cutoff]
    bottom_spans = [span for span in spans if _span_center_y(span) >= bottom_cutoff]
    if not top_spans or not bottom_spans:
        return False
    top_text = _flat_line_text_from_spans(top_spans).strip()
    bottom_text = _flat_line_text_from_spans(bottom_spans).strip()
    if _looks_like_fraction_prose_text(top_text) or _looks_like_fraction_prose_text(bottom_text):
        return False

    top_bbox = _line_bbox_from_spans(top_spans)
    bottom_bbox = _line_bbox_from_spans(bottom_spans)
    overlap = max(0.0, min(top_bbox[2], bottom_bbox[2]) - max(top_bbox[0], bottom_bbox[0]))
    top_width = top_bbox[2] - top_bbox[0]
    bottom_width = bottom_bbox[2] - bottom_bbox[0]
    narrower_width = max(0.1, min(top_width, bottom_width))
    if overlap / narrower_width < 0.45:
        return False
    center_delta = abs(((top_bbox[0] + top_bbox[2]) / 2) - ((bottom_bbox[0] + bottom_bbox[2]) / 2))
    if center_delta > max(median_size * 0.8, narrower_width * 0.45):
        return False
    if max(top_bbox[2], bottom_bbox[2]) - min(top_bbox[0], bottom_bbox[0]) < median_size * 1.4:
        return False

    combined_text = " ".join(_span_text(span).strip() for span in spans)
    return bool(re.search(r"[+\-=−]|(?:sin|cos|tan|sec|cosec|cot|ln|log)|[A-Za-z]\s", combined_text))


def _is_fraction_row_span_candidate(span: dict[str, Any], median_size: float) -> bool:
    text = _span_text(span).strip()
    if not text or _is_mark_token(text):
        return False
    if not re.search(r"[A-Za-z0-9θπ≡=+\-−*/]", text):
        return False
    if _looks_like_fraction_prose_text(text):
        return False

    x0, y0, x1, y1 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
    height = max(0.0, y1 - y0)
    width = max(0.0, x1 - x0)
    if height > median_size * 1.4 and not _is_control_parenthesis_text(text):
        return False
    if width > median_size * 22 and _span_prose_word_count(text) >= 1:
        return False
    return True


def _looks_like_fraction_prose_text(text: str) -> bool:
    normalized = " ".join(text.replace("\u00a0", " ").split())
    if not normalized:
        return False
    if re.search(r"\([a-z]\)", normalized, re.IGNORECASE):
        return True
    if re.search(r"\b(?:DO NOT WRITE|UCLES|Turn over)\b", normalized, re.IGNORECASE):
        return True
    prose_words = [
        word.lower()
        for word in re.findall(r"[A-Za-z]{2,}", normalized)
        if word.lower() not in _FRACTION_MATH_WORDS
    ]
    if any(word in _FRACTION_PROSE_WORDS for word in prose_words):
        return True
    return len(prose_words) >= 2


def _span_prose_word_count(text: str) -> int:
    return sum(
        1
        for word in re.findall(r"[A-Za-z]{2,}", text)
        if word.lower() not in _FRACTION_MATH_WORDS
    )


def _is_control_parenthesis_text(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(char in "()\x00\x01" for char in stripped)


_FRACTION_MATH_WORDS = {
    "sin",
    "cos",
    "tan",
    "sec",
    "cosec",
    "csc",
    "cot",
    "ln",
    "log",
    "exp",
}

_FRACTION_PROSE_WORDS = {
    "ascending",
    "coefficient",
    "coordinates",
    "curve",
    "equation",
    "exact",
    "expand",
    "expansion",
    "express",
    "find",
    "given",
    "hence",
    "identity",
    "including",
    "point",
    "powers",
    "prove",
    "show",
    "stationary",
    "term",
    "terms",
    "value",
    "where",
}


def _span_text(span: dict[str, Any] | None) -> str:
    if span is None:
        return ""
    text = str(span.get("text", ""))
    replacements = {
        "\x00": "(",
        "\x01": ")",
        "\x8f": "≡",
        "": "≡",
        "Å": "°",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _needs_operator_spacing(previous_text: str, text: str) -> bool:
    operators = {"+", "-", "=", "<", ">", "≤", "≥", "±"}
    return previous_text.strip() in operators or text.strip() in operators


def _is_margin_furniture_span(page: Any, span: dict[str, Any], config: AppConfig) -> bool:
    text = " ".join(str(span.get("text", "")).replace("\u00a0", " ").split())
    if not text:
        return False
    if not re.search(r"DO NOT WRITE IN THIS MARGIN", text, re.IGNORECASE):
        return False

    x0, y0, x1, y1 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    near_left = x0 <= config.detection.crop_left_margin
    near_right = x1 >= page_width - config.detection.crop_right_margin
    return width <= 80 and height >= page_height * 0.16 and (near_left or near_right)


def _is_control_artifact_span(span: dict[str, Any]) -> bool:
    text = str(span.get("text", ""))
    control_count = sum(1 for char in text if ord(char) < 32 and char not in "\n\t\r")
    if control_count < 4:
        return False
    visible_count = sum(1 for char in text if char.isalnum())
    return control_count >= max(4, visible_count)


def _repair_line_spacing(text: str) -> str:
    value = text
    value = re.sub(r"\b(ln|log)(?=[A-Za-z0-9(])", r"\1 ", value)
    value = re.sub(r"\b(ln|log)\s+\(", r"\1(", value)
    value = re.sub(r"\b(cosec|sin|cos|tan|sec|cot)(?=[0-9θxyz])", r"\1 ", value)
    value = re.sub(r"(?<=[A-Za-z0-9}])(?=(?:cosec|sin|cos|tan|sec|cot)(?:[0-9θxyz]|\b))", " ", value)
    value = re.sub(r"\b(cosec|sin|cos|tan|sec|cot)(?=[0-9θxyz])", r"\1 ", value)
    value = re.sub(r"(?<=\})(?=[A-Za-z]{2,}\b)", " ", value)
    value = re.sub(r"\b([A-Za-z]{2,})([A-Z][a-z]{2,})\b", r"\1 \2", value)
    return value


def _line_bbox_from_spans(spans: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    bboxes = [span.get("bbox", [0, 0, 0, 0]) for span in spans if span.get("text", "").strip()]
    if not bboxes:
        return (0, 0, 0, 0)
    return (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )


def _span_x0(span: dict[str, Any]) -> float:
    return float(span.get("bbox", [0, 0, 0, 0])[0])


def _span_center_y(span: dict[str, Any]) -> float:
    bbox = span.get("bbox", [0, 0, 0, 0])
    return (float(bbox[1]) + float(bbox[3])) / 2


def _line_center_y(line: list[dict[str, Any]]) -> float:
    sizes = [max(0.1, float(span.get("size", 0))) for span in line]
    weighted = sum(_span_center_y(span) * size for span, size in zip(line, sizes))
    return weighted / sum(sizes)


def _line_median_font_size(line: list[dict[str, Any]]) -> float:
    sizes = sorted(float(span.get("size", 0)) for span in line if float(span.get("size", 0)) > 0)
    if not sizes:
        return 0.0
    middle = len(sizes) // 2
    if len(sizes) % 2:
        return sizes[middle]
    return (sizes[middle - 1] + sizes[middle]) / 2


def _vertical_overlap_ratio(span: dict[str, Any], line: list[dict[str, Any]]) -> float:
    bbox = span.get("bbox", [0, 0, 0, 0])
    span_top = float(bbox[1])
    span_bottom = float(bbox[3])
    line_top = min(float(item.get("bbox", [0, 0, 0, 0])[1]) for item in line)
    line_bottom = max(float(item.get("bbox", [0, 0, 0, 0])[3]) for item in line)
    overlap = max(0.0, min(span_bottom, line_bottom) - max(span_top, line_top))
    span_height = max(0.1, span_bottom - span_top)
    return overlap / span_height


def _extract_graphics(page: Any) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect and rect.is_valid and not rect.is_empty:
            box = _visual_box_from_rect(page, rect)
            if _is_meaningful_graphic_box(box, page_width, page_height):
                boxes.append(box)

    try:
        image_infos = page.get_image_info(xrefs=True)
    except Exception:
        image_infos = []
    for image_info in image_infos:
        bbox = image_info.get("bbox")
        if bbox:
            box = _visual_box_from_rect(page, bbox)
            if _is_meaningful_graphic_box(box, page_width, page_height):
                boxes.append(box)
    return boxes


def _is_meaningful_graphic_box(box: BoundingBox, page_width: float, page_height: float) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    area = width * height
    if area < 36:
        return False

    page_area = max(1.0, page_width * page_height)
    if area >= page_area * 0.9:
        return False

    very_thin_horizontal = width >= page_width * 0.25 and height <= 2.5
    very_thin_vertical = height >= page_height * 0.25 and width <= 2.5
    if very_thin_horizontal or very_thin_vertical:
        return False

    near_page_edge = box.y0 <= 8 or box.y1 >= page_height - 8 or box.x0 <= 8 or box.x1 >= page_width - 8
    edge_artifact = near_page_edge and area < page_area * 0.01
    if edge_artifact:
        return False

    return True


def _merge_pdf_and_ocr_blocks(pdf_blocks: list[TextBlock], ocr_blocks: list[TextBlock]) -> list[TextBlock]:
    merged = list(pdf_blocks)
    for ocr_block in ocr_blocks:
        if any(
            _boxes_overlap_ratio(existing.bbox, ocr_block.bbox) >= 0.55
            and not _existing_block_should_yield_to_ocr(existing)
            for existing in pdf_blocks
        ):
            continue
        merged.append(ocr_block)
    return sorted(merged, key=lambda block: (block.bbox.y0, block.bbox.x0))


def _boxes_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    overlap_w = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    overlap_h = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    overlap_area = overlap_w * overlap_h
    if overlap_area <= 0:
        return 0.0
    min_area = max(1.0, min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0)))
    return overlap_area / min_area


def _visual_bbox(page: Any, bbox: Any) -> list[float]:
    box = _visual_box_from_rect(page, bbox)
    return [box.x0, box.y0, box.x1, box.y1]


def _visual_box_from_rect(page: Any, rect_like: Any) -> BoundingBox:
    try:
        import fitz

        rect = fitz.Rect(rect_like)
        if getattr(page, "rotation", 0):
            rect = rect * page.rotation_matrix
        return BoundingBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    except Exception:
        x0, y0, x1, y1 = rect_like
        return BoundingBox(float(x0), float(y0), float(x1), float(y1))


def _supplemental_sparse_lower_ocr_blocks(
    page: Any,
    page_number: int,
    pdf_blocks: list[TextBlock],
    config: AppConfig,
    fitz: Any,
) -> list[TextBlock]:
    clip = _sparse_lower_ocr_clip(page, pdf_blocks, config, fitz)
    if clip is None:
        return []

    ocr_blocks = _ocr_page(
        page,
        page_number,
        config,
        clip=clip,
        context="ocr_sparse_lower_region",
    )
    if not ocr_blocks:
        return []

    signal_blocks = [block for block in ocr_blocks if _is_sparse_lower_region_signal(block.text)]
    if not signal_blocks:
        return []

    return [
        block
        for block in ocr_blocks
        if _is_sparse_lower_region_keep_block(block.text)
    ]


def _sparse_lower_ocr_clip(
    page: Any,
    pdf_blocks: list[TextBlock],
    config: AppConfig,
    fitz: Any,
) -> Any | None:
    page_height = float(page.rect.height)
    body_top = float(config.detection.crop_top_margin)
    body_bottom = page_height - float(config.detection.crop_bottom_margin)
    substantive_blocks = [
        block
        for block in sorted(pdf_blocks, key=lambda item: (item.bbox.y0, item.bbox.x0))
        if _is_sparse_lower_region_body_block(block, page_height, config)
    ]
    if not substantive_blocks:
        return None

    last_body_block = substantive_blocks[-1]
    tail_gap = body_bottom - last_body_block.bbox.y1
    min_gap = max(150.0, config.detection.prompt_region_max_gap * 2.4)
    if tail_gap < min_gap:
        return None
    if last_body_block.bbox.y1 >= body_bottom - max(110.0, config.detection.prompt_region_max_gap * 1.4):
        return None

    start_y = max(
        last_body_block.bbox.y1 + config.detection.crop_padding + 6.0,
        body_top + 70.0,
    )
    if start_y >= body_bottom - 40:
        return None

    return fitz.Rect(0, start_y, float(page.rect.width), body_bottom)


def _is_sparse_lower_region_body_block(block: TextBlock, page_height: float, config: AppConfig) -> bool:
    if block.bbox.y1 < config.detection.crop_top_margin:
        return False
    if block.bbox.y0 > page_height - config.detection.bottom_margin:
        return False
    return _is_sparse_lower_region_keep_block(block.text)


def _is_sparse_lower_region_signal(text: str) -> bool:
    cleaned = _normalized_ocr_text(text)
    if not cleaned:
        return False
    if re.match(r"^\s*(?:\d+\s+(?:\([a-zivxlcdm]+\)\s*)?\S|\([a-zivxlcdm]+\)\s+\S)", cleaned, re.IGNORECASE):
        return True
    if re.search(r"\[\d{1,2}\]", cleaned):
        return True
    return sum(1 for char in cleaned if char.isalpha()) >= 8


def _is_sparse_lower_region_keep_block(text: str) -> bool:
    cleaned = _normalized_ocr_text(text)
    if not cleaned:
        return False
    if re.search(
        r"WRITE IN THIS MARGIN|DO NOT W(?:RITE)?|©\s*UCLES|Cambridge International|Turn over",
        cleaned,
        re.IGNORECASE,
    ):
        return False
    if re.fullmatch(r"[._\-–—=\s]{4,}", cleaned):
        return False
    alpha_numeric = sum(1 for char in cleaned if char.isalnum())
    if alpha_numeric >= 2:
        return True
    return bool(
        re.search(r"^\s*(?:\d+\s+(?:\([a-zivxlcdm]+\)\s*)?\S|\([a-zivxlcdm]+\)\s+\S)", cleaned, re.IGNORECASE)
        or re.search(r"\[\d{1,2}\]", cleaned)
    )


def _normalized_ocr_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def _existing_block_should_yield_to_ocr(block: TextBlock) -> bool:
    cleaned = _normalized_ocr_text(block.text)
    height = max(0.0, block.bbox.y1 - block.bbox.y0)
    if height >= 160:
        return True
    if re.search(r"WRITE IN THIS MARGIN|©\s*UCLES|Cambridge International|Turn over", cleaned, re.IGNORECASE):
        return True
    if re.fullmatch(r"[._\-–—=\s]{8,}", cleaned):
        return True
    return False


def _normalize_ocr_block_text(text: str) -> str:
    normalized = _normalized_ocr_text(text)
    normalized = re.sub(r"\{(\d{1,2})\]", r"[\1]", normalized)
    normalized = re.sub(r"\[(\d{1,2})\}", r"[\1]", normalized)
    normalized = re.sub(r"\((\d{1,2})\]", r"[\1]", normalized)
    normalized = re.sub(r"\[(\d{1,2})\)", r"[\1]", normalized)
    return normalized


def _ocr_page(
    page: Any,
    page_number: int,
    config: AppConfig,
    *,
    clip: Any | None = None,
    context: str = "ocr_page",
) -> list[TextBlock]:
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pytesseract and Pillow are required for OCR fallback.") from exc
    quiet_mupdf(fitz)
    if config.ocr.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.ocr.tesseract_cmd

    image, _used_zoom = render_pdf_area(
        page,
        fitz,
        dpi=config.ocr.dpi,
        source_file=getattr(page.parent, "name", "<pdf>"),
        page_number=page_number,
        context=context,
        clip=clip,
    )
    data = pytesseract.image_to_data(
        image,
        lang=config.ocr.language,
        output_type=pytesseract.Output.DICT,
        timeout=config.ocr.timeout_seconds,
    )

    grouped: dict[tuple[int, int, int], list[tuple[str, int, int, int, int, float]]] = defaultdict(list)
    for index, word in enumerate(data.get("text", [])):
        word = word.strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            confidence = -1
        if confidence >= 0 and confidence < config.ocr.min_confidence:
            continue
        key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
        grouped[key].append(
            (
                word,
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
                confidence,
            )
        )

    clip_rect = fitz.Rect(clip) if clip is not None else fitz.Rect(page.rect)
    scale_x = float(clip_rect.width) / image.width
    scale_y = float(clip_rect.height) / image.height
    blocks: list[TextBlock] = []
    for words in grouped.values():
        words.sort(key=lambda item: item[1])
        text = _normalize_ocr_block_text(" ".join(item[0] for item in words))
        left = min(item[1] for item in words)
        top = min(item[2] for item in words)
        right = max(item[1] + item[3] for item in words)
        bottom = max(item[2] + item[4] for item in words)
        confidences = [item[5] for item in words if item[5] >= 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        blocks.append(
            TextBlock(
                page_number=page_number,
                text=text,
                bbox=BoundingBox(
                    float(clip_rect.x0) + left * scale_x,
                    float(clip_rect.y0) + top * scale_y,
                    float(clip_rect.x0) + right * scale_x,
                    float(clip_rect.y0) + bottom * scale_y,
                ),
                source="ocr",
                confidence=avg_confidence,
            )
        )

    if blocks:
        return sorted(blocks, key=lambda block: (block.bbox.y0, block.bbox.x0))

    text = _normalize_ocr_block_text(
        pytesseract.image_to_string(image, lang=config.ocr.language, timeout=config.ocr.timeout_seconds).strip()
    )
    if not text:
        return []
    return [
        TextBlock(
            page_number=page_number,
            text=text,
            bbox=BoundingBox(
                float(clip_rect.x0),
                float(clip_rect.y0),
                float(clip_rect.x1),
                float(clip_rect.y1),
            ),
            source="ocr",
            confidence=None,
        )
    ]
