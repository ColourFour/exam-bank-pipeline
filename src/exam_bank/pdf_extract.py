from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .config import AppConfig
from .image_limits import render_pdf_area
from .models import BoundingBox, PageLayout, TextBlock
from .mupdf_tools import quiet_mupdf
from .ocr import score_text_candidate


_SPARSE_LOWER_OCR_CORRUPTION_REASONS = {
    "excessive_isolated_symbols",
    "merged_word_artifacts",
    "ocr_symbol_garbage",
    "pdf_control_or_replacement_garbage",
}


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
    legacy_fallback = _is_legacy_pdf(pdf_path)

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            blocks = _extract_text_blocks(page, page_number, config)
            graphics = _extract_graphics(page, legacy_fallback=legacy_fallback)
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

            ocr_hint_graphics = _ocr_hint_graphics(blocks, graphics, page_width=float(page.rect.width), page_height=float(page.rect.height), legacy_fallback=legacy_fallback)
            if ocr_hint_graphics:
                graphics = _dedupe_boxes([*graphics, *ocr_hint_graphics])
                warning = _append_warning(warning, "ocr_hint_figure_regions")

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

    return _normalize_encoded_digit_text(layouts)


def _is_legacy_pdf(pdf_path: Path) -> bool:
    match = re.search(r"_(?P<season>[msw])(?P<yy>\d{2})_", pdf_path.name.lower())
    if not match:
        return False
    return int(match.group("yy")) < 17


_PDF_GLYPH_EXACT_UNICODE = {
    "Pislant": "Π",
    "Sigma": "Σ",
}

_PDF_GLYPH_UNICODE_ALIASES = {
    "alpha": "α",
    "alphaslant": "α",
    "beta": "β",
    "betaslant": "β",
    "gamma": "γ",
    "gammaslant": "γ",
    "delta": "δ",
    "deltaslant": "δ",
    "epsilon": "ε",
    "epsilonslant": "ε",
    "eta": "η",
    "etaslant": "η",
    "theta": "θ",
    "thetaslant": "θ",
    "lambda": "λ",
    "lambdaslant": "λ",
    "mu": "μ",
    "muslant": "μ",
    "pi": "π",
    "pislant": "π",
    "sigma": "σ",
    "sigmaslant": "σ",
    "phi": "φ",
    "phislant": "φ",
    "degrees": "°",
    "textint": "∫",
    "displayint": "∫",
    "root": "√",
    "realset": "ℝ",
    # CAIE's ``zmath`` glyph is the italic complex variable z, not the
    # blackboard-bold integer-set symbol.
    "zmath": "z",
    "equivalent": "≡",
    "approxequal": "≈",
    "similar": "∼",
    "mapsto": "↦",
    "arrowright": "→",
    "multiply": "×",
    "minus": "−",
    "plus": "+",
    "equal": "=",
    "less": "<",
    "greater": ">",
    "lessequal": "≤",
    "greaterequal": "≥",
    "notequal": "≠",
    "element": "∈",
    "infinity": "∞",
    "prime": "′",
}


def _normalized_pdf_font_name(name: Any) -> str:
    value = _decode_pdf_name(str(name or "").lstrip("/"))
    value = re.sub(r"^[A-Za-z]{6}\+", "", value)
    return value.casefold()


def _decode_pdf_name(value: str) -> str:
    return re.sub(
        r"#([0-9A-Fa-f]{2})",
        lambda match: chr(int(match.group(1), 16)),
        value,
    )


def _unicode_for_pdf_glyph_name(name: str) -> str | None:
    exact = _PDF_GLYPH_EXACT_UNICODE.get(name)
    if exact is not None:
        return exact

    normalized = name.casefold()
    alias = _PDF_GLYPH_UNICODE_ALIASES.get(normalized)
    if alias is not None:
        return alias
    if normalized.startswith("surd"):
        return "√"
    if normalized.startswith(("lpar", "parenleft")):
        return "("
    if normalized.startswith(("rpar", "parenright")):
        return ")"
    if normalized.startswith(("lbrk", "bracketleft")):
        return "["
    if normalized.startswith(("rbrk", "bracketright")):
        return "]"
    if normalized in {"vert", "vertb"}:
        return "|"
    return None


def _font_encoding_differences(document: Any, font_xref: int) -> dict[int, str]:
    try:
        value_type, value = document.xref_get_key(font_xref, "Encoding")
    except Exception:
        return {}

    encoding_object = ""
    if value_type == "xref":
        match = re.match(r"\s*(\d+)\s+\d+\s+R\s*$", str(value))
        if match is None:
            return {}
        try:
            encoding_object = str(document.xref_object(int(match.group(1)), compressed=False))
        except Exception:
            return {}
    elif value_type in {"dict", "array"}:
        encoding_object = str(value)
    else:
        return {}

    differences_match = re.search(r"/Differences\s*\[(.*?)\]", encoding_object, re.DOTALL)
    if differences_match is None:
        return {}

    tokens = re.findall(r"\d+|/[^\s<>\[\](){}%/]+", differences_match.group(1))
    differences: dict[int, str] = {}
    current_code: int | None = None
    for token in tokens:
        if token.isdigit():
            current_code = int(token)
            continue
        if current_code is None or not token.startswith("/"):
            continue
        differences[current_code] = _decode_pdf_name(token[1:])
        current_code += 1
    return differences


def _page_font_fallback_replacements(page: Any) -> dict[str, dict[str, str]]:
    """Return unanimous font-scoped repairs for PyMuPDF's unknown-Unicode fallback.

    With the default ``TEXT_USE_CID_FOR_UNKNOWN_UNICODE`` behavior, an
    unmapped one-byte Type1 character is returned as ``chr(character_code)``.
    CAIE math fonts retain the intended semantic glyph name in their Encoding
    Differences array even when ToUnicode omits it.  Duplicate subset fonts can
    share the same normalized name on a page, so disagreeing definitions fail
    closed instead of guessing.
    """

    try:
        document = page.parent
        fonts = page.get_fonts(full=True)
    except Exception:
        return {}

    candidates: dict[str, dict[int, set[str | None]]] = defaultdict(lambda: defaultdict(set))
    for font in fonts:
        if not isinstance(font, (list, tuple)) or len(font) < 4:
            continue
        try:
            font_xref = int(font[0])
        except (TypeError, ValueError):
            continue
        if font_xref <= 0:
            continue
        font_name = _normalized_pdf_font_name(font[3])
        if not font_name:
            continue
        for code, glyph_name in _font_encoding_differences(document, font_xref).items():
            if not 0 <= code <= 0xFF:
                continue
            replacement = _unicode_for_pdf_glyph_name(glyph_name)
            candidates[font_name][code].add(replacement)

    replacements: dict[str, dict[str, str]] = {}
    for font_name, code_candidates in candidates.items():
        font_replacements: dict[str, str] = {}
        for code, values in code_candidates.items():
            if len(values) != 1:
                continue
            replacement = next(iter(values))
            if replacement is None:
                continue
            fallback = chr(code)
            if fallback != replacement:
                font_replacements[fallback] = replacement
        if font_replacements:
            replacements[font_name] = font_replacements
    return replacements


def _repair_font_encoded_span_text(
    span: dict[str, Any],
    replacements: dict[str, dict[str, str]],
) -> str:
    text = str(span.get("text", ""))
    font_name = _normalized_pdf_font_name(span.get("font", ""))
    font_replacements = replacements.get(font_name)
    if not font_replacements:
        return text
    return "".join(font_replacements.get(char, char) for char in text)


def _normalize_encoded_digit_text(layouts: list[PageLayout]) -> list[PageLayout]:
    digit_map = _encoded_digit_translation_from_page_numbers(layouts)
    if not digit_map:
        return layouts

    normalized_layouts: list[PageLayout] = []
    for layout in layouts:
        blocks = [
            TextBlock(
                page_number=block.page_number,
                text=_normalize_encoded_digit_block_text(block.text, digit_map),
                bbox=block.bbox,
                source=block.source,
                confidence=block.confidence,
                font_size=block.font_size,
                font_name=block.font_name,
                is_bold=block.is_bold,
            )
            for block in layout.blocks
        ]
        normalized_layouts.append(
            PageLayout(
                page_number=layout.page_number,
                width=layout.width,
                height=layout.height,
                blocks=blocks,
                graphics=layout.graphics,
                text_source=layout.text_source,
                extraction_warning=layout.extraction_warning,
            )
        )
    return normalized_layouts


def _encoded_digit_translation_from_page_numbers(layouts: list[PageLayout]) -> dict[str, str]:
    assignments: dict[str, set[str]] = defaultdict(set)
    for layout in layouts:
        expected = str(layout.page_number)
        if layout.page_number <= 1:
            continue
        for block in layout.blocks:
            if not (260 <= block.bbox.x0 <= 330 and 28 <= block.bbox.y0 <= 46):
                continue
            token = "".join(block.text.split())
            if not token or token == expected or token.isdigit() or len(token) != len(expected):
                continue
            for char, digit in zip(token, expected):
                if char != digit:
                    assignments[char].add(digit)

    digit_map = {
        char: next(iter(digits))
        for char, digits in assignments.items()
        if len(digits) == 1 and not char.isdigit()
    }
    if len(set(digit_map.values())) < 3:
        return {}
    return digit_map


_ENCODED_DIGIT_SEPARATOR_CHARS = {"\x81", "\x82", "{", "~"}


def _normalize_encoded_digit_block_text(text: str, digit_map: dict[str, str]) -> str:
    compact = "".join(text.split())
    if compact and len(compact) <= 3 and all(char.isdigit() or char in digit_map for char in compact):
        return text.translate(str.maketrans(digit_map))

    prefix: list[str] = []
    index = 0
    while index < len(text) and (text[index].isdigit() or text[index] in digit_map):
        prefix.append(digit_map.get(text[index], text[index]))
        index += 1
    if not prefix:
        return text
    if index < len(text) and text[index] not in _ENCODED_DIGIT_SEPARATOR_CHARS and not text[index].isspace():
        return text

    separator = " " if index < len(text) and text[index] in _ENCODED_DIGIT_SEPARATOR_CHARS else ""
    return "".join(prefix) + separator + text[index + len(separator) :]


def _extract_text_blocks(page: Any, page_number: int, config: AppConfig) -> list[TextBlock]:
    font_fallback_replacements = _page_font_fallback_replacements(page)
    text_dict = page.get_text("dict")
    spans: list[dict[str, Any]] = []
    for raw_block in text_dict.get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        for raw_line in raw_block.get("lines", []):
            for span in raw_line.get("spans", []):
                if not str(span.get("text", "")):
                    continue
                normalized_span = dict(span)
                normalized_span["text"] = _repair_font_encoded_span_text(
                    normalized_span,
                    font_fallback_replacements,
                )
                normalized_span["bbox"] = _visual_bbox(page, span.get("bbox", [0, 0, 0, 0]))
                if _is_control_artifact_span(normalized_span):
                    continue
                if _is_margin_furniture_span(page, normalized_span, config):
                    continue
                # Pure-space spans carry real word-boundary information.  PDF
                # geometry alone cannot recover every boundary: adjacent spans
                # may touch even when a separately encoded space was printed.
                # Preserve these spans through visual-line assembly and strip
                # only the completed line.
                spans.append(normalized_span)

    visual_lines = _group_spans_into_visual_lines(spans, config.detection.span_line_y_tolerance)
    serialized_lines = _serialize_table_visual_lines(visual_lines)
    blocks: list[TextBlock] = []
    for line_spans, text_override in serialized_lines:
        sorted_spans = sorted(line_spans, key=lambda span: (float(span.get("bbox", [0, 0, 0, 0])[0]), float(span.get("bbox", [0, 0, 0, 0])[1])))
        text = (
            text_override
            if text_override is not None
            else _line_text_from_spans(sorted_spans)
        ).strip()
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

    source_spans = [span for span in spans if str(span.get("text", ""))]
    font_sizes = [
        float(span.get("size", 0))
        for span in source_spans
        if _span_text(span).strip() and float(span.get("size", 0)) > 0
    ]
    body_size = median(font_sizes) if font_sizes else 0.0
    anchor_spans: list[dict[str, Any]] = []
    deferred_spans: list[dict[str, Any]] = []
    for span in source_spans:
        target = (
            deferred_spans
            if _is_deferred_small_line_token(span, body_size)
            else anchor_spans
        )
        target.append(span)

    lines: list[list[dict[str, Any]]] = []
    # Establish the printed prose/math baselines before assigning scripts.
    # A small exponent often sits between two baselines; processing it first
    # lets it pull both lines into one transitive y-cluster.
    for span_group in (anchor_spans, deferred_spans):
        for span in sorted(
            span_group,
            key=lambda item: (_span_center_y(item), _span_x0(item)),
        ):
            target_index = _matching_line_index(span, lines, y_tolerance)
            if target_index is None:
                lines.append([span])
            else:
                lines[target_index].append(span)

    normalized_lines: list[list[dict[str, Any]]] = []
    for line in lines:
        normalized_lines.append(sorted(line, key=lambda span: (_span_x0(span), _span_center_y(span))))

    ordered_lines = sorted(
        normalized_lines,
        key=lambda line: (_line_center_y(line), min(_span_x0(span) for span in line)),
    )
    return _merge_stacked_math_visual_lines(ordered_lines)


def _is_deferred_small_line_token(
    span: dict[str, Any],
    body_size: float,
) -> bool:
    text = _span_text(span).strip()
    if not text or not body_size or _is_mark_token(text):
        return False
    return (
        len(text) <= 3
        and 0 < float(span.get("size", 0)) <= body_size * 0.82
    )


def _serialize_table_visual_lines(
    lines: list[list[dict[str, Any]]],
) -> list[tuple[list[dict[str, Any]], str | None]]:
    """Serialize compact PDF tables while their cell coordinates still exist.

    Native PDF text contains no row or cell delimiters.  CAIE question prose
    reliably introduces these compact tables with the word ``table`` and the
    cell centres then repeat across adjacent visual rows.  Use both signals so
    ordinary equations and aligned prose remain untouched.
    """

    line_texts = [_line_text_from_spans(line).strip() for line in lines]
    replacements: dict[int, tuple[list[dict[str, Any]], str]] = {}
    skipped: set[int] = set()

    for anchor_index, anchor_text in enumerate(line_texts):
        prompt_kind = _table_prompt_kind(line_texts, anchor_index)
        if prompt_kind is None:
            continue
        if prompt_kind == "stem_leaf":
            stem_leaf = _stem_leaf_display_serialization(
                lines,
                line_texts,
                anchor_index,
            )
            if stem_leaf is not None:
                display_indices, display_spans, display_text = stem_leaf
                replacements[anchor_index] = (
                    [*lines[anchor_index], *display_spans],
                    f"{anchor_text} {display_text}",
                )
                skipped.update(display_indices)
            continue
        first_index = anchor_index + 1
        while (
            first_index < min(len(lines), anchor_index + 3)
            and len(_split_table_line_cells(lines[first_index])) < 2
            and re.match(r"^[a-z]", line_texts[first_index])
        ):
            first_index += 1
        if first_index >= len(lines) or first_index in skipped:
            continue

        anchor_bbox = _line_bbox_from_spans(lines[anchor_index])
        first_bbox = _line_bbox_from_spans(lines[first_index])
        if first_bbox[1] - anchor_bbox[3] > 36:
            continue

        physical_rows: list[
            tuple[int, list[dict[str, Any]], list[list[dict[str, Any]]]]
        ] = []
        previous_bbox: tuple[float, float, float, float] | None = None
        for index in range(first_index, min(len(lines), first_index + 8)):
            if index in skipped or index in replacements:
                break
            line = lines[index]
            cells = _split_table_line_cells(line)
            text = line_texts[index]
            if not _is_table_row_candidate(text, cells):
                break
            bbox = _line_bbox_from_spans(line)
            if previous_bbox is not None:
                row_size = max(1.0, _line_median_font_size(line))
                if bbox[1] - previous_bbox[3] > max(14.0, row_size * 1.3):
                    break
            physical_rows.append((index, line, cells))
            previous_bbox = bbox

        if not _is_geometric_table_run(physical_rows):
            continue
        if prompt_kind == "raw_data" and not _is_raw_data_table_run(physical_rows):
            continue

        table_rows, wrapped_header = _logical_table_rows(physical_rows)
        if prompt_kind == "raw_data":
            if any(re.search(r"[A-Za-z]", row[2][0][1]) for row in table_rows):
                table_rows = _merge_labeled_raw_data_rows(table_rows)
                for indices, row_spans, cells in table_rows:
                    cell_texts = [text for _cell_spans, text in cells if text]
                    replacements[indices[0]] = (
                        row_spans,
                        _punctuated_table_row(cell_texts, has_label=True),
                    )
                    skipped.update(indices[1:])
            else:
                cell_texts = [
                    text
                    for _indices, _row_spans, cells in table_rows
                    for _cell_spans, text in cells
                    if text
                ]
                row_indices = [index for indices, _spans, _cells in table_rows for index in indices]
                row_spans = [span for _indices, spans, _cells in table_rows for span in spans]
                replacements[anchor_index] = (
                    [*lines[anchor_index], *row_spans],
                    f"{anchor_text} {' '.join(cell_texts)}",
                )
                skipped.update(row_indices)
            continue

        max_cell_count = max(len(row[2]) for row in table_rows)
        for row_position, (indices, row_spans, cells) in enumerate(table_rows):
            cell_texts = [text for _cell_spans, text in cells if text]
            if len(cell_texts) < 2:
                continue
            if wrapped_header:
                serialized = " ".join(cell_texts)
            else:
                first_row_has_label = len(cell_texts) == max_cell_count
                has_label = row_position > 0 or first_row_has_label
                serialized = _punctuated_table_row(cell_texts, has_label=has_label)
            replacements[indices[0]] = (row_spans, serialized)
            skipped.update(indices[1:])

    serialized_lines: list[tuple[list[dict[str, Any]], str | None]] = []
    for index, line in enumerate(lines):
        if index in skipped:
            continue
        replacement = replacements.get(index)
        if replacement is None:
            serialized_lines.append((line, None))
        else:
            serialized_lines.append(replacement)
    return serialized_lines


def _table_prompt_kind(line_texts: list[str], index: int) -> str | None:
    text = line_texts[index]
    if re.search(r"\bstem-and-leaf diagram\b", text, re.IGNORECASE):
        return "stem_leaf"
    if re.search(r"\btable\b", text, re.IGNORECASE):
        return "table"
    if re.search(r"\bshown below\b", text, re.IGNORECASE):
        return "raw_data"
    if (
        index > 0
        and re.fullmatch(r"below[.:]?", text, re.IGNORECASE)
        and re.search(r"\bshown\s*$", line_texts[index - 1], re.IGNORECASE)
    ):
        return "raw_data"
    return None


def _stem_leaf_display_serialization(
    lines: list[list[dict[str, Any]]],
    line_texts: list[str],
    anchor_index: int,
) -> tuple[list[int], list[dict[str, Any]], str] | None:
    first_index = anchor_index + 1
    if first_index >= len(lines):
        return None
    key_index = next(
        (
            index
            for index in range(first_index, min(len(lines), first_index + 12))
            if re.search(r"\bKey\s*:", line_texts[index], re.IGNORECASE)
        ),
        None,
    )
    if key_index is None:
        return None

    header_spans = [
        span
        for span in lines[first_index]
        if re.search(r"[A-Za-z]", _span_text(span))
    ]
    if len(header_spans) != 2:
        return None
    header_spans.sort(key=_span_x0)
    left_header, right_header = header_spans
    left_bbox = _line_bbox_from_spans([left_header])
    right_bbox = _line_bbox_from_spans([right_header])
    stem_x = (left_bbox[2] + right_bbox[0]) / 2

    display_indices = list(range(first_index, key_index + 1))
    display_spans = [span for index in display_indices for span in lines[index]]
    key_marker = next(
        (
            span
            for span in lines[key_index]
            if re.match(r"\s*Key\s*:", _span_text(span), re.IGNORECASE)
        ),
        None,
    )
    if key_marker is None:
        return None
    key_y = _span_center_y(key_marker)
    numeric_spans = [
        span
        for span in display_spans
        if _span_text(span).strip().isdigit()
        and _span_center_y(span) < key_y - 6.0
    ]
    numeric_rows = _spans_grouped_by_baseline(numeric_spans, tolerance=3.0)
    if len(numeric_rows) < 3:
        return None

    serialized_rows: list[str] = []
    for row in numeric_rows:
        ordered = sorted(row, key=_span_x0)
        stem_index = min(
            range(len(ordered)),
            key=lambda index: abs(_table_cell_center_x([ordered[index]]) - stem_x),
        )
        left_values = [_span_text(span).strip() for span in ordered[:stem_index]]
        stem_value = _span_text(ordered[stem_index]).strip()
        right_values = [_span_text(span).strip() for span in ordered[stem_index + 1 :]]
        if not left_values or not right_values:
            return None
        serialized_rows.append(
            f"{' '.join(left_values)} | {stem_value} | {' '.join(right_values)}"
        )

    key_spans = [
        span
        for span in lines[key_index]
        if _span_center_y(span) >= key_y - 5.0
    ]
    key_text = _line_text_from_spans(key_spans).strip()
    key_text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", " | ", key_text)
    key_text = key_text.rstrip(" .") + "."
    header_text = (
        f"{_line_text_from_spans([left_header]).strip()} | Stem | "
        f"{_line_text_from_spans([right_header]).strip()}:"
    )
    display_text = f"{header_text} {'; '.join(serialized_rows)}. {key_text}"
    return display_indices, display_spans, display_text


def _spans_grouped_by_baseline(
    spans: list[dict[str, Any]],
    *,
    tolerance: float,
) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for span in sorted(spans, key=lambda item: (_span_center_y(item), _span_x0(item))):
        if not rows or abs(_span_center_y(span) - _line_center_y(rows[-1])) > tolerance:
            rows.append([span])
        else:
            rows[-1].append(span)
    return rows


def _split_table_line_cells(
    line: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    spans = sorted(
        [span for span in line if _span_text(span).strip()],
        key=lambda span: (_span_x0(span), _span_center_y(span)),
    )
    if not spans:
        return []
    font_sizes = [
        float(span.get("size", 0))
        for span in spans
        if float(span.get("size", 0)) > 0
    ]
    split_gap = max(7.0, (median(font_sizes) if font_sizes else 10.0) * 0.72)
    cells: list[list[dict[str, Any]]] = []
    for span in spans:
        if not cells:
            cells.append([span])
            continue
        previous_x1 = max(
            float(item.get("bbox", [0, 0, 0, 0])[2])
            for item in cells[-1]
        )
        if _span_x0(span) - previous_x1 > split_gap:
            cells.append([span])
        else:
            cells[-1].append(span)
    return cells


def _is_table_row_candidate(
    text: str,
    cells: list[list[dict[str, Any]]],
) -> bool:
    if len(cells) < 2 or re.search(r"\[\d{1,2}\]", text):
        return False
    if re.match(r"^\s*\((?:[a-z]|[ivx]+)\)\s+", text, re.IGNORECASE):
        return False
    cell_texts = [_line_text_from_spans(cell).strip() for cell in cells]
    return all(cell_texts) and all(len(value) <= 36 for value in cell_texts[1:])


def _is_geometric_table_run(
    rows: list[tuple[int, list[dict[str, Any]], list[list[dict[str, Any]]]]],
) -> bool:
    if len(rows) < 2 or max(len(cells) for _index, _line, cells in rows) < 3:
        return False
    return any(
        len(_aligned_table_cell_pairs(upper_cells, lower_cells)) >= 2
        for (_upper_index, _upper_line, upper_cells),
        (_lower_index, _lower_line, lower_cells) in zip(rows, rows[1:])
    )


def _is_raw_data_table_run(
    rows: list[tuple[int, list[dict[str, Any]], list[list[dict[str, Any]]]]],
) -> bool:
    number_re = re.compile(r"^-?\d+(?:\.\d+)?$")
    for _index, _line, cells in rows:
        values = [_normalize_table_cell_text(_table_cell_text(cell)) for cell in cells]
        if all(number_re.fullmatch(value) for value in values):
            continue
        if re.search(r"[A-Za-z]", values[0]) and all(
            number_re.fullmatch(value) for value in values[1:]
        ):
            continue
        return False
    return True


def _aligned_table_cell_pairs(
    upper_cells: list[list[dict[str, Any]]],
    lower_cells: list[list[dict[str, Any]]],
) -> list[tuple[int, int]]:
    tolerance = 8.0
    upper_centres = [_table_cell_center_x(cell) for cell in upper_cells]
    lower_centres = [_table_cell_center_x(cell) for cell in lower_cells]
    available = set(range(len(upper_cells)))
    pairs: list[tuple[int, int]] = []
    for lower_index, lower_centre in enumerate(lower_centres):
        matches = [
            (abs(upper_centres[upper_index] - lower_centre), upper_index)
            for upper_index in available
            if abs(upper_centres[upper_index] - lower_centre) <= tolerance
        ]
        if not matches:
            continue
        _distance, upper_index = min(matches)
        available.remove(upper_index)
        pairs.append((upper_index, lower_index))
    return pairs


def _table_cell_center_x(cell: list[dict[str, Any]]) -> float:
    bbox = _line_bbox_from_spans(cell)
    return (bbox[0] + bbox[2]) / 2


def _logical_table_rows(
    physical_rows: list[
        tuple[int, list[dict[str, Any]], list[list[dict[str, Any]]]]
    ],
) -> tuple[
    list[
        tuple[
            list[int],
            list[dict[str, Any]],
            list[tuple[list[dict[str, Any]], str]],
        ]
    ],
    bool,
]:
    rows: list[
        tuple[
            list[int],
            list[dict[str, Any]],
            list[tuple[list[dict[str, Any]], str]],
        ]
    ] = []
    index = 0
    while index < len(physical_rows):
        physical_index, line_spans, raw_cells = physical_rows[index]
        cells = [
            (cell, _normalize_table_cell_text(_table_cell_text(cell)))
            for cell in raw_cells
        ]
        indices = [physical_index]
        combined_spans = list(line_spans)
        if index + 1 < len(physical_rows):
            lower_index, lower_spans, lower_cells = physical_rows[index + 1]
            if _is_stacked_table_fraction_continuation(raw_cells, lower_cells):
                for upper_cell_index, lower_cell_index in _aligned_table_cell_pairs(
                    raw_cells,
                    lower_cells,
                ):
                    merged_cell_spans = [
                        *cells[upper_cell_index][0],
                        *lower_cells[lower_cell_index],
                    ]
                    cells[upper_cell_index] = (
                        merged_cell_spans,
                        _normalize_table_cell_text(
                            _line_text_from_spans(merged_cell_spans)
                        ),
                    )
                indices.append(lower_index)
                combined_spans.extend(lower_spans)
                index += 1
        rows.append((indices, combined_spans, cells))
        index += 1

    wrapped_header = len(rows) >= 3 and _is_wrapped_table_header(rows[0], rows[1])
    if wrapped_header:
        upper_indices, upper_spans, upper_cells = rows[0]
        lower_indices, lower_spans, lower_cells = rows[1]
        raw_upper_cells = [cell_spans for cell_spans, _text in upper_cells]
        raw_lower_cells = [cell_spans for cell_spans, _text in lower_cells]
        for upper_cell_index, lower_cell_index in _aligned_table_cell_pairs(
            raw_upper_cells,
            raw_lower_cells,
        ):
            upper_cell_spans, upper_text = upper_cells[upper_cell_index]
            lower_cell_spans, lower_text = lower_cells[lower_cell_index]
            upper_cells[upper_cell_index] = (
                [*upper_cell_spans, *lower_cell_spans],
                f"{upper_text} {lower_text}".strip(),
            )
        rows[0] = (
            [*upper_indices, *lower_indices],
            [*upper_spans, *lower_spans],
            upper_cells,
        )
        del rows[1]
    return rows, wrapped_header


def _merge_labeled_raw_data_rows(
    rows: list[
        tuple[
            list[int],
            list[dict[str, Any]],
            list[tuple[list[dict[str, Any]], str]],
        ]
    ],
) -> list[
    tuple[
        list[int],
        list[dict[str, Any]],
        list[tuple[list[dict[str, Any]], str]],
    ]
]:
    merged: list[
        tuple[
            list[int],
            list[dict[str, Any]],
            list[tuple[list[dict[str, Any]], str]],
        ]
    ] = []
    for indices, row_spans, cells in rows:
        if re.search(r"[A-Za-z]", cells[0][1]) or not merged:
            merged.append((list(indices), list(row_spans), list(cells)))
            continue
        prior_indices, prior_spans, prior_cells = merged[-1]
        merged[-1] = (
            [*prior_indices, *indices],
            [*prior_spans, *row_spans],
            [*prior_cells, *cells],
        )
    return merged


def _is_stacked_table_fraction_continuation(
    upper_cells: list[list[dict[str, Any]]],
    lower_cells: list[list[dict[str, Any]]],
) -> bool:
    if len(lower_cells) < 2 or len(lower_cells) >= len(upper_cells):
        return False
    pairs = _aligned_table_cell_pairs(upper_cells, lower_cells)
    if len(pairs) != len(lower_cells):
        return False
    for upper_index, lower_index in pairs:
        upper_text = _line_text_from_spans(upper_cells[upper_index]).strip()
        lower_text = _line_text_from_spans(lower_cells[lower_index]).strip()
        if not (upper_text.isdigit() and lower_text.isdigit()):
            return False
        upper_bbox = _line_bbox_from_spans(upper_cells[upper_index])
        lower_bbox = _line_bbox_from_spans(lower_cells[lower_index])
        if lower_bbox[1] - upper_bbox[3] > 4.5:
            return False
    return True


def _is_wrapped_table_header(
    upper: tuple[
        list[int],
        list[dict[str, Any]],
        list[tuple[list[dict[str, Any]], str]],
    ],
    lower: tuple[
        list[int],
        list[dict[str, Any]],
        list[tuple[list[dict[str, Any]], str]],
    ],
) -> bool:
    upper_cells = upper[2]
    lower_cells = lower[2]
    if len(lower_cells) < 2 or len(lower_cells) > len(upper_cells):
        return False
    all_text = [text for _spans, text in [*upper_cells, *lower_cells]]
    if any(re.search(r"\d|[=<>≤≥]", text) for text in all_text):
        return False
    upper_bbox = _line_bbox_from_spans(upper[1])
    lower_bbox = _line_bbox_from_spans(lower[1])
    if lower_bbox[1] - upper_bbox[3] > 3.5:
        return False
    pairs = _aligned_table_cell_pairs(
        [spans for spans, _text in upper_cells],
        [spans for spans, _text in lower_cells],
    )
    return len(pairs) == len(lower_cells)


def _normalize_table_cell_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip().rstrip(" ,.;:")
    return re.sub(r"(?<=\d)\s*[−–—-]\s*(?=\d)", "-", value)


def _table_cell_text(cell: list[dict[str, Any]]) -> str:
    nonempty = [span for span in cell if _span_text(span).strip()]
    if len(nonempty) == 2 and all(_span_text(span).strip().isdigit() for span in nonempty):
        upper, lower = sorted(nonempty, key=_span_center_y)
        upper_bbox = _line_bbox_from_spans([upper])
        lower_bbox = _line_bbox_from_spans([lower])
        overlap = max(
            0.0,
            min(upper_bbox[2], lower_bbox[2])
            - max(upper_bbox[0], lower_bbox[0]),
        )
        narrower = max(
            0.1,
            min(upper_bbox[2] - upper_bbox[0], lower_bbox[2] - lower_bbox[0]),
        )
        if (
            overlap / narrower >= 0.7
            and _span_center_y(lower) - _span_center_y(upper) >= 3.0
        ):
            return f"({_span_text(upper).strip()})/({_span_text(lower).strip()})"
    return _line_text_from_spans(cell)


def _punctuated_table_row(cell_texts: list[str], *, has_label: bool) -> str:
    if not has_label:
        return f"{', '.join(cell_texts)}."
    label, *values = cell_texts
    return f"{label}: {', '.join(values)}."


def _merge_stacked_math_visual_lines(
    lines: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Coalesce overlapping formula rows without joining prose baselines.

    Fractions and integral bounds are commonly emitted as two or three PDF
    lines whose boxes overlap the surrounding prose baseline.  Treating every
    y cluster as a separate reading line puts denominators after the sentence;
    treating any overlap as line membership lets a tall math glyph bridge an
    entire paragraph.  This bounded second pass merges only adjacent,
    vertically overlapping rows when at least one row is math-only.
    """

    merged: list[list[dict[str, Any]]] = []
    for line in lines:
        if merged and _should_merge_stacked_math_rows(merged[-1], line):
            merged[-1] = sorted(
                [*merged[-1], *line],
                key=lambda span: (_span_x0(span), _span_center_y(span)),
            )
        else:
            merged.append(line)
    return merged


def _should_merge_stacked_math_rows(
    upper: list[dict[str, Any]],
    lower: list[dict[str, Any]],
) -> bool:
    upper_bbox = _line_bbox_from_spans(upper)
    lower_bbox = _line_bbox_from_spans(lower)
    upper_height = max(0.1, upper_bbox[3] - upper_bbox[1])
    lower_height = max(0.1, lower_bbox[3] - lower_bbox[1])
    vertical_overlap = max(
        0.0,
        min(upper_bbox[3], lower_bbox[3]) - max(upper_bbox[1], lower_bbox[1]),
    )
    vertical_gap = max(0.0, lower_bbox[1] - upper_bbox[3])
    close_enough = (
        vertical_overlap / min(upper_height, lower_height) >= 0.12
        or vertical_gap <= 0.75
    )
    if not close_enough:
        return False

    horizontal_overlap = max(
        0.0,
        min(upper_bbox[2], lower_bbox[2]) - max(upper_bbox[0], lower_bbox[0]),
    )
    if horizontal_overlap <= 0:
        return False

    upper_prose = _line_prose_word_count(upper)
    lower_prose = _line_prose_word_count(lower)
    derivative_equation_rows = _rows_have_derivative_equation_text(upper, lower)
    if derivative_equation_rows:
        if _row_has_relation_symbol(lower):
            return False
        if _derivative_tokens_are_horizontally_aligned(upper, lower):
            return True
    if upper_prose and lower_prose:
        return False
    if (
        _row_is_standalone_math_statement(upper)
        or _row_is_standalone_math_statement(lower)
    ):
        return False
    return True


def _rows_have_derivative_equation_text(
    upper: list[dict[str, Any]],
    lower: list[dict[str, Any]],
) -> bool:
    upper_text = _flat_line_text_from_spans(upper).strip()
    lower_text = _flat_line_text_from_spans(lower).strip()
    return bool(
        re.search(
            r"(?:^|[^A-Za-z])[A-Za-zθ]?d[A-Za-zθ]\s*=",
            upper_text,
        )
        and re.match(r"d[A-Za-zθ](?:\s|$)", lower_text)
    )


def _derivative_tokens_are_horizontally_aligned(
    upper: list[dict[str, Any]],
    lower: list[dict[str, Any]],
) -> bool:
    upper_derivatives = _derivative_token_bboxes(upper)
    lower_derivatives = _derivative_token_bboxes(lower)
    for upper_bbox in upper_derivatives:
        for lower_bbox in lower_derivatives:
            overlap = max(
                0.0,
                min(upper_bbox[2], lower_bbox[2])
                - max(upper_bbox[0], lower_bbox[0]),
            )
            narrower = max(
                0.1,
                min(
                    upper_bbox[2] - upper_bbox[0],
                    lower_bbox[2] - lower_bbox[0],
                ),
            )
            if overlap / narrower >= 0.6:
                return True
    return False


def _derivative_token_bboxes(
    line: list[dict[str, Any]],
) -> list[tuple[float, float, float, float]]:
    spans = sorted(
        [span for span in line if _span_text(span).strip()],
        key=lambda span: (_span_x0(span), _span_center_y(span)),
    )
    boxes: list[tuple[float, float, float, float]] = []
    for index, current in enumerate(spans):
        current_text = "".join(_span_text(current).split())
        if re.fullmatch(r"d[A-Za-zθ]", current_text):
            boxes.append(_line_bbox_from_spans([current]))
            continue
        if current_text != "d" or index + 1 >= len(spans):
            continue

        following = spans[index + 1]
        following_text = "".join(_span_text(following).split())
        if not re.fullmatch(r"[A-Za-zθ]", following_text):
            continue
        max_size = max(
            float(current.get("size", 0)),
            float(following.get("size", 0)),
            1.0,
        )
        horizontal_gap = _span_x0(following) - float(
            current.get("bbox", [0, 0, 0, 0])[2]
        )
        if horizontal_gap > max(2.0, max_size * 0.35):
            continue
        center_delta = abs(_span_center_y(current) - _span_center_y(following))
        if center_delta > max(1.0, max_size * 0.2):
            continue
        boxes.append(_line_bbox_from_spans([current, following]))
    return boxes


def _row_has_relation_symbol(line: list[dict[str, Any]]) -> bool:
    return any(re.search(r"[=≡≤≥<>]", _span_text(span)) for span in line)


def _line_prose_word_count(line: list[dict[str, Any]]) -> int:
    return sum(_span_prose_word_count(_span_text(span)) for span in line)


def _row_is_standalone_math_statement(line: list[dict[str, Any]]) -> bool:
    return not _line_prose_word_count(line) and _row_has_relation_symbol(line)


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

        # Bounding boxes from tall math fonts often overlap the next printed
        # baseline even though their text does not.  Center distance is the
        # stable line-membership signal; using overlap as an override permits
        # one tall span to bridge an entire paragraph into a single line.
        if distance > tolerance:
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
    spans = _integral_composite_spans(spans)
    stacked_text = _stacked_fraction_line_text(spans)
    if stacked_text is not None:
        return _repair_line_spacing(stacked_text)
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
        delimiter_gap = _needs_delimiter_spacing(previous_text, text) and gap > 1.5
        integral_gap = "∫" in previous_text and text.lstrip()[:1].isalnum() and gap > 0.5
        threshold = max(2.0, float(span.get("size", max_size or 1)) * 0.35)
        if previous_x1 is not None and (
            operator_gap
            or delimiter_gap
            or integral_gap
            or gap > threshold
        ):
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
        adjacent_small_star = (
            normalized == "*"
            and bool(max_size)
            and bool(median_size)
            and size <= max_size * 0.85
            and previous_span is not None
            and bool(re.fullmatch(r"[A-Za-z]", previous_text_normalized))
            and previous_gap <= max(1.0, median_size * 0.15)
        )
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
                adjacent_small_star
                or vertical_shift >= max(1.0, median_size * 0.08)
                or baseline_shift_from_previous >= max(1.0, median_size * 0.15)
            )
        )
        if is_script_candidate:
            if adjacent_small_star:
                pieces.append(f"^{{{text}}}")
                previous_span = span
                previous_x1 = x1
                previous_text = text
                continue
            script_threshold = max(0.6, median_size * 0.04)
            script_reference_mid = (
                line_mid
                if previous_text_normalized in {")", "]", "}"}
                else previous_mid
            )
            if span_mid < script_reference_mid - script_threshold:
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
    all_spans = [span for span in spans if _span_text(span)]
    spans = [span for span in all_spans if _span_text(span).strip()]
    if len(spans) < 2:
        return None

    font_sizes = [float(span.get("size", 0)) for span in spans if _span_text(span).strip()]
    median_size = median(font_sizes) if font_sizes else 0
    if median_size <= 0:
        return None

    baseline = _math_baseline_y(spans, median_size)
    row_offset = max(0.8, median_size * 0.07)
    top_spans = [
        span
        for span in spans
        if _span_center_y(span) < baseline - row_offset
        and _is_fraction_row_span_candidate(span, median_size)
    ]
    bottom_spans = [
        span
        for span in spans
        if _span_center_y(span) > baseline + row_offset
        and _is_fraction_row_span_candidate(span, median_size)
    ]
    top_groups = _fraction_horizontal_groups(top_spans, median_size)
    bottom_groups = _fraction_horizontal_groups(bottom_spans, median_size)
    pairs = _matched_fraction_groups(
        top_groups,
        bottom_groups,
        median_size,
        context_spans=spans,
    )
    used_ids = {id(span) for pair in pairs for group in pair for span in group}
    baseline_spans = [
        span
        for span in spans
        if abs(_span_center_y(span) - baseline) <= row_offset
        and _is_fraction_row_span_candidate(span, median_size)
        and id(span) not in used_ids
    ]
    baseline_groups = _fraction_horizontal_groups(baseline_spans, median_size)
    remaining_top = [group for group in top_groups if not any(id(span) in used_ids for span in group)]
    remaining_bottom = [group for group in bottom_groups if not any(id(span) in used_ids for span in group)]
    one_sided_pairs = [
        *_matched_fraction_groups(
            remaining_top,
            baseline_groups,
            median_size,
            context_spans=spans,
        ),
        *_matched_fraction_groups(
            baseline_groups,
            remaining_bottom,
            median_size,
            context_spans=spans,
        ),
    ]
    for pair in one_sided_pairs:
        pair_ids = {id(span) for group in pair for span in group}
        if pair_ids & used_ids:
            continue
        pairs.append(pair)
        used_ids.update(pair_ids)
    if not pairs:
        return None

    consumed_ids = {
        id(span)
        for top_group, bottom_group in pairs
        for span in [*top_group, *bottom_group]
    }
    segments: list[dict[str, Any]] = []
    fraction_boxes: list[tuple[float, float, float, float]] = []
    for top_group, bottom_group in pairs:
        top_text = _flat_line_text_from_spans(top_group).strip()
        bottom_text = _flat_line_text_from_spans(bottom_group).strip()
        if not top_text or not bottom_text:
            continue
        x0, y0, x1, y1 = _line_bbox_from_spans([*top_group, *bottom_group])
        fraction_boxes.append((x0, y0, x1, y1))
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
    segments.extend(
        span
        for span in all_spans
        if id(span) not in consumed_ids
        and not _space_inside_fraction_box(span, fraction_boxes)
    )
    return _flat_line_text_from_spans(segments)


def _math_baseline_y(spans: list[dict[str, Any]], median_size: float) -> float:
    anchors: list[float] = []
    for span in spans:
        text = _span_text(span).strip()
        if not text:
            continue
        if (
            _span_prose_word_count(text) >= 1
            or _is_mark_token(text)
            or re.search(r"(?:^|\s)(?:=|≡|≤|≥|<|>)(?:\s|$)", text)
        ):
            anchors.append(_span_center_y(span))
    if anchors:
        return median(anchors)

    largest = [
        _span_center_y(span)
        for span in spans
        if float(span.get("size", 0)) >= median_size * 1.15
    ]
    if largest:
        return median(largest)
    return median(_span_center_y(span) for span in spans)


def _fraction_horizontal_groups(
    spans: list[dict[str, Any]],
    median_size: float,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    max_gap = max(2.0, median_size * 0.45)
    max_center_delta = max(2.0, median_size * 0.55)
    for span in sorted(spans, key=lambda item: (_span_x0(item), _span_center_y(item))):
        if not groups:
            groups.append([span])
            continue
        previous = groups[-1]
        previous_bbox = _line_bbox_from_spans(previous)
        gap = _span_x0(span) - previous_bbox[2]
        if gap <= max_gap and abs(_span_center_y(span) - _line_center_y(previous)) <= max_center_delta:
            previous.append(span)
        else:
            groups.append([span])
    return groups


def _matched_fraction_groups(
    top_groups: list[list[dict[str, Any]]],
    bottom_groups: list[list[dict[str, Any]]],
    median_size: float,
    *,
    context_spans: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    proposals: list[tuple[float, float, int, int]] = []
    for top_index, top_group in enumerate(top_groups):
        top_bbox = _line_bbox_from_spans(top_group)
        for bottom_index, bottom_group in enumerate(bottom_groups):
            bottom_bbox = _line_bbox_from_spans(bottom_group)
            overlap = max(0.0, min(top_bbox[2], bottom_bbox[2]) - max(top_bbox[0], bottom_bbox[0]))
            narrower = max(0.1, min(top_bbox[2] - top_bbox[0], bottom_bbox[2] - bottom_bbox[0]))
            overlap_ratio = overlap / narrower
            if overlap_ratio < 0.45:
                continue
            center_delta = abs(
                ((top_bbox[0] + top_bbox[2]) / 2)
                - ((bottom_bbox[0] + bottom_bbox[2]) / 2)
            )
            if center_delta > max(median_size * 0.85, narrower * 0.5):
                continue
            vertical_distance = abs(_line_center_y(top_group) - _line_center_y(bottom_group))
            proposals.append((vertical_distance, -overlap_ratio, top_index, bottom_index))

    used_top: set[int] = set()
    used_bottom: set[int] = set()
    pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for _distance, _negative_overlap, top_index, bottom_index in sorted(proposals):
        if top_index in used_top or bottom_index in used_bottom:
            continue
        top_group = top_groups[top_index]
        bottom_group = bottom_groups[bottom_index]
        top_text = _flat_line_text_from_spans(top_group).strip()
        bottom_text = _flat_line_text_from_spans(bottom_group).strip()
        if _looks_like_fraction_prose_text(top_text):
            continue
        if _looks_like_fraction_prose_text(bottom_text):
            continue
        if _looks_like_incomplete_derivative_fraction_pair(top_text, bottom_text):
            continue
        if (
            top_text.isdigit()
            and bottom_text.isdigit()
            and len(top_text) == 1
            and len(bottom_text) == 1
            and not _digit_fraction_has_context(
                top_group,
                bottom_group,
                context_spans,
                median_size,
            )
        ):
            continue
        used_top.add(top_index)
        used_bottom.add(bottom_index)
        pairs.append((top_group, bottom_group))
    return pairs


def _looks_like_incomplete_derivative_fraction_pair(
    top_text: str,
    bottom_text: str,
) -> bool:
    compact_top = "".join(top_text.split())
    compact_bottom = "".join(bottom_text.split())
    return bool(
        re.fullmatch(r"d[A-Za-zθ]", compact_top)
        and compact_bottom == "d"
    )


def _digit_fraction_has_context(
    top_group: list[dict[str, Any]],
    bottom_group: list[dict[str, Any]],
    context_spans: list[dict[str, Any]],
    median_size: float,
) -> bool:
    x0, _y0, x1, _y1 = _line_bbox_from_spans([*top_group, *bottom_group])
    member_ids = {id(span) for span in [*top_group, *bottom_group]}
    max_gap = max(4.0, median_size * 1.2)
    for span in context_spans:
        if id(span) in member_ids:
            continue
        text = _span_text(span).strip()
        if not text:
            continue
        sx0, _sy0, sx1, _sy1 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
        if 0 <= x0 - sx1 <= max_gap and re.search(r"(?:=|≡|∫|ln)\s*$", text):
            return True
        if 0 <= sx0 - x1 <= max_gap and re.match(r"(?:π|ln)\b", text):
            return True
    return False


def _space_inside_fraction_box(
    span: dict[str, Any],
    boxes: list[tuple[float, float, float, float]],
) -> bool:
    if _span_text(span).strip():
        return False
    x0, y0, x1, y1 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
    width = max(0.1, x1 - x0)
    return any(
        max(0.0, min(x1, bx1) - max(x0, bx0)) / width >= 0.4
        and max(0.0, min(y1, by1) - max(y0, by0)) > 0
        for bx0, by0, bx1, by1 in boxes
    )


def _integral_composite_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nonempty = [span for span in spans if _span_text(span).strip()]
    if not nonempty:
        return spans
    font_sizes = [float(span.get("size", 0)) for span in nonempty if float(span.get("size", 0)) > 0]
    median_size = median(font_sizes) if font_sizes else 0
    if median_size <= 0:
        return spans
    baseline = _math_baseline_y(nonempty, median_size)
    normalized_spans = [_contextual_integral_span(span, nonempty, baseline, median_size) for span in spans]

    consumed_ids: set[int] = set()
    replacements: list[dict[str, Any]] = []
    for integral in normalized_spans:
        if "∫" not in _span_text(integral) or id(integral) in consumed_ids:
            continue
        ix0, iy0, ix1, iy1 = [float(value) for value in integral.get("bbox", [0, 0, 0, 0])]
        bound_right = ix1 + max(8.0, median_size * 1.1)
        candidates = [
            span
            for span in normalized_spans
            if span is not integral
            and _span_text(span).strip()
            and _span_x0(span) >= ix0 - 0.5
            and float(span.get("bbox", [0, 0, 0, 0])[2]) <= bound_right
        ]
        upper = [span for span in candidates if _span_center_y(span) < baseline - max(1.0, median_size * 0.08)]
        lower = [span for span in candidates if _span_center_y(span) > baseline + max(1.0, median_size * 0.08)]
        if not upper and not lower:
            continue

        upper_text = _integral_bound_text(upper)
        lower_text = _integral_bound_text(lower)
        leading_space = " " if _span_text(integral)[:1].isspace() else ""
        text = leading_space + "∫"
        if lower_text:
            text += f"_{{{lower_text}}}"
        if upper_text:
            text += f"^{{{upper_text}}}"
        consumed = [integral, *upper, *lower]
        consumed_ids.update(id(span) for span in consumed)
        x0, y0, x1, y1 = _line_bbox_from_spans(consumed)
        replacements.append(
            {
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "size": median_size,
                "font": integral.get("font", ""),
            }
        )

    if not replacements:
        return normalized_spans
    return [*replacements, *(span for span in normalized_spans if id(span) not in consumed_ids)]


def _contextual_integral_span(
    span: dict[str, Any],
    all_spans: list[dict[str, Any]],
    baseline: float,
    median_size: float,
) -> dict[str, Any]:
    text = _span_text(span).strip()
    if text not in {"y", "Ó", "Ô"}:
        return span
    x0, y0, x1, y1 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
    if y1 - y0 < median_size * 1.65 or abs(_span_center_y(span) - baseline) > median_size:
        return span
    has_lower_bound = any(
        _span_center_y(item) > baseline + 1
        and _span_x0(item) >= x0 - 1
        and _span_x0(item) <= x1 + median_size
        and _span_text(item).strip().isalnum()
        for item in all_spans
        if item is not span
    )
    if not has_lower_bound:
        return span
    normalized = dict(span)
    normalized["text"] = "∫"
    return normalized


def _integral_bound_text(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return ""
    normalized_spans: list[dict[str, Any]] = []
    for span in spans:
        normalized = dict(span)
        raw_text = str(normalized.get("text", ""))
        if raw_text.strip() == "r":
            normalized["text"] = raw_text.replace("r", "π")
        normalized_spans.append(normalized)
    text = _line_text_from_spans(normalized_spans).strip()
    # Some recent opaque CAIE fonts expose the visual pi glyph as ``r``.
    # Restrict the repair to an already identified integral-bound region.
    text = re.sub(r"(?<=[0-9)}])\s*r\b", "π", text)
    return text


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
    if re.fullmatch(r"[=≡≤≥<>]", text):
        return False
    if not re.search(r"[A-Za-z0-9θπ√≡=+\-−*/]", text):
        return False
    if _looks_like_fraction_prose_text(text):
        return False

    x0, y0, x1, y1 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
    height = max(0.0, y1 - y0)
    width = max(0.0, x1 - x0)
    tall_math_operator = bool(re.fullmatch(r"[+\-=−*/]", text.strip()))
    tall_math_glyph = bool(re.fullmatch(r"[α-ωΑ-Ωπθ∫Σ√]", text.strip()))
    if height > median_size * 1.4 and not (
        _is_control_parenthesis_text(text)
        or tall_math_operator
        or tall_math_glyph
    ):
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
        "Å": "°",
        "◦": "°",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _needs_operator_spacing(previous_text: str, text: str) -> bool:
    operators = {"+", "-", "−", "=", "≡", "≠", "<", ">", "≤", "≥", "±", "×"}
    return previous_text.strip() in operators or text.strip() in operators


def _needs_delimiter_spacing(previous_text: str, text: str) -> bool:
    previous = previous_text.strip()
    current = text.lstrip()
    if not current.startswith("("):
        return False
    final_word = re.search(r"([A-Za-z]{2,})$", previous)
    return bool(final_word and final_word.group(1).lower() not in _FRACTION_MATH_WORDS)


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
    value = value.replace("↦→", "↦").replace("↦ →", "↦")
    value = value.replace("∼", "~")
    if re.search(r"\b(?:class interval|rectangle)\b", value, re.IGNORECASE):
        value = re.sub(r"\b(\d+)\s*[−–—-]\s*(\d+)\b", r"\1-\2", value)
    value = re.sub(
        r"Σ\(([A-Za-z])\^\{([^}]+)\}\)/\(([FL])\)",
        r"Σ\1_{\3}^{\2}",
        value,
    )
    value = re.sub(r"\^\{[-−]\}\s*(\d+)", r"^{-\1}", value)
    value = re.sub(r"(?<=[A-Za-z0-9)}])\s+\^\{", "^{", value)
    value = re.sub(r"_\{([A-Za-z])\}\s*\+\s*([0-9]+)", r"_{\1+\2}", value)
    value = re.sub(r"(?<=\d)\^\{°\}", "°", value)
    value = re.sub(
        r"\b(ln|log)\(([^()]+)\)/\(([^()]+)\)",
        r"\1((\2)/(\3))",
        value,
    )
    value = re.sub(r"\b(ln|log)(?=[A-Za-z0-9(])", r"\1 ", value)
    value = re.sub(r"\b(ln|log)\s+\(", r"\1(", value)
    value = re.sub(
        r"\b(cosec|sin|cos|tan|sec|cot)\s+([0-9]+)\s+([A-Za-zθ])\b",
        r"\1 \2\3",
        value,
    )
    value = re.sub(r"\b(cosec|sin|cos|tan|sec|cot)(?=[0-9θxyz])", r"\1 ", value)
    value = re.sub(r"(?<=[A-Za-z0-9}])(?=(?:cosec|sin|cos|tan|sec|cot)(?:[0-9θxyz]|\b))", " ", value)
    value = re.sub(r"\b(cosec|sin|cos|tan|sec|cot)(?=[0-9θxyz])", r"\1 ", value)
    value = re.sub(
        r"\b(cosec|sin|cos|tan|sec|cot)\s+([0-9]+)\s+([A-Za-zθ])\b",
        r"\1 \2\3",
        value,
    )
    value = re.sub(r"(?<=\})(?=[A-Za-z]{2,}\b)", " ", value)
    value = re.sub(r"\b([A-Za-z]{2,})([A-Z][a-z]{2,})\b", r"\1 \2", value)
    value = re.sub(r"\b([fg])\s+-\s*1(?=\s*\()", r"\1^{-1}", value)
    value = re.sub(
        r"\b(cosec|sin|cos|tan|sec|cot)(\^\{[^}]+\})(?=[0-9A-Za-zθ])",
        r"\1\2 ",
        value,
    )
    value = re.sub(r"√(?!\()(\d+|[A-Za-z])", r"√(\1)", value)
    value = re.sub(r"(?<=°)(?=[A-Za-z])", " ", value)
    value = re.sub(r"(?<=\d)(?=(?:kg|km|cm|mm|kW|N|J|W)\b)", " ", value)
    value = re.sub(r"(?<=\d)(?=m(?!\s*kg\b)(?:\s|[.,;:)]|$))", " ", value)
    value = re.sub(r"\s+([,.;:?!])", r"\1", value)
    value = re.sub(r"([,;:])(?=\S)", r"\1 ", value)
    value = re.sub(r"\s+\)", ")", value)
    value = re.sub(r"([\[(])\s+", r"\1", value)
    value = re.sub(r"\b([fg])\s+\(", r"\1(", value)
    value = re.sub(r"(?<!\s)(?=\[\d{1,2}\])", " ", value)
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
    span_height = max(0.1, span_bottom - span_top)
    # Compare against real member spans, not the union of the whole line.  A
    # union becomes taller every time a nearby baseline is accepted; it can
    # then bridge successive printed lines and collapse an entire paragraph
    # into one x-sorted line.  That destroys word order even though the PDF's
    # individual spans are correct.  The strongest pairwise overlap retains
    # support for mixed fonts and attached scripts without that transitive
    # growth failure.
    return max(
        (
            max(
                0.0,
                min(span_bottom, float(item.get("bbox", [0, 0, 0, 0])[3]))
                - max(span_top, float(item.get("bbox", [0, 0, 0, 0])[1])),
            )
            / span_height
            for item in line
        ),
        default=0.0,
    )


def _extract_graphics(page: Any, *, legacy_fallback: bool = False) -> list[BoundingBox]:
    candidates: list[tuple[BoundingBox, str]] = []
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    drawings = page.get_drawings()
    drawing_rects: list[BoundingBox] = []

    for drawing in drawings:
        rect = drawing.get("rect")
        if rect and rect.is_valid:
            box = _visual_box_from_rect(page, rect)
            drawing_rects.append(box)
            if not rect.is_empty and _is_meaningful_graphic_box(box, page_width, page_height):
                candidates.append((box, "vector_graphic"))
            elif not rect.is_empty and legacy_fallback and _is_low_confidence_legacy_graphic_box(
                box, page_width, page_height
            ):
                candidates.append((box, "legacy_low_confidence_vector"))
        if legacy_fallback:
            item_boxes = _drawing_item_boxes(page, drawing)
            cluster = _dense_non_text_cluster(item_boxes, page_width=page_width, page_height=page_height)
            if cluster is not None:
                candidates.append((cluster, "legacy_dense_non_text_cluster"))

    candidates.extend(
        (grid_box, "dense_grid_graphic")
        for grid_box in _dense_grid_graphic_boxes(
            drawing_rects,
            page_width=page_width,
            page_height=page_height,
        )
    )

    try:
        image_infos = page.get_image_info(xrefs=True)
    except Exception:
        image_infos = []
    for image_info in image_infos:
        bbox = image_info.get("bbox")
        if bbox:
            box = _visual_box_from_rect(page, bbox)
            if _is_meaningful_graphic_box(box, page_width, page_height):
                candidates.append((box, "embedded_image"))
            elif legacy_fallback and _is_low_confidence_legacy_graphic_box(box, page_width, page_height):
                candidates.append((box, "legacy_low_confidence_embedded_image"))
    return _dedupe_boxes([box for box, _method in candidates])


def _dense_grid_graphic_box(
    drawing_rects: list[BoundingBox],
    *,
    page_width: float,
    page_height: float,
) -> BoundingBox | None:
    """Return one grid only when the strokes form a single connected grid."""

    boxes = _dense_grid_graphic_boxes(
        drawing_rects,
        page_width=page_width,
        page_height=page_height,
    )
    return boxes[0] if len(boxes) == 1 else None


def _dense_grid_graphic_boxes(
    drawing_rects: list[BoundingBox],
    *,
    page_width: float,
    page_height: float,
) -> list[BoundingBox]:
    """Recover grids encoded as many zero-area vector strokes.

    PyMuPDF reports each long grid line as a drawing rectangle with zero width
    or height, so the normal area-based graphic filter intentionally drops it.
    Requiring several long lines in both directions distinguishes a real grid
    or table from answer rules, page borders, and ordinary graph axes. Connected
    components keep separate grids on the same page from becoming one oversized
    graphic region.
    """

    interior = [
        box
        for box in drawing_rects
        if box.x0 > 18
        and box.x1 < page_width - 18
        and box.y0 > 35
        and box.y1 < page_height - 35
    ]
    vertical = [
        box
        for box in interior
        if box.x1 - box.x0 <= 2.5 and box.y1 - box.y0 >= max(24.0, page_height * 0.05)
    ]
    if len(vertical) < 3:
        return []
    horizontal = [
        box
        for box in interior
        if box.y1 - box.y0 <= 2.5
        and box.x1 - box.x0 >= max(24.0, page_width * 0.04)
    ]
    if len(horizontal) < 3:
        return []

    vertical_to_horizontal: dict[int, set[int]] = {index: set() for index in range(len(vertical))}
    horizontal_to_vertical: dict[int, set[int]] = {index: set() for index in range(len(horizontal))}
    for vertical_index, vertical_rule in enumerate(vertical):
        center_x = (vertical_rule.x0 + vertical_rule.x1) / 2
        for horizontal_index, horizontal_rule in enumerate(horizontal):
            center_y = (horizontal_rule.y0 + horizontal_rule.y1) / 2
            if (
                horizontal_rule.x0 - 4 <= center_x <= horizontal_rule.x1 + 4
                and vertical_rule.y0 - 3 <= center_y <= vertical_rule.y1 + 3
            ):
                vertical_to_horizontal[vertical_index].add(horizontal_index)
                horizontal_to_vertical[horizontal_index].add(vertical_index)

    grid_boxes: list[BoundingBox] = []
    unvisited_vertical = set(range(len(vertical)))
    while unvisited_vertical:
        seed = unvisited_vertical.pop()
        component_vertical = {seed}
        component_horizontal: set[int] = set()
        pending: list[tuple[str, int]] = [("vertical", seed)]
        while pending:
            kind, index = pending.pop()
            if kind == "vertical":
                for horizontal_index in vertical_to_horizontal[index] - component_horizontal:
                    component_horizontal.add(horizontal_index)
                    pending.append(("horizontal", horizontal_index))
            else:
                for vertical_index in horizontal_to_vertical[index] - component_vertical:
                    component_vertical.add(vertical_index)
                    unvisited_vertical.discard(vertical_index)
                    pending.append(("vertical", vertical_index))

        component_vertical_rules = [vertical[index] for index in component_vertical]
        component_horizontal_rules = [horizontal[index] for index in component_horizontal]
        if (
            _distinct_rule_position_count(component_vertical_rules, vertical=True) < 3
            or _distinct_rule_position_count(component_horizontal_rules, vertical=False) < 3
        ):
            continue

        vertical_box = _union_boxes(component_vertical_rules)
        horizontal_box = _union_boxes(component_horizontal_rules)
        component_box = _union_boxes([vertical_box, horizontal_box])
        if component_box.x1 - component_box.x0 < 50 or component_box.y1 - component_box.y0 < 50:
            continue
        spanning_horizontal = [
            rule
            for rule in component_horizontal_rules
            if rule.x0 <= vertical_box.x0 + 4 and rule.x1 >= vertical_box.x1 - 4
        ]
        spanning_vertical = [
            rule
            for rule in component_vertical_rules
            if rule.y0 <= horizontal_box.y0 + 4 and rule.y1 >= horizontal_box.y1 - 4
        ]
        if (
            _distinct_rule_position_count(spanning_vertical, vertical=True) < 3
            or _distinct_rule_position_count(spanning_horizontal, vertical=False) < 3
        ):
            continue
        grid_boxes.append(_union_boxes([*spanning_vertical, *spanning_horizontal]))

    return _dedupe_boxes(grid_boxes)


def _distinct_rule_position_count(rules: list[BoundingBox], *, vertical: bool) -> int:
    positions = sorted(
        (rule.x0 + rule.x1) / 2 if vertical else (rule.y0 + rule.y1) / 2
        for rule in rules
    )
    distinct: list[float] = []
    for position in positions:
        if not distinct or position - distinct[-1] > 2.5:
            distinct.append(position)
    return len(distinct)


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


def _is_low_confidence_legacy_graphic_box(box: BoundingBox, page_width: float, page_height: float) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    area = width * height
    if area < 12:
        return False
    page_area = max(1.0, page_width * page_height)
    if area >= page_area * 0.92:
        return False
    if box.y1 < 35 or box.y0 > page_height - 35:
        return False
    if box.x1 < 18 or box.x0 > page_width - 18:
        return False
    return width >= 4 and height >= 4


def _drawing_item_boxes(page: Any, drawing: dict[str, Any]) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    for item in drawing.get("items", []) or []:
        for value in item:
            box = _rectlike_to_box(page, value)
            if box is not None:
                boxes.append(box)
    return boxes


def _rectlike_to_box(page: Any, value: Any) -> BoundingBox | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return None
    for attr in ("x0", "y0", "x1", "y1"):
        if not hasattr(value, attr):
            break
    else:
        return _visual_box_from_rect(page, value)
    if isinstance(value, (list, tuple)) and len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4]):
        return _visual_box_from_rect(page, value[:4])
    return None


def _dense_non_text_cluster(
    boxes: list[BoundingBox],
    *,
    page_width: float,
    page_height: float,
) -> BoundingBox | None:
    meaningful = [
        box
        for box in boxes
        if _is_low_confidence_legacy_graphic_box(box, page_width, page_height)
    ]
    if len(meaningful) < 3:
        return None
    union = _union_boxes(meaningful)
    width = max(0.0, union.x1 - union.x0)
    height = max(0.0, union.y1 - union.y0)
    if width < 18 or height < 18:
        return None
    union_area = max(1.0, width * height)
    source_area = sum(max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0) for box in meaningful)
    if source_area / union_area < 0.015:
        return None
    return union


def _ocr_hint_graphics(
    blocks: list[TextBlock],
    existing_graphics: list[BoundingBox],
    *,
    page_width: float,
    page_height: float,
    legacy_fallback: bool,
) -> list[BoundingBox]:
    hints: list[BoundingBox] = []
    radius = 110.0 if legacy_fallback else 70.0
    for block in blocks:
        if not _is_figure_hint_text(block.text):
            continue
        near = [
            graphic
            for graphic in existing_graphics
            if _distance_between_boxes(block.bbox, graphic) <= radius
        ]
        if near:
            hints.append(_union_boxes([block.bbox, *near]).padded(8, page_width, page_height))
        elif legacy_fallback:
            hints.append(block.bbox.padded(radius * 0.5, page_width, page_height))
    return hints


def _is_figure_hint_text(text: str) -> bool:
    cleaned = _normalized_ocr_text(text).lower()
    return bool(
        re.search(r"\b(?:figure|fig\.?|diagram|graph|curve|sketch|sector|circle|histogram|box plot|scatter diagram)\b", cleaned)
        or re.fullmatch(r"(?:[A-Z]|\d{1,2}|[()+\-−=])(?:\s+(?:[A-Z]|\d{1,2}|[()+\-−=])){1,5}", str(text).strip())
    )


def _distance_between_boxes(a: BoundingBox, b: BoundingBox) -> float:
    horizontal_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    vertical_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return (horizontal_gap**2 + vertical_gap**2) ** 0.5


def _dedupe_boxes(boxes: list[BoundingBox]) -> list[BoundingBox]:
    kept: list[BoundingBox] = []
    for box in sorted(boxes, key=lambda item: (item.y0, item.x0, item.y1, item.x1)):
        if any(_boxes_overlap_ratio(box, existing) >= 0.92 for existing in kept):
            continue
        kept.append(box)
    return kept


def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _append_warning(current: str | None, value: str) -> str:
    if not current:
        return value
    parts = [part for part in current.split(";") if part]
    if value not in parts:
        parts.append(value)
    return ";".join(parts)


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
    if not _should_run_sparse_lower_ocr(pdf_blocks, float(page.rect.height), config):
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


def _should_run_sparse_lower_ocr(
    pdf_blocks: list[TextBlock],
    page_height: float,
    config: AppConfig,
) -> bool:
    strategy = str(config.ocr.strategy or "adaptive").strip().lower()
    if strategy == "always":
        return True
    if strategy != "adaptive":
        raise ValueError(f"Unsupported OCR strategy: {config.ocr.strategy!r}")

    substantive_blocks = [
        block
        for block in sorted(pdf_blocks, key=lambda item: (item.bbox.y0, item.bbox.x0))
        if _is_sparse_lower_region_body_block(block, page_height, config)
    ]
    native_text = " ".join(block.text for block in substantive_blocks).strip()
    score = score_text_candidate(native_text, source="native")
    if score.rejection_reasons or score.score < int(config.ocr.native_text_min_score):
        return True
    if _SPARSE_LOWER_OCR_CORRUPTION_REASONS.intersection(score.reasons):
        return True
    return re.search(r"\[\d{1,2}\]\s*$", native_text) is None


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

    return fitz.Rect(
        float(config.detection.crop_left_margin),
        start_y,
        float(page.rect.width) - float(config.detection.crop_right_margin),
        body_bottom,
    )


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
