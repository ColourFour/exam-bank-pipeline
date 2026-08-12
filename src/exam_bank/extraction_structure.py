from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .config import AppConfig
from .models import BoundingBox, PageLayout, QuestionSpan, TextBlock
from .question_detection_layout import distance_to_box as _distance_to_box
from .question_detection_layout import (
    looks_like_diagram_axis_or_label_text as _shared_looks_like_diagram_axis_or_label_text,
)

_PART_LINE_RE = re.compile(
    r"^\s*(?:(?P<number>\d{1,2})\s*)?(?P<label>\((?:[a-h]|i{1,3}|iv|v|vi{0,3}|ix|x)\)(?:\((?:i{1,3}|iv|v|vi{0,3}|ix|x)\))?)",
    re.IGNORECASE,
)
_MATH_TOKEN_RE = re.compile(
    r"(?:=|<|>|≤|≥|\^|√|π|θ|∫|Σ|sin|cos|tan|sec|cosec|cot|ln|log|dy/dx|dx/dt|x\s*=|y\s*=|\|z\||arg|vector)",
    re.IGNORECASE,
)
_SUSPICIOUS_SYMBOL_RUN_RE = re.compile(r"[=<>^/*_]{4,}|[?�]{3,}|(?:[A-Za-z0-9][^A-Za-z0-9\s]){4,}")
@dataclass(frozen=True)
class StructuredQuestionText:
    body_text_raw: str
    body_text_normalized: str
    math_lines: list[str]
    diagram_text: list[str]
    extraction_quality_score: float
    extraction_quality_flags: list[str]
    combined_question_text: str
    part_texts: list[dict[str, object]]


@dataclass(frozen=True)
class _LineItem:
    text: str
    page_number: int
    bbox: BoundingBox
    font_size: float | None = None


def build_structured_question_text(
    span: QuestionSpan,
    layouts: list[PageLayout],
    config: AppConfig,
) -> StructuredQuestionText:
    lines = _lines_from_blocks(span.blocks)
    text_only_diagram_line_keys = _text_only_diagram_line_keys(lines, layouts, span, config)
    diagram_lines: list[str] = []
    body_lines: list[str] = []
    body_line_items: list[_LineItem] = []

    for line in lines:
        # Copyright acknowledgements are page furniture that follows the last
        # question on older papers.  Nothing after the first acknowledgement
        # line belongs to the question body.
        if _looks_like_document_footer_line(line.text):
            break
        if _looks_like_page_turn_furniture_line(line.text):
            continue
        layout = _layout_by_number(layouts, line.page_number)
        if body_line_items and _is_lowercase_body_continuation(line, body_line_items[-1]):
            body_lines.append(line.text)
            body_line_items.append(line)
            continue
        split_anchor = _split_question_anchor_diagram_label(line, layout, span, config)
        if split_anchor is not None:
            body_text, diagram_text = split_anchor
            if not _body_already_has_question_anchor(body_lines, body_text):
                body_lines.append(body_text)
                body_line_items.append(line)
            diagram_lines.append(diagram_text)
            continue
        if _is_duplicate_question_number_diagram_label(line, layout, span, body_lines, config):
            diagram_lines.append(line.text)
            continue
        if _looks_like_answer_filler_line(line.text):
            continue
        if _line_key(line) in text_only_diagram_line_keys:
            diagram_lines.append(line.text)
            continue
        if _looks_like_diagram_text(line, layout, span, config):
            diagram_lines.append(line.text)
        else:
            body_lines.append(line.text)
            body_line_items.append(line)

    body_text_raw = "\n".join(_clean_raw_line(line) for line in body_lines if _clean_raw_line(line)).strip()
    body_text_normalized = _normalize_preserving_structure(body_text_raw)
    math_lines = _extract_math_lines(body_text_raw)
    extraction_quality_flags = _extraction_quality_flags(body_text_raw, body_text_normalized, math_lines, diagram_lines)
    extraction_quality_score = _quality_score(extraction_quality_flags)
    combined_question_text = body_text_normalized
    part_texts = _part_texts(body_text_raw)

    return StructuredQuestionText(
        body_text_raw=body_text_raw,
        body_text_normalized=body_text_normalized,
        math_lines=math_lines,
        diagram_text=[_normalize_light(line) for line in diagram_lines if _normalize_light(line)],
        extraction_quality_score=extraction_quality_score,
        extraction_quality_flags=extraction_quality_flags,
        combined_question_text=combined_question_text,
        part_texts=part_texts,
    )


def _lines_from_blocks(blocks: list[TextBlock]) -> list[_LineItem]:
    items: list[_LineItem] = []
    for block in sorted(blocks, key=lambda item: (item.page_number, item.bbox.y0, item.bbox.x0)):
        raw_lines = [line for line in block.text.splitlines() if line.strip()]
        if not raw_lines:
            continue
        line_height = max(1.0, (block.bbox.y1 - block.bbox.y0) / max(1, len(raw_lines)))
        for index, line in enumerate(raw_lines):
            y0 = block.bbox.y0 + (index * line_height)
            y1 = min(block.bbox.y1, y0 + line_height)
            items.append(
                _LineItem(
                    text=line,
                    page_number=block.page_number,
                    bbox=BoundingBox(block.bbox.x0, y0, block.bbox.x1, y1),
                    font_size=block.font_size,
                )
            )
    return items


def _text_only_diagram_line_keys(
    lines: list[_LineItem],
    layouts: list[PageLayout],
    span: QuestionSpan,
    config: AppConfig,
) -> set[tuple[int, float, float, str]]:
    if not _span_has_visual_prompt(span):
        return set()

    candidates_by_page: dict[int, list[_LineItem]] = {}
    for line in lines:
        layout = _layout_by_number(layouts, line.page_number)
        if layout.graphics:
            continue
        if _looks_like_text_only_diagram_label(line, span, config):
            candidates_by_page.setdefault(line.page_number, []).append(line)

    keys: set[tuple[int, float, float, str]] = set()
    max_gap = max(config.detection.prompt_graphic_lookahead * 1.7, config.detection.prompt_region_max_gap * 4.0)
    for page_lines in candidates_by_page.values():
        cluster: list[_LineItem] = []
        previous: _LineItem | None = None
        for line in sorted(page_lines, key=lambda item: (item.bbox.y0, item.bbox.x0)):
            if previous is None or line.bbox.y0 - previous.bbox.y1 <= max_gap:
                cluster.append(line)
            else:
                if _text_only_diagram_cluster_is_strong(cluster):
                    keys.update(_line_key(item) for item in cluster)
                cluster = [line]
            previous = line
        if _text_only_diagram_cluster_is_strong(cluster):
            keys.update(_line_key(item) for item in cluster)
    return keys


def _text_only_diagram_cluster_is_strong(lines: list[_LineItem]) -> bool:
    if len(lines) < 2:
        return False
    cleaned = [_normalize_light(line.text) for line in lines]
    has_axis_label = any("(" in text and ")" in text for text in cleaned)
    has_numeric_or_origin_label = any(re.search(r"(?:^|\s)(?:O|-?\d+(?:\.\d+)?)(?:\s|$)", text) for text in cleaned)
    return has_axis_label or has_numeric_or_origin_label


def _looks_like_text_only_diagram_label(line: _LineItem, span: QuestionSpan, config: AppConfig) -> bool:
    cleaned = _normalize_light(line.text)
    if not cleaned:
        return False
    if re.search(r"\[\d{1,2}\]", cleaned):
        return False
    if _PART_LINE_RE.match(cleaned):
        return False
    parsed_anchor = re.match(rf"^\s*{re.escape(span.question_number)}\s+(.+?)\s*$", cleaned)
    if parsed_anchor:
        cleaned = parsed_anchor.group(1)
    elif cleaned == span.question_number and line.bbox.x0 <= config.detection.question_start_max_x + 20:
        return False

    sentence_like = bool(
        re.search(
            r"\b(?:the|find|show|calculate|solve|given|diagram|graph|sketch|draw|hence|for|from|with)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    if sentence_like and not _looks_like_diagram_axis_or_label_text(cleaned):
        return False
    if _looks_like_diagram_axis_or_label_text(cleaned):
        return True
    return len(cleaned) <= 28 and len(cleaned.split()) <= 4 and not sentence_like and bool(re.search(r"[A-Z0-9()°]", cleaned))


def _line_key(line: _LineItem) -> tuple[int, float, float, str]:
    return (line.page_number, round(line.bbox.y0, 2), round(line.bbox.x0, 2), _normalize_light(line.text))


def _span_has_visual_prompt(span: QuestionSpan) -> bool:
    text = _normalize_light(span.combined_text).lower()
    return bool(
        re.search(
            r"\b(?:diagram|graph|sketch|draw|shown|velocity-time|displacement-time|speed-time|force-time|v\s*\(|t\s*\(|shaded|figure)\b",
            text,
        )
    )


def _looks_like_diagram_text(line: _LineItem, layout: PageLayout, span: QuestionSpan, config: AppConfig) -> bool:
    cleaned = _normalize_light(line.text)
    if not cleaned:
        return False
    if re.search(r"\[\d{1,2}\]", cleaned):
        return False
    if re.match(rf"^\s*{re.escape(span.question_number)}(?:\b|[.)])", cleaned):
        return False
    if _PART_LINE_RE.match(cleaned):
        return False

    near_graphic = any(_distance_to_box(line.bbox, graphic) <= 32 for graphic in layout.graphics)
    short = len(cleaned) <= 16 and len(cleaned.split()) <= 4
    simple_label = _looks_like_diagram_axis_or_label_text(cleaned)
    sentence_like = bool(re.search(r"\b(the|find|show|calculate|solve|given|diagram)\b", cleaned, re.IGNORECASE))

    if near_graphic and short and not sentence_like:
        return True
    if near_graphic and _looks_like_diagram_axis_or_label_text(cleaned):
        return True
    if near_graphic and simple_label:
        return True
    if simple_label and short and line.bbox.x0 > config.detection.question_start_max_x + 40:
        return True
    return False


def _split_question_anchor_diagram_label(
    line: _LineItem,
    layout: PageLayout,
    span: QuestionSpan,
    config: AppConfig,
) -> tuple[str, str] | None:
    cleaned = _normalize_light(line.text)
    match = re.match(rf"^\s*({re.escape(span.question_number)})\s+(.+?)\s*$", cleaned)
    if not match:
        return None
    if not any(_distance_to_box(line.bbox, graphic) <= 32 for graphic in layout.graphics):
        return None
    diagram_tail = match.group(2)
    if not _looks_like_diagram_axis_or_label_text(diagram_tail):
        return None
    return match.group(1), diagram_tail


def _looks_like_diagram_axis_or_label_text(text: str) -> bool:
    cleaned = _normalize_light(text)
    if not cleaned:
        return False
    return _shared_looks_like_diagram_axis_or_label_text(cleaned)


def _body_already_has_question_anchor(body_lines: list[str], candidate: str) -> bool:
    normalized_candidate = _normalize_light(candidate)
    if not normalized_candidate.isdigit():
        return False
    return any(re.match(rf"^\s*{re.escape(normalized_candidate)}(?:\b|[.)])", _normalize_light(line)) for line in body_lines)


def _is_duplicate_question_number_diagram_label(
    line: _LineItem,
    layout: PageLayout,
    span: QuestionSpan,
    body_lines: list[str],
    config: AppConfig,
) -> bool:
    cleaned = _normalize_light(line.text)
    if cleaned != span.question_number:
        return False
    if not _body_already_has_question_anchor(body_lines, cleaned):
        return False
    return any(_distance_to_box(line.bbox, graphic) <= 32 for graphic in layout.graphics)


def _looks_like_answer_filler_line(text: str) -> bool:
    cleaned = _normalize_light(text)
    if not cleaned:
        return False
    if re.match(r"^[A-Za-z][A-Za-z0-9 ]{0,30}(?:[._\-–—]\s*){20,}$", cleaned):
        return True
    if re.match(r"^(?:[._\-–—]\s*){12,}", cleaned):
        return True
    visible_alnum = len(re.findall(r"[A-Za-z0-9]", cleaned))
    filler_count = len(re.findall(r"[._\-–—]", cleaned))
    return filler_count >= 24 and visible_alnum <= 4


def _looks_like_document_footer_line(text: str) -> bool:
    cleaned = _normalize_light(text)
    return bool(
        re.search(
            r"^(?:Permission to reproduce items|Every reasonable effort has been made by the publisher|"
            r"University of Cambridge International Examinations is part of|"
            r"To avoid the issue of disclosure of answer-related information|"
            r"All copyright acknowledgements are reproduced online)",
            cleaned,
            re.IGNORECASE,
        )
    )


def _looks_like_page_turn_furniture_line(text: str) -> bool:
    cleaned = _normalize_light(text)
    return bool(
        re.fullmatch(
            r"\[Questions?(?:\s+\d+|\s+\d+\s+\([ivx]+\))(?:\s+and\s+\d+)?\s+"
            r"(?:is|are)\s+printed\s+on\s+the\s+next\s+page\.\]",
            cleaned,
            re.IGNORECASE,
        )
    )


def _is_lowercase_body_continuation(line: _LineItem, previous: _LineItem) -> bool:
    """Keep a wrapped prose tail even when a broad graphic overlaps the text."""

    current_text = _normalize_light(line.text)
    previous_text = _normalize_light(previous.text)
    if not current_text or not previous_text or line.page_number != previous.page_number:
        return False
    if not re.match(r"^[a-z]", current_text):
        return False
    if not re.search(r"[.!?](?:\s*\[\d{1,2}\])?$", current_text):
        return False
    if re.search(r"[.!?:;](?:\s*\[\d{1,2}\])?$", previous_text):
        return False
    if abs(line.bbox.x0 - previous.bbox.x0) > 36:
        return False
    font_size = previous.font_size or line.font_size or 11.0
    vertical_gap = line.bbox.y0 - previous.bbox.y1
    return -2 <= vertical_gap <= max(20.0, font_size * 1.8)


def _normalize_preserving_structure(text: str) -> str:
    normalized_lines = [_normalize_light(line) for line in text.splitlines()]
    normalized = "\n".join(line for line in normalized_lines if line).strip()
    normalized = _repair_canonical_math_serialization(normalized)
    return _repair_cross_line_theta_context(normalized)


def _repair_cross_line_theta_context(text: str) -> str:
    has_theta_context = re.search(
        r"\bangle\s+[A-Z]{2,4}\s*=\s*θ\s*radians\b"
        r"|\b(?:sin|cos|tan|sec|cosec|cot)(?:\s*(?:\^\{(?!-\})[^}]+\})?)\s*θ\b"
        r"|\b0\s*(?:<|≤)\s*θ\s*(?:<|≤)",
        text,
        re.IGNORECASE,
    )
    if not has_theta_context:
        return text
    return re.sub(r"\b(value of )i\b", r"\1θ", text, flags=re.IGNORECASE)


def _normalize_light(text: str) -> str:
    value = _normalize_pdf_math_glyphs(text)
    value = value.replace("\u00a0", " ")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"[ \t]+", " ", value.strip())
    return value


def _clean_raw_line(text: str) -> str:
    value = _normalize_pdf_math_glyphs(text)
    value = value.replace("\u00a0", " ")
    value = value.replace("\r", " ")
    return value.rstrip()


def _normalize_pdf_math_glyphs(text: str) -> str:
    """Repair recurring CAIE/PDF math glyph extraction artifacts without claiming semantic certainty."""

    value = str(text or "")
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "−": "-",
        "–": "-",
        "—": "-",
        "\x00": "(",
        "\x01": ")",
        "\x02": "",
        "\x0e": "|",
        "\x10": "(",
        "\x11": ")",
        "\x8f": "≡",
        "Ó": "∫",
        "Ô": "∫",
        "Å": "°",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    # This control character is often a radical marker in the question papers,
    # but not reliably enough to replace globally. Keep the high-signal case
    # from substitution prompts and otherwise remove the noise character.
    value = re.sub(r"(\bu\s*=\s*)\x0f(?=\s*x\b)", r"\1√", value)
    value = value.replace("\x0f", "")
    value = re.sub(r"[\x03-\x08\x0b\x0c\x12-\x1f\x7f]", "", value)

    value = _normalize_common_math_ocr_substitutions(value)
    value = _repair_common_joined_words(value)
    return value


def _normalize_common_math_ocr_substitutions(text: str) -> str:
    value = text
    value = value.replace("↦→", "↦").replace("↦ →", "↦")
    value = re.sub(r"\(--→\)/\(([A-Z]{2})\)", r"\\overrightarrow{\1}", value)
    value = re.sub(r"\b([A-Za-z])\(([^()\[\]]+)\[(\d{1,2})\]\)([.?!])", r"\1(\2)\4 [\3]", value)
    value = re.sub(r"\b([A-Za-z])\(([^()\[\]]+)\[(\d{1,2})\]\)", r"\1(\2) [\3]", value)
    value = re.sub(r"(\[\d{1,2}\])\s*(?:[._\-–—]\s*){12,}.*$", r"\1", value)
    value = _repair_caie_math_delimiters(value)
    value = re.sub(
        r"(\binterval\s+)0\s*1\s*([xXθ])\s*1\s*(\^\{[^}]+\}_\{[^}]+\})r\b",
        r"\g<1>0 ≤ \2 ≤ \3π",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"(?<![A-Za-z])-\s*r\s*G\s*[i1θ]\s*G\s*r\b", "-π ≤ θ ≤ π", value)
    value = re.sub(r"\br\s*G\s*[i1θ]\s*G\s*r\b", "π ≤ θ ≤ π", value)
    value = re.sub(
        r"\b([0-9])\s*G\s*([xXiIθ])(?![A-FH-Za-z])",
        r"\1 ≤ \2",
        value,
    )
    value = re.sub(r"\b([xXiIθ])\s*G\s*(\^\{[^}]+\}|[0-9])", r"\1 ≤ \2", value)
    value = re.sub(
        r"\b(\d+)\s*G\s*([A-Za-zθ])\s*G\s*(\d+)\b",
        r"\1 ≤ \2 ≤ \3",
        value,
    )
    value = re.sub(r"\br20\b", "r > 0", value)
    # In older CAIE fonts the terminal pi in a standard 0-to-angle bound is
    # sometimes exposed as ``r``.  Restrict that repair to the complete range
    # expression; quantities such as 12r and common ratios must remain r.
    value = re.sub(
        r"(\b0\s*(?:<|≤)\s*[xXiIθ]\s*(?:<|≤)\s*"
        r"(?:\^\{[^}]+\}_\{[^}]+\}|[0-9]+))r\b",
        r"\1π",
        value,
    )
    value = re.sub(r"°\s*(0|90|180|360)\b", r"\1°", value)
    value = re.sub(r"\b(0|90|180|360)(°?)\s*<\s*1\s*<", r"\1\2 < θ <", value)
    value = re.sub(r"\b(0|90|180|360)(°?)\s*≤\s*1\s*≤", r"\1\2 ≤ θ ≤", value)
    value = re.sub(r"\b0\s*<\s*1\s*<", "0 < θ <", value)
    value = re.sub(r"\b0\s*≤\s*1\s*≤", "0 ≤ θ ≤", value)
    value = re.sub(r"\b(sin|cos|tan|sec|cosec|cot)\(\s*1(?=\s*[-+])", r"\1(θ", value)
    value = re.sub(r"\b(sin|cos|tan|sec|cosec|cot)(\s*(?:\^\{(?!-\})[^}]+\})?)\s*1\b", r"\1\2 θ", value)
    value = re.sub(r"\b(sin|cos|tan|sec|cosec|cot)(?=[0-9θxyzi])", r"\1 ", value)
    value = re.sub(r"\b(ln|log)(?=[A-Za-z0-9(])", r"\1 ", value)
    value = re.sub(r"\b(ln|log)\s+\(", r"\1(", value)
    value = re.sub(r"(?<=[A-Za-z0-9}])(?=(?:cosec|sin|cos|tan|sec|cot)(?:[0-9θxyzi]|\b))", " ", value)
    value = re.sub(r"\b(sin|cos|tan|sec|cosec|cot)(?=[0-9θxyzi])", r"\1 ", value)
    value = re.sub(r"\b(cosec|sin|cos|tan|sec|cot|ln|log)\s+([0-9]+)([A-Za-zθ])\b", r"\1 \2\3", value)
    value = re.sub(r"\b(tan|sin|cos|sec|cosec|cot)\^\{-\}\s*θ", r"\1^{-}1", value)
    value = _repair_trig_theta_placeholders(value)
    return _repair_canonical_math_serialization(value)


def _repair_canonical_math_serialization(text: str) -> str:
    """Finish narrow, source-faithful repairs that can cross PDF span lines."""

    value = text.replace("µ", "μ").replace("∼", "~")
    value = value.replace("↦→", "↦").replace("↦ →", "↦")
    value = re.sub(
        r"\b(\d+)\s*G\s*([A-Za-zθ])\s*G\s*(\d+)\b",
        r"\1 ≤ \2 ≤ \3",
        value,
    )
    value = re.sub(r"\^\{-\}\s*(\d+)", r"^{-\1}", value)
    value = re.sub(r"(?<=\})(?=[A-Za-z]{2,}\b)", " ", value)
    value = re.sub(r"(?<=[A-Za-z0-9)}])\s+\^\{", "^{", value)
    value = re.sub(
        r"([A-Za-z)])\^\{-\}\s*(\d+)_\{(\d+)\}",
        r"\1^{-(\2)/(\3)}",
        value,
    )
    value = re.sub(
        r"([A-Za-z)])\^\{(-?)(\d+)\}_\{(\d+)\}",
        lambda match: (
            f"{match.group(1)}^{{"
            f"{'-' if match.group(2) else ''}({match.group(3)})/({match.group(4)})"
            "}"
        ),
        value,
    )
    value = re.sub(
        r"(?<![A-Za-z0-9}])(-?)(\d+)_\{(\d+)\}",
        lambda match: f"{match.group(1)}({match.group(2)})/({match.group(3)})",
        value,
    )
    value = re.sub(r"_\{([A-Za-z])\}\s*\+\s*([0-9]+)", r"_{\1+\2}", value)
    value = re.sub(
        r"([A-Za-z0-9])\^\{([A-Za-z])\}\+\s*(\d+)",
        r"\1^{\2 + \3}",
        value,
    )
    value = re.sub(
        r"([A-Za-z0-9])\^\{([A-Za-z])\}\^\{\+\}\s*(\d+)",
        r"\1^{\2+\3}",
        value,
    )
    value = re.sub(r"_\{([A-Za-z])\}_\{\+\}(\d+)", r"_{\1+\2}", value)
    value = re.sub(
        r"e\^\{(-?\([^)]+\)/\([^)]+\))\}\^\{([A-Za-z])\}",
        r"e^{\1\2}",
        value,
    )
    value = re.sub(r"e\^\{-\}\s*([A-Za-z])", r"e^{-\1}", value)
    value = re.sub(
        r"e\^\{(-(?:\d+|\([^)]+\)/\([^)]+\)|[A-Za-z])|i)\}([A-Za-zθ])",
        r"e^{\1\2}",
        value,
    )
    value = re.sub(r"\b(?:ddxy|ddyx|xddy)\b", "(dy)/(dx)", value)
    value = re.sub(r"\bddθx\b", "(dx)/(dθ)", value)
    value = re.sub(r"\bf\^\{\s+′\}", "f'", value)
    value = re.sub(r"\b([fgh])\s+-\s*1\b", r"\1^{-1}", value)
    value = re.sub(r"√(?!\()(\d+|[A-Za-z])", r"√(\1)", value)
    value = re.sub(
        r"\b(cosec|sin|cos|tan|sec|cot)(\^\{[^}]+\})(?=[0-9A-Za-zθ])",
        r"\1\2 ",
        value,
    )
    value = re.sub(r"\bfor -\s+(?=\()", "for -", value)
    value = re.sub(r"([=(,])-\s+(?=\()", r"\1-", value)
    value = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "-", value)
    value = re.sub(r"~\s*(?=N\s*\()", "~ ", value)
    value = re.sub(r"\s*(≤|≥|≡|≈|∈|=|<|>)\s*", r" \1 ", value)
    value = re.sub(r"(?<=[A-Za-z0-9})|])-(?=\()", " - ", value)
    # Preserve unary negatives (for example coordinates and constants).  The
    # two compact binary-minus forms below are recurring complex-number font
    # artefacts and have enough surrounding syntax to repair unambiguously.
    value = re.sub(r"(\|z)\s+-\s*(?=\d)", r"\1 - ", value)
    value = re.sub(r"(arg\(z)\s+-\s*(?=[A-Za-z])", r"\1 - ", value)
    value = re.sub(r"\s*↦\s*", " ↦ ", value)
    # A decoded real-set glyph between a scalar and a number is the old-font
    # greater-than glyph. Genuine membership is already represented by ∈ ℝ.
    value = re.sub(r"(?<=[A-Za-z0-9})])\s*ℝ\s*(?=[0-9])", " > ", value)
    value = re.sub(
        r"(?<=[A-Za-z0-9})θ])\s*ℝ\s*(?=(?:sin|cos|tan|[fgh]\s*\())",
        " > ",
        value,
    )
    value = re.sub(r"(?<=°)(?=[A-Za-z])", " ", value)
    value = re.sub(r"([A-Za-zαβθ])\^\{°\}", r"\1°", value)
    value = re.sub(r"(?<=[0-9)])(?=(?:kg|km|cm|mm|kW|N|J|W)\b)", " ", value)
    value = re.sub(r"(?<=\d)(?=m(?!\s*kg\b)(?:\s|[.,;:)]|$))", " ", value)
    value = re.sub(r"\s+([,.;:?!])", r"\1", value)
    value = re.sub(r"([,;:])(?=\S)", r"\1 ", value)
    value = re.sub(r"([\[(])\s+", r"\1", value)
    value = re.sub(r"\s+\)", ")", value)
    value = re.sub(r"\b([fg])\s+\(", r"\1(", value)
    value = re.sub(r"\b([fg])\s*:\s*([A-Za-z])", r"\1 : \2", value)
    value = re.sub(
        r"\b([A-Z])\s+\((?=\s*-?(?:\d+(?:\.\d+)?|[a-z])\s*,)",
        r"\1(",
        value,
    )
    value = re.sub(r"(?<!\s)(?=\[\d{1,2}\])", " ", value)
    value = re.sub(r"\b(?:in\s+equal\s+it\s+y|in\s+equality)\b", "inequality", value)
    value = re.sub(r"\bco\s+sec\b", "cosec", value)
    value = value.replace("ln(cost)", "ln(cos t)")
    value = re.sub(r"(?<== )cott\b", "cot t", value)
    value = re.sub(r"(?<=[A-Za-z0-9)])\(see diagram\)", " (see diagram)", value)
    value = re.sub(r"\b(\d+(?:\.\d+)?)ms(?=\^\{-[12]\})", r"\1 m s", value)
    value = re.sub(r"\b([A-Za-z])ms(?=\^\{-[12]\})", r"\1 m s", value)
    value = re.sub(r"(?<=\))ms(?=\^\{-[12]\})", " m s", value)
    value = re.sub(
        r"\bms(\d+(?:\.\d+)?)\s+-([12])\b",
        r"\1 m s^{-\2}",
        value,
    )
    value = re.sub(
        r"\bm(\d+(?:\.\d+)?)\s+s\s+-([12])\b",
        r"\1 m s^{-\2}",
        value,
    )
    value = re.sub(r"\bm([vV])\s+s\s+-([12])\b", r"\1 m s^{-\2}", value)
    value = re.sub(
        r"\b(\d+(?:\.\d+)?)ms\s+-([12])\b",
        r"\1 m s^{-\2}",
        value,
    )
    value = re.sub(
        r"\bat\.(\d)\s+(\d+)([A-Za-z])ms\s+-([12])\b",
        r"at \1.\2\3 m s^{-\4}",
        value,
    )
    value = re.sub(r"\bms([vV])\s+-([12])\b", r"\1 m s^{-\2}", value)
    value = re.sub(
        r"\b(\d+(?:\.\d+)?)ms\^\{-\}",
        r"\1 m s^{-1}",
        value,
    )
    value = re.sub(r"\b((?:kW|W|J|N)\.)\^\{1\}", r"\1", value)
    value = re.sub(r"Σ\s+(?=[A-Za-z])", "Σ", value)
    value = re.sub(r"\bafter\s+noon\s+s\b", "afternoons", value)
    value = re.sub(r"\bwifi\s*connection\b", "wifi connection", value)
    value = re.sub(r"(∫_\{[^}]+\}\^\{[^}]+\})(?=\()", r"\1 ", value)
    value = re.sub(r"(\([^()]+\)/\([^()]+\))\s+\(", r"\1(", value)
    value = re.sub(r"\((\d+)\)/\(-\s*(\d+)\)", r"-(\1)/(\2)", value)
    value = re.sub(
        r"([A-Za-z0-9})])(\([^()]+\))/\(-\s*([^()]+)\)",
        r"\1 - \2/(\3)",
        value,
    )
    value = re.sub(
        r"([A-Za-z0-9})])\(\+\s*([^()]+)\)/\(([^()]+)\)",
        r"\1 + (\2)/(\3)",
        value,
    )
    value = re.sub(
        r"(\([^()]+\)/\()([^()]+)\s+-\)(\d+)",
        r"\1\2) - \3",
        value,
    )
    value = re.sub(
        r"([A-Za-z])\(\s*-\s*(\d+)\)/\((\d+)\)",
        r"\1^{-(\2)/(\3)}",
        value,
    )
    value = re.sub(r"([fgh])\s+-\s*1(?=\s*\()", r"\1^{-1}", value)
    value = re.sub(r"\((\d+)\)/(\d+)π\)", r"(\1)/(\2)π", value)
    value = re.sub(r"\((\d+)π\)/\((\d+)\)", r"(\1)/(\2)π", value)
    value = re.sub(r"√\((\d+)i\)", r"√(\1)i", value)
    value = re.sub(
        r"\b(sin|cos|tan|sec|cosec|cot)\s+\((\d+)\)/\((\d+)([A-Za-zθ])\)",
        r"\1((\2)/(\3)\4)",
        value,
    )
    value = re.sub(r"\((\d+)\)/\((\d+)π\)", r"(\1)/(\2)π", value)
    value = re.sub(
        r"--→([A-Z]{2})\.\s*--→([A-Z]{2})",
        r"\\overrightarrow{\1} · \\overrightarrow{\2}",
        value,
    )
    value = re.sub(r"\b([A-Zxy])2\s+([0-9])\b", r"\1 > \2", value)
    value = re.sub(r"\b([A-Za-z]{2,})_\{-\}", r"\1", value)
    value = re.sub(
        r"\bm\s+s\s+([12])(?=\s|[.,;:)])",
        r"m s^{-\1}",
        value,
    )
    value = re.sub(
        r"\bm\s+s\s+([12])(?=(?:and|respectively)\b)",
        r"m s^{-\1} ",
        value,
    )
    value = re.sub(
        r"\b(sin|cos|tan|sec|cosec|cot)\^\{-10\}\s+(\d+)\.",
        r"\1^{-1} 0.\2",
        value,
    )
    value = re.sub(r"(\S)\s+(\[\d{1,2}\])([.?!])", r"\1\3 \2", value)
    value = re.sub(r"\b(\d+)\s+(\d+)\.\.", r"\1.\2.", value)
    value = re.sub(
        r"^(\d+)\s+(?:[αβθ]|\d+(?:\.\d+)?\s+(?:m|kg|N))\s+"
        r"(?=(?:The diagram|Two particles|A particle))",
        r"\1 ",
        value,
    )
    if re.search(r"a,\s+b\s+and\s+c\s+respectively", value) and re.search(
        r"positive\s+constants", value
    ):
        value = re.sub(r"(?<=\d)°(?=\^\{2\}|\s|[.,])", "c", value)
    value = value.replace("an angle of °", "an angle of α°")
    value = value.replace("the value of °", "the value of α")
    value = re.sub(
        r"inclined\s+at\s+an\s+angle\s+to\s+the\s+horizontal,\s+where\s+sin\s*=",
        "inclined at an angle α to the horizontal, where sin α =",
        value,
    )
    value = _relocate_misplaced_degree(value)
    if "rectangle ABCD" in value and re.search(r"AQ\s+is\s+an\s+arc\s+of\s+a\s+circle\s+with\s+centre", value):
        value = re.sub(r"with centre\s+(?=\([ivx]+\))", "with centre D. ", value, count=1)
    compact_value = re.sub(r"\s+", " ", value)
    if (
        "sin = 0.8 and cos = 0.6" in compact_value
        or "sin β = 0.8 and cos β = 0.6" in compact_value
    ):
        value = value.replace("sin = 0.8", "sin β = 0.8")
        value = value.replace("cos = 0.6", "cos β = 0.6")
        value = value.replace("W cos)", "W cos α")
        value = value.replace("W sin)", "W sin α")
        value = value.replace("W and)", "W and α")
    value = re.sub(r"\bAs\s+tan\s+a\b", "Astana", value)
    value = re.sub(r"\bAs\s+tan(?=\s+and\s+Bejin\b)", "Astana", value)
    value = re.sub(r"\bno\s+on\b", "noon", value)
    value = re.sub(r"\bafter\s+noon\s+s\b", "afternoons", value)
    value = re.sub(
        r"\(([a-z]+)from([a-z]+)\)/\(-\)",
        r"\1 from \2",
        value,
    )
    if re.search(r"\ban\^\{1\}\s+angle\s+a°", value):
        value = re.sub(r"\ban\^\{1\}\s+angle\s+a°", "an angle α°", value)
        value = re.sub(r"\bvalue of a\b", "value of α", value)
    value = re.sub(
        r"(There is a constant)\s+"
        r"(\d+(?:\.\d+)?)\s+m\s+sresistance([12])\.to motion"
        r"([\s\S]*?\bhis acceleration is)",
        r"\1 resistance to motion\4 \2 m s^{-\3}.",
        value,
    )
    # Geometry-flattening repairs below require complete surrounding syntax;
    # keeping them at the end also makes the earlier glyph repairs available.
    value = re.sub(
        r"(\bprobability of (?:throwing|obtaining)[^.]{0,80}? is )(\d+)\^\{(\d+)\}",
        _replace_probability_power_fraction,
        value,
    )
    value = re.sub(
        r"\(\((\d+)\)/\(([^()]+?)\s+([23])\)\)",
        r"(\1)/((\2)^{\3})",
        value,
    )
    value = re.sub(
        r"e\^\{([^{}]+)\}_\{-\}([A-Za-z])\^\{([^{}]+)\}",
        r"e^{\1 - \2^{\3}}",
        value,
    )
    value = re.sub(r"\bk\((\d+)\)/\(([^()]+)\)", r"(k^{\1})/(\2)", value)
    value = re.sub(
        r"(\d+π)([A-Za-z])\((\d+)\s*\+\s*(\d+)\)/\(\2\)",
        r"\1\2^{\3} + (\4)/(\2)",
        value,
    )
    value = re.sub(r"\(--→\)/\(p([A-Z]{2})\)", r"p\\overrightarrow{\1}", value)
    value = re.sub(r"--→([A-Z]{2})", r"\\overrightarrow{\1}", value)
    value = re.sub(
        r"(\bposition vectors given by)\s*\d{1,2}\s+(?=\\overrightarrow\{OA\})",
        r"\1 ",
        value,
    )
    value = re.sub(
        r"^(\d+)\s+z\s+(?=The diagram shows a set of rectangular axes)",
        r"\1 ",
        value,
    )
    value = re.sub(
        r"\(([A-Za-z])\s+\1\s*\+\s*1\)/",
        r"(\1(\1 + 1))/",
        value,
    )
    value = re.sub(r"\(\)\s*d([ux])\b", r" d\1", value)
    value = re.sub(
        r"\b(sin|cos|tan|sec|cosec|cot)\s+2\^\{1\}([A-Za-zθ])",
        r"\1((1)/(2)\2)",
        value,
    )
    value = re.sub(
        r"\b(sin|cos|tan|sec|cosec|cot)\s+\(1\)/\(2\)([A-Za-zθ])",
        r"\1((1)/(2)\2)",
        value,
    )
    value = re.sub(r"\b(\d+)\^\{1\}(?=\()", r"(1)/(\1)", value)
    value = re.sub(
        r"(cos|sin|tan)\^\{-1\}\(\((\d+)√\)/\((\d+)\)(\d+)\)",
        r"\1^{-1}((\2)/(\3)√(\4))",
        value,
    )
    value = value.replace("√(s)in x", "√(sin x)")
    value = re.sub(
        r"([πθA-Za-z0-9}])\s*-\s*(?=(?:sin|cos|tan|sec|cosec|cot)\b)",
        r"\1 - ",
        value,
    )
    value = re.sub(r"e\^\{(-?[A-Za-z])\}_\{([A-Za-z])\}", r"e^{\1_{\2}}", value)
    value = re.sub(r"e\^\{2\}a(?=\s*-\s*4e\^\{a\})", r"e^{2a}", value)
    value = re.sub(
        r"\((1\s*-\s*e\^\{[^}]+\}),\s*for\)\s*x",
        r"(\1), for x",
        value,
    )
    value = re.sub(
        r"([A-Za-z0-9])\^\{([^{}]+)\}\s*-\s*(\d+[A-Za-z])\s*(?==)",
        r"\1^{\2 - \3} ",
        value,
    )
    value = re.sub(
        r"#(\d+)\^\{(-\d+)\}\s*([A-Za-z])",
        r" × \1^{\2\3}",
        value,
    )
    value = re.sub(
        r"(\([^()\n]+\))(\d+)\^\{(\d+)\}",
        r"\1^{(\3)/(\2)}",
        value,
    )
    value = re.sub(
        r"([=+\-])\s*(\d+)\s+([A-Za-z])(?=\^\{)",
        r"\1 \2\3",
        value,
    )
    value = re.sub(r"(iterative formula)\s+\(\)\s+", r"\1 ", value)
    value = re.sub(
        r"\(([^()]+)\)/\(([A-Za-z])\)\s+([A-Za-z])(?=\s+to determine)",
        r"(\1)/(\2_{\3})",
        value,
    )
    value = re.sub(
        r"\bexp\s+([^.,]+?)(?=\s+to determine)",
        r"exp(\1)",
        value,
    )
    value = re.sub(
        r"(crosses the)(\([^.;\n]+\)\.)(\s*x-axis.*?passes through the point)\s*"
        r"(?=\([ivx]+\))",
        r"\1\3 \2 ",
        value,
        flags=re.DOTALL,
    )
    return re.sub(r"[ \t]+", " ", value).strip()


def _replace_probability_power_fraction(match: re.Match[str]) -> str:
    denominator = int(match.group(2))
    numerator = int(match.group(3))
    if denominator <= numerator:
        return match.group(0)
    return f"{match.group(1)}({numerator})/({denominator})"


def _relocate_misplaced_degree(text: str) -> str:
    if "_{°}" not in text:
        return text
    value = text.replace("_{°}", "", 1)
    patterns = (
        r"(\bangle(?:\s+[A-Z]{3})?\s+is\s+)(\d+|[αθ])\b(?!°)",
        r"(\bangle\s+of\s+)(\d+|[αθ])\b(?!°)",
    )
    for pattern in patterns:
        repaired, count = re.subn(pattern, r"\1\2°", value, count=1)
        if count:
            return repaired
    return value


def _repair_caie_math_delimiters(text: str) -> str:
    value = text
    value = re.sub(r"@([^@\n]{1,80})A_\{(\d{1,2})\}", _replace_at_power_delimiter, value)
    value = re.sub(r"@([^@\n]{1,80})A", _replace_at_delimiter, value)
    value = re.sub(r"\bb([^()\n]{1,80})l(\^\{\d{1,2}\})", _replace_bl_power_delimiter, value)
    value = re.sub(r"`([^`\n]{1,80})j(\^\{\d{1,2}\})", _replace_backtick_power_delimiter, value)
    value = re.sub(r"`([^`\n]{1,80})j", _replace_backtick_delimiter, value)
    return value


def _replace_at_power_delimiter(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not _looks_like_compacted_math_inner(inner):
        return match.group(0)
    return f"({inner})^{{{match.group(2)}}}"


def _replace_at_delimiter(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not _looks_like_compacted_math_inner(inner):
        return match.group(0)
    return f"({inner})"


def _replace_bl_power_delimiter(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not _looks_like_compacted_math_inner(inner):
        return match.group(0)
    return f"({inner}){match.group(2)}"


def _replace_backtick_power_delimiter(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not _looks_like_compacted_math_inner(inner):
        return match.group(0)
    return f"({inner}){match.group(2)}"


def _replace_backtick_delimiter(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not _looks_like_compacted_math_inner(inner):
        return match.group(0)
    return f"({inner})"


def _looks_like_compacted_math_inner(value: str) -> bool:
    inner = value.strip()
    if not inner or len(inner.split()) > 10:
        return False
    return bool(
        re.search(r"[0-9][A-Za-z]|[A-Za-z][0-9]|[=+\-*/^_{}]|(?:sin|cos|tan|ln|log)\b", inner, re.IGNORECASE)
        and re.search(r"[0-9A-Za-z]", inner)
    )


def _repair_trig_theta_placeholders(text: str) -> str:
    value = text
    trig = r"cosec|sin|cos|tan|sec|cot"
    value = re.sub(rf"\b({trig})i(?=(?:{trig}))", r"\1 θ ", value, flags=re.IGNORECASE)
    value = re.sub(rf"\b({trig})(\s*(?:\^\{{(?!-\}})[^}}]+\}})?)\s*i\b", r"\1\2 θ", value, flags=re.IGNORECASE)
    value = re.sub(rf"\b({trig})\s*i\b", r"\1 θ", value, flags=re.IGNORECASE)
    value = re.sub(rf"\b({trig})\s*!", r"\1 θ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bangle\s+of\s+!", "angle of θ", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(angle\s+[A-Z]{2,4}\s*=\s*)i(?=\s*radians\b)", r"\1θ", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(angle\s+[A-Z]{2,4}\s+(?:is|=)\s*)i(?=\s*(?:radians|°)\b)", r"\1θ", value, flags=re.IGNORECASE)
    value = re.sub(r"\b([A-Z]{2,4}\s*=\s*)i(?=°)", r"\1θ", value)
    value = re.sub(r"\b(angle\s*)°i\b", r"\1θ°", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<=\bangle\s)°i(?=\s+to\b)", "θ°", value, flags=re.IGNORECASE)
    value = re.sub(r"(-?\s*\d{1,3}°)\s*1\s*i\s*1\s*(\d{1,3}°)", r"\1 < θ < \2", value)
    value = re.sub(r"\b0\s*1\s*i\s*1\s*(\^\{1\}_\{2\}π)", r"0 < θ < \1", value)
    value = re.sub(r"\b0\s*1\s*i\s*1\s*(π)", r"0 < θ < \1", value)
    if re.search(r"\bangle\s+[A-Z]{2,4}\s*=\s*θ\s*radians\b", value, re.IGNORECASE):
        value = re.sub(r"\b(value of )i\b", r"\1θ", value, flags=re.IGNORECASE)
    if re.search(r"\b(?:sin|cos|tan|sec|cosec|cot)(?:\s*(?:\^\{(?!-\})[^}]+\})?)\s*θ\b", value, re.IGNORECASE):
        value = re.sub(r"\b(value of )i\b", r"\1θ", value, flags=re.IGNORECASE)
    if re.search(r"\b(?:angle\s+[A-Z]{2,4}\s+(?:is|=)\s*θ\s*(?:radians|°)|[A-Z]{2,4}\s*=\s*θ°|angle\s+θ°)", value, re.IGNORECASE):
        value = re.sub(r"\b(value of )i\b", r"\1θ", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(0|90|180|360)(°?)\s*<\s*i\s*<", r"\1\2 < θ <", value)
    value = re.sub(r"\b(0|90|180|360)(°?)\s*≤\s*i\s*≤", r"\1\2 ≤ θ ≤", value)
    value = re.sub(r"\b0\s*<\s*i\s*<", "0 < θ <", value)
    value = re.sub(r"\b0\s*≤\s*i\s*≤", "0 ≤ θ ≤", value)
    value = re.sub(r"(?<=\d)c\b", "°", value)
    return value


def _repair_common_joined_words(text: str) -> str:
    value = text
    replacements = [
        (r"\bThediagramshows", "The diagram shows"),
        (r"\bThediagram", "The diagram"),
        (r"\bthediagramshows", "the diagram shows"),
        (r"\bthediagram", "the diagram"),
        (r"\bFindthe(?=\b|[a-z])", "Find the"),
        (r"\bfindthe(?=\b|[a-z])", "find the"),
        (r"\bGivethe(?=\b|[a-z])", "Give the"),
        (r"\bgivethe(?=\b|[a-z])", "give the"),
        (r"\bGiveyour(?=\b|[a-z])", "Give your"),
        (r"\bgiveyour(?=\b|[a-z])", "give your"),
        (r"\bByfirst\b", "By first"),
        (r"\bbyfirst\b", "by first"),
        (r"Bysketchingasuitable", "By sketching a suitable"),
        (r"bysketchingasuitable", "by sketching a suitable"),
        (r"\bfirstexpressing\b", "first expressing"),
        (r"\bfirstexpanding\b", "first expanding"),
        (r"ofgraphs,showthattheequation", "of graphs, show that the equation"),
        (r"showthattheequation", "show that the equation"),
        (r"\bmaybeexpressedintheforma\b", "may be expressed in the form a"),
        (r"\btheequation\b", "the equation"),
        (r"\bsolvethe\b", "solve the"),
        (r"\bintheform\b", "in the form"),
        (r"\banswerintheform(?=\b|[a-z])", "answer in the form"),
        (r"\bthevalue(?=\b|[a-z])", "the value"),
        (r"\bvalueof([A-Za-z])\b", r"value of \1"),
        (r"\bthevalueof(?=\b|[a-z])", "the value of"),
        (r"\byouranswer(?=\b|[a-z])", "your answer"),
        (r"\bwhereaandbare\b", "where a and b are"),
        (r"\bwhereaandb\b", "where a and b"),
        (r"\bandbare\b", "and b are"),
        (r"\bthewater\b", "the water"),
        (r"\binthetank\b", "in the tank"),
        (r"\bshowthat\b", "show that"),
        (r"\bmaybe\b", "may be"),
        (r"\bStatethe\b", "State the"),
        (r"\bstatethe\b", "state the"),
        (r"\bandshows", "and shows"),
        (r"\band shows(?=[A-Z])", "and shows "),
        (r"\bgraphthe\b", "graph the"),
        (r"\bgraphof\b", "graph of"),
        (r"\bisof\b", "is of"),
        (r"\bfactor is in g\b", "factorising"),
        (r"\bm or e\b", "more"),
        (r"\bbe in g\b", "being"),
        (r"\bkeep in g\b", "keeping"),
        (r"\bexpress i on\b", "expression"),
        (r"\bthethe\b", "the"),
        (r"\btheformgraph", "the form graph"),
        (r"\btwostraight", "two straight"),
        (r"\bAfairspinner", "A fair spinner"),
        (r"\bfairspinner\b", "fair spinner"),
        (r"\bsidesnumbered\b", "sides numbered"),
        (r"\bisspun\b", "is spun"),
        (r"\bandisspun\b", "and is spun"),
        (r"\bscoreon\b", "score on"),
        (r"\bsideonwhich\b", "side on which"),
        (r"\bcomestorest\b", "comes to rest"),
        (r"\bAtacompany\b", "At a company"),
        (r"\bThereisaresistancetothemotionoftheblock\b", "There is a resistance to the motion of the block"),
        (r"\bwhichthecranedoes\b", "which the crane does"),
        (r"ofworkto\b", "of work to"),
        (r"\bGiventhattheaveragepowerexertedbythecraneis\b", "Given that the average power exerted by the crane is"),
        (r"\bthetotaltimeforwhichthe\b", "the total time for which the"),
        (r"\bForanothercompetition\b", "For another competition"),
        (r"\bForanother\b", "For another"),
        (r"\bateamof\b", "a team of"),
        (r"\bconsistsof\b", "consists of"),
        (r"\bUseanenergy\b", "Use an energy"),
        (r"\bmethodtofindthe\b", "method to find the"),
        (r"\bcoeﬃcientoffrictionbetweenthe\b", "coeﬃcient of friction between the"),
        (r"\bcoefficientoffrictionbetweenthe\b", "coefficient of friction between the"),
    ]
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"(?<=[0-9])(?=Byfirst)", " ", value)
    value = re.sub(r"\bByfirst(?=[a-z])", "By first", value)
    value = re.sub(r"(?<=[a-z])(?=Express\b)", " ", value)
    value = re.sub(r"(?<=[a-z])(?=Expand\b)", " ", value)
    value = re.sub(r"(?<=[a-z])(?=Solve\b)", " ", value)
    value = re.sub(r"\bquadratic equationin\b", "quadratic equation in ", value)
    value = _repair_joined_prose_tokens(value)
    value = re.sub(r"\bnumbered\s*(\d+)to(\d+)andisspun\b", r"numbered \1 to \2 and is spun", value)
    value = re.sub(r",(?=which\b)", ", ", value)
    value = re.sub(r"(?<=J)of work", " of work", value)
    value = re.sub(r"\b(for which|magnitude)(?=\d)", r"\1 ", value)
    value = re.sub(
        r"\b(mass|radius|length|height|speed|period|magnitude|force|value|rate|distance|of|at|to|for|is|does)(?=\d)",
        r"\1 ",
        value,
    )
    value = re.sub(r"\bpart(?=\([a-z]\))", "part ", value)
    value = re.sub(r"\)(?=to\b)", ") ", value)
    value = re.sub(r"\)\s*on\s+to\b", ") onto", value)
    value = re.sub(r"\)(?=onto\b)", ") ", value)
    value = re.sub(r"(?<=[A-Za-z0-9]\.)(?=[A-Z][a-z])", " ", value)
    return value


_JOINED_PROSE_TOKEN_RE = re.compile(r"(?<![A-Za-z])([A-Za-z]{10,})(?=[0-9(]|\b)")
_JOINED_PROSE_WORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "acceleration",
        "accelerates",
        "according",
        "against",
        "again",
        "along",
        "all",
        "also",
        "an",
        "and",
        "angle",
        "another",
        "answer",
        "applied",
        "applying",
        "arc",
        "are",
        "area",
        "arithmetic",
        "as",
        "at",
        "attempt",
        "attempts",
        "average",
        "bag",
        "based",
        "be",
        "before",
        "between",
        "bicycle",
        "blue",
        "block",
        "both",
        "bottom",
        "bounded",
        "boundary",
        "broadband",
        "by",
        "calculate",
        "can",
        "car",
        "caravan",
        "care",
        "centre",
        "circle",
        "circular",
        "coefficient",
        "collide",
        "collision",
        "combined",
        "common",
        "competition",
        "complex",
        "connected",
        "constant",
        "consists",
        "coordinate",
        "coordinates",
        "coming",
        "comes",
        "correct",
        "crane",
        "curve",
        "curved",
        "daily",
        "day",
        "decelerating",
        "defined",
        "denoted",
        "depth",
        "decreasing",
        "describe",
        "determine",
        "diagram",
        "dice",
        "differential",
        "direction",
        "distance",
        "distances",
        "divides",
        "down",
        "driving",
        "each",
        "axis",
        "chosen",
        "cm",
        "end",
        "ends",
        "energy",
        "engine",
        "equation",
        "equal",
        "event",
        "exact",
        "exactly",
        "being",
        "chooses",
        "eats",
        "exerted",
        "expanding",
        "expression",
        "express",
        "expressing",
        "fair",
        "families",
        "find",
        "first",
        "fixed",
        "force",
        "for",
        "factor",
        "factorising",
        "formula",
        "form",
        "from",
        "friction",
        "function",
        "geometric",
        "given",
        "giving",
        "good",
        "graph",
        "greatest",
        "has",
        "have",
        "height",
        "hence",
        "hill",
        "horizontal",
        "household",
        "identity",
        "if",
        "in",
        "inclined",
        "including",
        "independent",
        "initial",
        "initially",
        "instant",
        "instantaneous",
        "integer",
        "integers",
        "interval",
        "into",
        "increased",
        "increase",
        "infected",
        "inextensible",
        "is",
        "it",
        "iterative",
        "its",
        "keep",
        "keeping",
        "kg",
        "large",
        "later",
        "length",
        "level",
        "line",
        "lines",
        "lies",
        "light",
        "load",
        "long",
        "made",
        "magnitude",
        "marbles",
        "male",
        "mass",
        "maximum",
        "means",
        "member",
        "method",
        "metres",
        "minimum",
        "modelled",
        "models",
        "more",
        "motion",
        "moves",
        "moving",
        "normal",
        "number",
        "numbered",
        "numbers",
        "obtain",
        "of",
        "on",
        "once",
        "one",
        "only",
        "or",
        "original",
        "other",
        "onto",
        "over",
        "overcome",
        "particle",
        "particles",
        "part",
        "pass",
        "passes",
        "period",
        "perimeter",
        "perpendicular",
        "piece",
        "pieces",
        "plane",
        "plate",
        "plays",
        "leopard",
        "point",
        "points",
        "possible",
        "positive",
        "power",
        "probability",
        "progression",
        "projected",
        "proportional",
        "prove",
        "pull",
        "pumped",
        "random",
        "randomly",
        "rate",
        "ratio",
        "reaches",
        "real",
        "red",
        "reflected",
        "released",
        "removing",
        "representing",
        "required",
        "resistance",
        "respectively",
        "rest",
        "result",
        "resulting",
        "riding",
        "road",
        "roots",
        "same",
        "scarf",
        "second",
        "score",
        "section",
        "sector",
        "segment",
        "segments",
        "sequence",
        "service",
        "shaded",
        "show",
        "shows",
        "side",
        "sides",
        "single",
        "smooth",
        "solve",
        "speed",
        "sphere",
        "spins",
        "spinner",
        "springs",
        "spun",
        "square",
        "scale",
        "straight",
        "stretched",
        "string",
        "student",
        "students",
        "subsequent",
        "such",
        "sum",
        "table",
        "takes",
        "tangent",
        "term",
        "terms",
        "test",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "three",
        "threaded",
        "through",
        "time",
        "times",
        "to",
        "total",
        "towards",
        "track",
        "transformation",
        "transformations",
        "travel",
        "travels",
        "travelling",
        "triangle",
        "two",
        "until",
        "up",
        "use",
        "used",
        "value",
        "variable",
        "variables",
        "vertical",
        "vertically",
        "velocity",
        "volume",
        "when",
        "where",
        "which",
        "while",
        "wire",
        "with",
        "work",
        "written",
        "your",
        "yellow",
        "answers",
        "allowed",
        "after",
        "argand",
        "been",
        "change",
        "colours",
        "crosses",
        "different",
        "differentiate",
        "directly",
        "do",
        "does",
        "drawing",
        "fail",
        "fully",
        "give",
        "gradient",
        "heights",
        "identical",
        "industrial",
        "locus",
        "metal",
        "no",
        "not",
        "operates",
        "players",
        "pulled",
        "pulley",
        "pulling",
        "rod",
        "rope",
        "rough",
        "region",
        "resultant",
        "rigid",
        "satisfying",
        "she",
        "shown",
        "sketch",
        "sloping",
        "symmetrical",
        "transformed",
        "turn",
        "twice",
        "wears",
        "weighs",
        "winch",
        "working",
        "map",
        "maps",
        "tank",
        "twenty",
        "water",
        "jacob",
        "georgie",
        "george",
        "isabella",
        "maria",
        "alisa",
        "sharma",
    }
)
_JOINED_SINGLE_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")
_JOINED_MAX_WORD_LEN = max(len(word) for word in _JOINED_PROSE_WORDS)


def _repair_joined_prose_tokens(text: str) -> str:
    return _JOINED_PROSE_TOKEN_RE.sub(lambda match: _repair_joined_prose_token(match.group(0)), text)


def _repair_joined_prose_token(token: str) -> str:
    if token.isupper():
        return token
    lowered = token.lower()
    if lowered in _JOINED_PROSE_WORDS:
        return token

    spans = _joined_prose_token_spans(lowered)
    if not spans or len(spans) < 2:
        return token
    pieces = [token[start:end] for start, end in spans]
    if sum(1 for piece in pieces if len(piece) > 1) < 2:
        return token
    if _has_consecutive_single_letter_segments(pieces):
        return token
    return " ".join(pieces)


def _has_consecutive_single_letter_segments(pieces: list[str]) -> bool:
    run = 0
    for piece in pieces:
        if len(piece) == 1:
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def _joined_prose_token_spans(lowered: str) -> list[tuple[int, int]] | None:
    @lru_cache(maxsize=None)
    def best(index: int) -> tuple[float, tuple[tuple[int, int], ...]] | None:
        if index == len(lowered):
            return 0.0, ()

        best_result: tuple[float, tuple[tuple[int, int], ...]] | None = None
        max_end = min(len(lowered), index + _JOINED_MAX_WORD_LEN)
        for end in range(max_end, index, -1):
            word = lowered[index:end]
            if word not in _JOINED_PROSE_WORDS and not _is_joined_single_letter(word):
                continue
            suffix = best(end)
            if suffix is None:
                continue
            suffix_score, suffix_spans = suffix
            score = _joined_word_score(word) + suffix_score
            candidate = (score, ((index, end),) + suffix_spans)
            if best_result is None or candidate[0] > best_result[0]:
                best_result = candidate
        return best_result

    result = best(0)
    if result is None:
        return None
    score, spans = result
    if score < len(lowered) * 0.55:
        return None
    return list(spans)


def _is_joined_single_letter(word: str) -> bool:
    return len(word) == 1 and word in _JOINED_SINGLE_LETTERS


def _joined_word_score(word: str) -> float:
    if word == "a":
        return 1.5
    if len(word) == 1:
        return -5.0
    if len(word) == 2:
        return 1.0
    return float(len(word) * 2 - 1)


def _extract_math_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_light(raw_line)
        if not line:
            continue
        if _line_is_math_heavy(line):
            lines.append(line)
    return lines


def _line_is_math_heavy(line: str) -> bool:
    math_tokens = len(_MATH_TOKEN_RE.findall(line))
    symbol_count = len(re.findall(r"[=<>^/√πθ∫Σ()+\-]", line))
    alpha_count = sum(char.isalpha() for char in line)
    digit_count = sum(char.isdigit() for char in line)
    return math_tokens >= 1 or symbol_count >= 3 or (digit_count >= 2 and alpha_count >= 2 and "=" in line)


def _extraction_quality_flags(
    body_text_raw: str,
    body_text_normalized: str,
    math_lines: list[str],
    diagram_lines: list[str],
) -> list[str]:
    flags: list[str] = []
    if len(body_text_normalized) < 24 or body_text_normalized.count("\n") == 0:
        flags.append("weak_text_structure")
    if math_lines and len(body_text_normalized.splitlines()) <= 2:
        flags.append("flattened_display_math")
    if diagram_lines and re.search(r"\n(?:[A-Z](?:\s+[A-Z]){1,}|x\s+y|\d+\s+cm)\n", f"\n{body_text_raw}\n"):
        flags.append("diagram_text_mixed_with_body")
    if _unmatched_parentheses(body_text_normalized):
        flags.append("broken_fraction_structure")
    if re.search(r"\b(?:sin|cos|tan|sec|cosec|cot)\s+[A-Za-zθ]\s+\d\b", body_text_normalized) or re.search(
        r"\b(?:ln|log)\s+(?:ln|log)\b", body_text_normalized
    ):
        flags.append("broken_superscript_or_power")
    if _SUSPICIOUS_SYMBOL_RUN_RE.search(body_text_normalized):
        flags.append("suspicious_symbol_run")
    if len(math_lines) >= max(2, len(body_text_normalized.splitlines()) // 2):
        flags.append("heavy_math_density")
    if any(flag in flags for flag in {"broken_superscript_or_power", "broken_fraction_structure", "suspicious_symbol_run"}):
        flags.append("math_corruption_suspected")
    if "math_corruption_suspected" in flags or ("diagram_text_mixed_with_body" in flags and "heavy_math_density" in flags):
        flags.append("likely_needs_visual_review")
    return sorted(set(flags))


def _unmatched_parentheses(text: str) -> bool:
    opens = text.count("(") + text.count("[") + text.count("{")
    closes = text.count(")") + text.count("]") + text.count("}")
    return opens != closes


def _quality_score(flags: list[str]) -> float:
    score = 1.0
    penalties = {
        "weak_text_structure": 0.18,
        "flattened_display_math": 0.14,
        "diagram_text_mixed_with_body": 0.12,
        "broken_superscript_or_power": 0.14,
        "broken_fraction_structure": 0.14,
        "suspicious_symbol_run": 0.18,
        "heavy_math_density": 0.08,
        "math_corruption_suspected": 0.12,
        "likely_needs_visual_review": 0.08,
    }
    for flag in flags:
        score -= penalties.get(flag, 0.0)
    return max(0.05, min(1.0, score))


def _part_texts(body_text_raw: str) -> list[dict[str, object]]:
    lines = body_text_raw.splitlines()
    if not any(_PART_LINE_RE.match(line) for line in lines):
        return []

    parts: list[dict[str, object]] = []
    current_label = ""
    current_lines: list[str] = []
    for line in lines:
        match = _PART_LINE_RE.match(line)
        if match:
            if current_label and current_lines:
                raw = "\n".join(current_lines).strip()
                parts.append(
                    {
                        "part_label": current_label,
                        "raw_text": raw,
                        "normalized_text": _normalize_preserving_structure(raw),
                        "math_lines": _extract_math_lines(raw),
                    }
                )
            current_label = match.group("label").lower()
            current_lines = [line]
        elif current_label:
            current_lines.append(line)
    if current_label and current_lines:
        raw = "\n".join(current_lines).strip()
        parts.append(
            {
                "part_label": current_label,
                "raw_text": raw,
                "normalized_text": _normalize_preserving_structure(raw),
                "math_lines": _extract_math_lines(raw),
            }
        )
    return parts


def _layout_by_number(layouts: list[PageLayout], page_number: int) -> PageLayout:
    for layout in layouts:
        if layout.page_number == page_number:
            return layout
    raise KeyError(f"Missing layout for page {page_number}")
