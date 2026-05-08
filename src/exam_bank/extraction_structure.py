from __future__ import annotations

from dataclasses import dataclass
import re

from .config import AppConfig
from .models import BoundingBox, PageLayout, QuestionSpan, TextBlock
from .question_detection_layout import distance_to_box as _distance_to_box
from .question_detection_layout import looks_like_diagram_axis_or_label_text as _shared_looks_like_diagram_axis_or_label_text


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
    diagram_lines: list[str] = []
    body_lines: list[str] = []

    for line in lines:
        layout = _layout_by_number(layouts, line.page_number)
        split_anchor = _split_question_anchor_diagram_label(line, layout, span, config)
        if split_anchor is not None:
            body_text, diagram_text = split_anchor
            if not _body_already_has_question_anchor(body_lines, body_text):
                body_lines.append(body_text)
            diagram_lines.append(diagram_text)
            continue
        if _is_duplicate_question_number_diagram_label(line, layout, span, body_lines, config):
            diagram_lines.append(line.text)
            continue
        if _looks_like_answer_filler_line(line.text):
            continue
        if _looks_like_diagram_text(line, layout, span, config):
            diagram_lines.append(line.text)
        else:
            body_lines.append(line.text)

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


def _looks_like_diagram_text(line: _LineItem, layout: PageLayout, span: QuestionSpan, config: AppConfig) -> bool:
    cleaned = _normalize_light(line.text)
    if not cleaned:
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
    if line.bbox.x0 <= config.detection.question_start_max_x + 40:
        return False
    return any(_distance_to_box(line.bbox, graphic) <= 32 for graphic in layout.graphics)


def _looks_like_answer_filler_line(text: str) -> bool:
    cleaned = _normalize_light(text)
    if not cleaned:
        return False
    if re.match(r"^(?:[._\-–—]\s*){12,}", cleaned):
        return True
    visible_alnum = len(re.findall(r"[A-Za-z0-9]", cleaned))
    filler_count = len(re.findall(r"[._\-–—]", cleaned))
    return filler_count >= 24 and visible_alnum <= 4


def _normalize_preserving_structure(text: str) -> str:
    normalized_lines = [_normalize_light(line) for line in text.splitlines()]
    return "\n".join(line for line in normalized_lines if line).strip()


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
        "\ufb01": "fi",
        "\ufb02": "fl",
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
        "": "≡",
        "Ó": "∫",
        "Ô": "∫",
        "Å": "°",
        "": "(",
        "": ")",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    # This control character is often a radical marker in the question papers,
    # but not reliably enough to replace globally. Keep the high-signal case
    # from substitution prompts and otherwise remove the noise character.
    value = re.sub(r"(\bu\s*=\s*)\x0f(?=\s*x\b)", r"\1√", value)
    value = value.replace("\x0f", "")

    value = _normalize_common_math_ocr_substitutions(value)
    value = _repair_common_joined_words(value)
    return value


def _normalize_common_math_ocr_substitutions(text: str) -> str:
    value = text
    value = re.sub(r"\b([A-Za-z])\(([^()\[\]]+)\[(\d{1,2})\]\)([.?!])", r"\1(\2)\4 [\3]", value)
    value = re.sub(r"\b([A-Za-z])\(([^()\[\]]+)\[(\d{1,2})\]\)", r"\1(\2) [\3]", value)
    value = re.sub(r"(\[\d{1,2}\])\s*(?:[._\-–—]\s*){12,}.*$", r"\1", value)
    value = re.sub(r"\b([0-9])\s*G\s*([xXiIθ])", r"\1 ≤ \2", value)
    value = re.sub(r"\b([xXiIθ])\s*G\s*(\^\{[^}]+\}|[0-9])", r"\1 ≤ \2", value)
    value = re.sub(r"\b([0-9])G([A-Za-zθ])G([0-9])\b", r"\1 ≤ \2 ≤ \3", value)
    value = re.sub(r"\br20\b", "r > 0", value)
    value = re.sub(r"(?<=[0-9}_])r\b", "π", value)
    value = re.sub(r"\b(0|90|180|360)°?\s*<\s*1\s*<", r"\1° < θ <", value)
    value = re.sub(r"\b(0|90|180|360)°?\s*≤\s*1\s*≤", r"\1° ≤ θ ≤", value)
    value = re.sub(r"\b0\s*<\s*1\s*<", "0 < θ <", value)
    value = re.sub(r"\b0\s*≤\s*1\s*≤", "0 ≤ θ ≤", value)
    value = re.sub(r"\b(sin|cos|tan|sec|cosec|cot)\(\s*1(?=\s*[-+])", r"\1(θ", value)
    value = re.sub(r"\b(sin|cos|tan|sec|cosec|cot)(\s*(?:\^\{(?!-\})[^}]+\})?)\s*1\b", r"\1\2 θ", value)
    value = re.sub(r"\b(sin|cos|tan|sec|cosec|cot)(?=[0-9θxyz])", r"\1 ", value)
    value = re.sub(r"\b(ln|log)(?=[A-Za-z0-9(])", r"\1 ", value)
    value = re.sub(r"\b(ln|log)\s+\(", r"\1(", value)
    value = re.sub(r"(?<=[A-Za-z0-9}])(?=(?:cosec|sin|cos|tan|sec|cot)(?:[0-9θxyz]|\b))", " ", value)
    value = re.sub(r"\b(sin|cos|tan|sec|cosec|cot)(?=[0-9θxyz])", r"\1 ", value)
    value = re.sub(r"\b(cosec|sin|cos|tan|sec|cot|ln|log)\s+([0-9]+)([A-Za-zθ])\b", r"\1 \2\3", value)
    value = re.sub(r"\b(tan|sin|cos|sec|cosec|cot)\^\{-\}\s*θ", r"\1^{-}1", value)
    return value


def _repair_common_joined_words(text: str) -> str:
    value = text
    replacements = [
        (r"\bFindthe(?=\b|[a-z])", "Find the"),
        (r"\bfindthe(?=\b|[a-z])", "find the"),
        (r"\bGivethe(?=\b|[a-z])", "Give the"),
        (r"\bgivethe(?=\b|[a-z])", "give the"),
        (r"\bGiveyour(?=\b|[a-z])", "Give your"),
        (r"\bgiveyour(?=\b|[a-z])", "give your"),
        (r"\bByfirst\b", "By first"),
        (r"\bbyfirst\b", "by first"),
        (r"\bfirstexpressing\b", "first expressing"),
        (r"\bfirstexpanding\b", "first expanding"),
        (r"\btheequation\b", "the equation"),
        (r"\bsolvethe\b", "solve the"),
        (r"\bintheform\b", "in the form"),
        (r"\banswerintheform(?=\b|[a-z])", "answer in the form"),
        (r"\bthevalue(?=\b|[a-z])", "the value"),
        (r"\byouranswer(?=\b|[a-z])", "your answer"),
        (r"\bwhereaandbare\b", "where a and b are"),
        (r"\bwhereaandb\b", "where a and b"),
        (r"\bandbare\b", "and b are"),
        (r"\bshowthat\b", "show that"),
        (r"\bmaybe\b", "may be"),
        (r"\bStatethe\b", "State the"),
        (r"\bstatethe\b", "state the"),
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
    return value


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
    if _unmatched_parentheses(body_text_raw):
        flags.append("broken_fraction_structure")
    if re.search(r"\b(?:sin|cos|tan|sec|cosec|cot)\s+[A-Za-zθ]\s+\d\b", body_text_raw) or re.search(
        r"\b(?:ln|log)\s+(?:ln|log)\b", body_text_raw
    ):
        flags.append("broken_superscript_or_power")
    if _SUSPICIOUS_SYMBOL_RUN_RE.search(body_text_raw):
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
