from __future__ import annotations

import re

from .config import AppConfig
from .models import BoundingBox, PageLayout


_PROMPT_WORD_RE = re.compile(
    r"\b(?:find|show|calculate|state|determine|prove|sketch|draw|describe|given|hence|use|solve|evaluate)\b",
    re.IGNORECASE,
)
_SIMPLE_DIAGRAM_LABEL_RE = re.compile(
    r"^(?:[A-Z](?:\^\{[′']\}|[′'])?(?:\s+[A-Z](?:\^\{[′']\}|[′'])?){0,4}|[xyXY]|θ|π|"
    r"\d+(?:\.\d+)?\s*(?:cm|mm|m|kg|rad)|\d+(?:\.\d+)?°?)$"
)
_GRAPH_EQUATION_LABEL_RE = re.compile(
    r"^(?:-?\d+(?:\.\d+)?\s+)?[xyXY]\s*=\s*[-+A-Za-z0-9πθ().{}\[\]^_/*\s]+$"
)
_AXIS_UNIT_LABEL_RE = re.compile(
    r"^(?:-?\d+(?:\.\d+)?\s+)?[A-Za-z]\s*\([^)]{1,40}\)$"
)


def clean_layout_text(text: str) -> str:
    return " ".join(strip_control_chars(text).replace("\u00a0", " ").replace("−", "-").split())


def strip_control_chars(text: str) -> str:
    return "".join(char if ord(char) >= 32 or char in "\n\t\r" else " " for char in text)


def looks_like_diagram_axis_or_label_text(text: str) -> bool:
    cleaned = clean_layout_text(text)
    if not cleaned:
        return False
    if len(cleaned) > 110:
        return False
    if re.search(r"\[\d{1,2}\]", cleaned):
        return False
    if _SIMPLE_DIAGRAM_LABEL_RE.match(cleaned):
        return True
    if _GRAPH_EQUATION_LABEL_RE.fullmatch(cleaned):
        return not _PROMPT_WORD_RE.search(cleaned)
    if _AXIS_UNIT_LABEL_RE.fullmatch(cleaned):
        return True
    if re.fullmatch(r"(?:-?\d+(?:\.\d+)?\s*){2,}[xyXY]?", cleaned):
        return True
    if re.fullmatch(r"(?:O\s+)?(?:-?\d+(?:\.\d+)?|π|(?:\d+_?\{?\d+\}?π))(\s+(?:-?\d+(?:\.\d+)?|π|(?:\d+_?\{?\d+\}?π))){1,}\s*[xyXY]?", cleaned):
        return True
    if re.fullmatch(r"\d+_\{\d+\}π", cleaned):
        return True
    if re.fullmatch(r"(?:(?:Bag|Box)\s+[A-Z](?:\s+|$)){2,}\d?", cleaned):
        return True

    alpha_words = re.findall(r"[A-Za-z]{2,}", cleaned)
    non_unit_words = [word for word in alpha_words if word.lower() not in {"cm", "mm", "kg", "rad"}]
    has_diagram_symbols = bool(re.search(r"[0-9=π]|[A-Z]", cleaned))
    if not non_unit_words and has_diagram_symbols:
        return True
    return False


def line_position_supports_diagram_label(
    *,
    text: str,
    bbox: BoundingBox | None,
    layout: PageLayout | None,
    config: AppConfig,
) -> bool:
    if not looks_like_diagram_axis_or_label_text(text):
        return False
    if bbox is None or layout is None:
        return True
    near_graphic = any(distance_to_box(bbox, graphic) <= 30 for graphic in layout.graphics)
    right_of_question_anchor = bbox.x0 > config.detection.question_start_max_x + 20
    return near_graphic or right_of_question_anchor


def distance_to_box(a: BoundingBox, b: BoundingBox) -> float:
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0)
    return max(dx, dy)
