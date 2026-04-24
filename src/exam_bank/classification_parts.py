from __future__ import annotations

import re
from typing import Any

from .classification_models import QuestionPartSegment


_ALPHA_PART_ANCHOR_RE = re.compile(
    r"(?m)^(?P<prefix>\s*(?:(?P<question>\d{1,2})\s*)?\((?P<label>[a-h])\)(?:\s*\([ivx]+\))?\s*)",
    re.IGNORECASE,
)
_ROMAN_PART_ANCHOR_RE = re.compile(
    r"(?m)^(?P<prefix>\s*(?:(?P<question>\d{1,2})\s*)?\((?P<label>i{1,3}|iv|v|vi{0,3}|ix|x)\)\s*)",
    re.IGNORECASE,
)


def split_question_parts(text: str, question_number: str) -> list[QuestionPartSegment]:
    cleaned = text.strip()
    if not cleaned:
        return []

    alpha_segments = _segments_from_anchors(cleaned, question_number, _ALPHA_PART_ANCHOR_RE)
    if alpha_segments:
        return alpha_segments
    return _segments_from_anchors(cleaned, question_number, _ROMAN_PART_ANCHOR_RE)


def index_structured_part_texts(part_texts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, part in enumerate(part_texts):
        label = normalize_part_key(str(part.get("part_label", "")))
        if label:
            indexed[label] = part
        indexed[str(index)] = part
    return indexed


def normalize_part_key(label: str) -> str:
    match = re.search(r"\((?P<label>[a-zivx]+)\)", label.lower())
    if match:
        return match.group("label")
    return label.strip().lower()


def _segments_from_anchors(text: str, question_number: str, pattern: re.Pattern[str]) -> list[QuestionPartSegment]:
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    preamble = text[: matches[0].start()].strip()
    segments: list[QuestionPartSegment] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        part_text = text[match.start() : next_start].strip()
        if len(part_text) < 12:
            continue

        label = match.group("label").lower()
        part_label = f"{question_number}({label})" if question_number else f"({label})"
        classification_text = f"{preamble}\n{part_text}".strip() if preamble else part_text
        segments.append(QuestionPartSegment(part_label=part_label, text=part_text, classification_text=classification_text))
    return segments
