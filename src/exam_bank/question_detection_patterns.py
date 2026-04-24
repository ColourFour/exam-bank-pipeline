from __future__ import annotations

import re

from .config import AppConfig


QUESTION_START_RE = re.compile(
    r"^\s*(?P<number>[1-9]\d{0,2})(?P<label>(?:\s*\([a-zivxlcdm]+\))*)"
    r"(?=\s|[).,:;\-–—]|$)",
    re.IGNORECASE,
)
MARK_RE = re.compile(r"\[(?P<marks>\d{1,2})\]")
TERMINAL_MARK_RE = re.compile(r"\[(?P<marks>\d{1,2})\]\s*$")
SUBPART_RE = re.compile(
    r"^\s*(?:\d+\s*)?(?P<label>\((?:viii|vii|vi|iv|ix|iii|ii|i|v|x|[a-z])\)(?:\([ivxlcdm]+\))*)",
    re.IGNORECASE,
)


def parse_question_start(text: str, config: AppConfig) -> tuple[str, str] | None:
    """Return top-level question number and visible label if text starts a question."""

    match = QUESTION_START_RE.match(text.strip())
    if not match:
        return None
    number = int(match.group("number"))
    if number > config.detection.max_question_number:
        return None
    label = f"{number}{match.group('label').replace(' ', '')}"
    return str(number), label


def extract_marks_from_text(text: str) -> int | None:
    """Extract the sum of bracketed Cambridge-style marks, e.g. [3] [4]."""

    marks = [int(match.group("marks")) for match in MARK_RE.finditer(text)]
    if not marks:
        return None
    return sum(marks)


def extract_question_total_from_text(text: str) -> int | None:
    """Prefer the terminal total when present, otherwise fall back to summed marks."""

    normalized = _strip_control_chars(text).replace("\u00a0", " ")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    terminal_mark_total = None
    for line in reversed(lines[-3:]):
        match = TERMINAL_MARK_RE.search(line)
        if match:
            terminal_mark_total = int(match.group("marks"))
            break
    mark_values = [int(match.group("marks")) for match in MARK_RE.finditer(normalized)]
    subparts = _question_subparts_from_text_for_validation(normalized)
    return detected_question_total(mark_values, terminal_mark_total, subparts)


def detected_question_total(mark_values: list[int], terminal_mark_total: int | None, subparts: list[str]) -> int | None:
    if not mark_values:
        return terminal_mark_total
    if not subparts or len(mark_values) == 1:
        return terminal_mark_total if terminal_mark_total is not None else sum(mark_values)
    if terminal_mark_total is None:
        return sum(mark_values)
    if len(mark_values) > len(subparts):
        return terminal_mark_total
    if len(mark_values) == len(subparts) and terminal_mark_total == mark_values[-1] and sum(mark_values) > terminal_mark_total:
        return sum(mark_values)
    return terminal_mark_total


def _question_subparts_from_text_for_validation(text: str) -> list[str]:
    labels: list[str] = []
    for line in text.splitlines():
        line = _clean_text_line(line)
        if not line:
            continue
        label = _subpart_label_from_text(line)
        if label and label not in labels:
            labels.append(label)
    return labels


def _subpart_label_from_text(text: str) -> str | None:
    match = SUBPART_RE.match(text.strip())
    if not match:
        return None
    labels = re.findall(r"\(([^)]+)\)", match.group("label"))
    return labels[-1].lower() if labels else None


def _clean_text_line(text: str) -> str:
    return " ".join(_strip_control_chars(text).replace("\u00a0", " ").split())


def _strip_control_chars(text: str) -> str:
    return "".join(char if ord(char) >= 32 or char in "\n\t\r" else " " for char in text)
