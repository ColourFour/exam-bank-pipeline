from __future__ import annotations

from pathlib import Path
import re
from typing import Any


KNOWN_MISSING_MARK_SCHEME_COMPANIONS = {
    "9709_2025_November_33": {
        "paper": "33winter25",
        "reason": "The source mark scheme PDF for 9709 Mathematics November 2025 Paper 33 is missing.",
    },
}

_SOURCE_COMPANION_RE = re.compile(
    r"(?P<syllabus>\d{4}).*?(?P<session>March|June|November)\s+"
    r"(?P<year>20\d{2}).*?(?P<component>\d{2})(?!.*\d)",
    re.IGNORECASE,
)
_PAPER_ID_RE = re.compile(r"^(?P<component>\d{2})(?P<season>spring|summer|autumn|winter)(?P<year>\d{2})$", re.IGNORECASE)
_PAPER_SEASON_TO_SESSION = {
    "spring": "March",
    "summer": "June",
    "autumn": "November",
    "winter": "November",
}


def source_companion_key(record: dict[str, Any]) -> str:
    for field in ["mark_scheme_source_pdf", "source_pdf"]:
        value = _clean_text(_note_or_top(record, field))
        if not value:
            continue
        match = _SOURCE_COMPANION_RE.search(Path(value).name)
        if match:
            return "_".join(
                [
                    match.group("syllabus"),
                    match.group("year"),
                    match.group("session").title(),
                    match.group("component"),
                ]
            )
    paper = _clean_text(record.get("paper"))
    for key, details in KNOWN_MISSING_MARK_SCHEME_COMPANIONS.items():
        if paper and paper == details.get("paper"):
            return key
    return source_companion_key_from_paper(paper)


def source_companion_key_from_paper(paper: str) -> str:
    match = _PAPER_ID_RE.fullmatch(paper)
    if not match:
        return ""
    session = _PAPER_SEASON_TO_SESSION.get(match.group("season").lower())
    if not session:
        return ""
    return f"9709_20{match.group('year')}_{session}_{match.group('component')}"


def is_known_missing_mark_scheme_companion(record: dict[str, Any]) -> bool:
    return source_companion_key(record) in KNOWN_MISSING_MARK_SCHEME_COMPANIONS


def _note_or_top(record: dict[str, Any], field: str) -> Any:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    if field in notes:
        return notes.get(field)
    return record.get(field)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()
