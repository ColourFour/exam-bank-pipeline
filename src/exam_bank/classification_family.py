from __future__ import annotations

from pathlib import Path
import re

from .document_metadata import parse_filename_metadata
from .trust import Confidence


_PAPER_CODE_RE = re.compile(r"(?:qp|ms|paper|p)[_\-\s]*(?P<code>[1-6][0-9])\b", re.IGNORECASE)
_PAPER_FAMILY_RE = re.compile(r"\bP(?P<family>[1-6])\b", re.IGNORECASE)


def infer_source_paper_family(source_name: str | None) -> tuple[str, str]:
    metadata = parse_filename_metadata(source_name or "")
    if metadata.paper_family != "unknown":
        return metadata.paper_family, Confidence.HIGH
    code, confidence = infer_source_paper_code(source_name)
    if code:
        return f"P{code[0]}", confidence
    if not source_name:
        return "unknown", Confidence.LOW
    name = Path(source_name).name
    direct = _PAPER_FAMILY_RE.search(name)
    if direct:
        return f"P{direct.group('family')}", Confidence.HIGH
    return "unknown", Confidence.LOW


def infer_source_paper_code(source_name: str | None) -> tuple[str, str]:
    if not source_name:
        return "", Confidence.LOW
    metadata = parse_filename_metadata(source_name)
    if metadata.component:
        return metadata.component, Confidence.HIGH
    name = Path(source_name).name
    match = _PAPER_CODE_RE.search(name)
    if match:
        return match.group("code"), Confidence.HIGH
    qp_match = re.search(r"(?:^|[_\-\s])(?:qp|ms)[_\-\s]*(?P<code>[1-6][0-9])(?:\D|$)", name, re.IGNORECASE)
    if qp_match:
        return qp_match.group("code"), Confidence.HIGH
    return "", Confidence.LOW
