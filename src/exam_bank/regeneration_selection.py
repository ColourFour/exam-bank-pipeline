from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def normalize_lookup_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = Path(text).name if "/" in text else text
    return text.removesuffix(".png").strip()


def normalize_requested_ids(values: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        for part in str(value or "").replace("\n", ",").split(","):
            cleaned = normalize_lookup_key(part)
            if cleaned:
                normalized.add(cleaned)
    return normalized
