from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel_path(path: str | Path, root: str | Path = ".") -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def stable_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "document"


def stable_source_stem(source_path: str | Path) -> str:
    path = Path(source_path)
    digest = hashlib.sha1(rel_path(path).encode("utf-8")).hexdigest()[:8]
    return f"{stable_slug(path.stem)}-{digest}"


def session_slug(session: str, year: str) -> str:
    session_value = {
        "March": "march",
        "MayJune": "june",
        "November": "november",
        "OctNov": "november",
        "October": "october",
    }.get(session, stable_slug(session))
    return f"{session_value}_{year}" if year else session_value


def load_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def records_from_question_bank(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def first_ints_after(lines: list[str], start_index: int, count: int) -> tuple[list[int], int]:
    values: list[int] = []
    index = start_index
    while index < len(lines) and len(values) < count:
        if re.fullmatch(r"-?\d+", lines[index]):
            values.append(int(lines[index]))
        index += 1
    return values, index

