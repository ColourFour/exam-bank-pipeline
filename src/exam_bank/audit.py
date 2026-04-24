from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


def load_question_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return list(payload["questions"])
    if isinstance(payload, list):
        return payload
    raise ValueError("Audit input must be a question bank document or a list of question records.")


def audit_question_bank(records: Iterable[dict[str, Any]], *, example_limit: int = 10) -> dict[str, Any]:
    rows = list(records)
    visual_flags = Counter(flag for record in rows for flag in _list_field(record, "visual_reason_flags"))

    clean_but_visual = [
        _example(record)
        for record in rows
        if _note_or_top(record, "text_fidelity_status") == "clean" and bool(_note_or_top(record, "visual_required"))
    ][:example_limit]
    readable_with_corruption = [
        _example(record)
        for record in rows
        if _note_or_top(record, "question_text_role") == "readable_text"
        and any("corruption" in flag or "garbage" in flag or flag == "text_order_unreliable" for flag in _list_field(record, "visual_reason_flags"))
    ][:example_limit]

    return {
        "record_count": len(rows),
        "question_text_role_counts": _counts(rows, "question_text_role"),
        "question_text_trust_counts": _counts(rows, "question_text_trust"),
        "visual_required_counts": _counts(rows, "visual_required"),
        "visual_curation_status_counts": _counts(rows, "visual_curation_status"),
        "text_only_status_counts": _counts(rows, "text_only_status"),
        "visual_reason_flag_counts": dict(sorted(visual_flags.items(), key=lambda item: (-item[1], item[0]))),
        "examples_clean_text_fidelity_but_visual_required": clean_but_visual,
        "examples_readable_text_with_corruption_flags": readable_with_corruption,
    }


def write_audit(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    report = audit_question_bank(load_question_records(input_path))
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = _note_or_top(record, key)
        if isinstance(value, bool):
            label = str(value).lower()
        else:
            label = str(value or "missing")
        counts[label] += 1
    return dict(sorted(counts.items()))


def _note_or_top(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record.get(key)
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(key)
    return None


def _list_field(record: dict[str, Any], key: str) -> list[str]:
    value = _note_or_top(record, key)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _example(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": record.get("question_id"),
        "question_number": record.get("question_number"),
        "question_text_snippet": str(record.get("question_text") or "")[:240],
        "visual_reason_flags": _list_field(record, "visual_reason_flags"),
        "question_image_path": record.get("question_image_path") or _first(record.get("question_image_paths")),
    }


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value
