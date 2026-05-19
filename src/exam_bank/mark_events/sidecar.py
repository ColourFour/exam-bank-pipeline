from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.mark_events import MARK_EVENTS_SCHEMA_NAME, MARK_EVENTS_SCHEMA_VERSION
from exam_bank.mark_events.parsing import parse_mark_scheme_text


SERIOUS_REVIEW_FLAGS = {
    "missing_question_id",
    "missing_mark_scheme_text",
    "missing_mark_scheme_image_path",
    "mark_scheme_image_file_missing",
    "mapping_status_fail",
    "no_detected_mark_events",
    "total_marks_mismatch",
    "question_total_mark_scheme_total_disagree",
    "unknown_mark_code",
    "unknown_dependent_mark_code",
    "dependent_mark_without_deterministic_prior_event",
}


def build_mark_events_sidecar(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    artifact_root: str | Path = "output",
    output_path: str | Path | None = None,
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    payload = _load_json(question_bank_path)
    records = _question_records(payload)
    generated_at = generated_at or _utc_now_iso()
    sidecar_records = [
        build_mark_event_record(record, artifact_root=Path(artifact_root), question_index=index)
        for index, record in enumerate(records)
    ]
    sidecar = {
        "schema_name": MARK_EVENTS_SCHEMA_NAME,
        "schema_version": MARK_EVENTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_question_bank_path": _rel_path(question_bank_path),
        "source_question_bank_sha256": _sha256_file(question_bank_path),
        "record_count": len(sidecar_records),
        "records": sidecar_records,
    }
    if output_path and not dry_run:
        write_atomic_json(sidecar, output_path)
    return sidecar


def build_mark_event_record(record: dict[str, Any], *, artifact_root: Path, question_index: int = 0) -> dict[str, Any]:
    question_id = str(record.get("question_id") or "")
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    text = str(record.get("mark_scheme_text") or "")
    source_path = _first_path(record.get("mark_scheme_image_path"), record.get("mark_scheme_image_paths"))
    source_exists = _path_exists(source_path, artifact_root) if source_path else False
    parse_result = parse_mark_scheme_text(text, question_id=question_id or f"record_{question_index:04d}")
    events = parse_result.events
    expected_total = _first_int(record.get("question_solution_marks"), notes.get("question_solution_marks"))
    detected_total = _first_int(notes.get("mark_scheme_total_detected"), _nested(notes, "mark_scheme_structure_detected", "mark_scheme_total_detected"))
    question_total = _first_int(notes.get("question_total_detected"), _nested(notes, "question_structure_detected", "question_total_detected"))
    if detected_total is None:
        detected_total = sum(event["mark_value"] for event in events) if events else None
    total_match = _total_match(detected_total, expected_total)
    review_flags = _record_review_flags(
        record=record,
        notes=notes,
        question_id=question_id,
        text=text,
        source_path=source_path,
        source_exists=source_exists,
        events=events,
        total_match=total_match,
        detected_total=detected_total,
        expected_total=expected_total,
        question_total=question_total,
        parser_flags=parse_result.review_flags,
    )
    extraction_status = _extraction_status(text=text, events=events, review_flags=review_flags)
    safe_for_advisory = _safe_for_advisory_use(
        extraction_status=extraction_status,
        source_exists=source_exists,
        events=events,
        total_match=total_match,
        review_flags=review_flags,
    )
    return {
        "question_id": question_id,
        "paper_id": record.get("paper"),
        "paper": record.get("paper"),
        "paper_family": record.get("paper_family"),
        "session": _session_from_paper(str(record.get("paper") or "")),
        "variant": _variant_from_notes(notes),
        "question_number": record.get("question_number"),
        "part_path": [],
        "source_mark_scheme_image_path": source_path,
        "source_mark_scheme_image_exists": source_exists,
        "source_text_kind": "native" if text.strip() else "unknown",
        "extraction_status": extraction_status,
        "safe_for_advisory_use": safe_for_advisory,
        "safe_for_marking_use": False,
        "total_marks_detected": detected_total,
        "total_marks_expected": expected_total,
        "question_total_detected": question_total,
        "total_marks_match": total_match,
        "review_flags": review_flags,
        "part_summaries": _part_summaries(events),
        "mark_events": events,
        "unparsed_evidence": parse_result.unparsed_evidence[:25],
        "top_unparsed_patterns": parse_result.top_unparsed_patterns,
    }


def sidecar_summary(sidecar: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in sidecar.get("records", []) if isinstance(record, dict)]
    events = [event for record in records for event in record.get("mark_events", []) if isinstance(event, dict)]
    return {
        "record_count": len(records),
        "event_count": len(events),
        "safe_for_advisory_count": sum(1 for record in records if record.get("safe_for_advisory_use") is True),
        "safe_for_marking_count": sum(1 for record in records if record.get("safe_for_marking_use") is True),
        "extraction_status_counts": dict(Counter(str(record.get("extraction_status") or "unknown") for record in records)),
        "mark_code_counts": _mark_code_counts(events),
        "follow_through_count": sum(1 for event in events if event.get("is_follow_through") is True),
        "dependent_count": sum(1 for event in events if event.get("is_dependent") is True),
        "total_mismatch_count": sum(
            1
            for record in records
            if record.get("total_marks_match") is False
            or "question_total_mark_scheme_total_disagree" in set(record.get("review_flags") or [])
        ),
        "question_total_disagreement_count": sum(
            1 for record in records if "question_total_mark_scheme_total_disagree" in set(record.get("review_flags") or [])
        ),
        "missing_mark_scheme_image_count": sum(1 for record in records if not record.get("source_mark_scheme_image_exists")),
        "no_detected_mark_events_count": sum(1 for record in records if not record.get("mark_events")),
        "ambiguous_part_mapping_count": sum(
            1 for record in records if "ambiguous_part_mapping" in set(record.get("review_flags") or [])
        ),
        "unsafe_for_advisory_reasons": dict(
            Counter(flag for record in records if not record.get("safe_for_advisory_use") for flag in record.get("review_flags", []))
        ),
        "top_unparsed_patterns": _top_patterns(records),
    }


def _record_review_flags(
    *,
    record: dict[str, Any],
    notes: dict[str, Any],
    question_id: str,
    text: str,
    source_path: str,
    source_exists: bool,
    events: list[dict[str, Any]],
    total_match: bool | None,
    detected_total: int | None,
    expected_total: int | None,
    question_total: int | None,
    parser_flags: list[str],
) -> list[str]:
    flags: list[str] = list(parser_flags)
    if not question_id:
        flags.append("missing_question_id")
    if not text.strip():
        flags.append("missing_mark_scheme_text")
    if not source_path:
        flags.append("missing_mark_scheme_image_path")
    elif not source_exists:
        flags.append("mark_scheme_image_file_missing")
    if str(notes.get("mapping_status") or "").lower() == "fail":
        flags.append("mapping_status_fail")
    if not events:
        flags.append("no_detected_mark_events")
    if total_match is False:
        flags.append("total_marks_mismatch")
    if (
        detected_total is not None
        and question_total is not None
        and detected_total != question_total
        and expected_total == detected_total
    ):
        flags.append("question_total_mark_scheme_total_disagree")
    for event in events:
        flags.extend(str(flag) for flag in event.get("review_flags", []) if flag)
    return _dedupe(flags)


def _extraction_status(*, text: str, events: list[dict[str, Any]], review_flags: list[str]) -> str:
    if not text.strip():
        return "failed"
    if not events:
        return "review"
    if any(flag in SERIOUS_REVIEW_FLAGS for flag in review_flags):
        return "review"
    if review_flags:
        return "partial"
    return "parsed"


def _safe_for_advisory_use(
    *,
    extraction_status: str,
    source_exists: bool,
    events: list[dict[str, Any]],
    total_match: bool | None,
    review_flags: list[str],
) -> bool:
    if extraction_status not in {"parsed", "partial"}:
        return False
    if not source_exists or not events or total_match is not True:
        return False
    return not any(flag in SERIOUS_REVIEW_FLAGS for flag in review_flags)


def _part_summaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_part: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_part[tuple(event.get("part_path") or [])].append(event)
    summaries: list[dict[str, Any]] = []
    for part, part_events in sorted(by_part.items()):
        summaries.append(
            {
                "part_path": list(part),
                "mark_event_count": len(part_events),
                "mark_value_sum": sum(int(event.get("mark_value") or 0) for event in part_events),
                "mark_code_counts": _mark_code_counts(part_events),
                "follow_through_count": sum(1 for event in part_events if event.get("is_follow_through") is True),
                "dependent_count": sum(1 for event in part_events if event.get("is_dependent") is True),
            }
        )
    return summaries


def _mark_code_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in events:
        code = str(event.get("mark_code_raw") or "").upper()
        if code.startswith("DM"):
            key = "DM"
        elif code.startswith("M"):
            key = "M"
        elif code.startswith("A") or code.startswith("DA"):
            key = "A"
        elif code.startswith("B") or code.startswith("DB"):
            key = "B"
        elif code.startswith("E") or code.startswith("DE"):
            key = "E"
        else:
            key = "unknown"
        if event.get("mark_type") == "unknown":
            key = "unknown"
        counts[key] += 1
    return {key: counts.get(key, 0) for key in ["M", "A", "B", "E", "DM", "unknown"] if counts.get(key, 0)}


def _top_patterns(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for pattern, count in (record.get("top_unparsed_patterns") or {}).items():
            counter[str(pattern)] += int(count)
    return {key: counter[key] for key in [key for key, _ in counter.most_common(20)]}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _first_path(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def _path_exists(path_value: str, artifact_root: Path) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    if path.is_absolute():
        return path.is_file()
    return (artifact_root / path).is_file() or path.is_file()


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value in ("", None):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _nested(data: dict[str, Any], outer: str, inner: str) -> Any:
    value = data.get(outer)
    if isinstance(value, dict):
        return value.get(inner)
    return None


def _total_match(detected: int | None, expected: int | None) -> bool | None:
    if detected is None or expected is None:
        return None
    return detected == expected


def _session_from_paper(paper: str) -> str:
    lowered = paper.lower()
    if "spring" in lowered:
        return "March"
    if "summer" in lowered:
        return "MayJune"
    if "autumn" in lowered:
        return "November"
    return ""


def _variant_from_notes(notes: dict[str, Any]) -> str:
    return str(notes.get("source_paper_code") or "").strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel_path(path: str | Path, root: str | Path = ".") -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
