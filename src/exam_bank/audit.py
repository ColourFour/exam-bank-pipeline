from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


_CANDIDATE_FIELDS = [
    "text_candidate_source",
    "text_candidate_decision",
    "ocr_selected",
    "native_text_score",
    "ocr_text_score",
    "selected_text_score",
]

_COMPARE_FIELDS = [
    "question_text",
    "text_candidate_source",
    "ocr_selected",
    "text_fidelity_status",
    "text_only_status",
    "visual_curation_status",
    "question_text_trust",
    "topic",
    "mapping_status",
    "validation_status",
]

_CURATION_ORDER = {"ready": 0, "review": 1, "fail": 2}
_TRUST_ORDER = {"high": 0, "medium": 1, "low": 2, "unusable": 3}
_VALIDATION_ORDER = {"pass": 0, "review": 1, "fail": 2}
_MAPPING_ORDER = {"pass": 0, "fail": 1}


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


def audit_ocr_candidates(
    records: Iterable[dict[str, Any]],
    *,
    baseline_records: Iterable[dict[str, Any]] | None = None,
    sample_limit: int = 20,
) -> dict[str, Any]:
    rows = list(records)
    baseline_rows = list(baseline_records) if baseline_records is not None else None
    metadata_presence = _candidate_metadata_presence(rows)
    data_quality_findings = _candidate_data_quality_findings(metadata_presence, len(rows))

    report: dict[str, Any] = {
        "record_count": len(rows),
        "candidate_metadata_presence": metadata_presence,
        "data_quality_findings": data_quality_findings,
        "ocr_selected_count": sum(1 for record in rows if _note_or_top(record, "ocr_selected") is True),
        "text_candidate_source_counts": _counts(rows, "text_candidate_source"),
        "text_candidate_decision_counts": _counts(rows, "text_candidate_decision"),
        "ocr_selected_counts": _counts(rows, "ocr_selected"),
        "ocr_rejected_reason_counts": _reason_counts(rows, "ocr_rejected_reasons"),
        "score_summaries": {
            "native_text_score": _score_summary(_note_or_top(record, "native_text_score") for record in rows),
            "ocr_text_score": _score_summary(_note_or_top(record, "ocr_text_score") for record in rows),
            "selected_text_score": _score_summary(_note_or_top(record, "selected_text_score") for record in rows),
        },
        "text_fidelity_status_counts": _counts(rows, "text_fidelity_status"),
        "text_only_status_counts": _counts(rows, "text_only_status"),
        "visual_curation_status_counts": _counts(rows, "visual_curation_status"),
        "question_text_trust_counts": _counts(rows, "question_text_trust"),
        "suspicious_ocr_selected_records": _suspicious_ocr_selected_records(rows, sample_limit=sample_limit),
        "readiness_inflation_risk_records": _readiness_inflation_risk_records(rows, sample_limit=sample_limit),
        "representative_records": _representative_ocr_records(rows, sample_limit=sample_limit),
        "baseline_comparison": _compare_with_baseline(rows, baseline_rows, sample_limit=sample_limit),
    }
    return report


def write_audit(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    report = audit_question_bank(load_question_records(input_path))
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_ocr_candidate_audit(
    input_path: str | Path,
    *,
    baseline_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline_records = load_question_records(baseline_path) if baseline_path else None
    report = audit_ocr_candidates(load_question_records(input_path), baseline_records=baseline_records)
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


def _candidate_metadata_presence(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    presence: dict[str, dict[str, int]] = {}
    for field in _CANDIDATE_FIELDS:
        present = sum(1 for record in records if _present(_note_or_top(record, field)))
        presence[field] = {"present": present, "missing": len(records) - present}
    return presence


def _candidate_data_quality_findings(presence: dict[str, dict[str, int]], total: int) -> list[str]:
    findings: list[str] = []
    missing_all = [field for field, counts in presence.items() if counts["present"] == 0 and total > 0]
    partial = [field for field, counts in presence.items() if 0 < counts["present"] < total]
    if missing_all:
        findings.append("candidate_metadata_missing_for_all_records:" + ",".join(missing_all))
    if len(missing_all) == len(_CANDIDATE_FIELDS) and total > 0:
        findings.append("stale_or_candidate_unaware_export")
    if partial:
        findings.append("candidate_metadata_partially_populated:" + ",".join(partial))
    if total == 0:
        findings.append("empty_question_bank")
    return findings


def _reason_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(reason for record in records for reason in _list_field(record, key))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _score_summary(values: Iterable[Any]) -> dict[str, Any]:
    scores = sorted(float(value) for value in values if isinstance(value, int | float) and not isinstance(value, bool))
    if not scores:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    return {
        "count": len(scores),
        "min": _round_score(scores[0]),
        "p25": _round_score(_percentile(scores, 0.25)),
        "median": _round_score(_percentile(scores, 0.50)),
        "p75": _round_score(_percentile(scores, 0.75)),
        "max": _round_score(scores[-1]),
        "mean": _round_score(mean(scores)),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _round_score(value: float) -> int | float:
    rounded = round(value, 3)
    return int(rounded) if rounded.is_integer() else rounded


def _suspicious_ocr_selected_records(records: list[dict[str, Any]], *, sample_limit: int) -> list[dict[str, Any]]:
    suspicious: list[dict[str, Any]] = []
    for record in records:
        if _note_or_top(record, "ocr_selected") is not True:
            continue
        reasons = _ocr_selection_risk_reasons(record)
        if reasons:
            suspicious.append(_ocr_record_example(record, extra={"risk_reasons": reasons}))
    return sorted(suspicious, key=lambda item: str(item.get("question_id") or ""))[:sample_limit]


def _readiness_inflation_risk_records(records: list[dict[str, Any]], *, sample_limit: int) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for record in records:
        if _note_or_top(record, "ocr_selected") is not True:
            continue
        reasons: list[str] = []
        text_only_status = _note_or_top(record, "text_only_status")
        visual_curation_status = _note_or_top(record, "visual_curation_status")
        trust = _note_or_top(record, "question_text_trust")
        fidelity = _note_or_top(record, "text_fidelity_status")
        if text_only_status == "ready" and trust != "high":
            reasons.append("text_only_ready_without_high_trust")
        if text_only_status == "ready" and fidelity != "clean":
            reasons.append("text_only_ready_without_clean_fidelity")
        if visual_curation_status == "ready" and _note_or_top(record, "scope_quality_status") != "clean":
            reasons.append("visual_ready_without_clean_scope")
        if reasons:
            risks.append(_ocr_record_example(record, extra={"risk_reasons": reasons}))
    return sorted(risks, key=lambda item: str(item.get("question_id") or ""))[:sample_limit]


def _representative_ocr_records(records: list[dict[str, Any]], *, sample_limit: int) -> list[dict[str, Any]]:
    selected = [record for record in records if _note_or_top(record, "ocr_selected") is True]
    if selected:
        return [_ocr_record_example(record, extra={"sample_reason": "ocr_selected"}) for record in _balanced_sample(selected, sample_limit)]

    rejected_candidates = [
        record
        for record in records
        if _numeric(_note_or_top(record, "ocr_text_score")) is not None and _numeric(_note_or_top(record, "native_text_score")) is not None
    ]
    rejected_candidates.sort(key=lambda record: _score_margin(record), reverse=True)
    return [
        _ocr_record_example(record, extra={"sample_reason": "highest_margin_rejected_or_retained"})
        for record in rejected_candidates[:sample_limit]
    ]


def _balanced_sample(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(
        records,
        key=lambda item: (
            str(_note_or_top(item, "paper_family") or item.get("paper_family") or ""),
            abs((_score_margin(item) or 30) - 30),
            str(item.get("question_id") or ""),
        ),
    ):
        family = str(record.get("paper_family") or "missing")
        by_family.setdefault(family, []).append(record)
    sample: list[dict[str, Any]] = []
    while len(sample) < limit and any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family] and len(sample) < limit:
                sample.append(by_family[family].pop(0))
    return sample


def _ocr_record_example(record: dict[str, Any], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {
        "question_id": record.get("question_id"),
        "paper_family": record.get("paper_family"),
        "question_number": record.get("question_number"),
        "question_text_snippet": str(record.get("question_text") or "")[:240],
        "ocr_text_snippet": str(record.get("ocr_text") or "")[:240],
        "native_text_score": _note_or_top(record, "native_text_score"),
        "ocr_text_score": _note_or_top(record, "ocr_text_score"),
        "selected_text_score": _note_or_top(record, "selected_text_score"),
        "score_margin": _score_margin(record),
        "text_candidate_source": _note_or_top(record, "text_candidate_source"),
        "text_candidate_decision": _note_or_top(record, "text_candidate_decision"),
        "decision_reasons": _list_field(record, "text_candidate_decision_reasons"),
        "ocr_rejected_reasons": _list_field(record, "ocr_rejected_reasons"),
        "scope_quality_status": _note_or_top(record, "scope_quality_status"),
        "text_fidelity_status": _note_or_top(record, "text_fidelity_status"),
        "text_only_status": _note_or_top(record, "text_only_status"),
        "visual_curation_status": _note_or_top(record, "visual_curation_status"),
        "question_text_trust": _note_or_top(record, "question_text_trust"),
        "human_judgment": "not_reviewed",
    }
    if extra:
        item.update(extra)
    return item


def _ocr_selection_risk_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    text = f"{record.get('question_text') or ''} {record.get('ocr_text') or ''}"
    if _note_or_top(record, "scope_quality_status") == "fail":
        reasons.append("scope_quality_failed")
    if not _present(_note_or_top(record, "question_number")):
        reasons.append("missing_question_number")
    if not _present(_note_or_top(record, "question_solution_marks")):
        reasons.append("missing_marks")
    if any(_note_or_top(record, field) == "fail" for field in ["validation_status", "mapping_status", "scope_quality_status"]):
        reasons.append("selected_with_hard_failure")
    if _list_field(record, "ocr_rejected_reasons"):
        reasons.append("selected_with_rejected_reasons")
    if _note_or_top(record, "text_fidelity_status") in {"degraded", "unusable"}:
        reasons.append("selected_text_not_clean")
    if any(flag in _list_field(record, "visual_reason_flags") for flag in ["contains_pdf_control_garbage", "native_text_missing_or_sparse"]):
        reasons.append("visual_text_quality_flag_present")
    if any(fragment.lower() in text.lower() for fragment in ["ucles", "cambridge", "blank page", "question paper", "mark scheme"]):
        reasons.append("possible_page_furniture_text")
    return sorted(set(reasons))


def _compare_with_baseline(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]] | None,
    *,
    sample_limit: int,
) -> dict[str, Any]:
    if baseline_records is None:
        return {"available": False, "reason": "missing_baseline"}

    current_by_id = {str(record.get("question_id")): record for record in records if record.get("question_id")}
    baseline_by_id = {str(record.get("question_id")): record for record in baseline_records if record.get("question_id")}
    shared_ids = sorted(set(current_by_id) & set(baseline_by_id))
    field_change_counts: dict[str, int] = {}
    changed_examples: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []

    for question_id in shared_ids:
        current = current_by_id[question_id]
        baseline = baseline_by_id[question_id]
        changed_fields = [field for field in _COMPARE_FIELDS if _note_or_top(current, field) != _note_or_top(baseline, field)]
        for field in changed_fields:
            field_change_counts[field] = field_change_counts.get(field, 0) + 1
        if changed_fields and len(changed_examples) < sample_limit:
            changed_examples.append(
                {
                    "question_id": question_id,
                    "changed_fields": changed_fields,
                    "baseline_question_text_snippet": str(baseline.get("question_text") or "")[:160],
                    "current_question_text_snippet": str(current.get("question_text") or "")[:160],
                }
            )
        movement = _status_movement(current, baseline)
        if movement["improved"] and len(improved) < sample_limit:
            improved.append({"question_id": question_id, "fields": movement["improved"]})
        if movement["worsened"] and len(worsened) < sample_limit:
            worsened.append({"question_id": question_id, "fields": movement["worsened"]})

    return {
        "available": True,
        "baseline_record_count": len(baseline_records),
        "current_record_count": len(records),
        "shared_record_count": len(shared_ids),
        "missing_from_current_count": len(set(baseline_by_id) - set(current_by_id)),
        "new_in_current_count": len(set(current_by_id) - set(baseline_by_id)),
        "field_change_counts": dict(sorted(field_change_counts.items())),
        "changed_record_examples": changed_examples,
        "improved_records": improved,
        "worsened_records": worsened,
    }


def _status_movement(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, list[str]]:
    improved: list[str] = []
    worsened: list[str] = []
    orders = {
        "text_only_status": _CURATION_ORDER,
        "visual_curation_status": _CURATION_ORDER,
        "question_text_trust": _TRUST_ORDER,
        "validation_status": _VALIDATION_ORDER,
        "mapping_status": _MAPPING_ORDER,
    }
    for field, order in orders.items():
        current_value = _note_or_top(current, field)
        baseline_value = _note_or_top(baseline, field)
        if current_value not in order or baseline_value not in order or current_value == baseline_value:
            continue
        if order[str(current_value)] < order[str(baseline_value)]:
            improved.append(field)
        else:
            worsened.append(field)
    return {"improved": improved, "worsened": worsened}


def _score_margin(record: dict[str, Any]) -> int | float | None:
    ocr_score = _numeric(_note_or_top(record, "ocr_text_score"))
    native_score = _numeric(_note_or_top(record, "native_text_score"))
    if ocr_score is None or native_score is None:
        return None
    return _round_score(ocr_score - native_score)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != []


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
