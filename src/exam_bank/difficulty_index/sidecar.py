from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.difficulty_index import DIFFICULTY_INDEX_SCHEMA_NAME, DIFFICULTY_INDEX_SCHEMA_VERSION


BAND_LABELS = {
    1: "easiest quintile within paper",
    2: "lower-middle quintile within paper",
    3: "middle quintile within paper",
    4: "upper-middle quintile within paper",
    5: "hardest quintile within paper",
}

SERIOUS_MARK_EVENT_FLAGS = {
    "total_marks_mismatch",
    "question_total_mark_scheme_total_disagree",
    "missing_mark_scheme_image_path",
    "mark_scheme_image_file_missing",
    "mapping_status_fail",
}

UNSAFE_QUESTION_FLAGS = {
    "missing_question_image",
    "missing_mark_scheme_image",
    "question_total_mark_scheme_total_disagree",
    "total_marks_mismatch",
    "mark_scheme_total_mismatch",
    "mapping_status_fail",
    "validation_status_fail",
    "unsafe_advisory_evidence",
}


def build_difficulty_index_sidecar(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    output_path: str | Path | None = "output/json/question_bank.difficulty_index.v1.json",
    reports_dir: str | Path | None = "reports",
    artifact_root: str | Path = "output",
    mark_events_path: str | Path | None = "output/json/question_bank.mark_events.v1.json",
    topic_routing_path: str | Path | None = "output/json/question_bank.topic_routing.v1.json",
    advisory_evidence_path: str | Path | None = "output/advisory_evidence/question_bank.advisory_evidence.v1.json",
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    question_bank = _load_json(question_bank_path)
    questions = _question_records(question_bank)
    mark_events = _records_by_question_id(_load_optional_json(mark_events_path).get("records", []))
    topic_routing = _topic_routing_by_question_id(_load_optional_json(topic_routing_path).get("records", {}))
    advisory = _records_by_question_id(_load_optional_json(advisory_evidence_path).get("records", []))

    records = [
        _build_record(
            question,
            artifact_root=Path(artifact_root),
            mark_event=mark_events.get(str(question.get("question_id") or "")),
            topic_route=topic_routing.get(str(question.get("question_id") or "")),
            advisory_record=advisory.get(str(question.get("question_id") or "")),
            question_index=index,
        )
        for index, question in enumerate(questions)
    ]
    _assign_paper_relative_bands(records)
    sidecar = {
        "schema_name": DIFFICULTY_INDEX_SCHEMA_NAME,
        "schema_version": DIFFICULTY_INDEX_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "source_question_bank_path": _rel_path(question_bank_path),
        "source_question_bank_sha256": _sha256_file(question_bank_path),
        "interpretation": {
            "difficulty_index_0_100": "Internal advisory sorting score only; not a psychometric measurement or candidate success-rate claim.",
            "paper_relative_difficulty_band": "Assigned within each paper after sorting by the advisory index.",
            "student_use": "v1 does not enable student-facing sequencing.",
        },
        "record_count": len(records),
        "records": records,
    }
    if output_path and not dry_run:
        write_atomic_json(sidecar, output_path, sort_keys=True)
    if reports_dir:
        from exam_bank.difficulty_index.reports import write_difficulty_index_reports

        write_difficulty_index_reports(sidecar, reports_dir=reports_dir, dry_run=dry_run)
    return sidecar


def sidecar_summary(sidecar: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in sidecar.get("records", []) if isinstance(record, dict)]
    missing = Counter()
    for record in records:
        features = record.get("features") if isinstance(record.get("features"), dict) else {}
        if (features.get("mark_total") or {}).get("value") is None:
            missing["mark_total"] += 1
        if (features.get("mark_event_structure") or {}).get("available") is not True:
            missing["mark_event_structure"] += 1
        if (features.get("topic_routing") or {}).get("used") is not True:
            missing["safe_topic_routing"] += 1
        if (features.get("examiner_report") or {}).get("used") is not True:
            missing["clean_examiner_report_signal"] += 1
        if (features.get("grade_threshold_context") or {}).get("used") is not True:
            missing["grade_threshold_context"] += 1
    return {
        "record_count": len(records),
        "confidence_counts": dict(Counter(str(record.get("confidence") or "unknown") for record in records)),
        "band_counts": dict(Counter(str(record.get("paper_relative_difficulty_band") or "missing") for record in records)),
        "unsafe_count": sum(1 for record in records if record.get("confidence") == "unsafe"),
        "low_confidence_count": sum(1 for record in records if record.get("confidence") == "low"),
        "teacher_filtering_safe_count": sum(1 for record in records if record.get("safe_for_teacher_filtering") is True),
        "student_sequencing_safe_count": sum(1 for record in records if record.get("safe_for_student_sequencing") is True),
        "missing_important_features": dict(missing),
        "question_total_mark_scheme_total_disagreement_count": sum(
            1
            for record in records
            if "question_total_mark_scheme_total_disagree" in set(record.get("unsafe_reasons") or [])
            or "total_marks_mismatch" in set(record.get("unsafe_reasons") or [])
        ),
        "review_queue_count": sum(1 for record in records if record_needs_review(record)),
    }


def record_needs_review(record: dict[str, Any]) -> bool:
    queue_reasons = set(record.get("unsafe_reasons") or [])
    queue_reasons.update(str(reason) for reason in record.get("review_reasons") or [] if str(reason).startswith("suspicious_"))
    queue_reasons.update(
        str(reason)
        for reason in record.get("review_reasons") or []
        if str(reason)
        in {
            "topic_routing_review_or_low_confidence",
            "mark_event_sidecar_not_advisory_safe",
            "many_tied_scores_within_paper",
            "examiner_report_signal_contradicts_deterministic_score",
        }
    )
    return (
        record.get("confidence") in {"low", "unsafe"}
        or bool(queue_reasons)
        or any(str(warning).startswith("missing_") for warning in record.get("warnings") or [])
    )


def _build_record(
    question: dict[str, Any],
    *,
    artifact_root: Path,
    mark_event: dict[str, Any] | None,
    topic_route: dict[str, Any] | None,
    advisory_record: dict[str, Any] | None,
    question_index: int,
) -> dict[str, Any]:
    notes = question.get("notes") if isinstance(question.get("notes"), dict) else {}
    mark_total = _first_int(
        question.get("question_solution_marks"),
        question.get("marks"),
        question.get("marks_if_available"),
        notes.get("question_solution_marks"),
    )
    mark_scheme_total = _first_int(
        _nested(notes, "mark_scheme_structure_detected", "mark_scheme_total_detected"),
        notes.get("mark_scheme_total_detected"),
        mark_event.get("total_marks_detected") if mark_event else None,
    )
    question_total = _first_int(
        _nested(notes, "question_structure_detected", "question_total_detected"),
        notes.get("question_total_detected"),
        mark_event.get("question_total_detected") if mark_event else None,
    )
    question_number = _first_int(question.get("question_number"))
    paper = str(question.get("paper") or "")
    question_image_path = _first_path(
        question.get("question_image_path"),
        question.get("canonical_question_artifact"),
        question.get("question_image_paths"),
    )
    mark_scheme_image_path = _first_path(question.get("mark_scheme_image_path"), question.get("mark_scheme_image_paths"))
    question_image_exists = _path_exists(question_image_path, artifact_root) if question_image_path else False
    mark_scheme_image_exists = _path_exists(mark_scheme_image_path, artifact_root) if mark_scheme_image_path else False
    warnings: list[str] = []
    review_reasons: list[str] = []
    unsafe_reasons: list[str] = []
    if not question_image_path or not question_image_exists:
        unsafe_reasons.append("missing_question_image")
    if not mark_scheme_image_path or not mark_scheme_image_exists:
        unsafe_reasons.append("missing_mark_scheme_image")
    if mark_scheme_total is not None and question_total is not None and mark_scheme_total != question_total:
        unsafe_reasons.append("question_total_mark_scheme_total_disagree")
    if mark_event and mark_event.get("total_marks_match") is False:
        unsafe_reasons.append("total_marks_mismatch")
    if str(notes.get("mapping_status") or "").lower() == "fail":
        unsafe_reasons.append("mapping_status_fail")
    if str(notes.get("validation_status") or "").lower() == "fail":
        unsafe_reasons.append("validation_status_fail")
    if mark_event:
        for flag in mark_event.get("review_flags") or []:
            if flag in SERIOUS_MARK_EVENT_FLAGS:
                unsafe_reasons.append(str(flag))
            elif flag:
                warnings.append(f"mark_event:{flag}")
    if not mark_event:
        warnings.append("mark_event_sidecar_missing")
    elif mark_event.get("safe_for_advisory_use") is not True:
        review_reasons.append("mark_event_sidecar_not_advisory_safe")

    score, features, evidence_used, feature_warnings, feature_reviews = _score_features(
        question=question,
        notes=notes,
        mark_event=mark_event,
        topic_route=topic_route,
        advisory_record=advisory_record,
        mark_total=mark_total,
        mark_scheme_total=mark_scheme_total,
        question_number=question_number,
    )
    warnings.extend(feature_warnings)
    review_reasons.extend(feature_reviews)
    confidence = _confidence(
        unsafe_reasons=unsafe_reasons,
        warnings=warnings,
        review_reasons=review_reasons,
        mark_total=mark_total,
        mark_event=mark_event,
        question_image_exists=question_image_exists,
        mark_scheme_image_exists=mark_scheme_image_exists,
        topic_route=topic_route,
        advisory_record=advisory_record,
    )
    if confidence == "unsafe":
        unsafe_reasons.append("unsafe_for_difficulty_use")
    return {
        "question_id": str(question.get("question_id") or f"record_{question_index:04d}"),
        "paper": paper,
        "paper_family": _lower_or_empty(question.get("paper_family")),
        "family": _lower_or_empty(question.get("paper_family")),
        "component": str(question.get("component") or notes.get("source_paper_code") or ""),
        "session": str(question.get("session") or _session_from_paper(paper)),
        "year": str(question.get("year") or _year_from_paper(paper)),
        "question_number": question.get("question_number"),
        "difficulty_index_0_100": score,
        "paper_relative_percentile": None,
        "paper_relative_difficulty_band": None,
        "paper_relative_band_label": "",
        "confidence": confidence,
        "safe_for_teacher_filtering": confidence != "unsafe",
        "safe_for_student_sequencing": False,
        "evidence_used": sorted(set(evidence_used)),
        "features": features,
        "warnings": _dedupe(warnings),
        "review_reasons": _dedupe(review_reasons),
        "unsafe_reasons": _dedupe(unsafe_reasons),
        "source_paths": {
            "question_image_path": question_image_path,
            "mark_scheme_image_path": mark_scheme_image_path,
            "question_image_exists": question_image_exists,
            "mark_scheme_image_exists": mark_scheme_image_exists,
        },
    }


def _score_features(
    *,
    question: dict[str, Any],
    notes: dict[str, Any],
    mark_event: dict[str, Any] | None,
    topic_route: dict[str, Any] | None,
    advisory_record: dict[str, Any] | None,
    mark_total: int | None,
    mark_scheme_total: int | None,
    question_number: int | None,
) -> tuple[int, dict[str, Any], list[str], list[str], list[str]]:
    evidence_used: list[str] = []
    warnings: list[str] = []
    review_reasons: list[str] = []
    features: dict[str, Any] = {}
    score = 12.0

    effective_marks = mark_scheme_total if mark_scheme_total is not None else mark_total
    mark_component = min(max((effective_marks or 0) / 12.0, 0), 1) * 28
    if effective_marks is None:
        warnings.append("missing_mark_total")
    else:
        evidence_used.append("mark_total")
    score += mark_component
    features["mark_total"] = {"value": effective_marks, "contribution": round(mark_component, 3)}

    position_component = min(max(((question_number or 1) - 1) / 10.0, 0), 1) * 24
    if question_number is None:
        warnings.append("missing_question_number")
    else:
        evidence_used.append("question_position")
    score += position_component
    features["question_position"] = {"question_number": question_number, "contribution": round(position_component, 3)}

    subparts = question.get("subparts") if isinstance(question.get("subparts"), list) else []
    notes_structure = notes.get("question_structure_detected") if isinstance(notes.get("question_structure_detected"), dict) else {}
    structure_subparts = notes_structure.get("subparts") if isinstance(notes_structure.get("subparts"), list) else []
    part_count = max(len(subparts), len(structure_subparts))
    scaffolding_adjustment = -min(max(part_count - 1, 0), 4) * 2.0
    linked = "hence" in f"{question.get('question_text') or ''} {question.get('ocr_text') or ''}".lower()
    if linked:
        scaffolding_adjustment += 4.0
    score += scaffolding_adjustment
    features["scaffolding"] = {
        "subpart_count": part_count,
        "linked_keyword_present": linked,
        "contribution": round(scaffolding_adjustment, 3),
    }
    if part_count:
        evidence_used.append("subpart_structure")

    if mark_event:
        events = mark_event.get("mark_events") if isinstance(mark_event.get("mark_events"), list) else []
        method_like = sum(1 for event in events if str(event.get("mark_type") or "") in {"method", "dependent_method", "follow_through"})
        dependent = sum(1 for event in events if event.get("is_dependent") is True)
        unknown = sum(1 for event in events if str(event.get("mark_type") or "") == "unknown")
        event_component = min(len(events), 12) * 1.0 + min(method_like, 8) * 1.2 + min(dependent, 4) * 1.5 - min(unknown, 4) * 1.5
        event_component = max(min(event_component, 18), -4)
        score += event_component
        evidence_used.append("mark_event_structure")
        features["mark_event_structure"] = {
            "available": True,
            "event_count": len(events),
            "method_like_count": method_like,
            "dependent_count": dependent,
            "unknown_count": unknown,
            "contribution": round(event_component, 3),
            "safe_for_advisory_use": mark_event.get("safe_for_advisory_use"),
        }
    else:
        features["mark_event_structure"] = {"available": False, "contribution": 0}

    topic_component = 0.0
    if topic_route and topic_route.get("review_required") is not True and str(topic_route.get("confidence") or "") in {"high", "medium"}:
        topic_id = str(topic_route.get("primary_topic_id") or "")
        topic_component = _topic_complexity_prior(topic_id)
        evidence_used.append("safe_topic_routing")
    elif topic_route:
        review_reasons.append("topic_routing_review_or_low_confidence")
    else:
        warnings.append("topic_routing_missing")
    score += topic_component
    features["topic_routing"] = {
        "used": topic_component != 0,
        "primary_topic_id": topic_route.get("primary_topic_id") if topic_route else "",
        "confidence": topic_route.get("confidence") if topic_route else "",
        "review_required": topic_route.get("review_required") if topic_route else None,
        "contribution": round(topic_component, 3),
    }

    examiner_component, examiner_used, examiner_warning = _examiner_component(advisory_record)
    if examiner_used:
        evidence_used.append("clean_examiner_report_signal")
    if examiner_warning:
        review_reasons.append(examiner_warning)
    score += examiner_component
    features["examiner_report"] = {
        "used": examiner_used,
        "contribution": round(examiner_component, 3),
        "signal": _nested(_nested(advisory_record or {}, "examiner_report_difficulty"), "item_signal"),
    }

    threshold_component, threshold_used, threshold_warning = _threshold_component(advisory_record)
    if threshold_used:
        evidence_used.append("weak_grade_threshold_paper_context")
    if threshold_warning:
        warnings.append(threshold_warning)
    score += threshold_component
    features["grade_threshold_context"] = {
        "used": threshold_used,
        "contribution": round(threshold_component, 3),
        "context_label": _nested(
            _nested(_nested(advisory_record or {}, "advisory_evidence"), "grade_threshold_context"),
            "component_context_label",
        ),
    }

    legacy_score = _first_int(question.get("difficulty_score"), notes.get("difficulty_score"))
    if legacy_score is not None and not notes.get("difficulty_uncertain"):
        legacy_component = ((legacy_score - 50) / 50) * 5
        score += legacy_component
        evidence_used.append("legacy_internal_difficulty_score_weak")
        features["legacy_internal_difficulty_score"] = {"value": legacy_score, "contribution": round(legacy_component, 3)}
    else:
        features["legacy_internal_difficulty_score"] = {"value": legacy_score, "contribution": 0}

    return int(round(max(0, min(100, score)))), features, evidence_used, warnings, review_reasons


def _assign_paper_relative_bands(records: list[dict[str, Any]]) -> None:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_paper[str(record.get("paper") or "")].append(record)
    for paper_records in by_paper.values():
        sorted_records = sorted(
            paper_records,
            key=lambda record: (
                float(record.get("difficulty_index_0_100") or 0),
                -(_first_int((record.get("features") or {}).get("mark_total", {}).get("value")) or 0),
                -(_first_int(record.get("question_number")) or 0),
                str(record.get("question_id") or ""),
            ),
        )
        n = len(sorted_records)
        sizes = _middle_weighted_band_sizes(n)
        bands: list[int] = []
        for band, size in enumerate(sizes, start=1):
            bands.extend([band] * size)
        for index, record in enumerate(sorted_records):
            record["paper_relative_difficulty_band"] = bands[index]
            record["paper_relative_band_label"] = BAND_LABELS[bands[index]]
            record["paper_relative_percentile"] = 50.0 if n == 1 else round((index / (n - 1)) * 100, 2)
            _add_suspicious_placement_reasons(record, n)
        if _many_tied_scores(sorted_records):
            for record in sorted_records:
                record["review_reasons"] = _dedupe(list(record.get("review_reasons") or []) + ["many_tied_scores_within_paper"])


def _middle_weighted_band_sizes(n: int) -> list[int]:
    if n <= 0:
        return [0, 0, 0, 0, 0]
    base, remainder = divmod(n, 5)
    sizes = [base] * 5
    for band_index in [2, 1, 3, 0, 4][:remainder]:
        sizes[band_index] += 1
    return sizes


def _add_suspicious_placement_reasons(record: dict[str, Any], paper_size: int) -> None:
    reasons = list(record.get("review_reasons") or [])
    band = record.get("paper_relative_difficulty_band")
    mark_total = _first_int((record.get("features") or {}).get("mark_total", {}).get("value"))
    qnum = _first_int(record.get("question_number"))
    if mark_total is not None and mark_total <= 3 and band == 5:
        reasons.append("suspicious_low_mark_total_in_band_5")
    if mark_total is not None and mark_total >= 10 and band == 1:
        reasons.append("suspicious_high_mark_total_in_band_1")
    if qnum is not None and qnum <= 2 and band == 5:
        reasons.append("suspicious_early_question_in_band_5")
    if qnum is not None and paper_size >= 5 and qnum >= max(1, paper_size - 1) and band == 1:
        reasons.append("suspicious_late_question_in_band_1")
    record["review_reasons"] = _dedupe(reasons)


def _many_tied_scores(records: list[dict[str, Any]]) -> bool:
    if len(records) < 5:
        return False
    counts = Counter(record.get("difficulty_index_0_100") for record in records)
    return any(count >= max(4, len(records) // 3) for count in counts.values())


def _confidence(
    *,
    unsafe_reasons: list[str],
    warnings: list[str],
    review_reasons: list[str],
    mark_total: int | None,
    mark_event: dict[str, Any] | None,
    question_image_exists: bool,
    mark_scheme_image_exists: bool,
    topic_route: dict[str, Any] | None,
    advisory_record: dict[str, Any] | None,
) -> str:
    if any(reason in UNSAFE_QUESTION_FLAGS or reason.startswith("missing_") for reason in unsafe_reasons):
        return "unsafe"
    if not question_image_exists or not mark_scheme_image_exists:
        return "unsafe"
    if mark_event and mark_event.get("safe_for_advisory_use") is False:
        return "unsafe"
    evidence_count = 0
    if mark_total is not None:
        evidence_count += 1
    if mark_event and mark_event.get("safe_for_advisory_use") is True:
        evidence_count += 1
    if topic_route and topic_route.get("review_required") is not True and topic_route.get("confidence") in {"high", "medium"}:
        evidence_count += 1
    if _examiner_component(advisory_record)[1]:
        evidence_count += 1
    if warnings or review_reasons:
        return "low" if evidence_count <= 2 else "medium"
    if evidence_count >= 4:
        return "high"
    if evidence_count >= 2:
        return "medium"
    return "low"


def _topic_complexity_prior(topic_id: str) -> float:
    lowered = topic_id.lower()
    hard_terms = ("vectors", "complex", "differential", "integration", "mechanics", "hypothesis", "normal", "trigonometry")
    easy_terms = ("series", "quadratics", "algebra", "coordinate_geometry", "straight_line")
    if any(term in lowered for term in hard_terms):
        return 5.0
    if any(term in lowered for term in easy_terms):
        return -2.0
    return 0.0


def _examiner_component(advisory_record: dict[str, Any] | None) -> tuple[float, bool, str]:
    difficulty = advisory_record.get("examiner_report_difficulty") if isinstance(advisory_record, dict) else {}
    if not isinstance(difficulty, dict) or not difficulty:
        return 0.0, False, ""
    if difficulty.get("review_required") is True or difficulty.get("warnings"):
        return 0.0, False, ""
    if difficulty.get("confidence") not in {"high", "medium"}:
        return 0.0, False, ""
    signal = str(difficulty.get("item_signal") or "").lower()
    if signal in {"hard", "challenging", "difficult"}:
        return 6.0, True, ""
    if signal in {"easy", "well_answered", "routine"}:
        return -5.0, True, ""
    return 0.0, False, ""


def _threshold_component(advisory_record: dict[str, Any] | None) -> tuple[float, bool, str]:
    context = _nested(_nested(advisory_record or {}, "advisory_evidence"), "grade_threshold_context")
    if not isinstance(context, dict) or context.get("available") is not True:
        return 0.0, False, ""
    if context.get("warnings"):
        return 0.0, False, "grade_threshold_context_warned"
    label = str(context.get("component_context_label") or "")
    if label == "paper_context_hard":
        return 2.0, True, ""
    if label == "paper_context_easy":
        return -2.0, True, ""
    if label == "paper_context_typical":
        return 0.0, True, ""
    return 0.0, False, ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return _load_json(path)


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _records_by_question_id(records: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        return {}
    return {str(record.get("question_id")): record for record in records if isinstance(record, dict) and record.get("question_id")}


def _topic_routing_by_question_id(records: Any) -> dict[str, dict[str, Any]]:
    if isinstance(records, dict):
        return {str(key): value for key, value in records.items() if isinstance(value, dict)}
    return _records_by_question_id(records)


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value in ("", None):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            match = re.search(r"\d+", str(value))
            if match:
                return int(match.group(0))
    return None


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
    path = Path(path_value)
    if path.is_absolute():
        return path.is_file()
    return (artifact_root / path).is_file() or path.is_file()


def _nested(data: dict[str, Any], outer: str, inner: str | None = None) -> Any:
    if not isinstance(data, dict):
        return None
    value = data.get(outer)
    if inner is None:
        return value
    if isinstance(value, dict):
        return value.get(inner)
    return None


def _session_from_paper(paper: str) -> str:
    lowered = paper.lower()
    if "spring" in lowered:
        return "March"
    if "summer" in lowered:
        return "MayJune"
    if "autumn" in lowered:
        return "November"
    return ""


def _year_from_paper(paper: str) -> str:
    match = re.search(r"(\d{2})$", paper)
    if not match:
        return ""
    year = int(match.group(1))
    return str(2000 + year)


def _lower_or_empty(value: Any) -> str:
    return str(value or "").lower()


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
