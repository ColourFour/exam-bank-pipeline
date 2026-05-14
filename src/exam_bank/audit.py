from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PureWindowsPath
import re
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

CURRENT_OUTPUT_INTEGRITY_SCHEMA_NAME = "exam_bank.current_output_integrity_audit"
CURRENT_OUTPUT_INTEGRITY_SCHEMA_VERSION = 1

KNOWN_MISSING_MARK_SCHEME_COMPANIONS = {
    "9709_2025_November_33": {
        "paper": "33autumn25",
        "reason": "The source mark scheme PDF for 9709 Mathematics November 2025 Paper 33 is missing.",
    },
}

_SOURCE_COMPANION_RE = re.compile(
    r"(?P<syllabus>\d{4}).*?(?P<session>March|June|November)\s+"
    r"(?P<year>20\d{2}).*?(?P<component>\d{2})(?!.*\d)",
    re.IGNORECASE,
)


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


def audit_current_output_integrity(
    input_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    example_limit: int = 10,
) -> dict[str, Any]:
    input_path = Path(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = _records_from_payload(payload)
    root = Path(artifact_root) if artifact_root is not None else _infer_artifact_root(input_path, payload)
    return audit_current_output_integrity_records(
        records,
        payload=payload,
        input_path=input_path,
        artifact_root=root,
        example_limit=example_limit,
    )


def audit_current_output_integrity_records(
    records: Iterable[dict[str, Any]],
    *,
    payload: dict[str, Any] | None = None,
    input_path: str | Path | None = None,
    artifact_root: str | Path,
    example_limit: int = 10,
) -> dict[str, Any]:
    rows = list(records)
    root = Path(artifact_root)

    missing_question_ids: list[dict[str, Any]] = []
    missing_paper_question_pairs: list[dict[str, Any]] = []
    question_id_records: list[tuple[int, str, dict[str, Any]]] = []
    paper_question_records: list[tuple[int, tuple[str, str], dict[str, Any]]] = []

    missing_question_image_paths: list[dict[str, Any]] = []
    bad_question_path_entries: dict[str, list[dict[str, Any]]] = {
        "absolute_question_image_path": [],
        "missing_question_image_file": [],
    }
    missing_mark_scheme_paths: list[dict[str, Any]] = []
    allowed_missing_mark_scheme_paths: list[dict[str, Any]] = []
    bad_mark_scheme_path_entries: dict[str, list[dict[str, Any]]] = {
        "absolute_mark_scheme_image_path": [],
        "missing_mark_scheme_image_file": [],
    }
    observed_allowed_companions: dict[str, list[str]] = {key: [] for key in KNOWN_MISSING_MARK_SCHEME_COMPANIONS}

    question_path_entry_count = 0
    mark_scheme_path_entry_count = 0
    nonblank_mark_scheme_record_count = 0

    for index, record in enumerate(rows):
        question_id = _clean_text(record.get("question_id"))
        paper = _clean_text(record.get("paper"))
        question_number = _clean_text(record.get("question_number"))
        companion = _source_companion_key(record)

        if question_id:
            question_id_records.append((index, question_id, record))
        else:
            missing_question_ids.append(_record_ref(index, record, companion=companion))

        if paper and question_number:
            paper_question_records.append((index, (paper, question_number), record))
        else:
            missing_paper_question_pairs.append(_record_ref(index, record, companion=companion))

        question_entries = _question_image_path_entries(record)
        primary_question_path = _clean_text(record.get("question_image_path"))
        if not primary_question_path:
            missing_question_image_paths.append(_record_ref(index, record, companion=companion))
        question_path_entry_count += len(question_entries)
        for entry in question_entries:
            _check_path_entry(
                entry,
                index=index,
                record=record,
                companion=companion,
                artifact_root=root,
                absolute_code="absolute_question_image_path",
                missing_file_code="missing_question_image_file",
                failures=bad_question_path_entries,
            )

        mark_scheme_entries = _mark_scheme_image_path_entries(record)
        if mark_scheme_entries:
            nonblank_mark_scheme_record_count += 1
        else:
            ref = _record_ref(index, record, companion=companion)
            if companion in KNOWN_MISSING_MARK_SCHEME_COMPANIONS:
                allowed_missing_mark_scheme_paths.append(ref)
                observed_allowed_companions.setdefault(companion, []).append(question_id or f"index:{index}")
            else:
                missing_mark_scheme_paths.append(ref)
        mark_scheme_path_entry_count += len(mark_scheme_entries)
        for entry in mark_scheme_entries:
            _check_path_entry(
                entry,
                index=index,
                record=record,
                companion=companion,
                artifact_root=root,
                absolute_code="absolute_mark_scheme_image_path",
                missing_file_code="missing_mark_scheme_image_file",
                failures=bad_mark_scheme_path_entries,
            )

    duplicate_question_id_values = {
        value for value, count in Counter(value for _, value, _ in question_id_records).items() if count > 1
    }
    duplicate_paper_question_values = {
        value for value, count in Counter(value for _, value, _ in paper_question_records).items() if count > 1
    }
    duplicate_question_id_records = [
        _record_ref(index, record, companion=_source_companion_key(record), duplicate_key=question_id)
        for index, question_id, record in question_id_records
        if question_id in duplicate_question_id_values
    ]
    duplicate_paper_question_records = [
        _record_ref(index, record, companion=_source_companion_key(record), duplicate_key=f"{pair[0]}:{pair[1]}")
        for index, pair, record in paper_question_records
        if pair in duplicate_paper_question_values
    ]

    record_count_mismatch = []
    declared_count = payload.get("record_count") if isinstance(payload, dict) else None
    if isinstance(declared_count, int) and declared_count != len(rows):
        record_count_mismatch.append(
            {
                "declared_record_count": declared_count,
                "actual_record_count": len(rows),
            }
        )

    failures = [
        _failure(
            "record_count_mismatch",
            "Top-level record_count does not match the number of question records.",
            record_count_mismatch,
            example_limit=example_limit,
        ),
        _failure("missing_question_id", "One or more records are missing question_id.", missing_question_ids, example_limit=example_limit),
        _failure(
            "duplicate_question_id",
            "question_id values must be unique.",
            duplicate_question_id_records,
            example_limit=example_limit,
        ),
        _failure(
            "missing_paper_or_question_number",
            "One or more records are missing paper or question_number.",
            missing_paper_question_pairs,
            example_limit=example_limit,
        ),
        _failure(
            "duplicate_paper_question",
            "(paper, question_number) pairs must be unique.",
            duplicate_paper_question_records,
            example_limit=example_limit,
        ),
        _failure(
            "missing_question_image_path",
            "Every record must have a question_image_path.",
            missing_question_image_paths,
            example_limit=example_limit,
        ),
        _failure(
            "absolute_question_image_path",
            "Question image paths must be relative artifact paths.",
            bad_question_path_entries["absolute_question_image_path"],
            example_limit=example_limit,
        ),
        _failure(
            "missing_question_image_file",
            "Question image paths must resolve to files under the artifact root.",
            bad_question_path_entries["missing_question_image_file"],
            example_limit=example_limit,
        ),
        _failure(
            "missing_mark_scheme_image_path",
            "Missing mark-scheme image paths are only allowed for documented source companions.",
            missing_mark_scheme_paths,
            example_limit=example_limit,
        ),
        _failure(
            "absolute_mark_scheme_image_path",
            "Nonblank mark-scheme image paths must be relative artifact paths.",
            bad_mark_scheme_path_entries["absolute_mark_scheme_image_path"],
            example_limit=example_limit,
        ),
        _failure(
            "missing_mark_scheme_image_file",
            "Nonblank mark-scheme image paths must resolve to files under the artifact root.",
            bad_mark_scheme_path_entries["missing_mark_scheme_image_file"],
            example_limit=example_limit,
        ),
    ]
    failures = [item for item in failures if item["count"]]

    report = {
        "schema_name": CURRENT_OUTPUT_INTEGRITY_SCHEMA_NAME,
        "schema_version": CURRENT_OUTPUT_INTEGRITY_SCHEMA_VERSION,
        "input": str(input_path) if input_path is not None else "",
        "artifact_root": str(root),
        "ok": not failures,
        "record_count": len(rows),
        "checks": {
            "declared_record_count_matches": not record_count_mismatch,
            "question_id_present_and_unique": not missing_question_ids and not duplicate_question_id_records,
            "paper_question_pairs_present_and_unique": not missing_paper_question_pairs and not duplicate_paper_question_records,
            "question_image_paths_relative_and_exist": not missing_question_image_paths
            and not bad_question_path_entries["absolute_question_image_path"]
            and not bad_question_path_entries["missing_question_image_file"],
            "nonblank_mark_scheme_image_paths_relative_and_exist": not bad_mark_scheme_path_entries["absolute_mark_scheme_image_path"]
            and not bad_mark_scheme_path_entries["missing_mark_scheme_image_file"],
            "missing_mark_scheme_paths_only_known_companions": not missing_mark_scheme_paths,
        },
        "counts": {
            "declared_record_count": declared_count if isinstance(declared_count, int) else None,
            "unique_question_id_count": len({value for _, value, _ in question_id_records}),
            "missing_question_id_count": len(missing_question_ids),
            "duplicate_question_id_value_count": len(duplicate_question_id_values),
            "duplicate_question_id_record_count": len(duplicate_question_id_records),
            "missing_paper_or_question_number_count": len(missing_paper_question_pairs),
            "duplicate_paper_question_pair_count": len(duplicate_paper_question_values),
            "duplicate_paper_question_record_count": len(duplicate_paper_question_records),
            "question_image_path_entry_count": question_path_entry_count,
            "missing_question_image_path_count": len(missing_question_image_paths),
            "absolute_question_image_path_count": len(bad_question_path_entries["absolute_question_image_path"]),
            "missing_question_image_file_count": len(bad_question_path_entries["missing_question_image_file"]),
            "nonblank_mark_scheme_image_record_count": nonblank_mark_scheme_record_count,
            "mark_scheme_image_path_entry_count": mark_scheme_path_entry_count,
            "missing_mark_scheme_image_path_count": len(missing_mark_scheme_paths) + len(allowed_missing_mark_scheme_paths),
            "allowed_missing_mark_scheme_image_path_count": len(allowed_missing_mark_scheme_paths),
            "unexpected_missing_mark_scheme_image_path_count": len(missing_mark_scheme_paths),
            "absolute_mark_scheme_image_path_count": len(bad_mark_scheme_path_entries["absolute_mark_scheme_image_path"]),
            "missing_mark_scheme_image_file_count": len(bad_mark_scheme_path_entries["missing_mark_scheme_image_file"]),
        },
        "known_missing_mark_scheme_companions": [
            {
                "source_companion": key,
                "paper": str(details["paper"]),
                "reason": str(details["reason"]),
                "observed_missing_count": len(observed_allowed_companions.get(key, [])),
                "observed_question_ids": sorted(observed_allowed_companions.get(key, [])),
            }
            for key, details in sorted(KNOWN_MISSING_MARK_SCHEME_COMPANIONS.items())
        ],
        "failures": failures,
    }
    return report


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


def audit_difficulty(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    score_values = [_note_or_top(record, "difficulty_score") for record in rows]
    return {
        "record_count": len(rows),
        "difficulty_label_counts": _counts(rows, "difficulty"),
        "difficulty_band_counts": _counts(rows, "difficulty_band"),
        "difficulty_confidence_counts": _counts(rows, "difficulty_confidence"),
        "difficulty_score_summary": _score_summary(score_values),
        "difficulty_score_bucket_counts": _difficulty_score_bucket_counts(rows),
        "difficulty_counts_by_paper_family": _difficulty_counts_by(rows, "paper_family", "difficulty"),
        "difficulty_score_summary_by_paper_family": _difficulty_score_summary_by(rows, "paper_family"),
        "difficulty_counts_by_topic": _difficulty_counts_by(rows, "topic", "difficulty"),
        "difficulty_counts_by_marks_bucket": _difficulty_counts_by(rows, "question_solution_marks", "difficulty", bucket_values=True),
        "difficulty_review_flag_counts": _reason_counts(rows, "difficulty_review_flags"),
        "missing_difficulty_metadata": _missing_difficulty_metadata(rows),
    }


def write_audit(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    report = audit_question_bank(load_question_records(input_path))
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_current_output_integrity_audit(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    artifact_root: str | Path | None = None,
    example_limit: int = 10,
) -> dict[str, Any]:
    report = audit_current_output_integrity(input_path, artifact_root=artifact_root, example_limit=example_limit)
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


def write_difficulty_audit(input_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    report = audit_difficulty(load_question_records(input_path))
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return list(payload["questions"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("Integrity audit input must be a question bank document or a list of question records.")


def _infer_artifact_root(input_path: Path, payload: Any) -> Path:
    if isinstance(payload, dict):
        manifest = payload.get("run_manifest")
        if isinstance(manifest, dict) and _clean_text(manifest.get("artifact_root")):
            return Path(str(manifest["artifact_root"]))
    if input_path.parent.name == "json":
        return input_path.parent.parent
    return input_path.parent


def _question_image_path_entries(record: dict[str, Any]) -> list[dict[str, str]]:
    return _path_entries(record, "question_image_path") + _path_entries(record, "canonical_question_artifact") + _path_entries(
        record,
        "question_image_paths",
    )


def _mark_scheme_image_path_entries(record: dict[str, Any]) -> list[dict[str, str]]:
    return _path_entries(record, "mark_scheme_image_path") + _path_entries(record, "mark_scheme_image_paths")


def _path_entries(record: dict[str, Any], field: str) -> list[dict[str, str]]:
    value = record.get(field)
    if isinstance(value, list):
        return [
            {"field": f"{field}[{index}]", "path": str(item).strip()}
            for index, item in enumerate(value)
            if str(item).strip()
        ]
    if _clean_text(value):
        return [{"field": field, "path": str(value).strip()}]
    return []


def _check_path_entry(
    entry: dict[str, str],
    *,
    index: int,
    record: dict[str, Any],
    companion: str,
    artifact_root: Path,
    absolute_code: str,
    missing_file_code: str,
    failures: dict[str, list[dict[str, Any]]],
) -> None:
    path_value = entry["path"]
    ref = _record_ref(index, record, companion=companion)
    ref["field"] = entry["field"]
    ref["path"] = path_value

    if not _is_relative_artifact_path(path_value):
        failures[absolute_code].append(ref)
    if not _artifact_file_exists(path_value, artifact_root):
        missing_ref = dict(ref)
        missing_ref["resolved_path"] = str(artifact_root / path_value) if _is_relative_artifact_path(path_value) else path_value
        failures[missing_file_code].append(missing_ref)


def _is_relative_artifact_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped or "://" in stripped:
        return False
    path = Path(stripped)
    windows_path = PureWindowsPath(stripped)
    return not path.is_absolute() and not windows_path.is_absolute() and not windows_path.drive


def _artifact_file_exists(value: str, artifact_root: Path) -> bool:
    if not _is_relative_artifact_path(value):
        return Path(value).is_file()
    return (artifact_root / value).is_file()


def _record_ref(index: int, record: dict[str, Any], *, companion: str, duplicate_key: str = "") -> dict[str, Any]:
    ref = {
        "index": index,
        "question_id": _clean_text(record.get("question_id")),
        "paper": _clean_text(record.get("paper")),
        "question_number": _clean_text(record.get("question_number")),
        "source_companion": companion,
    }
    if duplicate_key:
        ref["duplicate_key"] = duplicate_key
    return ref


def _failure(code: str, message: str, items: list[dict[str, Any]], *, example_limit: int) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "count": len(items),
        "examples": items[:example_limit],
    }


def _source_companion_key(record: dict[str, Any]) -> str:
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
    return ""


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


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


def _difficulty_score_bucket_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    buckets: Counter[str] = Counter()
    for record in records:
        score = _numeric(_note_or_top(record, "difficulty_score"))
        if score is None:
            buckets["missing"] += 1
        elif score <= 34:
            buckets["0-34"] += 1
        elif score <= 69:
            buckets["35-69"] += 1
        else:
            buckets["70-100"] += 1
    return dict(sorted(buckets.items()))


def _difficulty_counts_by(
    records: list[dict[str, Any]],
    group_key: str,
    value_key: str,
    *,
    bucket_values: bool = False,
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for record in records:
        group_value = _note_or_top(record, group_key)
        group = _marks_bucket(group_value) if bucket_values else str(group_value or "missing")
        value = str(_note_or_top(record, value_key) or "missing")
        grouped.setdefault(group, Counter())[value] += 1
    return {group: dict(sorted(counts.items())) for group, counts in sorted(grouped.items())}


def _difficulty_score_summary_by(records: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        group = str(_note_or_top(record, group_key) or "missing")
        grouped.setdefault(group, []).append(_note_or_top(record, "difficulty_score"))
    return {group: _score_summary(values) for group, values in sorted(grouped.items())}


def _missing_difficulty_metadata(records: list[dict[str, Any]]) -> dict[str, int]:
    fields = [
        "difficulty",
        "difficulty_score",
        "difficulty_band",
        "difficulty_confidence",
        "difficulty_evidence",
        "difficulty_features",
        "difficulty_model_version",
    ]
    return {field: sum(1 for record in records if not _present(_note_or_top(record, field))) for field in fields}


def _marks_bucket(value: Any) -> str:
    mark = _numeric(value)
    if mark is None:
        return "missing"
    if mark <= 3:
        return "1-3"
    if mark <= 6:
        return "4-6"
    if mark <= 9:
        return "7-9"
    return "10+"


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
