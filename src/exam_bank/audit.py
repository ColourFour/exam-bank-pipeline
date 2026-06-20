from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PureWindowsPath
import re
from statistics import mean
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


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
SUSPICIOUS_RENDERED_CROP_MIN_HEIGHT_PX = 1200
SUSPICIOUS_RENDERED_CROP_ABSOLUTE_HEIGHT_PX = 5000
SUSPICIOUS_RENDERED_CROP_MAX_ASPECT_RATIO = 5.0
SUSPICIOUS_RENDERED_CROP_MARGIN_MIN_DIMENSION_PX = 120
SUSPICIOUS_RENDERED_CROP_MAX_BLANK_MARGIN_RATIO = 0.75
SUSPICIOUS_RENDERED_CROP_MIN_CONTENT_AREA_RATIO = 0.08
SUSPICIOUS_RENDERED_CROP_NONWHITE_THRESHOLD = 245

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
_PAPER_ID_RE = re.compile(r"^(?P<component>\d{2})(?P<season>spring|summer|autumn|winter)(?P<year>\d{2})$", re.IGNORECASE)
_PAPER_SEASON_TO_SESSION = {
    "spring": "March",
    "summer": "June",
    "autumn": "November",
    "winter": "November",
}


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
    known_question_numbers_by_paper = _known_question_numbers_by_paper(rows)

    missing_question_ids: list[dict[str, Any]] = []
    missing_paper_question_pairs: list[dict[str, Any]] = []
    question_id_records: list[tuple[int, str, dict[str, Any]]] = []
    paper_question_records: list[tuple[int, tuple[str, str], dict[str, Any]]] = []

    missing_question_image_paths: list[dict[str, Any]] = []
    question_image_path_records: list[tuple[str, dict[str, Any]]] = []
    question_image_path_kind_mismatches: list[dict[str, Any]] = []
    bad_question_path_entries: dict[str, list[dict[str, Any]]] = {
        "absolute_question_image_path": [],
        "missing_question_image_file": [],
    }
    missing_mark_scheme_paths: list[dict[str, Any]] = []
    allowed_missing_mark_scheme_paths: list[dict[str, Any]] = []
    mark_scheme_image_path_records: list[tuple[str, dict[str, Any]]] = []
    mark_scheme_image_path_kind_mismatches: list[dict[str, Any]] = []
    bad_mark_scheme_path_entries: dict[str, list[dict[str, Any]]] = {
        "absolute_mark_scheme_image_path": [],
        "missing_mark_scheme_image_file": [],
    }
    suspicious_rendered_crop_dimensions: list[dict[str, Any]] = []
    suspicious_rendered_crop_whitespace: list[dict[str, Any]] = []
    observed_allowed_companions: dict[str, list[str]] = {key: [] for key in KNOWN_MISSING_MARK_SCHEME_COMPANIONS}

    question_path_entry_count = 0
    mark_scheme_path_entry_count = 0
    nonblank_mark_scheme_record_count = 0
    mark_scheme_text_foreign_question_label_records: list[dict[str, Any]] = []

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
        for entry in _unique_path_entries(question_entries):
            question_image_path_records.append(_path_mapping_record(entry, index=index, record=record, companion=companion))
            _check_image_path_kind(
                entry,
                expected_image_kind="question",
                index=index,
                record=record,
                companion=companion,
                findings=question_image_path_kind_mismatches,
            )
            _check_rendered_crop_dimensions(
                entry,
                image_kind="question",
                index=index,
                record=record,
                companion=companion,
                artifact_root=root,
                findings=suspicious_rendered_crop_dimensions,
            )
            _check_rendered_crop_whitespace(
                entry,
                image_kind="question",
                index=index,
                record=record,
                companion=companion,
                artifact_root=root,
                findings=suspicious_rendered_crop_whitespace,
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
        for entry in _unique_path_entries(mark_scheme_entries):
            mark_scheme_image_path_records.append(_path_mapping_record(entry, index=index, record=record, companion=companion))
            _check_image_path_kind(
                entry,
                expected_image_kind="mark_scheme",
                index=index,
                record=record,
                companion=companion,
                findings=mark_scheme_image_path_kind_mismatches,
            )
            _check_rendered_crop_dimensions(
                entry,
                image_kind="mark_scheme",
                index=index,
                record=record,
                companion=companion,
                artifact_root=root,
                findings=suspicious_rendered_crop_dimensions,
            )
            _check_rendered_crop_whitespace(
                entry,
                image_kind="mark_scheme",
                index=index,
                record=record,
                companion=companion,
                artifact_root=root,
                findings=suspicious_rendered_crop_whitespace,
            )

        foreign_text_labels = _mark_scheme_text_foreign_question_labels(
            record,
            known_question_numbers=known_question_numbers_by_paper.get(paper, set()),
        )
        if foreign_text_labels:
            ref = _record_ref(index, record, companion=companion)
            ref["foreign_question_labels"] = sorted(
                {item["question_number"] for item in foreign_text_labels},
                key=_natural_sort_key,
            )
            ref["line_examples"] = [item["line"] for item in foreign_text_labels[:5]]
            mark_scheme_text_foreign_question_label_records.append(ref)

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
    duplicate_question_image_path_records = _duplicate_path_mapping_records(question_image_path_records)
    duplicate_mark_scheme_image_path_records = _duplicate_path_mapping_records(mark_scheme_image_path_records)
    duplicate_question_image_path_values = {
        path for path, count in Counter(path for path, _ in question_image_path_records).items() if count > 1
    }
    duplicate_mark_scheme_image_path_values = {
        path for path, count in Counter(path for path, _ in mark_scheme_image_path_records).items() if count > 1
    }
    suspicious_rendered_crop_dimensions = _rank_suspicious_rendered_crop_dimensions(suspicious_rendered_crop_dimensions)
    suspicious_rendered_crop_whitespace = _rank_suspicious_rendered_crop_whitespace(suspicious_rendered_crop_whitespace)

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
            "duplicate_question_image_path",
            "Question image paths should not be shared by multiple records.",
            duplicate_question_image_path_records,
            example_limit=example_limit,
        ),
        _failure(
            "question_image_path_kind_mismatch",
            "Question image fields must not point at mark-scheme-looking artifact paths.",
            question_image_path_kind_mismatches,
            example_limit=example_limit,
        ),
        _failure(
            "duplicate_mark_scheme_image_path",
            "Mark-scheme image paths should not be shared by multiple records.",
            duplicate_mark_scheme_image_path_records,
            example_limit=example_limit,
        ),
        _failure(
            "mark_scheme_image_path_kind_mismatch",
            "Mark-scheme image fields must not point at question-looking artifact paths.",
            mark_scheme_image_path_kind_mismatches,
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
        _failure(
            "mark_scheme_text_foreign_question_label",
            "mark_scheme_text must not contain top-level labels for other questions.",
            mark_scheme_text_foreign_question_label_records,
            example_limit=example_limit,
        ),
        _failure(
            "suspicious_rendered_crop_dimensions",
            "Rendered question and mark-scheme crops must not have extreme dimensions without review.",
            suspicious_rendered_crop_dimensions,
            example_limit=example_limit,
        ),
        _failure(
            "suspicious_rendered_crop_whitespace",
            "Rendered question and mark-scheme crops must not contain excessive blank margins.",
            suspicious_rendered_crop_whitespace,
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
            "image_paths_unique_across_records": not duplicate_question_image_path_records
            and not duplicate_mark_scheme_image_path_records,
            "image_path_roles_match_fields": not question_image_path_kind_mismatches
            and not mark_scheme_image_path_kind_mismatches,
            "question_image_paths_relative_and_exist": not missing_question_image_paths
            and not bad_question_path_entries["absolute_question_image_path"]
            and not bad_question_path_entries["missing_question_image_file"],
            "nonblank_mark_scheme_image_paths_relative_and_exist": not bad_mark_scheme_path_entries["absolute_mark_scheme_image_path"]
            and not bad_mark_scheme_path_entries["missing_mark_scheme_image_file"],
            "missing_mark_scheme_paths_only_known_companions": not missing_mark_scheme_paths,
            "mark_scheme_text_has_no_foreign_question_labels": not mark_scheme_text_foreign_question_label_records,
            "rendered_crop_dimensions_not_suspicious": not suspicious_rendered_crop_dimensions,
            "rendered_crop_whitespace_not_suspicious": not suspicious_rendered_crop_whitespace,
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
            "duplicate_question_image_path_value_count": len(duplicate_question_image_path_values),
            "duplicate_question_image_path_record_count": len(duplicate_question_image_path_records),
            "duplicate_mark_scheme_image_path_value_count": len(duplicate_mark_scheme_image_path_values),
            "duplicate_mark_scheme_image_path_record_count": len(duplicate_mark_scheme_image_path_records),
            "question_image_path_kind_mismatch_count": len(question_image_path_kind_mismatches),
            "mark_scheme_image_path_kind_mismatch_count": len(mark_scheme_image_path_kind_mismatches),
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
            "mark_scheme_text_foreign_question_label_record_count": len(
                mark_scheme_text_foreign_question_label_records
            ),
            "suspicious_rendered_crop_dimension_count": len(suspicious_rendered_crop_dimensions),
            "suspicious_rendered_crop_whitespace_count": len(suspicious_rendered_crop_whitespace),
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
        "unexpected_missing_mark_scheme_groups": _missing_mark_scheme_groups(missing_mark_scheme_paths),
        "mark_scheme_text_foreign_label_summary": _mark_scheme_text_foreign_label_summary(
            mark_scheme_text_foreign_question_label_records
        ),
        "mark_scheme_text_foreign_label_review_candidates": _mark_scheme_text_foreign_label_review_candidates(
            mark_scheme_text_foreign_question_label_records,
            limit=example_limit,
        ),
        "suspicious_rendered_crop_dimension_summary": _suspicious_rendered_crop_dimension_summary(
            suspicious_rendered_crop_dimensions
        ),
        "suspicious_rendered_crop_review_candidates": _suspicious_rendered_crop_review_candidates(
            suspicious_rendered_crop_dimensions,
            limit=example_limit,
        ),
        "suspicious_rendered_crop_whitespace_summary": _suspicious_rendered_crop_whitespace_summary(
            suspicious_rendered_crop_whitespace
        ),
        "suspicious_rendered_crop_whitespace_review_candidates": _suspicious_rendered_crop_whitespace_review_candidates(
            suspicious_rendered_crop_whitespace,
            limit=example_limit,
        ),
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


def _unique_path_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for entry in entries:
        path_value = entry["path"]
        if path_value in seen:
            continue
        seen.add(path_value)
        unique.append(entry)
    return unique


def _path_mapping_record(
    entry: dict[str, str],
    *,
    index: int,
    record: dict[str, Any],
    companion: str,
) -> tuple[str, dict[str, Any]]:
    ref = _record_ref(index, record, companion=companion)
    ref["field"] = entry["field"]
    ref["path"] = entry["path"]
    return entry["path"], ref


def _duplicate_path_mapping_records(path_records: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    duplicate_values = {path for path, count in Counter(path for path, _ in path_records).items() if count > 1}
    duplicate_records: list[dict[str, Any]] = []
    for path_value, ref in path_records:
        if path_value not in duplicate_values:
            continue
        duplicate_ref = dict(ref)
        duplicate_ref["duplicate_key"] = path_value
        duplicate_records.append(duplicate_ref)
    return duplicate_records


def _known_question_numbers_by_paper(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for record in records:
        paper = _clean_text(record.get("paper"))
        question_number = parent_question_number(_clean_text(record.get("question_number")))
        if not paper or not question_number:
            continue
        known.setdefault(paper, set()).add(question_number)
    return known


def _mark_scheme_text_foreign_question_labels(
    record: dict[str, Any],
    *,
    known_question_numbers: set[str],
) -> list[dict[str, str]]:
    current_number = _clean_text(record.get("question_number"))
    if not current_number:
        return []
    current_number = parent_question_number(current_number)
    text = _clean_text(record.get("mark_scheme_text"))
    if not text:
        return []

    findings: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.replace("\u00a0", " ").split())
        if not line:
            continue
        label = _top_level_mark_scheme_text_label(line)
        if label is None or label == current_number:
            continue
        if known_question_numbers and label not in known_question_numbers:
            continue
        findings.append({"question_number": label, "line": line})
    return findings


def _top_level_mark_scheme_text_label(line: str) -> str | None:
    match = re.match(r"^(?P<number>\d{1,2})(?P<rest>.*)$", line)
    if not match:
        return None
    number = str(int(match.group("number")))
    rest = match.group("rest")
    if not rest:
        return None
    if re.match(r"^\([a-zivx]+\)", rest, re.IGNORECASE):
        return number
    if re.match(r"^\s+\([a-zivx]+\)", rest, re.IGNORECASE):
        return number
    if re.match(r"^\s{2,}\([a-zivx]+\)", rest, re.IGNORECASE):
        return number
    return None


def parent_question_number(value: str) -> str:
    match = re.match(r"^(\d{1,2})", value.strip())
    return str(int(match.group(1))) if match else value.strip()


def _check_image_path_kind(
    entry: dict[str, str],
    *,
    expected_image_kind: str,
    index: int,
    record: dict[str, Any],
    companion: str,
    findings: list[dict[str, Any]],
) -> None:
    mismatch = _image_path_kind_mismatch(entry["path"], expected_image_kind=expected_image_kind)
    if mismatch is None:
        return
    ref = _record_ref(index, record, companion=companion)
    ref.update(
        {
            "field": entry["field"],
            "path": entry["path"],
            "expected_image_kind": expected_image_kind,
            "inferred_path_role": mismatch["inferred_path_role"],
            "reason": mismatch["reason"],
        }
    )
    findings.append(ref)


def _image_path_kind_mismatch(path_value: str, *, expected_image_kind: str) -> dict[str, str] | None:
    roles = _strong_image_path_roles(path_value)
    if expected_image_kind == "question" and "mark_scheme" in roles:
        return {
            "inferred_path_role": "mark_scheme",
            "reason": "path_looks_like_mark_scheme_artifact",
        }
    if expected_image_kind == "mark_scheme" and "question" in roles:
        return {
            "inferred_path_role": "question",
            "reason": "path_looks_like_question_artifact",
        }
    return None


def _strong_image_path_roles(path_value: str) -> set[str]:
    normalized = path_value.replace("\\", "/").lower()
    segments = [segment for segment in normalized.split("/") if segment]
    basename = segments[-1] if segments else normalized
    roles: set[str] = set()

    if any(segment in {"mark_scheme", "mark_schemes"} for segment in segments):
        roles.add("mark_scheme")
    if "markscheme" in basename or "_ms_" in basename or basename.startswith("ms_"):
        roles.add("mark_scheme")

    if any(segment == "questions" for segment in segments):
        roles.add("question")
    if "_question" in basename or "_qp_" in basename or basename.startswith("qp_"):
        roles.add("question")

    return roles


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


def _check_rendered_crop_dimensions(
    entry: dict[str, str],
    *,
    image_kind: str,
    index: int,
    record: dict[str, Any],
    companion: str,
    artifact_root: Path,
    findings: list[dict[str, Any]],
) -> None:
    path_value = entry["path"]
    if not _is_relative_artifact_path(path_value):
        return
    resolved_path = artifact_root / path_value
    if not resolved_path.is_file():
        return
    dimensions = _read_image_dimensions(resolved_path)
    if dimensions is None:
        return
    width, height = dimensions
    reasons = _suspicious_rendered_crop_dimension_reasons(width=width, height=height)
    if not reasons:
        return

    ref = _record_ref(index, record, companion=companion)
    ref.update(
        {
            "field": entry["field"],
            "path": path_value,
            "resolved_path": str(resolved_path),
            "image_kind": image_kind,
            "width_px": width,
            "height_px": height,
            "aspect_ratio": round(height / width, 3) if width else None,
            "reasons": reasons,
        }
    )
    findings.append(ref)


def _check_rendered_crop_whitespace(
    entry: dict[str, str],
    *,
    image_kind: str,
    index: int,
    record: dict[str, Any],
    companion: str,
    artifact_root: Path,
    findings: list[dict[str, Any]],
) -> None:
    path_value = entry["path"]
    if not _is_relative_artifact_path(path_value):
        return
    resolved_path = artifact_root / path_value
    if not resolved_path.is_file():
        return
    metrics = _read_image_content_metrics(resolved_path)
    if metrics is None:
        return
    reasons = _suspicious_rendered_crop_whitespace_reasons(metrics)
    if not reasons:
        return

    ref = _record_ref(index, record, companion=companion)
    ref.update(
        {
            "field": entry["field"],
            "path": path_value,
            "resolved_path": str(resolved_path),
            "image_kind": image_kind,
            "width_px": metrics["width_px"],
            "height_px": metrics["height_px"],
            "content_bbox": metrics["content_bbox"],
            "blank_top_ratio": metrics["blank_top_ratio"],
            "blank_bottom_ratio": metrics["blank_bottom_ratio"],
            "blank_left_ratio": metrics["blank_left_ratio"],
            "blank_right_ratio": metrics["blank_right_ratio"],
            "content_area_ratio": metrics["content_area_ratio"],
            "reasons": reasons,
        }
    )
    findings.append(ref)


def _read_image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError):
        return None


def _read_image_content_metrics(path: Path) -> dict[str, Any] | None:
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            width, height = grayscale.size
            if width <= 0 or height <= 0:
                return None
            mask = grayscale.point(
                lambda pixel: 255 if pixel < SUSPICIOUS_RENDERED_CROP_NONWHITE_THRESHOLD else 0,
                mode="1",
            )
            bbox = mask.getbbox()
    except (OSError, UnidentifiedImageError):
        return None
    if bbox is None:
        return {
            "width_px": width,
            "height_px": height,
            "content_bbox": None,
            "blank_top_ratio": 1.0,
            "blank_bottom_ratio": 1.0,
            "blank_left_ratio": 1.0,
            "blank_right_ratio": 1.0,
            "content_area_ratio": 0.0,
        }

    left, top, right, bottom = bbox
    content_width = max(0, right - left)
    content_height = max(0, bottom - top)
    return {
        "width_px": width,
        "height_px": height,
        "content_bbox": [left, top, right, bottom],
        "blank_top_ratio": round(top / height, 3),
        "blank_bottom_ratio": round((height - bottom) / height, 3),
        "blank_left_ratio": round(left / width, 3),
        "blank_right_ratio": round((width - right) / width, 3),
        "content_area_ratio": round((content_width * content_height) / (width * height), 3),
    }


def _suspicious_rendered_crop_dimension_reasons(*, width: int, height: int) -> list[str]:
    reasons: list[str] = []
    if width > 0 and height >= SUSPICIOUS_RENDERED_CROP_MIN_HEIGHT_PX:
        aspect_ratio = height / width
        if aspect_ratio >= SUSPICIOUS_RENDERED_CROP_MAX_ASPECT_RATIO:
            reasons.append("extreme_aspect_ratio")
            if height >= SUSPICIOUS_RENDERED_CROP_ABSOLUTE_HEIGHT_PX:
                reasons.append("very_tall_rendered_crop")
    return reasons


def _suspicious_rendered_crop_whitespace_reasons(metrics: dict[str, Any]) -> list[str]:
    width = int(metrics.get("width_px") or 0)
    height = int(metrics.get("height_px") or 0)
    if width < SUSPICIOUS_RENDERED_CROP_MARGIN_MIN_DIMENSION_PX or height < SUSPICIOUS_RENDERED_CROP_MARGIN_MIN_DIMENSION_PX:
        return []
    if metrics.get("content_bbox") is None:
        return []

    reasons: list[str] = []
    if float(metrics.get("blank_top_ratio") or 0) >= SUSPICIOUS_RENDERED_CROP_MAX_BLANK_MARGIN_RATIO:
        reasons.append("excessive_blank_top_margin")
    if float(metrics.get("blank_bottom_ratio") or 0) >= SUSPICIOUS_RENDERED_CROP_MAX_BLANK_MARGIN_RATIO:
        reasons.append("excessive_blank_bottom_margin")
    if float(metrics.get("content_area_ratio") or 0) <= SUSPICIOUS_RENDERED_CROP_MIN_CONTENT_AREA_RATIO:
        reasons.append("very_low_content_area")
    return reasons


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


def _missing_mark_scheme_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        key = (_clean_text(item.get("paper")) or "missing", _clean_text(item.get("source_companion")) or "missing")
        grouped.setdefault(key, []).append(item)
    result: list[dict[str, Any]] = []
    for (paper, companion), records in sorted(grouped.items()):
        result.append(
            {
                "paper": paper,
                "source_companion": companion,
                "count": len(records),
                "question_ids": sorted(_clean_text(record.get("question_id")) for record in records if _clean_text(record.get("question_id"))),
                "question_numbers": sorted(
                    (_clean_text(record.get("question_number")) for record in records if _clean_text(record.get("question_number"))),
                    key=_natural_sort_key,
                ),
            }
        )
    return result


def _mark_scheme_text_foreign_label_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = Counter(label for item in items for label in _list_field(item, "foreign_question_labels"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(_clean_text(item.get("paper")) or "missing", []).append(item)

    top_papers: list[dict[str, Any]] = []
    for paper, paper_items in sorted(grouped.items(), key=lambda entry: (-len(entry[1]), entry[0]))[:10]:
        label_counts = Counter(label for item in paper_items for label in _list_field(item, "foreign_question_labels"))
        top_papers.append(
            {
                "paper": paper,
                "count": len(paper_items),
                "question_ids": sorted(
                    {
                        _clean_text(item.get("question_id"))
                        for item in paper_items
                        if _clean_text(item.get("question_id"))
                    }
                ),
                "foreign_label_counts": dict(sorted(label_counts.items(), key=lambda item: _natural_sort_key(item[0]))),
            }
        )
    return {
        "by_foreign_label": dict(sorted(by_label.items(), key=lambda item: _natural_sort_key(item[0]))),
        "top_papers": top_papers,
    }


def _mark_scheme_text_foreign_label_review_candidates(
    items: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        items,
        key=lambda item: (
            -len(_list_field(item, "foreign_question_labels")),
            -len(_list_field(item, "line_examples")),
            _clean_text(item.get("paper")),
            _natural_sort_key(_clean_text(item.get("question_number"))),
            _clean_text(item.get("question_id")),
        ),
    )
    return [_mark_scheme_text_foreign_label_candidate(item) for item in ranked[: max(0, limit)]]


def _mark_scheme_text_foreign_label_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": _clean_text(item.get("question_id")),
        "paper": _clean_text(item.get("paper")),
        "question_number": _clean_text(item.get("question_number")),
        "foreign_question_labels": _list_field(item, "foreign_question_labels"),
        "line_examples": _list_field(item, "line_examples"),
    }


def _suspicious_rendered_crop_dimension_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_image_kind = Counter(_clean_text(item.get("image_kind")) or "missing" for item in items)
    by_reason = Counter(reason for item in items for reason in _list_field(item, "reasons"))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(_clean_text(item.get("paper")) or "missing", []).append(item)

    top_papers: list[dict[str, Any]] = []
    for paper, paper_items in sorted(grouped.items(), key=lambda entry: (-len(entry[1]), entry[0]))[:10]:
        image_kind_counts = Counter(_clean_text(item.get("image_kind")) or "missing" for item in paper_items)
        top_papers.append(
            {
                "paper": paper,
                "count": len(paper_items),
                "question_ids": sorted(
                    {
                        _clean_text(item.get("question_id"))
                        for item in paper_items
                        if _clean_text(item.get("question_id"))
                    }
                ),
                "image_kind_counts": dict(sorted(image_kind_counts.items())),
            }
        )

    max_height_candidate = max(
        items,
        key=lambda item: (_numeric(item.get("height_px")) or -1, _clean_text(item.get("path"))),
        default=None,
    )
    max_aspect_ratio_candidate = max(
        items,
        key=lambda item: (_numeric(item.get("aspect_ratio")) or -1, _clean_text(item.get("path"))),
        default=None,
    )

    return {
        "by_image_kind": dict(sorted(by_image_kind.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "top_papers": top_papers,
        "max_height_candidate": _dimension_candidate_summary(max_height_candidate),
        "max_aspect_ratio_candidate": _dimension_candidate_summary(max_aspect_ratio_candidate),
    }


def _rank_suspicious_rendered_crop_dimensions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=_suspicious_rendered_crop_sort_key)


def _suspicious_rendered_crop_sort_key(item: dict[str, Any]) -> tuple[int, float, float, str]:
    return (
        -len(_list_field(item, "reasons")),
        -float(_numeric(item.get("height_px")) or 0),
        -float(_numeric(item.get("aspect_ratio")) or 0),
        _clean_text(item.get("path")),
    )


def _suspicious_rendered_crop_review_candidates(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in (_dimension_candidate_summary(item) for item in items[: max(0, limit)])
        if candidate is not None
    ]


def _suspicious_rendered_crop_whitespace_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_image_kind = Counter(_clean_text(item.get("image_kind")) or "missing" for item in items)
    by_reason = Counter(reason for item in items for reason in _list_field(item, "reasons"))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(_clean_text(item.get("paper")) or "missing", []).append(item)

    top_papers: list[dict[str, Any]] = []
    for paper, paper_items in sorted(grouped.items(), key=lambda entry: (-len(entry[1]), entry[0]))[:10]:
        image_kind_counts = Counter(_clean_text(item.get("image_kind")) or "missing" for item in paper_items)
        top_papers.append(
            {
                "paper": paper,
                "count": len(paper_items),
                "question_ids": sorted(
                    {
                        _clean_text(item.get("question_id"))
                        for item in paper_items
                        if _clean_text(item.get("question_id"))
                    }
                ),
                "image_kind_counts": dict(sorted(image_kind_counts.items())),
            }
        )

    max_bottom_margin_candidate = max(
        items,
        key=lambda item: (_numeric(item.get("blank_bottom_ratio")) or -1, _clean_text(item.get("path"))),
        default=None,
    )
    min_content_area_candidate = min(
        items,
        key=lambda item: (_numeric(item.get("content_area_ratio")) or 2, _clean_text(item.get("path"))),
        default=None,
    )

    return {
        "by_image_kind": dict(sorted(by_image_kind.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "top_papers": top_papers,
        "max_bottom_margin_candidate": _whitespace_candidate_summary(max_bottom_margin_candidate),
        "min_content_area_candidate": _whitespace_candidate_summary(min_content_area_candidate),
    }


def _rank_suspicious_rendered_crop_whitespace(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=_suspicious_rendered_crop_whitespace_sort_key)


def _suspicious_rendered_crop_whitespace_sort_key(item: dict[str, Any]) -> tuple[int, float, float, str]:
    return (
        -len(_list_field(item, "reasons")),
        -float(_numeric(item.get("blank_bottom_ratio")) or 0),
        float(_numeric(item.get("content_area_ratio")) or 1),
        _clean_text(item.get("path")),
    )


def _suspicious_rendered_crop_whitespace_review_candidates(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in (_whitespace_candidate_summary(item) for item in items[: max(0, limit)])
        if candidate is not None
    ]


def _whitespace_candidate_summary(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "question_id": _clean_text(item.get("question_id")),
        "paper": _clean_text(item.get("paper")),
        "question_number": _clean_text(item.get("question_number")),
        "image_kind": _clean_text(item.get("image_kind")),
        "path": _clean_text(item.get("path")),
        "width_px": item.get("width_px"),
        "height_px": item.get("height_px"),
        "content_bbox": item.get("content_bbox"),
        "blank_top_ratio": item.get("blank_top_ratio"),
        "blank_bottom_ratio": item.get("blank_bottom_ratio"),
        "content_area_ratio": item.get("content_area_ratio"),
        "reasons": _list_field(item, "reasons"),
    }


def _dimension_candidate_summary(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "question_id": _clean_text(item.get("question_id")),
        "paper": _clean_text(item.get("paper")),
        "question_number": _clean_text(item.get("question_number")),
        "image_kind": _clean_text(item.get("image_kind")),
        "path": _clean_text(item.get("path")),
        "width_px": item.get("width_px"),
        "height_px": item.get("height_px"),
        "aspect_ratio": item.get("aspect_ratio"),
        "reasons": _list_field(item, "reasons"),
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
    inferred = _source_companion_key_from_paper(paper)
    if inferred:
        return inferred
    return ""


def _source_companion_key_from_paper(paper: str) -> str:
    match = _PAPER_ID_RE.fullmatch(paper)
    if not match:
        return ""
    session = _PAPER_SEASON_TO_SESSION.get(match.group("season").lower())
    if not session:
        return ""
    return f"9709_20{match.group('year')}_{session}_{match.group('component')}"


def _natural_sort_key(value: str) -> list[int | str]:
    parts: list[int | str] = []
    for part in re.split(r"(\d+)", value):
        if not part:
            continue
        parts.append(int(part) if part.isdigit() else part)
    return parts


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
