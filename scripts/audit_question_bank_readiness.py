from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


AUDIT_VERSION = "iteration_001_readiness_audit_v1"
SAMPLE_SEED = "20260510"


TOP_LEVEL_PREFERRED_FIELDS = {
    "question_id",
    "paper",
    "paper_family",
    "question_number",
    "canonical_question_artifact",
    "question_image_path",
    "question_image_paths",
    "mark_scheme_image_path",
    "mark_scheme_image_paths",
    "page_refs",
    "question_text",
    "question_text_role",
    "question_text_trust",
    "ocr_ran",
    "ocr_engine",
    "ocr_text",
    "ocr_text_trust",
    "ocr_failure_reason",
    "visual_required",
    "visual_reason_flags",
    "visual_curation_status",
    "text_only_status",
    "mark_scheme_text",
    "question_solution_marks",
    "difficulty",
    "difficulty_score",
    "difficulty_band",
    "subparts",
    "subparts_solution_marks",
    "topic",
}

NOTES_PREFERRED_FIELDS = {
    "subtopic",
    "source_pdf",
    "mark_scheme_source_pdf",
    "source_paper_code",
    "full_question_label",
    "topic_confidence",
    "topic_uncertain",
    "topic_trust_status",
    "difficulty_confidence",
    "difficulty_evidence",
    "difficulty_uncertain",
    "difficulty_score_scale",
    "difficulty_features",
    "difficulty_review_flags",
    "difficulty_model_version",
    "mapping_status",
    "mapping_failure_reason",
    "scope_quality_status",
    "question_crop_confidence",
    "text_source_profile",
    "text_fidelity_status",
    "text_fidelity_flags",
    "mark_scheme_crop_confidence",
    "review_flags",
    "extraction_quality_score",
    "extraction_quality_flags",
    "validation_status",
    "validation_flags",
    "recovery_attempted",
    "recovery_result",
    "ocr_text_role",
    "text_candidate_source",
    "native_text_score",
    "ocr_text_score",
    "selected_text_score",
    "text_candidate_decision",
    "text_candidate_decision_reasons",
    "ocr_selected",
    "ocr_rejected_reasons",
    "question_structure_detected",
    "mark_scheme_structure_detected",
    "question_total_detected",
    "mark_scheme_total_detected",
    "question_format_profile",
    "paper_total_expected",
    "paper_total_detected",
    "paper_total_status",
    "rescan_triggered",
    "rescan_result",
    "paper_total_before_rescan",
    "paper_total_after_rescan",
    "paper_total_focus_questions",
    "paper_total_focus_pages",
    "paper_total_focus_reason",
}

EXPECTED_FIELDS = [
    "question_id",
    "paper",
    "paper_family",
    "question_number",
    "question_image_path",
    "mark_scheme_image_path",
    "question_text",
    "ocr_text",
    "ocr_ran",
    "ocr_engine",
    "ocr_text_trust",
    "text_candidate_source",
    "native_text_score",
    "ocr_text_score",
    "selected_text_score",
    "text_candidate_decision",
    "text_candidate_decision_reasons",
    "ocr_selected",
    "ocr_rejected_reasons",
    "question_text_trust",
    "text_fidelity_status",
    "text_only_status",
    "visual_required",
    "visual_curation_status",
    "question_crop_confidence",
    "mark_scheme_crop_confidence",
    "mapping_status",
    "validation_status",
    "question_structure_detected",
    "mark_scheme_structure_detected",
    "question_total_detected",
    "mark_scheme_total_detected",
    "paper_total_status",
    "subparts",
    "subparts_solution_marks",
    "mark_scheme_text",
    "question_solution_marks",
    "question_text_role",
    "ocr_text_role",
    "scope_quality_status",
    "question_format_profile",
    "review_flags",
    "validation_flags",
    "text_fidelity_flags",
    "visual_reason_flags",
    "mapping_failure_reason",
    "topic",
    "subtopic",
    "topic_confidence",
    "topic_uncertain",
    "topic_trust_status",
    "difficulty",
    "difficulty_confidence",
    "difficulty_score",
    "difficulty_band",
    "difficulty_model_version",
]

RUN_METADATA_FIELDS = [
    "generated_at",
    "run_id",
    "pipeline_version",
    "git_commit",
    "model_versions",
    "ocr_engine_version",
    "input_manifest_sha256",
    "artifact_root",
    "qa_summary",
]

OCR_MEASUREMENT_FIELDS = {
    "ocr_ran",
    "ocr_engine",
    "ocr_text",
    "ocr_text_trust",
    "text_candidate_source",
    "native_text_score",
    "ocr_text_score",
    "selected_text_score",
    "text_candidate_decision",
    "text_candidate_decision_reasons",
    "ocr_selected",
    "ocr_rejected_reasons",
}

REVIEW_FLAG_NAMES = [
    "crop_uncertain",
    "low_confidence_question_crop",
    "weak_question_anchor",
    "missing_terminal_mark_total",
    "partial_question_block",
    "ocr_merged_sparse_lower_region",
    "crop_split_prompt_regions",
    "markscheme_image_stitched",
    "markscheme_image_uncertain",
    "side_panel_excluded",
    "barcode_excluded",
    "page_diagram_union_used",
    "page_diagram_union_skipped_neighbor_question",
    "question_text_figure_overlap_prevented",
    "text_figure_overlap_trimmed",
    "excluded_boilerplate_copyright_footer",
]

STATUS_ORDER = {
    "fail": 0,
    "unusable": 0,
    "degraded": 0,
    "review": 1,
    "ready": 2,
    "pass": 2,
    "clean": 2,
}
TRUST_ORDER = {
    "unusable": -1,
    "low": 0,
    "medium": 1,
    "high": 2,
}
FIDELITY_ORDER = {
    "unusable": 0,
    "degraded": 0,
    "review": 1,
    "clean": 2,
}
CONFIDENCE_ORDER = {
    "": 0,
    "blank": 0,
    "missing": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

FIELD_ORDERINGS = {
    "text_fidelity_status": FIDELITY_ORDER,
    "text_only_status": STATUS_ORDER,
    "visual_curation_status": STATUS_ORDER,
    "question_text_trust": TRUST_ORDER,
    "mapping_status": STATUS_ORDER,
    "validation_status": STATUS_ORDER,
    "question_crop_confidence": CONFIDENCE_ORDER,
    "mark_scheme_crop_confidence": CONFIDENCE_ORDER,
}

COMPARE_FIELDS = [
    "question_text",
    "text_candidate_source",
    "ocr_selected",
    "text_candidate_decision",
    "native_text_score",
    "ocr_text_score",
    "selected_text_score",
    "text_fidelity_status",
    "text_only_status",
    "visual_curation_status",
    "question_text_trust",
    "topic",
    "subtopic",
    "topic_confidence",
    "difficulty",
    "difficulty_confidence",
    "mapping_status",
    "validation_status",
    "question_crop_confidence",
    "mark_scheme_crop_confidence",
    "question_solution_marks",
    "question_total_detected",
    "mark_scheme_total_detected",
]

QUESTION_NUMBER_RE_TEMPLATE = r"(?:^|\s|\n){number}(?:\s|\.|\(|\)|$)"
MARK_RE = re.compile(r"\[(\d{1,2})\]")
ALPHA_SUBPART_RE = re.compile(r"\(([a-h])\)", re.IGNORECASE)
ROMAN_SUBPART_RE = re.compile(r"\((viii|vii|vi|iv|ix|iii|ii|i|v|x)\)", re.IGNORECASE)
MATH_TOKEN_RE = re.compile(
    r"(?:=|<|>|<=|>=|≤|≥|\^|√|π|θ|∫|Σ|sin|cos|tan|sec|cosec|cot|ln|log|dy/dx|dx/dt|arg|vector|modulus|\|z\|)",
    re.IGNORECASE,
)
FUNCTION_TOKEN_RE = re.compile(r"\b(?:sin|cos|tan|sec|cosec|cot|ln|log|exp)\s*(?:\(|[A-Za-z0-9θ])", re.IGNORECASE)
PAGE_FURNITURE_RE = re.compile(
    r"\b(?:UCLES|Cambridge|BLANK PAGE|Additional Materials|READ THESE INSTRUCTIONS|INSTRUCTIONS|Question Paper|Mark Scheme)\b",
    re.IGNORECASE,
)
BARCODE_RE = re.compile(r"\b(?:PUTT|RT TT|VARTA|ARTY|RACY|BLANK|[A-Z]{4,}\s+[A-Z]{4,})\b")
COPYRIGHT_RE = re.compile(r"\b(?:copyright|©|Cambridge Assessment|University of Cambridge|UCLES)\b", re.IGNORECASE)
MARGIN_RE = re.compile(r"\b(?:Turn over|For Examiner'?s Use|Additional page|Answer all the questions)\b", re.IGNORECASE)
NEXT_QUESTION_RE = re.compile(r"\[\d{1,2}\]\s+\d{1,2}\s+(?:Find|Show|Solve|Given|The diagram|Use|Calculate)", re.IGNORECASE)
ISOLATED_SYMBOL_RE = re.compile(r"(?:^|\s)[^\w\s()\[\].,;:=+\-*/^]{1,2}(?=\s|$)")
MATH_MANGLING_RE = re.compile(
    r"(?:\b(?:sin|cos|tan|ln|log)\s+[A-Za-zθ]\s+\d\b|[?�]{2,}|[=<>^/*_]{4,}|(?:\d\s+[A-Za-z]\s+\d\s+[A-Za-z]\s+\d)|(?:\b[il]\b\s*[=<]\s*\d))",
    re.IGNORECASE,
)
MERGED_PROSE_RE = re.compile(
    r"\b(?:Findthe|findthe|showthat|Showthat|Givethe|Giveyour|answerintheform|whereaandbare|Thediagram|thediagram|thegraph|formgraph|straightof|andshows|solvethe|intheform)\b"
)
PAPER_RE = re.compile(r"^(?P<component>\d{2})(?P<season>spring|summer|autumn)(?P<yy>\d{2})$", re.IGNORECASE)


class FieldResolver:
    def __init__(self) -> None:
        self.disagreements: dict[tuple[str, str], dict[str, Any]] = {}

    def preferred_source(self, field: str) -> str:
        if field in NOTES_PREFERRED_FIELDS:
            return "notes"
        return "top_level"

    def get(self, record: dict[str, Any], field: str) -> Any:
        top_has = field in record
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        notes_has = field in notes
        top_value = record.get(field) if top_has else None
        notes_value = notes.get(field) if notes_has else None
        question_id = str(record.get("question_id") or "<missing>")

        if top_has and notes_has and not _values_equal(top_value, notes_value):
            self.disagreements.setdefault(
                (question_id, field),
                {
                    "question_id": question_id,
                    "field": field,
                    "preferred_source": self.preferred_source(field),
                    "top_level_value": top_value,
                    "notes_value": notes_value,
                },
            )

        if top_has and notes_has:
            return notes_value if self.preferred_source(field) == "notes" else top_value
        if top_has:
            return top_value
        if notes_has:
            return notes_value
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit question-bank OCR selection, extraction readiness, and Asterion tiers.")
    parser.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    parser.add_argument("--baseline", default="", help="Optional baseline question_bank.json.")
    parser.add_argument("--artifact-root", default="", help="Optional root used to verify relative question and mark-scheme image paths.")
    parser.add_argument("--out-dir", required=True, help="Directory for audit reports.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    baseline_path = Path(args.baseline) if args.baseline else None
    artifact_root = Path(args.artifact_root) if args.artifact_root else None

    payload, records = load_question_bank(input_path)
    baseline_payload: dict[str, Any] | None = None
    baseline_records: list[dict[str, Any]] | None = None
    if baseline_path:
        baseline_payload, baseline_records = load_question_bank(baseline_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_audit_report(
        payload=payload,
        records=records,
        input_path=input_path,
        out_dir=out_dir,
        artifact_root=artifact_root,
        baseline_payload=baseline_payload,
        baseline_records=baseline_records,
        baseline_path=baseline_path,
    )

    write_json(out_dir / "audit_summary.json", report["summary"])
    write_markdown(out_dir / "audit_summary.md", render_summary_md(report["summary"]))
    write_markdown(out_dir / "next_iteration_recommendations.md", render_recommendations_md(report["summary"]))
    write_csv(out_dir / "field_presence_report.csv", report["field_presence"])
    write_csv(out_dir / "ocr_candidate_audit.csv", report["ocr_rows"])
    write_csv(out_dir / "ocr_suspicious_records.csv", report["ocr_suspicious_rows"])
    write_csv(out_dir / "possible_ocr_false_negatives.csv", report["ocr_false_negative_rows"])
    write_csv(out_dir / "readiness_tiers.csv", report["readiness_rows"])
    write_csv(out_dir / "hard_blockers.csv", report["hard_blocker_rows"])
    write_csv(out_dir / "crop_quality_report.csv", report["crop_quality_rows"])
    write_csv(out_dir / "mapping_validation_report.csv", report["mapping_validation_rows"])
    write_csv(out_dir / "subpart_marks_report.csv", report["subpart_rows"])
    write_csv(out_dir / "representative_review_sample.csv", report["sample_rows"])

    if report["baseline_rows"] is not None:
        write_csv(out_dir / "baseline_comparison.csv", report["baseline_rows"])
        write_json(out_dir / "baseline_comparison_summary.json", report["baseline_summary"])

    print(f"Wrote audit reports to {out_dir}")
    print(f"Records audited: {len(records)}")
    if baseline_path:
        print(f"Baseline compared: {baseline_path}")
    if artifact_root:
        print(f"Artifact root checked: {artifact_root}")
    return 0


def load_question_bank(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return payload, list(payload["questions"])
    if isinstance(payload, list):
        return {"questions": payload}, list(payload)
    raise ValueError(f"{path} is not a question-bank JSON document or record list.")


def build_audit_report(
    *,
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    input_path: Path,
    out_dir: Path,
    artifact_root: Path | None,
    baseline_payload: dict[str, Any] | None,
    baseline_records: list[dict[str, Any]] | None,
    baseline_path: Path | None,
) -> dict[str, Any]:
    resolver = FieldResolver()
    field_inventory = build_field_inventory(payload, records, resolver)
    field_presence = build_field_presence_report(records, resolver)
    schema_blocker = ocr_measurement_blocker(field_presence)

    ocr_rows, suspicious_rows, false_negative_rows = build_ocr_candidate_rows(records, resolver)
    readiness_rows, hard_blocker_rows = build_readiness_rows(records, resolver, artifact_root=artifact_root)
    subpart_rows = build_subpart_rows(records, resolver)
    mapping_validation_rows = build_mapping_validation_rows(records, resolver)
    crop_quality_rows = build_crop_quality_rows(records, resolver, readiness_rows, false_negative_rows)
    baseline_summary: dict[str, Any] | None = None
    baseline_rows: list[dict[str, Any]] | None = None
    if baseline_records is not None:
        baseline_summary, baseline_rows = build_baseline_comparison(
            current_records=records,
            baseline_records=baseline_records,
            resolver=resolver,
            artifact_root=artifact_root,
            baseline_payload=baseline_payload or {},
        )

    sample_rows = build_representative_sample(
        records=records,
        resolver=resolver,
        suspicious_rows=suspicious_rows,
        false_negative_rows=false_negative_rows,
        readiness_rows=readiness_rows,
        mapping_rows=mapping_validation_rows,
        subpart_rows=subpart_rows,
        baseline_by_id=_records_by_id(baseline_records or []),
        baseline_rows=baseline_rows,
    )

    summary = build_summary(
        payload=payload,
        records=records,
        resolver=resolver,
        input_path=input_path,
        out_dir=out_dir,
        artifact_root=artifact_root,
        baseline_path=baseline_path,
        field_inventory=field_inventory,
        field_presence=field_presence,
        schema_blocker=schema_blocker,
        ocr_rows=ocr_rows,
        suspicious_rows=suspicious_rows,
        false_negative_rows=false_negative_rows,
        readiness_rows=readiness_rows,
        hard_blocker_rows=hard_blocker_rows,
        subpart_rows=subpart_rows,
        mapping_validation_rows=mapping_validation_rows,
        crop_quality_rows=crop_quality_rows,
        baseline_summary=baseline_summary,
        baseline_records=baseline_records,
    )

    return {
        "summary": summary,
        "field_presence": field_presence,
        "ocr_rows": ocr_rows,
        "ocr_suspicious_rows": suspicious_rows,
        "ocr_false_negative_rows": false_negative_rows,
        "readiness_rows": readiness_rows,
        "hard_blocker_rows": hard_blocker_rows,
        "crop_quality_rows": crop_quality_rows,
        "mapping_validation_rows": mapping_validation_rows,
        "subpart_rows": subpart_rows,
        "sample_rows": sample_rows,
        "baseline_summary": baseline_summary,
        "baseline_rows": baseline_rows,
    }


def build_field_inventory(payload: dict[str, Any], records: list[dict[str, Any]], resolver: FieldResolver) -> dict[str, Any]:
    top_counter: Counter[str] = Counter()
    notes_counter: Counter[str] = Counter()
    duplicate_counter: Counter[str] = Counter()
    duplicate_disagreement_counter: Counter[str] = Counter()

    for record in records:
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        for key in record:
            if key != "notes":
                top_counter[key] += 1
        for key in notes:
            notes_counter[key] += 1
        for key in set(record) & set(notes):
            if key == "notes":
                continue
            duplicate_counter[key] += 1
            if not _values_equal(record.get(key), notes.get(key)):
                duplicate_disagreement_counter[key] += 1

    ids = [str(record.get("question_id") or "") for record in records]
    duplicate_ids = sum(count - 1 for count in Counter(item for item in ids if item).values() if count > 1)
    missing_ids = sum(1 for item in ids if not item)

    return {
        "schema_name": payload.get("schema_name"),
        "schema_version": payload.get("schema_version"),
        "declared_record_count": payload.get("record_count"),
        "actual_record_count": len(records),
        "duplicate_question_id_count": duplicate_ids,
        "missing_question_id_count": missing_ids,
        "counts_by_paper": count_values(records, resolver, "paper"),
        "counts_by_paper_family": count_values(records, resolver, "paper_family"),
        "counts_by_question_format_profile": count_values(records, resolver, "question_format_profile"),
        "counts_by_notes_question_format_profile": count_values(records, resolver, "question_format_profile", force_notes=True),
        "counts_by_year_season": count_year_season(records, resolver),
        "top_level_fields": dict(sorted(top_counter.items())),
        "notes_fields": dict(sorted(notes_counter.items())),
        "duplicated_fields": dict(sorted(duplicate_counter.items())),
        "duplicated_field_disagreements": dict(sorted(duplicate_disagreement_counter.items())),
    }


def build_field_presence_report(records: list[dict[str, Any]], resolver: FieldResolver) -> list[dict[str, Any]]:
    all_fields = sorted(set(EXPECTED_FIELDS) | set(TOP_LEVEL_PREFERRED_FIELDS) | set(NOTES_PREFERRED_FIELDS))
    rows: list[dict[str, Any]] = []
    total = len(records)
    for field in all_fields:
        top_present = 0
        notes_present = 0
        resolved_present = 0
        blank_count = 0
        values: list[Any] = []
        disagreements = 0
        for record in records:
            notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
            top_has = field in record
            notes_has = field in notes
            if top_has:
                top_present += 1
            if notes_has:
                notes_present += 1
            if top_has and notes_has and not _values_equal(record.get(field), notes.get(field)):
                disagreements += 1
            value = resolver.get(record, field)
            if _is_blank(value):
                blank_count += 1
            else:
                resolved_present += 1
                values.append(value)
        rows.append(
            {
                "field": field,
                "preferred_source": resolver.preferred_source(field),
                "exists_top_level": top_present > 0,
                "exists_notes": notes_present > 0,
                "top_level_key_count": top_present,
                "notes_key_count": notes_present,
                "present_count": resolved_present,
                "missing_count": total - resolved_present,
                "null_blank_count": blank_count,
                "duplicate_disagreement_count": disagreements,
                "summary": summarize_values(values, total),
                "missingness_blocks_measurement": missingness_blocks_measurement(field, resolved_present, total),
            }
        )
    return rows


def ocr_measurement_blocker(field_presence: list[dict[str, Any]]) -> str:
    presence_by_field = {row["field"]: int(row["present_count"]) for row in field_presence}
    missing = sorted(field for field in OCR_MEASUREMENT_FIELDS if presence_by_field.get(field, 0) == 0)
    if len(missing) == len(OCR_MEASUREMENT_FIELDS):
        return "no_usable_ocr_candidate_fields"
    if missing:
        return "partial_ocr_candidate_fields_missing:" + ",".join(missing)
    return ""


def build_ocr_candidate_rows(
    records: list[dict[str, Any]],
    resolver: FieldResolver,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    suspicious_rows: list[dict[str, Any]] = []
    false_negative_rows: list[dict[str, Any]] = []
    for record in sorted(records, key=record_sort_key):
        qid = str(resolver.get(record, "question_id") or "")
        qn = str(resolver.get(record, "question_number") or "")
        selected_text = str(resolver.get(record, "question_text") or "")
        ocr_text = str(resolver.get(record, "ocr_text") or "")
        native_text = native_text_candidate(record, resolver)
        expected_subparts = expected_subpart_labels(record, resolver)
        selected_metrics = text_metrics(selected_text, qn, expected_subparts)
        ocr_metrics = text_metrics(ocr_text, qn, expected_subparts)
        native_metrics = text_metrics(native_text, qn, expected_subparts)
        ocr_selected = as_bool(resolver.get(record, "ocr_selected"))
        margin = score_margin(record, resolver)
        suspicious_flags = ocr_selection_flags(
            record=record,
            resolver=resolver,
            selected_metrics=selected_metrics,
            ocr_metrics=ocr_metrics,
            native_metrics=native_metrics,
            margin=margin,
        )
        false_negative_flags = ocr_false_negative_flags(
            record=record,
            resolver=resolver,
            selected_metrics=selected_metrics,
            ocr_metrics=ocr_metrics,
            native_metrics=native_metrics,
            margin=margin,
        )
        row = {
            "question_id": qid,
            "paper": resolver.get(record, "paper"),
            "paper_family": resolver.get(record, "paper_family"),
            "question_format_profile": resolver.get(record, "question_format_profile"),
            "year_season": year_season_label(resolver.get(record, "paper")),
            "question_number": qn,
            "ocr_ran": resolver.get(record, "ocr_ran"),
            "ocr_engine": resolver.get(record, "ocr_engine"),
            "ocr_selected": ocr_selected,
            "text_candidate_source": resolver.get(record, "text_candidate_source"),
            "text_candidate_decision": resolver.get(record, "text_candidate_decision"),
            "text_candidate_decision_reasons": list_field(record, resolver, "text_candidate_decision_reasons"),
            "ocr_rejected_reasons": list_field(record, resolver, "ocr_rejected_reasons"),
            "native_text_score": resolver.get(record, "native_text_score"),
            "ocr_text_score": resolver.get(record, "ocr_text_score"),
            "selected_text_score": resolver.get(record, "selected_text_score"),
            "ocr_minus_native_margin": margin,
            "selected_minus_native_margin": selected_minus_native_margin(record, resolver),
            "selected_minus_ocr_margin": selected_minus_ocr_margin(record, resolver),
            "ocr_text_trust": resolver.get(record, "ocr_text_trust"),
            "question_text_trust": resolver.get(record, "question_text_trust"),
            "question_text_role": resolver.get(record, "question_text_role"),
            "ocr_text_role": resolver.get(record, "ocr_text_role"),
            "text_fidelity_status": resolver.get(record, "text_fidelity_status"),
            "text_only_status": resolver.get(record, "text_only_status"),
            "visual_curation_status": resolver.get(record, "visual_curation_status"),
            "mapping_status": resolver.get(record, "mapping_status"),
            "validation_status": resolver.get(record, "validation_status"),
            "scope_quality_status": resolver.get(record, "scope_quality_status"),
            "question_crop_confidence": resolver.get(record, "question_crop_confidence"),
            "mark_scheme_crop_confidence": resolver.get(record, "mark_scheme_crop_confidence"),
            "expected_question_number_present_selected": selected_metrics["question_number_present"],
            "expected_question_number_present_ocr": ocr_metrics["question_number_present"],
            "expected_question_number_present_native": native_metrics["question_number_present"],
            "selected_mark_bracket_count": selected_metrics["mark_bracket_count"],
            "ocr_mark_bracket_count": ocr_metrics["mark_bracket_count"],
            "native_mark_bracket_count": native_metrics["mark_bracket_count"],
            "selected_terminal_mark_total_present": selected_metrics["terminal_mark_total_present"],
            "ocr_terminal_mark_total_present": ocr_metrics["terminal_mark_total_present"],
            "native_terminal_mark_total_present": native_metrics["terminal_mark_total_present"],
            "selected_alpha_subpart_count": selected_metrics["alpha_subpart_count"],
            "ocr_alpha_subpart_count": ocr_metrics["alpha_subpart_count"],
            "native_alpha_subpart_count": native_metrics["alpha_subpart_count"],
            "selected_roman_subpart_count": selected_metrics["roman_subpart_count"],
            "ocr_roman_subpart_count": ocr_metrics["roman_subpart_count"],
            "native_roman_subpart_count": native_metrics["roman_subpart_count"],
            "selected_length": selected_metrics["length"],
            "ocr_length": ocr_metrics["length"],
            "native_length": native_metrics["length"],
            "selected_normalized_length": selected_metrics["normalized_length"],
            "ocr_normalized_length": ocr_metrics["normalized_length"],
            "native_normalized_length": native_metrics["normalized_length"],
            "selected_math_token_count": selected_metrics["math_token_count"],
            "ocr_math_token_count": ocr_metrics["math_token_count"],
            "native_math_token_count": native_metrics["math_token_count"],
            "selected_function_token_count": selected_metrics["function_token_count"],
            "ocr_function_token_count": ocr_metrics["function_token_count"],
            "native_function_token_count": native_metrics["function_token_count"],
            "selected_page_furniture_token_count": selected_metrics["page_furniture_token_count"],
            "ocr_page_furniture_token_count": ocr_metrics["page_furniture_token_count"],
            "native_page_furniture_token_count": native_metrics["page_furniture_token_count"],
            "audit_flags": suspicious_flags if ocr_selected else false_negative_flags,
            "suggested_human_judgment": suggested_human_judgment(ocr_selected, suspicious_flags, false_negative_flags, record, resolver),
            "question_image_path": resolver.get(record, "question_image_path"),
            "mark_scheme_image_path": resolver.get(record, "mark_scheme_image_path"),
        }
        rows.append(row)
        if ocr_selected and suspicious_flags:
            suspicious_rows.append({**row, "audit_flags": suspicious_flags})
        if not ocr_selected and false_negative_flags:
            false_negative_rows.append({**row, "audit_flags": false_negative_flags})
    return rows, suspicious_rows, false_negative_rows


def build_readiness_rows(
    records: list[dict[str, Any]],
    resolver: FieldResolver,
    *,
    artifact_root: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for record in sorted(records, key=record_sort_key):
        qid = str(resolver.get(record, "question_id") or "")
        tier_info = classify_readiness(record, resolver, artifact_root=artifact_root)
        row = {
            "question_id": qid,
            "paper": resolver.get(record, "paper"),
            "paper_family": resolver.get(record, "paper_family"),
            "question_format_profile": resolver.get(record, "question_format_profile"),
            "year_season": year_season_label(resolver.get(record, "paper")),
            "question_number": resolver.get(record, "question_number"),
            "highest_tier": tier_info["tier"],
            "tier_label": tier_info["label"],
            "tier_0_hard_blocker": tier_info["tier"] == 0,
            "tier_2_multimodal_candidate": tier_info["qualifies_tier_2"],
            "tier_3_visual_ready": tier_info["qualifies_tier_3"],
            "tier_4_text_ready": tier_info["qualifies_tier_4"],
            "tier_5_gold_pilot": tier_info["qualifies_tier_5"],
            "tier_reasons": tier_info["reasons"],
            "hard_blockers": tier_info["hard_blockers"],
            "review_blockers": tier_info["review_blockers"],
            "mapping_status": resolver.get(record, "mapping_status"),
            "validation_status": resolver.get(record, "validation_status"),
            "scope_quality_status": resolver.get(record, "scope_quality_status"),
            "visual_curation_status": resolver.get(record, "visual_curation_status"),
            "text_only_status": resolver.get(record, "text_only_status"),
            "question_text_trust": resolver.get(record, "question_text_trust"),
            "text_fidelity_status": resolver.get(record, "text_fidelity_status"),
            "visual_required": resolver.get(record, "visual_required"),
            "question_crop_confidence": resolver.get(record, "question_crop_confidence"),
            "mark_scheme_crop_confidence": resolver.get(record, "mark_scheme_crop_confidence"),
            "question_solution_marks": resolver.get(record, "question_solution_marks"),
            "question_total_detected": resolver.get(record, "question_total_detected"),
            "mark_scheme_total_detected": resolver.get(record, "mark_scheme_total_detected"),
            "paper_total_status": resolver.get(record, "paper_total_status"),
            "question_image_path": resolver.get(record, "question_image_path"),
            "mark_scheme_image_path": resolver.get(record, "mark_scheme_image_path"),
        }
        rows.append(row)
        if tier_info["hard_blockers"]:
            blockers.append(
                {
                    "question_id": qid,
                    "paper": resolver.get(record, "paper"),
                    "paper_family": resolver.get(record, "paper_family"),
                    "question_format_profile": resolver.get(record, "question_format_profile"),
                    "question_number": resolver.get(record, "question_number"),
                    "hard_blockers": tier_info["hard_blockers"],
                    "mapping_status": resolver.get(record, "mapping_status"),
                    "validation_status": resolver.get(record, "validation_status"),
                    "scope_quality_status": resolver.get(record, "scope_quality_status"),
                    "question_crop_confidence": resolver.get(record, "question_crop_confidence"),
                    "mark_scheme_crop_confidence": resolver.get(record, "mark_scheme_crop_confidence"),
                    "question_image_path": resolver.get(record, "question_image_path"),
                    "mark_scheme_image_path": resolver.get(record, "mark_scheme_image_path"),
                }
            )
    return rows, blockers


def classify_readiness(record: dict[str, Any], resolver: FieldResolver, *, artifact_root: Path | None) -> dict[str, Any]:
    hard: list[str] = []
    review: list[str] = []
    qid = resolver.get(record, "question_id")
    question_image = resolver.get(record, "question_image_path")
    mark_image = resolver.get(record, "mark_scheme_image_path")
    mark_text = resolver.get(record, "mark_scheme_text")
    mapping_status = str(resolver.get(record, "mapping_status") or "").lower()
    validation_status = str(resolver.get(record, "validation_status") or "").lower()
    scope_status = str(resolver.get(record, "scope_quality_status") or "").lower()
    question_total = numeric(resolver.get(record, "question_total_detected"))
    mark_total = numeric(resolver.get(record, "mark_scheme_total_detected"))
    solution_marks = numeric(resolver.get(record, "question_solution_marks"))
    question_structure = dict_field(record, resolver, "question_structure_detected")
    validation_flags = set(list_field(record, resolver, "validation_flags"))
    review_flags = set(list_field(record, resolver, "review_flags"))

    if _is_blank(qid):
        hard.append("missing_question_id")
    if _is_blank(question_image):
        hard.append("missing_question_image_path")
    if _is_blank(mark_image):
        hard.append("missing_mark_scheme_image_path")
    if _is_blank(mark_text):
        hard.append("missing_mark_scheme_text")
    if mapping_status == "fail":
        hard.append("mapping_status_fail")
    if validation_status == "fail":
        hard.append("validation_status_fail")
    if question_total is not None and mark_total is not None and question_total != mark_total:
        hard.append("question_total_mark_scheme_total_disagree")
    if solution_marks is None and question_total is None and mark_total is None:
        hard.append("question_has_no_usable_marks")
    if solution_marks is not None and solution_marks <= 0:
        hard.append("question_has_no_usable_marks")
    if scope_status == "fail":
        hard.append("scope_quality_status_fail")
    if question_structure.get("contamination_detected"):
        hard.append("scope_contamination_detected")
    if question_structure.get("likely_truncated"):
        hard.append("likely_truncated_question_crop")
    if validation_flags & {"question_scope_contaminated", "likely_truncated_question_crop"}:
        hard.append("hard_scope_validation_flag")
    if artifact_root:
        for field, value in [("question_image_path", question_image), ("mark_scheme_image_path", mark_image)]:
            if not _is_blank(value) and not artifact_exists(value, artifact_root):
                hard.append(f"artifact_missing:{field}")

    if mapping_status == "review":
        review.append("mapping_status_review")
    if validation_status == "review":
        review.append("validation_status_review")
    if str(resolver.get(record, "visual_curation_status") or "").lower() in {"review", "fail"}:
        review.append("visual_curation_not_ready")
    if str(resolver.get(record, "text_only_status") or "").lower() == "fail":
        review.append("text_only_fail")
    if str(resolver.get(record, "question_crop_confidence") or "").lower() == "low":
        review.append("low_question_crop_confidence")
    if str(resolver.get(record, "mark_scheme_crop_confidence") or "").lower() == "low":
        review.append("low_mark_scheme_crop_confidence")
    if as_bool(resolver.get(record, "ocr_selected")):
        expected = expected_subpart_labels(record, resolver)
        metrics = text_metrics(str(resolver.get(record, "question_text") or ""), str(resolver.get(record, "question_number") or ""), expected)
        flags = ocr_selection_flags(
            record=record,
            resolver=resolver,
            selected_metrics=metrics,
            ocr_metrics=text_metrics(str(resolver.get(record, "ocr_text") or ""), str(resolver.get(record, "question_number") or ""), expected),
            native_metrics=text_metrics(native_text_candidate(record, resolver), str(resolver.get(record, "question_number") or ""), expected),
            margin=score_margin(record, resolver),
        )
        if flags:
            review.append("suspicious_ocr_selection")
    subparts = list_field(record, resolver, "subparts")
    subpart_marks = resolver.get(record, "subparts_solution_marks")
    if subparts and subpart_marks_all_null(subpart_marks):
        review.append("subpart_marks_missing")
    topic_conf = str(resolver.get(record, "topic_confidence") or "").lower()
    if topic_conf == "low" and str(resolver.get(record, "text_fidelity_status") or "").lower() != "clean":
        review.append("topic_confidence_low_with_degraded_text")
    if review_flags & {"crop_uncertain", "low_confidence_question_crop", "weak_question_anchor"}:
        review.append("crop_or_anchor_review_flag")

    tier2 = (
        not hard
        and mapping_status == "pass"
        and validation_status == "pass"
        and not _is_blank(question_image)
        and not _is_blank(mark_image)
        and not _is_blank(mark_text)
        and totals_agree_or_missing(record, resolver)
    )
    tier3 = (
        tier2
        and str(resolver.get(record, "visual_curation_status") or "").lower() == "ready"
        and str(resolver.get(record, "question_crop_confidence") or "").lower() == "high"
        and str(resolver.get(record, "mark_scheme_crop_confidence") or "").lower() == "high"
        and scope_status == "clean"
    )
    selected_metrics = text_metrics(
        str(resolver.get(record, "question_text") or ""),
        str(resolver.get(record, "question_number") or ""),
        expected_subpart_labels(record, resolver),
    )
    tier4 = (
        tier2
        and str(resolver.get(record, "text_only_status") or "").lower() == "ready"
        and str(resolver.get(record, "question_text_trust") or "").lower() == "high"
        and str(resolver.get(record, "text_fidelity_status") or "").lower() == "clean"
        and selected_metrics["question_number_present"]
        and selected_metrics["expected_subparts_present"]
        and selected_metrics["mark_bracket_count"] > 0
        and as_bool(resolver.get(record, "visual_required")) is False
    )
    tier5 = (tier3 and tier4) or (
        tier2
        and tier3
        and as_bool(resolver.get(record, "visual_required")) is True
        and str(resolver.get(record, "text_fidelity_status") or "").lower() == "clean"
        and not contradiction_flags(record, resolver)
    )

    if hard:
        tier = 0
        label = "Tier 0 - Hard blocker"
    elif tier5:
        tier = 5
        label = "Tier 5 - Gold pilot set"
    elif tier4:
        tier = 4
        label = "Tier 4 - Asterion text-ready"
    elif tier3:
        tier = 3
        label = "Tier 3 - Asterion visual-ready"
    elif tier2:
        tier = 2
        label = "Tier 2 - Asterion multimodal candidate"
    else:
        tier = 1
        label = "Tier 1 - Master/review only"

    reasons = hard or review or ["tier_conditions_met"]
    return {
        "tier": tier,
        "label": label,
        "hard_blockers": sorted(set(hard)),
        "review_blockers": sorted(set(review)),
        "reasons": sorted(set(reasons)),
        "qualifies_tier_2": tier2,
        "qualifies_tier_3": tier3,
        "qualifies_tier_4": tier4,
        "qualifies_tier_5": tier5,
    }


def build_subpart_rows(records: list[dict[str, Any]], resolver: FieldResolver) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=record_sort_key):
        structure = dict_field(record, resolver, "question_structure_detected")
        subparts = list_field(record, resolver, "subparts")
        subpart_marks = resolver.get(record, "subparts_solution_marks")
        mark_values = structure.get("mark_values_detected")
        if not isinstance(mark_values, list):
            mark_values = []
        mark_values_numeric = [int(value) for value in mark_values if isinstance(value, int) and not isinstance(value, bool)]
        question_total = numeric(resolver.get(record, "question_total_detected"))
        solution_marks = numeric(resolver.get(record, "question_solution_marks"))
        target_total = question_total if question_total is not None else solution_marks
        likely_nested = has_nested_subparts(subparts, structure, mark_values_numeric)
        fillable_simple = bool(
            subparts
            and subpart_marks_all_null(subpart_marks)
            and mark_values_numeric
            and len(mark_values_numeric) == len(subparts)
            and target_total is not None
            and sum(mark_values_numeric) == int(target_total)
            and not likely_nested
        )
        reason = "fillable_simple" if fillable_simple else "not_fillable"
        if not subparts:
            reason = "no_subparts"
        elif not subpart_marks_all_null(subpart_marks):
            reason = "subpart_marks_already_populated"
        elif not mark_values_numeric:
            reason = "no_detected_mark_values"
        elif target_total is None:
            reason = "missing_total"
        elif sum(mark_values_numeric) != int(target_total):
            reason = "detected_marks_do_not_sum_to_total"
        elif len(mark_values_numeric) != len(subparts):
            reason = "detected_marks_count_does_not_equal_subpart_count"
        elif likely_nested:
            reason = "likely_nested_subparts"

        rows.append(
            {
                "question_id": resolver.get(record, "question_id"),
                "paper": resolver.get(record, "paper"),
                "paper_family": resolver.get(record, "paper_family"),
                "question_format_profile": resolver.get(record, "question_format_profile"),
                "question_number": resolver.get(record, "question_number"),
                "subparts": subparts,
                "subparts_solution_marks": subpart_marks,
                "mark_values_detected": mark_values_numeric,
                "question_solution_marks": resolver.get(record, "question_solution_marks"),
                "question_total_detected": resolver.get(record, "question_total_detected"),
                "mark_scheme_total_detected": resolver.get(record, "mark_scheme_total_detected"),
                "paper_total_status": resolver.get(record, "paper_total_status"),
                "fillable_simple": fillable_simple,
                "likely_nested": likely_nested,
                "reason": reason,
            }
        )
    return rows


def build_mapping_validation_rows(records: list[dict[str, Any]], resolver: FieldResolver) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=record_sort_key):
        flags = contradiction_flags(record, resolver)
        rows.append(
            {
                "question_id": resolver.get(record, "question_id"),
                "paper": resolver.get(record, "paper"),
                "paper_family": resolver.get(record, "paper_family"),
                "question_format_profile": resolver.get(record, "question_format_profile"),
                "question_number": resolver.get(record, "question_number"),
                "mapping_status": resolver.get(record, "mapping_status"),
                "mapping_failure_reason": resolver.get(record, "mapping_failure_reason"),
                "validation_status": resolver.get(record, "validation_status"),
                "validation_flags": list_field(record, resolver, "validation_flags"),
                "scope_quality_status": resolver.get(record, "scope_quality_status"),
                "mark_scheme_text_missing": _is_blank(resolver.get(record, "mark_scheme_text")),
                "mark_scheme_image_path_missing": _is_blank(resolver.get(record, "mark_scheme_image_path")),
                "mark_scheme_crop_confidence": resolver.get(record, "mark_scheme_crop_confidence"),
                "question_total_detected": resolver.get(record, "question_total_detected"),
                "mark_scheme_total_detected": resolver.get(record, "mark_scheme_total_detected"),
                "question_solution_marks": resolver.get(record, "question_solution_marks"),
                "paper_total_status": resolver.get(record, "paper_total_status"),
                "contradiction_flags": flags,
            }
        )
    return rows


def build_crop_quality_rows(
    records: list[dict[str, Any]],
    resolver: FieldResolver,
    readiness_rows: list[dict[str, Any]],
    false_negative_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_qid = {str(row["question_id"]): row for row in readiness_rows}
    false_negative_ids = {str(row["question_id"]) for row in false_negative_rows}
    group_specs = [
        ("question_format_profile", lambda record: str(resolver.get(record, "question_format_profile") or "missing")),
        ("year_season", lambda record: year_season_label(resolver.get(record, "paper"))),
        ("paper_family", lambda record: str(resolver.get(record, "paper_family") or "missing")),
    ]
    rows: list[dict[str, Any]] = []
    for group_type, group_func in group_specs:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[group_func(record)].append(record)
        for group_value in sorted(grouped):
            items = grouped[group_value]
            qids = [str(resolver.get(record, "question_id") or "") for record in items]
            selected_count = sum(1 for record in items if as_bool(resolver.get(record, "ocr_selected")))
            flag_counts = Counter(
                flag
                for record in items
                for flag in list_field(record, resolver, "review_flags") + list_field(record, resolver, "validation_flags")
                if flag in REVIEW_FLAG_NAMES
            )
            tier_counts = Counter(str(by_qid.get(qid, {}).get("highest_tier", "missing")) for qid in qids)
            hard_count = sum(1 for qid in qids if int(by_qid.get(qid, {}).get("highest_tier", 99)) == 0)
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "record_count": len(items),
                    "validation_status_distribution": count_values(items, resolver, "validation_status"),
                    "mapping_status_distribution": count_values(items, resolver, "mapping_status"),
                    "question_crop_confidence_distribution": count_values(items, resolver, "question_crop_confidence"),
                    "mark_scheme_crop_confidence_distribution": count_values(items, resolver, "mark_scheme_crop_confidence"),
                    "visual_curation_status_distribution": count_values(items, resolver, "visual_curation_status"),
                    "text_only_status_distribution": count_values(items, resolver, "text_only_status"),
                    "question_text_trust_distribution": count_values(items, resolver, "question_text_trust"),
                    "ocr_selected_count": selected_count,
                    "ocr_selected_rate": rate(selected_count, len(items)),
                    "ocr_false_negative_candidate_count": sum(1 for qid in qids if qid in false_negative_ids),
                    "ocr_false_negative_candidate_rate": rate(sum(1 for qid in qids if qid in false_negative_ids), len(items)),
                    "hard_blocker_count": hard_count,
                    "asterion_tier_distribution": dict(sorted(tier_counts.items())),
                    "review_flag_counts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
                }
            )
    return rows


def build_baseline_comparison(
    *,
    current_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    resolver: FieldResolver,
    artifact_root: Path | None,
    baseline_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current_by_id = _records_by_id(current_records)
    baseline_by_id = _records_by_id(baseline_records)
    current_duplicates = duplicate_ids(current_records)
    baseline_duplicates = duplicate_ids(baseline_records)
    shared_ids = sorted(set(current_by_id) & set(baseline_by_id))
    added_ids = sorted(set(current_by_id) - set(baseline_by_id))
    removed_ids = sorted(set(baseline_by_id) - set(current_by_id))
    rows: list[dict[str, Any]] = []
    field_change_counts: Counter[str] = Counter()
    improved_counts: Counter[str] = Counter()
    worsened_counts: Counter[str] = Counter()
    status_unknown_counts: Counter[str] = Counter()
    tier_change_counts: Counter[str] = Counter()
    current_resolver = resolver
    baseline_resolver = FieldResolver()

    for question_id in shared_ids:
        current = current_by_id[question_id]
        baseline = baseline_by_id[question_id]
        changed_fields = []
        old_new: dict[str, Any] = {}
        for field in COMPARE_FIELDS:
            current_value = current_resolver.get(current, field)
            baseline_value = baseline_resolver.get(baseline, field)
            if normalize_compare_value(current_value) != normalize_compare_value(baseline_value):
                changed_fields.append(field)
                field_change_counts[field] += 1
                old_new[f"old_{field}"] = baseline_value
                old_new[f"new_{field}"] = current_value
        movement = status_movement(current, baseline, current_resolver, baseline_resolver)
        for field in movement["improved"]:
            improved_counts[field] += 1
        for field in movement["worsened"]:
            worsened_counts[field] += 1
        for field in movement["unknown"]:
            status_unknown_counts[field] += 1
        current_tier = classify_readiness(current, current_resolver, artifact_root=artifact_root)["tier"]
        baseline_tier = classify_readiness(baseline, baseline_resolver, artifact_root=None)["tier"]
        tier_delta = current_tier - baseline_tier
        tier_key = "unchanged"
        if tier_delta > 0:
            tier_key = "improved"
        elif tier_delta < 0:
            tier_key = "worsened"
        tier_change_counts[tier_key] += 1
        current_metrics = text_metrics(
            str(current_resolver.get(current, "question_text") or ""),
            str(current_resolver.get(current, "question_number") or ""),
            expected_subpart_labels(current, current_resolver),
        )
        baseline_metrics = text_metrics(
            str(baseline_resolver.get(baseline, "question_text") or ""),
            str(baseline_resolver.get(baseline, "question_number") or ""),
            expected_subpart_labels(baseline, baseline_resolver),
        )
        text_regression_flags = text_regression_flags_from_metrics(current_metrics, baseline_metrics)
        text_improvement_flags = text_improvement_flags_from_metrics(current_metrics, baseline_metrics)
        status_inflation_flags = []
        if movement["improved"] and not text_improvement_flags and text_regression_flags:
            status_inflation_flags.append("status_improved_with_text_regression")
        elif movement["improved"] and not text_improvement_flags and changed_fields:
            status_inflation_flags.append("status_improved_without_text_quality_evidence")

        rows.append(
            {
                "question_id": question_id,
                "old_tier": baseline_tier,
                "new_tier": current_tier,
                "tier_delta": tier_delta,
                "tier_movement": tier_key,
                "changed_fields": changed_fields,
                "improved_status_fields": movement["improved"],
                "worsened_status_fields": movement["worsened"],
                "unknown_status_fields": movement["unknown"],
                "exact_question_text_changed": current_resolver.get(current, "question_text") != baseline_resolver.get(baseline, "question_text"),
                "normalized_question_text_changed": normalize_text(current_resolver.get(current, "question_text"))
                != normalize_text(baseline_resolver.get(baseline, "question_text")),
                "old_question_text": baseline_resolver.get(baseline, "question_text"),
                "new_question_text": current_resolver.get(current, "question_text"),
                "old_text_candidate_source": baseline_resolver.get(baseline, "text_candidate_source"),
                "new_text_candidate_source": current_resolver.get(current, "text_candidate_source"),
                "old_ocr_selected": baseline_resolver.get(baseline, "ocr_selected"),
                "new_ocr_selected": current_resolver.get(current, "ocr_selected"),
                "old_native_text_score": baseline_resolver.get(baseline, "native_text_score"),
                "new_native_text_score": current_resolver.get(current, "native_text_score"),
                "old_ocr_text_score": baseline_resolver.get(baseline, "ocr_text_score"),
                "new_ocr_text_score": current_resolver.get(current, "ocr_text_score"),
                "old_selected_text_score": baseline_resolver.get(baseline, "selected_text_score"),
                "new_selected_text_score": current_resolver.get(current, "selected_text_score"),
                "old_margin": score_margin(baseline, baseline_resolver),
                "new_margin": score_margin(current, current_resolver),
                "text_regression_flags": text_regression_flags,
                "text_improvement_flags": text_improvement_flags,
                "status_inflation_flags": status_inflation_flags,
                **old_new,
            }
        )

    for question_id in added_ids:
        rows.append({"question_id": question_id, "record_status": "added"})
    for question_id in removed_ids:
        rows.append({"question_id": question_id, "record_status": "removed"})

    summary = {
        "available": True,
        "reliable": not current_duplicates and not baseline_duplicates,
        "unreliable_reason": "duplicate_question_ids" if current_duplicates or baseline_duplicates else "",
        "baseline_schema_name": baseline_payload.get("schema_name"),
        "baseline_schema_version": baseline_payload.get("schema_version"),
        "baseline_declared_record_count": baseline_payload.get("record_count"),
        "baseline_record_count": len(baseline_records),
        "current_record_count": len(current_records),
        "records_added": len(added_ids),
        "records_removed": len(removed_ids),
        "records_present_in_both": len(shared_ids),
        "current_duplicate_question_ids": current_duplicates,
        "baseline_duplicate_question_ids": baseline_duplicates,
        "field_change_counts": dict(sorted(field_change_counts.items())),
        "improved_status_counts": dict(sorted(improved_counts.items())),
        "worsened_status_counts": dict(sorted(worsened_counts.items())),
        "unknown_status_movement_counts": dict(sorted(status_unknown_counts.items())),
        "asterion_tier_change_counts": dict(sorted(tier_change_counts.items())),
        "ordering_used": {
            "STATUS_ORDER": STATUS_ORDER,
            "TRUST_ORDER": TRUST_ORDER,
            "FIDELITY_ORDER": FIDELITY_ORDER,
            "CONFIDENCE_ORDER": CONFIDENCE_ORDER,
        },
    }
    return summary, sorted(rows, key=lambda row: str(row.get("question_id") or ""))


def build_representative_sample(
    *,
    records: list[dict[str, Any]],
    resolver: FieldResolver,
    suspicious_rows: list[dict[str, Any]],
    false_negative_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    subpart_rows: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    baseline_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows_by_id = {str(resolver.get(record, "question_id") or ""): record for record in records}
    baseline_change_ids = {
        str(row.get("question_id"))
        for row in (baseline_rows or [])
        if row.get("changed_fields") or row.get("improved_status_fields") or row.get("worsened_status_fields")
    }
    mapping_contradiction_ids = {str(row.get("question_id")) for row in mapping_rows if row.get("contradiction_flags")}
    low_crop_ids = {
        str(row.get("question_id"))
        for row in readiness_rows
        if str(row.get("question_crop_confidence") or "").lower() == "low"
    }
    fillable_subpart_ids = {str(row.get("question_id")) for row in subpart_rows if as_bool(row.get("fillable_simple"))}
    suspicious_ids = {str(row.get("question_id")) for row in suspicious_rows}
    false_negative_ids = {str(row.get("question_id")) for row in false_negative_rows}

    buckets: list[tuple[str, list[dict[str, Any]], str]] = []
    selected = [record for record in records if as_bool(resolver.get(record, "ocr_selected"))]
    high_margin = sorted([record for record in selected if (score_margin(record, resolver) or -math.inf) >= 60], key=lambda record: (-(score_margin(record, resolver) or 0), record_sort_key(record)))
    low_margin = sorted([record for record in selected if (score_margin(record, resolver) or math.inf) < 40], key=lambda record: ((score_margin(record, resolver) or 0), record_sort_key(record)))
    native_selected = [record for record in records if not as_bool(resolver.get(record, "ocr_selected"))]
    high_ocr_score = sorted(
        [record for record in native_selected if (numeric(resolver.get(record, "ocr_text_score")) or -math.inf) >= 80],
        key=lambda record: (-(numeric(resolver.get(record, "ocr_text_score")) or 0), record_sort_key(record)),
    )
    high_margin_rejected = sorted(
        [record for record in native_selected if (score_margin(record, resolver) or -math.inf) >= 30],
        key=lambda record: (-(score_margin(record, resolver) or 0), record_sort_key(record)),
    )
    format_2024_2025 = [record for record in records if is_2024_2025_record(record, resolver)]
    native_controls = [record for record in native_selected if str(resolver.get(record, "mapping_status") or "") == "pass"]
    visual_ready = [record for record in records if str(resolver.get(record, "visual_curation_status") or "") == "ready"]
    text_ready = [record for record in records if str(resolver.get(record, "text_only_status") or "") == "ready"]

    buckets.extend(
        [
            ("high-margin OCR-selected records", high_margin, "good_selection"),
            ("low-margin OCR-selected records", low_margin, "questionable_selection"),
            ("OCR-selected records with suspicious flags", [rows_by_id[qid] for qid in sorted(suspicious_ids) if qid in rows_by_id], "bad_selection"),
            ("native-selected records where OCR was rejected despite high OCR score", high_ocr_score, "possible_false_negative"),
            ("native-selected records where OCR was rejected despite high OCR-over-native margin", high_margin_rejected, "possible_false_negative"),
            ("records with readiness/trust/status changes", [rows_by_id[qid] for qid in sorted(baseline_change_ids) if qid in rows_by_id], "not_enough_evidence"),
            ("records with mapping/validation contradictions", [rows_by_id[qid] for qid in sorted(mapping_contradiction_ids) if qid in rows_by_id], "not_enough_evidence"),
            ("records with low question crop confidence", [rows_by_id[qid] for qid in sorted(low_crop_ids) if qid in rows_by_id], "not_enough_evidence"),
            ("records from 2024-2025 format/profile", format_2024_2025, "not_enough_evidence"),
            ("records with missing/null subpart marks but detected mark values", [rows_by_id[qid] for qid in sorted(fillable_subpart_ids) if qid in rows_by_id], "not_enough_evidence"),
            ("random native-selected controls across paper families", native_controls, "not_enough_evidence"),
            ("random visual-ready controls", visual_ready, "not_enough_evidence"),
            ("random text-ready controls", text_ready, "not_enough_evidence"),
        ]
    )

    sample_rows: list[dict[str, Any]] = []
    seen_bucket_qids: set[tuple[str, str]] = set()
    for bucket_name, bucket_records, default_judgment in buckets:
        if bucket_name.startswith("random"):
            selected_records = balanced_sample_by_family(bucket_records, bucket_name, 12, resolver)
        else:
            selected_records = bucket_records[:12]
        for record in selected_records:
            qid = str(resolver.get(record, "question_id") or "")
            if not qid or (bucket_name, qid) in seen_bucket_qids:
                continue
            seen_bucket_qids.add((bucket_name, qid))
            expected = expected_subpart_labels(record, resolver)
            selected_text = str(resolver.get(record, "question_text") or "")
            ocr_text = str(resolver.get(record, "ocr_text") or "")
            native_text = native_text_candidate(record, resolver)
            flags: list[str]
            if qid in suspicious_ids:
                flags = next((list(row.get("audit_flags") or []) for row in suspicious_rows if str(row.get("question_id")) == qid), [])
            elif qid in false_negative_ids:
                flags = next((list(row.get("audit_flags") or []) for row in false_negative_rows if str(row.get("question_id")) == qid), [])
            else:
                flags = []
            baseline_record = baseline_by_id.get(qid, {})
            sample_rows.append(
                {
                    "sample_bucket": bucket_name,
                    "question_id": qid,
                    "paper": resolver.get(record, "paper"),
                    "paper_family": resolver.get(record, "paper_family"),
                    "question_format_profile": resolver.get(record, "question_format_profile"),
                    "question_number": resolver.get(record, "question_number"),
                    "old_question_text": short_text(baseline_record.get("question_text")),
                    "new_question_text": short_text(selected_text),
                    "native_pdf_text": short_text(native_text),
                    "ocr_text": short_text(ocr_text),
                    "selected_text": short_text(selected_text),
                    "native_text_score": resolver.get(record, "native_text_score"),
                    "ocr_text_score": resolver.get(record, "ocr_text_score"),
                    "selected_text_score": resolver.get(record, "selected_text_score"),
                    "score_margin": score_margin(record, resolver),
                    "decision_reasons": list_field(record, resolver, "text_candidate_decision_reasons"),
                    "rejected_reasons": list_field(record, resolver, "ocr_rejected_reasons"),
                    "text_fidelity_status": resolver.get(record, "text_fidelity_status"),
                    "text_only_status": resolver.get(record, "text_only_status"),
                    "visual_curation_status": resolver.get(record, "visual_curation_status"),
                    "question_text_trust": resolver.get(record, "question_text_trust"),
                    "question_crop_confidence": resolver.get(record, "question_crop_confidence"),
                    "mark_scheme_crop_confidence": resolver.get(record, "mark_scheme_crop_confidence"),
                    "mapping_status": resolver.get(record, "mapping_status"),
                    "validation_status": resolver.get(record, "validation_status"),
                    "question_solution_marks": resolver.get(record, "question_solution_marks"),
                    "question_total_detected": resolver.get(record, "question_total_detected"),
                    "mark_scheme_total_detected": resolver.get(record, "mark_scheme_total_detected"),
                    "subparts": list_field(record, resolver, "subparts"),
                    "subparts_solution_marks": resolver.get(record, "subparts_solution_marks"),
                    "mark_values_detected": dict_field(record, resolver, "question_structure_detected").get("mark_values_detected"),
                    "question_image_path": resolver.get(record, "question_image_path"),
                    "mark_scheme_image_path": resolver.get(record, "mark_scheme_image_path"),
                    "audit_flags": flags,
                    "suggested_human_judgment": suggested_human_judgment(
                        as_bool(resolver.get(record, "ocr_selected")),
                        flags if qid in suspicious_ids else [],
                        flags if qid in false_negative_ids else [],
                        record,
                        resolver,
                        fallback=default_judgment,
                    ),
                    "selected_structure_metrics": text_metrics(selected_text, str(resolver.get(record, "question_number") or ""), expected),
                    "ocr_structure_metrics": text_metrics(ocr_text, str(resolver.get(record, "question_number") or ""), expected),
                }
            )
    return sample_rows


def build_summary(
    *,
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    resolver: FieldResolver,
    input_path: Path,
    out_dir: Path,
    artifact_root: Path | None,
    baseline_path: Path | None,
    field_inventory: dict[str, Any],
    field_presence: list[dict[str, Any]],
    schema_blocker: str,
    ocr_rows: list[dict[str, Any]],
    suspicious_rows: list[dict[str, Any]],
    false_negative_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    hard_blocker_rows: list[dict[str, Any]],
    subpart_rows: list[dict[str, Any]],
    mapping_validation_rows: list[dict[str, Any]],
    crop_quality_rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
    baseline_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    tier_counts = Counter(str(row["highest_tier"]) for row in readiness_rows)
    tier_ids: dict[str, list[str]] = defaultdict(list)
    for row in readiness_rows:
        tier_ids[str(row["highest_tier"])].append(str(row["question_id"]))
    top_decision_reasons = Counter(reason for row in ocr_rows for reason in as_list(row.get("text_candidate_decision_reasons")))
    top_rejected_reasons = Counter(reason for row in ocr_rows for reason in as_list(row.get("ocr_rejected_reasons")))
    mark_summary = build_mark_total_summary(records, resolver, subpart_rows)
    mapping_summary = build_mapping_summary(mapping_validation_rows)
    topic_difficulty_summary = build_topic_difficulty_summary(records, resolver, baseline_records)
    run_metadata = {field: payload.get(field) for field in RUN_METADATA_FIELDS}
    run_metadata_missing = [field for field in RUN_METADATA_FIELDS if field not in payload]
    ocr_selected_count = sum(1 for row in ocr_rows if as_bool(row.get("ocr_selected")))
    ocr_ran_count = sum(1 for row in ocr_rows if as_bool(row.get("ocr_ran")))

    summary = {
        "audit_version": AUDIT_VERSION,
        "sample_seed": SAMPLE_SEED,
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "artifact_root": str(artifact_root) if artifact_root else "",
        "baseline_path": str(baseline_path) if baseline_path else "",
        "source_of_truth_policy": {
            "top_level_preferred_fields": sorted(TOP_LEVEL_PREFERRED_FIELDS),
            "notes_preferred_fields": sorted(NOTES_PREFERRED_FIELDS),
            "rule": "Prefer top-level export-contract fields; prefer notes for pipeline diagnostics. If both exist and disagree, use the preferred location and record the disagreement.",
        },
        "schema": field_inventory,
        "schema_blocker": schema_blocker,
        "run_metadata": {
            "present_values": run_metadata,
            "missing_fields": run_metadata_missing,
            "fully_auditable_from_json_alone": not run_metadata_missing,
        },
        "record_count": len(records),
        "counts_by_paper": count_values(records, resolver, "paper"),
        "counts_by_paper_family": count_values(records, resolver, "paper_family"),
        "counts_by_question_format_profile": count_values(records, resolver, "question_format_profile"),
        "counts_by_year_season": count_year_season(records, resolver),
        "ocr_candidate_measurement": {
            "ocr_ran_count": ocr_ran_count,
            "ocr_ran_distribution": count_values(records, resolver, "ocr_ran"),
            "ocr_engine_distribution": count_values(records, resolver, "ocr_engine"),
            "ocr_selected_count": ocr_selected_count,
            "ocr_selected_rate": rate(ocr_selected_count, len(records)),
            "text_candidate_source_distribution": count_values(records, resolver, "text_candidate_source"),
            "text_candidate_decision_distribution": count_values(records, resolver, "text_candidate_decision"),
            "top_text_candidate_decision_reasons": top_counter_dict(top_decision_reasons),
            "top_ocr_rejected_reasons": top_counter_dict(top_rejected_reasons),
            "native_text_score_summary": numeric_summary([row.get("native_text_score") for row in ocr_rows], len(ocr_rows)),
            "ocr_text_score_summary": numeric_summary([row.get("ocr_text_score") for row in ocr_rows], len(ocr_rows)),
            "selected_text_score_summary": numeric_summary([row.get("selected_text_score") for row in ocr_rows], len(ocr_rows)),
            "ocr_minus_native_margin_summary": numeric_summary([row.get("ocr_minus_native_margin") for row in ocr_rows], len(ocr_rows)),
            "selected_minus_native_margin_summary": numeric_summary([row.get("selected_minus_native_margin") for row in ocr_rows], len(ocr_rows)),
            "selected_minus_ocr_margin_summary": numeric_summary([row.get("selected_minus_ocr_margin") for row in ocr_rows], len(ocr_rows)),
            "ocr_text_trust_distribution": count_values(records, resolver, "ocr_text_trust"),
            "question_text_trust_distribution": count_values(records, resolver, "question_text_trust"),
            "question_text_role_distribution": count_values(records, resolver, "question_text_role"),
            "ocr_text_role_distribution": count_values(records, resolver, "ocr_text_role"),
            "text_fidelity_status_distribution": count_values(records, resolver, "text_fidelity_status"),
            "text_only_status_distribution": count_values(records, resolver, "text_only_status"),
            "visual_curation_status_distribution": count_values(records, resolver, "visual_curation_status"),
            "mapping_status_distribution": count_values(records, resolver, "mapping_status"),
            "validation_status_distribution": count_values(records, resolver, "validation_status"),
            "suspicious_ocr_selected_count": len(suspicious_rows),
            "possible_ocr_false_negative_count": len(false_negative_rows),
        },
        "readiness": {
            "highest_tier_counts": dict(sorted(tier_counts.items())),
            "tier_question_ids": {tier: sorted(ids) for tier, ids in sorted(tier_ids.items())},
            "hard_blocker_count": len(hard_blocker_rows),
            "hard_blocker_reason_counts": top_counter_dict(Counter(reason for row in hard_blocker_rows for reason in as_list(row.get("hard_blockers")))),
        },
        "mapping_validation": mapping_summary,
        "mark_totals_and_subparts": mark_summary,
        "topic_difficulty_side_effects": topic_difficulty_summary,
        "crop_quality_group_count": len(crop_quality_rows),
        "field_presence_blockers": [row for row in field_presence if row["missingness_blocks_measurement"]],
        "field_disagreement_count": len(resolver.disagreements),
        "field_disagreements_sample": sorted(resolver.disagreements.values(), key=lambda item: (item["field"], item["question_id"]))[:50],
        "baseline_comparison": baseline_summary or {"available": False, "reason": "no_baseline_argument"},
    }
    summary["next_iteration_recommendations"] = recommend_next_iteration(summary, crop_quality_rows, false_negative_rows, subpart_rows)
    return summary


def build_mark_total_summary(records: list[dict[str, Any]], resolver: FieldResolver, subpart_rows: list[dict[str, Any]]) -> dict[str, Any]:
    q_solution_missing = []
    q_total_missing = []
    ms_total_missing = []
    solution_vs_q_total = []
    q_vs_ms_total = []
    paper_not_matched = []
    paper_matched_local_disagree = []
    subparts_all_null = []
    mark_values_fillable = []
    for record in records:
        qid = resolver.get(record, "question_id")
        solution = numeric(resolver.get(record, "question_solution_marks"))
        q_total = numeric(resolver.get(record, "question_total_detected"))
        ms_total = numeric(resolver.get(record, "mark_scheme_total_detected"))
        paper_status = str(resolver.get(record, "paper_total_status") or "")
        if solution is None:
            q_solution_missing.append(qid)
        if q_total is None:
            q_total_missing.append(qid)
        if ms_total is None:
            ms_total_missing.append(qid)
        if solution is not None and q_total is not None and solution != q_total:
            solution_vs_q_total.append(qid)
        if q_total is not None and ms_total is not None and q_total != ms_total:
            q_vs_ms_total.append(qid)
        if paper_status not in {"matched", "recovered_after_rescan"}:
            paper_not_matched.append(qid)
        if paper_status in {"matched", "recovered_after_rescan"} and q_total is not None and ms_total is not None and q_total != ms_total:
            paper_matched_local_disagree.append(qid)
        if list_field(record, resolver, "subparts") and subpart_marks_all_null(resolver.get(record, "subparts_solution_marks")):
            subparts_all_null.append(qid)
        structure = dict_field(record, resolver, "question_structure_detected")
        marks = structure.get("mark_values_detected")
        if marks:
            mark_values_fillable.append(qid)
    return {
        "records_with_question_solution_marks_missing": len(q_solution_missing),
        "records_with_question_total_detected_missing": len(q_total_missing),
        "records_with_mark_scheme_total_detected_missing": len(ms_total_missing),
        "records_question_solution_marks_not_equal_question_total_detected": len(solution_vs_q_total),
        "records_question_total_not_equal_mark_scheme_total": len(q_vs_ms_total),
        "records_paper_total_status_not_matched": len(paper_not_matched),
        "records_paper_total_matched_but_local_totals_disagree": len(paper_matched_local_disagree),
        "records_with_subparts_present_but_all_subpart_marks_null": len(subparts_all_null),
        "total_subpart_entries": sum(len(as_list(row.get("subparts"))) for row in subpart_rows),
        "total_null_subpart_mark_entries": sum(null_subpart_mark_count(row.get("subparts_solution_marks")) for row in subpart_rows),
        "records_with_detected_mark_values": len(mark_values_fillable),
        "records_fillable_simple_future_sprint": sum(1 for row in subpart_rows if as_bool(row.get("fillable_simple"))),
        "records_likely_nested": sum(1 for row in subpart_rows if as_bool(row.get("likely_nested"))),
    }


def build_mapping_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mapping_status_distribution": count_row_values(rows, "mapping_status"),
        "validation_status_distribution": count_row_values(rows, "validation_status"),
        "mapping_failure_reason_counts": top_counter_dict(Counter(str(row.get("mapping_failure_reason") or "missing") for row in rows)),
        "validation_flag_counts": top_counter_dict(Counter(flag for row in rows for flag in as_list(row.get("validation_flags")))),
        "missing_mark_scheme_text_count": sum(1 for row in rows if as_bool(row.get("mark_scheme_text_missing"))),
        "missing_mark_scheme_image_path_count": sum(1 for row in rows if as_bool(row.get("mark_scheme_image_path_missing"))),
        "mark_scheme_crop_confidence_distribution": count_row_values(rows, "mark_scheme_crop_confidence"),
        "mapping_fail_but_validation_pass_count": sum(1 for row in rows if "mapping_fail_but_validation_pass" in as_list(row.get("contradiction_flags"))),
        "missing_mark_scheme_but_validation_pass_count": sum(1 for row in rows if "missing_mark_scheme_but_validation_pass" in as_list(row.get("contradiction_flags"))),
        "mark_total_mismatch_count": sum(1 for row in rows if "question_total_mark_scheme_total_disagree" in as_list(row.get("contradiction_flags"))),
        "paper_total_matched_but_local_validation_failed_count": sum(
            1 for row in rows if "paper_total_matched_but_local_validation_failed" in as_list(row.get("contradiction_flags"))
        ),
    }


def build_topic_difficulty_summary(
    records: list[dict[str, Any]],
    resolver: FieldResolver,
    baseline_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    high_topic_degraded = []
    high_topic_low_trust = []
    high_difficulty_degraded = []
    for record in records:
        qid = resolver.get(record, "question_id")
        topic_conf = str(resolver.get(record, "topic_confidence") or "").lower()
        difficulty_conf = str(resolver.get(record, "difficulty_confidence") or "").lower()
        text_fidelity = str(resolver.get(record, "text_fidelity_status") or "").lower()
        trust = str(resolver.get(record, "question_text_trust") or "").lower()
        mapping = str(resolver.get(record, "mapping_status") or "").lower()
        validation = str(resolver.get(record, "validation_status") or "").lower()
        if topic_conf == "high" and text_fidelity not in {"clean", ""}:
            high_topic_degraded.append(qid)
        if topic_conf == "high" and trust in {"low", "unusable"}:
            high_topic_low_trust.append(qid)
        if difficulty_conf == "high" and (text_fidelity not in {"clean", ""} or mapping != "pass" or validation != "pass"):
            high_difficulty_degraded.append(qid)

    baseline_topic_changes = 0
    baseline_difficulty_changes = 0
    if baseline_records is not None:
        current_by_id = _records_by_id(records)
        baseline_by_id = _records_by_id(baseline_records)
        baseline_resolver = FieldResolver()
        for qid in set(current_by_id) & set(baseline_by_id):
            if resolver.get(current_by_id[qid], "topic") != baseline_resolver.get(baseline_by_id[qid], "topic"):
                baseline_topic_changes += 1
            if resolver.get(current_by_id[qid], "difficulty") != baseline_resolver.get(baseline_by_id[qid], "difficulty"):
                baseline_difficulty_changes += 1

    return {
        "topic_distribution": count_values(records, resolver, "topic"),
        "topic_confidence_distribution": count_values(records, resolver, "topic_confidence"),
        "topic_uncertain_count": sum(1 for record in records if as_bool(resolver.get(record, "topic_uncertain"))),
        "topic_trust_status_distribution": count_values(records, resolver, "topic_trust_status"),
        "difficulty_distribution": count_values(records, resolver, "difficulty"),
        "difficulty_confidence_distribution": count_values(records, resolver, "difficulty_confidence"),
        "difficulty_model_version_distribution": count_values(records, resolver, "difficulty_model_version"),
        "high_topic_confidence_but_degraded_text_count": len(high_topic_degraded),
        "high_topic_confidence_but_low_question_text_trust_count": len(high_topic_low_trust),
        "high_difficulty_confidence_but_degraded_marks_text_mapping_count": len(high_difficulty_degraded),
        "baseline_topic_change_count": baseline_topic_changes,
        "baseline_difficulty_change_count": baseline_difficulty_changes,
    }


def render_summary_md(summary: dict[str, Any]) -> str:
    ocr = summary["ocr_candidate_measurement"]
    readiness = summary["readiness"]
    lines = [
        "# Question Bank Readiness Audit - iteration_001",
        "",
        f"- Input: `{summary['input_path']}`",
        f"- Records: `{summary['record_count']}`",
        f"- Schema: `{summary['schema'].get('schema_name')}` version `{summary['schema'].get('schema_version')}`",
        f"- Audit version: `{summary['audit_version']}`",
        f"- Sample seed: `{summary['sample_seed']}`",
        "",
        "## Field Source Policy",
        "",
        summary["source_of_truth_policy"]["rule"],
        "",
        f"- Field disagreements detected: `{summary['field_disagreement_count']}`",
        f"- OCR measurement blocker: `{summary['schema_blocker'] or 'none'}`",
        "",
        "## OCR Candidate Selection",
        "",
        f"- OCR ran count: `{ocr['ocr_ran_count']}`",
        f"- OCR selected count: `{ocr['ocr_selected_count']}` ({ocr['ocr_selected_rate']})",
        f"- Text candidate source distribution: `{json.dumps(ocr['text_candidate_source_distribution'], sort_keys=True)}`",
        f"- Text candidate decision distribution: `{json.dumps(ocr['text_candidate_decision_distribution'], sort_keys=True)}`",
        f"- Top OCR rejected reasons: `{json.dumps(ocr['top_ocr_rejected_reasons'], sort_keys=True)}`",
        f"- Suspicious OCR-selected records: `{ocr['suspicious_ocr_selected_count']}`",
        f"- Possible OCR false negatives: `{ocr['possible_ocr_false_negative_count']}`",
        "",
        "## Readiness",
        "",
        f"- Highest tier counts: `{json.dumps(readiness['highest_tier_counts'], sort_keys=True)}`",
        f"- Hard blocker count: `{readiness['hard_blocker_count']}`",
        f"- Hard blocker reasons: `{json.dumps(readiness['hard_blocker_reason_counts'], sort_keys=True)}`",
        "",
        "## Mapping And Validation",
        "",
        f"- Mapping status: `{json.dumps(summary['mapping_validation']['mapping_status_distribution'], sort_keys=True)}`",
        f"- Validation status: `{json.dumps(summary['mapping_validation']['validation_status_distribution'], sort_keys=True)}`",
        f"- Mapping fail but validation pass: `{summary['mapping_validation']['mapping_fail_but_validation_pass_count']}`",
        f"- Missing mark scheme but validation pass: `{summary['mapping_validation']['missing_mark_scheme_but_validation_pass_count']}`",
        "",
        "## Marks And Subparts",
        "",
        f"- Records with all subpart marks null: `{summary['mark_totals_and_subparts']['records_with_subparts_present_but_all_subpart_marks_null']}`",
        f"- Simple future-fillable subpart mark records: `{summary['mark_totals_and_subparts']['records_fillable_simple_future_sprint']}`",
        f"- Local question/mark-scheme total mismatches: `{summary['mark_totals_and_subparts']['records_question_total_not_equal_mark_scheme_total']}`",
        "",
        "## Run Metadata",
        "",
        f"- Fully auditable from JSON alone: `{summary['run_metadata']['fully_auditable_from_json_alone']}`",
        f"- Missing run metadata fields: `{', '.join(summary['run_metadata']['missing_fields']) or 'none'}`",
        "",
        "## Baseline",
        "",
        f"- Baseline comparison: `{json.dumps(summary['baseline_comparison'], sort_keys=True)[:2000]}`",
        "",
        "## Next Recommendation",
        "",
    ]
    for item in summary["next_iteration_recommendations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def render_recommendations_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Next Iteration Recommendations",
        "",
        "These recommendations are derived from the reporting pass only. No OCR thresholds, extraction behavior, trust gates, crop logic, mapping logic, topic labels, or difficulty labels were changed.",
        "",
    ]
    for index, item in enumerate(summary["next_iteration_recommendations"], start=1):
        lines.append(f"{index}. {item}")
    lines.extend(
        [
            "",
            "## Explicit Non-Recommendations",
            "",
            "- Do not tune OCR selection until a candidate-aware OCR-enabled export is audited.",
            "- Do not promote subpart marks automatically in this iteration; use `subpart_marks_report.csv` to scope that sprint.",
            "- Do not treat text-ready counts as student-facing truth for visual-required questions.",
            "- Do not trust topic or difficulty labels independently of text fidelity, mapping, and validation status.",
            "",
        ]
    )
    return "\n".join(lines)


def recommend_next_iteration(summary: dict[str, Any], crop_quality_rows: list[dict[str, Any]], false_negative_rows: list[dict[str, Any]], subpart_rows: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    ocr = summary["ocr_candidate_measurement"]
    readiness = summary["readiness"]
    mapping = summary["mapping_validation"]
    marks = summary["mark_totals_and_subparts"]
    run_metadata = summary["run_metadata"]

    if ocr["ocr_ran_count"] == 0:
        recommendations.append("Priority 1: produce or select an OCR-enabled candidate export before tuning OCR selection; current input has no OCR runs selected for measurement.")
    elif false_negative_rows and len(false_negative_rows) / max(1, summary["record_count"]) >= 0.03:
        recommendations.append("Priority 1: review OCR candidate-selection conservatism using `possible_ocr_false_negatives.csv` before changing crop or mapping logic.")
    elif ocr["suspicious_ocr_selected_count"] > 0:
        recommendations.append("Priority 1: review OCR-selected risk cases before considering any looser selection threshold.")

    if mapping["mapping_status_distribution"].get("fail", 0) or mapping["validation_status_distribution"].get("fail", 0):
        recommendations.append("Priority 2: address mapping/validation hard blockers; validation pass must not hide mapping failures.")

    high_2024_hard = format_profile_or_recent_hard_rate(crop_quality_rows)
    if high_2024_hard:
        recommendations.append("Priority 3: measure and fix 2024-2025 layout/crop handling, which is overrepresented in hard blockers or review tiers.")

    low_crop_rows = [row for row in crop_quality_rows if row["group_type"] == "question_format_profile" and "low" in str(row.get("question_crop_confidence_distribution"))]
    if low_crop_rows:
        recommendations.append("Priority 4: inspect crop/scope detection where low question or mark-scheme crop confidence is limiting visual readiness.")

    if marks["records_fillable_simple_future_sprint"]:
        recommendations.append(
            f"Priority 5: plan subpart mark promotion; `{marks['records_fillable_simple_future_sprint']}` records appear simple-fillable from detected mark values."
        )

    if not run_metadata["fully_auditable_from_json_alone"]:
        recommendations.append("Priority 6: add a top-level run manifest in a later export iteration so model/pipeline freshness is auditable from JSON alone.")

    if summary["topic_difficulty_side_effects"]["high_topic_confidence_but_degraded_text_count"]:
        recommendations.append("Priority 7: audit topic/difficulty reruns only after text fidelity and mapping blockers are reduced.")

    if not recommendations:
        recommendations.append("No single blocker dominates; use the representative sample and tier reports to choose the next narrow sprint.")
    return recommendations


def format_profile_or_recent_hard_rate(crop_quality_rows: list[dict[str, Any]]) -> bool:
    for row in crop_quality_rows:
        if row.get("group_type") != "year_season":
            continue
        value = str(row.get("group_value") or "")
        if not (value.startswith("2024") or value.startswith("2025")):
            continue
        count = int(row.get("record_count") or 0)
        hard = int(row.get("hard_blocker_count") or 0)
        if count and hard / count >= 0.15:
            return True
    return False


def ocr_selection_flags(
    *,
    record: dict[str, Any],
    resolver: FieldResolver,
    selected_metrics: dict[str, Any],
    ocr_metrics: dict[str, Any],
    native_metrics: dict[str, Any],
    margin: float | None,
) -> list[str]:
    if not as_bool(resolver.get(record, "ocr_selected")):
        return []
    flags: list[str] = []
    if not selected_metrics["question_number_present"]:
        flags.append("selected_ocr_missing_expected_question_number")
    expected = expected_subpart_labels(record, resolver)
    if expected and not selected_metrics["expected_subparts_present"]:
        flags.append("selected_ocr_missing_expected_subparts")
    if selected_metrics["mark_bracket_count"] == 0 and any(
        numeric(resolver.get(record, field)) for field in ["question_solution_marks", "question_total_detected", "mark_scheme_total_detected"]
    ):
        flags.append("selected_ocr_missing_expected_mark_brackets")
    if native_metrics["alpha_subpart_count"] and selected_metrics["alpha_subpart_count"] < native_metrics["alpha_subpart_count"]:
        flags.append("selected_ocr_fewer_alpha_subparts_than_native")
    if native_metrics["roman_subpart_count"] and selected_metrics["roman_subpart_count"] < native_metrics["roman_subpart_count"]:
        flags.append("selected_ocr_fewer_roman_subparts_than_native")
    structure = dict_field(record, resolver, "question_structure_detected")
    if structure.get("has_terminal_mark_total") and not selected_metrics["terminal_mark_total_present"]:
        flags.append("selected_ocr_lost_terminal_mark_total")
    if native_metrics["normalized_length"]:
        if selected_metrics["normalized_length"] > native_metrics["normalized_length"] * 1.8 and selected_metrics["normalized_length"] - native_metrics["normalized_length"] >= 80:
            flags.append("selected_ocr_much_longer_than_native")
        if selected_metrics["normalized_length"] < native_metrics["normalized_length"] * 0.55 and native_metrics["normalized_length"] - selected_metrics["normalized_length"] >= 80:
            flags.append("selected_ocr_much_shorter_than_native")
    flags.extend(text_noise_flags(selected_metrics, prefix="selected_ocr"))
    if str(resolver.get(record, "mapping_status") or "").lower() in {"fail", "review"}:
        flags.append("ocr_selected_with_mapping_not_pass")
    if str(resolver.get(record, "validation_status") or "").lower() in {"fail", "review"}:
        flags.append("ocr_selected_with_validation_not_pass")
    if str(resolver.get(record, "question_crop_confidence") or "").lower() == "low":
        flags.append("ocr_selected_with_low_question_crop_confidence")
    if str(resolver.get(record, "scope_quality_status") or "").lower() in {"fail", "review"}:
        flags.append("ocr_selected_with_uncertain_or_failed_scope")
    if str(resolver.get(record, "scope_quality_status") or "").lower() == "fail":
        flags.append("ocr_selected_for_hard_scope_failure")
    if str(resolver.get(record, "text_only_status") or "").lower() == "ready" and str(resolver.get(record, "question_text_trust") or "").lower() != "high":
        flags.append("ocr_selected_text_only_ready_without_high_trust")
    if str(resolver.get(record, "question_text_trust") or "").lower() == "high" and str(resolver.get(record, "text_fidelity_status") or "").lower() != "clean":
        flags.append("ocr_selected_high_trust_without_clean_fidelity")
    if margin is None:
        flags.append("ocr_selected_missing_score_margin")
    elif margin < 30:
        flags.append("ocr_selected_small_score_margin")
    return sorted(set(flags))


def ocr_false_negative_flags(
    *,
    record: dict[str, Any],
    resolver: FieldResolver,
    selected_metrics: dict[str, Any],
    ocr_metrics: dict[str, Any],
    native_metrics: dict[str, Any],
    margin: float | None,
) -> list[str]:
    if as_bool(resolver.get(record, "ocr_selected")):
        return []
    flags: list[str] = []
    ocr_text = str(resolver.get(record, "ocr_text") or "")
    if not ocr_text.strip():
        return []
    if margin is not None and margin >= 30:
        flags.append("ocr_score_much_higher_than_native")
    if str(resolver.get(record, "ocr_text_trust") or "").lower() == "high" and str(resolver.get(record, "question_text_trust") or "").lower() in {"low", "unusable"}:
        flags.append("high_ocr_trust_low_question_text_trust")
    if ocr_metrics["readable_spacing"] and not selected_metrics["readable_spacing"]:
        flags.append("ocr_readable_native_merged_or_sparse")
    if (
        ocr_metrics["question_number_present"]
        and ocr_metrics["mark_bracket_count"] >= selected_metrics["mark_bracket_count"]
        and ocr_metrics["subpart_count"] >= selected_metrics["subpart_count"]
        and (not selected_metrics["question_number_present"] or selected_metrics["mark_bracket_count"] == 0 or not selected_metrics["expected_subparts_present"])
    ):
        flags.append("ocr_preserves_structure_better_than_native")
    rejected_reasons = list_field(record, resolver, "ocr_rejected_reasons")
    if rejected_reasons == ["ocr_not_clearly_better"] and margin is not None and margin >= 20:
        flags.append("only_not_clearly_better_but_margin_positive")
    if selected_metrics["merged_prose_artifact_count"] and not ocr_metrics["merged_prose_artifact_count"] and not text_noise_flags(ocr_metrics, prefix="ocr"):
        flags.append("ocr_fixes_known_native_merged_prose_without_obvious_risk")
    return sorted(set(flags))


def text_noise_flags(metrics: dict[str, Any], *, prefix: str) -> list[str]:
    flags: list[str] = []
    if metrics["page_furniture_token_count"]:
        flags.append(f"{prefix}_contains_likely_page_furniture")
    if metrics["barcode_fragment_count"]:
        flags.append(f"{prefix}_contains_barcode_fragments")
    if metrics["copyright_footer_count"]:
        flags.append(f"{prefix}_contains_copyright_footer")
    if metrics["margin_text_count"]:
        flags.append(f"{prefix}_contains_margin_text")
    if metrics["next_question_contamination"]:
        flags.append(f"{prefix}_contains_next_question_contamination")
    if metrics["isolated_symbol_count"] > max(6, metrics["word_count"] // 6):
        flags.append(f"{prefix}_has_suspicious_isolated_symbols")
    if metrics["math_mangling_count"]:
        flags.append(f"{prefix}_has_obvious_math_notation_mangling")
    return flags


def text_metrics(text: str, question_number: str, expected_subparts: list[str]) -> dict[str, Any]:
    raw = str(text or "")
    normalized = normalize_text(raw)
    alpha = sorted(set(label.lower() for label in ALPHA_SUBPART_RE.findall(normalized)), key=alpha_sort_key)
    roman = sorted(set(label.lower() for label in ROMAN_SUBPART_RE.findall(normalized)), key=roman_sort_key)
    mark_values = [int(value) for value in MARK_RE.findall(normalized)]
    qnum_present = contains_question_number(normalized, question_number)
    expected = [label.lower() for label in expected_subparts]
    candidate_subparts = set(alpha) | set(roman)
    expected_present = not expected or set(expected).issubset(candidate_subparts)
    words = re.findall(r"[A-Za-z]{3,}", normalized)
    long_tokens = [word for word in words if len(word) >= 24]
    readable_spacing = bool(len(words) >= 8 and len(long_tokens) < 2 and normalized.count(" ") / max(1, sum(char.isalpha() for char in normalized)) >= 0.12)
    return {
        "length": len(raw),
        "normalized_length": len(normalized),
        "word_count": len(words),
        "question_number_present": qnum_present,
        "mark_bracket_count": len(mark_values),
        "terminal_mark_total_present": bool(re.search(r"\[\d{1,2}\]\s*$", normalized) or re.search(r"\[\d{1,2}\]\s*[.?!]?\s*$", normalized)),
        "alpha_subparts": alpha,
        "alpha_subpart_count": len(alpha),
        "roman_subparts": roman,
        "roman_subpart_count": len(roman),
        "subpart_count": len(candidate_subparts),
        "expected_subparts_present": expected_present,
        "math_token_count": len(MATH_TOKEN_RE.findall(normalized)),
        "function_token_count": len(FUNCTION_TOKEN_RE.findall(normalized)),
        "page_furniture_token_count": len(PAGE_FURNITURE_RE.findall(normalized)),
        "barcode_fragment_count": len(BARCODE_RE.findall(normalized)),
        "copyright_footer_count": len(COPYRIGHT_RE.findall(normalized)),
        "margin_text_count": len(MARGIN_RE.findall(normalized)),
        "next_question_contamination": bool(NEXT_QUESTION_RE.search(normalized)),
        "isolated_symbol_count": len(ISOLATED_SYMBOL_RE.findall(normalized)),
        "math_mangling_count": len(MATH_MANGLING_RE.findall(normalized)),
        "merged_prose_artifact_count": len(MERGED_PROSE_RE.findall(normalized)),
        "readable_spacing": readable_spacing,
    }


def text_regression_flags_from_metrics(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if baseline["question_number_present"] and not current["question_number_present"]:
        flags.append("lost_question_number")
    if current["mark_bracket_count"] < baseline["mark_bracket_count"]:
        flags.append("lost_mark_brackets")
    if current["subpart_count"] < baseline["subpart_count"]:
        flags.append("lost_subpart_labels")
    if current["math_token_count"] + 2 < baseline["math_token_count"]:
        flags.append("lost_math_tokens")
    if current["page_furniture_token_count"] > baseline["page_furniture_token_count"]:
        flags.append("introduced_page_furniture")
    return flags


def text_improvement_flags_from_metrics(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if current["question_number_present"] and not baseline["question_number_present"]:
        flags.append("restored_question_number")
    if current["mark_bracket_count"] > baseline["mark_bracket_count"]:
        flags.append("more_mark_brackets")
    if current["subpart_count"] > baseline["subpart_count"]:
        flags.append("more_subpart_labels")
    if current["merged_prose_artifact_count"] < baseline["merged_prose_artifact_count"]:
        flags.append("less_merged_prose")
    if current["page_furniture_token_count"] < baseline["page_furniture_token_count"]:
        flags.append("less_page_furniture")
    return flags


def suggested_human_judgment(
    ocr_selected: bool,
    suspicious_flags: list[str],
    false_negative_flags: list[str],
    record: dict[str, Any],
    resolver: FieldResolver,
    *,
    fallback: str = "not_enough_evidence",
) -> str:
    if ocr_selected:
        if any(
            flag
            for flag in suspicious_flags
            if any(token in flag for token in ["missing", "lost", "hard_scope", "page_furniture", "barcode", "copyright", "next_question", "math_notation"])
        ):
            return "bad_selection"
        if suspicious_flags:
            return "questionable_selection"
        if (score_margin(record, resolver) or 0) >= 30:
            return "good_selection"
        return "questionable_selection"
    if false_negative_flags:
        return "possible_false_negative"
    return fallback


def status_movement(
    current: dict[str, Any],
    baseline: dict[str, Any],
    current_resolver: FieldResolver,
    baseline_resolver: FieldResolver,
) -> dict[str, list[str]]:
    improved: list[str] = []
    worsened: list[str] = []
    unknown: list[str] = []
    for field, order in FIELD_ORDERINGS.items():
        current_value = normalize_label(current_resolver.get(current, field))
        baseline_value = normalize_label(baseline_resolver.get(baseline, field))
        if current_value == baseline_value:
            continue
        if current_value not in order or baseline_value not in order:
            unknown.append(field)
            continue
        if order[current_value] > order[baseline_value]:
            improved.append(field)
        else:
            worsened.append(field)
    return {"improved": improved, "worsened": worsened, "unknown": unknown}


def contradiction_flags(record: dict[str, Any], resolver: FieldResolver) -> list[str]:
    flags: list[str] = []
    mapping = str(resolver.get(record, "mapping_status") or "").lower()
    validation = str(resolver.get(record, "validation_status") or "").lower()
    q_total = numeric(resolver.get(record, "question_total_detected"))
    ms_total = numeric(resolver.get(record, "mark_scheme_total_detected"))
    paper_status = str(resolver.get(record, "paper_total_status") or "").lower()
    if mapping == "fail" and validation == "pass":
        flags.append("mapping_fail_but_validation_pass")
    if (_is_blank(resolver.get(record, "mark_scheme_text")) or _is_blank(resolver.get(record, "mark_scheme_image_path"))) and validation == "pass":
        flags.append("missing_mark_scheme_but_validation_pass")
    if q_total is not None and ms_total is not None and q_total != ms_total:
        flags.append("question_total_mark_scheme_total_disagree")
    if paper_status in {"matched", "recovered_after_rescan"} and validation == "fail":
        flags.append("paper_total_matched_but_local_validation_failed")
    return sorted(flags)


def count_values(records: Iterable[dict[str, Any]], resolver: FieldResolver, field: str, *, force_notes: bool = False) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if force_notes:
            notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
            value = notes.get(field)
        else:
            value = resolver.get(record, field)
        counts[value_label(value)] += 1
    return dict(sorted(counts.items()))


def count_row_values(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(value_label(row.get(field)) for row in rows)
    return dict(sorted(counts.items()))


def count_year_season(records: list[dict[str, Any]], resolver: FieldResolver) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[year_season_label(resolver.get(record, "paper"))] += 1
    return dict(sorted(counts.items()))


def year_season_label(paper: Any) -> str:
    parsed = parse_paper_id(paper)
    if not parsed:
        return "missing"
    return f"{parsed['year']}_{parsed['season']}"


def parse_paper_id(paper: Any) -> dict[str, Any] | None:
    match = PAPER_RE.match(str(paper or ""))
    if not match:
        return None
    yy = int(match.group("yy"))
    return {
        "component": match.group("component"),
        "season": match.group("season").lower(),
        "year": 2000 + yy,
    }


def is_2024_2025_record(record: dict[str, Any], resolver: FieldResolver) -> bool:
    parsed = parse_paper_id(resolver.get(record, "paper"))
    if parsed and parsed["year"] in {2024, 2025}:
        return True
    profile = str(resolver.get(record, "question_format_profile") or "").lower()
    return "2024" in profile or "2025" in profile or profile not in {"", "missing", "legacy"}


def score_margin(record: dict[str, Any], resolver: FieldResolver) -> float | None:
    ocr_score = numeric(resolver.get(record, "ocr_text_score"))
    native_score = numeric(resolver.get(record, "native_text_score"))
    if ocr_score is None or native_score is None:
        return None
    return round_number(ocr_score - native_score)


def selected_minus_native_margin(record: dict[str, Any], resolver: FieldResolver) -> float | None:
    selected = numeric(resolver.get(record, "selected_text_score"))
    native = numeric(resolver.get(record, "native_text_score"))
    if selected is None or native is None:
        return None
    return round_number(selected - native)


def selected_minus_ocr_margin(record: dict[str, Any], resolver: FieldResolver) -> float | None:
    selected = numeric(resolver.get(record, "selected_text_score"))
    ocr_score = numeric(resolver.get(record, "ocr_text_score"))
    if selected is None or ocr_score is None:
        return None
    return round_number(selected - ocr_score)


def numeric_summary(values: Iterable[Any], total_count: int | None = None) -> dict[str, Any]:
    raw = list(values)
    scores = sorted(value for value in (numeric(value) for value in raw) if value is not None)
    total = len(raw) if total_count is None else total_count
    if not scores:
        return {
            "count": 0,
            "missing_count": total,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(scores),
        "missing_count": max(0, total - len(scores)),
        "min": round_number(scores[0]),
        "p25": round_number(percentile(scores, 0.25)),
        "median": round_number(percentile(scores, 0.50)),
        "p75": round_number(percentile(scores, 0.75)),
        "max": round_number(scores[-1]),
        "mean": round_number(mean(scores)),
    }


def percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize_values(values: list[Any], total: int) -> str:
    numeric_values = [numeric(value) for value in values]
    numeric_values = [value for value in numeric_values if value is not None]
    if numeric_values and len(numeric_values) == len(values):
        return json.dumps(numeric_summary(numeric_values, total_count=total), sort_keys=True)
    counts = Counter(value_label(value) for value in values)
    if len(counts) <= 25:
        return json.dumps(dict(sorted(counts.items())), sort_keys=True)
    top = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:25])
    return json.dumps({"distinct_count": len(counts), "top_values": top}, sort_keys=True)


def missingness_blocks_measurement(field: str, present_count: int, total: int) -> str:
    if total == 0:
        return "empty_input"
    if field in OCR_MEASUREMENT_FIELDS and present_count == 0:
        return "blocks_ocr_candidate_measurement"
    if field in {"question_id", "question_image_path", "mark_scheme_image_path", "mapping_status", "validation_status"} and present_count < total:
        return "blocks_readiness_measurement"
    if field in {"question_crop_confidence", "mark_scheme_crop_confidence"} and present_count == 0:
        return "blocks_crop_quality_measurement"
    return ""


def native_text_candidate(record: dict[str, Any], resolver: FieldResolver) -> str:
    for field in ["native_text", "pdf_text", "body_text_normalized", "body_text_raw"]:
        value = resolver.get(record, field)
        if isinstance(value, str) and value.strip():
            return value
    if str(resolver.get(record, "text_candidate_source") or "").lower() in {"native", "native_pdf"}:
        return str(resolver.get(record, "question_text") or "")
    return ""


def expected_subpart_labels(record: dict[str, Any], resolver: FieldResolver) -> list[str]:
    subparts = list_field(record, resolver, "subparts")
    if subparts:
        return [str(label).strip("()").lower() for label in subparts if str(label).strip()]
    structure = dict_field(record, resolver, "question_structure_detected")
    value = structure.get("subparts")
    if isinstance(value, list):
        return [str(label).strip("()").lower() for label in value if str(label).strip()]
    return []


def totals_agree_or_missing(record: dict[str, Any], resolver: FieldResolver) -> bool:
    q_total = numeric(resolver.get(record, "question_total_detected"))
    ms_total = numeric(resolver.get(record, "mark_scheme_total_detected"))
    return q_total is None or ms_total is None or q_total == ms_total


def artifact_exists(value: Any, artifact_root: Path) -> bool:
    path = Path(str(value))
    if path.is_absolute():
        return path.exists()
    return (artifact_root / path).exists()


def subpart_marks_all_null(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(_is_blank(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(_is_blank(item) for item in value)
    return _is_blank(value)


def null_subpart_mark_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(1 for item in value.values() if _is_blank(item))
    if isinstance(value, list):
        return sum(1 for item in value if _is_blank(item))
    return 0


def has_nested_subparts(subparts: list[str], structure: dict[str, Any], mark_values: list[int]) -> bool:
    if structure.get("subpart_type") == "nested":
        return True
    if any(label in {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"} for label in subparts):
        return True
    return bool(subparts and mark_values and len(mark_values) > len(subparts))


def duplicate_ids(records: list[dict[str, Any]]) -> list[str]:
    counts = Counter(str(record.get("question_id") or "") for record in records if record.get("question_id"))
    return sorted(qid for qid, count in counts.items() if count > 1)


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record.get("question_id") or "")
        if qid and qid not in result:
            result[qid] = record
    return result


def list_field(record: dict[str, Any], resolver: FieldResolver, field: str) -> list[str]:
    return [str(item) for item in as_list(resolver.get(record, field))]


def dict_field(record: dict[str, Any], resolver: FieldResolver, field: str) -> dict[str, Any]:
    value = resolver.get(record, field)
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _is_blank(value):
        return []
    return [value]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, int):
        return value != 0
    return False


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def contains_question_number(text: str, question_number: str) -> bool:
    if not question_number:
        return False
    return bool(re.search(QUESTION_NUMBER_RE_TEMPLATE.format(number=re.escape(str(question_number))), text))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split())


def normalize_label(value: Any) -> str:
    if _is_blank(value):
        return "missing"
    return str(value).strip().lower()


def normalize_compare_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    return value


def value_label(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if _is_blank(value):
        return "missing"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def short_text(value: Any, limit: int = 2000) -> str:
    text = str(value or "").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20] + " ...[truncated]"


def _is_blank(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _values_equal(left: Any, right: Any) -> bool:
    return normalize_compare_value(left) == normalize_compare_value(right)


def record_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (str(record.get("paper_family") or ""), str(record.get("paper") or ""), str(record.get("question_id") or record.get("question_number") or ""))


def alpha_sort_key(label: str) -> int:
    order = ["a", "b", "c", "d", "e", "f", "g", "h"]
    return order.index(label) if label in order else len(order)


def roman_sort_key(label: str) -> int:
    order = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
    return order.index(label) if label in order else len(order)


def balanced_sample_by_family(records: list[dict[str, Any]], bucket_name: str, limit: int, resolver: FieldResolver) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda item: stable_hash(bucket_name, str(resolver.get(item, "question_id") or ""))):
        grouped[str(resolver.get(record, "paper_family") or "missing")].append(record)
    sample: list[dict[str, Any]] = []
    while len(sample) < limit and any(grouped.values()):
        for family in sorted(grouped):
            if grouped[family] and len(sample) < limit:
                sample.append(grouped[family].pop(0))
    return sample


def stable_hash(bucket: str, question_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}:{bucket}:{question_id}".encode("utf-8")).hexdigest()


def top_counter_dict(counter: Counter[str], limit: int = 25) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit])


def rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 6)


def round_number(value: float) -> int | float:
    rounded = round(float(value), 3)
    return int(rounded) if rounded.is_integer() else rounded


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("\r", " ").replace("\n", "\\n")


if __name__ == "__main__":
    raise SystemExit(main())
