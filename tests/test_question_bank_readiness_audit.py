from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

from scripts.audit_question_bank_readiness import main


def _base_record(question_id: str = "12spring24_q01", question_number: str = "1") -> dict:
    return {
        "question_id": question_id,
        "paper": "12spring24",
        "paper_family": "p1",
        "question_number": question_number,
        "question_image_path": f"p1/12spring24/questions/q{int(question_number):02d}.png",
        "mark_scheme_image_path": f"p1/12spring24/mark_scheme/q{int(question_number):02d}.png",
        "question_text": f"{question_number} (a) Find x. [2] (b) Find y. [3]",
        "ocr_text": "",
        "ocr_ran": False,
        "ocr_engine": "",
        "ocr_text_trust": "unusable",
        "question_text_trust": "high",
        "question_text_role": "readable_text",
        "text_only_status": "ready",
        "visual_curation_status": "ready",
        "visual_required": False,
        "visual_reason_flags": [],
        "mark_scheme_text": "M1 A1 M1 A1 A1",
        "question_solution_marks": 5,
        "subparts": ["a", "b"],
        "subparts_solution_marks": {"a": None, "b": None},
        "topic": "algebra",
        "difficulty": "average",
        "difficulty_score": 45,
        "difficulty_band": "average",
        "notes": {
            "mapping_status": "pass",
            "mapping_failure_reason": "",
            "validation_status": "pass",
            "validation_flags": [],
            "scope_quality_status": "clean",
            "question_crop_confidence": "high",
            "mark_scheme_crop_confidence": "high",
            "text_fidelity_status": "clean",
            "text_fidelity_flags": [],
            "text_source_profile": "native_pdf",
            "question_text_trust": "high",
            "question_text_role": "readable_text",
            "visual_curation_status": "ready",
            "text_only_status": "ready",
            "visual_required": False,
            "visual_reason_flags": [],
            "review_flags": [],
            "extraction_quality_flags": [],
            "ocr_ran": False,
            "ocr_engine": "",
            "ocr_text_trust": "unusable",
            "ocr_text_role": "missing",
            "text_candidate_source": "native",
            "native_text_score": 50,
            "ocr_text_score": -100,
            "selected_text_score": 50,
            "text_candidate_decision": "native_retained",
            "text_candidate_decision_reasons": ["expected_question_number_present"],
            "ocr_selected": False,
            "ocr_rejected_reasons": ["empty_ocr_text"],
            "question_structure_detected": {
                "format_profile": "legacy",
                "subparts": ["a", "b"],
                "subpart_type": "alpha",
                "mark_values_detected": [2, 3],
                "has_terminal_mark_total": True,
                "question_total_detected": 5,
                "contamination_detected": False,
                "likely_truncated": False,
            },
            "mark_scheme_structure_detected": {
                "subparts": ["a", "b"],
                "question_total_detected": 5,
                "mark_scheme_total_detected": 5,
            },
            "question_total_detected": 5,
            "mark_scheme_total_detected": 5,
            "question_format_profile": "legacy",
            "paper_total_status": "matched",
            "topic_confidence": "high",
            "topic_uncertain": False,
            "topic_trust_status": "normal",
            "difficulty_confidence": "high",
            "difficulty_model_version": "local-difficulty-v1",
        },
    }


def _write_bank(path: Path, records: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.question_bank",
                "schema_version": 2,
                "record_count": len(records),
                "questions": records,
            }
        ),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_audit(
    tmp_path: Path,
    records: list[dict],
    *,
    baseline_records: list[dict] | None = None,
    artifact_root: Path | None = None,
    out_name: str = "audit",
) -> tuple[Path, dict]:
    input_path = tmp_path / f"{out_name}_question_bank.json"
    out_dir = tmp_path / out_name
    _write_bank(input_path, records)
    args = ["--input", str(input_path), "--out-dir", str(out_dir)]
    if baseline_records is not None:
        baseline_path = tmp_path / f"{out_name}_baseline.json"
        _write_bank(baseline_path, baseline_records)
        args.extend(["--baseline", str(baseline_path)])
    if artifact_root is not None:
        args.extend(["--artifact-root", str(artifact_root)])

    assert main(args) == 0
    return out_dir, json.loads((out_dir / "audit_summary.json").read_text(encoding="utf-8"))


def _field_row(out_dir: Path, field: str) -> dict[str, str]:
    rows = _read_csv(out_dir / "field_presence_report.csv")
    return next(row for row in rows if row["field"] == field)


def _rows_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["question_id"]: row for row in rows}


def _clone_record(question_id: str, question_number: str) -> dict:
    return copy.deepcopy(_base_record(question_id, question_number))


def _set_note(record: dict, **values: object) -> dict:
    record.setdefault("notes", {}).update(values)
    return record


def _set_question_totals(record: dict, question_total: int, mark_total: int | None = None) -> dict:
    if mark_total is None:
        mark_total = question_total
    record["question_solution_marks"] = question_total
    record["notes"]["question_total_detected"] = question_total
    record["notes"]["mark_scheme_total_detected"] = mark_total
    record["notes"]["question_structure_detected"]["question_total_detected"] = question_total
    record["notes"]["mark_scheme_structure_detected"]["question_total_detected"] = question_total
    record["notes"]["mark_scheme_structure_detected"]["mark_scheme_total_detected"] = mark_total
    return record


def _json_list(value: str) -> list[str]:
    return json.loads(value) if value else []


def test_readiness_audit_writes_reports_and_baseline_comparison(tmp_path: Path) -> None:
    good = _base_record()
    blocker = _base_record("12spring24_q02", "2")
    blocker["mark_scheme_image_path"] = ""
    blocker["mark_scheme_text"] = ""
    blocker["question_text_trust"] = "low"
    blocker["notes"]["question_text_trust"] = "high"
    blocker["notes"]["mapping_status"] = "fail"
    blocker["notes"]["validation_status"] = "pass"
    blocker["notes"]["native_text_score"] = None
    blocker["notes"]["ocr_text_score"] = None
    blocker["notes"]["selected_text_score"] = None

    baseline_good = _base_record()
    baseline_good["question_text"] = "1 (a) Find x. [2]"
    baseline_good["notes"]["validation_status"] = "review"
    removed = _base_record("12spring24_q03", "3")

    input_path = tmp_path / "question_bank.json"
    baseline_path = tmp_path / "baseline.json"
    out_dir = tmp_path / "audit"
    _write_bank(input_path, [good, blocker])
    _write_bank(baseline_path, [baseline_good, removed])

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--baseline",
            str(baseline_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((out_dir / "audit_summary.json").read_text(encoding="utf-8"))
    baseline_summary = json.loads((out_dir / "baseline_comparison_summary.json").read_text(encoding="utf-8"))

    assert summary["record_count"] == 2
    assert summary["field_disagreement_count"] == 1
    assert summary["ocr_candidate_measurement"]["native_text_score_summary"]["count"] == 1
    assert summary["ocr_candidate_measurement"]["native_text_score_summary"]["missing_count"] == 1
    assert summary["mapping_validation"]["mapping_fail_but_validation_pass_count"] == 1
    assert summary["mark_totals_and_subparts"]["records_fillable_simple_future_sprint"] == 2
    assert baseline_summary["records_added"] == 1
    assert baseline_summary["records_removed"] == 1
    assert baseline_summary["records_present_in_both"] == 1
    assert (out_dir / "field_presence_report.csv").exists()
    assert (out_dir / "readiness_tiers.csv").exists()
    assert (out_dir / "subpart_marks_report.csv").exists()

    hard_blockers = _read_csv(out_dir / "hard_blockers.csv")
    assert hard_blockers[0]["question_id"] == "12spring24_q02"
    assert "missing_mark_scheme_image_path" in hard_blockers[0]["hard_blockers"]


def test_readiness_audit_surfaces_suspicious_ocr_and_false_negatives(tmp_path: Path) -> None:
    suspicious = _base_record("12spring24_q01", "1")
    suspicious["question_text"] = "Cambridge International Find x"
    suspicious["ocr_text"] = "Cambridge International Find x"
    suspicious["ocr_ran"] = True
    suspicious["ocr_engine"] = "tesseract"
    suspicious["question_solution_marks"] = None
    suspicious["notes"].update(
        {
            "ocr_ran": True,
            "ocr_engine": "tesseract",
            "text_candidate_source": "ocr",
            "text_candidate_decision": "ocr_selected",
            "ocr_selected": True,
            "native_text_score": 5,
            "ocr_text_score": 42,
            "selected_text_score": 42,
            "ocr_rejected_reasons": [],
            "mapping_status": "fail",
            "validation_status": "fail",
            "scope_quality_status": "fail",
            "question_crop_confidence": "low",
        }
    )

    false_negative = _base_record("12spring24_q02", "2")
    false_negative["question_text"] = "2 Findthevalueofx. [2]"
    false_negative["ocr_text"] = "2 Find the value of x. [2]"
    false_negative["notes"].update(
        {
            "native_text_score": 10,
            "ocr_text_score": 55,
            "selected_text_score": 10,
            "ocr_text_trust": "high",
            "question_text_trust": "low",
            "ocr_rejected_reasons": ["ocr_not_clearly_better"],
        }
    )

    input_path = tmp_path / "question_bank.json"
    out_dir = tmp_path / "audit"
    _write_bank(input_path, [suspicious, false_negative])

    assert main(["--input", str(input_path), "--out-dir", str(out_dir)]) == 0

    suspicious_rows = _read_csv(out_dir / "ocr_suspicious_records.csv")
    false_negative_rows = _read_csv(out_dir / "possible_ocr_false_negatives.csv")

    assert suspicious_rows[0]["question_id"] == "12spring24_q01"
    assert "ocr_selected_with_mapping_not_pass" in suspicious_rows[0]["audit_flags"]
    assert "ocr_selected_for_hard_scope_failure" in suspicious_rows[0]["audit_flags"]
    assert "selected_ocr_contains_likely_page_furniture" in suspicious_rows[0]["audit_flags"]

    assert false_negative_rows[0]["question_id"] == "12spring24_q02"
    assert "ocr_score_much_higher_than_native" in false_negative_rows[0]["audit_flags"]
    assert "only_not_clearly_better_but_margin_positive" in false_negative_rows[0]["audit_flags"]


def test_readiness_audit_artifact_root_missing_files_become_blockers(tmp_path: Path) -> None:
    record = _base_record()
    input_path = tmp_path / "question_bank.json"
    artifact_root = tmp_path / "output"
    out_dir = tmp_path / "audit"
    _write_bank(input_path, [record])

    assert main(["--input", str(input_path), "--artifact-root", str(artifact_root), "--out-dir", str(out_dir)]) == 0

    hard_blockers = _read_csv(out_dir / "hard_blockers.csv")
    assert hard_blockers[0]["question_id"] == "12spring24_q01"
    assert "artifact_missing:question_image_path" in hard_blockers[0]["hard_blockers"]
    assert "artifact_missing:mark_scheme_image_path" in hard_blockers[0]["hard_blockers"]


def test_readiness_audit_resolves_top_level_notes_conflicts_and_missing_fields(tmp_path: Path) -> None:
    top_only = _clone_record("12spring24_q01", "1")
    top_only["question_text_trust"] = "high"
    top_only["notes"].pop("question_text_trust")

    notes_only = _clone_record("12spring24_q02", "2")
    notes_only.pop("question_text_trust")
    notes_only["notes"]["question_text_trust"] = "medium"

    both_equal = _clone_record("12spring24_q03", "3")
    both_equal["question_text_trust"] = "high"
    both_equal["notes"]["question_text_trust"] = "high"

    conflict = _clone_record("12spring24_q04", "4")
    conflict["question_text_trust"] = "low"
    conflict["notes"]["question_text_trust"] = "high"

    missing = _clone_record("12spring24_q05", "5")
    missing.pop("question_text_trust")
    missing["notes"].pop("question_text_trust")

    out_dir, summary = _run_audit(tmp_path, [top_only, notes_only, both_equal, conflict, missing])
    trust_presence = _field_row(out_dir, "question_text_trust")

    assert trust_presence["preferred_source"] == "top_level"
    assert trust_presence["top_level_key_count"] == "3"
    assert trust_presence["notes_key_count"] == "3"
    assert trust_presence["present_count"] == "4"
    assert trust_presence["missing_count"] == "1"
    assert trust_presence["duplicate_disagreement_count"] == "1"
    assert summary["field_disagreement_count"] == 1
    assert summary["field_disagreements_sample"] == [
        {
            "field": "question_text_trust",
            "notes_value": "high",
            "preferred_source": "top_level",
            "question_id": "12spring24_q04",
            "top_level_value": "low",
        }
    ]
    assert summary["ocr_candidate_measurement"]["question_text_trust_distribution"] == {
        "high": 2,
        "low": 1,
        "medium": 1,
        "missing": 1,
    }


def test_readiness_audit_handles_missing_optional_fields_and_score_summaries(tmp_path: Path) -> None:
    missing = _clone_record("12spring24_q01", "1")
    for field in [
        "ocr_text",
        "ocr_ran",
        "ocr_selected",
        "native_text_score",
        "ocr_text_score",
        "selected_text_score",
        "text_candidate_source",
        "text_candidate_decision",
        "ocr_rejected_reasons",
        "visual_curation_status",
        "text_only_status",
        "question_text_trust",
        "question_crop_confidence",
        "mark_scheme_crop_confidence",
    ]:
        missing.pop(field, None)
        missing["notes"].pop(field, None)

    null_scores = _clone_record("12spring24_q02", "2")
    null_scores["notes"]["native_text_score"] = None
    null_scores["notes"]["ocr_text_score"] = None
    null_scores["notes"]["selected_text_score"] = None

    non_numeric_scores = _clone_record("12spring24_q03", "3")
    non_numeric_scores["notes"]["native_text_score"] = "42"
    non_numeric_scores["notes"]["ocr_text_score"] = "not-a-number"
    non_numeric_scores["notes"]["selected_text_score"] = "12"

    numeric_scores = _clone_record("12spring24_q04", "4")
    numeric_scores["notes"]["native_text_score"] = 10
    numeric_scores["notes"]["ocr_text_score"] = 40
    numeric_scores["notes"]["selected_text_score"] = 10

    out_dir, summary = _run_audit(tmp_path, [missing, null_scores, non_numeric_scores, numeric_scores])

    assert _field_row(out_dir, "ocr_text")["missing_count"] == "4"
    assert _field_row(out_dir, "ocr_ran")["missing_count"] == "1"
    assert _field_row(out_dir, "question_crop_confidence")["missing_count"] == "1"
    assert _field_row(out_dir, "mark_scheme_crop_confidence")["missing_count"] == "1"
    assert summary["schema_blocker"].startswith("partial_ocr_candidate_fields_missing:")
    assert "ocr_text" in summary["schema_blocker"]
    assert summary["ocr_candidate_measurement"]["possible_ocr_false_negative_count"] == 0
    assert summary["ocr_candidate_measurement"]["suspicious_ocr_selected_count"] == 0

    assert summary["ocr_candidate_measurement"]["native_text_score_summary"] == {
        "count": 1,
        "missing_count": 3,
        "min": 10,
        "p25": 10,
        "median": 10,
        "p75": 10,
        "max": 10,
        "mean": 10,
    }
    assert summary["ocr_candidate_measurement"]["ocr_text_score_summary"]["count"] == 1
    assert summary["ocr_candidate_measurement"]["ocr_text_score_summary"]["missing_count"] == 3
    assert summary["ocr_candidate_measurement"]["ocr_text_score_summary"]["mean"] == 40
    assert summary["ocr_candidate_measurement"]["selected_text_score_summary"]["count"] == 1
    assert summary["ocr_candidate_measurement"]["selected_text_score_summary"]["missing_count"] == 3


def test_readiness_audit_distinguishes_ocr_states_and_risk_outputs(tmp_path: Path) -> None:
    inactive = _clone_record("12spring24_q01", "1")
    inactive["ocr_text"] = ""
    inactive["notes"].update(
        {
            "ocr_ran": False,
            "ocr_selected": False,
            "text_candidate_source": "native",
            "text_candidate_decision": "native_retained",
            "ocr_rejected_reasons": ["empty_ocr_text"],
        }
    )

    selected_good = _clone_record("12spring24_q02", "2")
    selected_good["ocr_text"] = selected_good["question_text"]
    selected_good["ocr_ran"] = True
    selected_good["ocr_engine"] = "tesseract"
    selected_good["notes"].update(
        {
            "ocr_ran": True,
            "ocr_engine": "tesseract",
            "ocr_selected": True,
            "text_candidate_source": "ocr",
            "text_candidate_decision": "ocr_selected",
            "native_text_score": 10,
            "ocr_text_score": 70,
            "selected_text_score": 70,
            "ocr_rejected_reasons": [],
        }
    )

    rejected = _clone_record("12spring24_q03", "3")
    rejected["ocr_text"] = rejected["question_text"]
    rejected["ocr_ran"] = True
    rejected["ocr_engine"] = "tesseract"
    rejected["notes"].update(
        {
            "ocr_ran": True,
            "ocr_engine": "tesseract",
            "ocr_selected": False,
            "text_candidate_source": "native",
            "text_candidate_decision": "native_retained",
            "native_text_score": 70,
            "ocr_text_score": 50,
            "selected_text_score": 70,
            "ocr_rejected_reasons": ["ocr_not_clearly_better"],
        }
    )

    false_negative = _clone_record("12spring24_q04", "4")
    false_negative["question_text"] = "4 Findthevalueofxwhenyisgivenandshowyourworkingclearly"
    false_negative["ocr_text"] = "4 Find the value of x when y is given and show your working clearly. [2]"
    false_negative["ocr_ran"] = True
    false_negative["ocr_engine"] = "tesseract"
    false_negative["ocr_text_trust"] = "high"
    false_negative["question_text_trust"] = "low"
    false_negative["subparts"] = []
    false_negative["subparts_solution_marks"] = {}
    false_negative["notes"]["question_structure_detected"]["subparts"] = []
    false_negative["notes"].update(
        {
            "ocr_ran": True,
            "ocr_engine": "tesseract",
            "ocr_selected": False,
            "text_candidate_source": "native",
            "text_candidate_decision": "native_retained",
            "native_text_score": 5,
            "ocr_text_score": 65,
            "selected_text_score": 5,
            "ocr_text_trust": "high",
            "question_text_trust": "low",
            "ocr_rejected_reasons": ["ocr_not_clearly_better"],
        }
    )

    suspicious = _clone_record("12spring24_q05", "5")
    suspicious["question_text"] = "Cambridge International"
    suspicious["ocr_text"] = suspicious["question_text"]
    suspicious["ocr_ran"] = True
    suspicious["ocr_engine"] = "tesseract"
    suspicious["native_text"] = "5 " + ("Find x and show your full working. " * 15) + "[5]"
    suspicious["subparts"] = []
    suspicious["subparts_solution_marks"] = {}
    suspicious["notes"]["question_structure_detected"]["subparts"] = []
    suspicious["notes"].update(
        {
            "ocr_ran": True,
            "ocr_engine": "tesseract",
            "ocr_selected": True,
            "text_candidate_source": "ocr",
            "text_candidate_decision": "ocr_selected",
            "native_text_score": 5,
            "ocr_text_score": 60,
            "selected_text_score": 60,
            "ocr_rejected_reasons": [],
            "mapping_status": "fail",
            "validation_status": "fail",
            "question_crop_confidence": "low",
        }
    )
    _set_question_totals(suspicious, 5)

    out_dir, summary = _run_audit(tmp_path, [inactive, selected_good, rejected, false_negative, suspicious])

    assert summary["ocr_candidate_measurement"]["ocr_ran_count"] == 4
    assert summary["ocr_candidate_measurement"]["ocr_selected_count"] == 2
    assert summary["ocr_candidate_measurement"]["text_candidate_source_distribution"] == {"native": 3, "ocr": 2}
    assert summary["ocr_candidate_measurement"]["text_candidate_decision_distribution"] == {"native_retained": 3, "ocr_selected": 2}
    assert summary["ocr_candidate_measurement"]["possible_ocr_false_negative_count"] == 1
    assert summary["ocr_candidate_measurement"]["suspicious_ocr_selected_count"] == 1

    suspicious_rows = _rows_by_id(_read_csv(out_dir / "ocr_suspicious_records.csv"))
    false_negative_rows = _rows_by_id(_read_csv(out_dir / "possible_ocr_false_negatives.csv"))
    suspicious_flags = set(_json_list(suspicious_rows["12spring24_q05"]["audit_flags"]))
    false_negative_flags = set(_json_list(false_negative_rows["12spring24_q04"]["audit_flags"]))

    assert suspicious_flags >= {
        "selected_ocr_missing_expected_question_number",
        "selected_ocr_missing_expected_mark_brackets",
        "ocr_selected_with_mapping_not_pass",
        "ocr_selected_with_validation_not_pass",
        "ocr_selected_with_low_question_crop_confidence",
        "selected_ocr_much_shorter_than_native",
    }
    assert false_negative_flags >= {
        "ocr_score_much_higher_than_native",
        "high_ocr_trust_low_question_text_trust",
        "ocr_readable_native_merged_or_sparse",
        "ocr_preserves_structure_better_than_native",
        "only_not_clearly_better_but_margin_positive",
    }


def test_readiness_audit_classifies_tiers_from_implemented_rules(tmp_path: Path) -> None:
    hard = _clone_record("12spring24_q01", "1")
    hard["question_image_path"] = ""

    tier2 = _clone_record("12spring24_q02", "2")
    tier2["question_text_trust"] = "medium"
    tier2["text_only_status"] = "review"
    tier2["visual_curation_status"] = "review"
    tier2["notes"].update(
        {
            "question_text_trust": "medium",
            "text_only_status": "review",
            "visual_curation_status": "review",
            "question_crop_confidence": "medium",
            "mark_scheme_crop_confidence": "medium",
        }
    )

    tier3 = _clone_record("12spring24_q03", "3")
    tier3["text_only_status"] = "review"
    tier3["notes"]["text_only_status"] = "review"

    tier4 = _clone_record("12spring24_q04", "4")
    tier4["visual_curation_status"] = "review"
    tier4["notes"]["visual_curation_status"] = "review"

    tier5 = _clone_record("12spring24_q05", "5")

    out_dir, summary = _run_audit(tmp_path, [hard, tier2, tier3, tier4, tier5])
    rows = _rows_by_id(_read_csv(out_dir / "readiness_tiers.csv"))

    assert summary["readiness"]["highest_tier_counts"] == {"0": 1, "2": 1, "3": 1, "4": 1, "5": 1}
    assert rows["12spring24_q01"]["highest_tier"] == "0"
    assert rows["12spring24_q01"]["tier_label"] == "Tier 0 - Hard blocker"
    assert rows["12spring24_q02"]["highest_tier"] == "2"
    assert rows["12spring24_q02"]["tier_2_multimodal_candidate"] == "true"
    assert rows["12spring24_q02"]["tier_3_visual_ready"] == "false"
    assert rows["12spring24_q03"]["highest_tier"] == "3"
    assert rows["12spring24_q03"]["tier_3_visual_ready"] == "true"
    assert rows["12spring24_q03"]["tier_4_text_ready"] == "false"
    assert rows["12spring24_q04"]["highest_tier"] == "4"
    assert rows["12spring24_q04"]["tier_4_text_ready"] == "true"
    assert rows["12spring24_q04"]["tier_3_visual_ready"] == "false"
    assert rows["12spring24_q05"]["highest_tier"] == "5"
    assert rows["12spring24_q05"]["tier_5_gold_pilot"] == "true"


def test_readiness_audit_detects_hard_blocker_reasons_including_artifacts(tmp_path: Path) -> None:
    missing_id = _clone_record("", "1")
    missing_question_image = _clone_record("12spring24_q02", "2")
    missing_question_image["question_image_path"] = ""
    missing_mark_image = _clone_record("12spring24_q03", "3")
    missing_mark_image["mark_scheme_image_path"] = ""
    mapping_fail = _set_note(_clone_record("12spring24_q04", "4"), mapping_status="fail")
    validation_fail = _set_note(_clone_record("12spring24_q05", "5"), validation_status="fail")
    total_mismatch = _set_question_totals(_clone_record("12spring24_q06", "6"), 6, mark_total=5)
    artifact_missing = _clone_record("12spring24_q07", "7")

    records = [missing_id, missing_question_image, missing_mark_image, mapping_fail, validation_fail, total_mismatch, artifact_missing]
    artifact_root = tmp_path / "artifacts"
    for record in records:
        if record is artifact_missing:
            continue
        for field in ["question_image_path", "mark_scheme_image_path"]:
            value = record.get(field)
            if value:
                path = artifact_root / value
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("image", encoding="utf-8")

    out_dir, summary = _run_audit(tmp_path, records, artifact_root=artifact_root)
    rows = _read_csv(out_dir / "hard_blockers.csv")
    all_reasons = {reason for row in rows for reason in _json_list(row["hard_blockers"])}

    assert summary["readiness"]["hard_blocker_count"] == 7
    assert all_reasons >= {
        "missing_question_id",
        "missing_question_image_path",
        "missing_mark_scheme_image_path",
        "mapping_status_fail",
        "validation_status_fail",
        "question_total_mark_scheme_total_disagree",
        "artifact_missing:question_image_path",
        "artifact_missing:mark_scheme_image_path",
    }


def test_readiness_audit_counts_mapping_and_validation_distributions(tmp_path: Path) -> None:
    mapping_pass = _set_note(_clone_record("12spring24_q01", "1"), mapping_status="pass", validation_status="pass")
    mapping_review = _set_note(_clone_record("12spring24_q02", "2"), mapping_status="review", validation_status="review")
    mapping_fail = _set_note(_clone_record("12spring24_q03", "3"), mapping_status="fail", validation_status="fail")
    missing = _clone_record("12spring24_q04", "4")
    missing["notes"].pop("mapping_status")
    missing["notes"].pop("validation_status")

    _, summary = _run_audit(tmp_path, [mapping_pass, mapping_review, mapping_fail, missing])

    assert summary["mapping_validation"]["mapping_status_distribution"] == {
        "fail": 1,
        "missing": 1,
        "pass": 1,
        "review": 1,
    }
    assert summary["mapping_validation"]["validation_status_distribution"] == {
        "fail": 1,
        "missing": 1,
        "pass": 1,
        "review": 1,
    }


def test_readiness_audit_reports_subpart_mark_fillability_reasons(tmp_path: Path) -> None:
    fillable = _clone_record("12spring24_q01", "1")
    fillable["subparts"] = ["a", "b"]
    fillable["subparts_solution_marks"] = {"a": None, "b": None}
    fillable["notes"]["question_structure_detected"].update({"subparts": ["a", "b"], "mark_values_detected": [2, 3]})
    _set_question_totals(fillable, 5)

    wrong_sum = _clone_record("12spring24_q02", "2")
    wrong_sum["subparts"] = ["a", "b"]
    wrong_sum["subparts_solution_marks"] = {"a": None, "b": None}
    wrong_sum["notes"]["question_structure_detected"].update({"subparts": ["a", "b"], "mark_values_detected": [2, 2]})
    _set_question_totals(wrong_sum, 5)

    likely_nested = _clone_record("12spring24_q03", "3")
    likely_nested["subparts"] = ["a"]
    likely_nested["subparts_solution_marks"] = {"a": None}
    likely_nested["notes"]["question_structure_detected"].update({"subparts": ["a"], "mark_values_detected": [1, 2]})
    _set_question_totals(likely_nested, 3)

    populated = _clone_record("12spring24_q04", "4")
    populated["subparts"] = ["a", "b"]
    populated["subparts_solution_marks"] = {"a": 2, "b": 3}
    populated["notes"]["question_structure_detected"].update({"subparts": ["a", "b"], "mark_values_detected": [2, 3]})
    _set_question_totals(populated, 5)

    out_dir, summary = _run_audit(tmp_path, [fillable, wrong_sum, likely_nested, populated])
    rows = _rows_by_id(_read_csv(out_dir / "subpart_marks_report.csv"))

    assert summary["mark_totals_and_subparts"]["records_with_subparts_present_but_all_subpart_marks_null"] == 3
    assert summary["mark_totals_and_subparts"]["records_fillable_simple_future_sprint"] == 1
    assert summary["mark_totals_and_subparts"]["records_likely_nested"] == 1
    assert rows["12spring24_q01"]["fillable_simple"] == "true"
    assert rows["12spring24_q01"]["reason"] == "fillable_simple"
    assert rows["12spring24_q02"]["fillable_simple"] == "false"
    assert rows["12spring24_q02"]["reason"] == "detected_marks_do_not_sum_to_total"
    assert rows["12spring24_q03"]["likely_nested"] == "true"
    assert rows["12spring24_q03"]["reason"] == "detected_marks_count_does_not_equal_subpart_count"
    assert rows["12spring24_q04"]["reason"] == "subpart_marks_already_populated"


def test_readiness_audit_baseline_comparison_tracks_ids_text_status_and_tiers(tmp_path: Path) -> None:
    exact_only_baseline = _clone_record("12spring24_q01", "1")
    exact_only_current = _clone_record("12spring24_q01", "1")
    exact_only_baseline["question_text"] = "1 Find x. [2]"
    exact_only_current["question_text"] = "1   Find   x.   [2]"

    improved_baseline = _clone_record("12spring24_q02", "2")
    improved_baseline["question_text"] = "2 Findthevalueofx. [2]"
    improved_baseline["notes"].update(
        {
            "mapping_status": "review",
            "validation_status": "review",
            "question_text_trust": "low",
            "text_only_status": "fail",
            "visual_curation_status": "review",
            "text_fidelity_status": "degraded",
            "ocr_selected": False,
            "text_candidate_decision": "native_retained",
        }
    )
    improved_baseline["question_text_trust"] = "low"
    improved_baseline["text_only_status"] = "fail"
    improved_baseline["visual_curation_status"] = "review"

    improved_current = _clone_record("12spring24_q02", "2")
    improved_current["question_text"] = "2 Find the value of x. [2]"
    improved_current["ocr_text"] = improved_current["question_text"]
    improved_current["notes"].update(
        {
            "ocr_selected": True,
            "text_candidate_source": "ocr",
            "text_candidate_decision": "ocr_selected",
            "native_text_score": 5,
            "ocr_text_score": 60,
            "selected_text_score": 60,
        }
    )

    tier_baseline = _set_note(_clone_record("12spring24_q03", "3"), mapping_status="fail", validation_status="fail")
    tier_current = _clone_record("12spring24_q03", "3")
    added = _clone_record("12spring24_q04", "4")
    removed = _clone_record("12spring24_q05", "5")

    out_dir, summary = _run_audit(
        tmp_path,
        [exact_only_current, improved_current, tier_current, added],
        baseline_records=[exact_only_baseline, improved_baseline, tier_baseline, removed],
    )
    rows = _rows_by_id(_read_csv(out_dir / "baseline_comparison.csv"))
    baseline_summary = json.loads((out_dir / "baseline_comparison_summary.json").read_text(encoding="utf-8"))

    assert baseline_summary["records_added"] == 1
    assert baseline_summary["records_removed"] == 1
    assert baseline_summary["records_present_in_both"] == 3
    assert baseline_summary["field_change_counts"]["question_text"] == 1
    assert baseline_summary["field_change_counts"]["ocr_selected"] == 1
    assert baseline_summary["field_change_counts"]["text_candidate_decision"] == 1
    assert baseline_summary["field_change_counts"]["mapping_status"] == 2
    assert baseline_summary["field_change_counts"]["validation_status"] == 2
    assert baseline_summary["improved_status_counts"]["mapping_status"] == 2
    assert baseline_summary["improved_status_counts"]["validation_status"] == 2
    assert baseline_summary["improved_status_counts"]["question_text_trust"] == 1
    assert baseline_summary["asterion_tier_change_counts"]["improved"] == 2
    assert summary["baseline_comparison"]["records_added"] == 1

    assert rows["12spring24_q01"]["exact_question_text_changed"] == "true"
    assert rows["12spring24_q01"]["normalized_question_text_changed"] == "false"
    assert rows["12spring24_q02"]["normalized_question_text_changed"] == "true"
    assert "question_text" in _json_list(rows["12spring24_q02"]["changed_fields"])
    assert "ocr_selected" in _json_list(rows["12spring24_q02"]["changed_fields"])
    assert rows["12spring24_q03"]["tier_movement"] == "improved"
    assert rows["12spring24_q04"]["record_status"] == "added"
    assert rows["12spring24_q05"]["record_status"] == "removed"


def test_readiness_audit_outputs_are_deterministic_for_same_fixture(tmp_path: Path) -> None:
    records = [
        _clone_record("12spring24_q03", "3"),
        _set_note(_clone_record("12spring24_q01", "1"), mapping_status="review", validation_status="review"),
        _set_note(_clone_record("12spring24_q02", "2"), mapping_status="fail", validation_status="fail"),
    ]
    input_path = tmp_path / "question_bank.json"
    out_dir = tmp_path / "audit"
    _write_bank(input_path, records)

    assert main(["--input", str(input_path), "--out-dir", str(out_dir)]) == 0
    report_names = [
        "audit_summary.json",
        "field_presence_report.csv",
        "ocr_candidate_audit.csv",
        "readiness_tiers.csv",
        "hard_blockers.csv",
        "mapping_validation_report.csv",
        "subpart_marks_report.csv",
        "representative_review_sample.csv",
    ]
    first_run = {name: (out_dir / name).read_bytes() for name in report_names}

    assert main(["--input", str(input_path), "--out-dir", str(out_dir)]) == 0
    second_run = {name: (out_dir / name).read_bytes() for name in report_names}

    assert second_run == first_run
