from __future__ import annotations

import json
from pathlib import Path

from exam_bank.audit import (
    audit_current_output_integrity,
    audit_difficulty,
    audit_ocr_candidates,
    audit_question_bank,
    write_audit,
    write_current_output_integrity_audit,
    write_difficulty_audit,
    write_ocr_candidate_audit,
)
from exam_bank.cli import main as cli_main


def test_visual_first_audit_reports_text_and_curation_distributions(tmp_path: Path) -> None:
    records = [
        {
            "question_id": "12spring24_q01",
            "question_number": "1",
            "question_text": "Sketch the graph of y x 3 6 = -.",
            "question_text_role": "untrusted_math_text",
            "question_text_trust": "low",
            "visual_required": True,
            "visual_reason_flags": ["contains_graph_or_diagram_prompt", "contains_math_text_corruption"],
            "visual_curation_status": "ready",
            "text_only_status": "fail",
            "question_image_path": "p1/12spring24/questions/q01.png",
            "notes": {"text_fidelity_status": "degraded"},
        },
        {
            "question_id": "12spring24_q02",
            "question_number": "2",
            "question_text": "A committee has 5 members chosen from 8 people.",
            "question_text_role": "readable_text",
            "question_text_trust": "high",
            "visual_required": False,
            "visual_reason_flags": [],
            "visual_curation_status": "ready",
            "text_only_status": "ready",
            "notes": {"text_fidelity_status": "clean"},
        },
        {
            "question_id": "12spring24_q03",
            "question_number": "3",
            "question_text": "Solve tan x = 1.",
            "question_text_role": "search_hint",
            "question_text_trust": "medium",
            "visual_required": True,
            "visual_reason_flags": ["contains_trig_expression", "contains_equation_layout"],
            "visual_curation_status": "review",
            "text_only_status": "review",
            "notes": {"text_fidelity_status": "clean"},
        },
    ]

    report = audit_question_bank(records)

    assert report["question_text_role_counts"] == {
        "readable_text": 1,
        "search_hint": 1,
        "untrusted_math_text": 1,
    }
    assert report["question_text_trust_counts"] == {"high": 1, "low": 1, "medium": 1}
    assert report["visual_required_counts"] == {"false": 1, "true": 2}
    assert report["visual_curation_status_counts"] == {"ready": 2, "review": 1}
    assert report["text_only_status_counts"] == {"fail": 1, "ready": 1, "review": 1}
    assert report["visual_reason_flag_counts"]["contains_equation_layout"] == 1
    assert report["examples_clean_text_fidelity_but_visual_required"][0]["question_id"] == "12spring24_q03"

    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "audit.json"
    input_path.write_text(json.dumps({"questions": records}), encoding="utf-8")
    written = write_audit(input_path, output_path)

    assert written["record_count"] == 3
    assert json.loads(output_path.read_text(encoding="utf-8"))["record_count"] == 3


def test_ocr_candidate_audit_reports_missing_metadata_counts_and_statuses() -> None:
    records = [
        {
            "question_id": "12spring24_q01",
            "question_number": "1",
            "question_text": "1 Find x.",
            "question_text_trust": "low",
            "text_only_status": "fail",
            "visual_curation_status": "review",
            "notes": {
                "text_fidelity_status": "degraded",
                "ocr_rejected_reasons": ["ocr_not_clearly_better", "page_furniture_or_header_text"],
            },
        },
        {
            "question_id": "12spring24_q02",
            "question_number": "2",
            "question_text": "2 Find y.",
            "question_text_trust": "high",
            "text_only_status": "ready",
            "visual_curation_status": "ready",
            "notes": {
                "text_fidelity_status": "clean",
                "text_candidate_source": "ocr",
                "text_candidate_decision": "ocr_selected",
                "ocr_selected": True,
                "native_text_score": 10,
                "ocr_text_score": 40,
                "selected_text_score": 40,
                "ocr_rejected_reasons": ["ocr_not_clearly_better"],
            },
        },
    ]

    report = audit_ocr_candidates(records)

    assert report["candidate_metadata_presence"]["text_candidate_source"] == {"present": 1, "missing": 1}
    assert report["candidate_metadata_presence"]["native_text_score"] == {"present": 1, "missing": 1}
    assert "candidate_metadata_partially_populated:text_candidate_source,text_candidate_decision,ocr_selected,native_text_score,ocr_text_score,selected_text_score" in report[
        "data_quality_findings"
    ]
    assert report["ocr_selected_count"] == 1
    assert report["text_candidate_source_counts"] == {"missing": 1, "ocr": 1}
    assert report["text_candidate_decision_counts"] == {"missing": 1, "ocr_selected": 1}
    assert report["ocr_selected_counts"] == {"missing": 1, "true": 1}
    assert report["ocr_rejected_reason_counts"] == {
        "ocr_not_clearly_better": 2,
        "page_furniture_or_header_text": 1,
    }
    assert report["score_summaries"]["ocr_text_score"] == {
        "count": 1,
        "min": 40,
        "p25": 40,
        "median": 40,
        "p75": 40,
        "max": 40,
        "mean": 40,
    }
    assert report["text_fidelity_status_counts"] == {"clean": 1, "degraded": 1}
    assert report["text_only_status_counts"] == {"fail": 1, "ready": 1}
    assert report["visual_curation_status_counts"] == {"ready": 1, "review": 1}
    assert report["question_text_trust_counts"] == {"high": 1, "low": 1}


def test_ocr_candidate_audit_flags_stale_export_and_ignores_null_scores() -> None:
    report = audit_ocr_candidates(
        [
            {
                "question_id": "12spring24_q01",
                "question_text": "1 Find x.",
                "notes": {
                    "text_candidate_source": None,
                    "text_candidate_decision": "",
                    "ocr_selected": None,
                    "native_text_score": None,
                    "ocr_text_score": None,
                    "selected_text_score": None,
                },
            }
        ]
    )

    assert report["data_quality_findings"] == [
        "candidate_metadata_missing_for_all_records:text_candidate_source,text_candidate_decision,ocr_selected,native_text_score,ocr_text_score,selected_text_score",
        "stale_or_candidate_unaware_export",
    ]
    assert report["score_summaries"]["native_text_score"]["count"] == 0
    assert report["score_summaries"]["native_text_score"]["mean"] is None


def test_ocr_candidate_audit_compares_baseline_and_status_movement() -> None:
    baseline = [
        {
            "question_id": "12spring24_q01",
            "question_text": "1 Find x from the diagram.",
            "question_text_trust": "low",
            "text_only_status": "fail",
            "visual_curation_status": "review",
            "notes": {"mapping_status": "fail", "validation_status": "review", "text_fidelity_status": "degraded"},
        },
        {
            "question_id": "12spring24_q02",
            "question_text": "2 Find y.",
            "question_text_trust": "high",
            "text_only_status": "ready",
            "visual_curation_status": "ready",
            "notes": {"mapping_status": "pass", "validation_status": "pass"},
        },
    ]
    current = [
        {
            "question_id": "12spring24_q01",
            "question_text": "1 Find x.",
            "question_text_trust": "high",
            "text_only_status": "ready",
            "visual_curation_status": "ready",
            "notes": {"mapping_status": "pass", "validation_status": "pass", "text_fidelity_status": "clean"},
        },
        {
            "question_id": "12spring24_q02",
            "question_text": "2 Find y.",
            "question_text_trust": "medium",
            "text_only_status": "review",
            "visual_curation_status": "ready",
            "notes": {"mapping_status": "pass", "validation_status": "fail"},
        },
        {
            "question_id": "12spring24_q03",
            "question_text": "3 Find z.",
        },
    ]

    comparison = audit_ocr_candidates(current, baseline_records=baseline)["baseline_comparison"]

    assert comparison["available"] is True
    assert comparison["shared_record_count"] == 2
    assert comparison["new_in_current_count"] == 1
    assert comparison["missing_from_current_count"] == 0
    assert comparison["field_change_counts"]["question_text"] == 1
    assert comparison["field_change_counts"]["question_text_trust"] == 2
    assert comparison["field_change_counts"]["text_only_status"] == 2
    assert comparison["field_change_counts"]["validation_status"] == 2
    assert comparison["improved_records"] == [
        {"question_id": "12spring24_q01", "fields": ["text_only_status", "visual_curation_status", "question_text_trust", "validation_status", "mapping_status"]}
    ]
    assert comparison["worsened_records"] == [
        {"question_id": "12spring24_q02", "fields": ["text_only_status", "question_text_trust", "validation_status"]}
    ]


def test_ocr_candidate_audit_flags_suspicious_selected_records_and_readiness_inflation() -> None:
    report = audit_ocr_candidates(
        [
            {
                "question_id": "12spring24_q01",
                "paper_family": "p1",
                "question_number": "",
                "question_text": "Cambridge International. Find x.",
                "question_solution_marks": None,
                "question_text_trust": "low",
                "text_only_status": "ready",
                "visual_curation_status": "ready",
                "notes": {
                    "ocr_selected": True,
                    "text_candidate_source": "ocr",
                    "text_candidate_decision": "ocr_selected",
                    "native_text_score": 5,
                    "ocr_text_score": 45,
                    "selected_text_score": 45,
                    "scope_quality_status": "fail",
                    "text_fidelity_status": "degraded",
                    "validation_status": "fail",
                },
            }
        ]
    )

    suspicious = report["suspicious_ocr_selected_records"]
    assert suspicious[0]["question_id"] == "12spring24_q01"
    assert set(suspicious[0]["risk_reasons"]) >= {
        "scope_quality_failed",
        "selected_text_not_clean",
        "possible_page_furniture_text",
        "missing_question_number",
        "missing_marks",
        "selected_with_hard_failure",
    }
    assert report["readiness_inflation_risk_records"][0]["risk_reasons"] == [
        "text_only_ready_without_high_trust",
        "text_only_ready_without_clean_fidelity",
        "visual_ready_without_clean_scope",
    ]


def test_write_ocr_candidate_audit_writes_deterministic_json(tmp_path: Path) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "ocr_audit.json"
    input_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": "12spring24_q01",
                        "question_text": "1 Find x.",
                        "question_text_trust": "high",
                        "text_only_status": "ready",
                        "visual_curation_status": "ready",
                        "notes": {
                            "text_fidelity_status": "clean",
                            "text_candidate_source": "native",
                            "text_candidate_decision": "native_retained",
                            "ocr_selected": False,
                            "native_text_score": 30,
                            "ocr_text_score": 20,
                            "selected_text_score": 30,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = write_ocr_candidate_audit(input_path, output_path=output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written == report
    assert list(written) == [
        "record_count",
        "candidate_metadata_presence",
        "data_quality_findings",
        "ocr_selected_count",
        "text_candidate_source_counts",
        "text_candidate_decision_counts",
        "ocr_selected_counts",
        "ocr_rejected_reason_counts",
        "score_summaries",
        "text_fidelity_status_counts",
        "text_only_status_counts",
        "visual_curation_status_counts",
        "question_text_trust_counts",
        "suspicious_ocr_selected_records",
        "readiness_inflation_risk_records",
        "representative_records",
        "baseline_comparison",
    ]


def test_difficulty_audit_reports_distribution_and_missing_metadata(tmp_path: Path) -> None:
    records = [
        {
            "question_id": "12spring24_q01",
            "paper_family": "p1",
            "topic": "differentiation",
            "difficulty": "easy",
            "difficulty_score": 18,
            "difficulty_band": "easy",
            "question_solution_marks": 2,
            "notes": {
                "difficulty_confidence": "high",
                "difficulty_evidence": "direct routine method",
                "difficulty_features": {"marks": {"marks": 2}},
                "difficulty_review_flags": [],
                "difficulty_model_version": "local-difficulty-v1",
            },
        },
        {
            "question_id": "32spring24_q09",
            "paper_family": "p3",
            "topic": "complex_numbers",
            "difficulty": "difficult",
            "difficulty_score": 73,
            "difficulty_band": "difficult",
            "question_solution_marks": 12,
            "notes": {
                "difficulty_confidence": "medium",
                "difficulty_evidence": "proof wording",
                "difficulty_features": {"topic_prior": {"difficult_topic_prior": True}},
                "difficulty_review_flags": ["marks_missing_for_difficulty"],
                "difficulty_model_version": "local-difficulty-v1",
            },
        },
        {"question_id": "52spring24_q03", "paper_family": "p5", "topic": "probability"},
    ]

    report = audit_difficulty(records)

    assert report["difficulty_label_counts"] == {"difficult": 1, "easy": 1, "missing": 1}
    assert report["difficulty_score_summary"]["count"] == 2
    assert report["difficulty_score_bucket_counts"] == {"0-34": 1, "70-100": 1, "missing": 1}
    assert report["difficulty_counts_by_paper_family"]["p3"] == {"difficult": 1}
    assert report["difficulty_counts_by_marks_bucket"]["10+"] == {"difficult": 1}
    assert report["difficulty_review_flag_counts"] == {"marks_missing_for_difficulty": 1}
    assert report["missing_difficulty_metadata"]["difficulty_score"] == 1

    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "difficulty_audit.json"
    input_path.write_text(json.dumps({"questions": records}), encoding="utf-8")
    written_report = write_difficulty_audit(input_path, output_path=output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == written_report


def test_current_output_integrity_allows_documented_missing_mark_scheme_companion(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    question_path = "p1/12spring24/questions/q01.png"
    mark_scheme_path = "p1/12spring24/mark_scheme/q01.png"
    known_missing_question_path = "p3/33autumn25/questions/q01.png"
    _write_file(artifact_root / question_path)
    _write_file(artifact_root / mark_scheme_path)
    _write_file(artifact_root / known_missing_question_path)

    records = [
        _integrity_record(
            "12spring24_q01",
            paper="12spring24",
            question_number="1",
            question_image_path=question_path,
            mark_scheme_image_path=mark_scheme_path,
        ),
        _integrity_record(
            "33autumn25_q01",
            paper="33autumn25",
            question_number="1",
            question_image_path=known_missing_question_path,
            mark_scheme_image_path="",
            source_pdf="input/question_papers/9709 Mathematics November 2025 Question Paper  33.pdf",
        ),
    ]
    input_path = artifact_root / "json" / "question_bank.json"
    output_path = tmp_path / "integrity.json"
    _write_integrity_bank(input_path, records, artifact_root=artifact_root)

    report = write_current_output_integrity_audit(input_path, output_path=output_path)

    assert report["ok"] is True
    assert report["counts"]["unique_question_id_count"] == 2
    assert report["counts"]["missing_question_image_path_count"] == 0
    assert report["counts"]["missing_question_image_file_count"] == 0
    assert report["counts"]["nonblank_mark_scheme_image_record_count"] == 1
    assert report["counts"]["allowed_missing_mark_scheme_image_path_count"] == 1
    assert report["counts"]["unexpected_missing_mark_scheme_image_path_count"] == 0
    assert report["known_missing_mark_scheme_companions"] == [
        {
            "source_companion": "9709_2025_November_33",
            "paper": "33autumn25",
            "reason": "The source mark scheme PDF for 9709 Mathematics November 2025 Paper 33 is missing.",
            "observed_missing_count": 1,
            "observed_question_ids": ["33autumn25_q01"],
        }
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert cli_main(["output-integrity-audit", "--input", str(input_path)]) == 0


def test_current_output_integrity_fails_on_new_missing_paths_and_duplicates(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    good_question_path = "p1/12spring24/questions/q01.png"
    good_mark_scheme_path = "p1/12spring24/mark_scheme/q01.png"
    _write_file(artifact_root / good_question_path)
    _write_file(artifact_root / good_mark_scheme_path)

    absolute_question_path = str((tmp_path / "outside.png").resolve())
    absolute_mark_scheme_path = str((tmp_path / "outside_mark_scheme.png").resolve())
    records = [
        _integrity_record(
            "12spring24_q01",
            paper="12spring24",
            question_number="1",
            question_image_path=good_question_path,
            mark_scheme_image_path=good_mark_scheme_path,
        ),
        _integrity_record(
            "12spring24_q01",
            paper="12spring24",
            question_number="1",
            question_image_path=absolute_question_path,
            mark_scheme_image_path=absolute_mark_scheme_path,
        ),
        _integrity_record(
            "12spring24_q03",
            paper="12spring24",
            question_number="3",
            question_image_path="",
            mark_scheme_image_path="",
        ),
    ]
    input_path = artifact_root / "json" / "question_bank.json"
    _write_integrity_bank(input_path, records, artifact_root=artifact_root)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["record_count"] = 999
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_current_output_integrity(input_path)

    assert report["ok"] is False
    assert {failure["code"] for failure in report["failures"]} >= {
        "record_count_mismatch",
        "duplicate_question_id",
        "duplicate_paper_question",
        "missing_question_image_path",
        "absolute_question_image_path",
        "missing_question_image_file",
        "missing_mark_scheme_image_path",
        "absolute_mark_scheme_image_path",
        "missing_mark_scheme_image_file",
    }
    assert report["checks"]["declared_record_count_matches"] is False
    assert report["counts"]["duplicate_question_id_value_count"] == 1
    assert report["counts"]["duplicate_paper_question_pair_count"] == 1
    assert report["counts"]["unexpected_missing_mark_scheme_image_path_count"] == 1
    assert cli_main(["output-integrity-audit", "--input", str(input_path)]) == 1


def _integrity_record(
    question_id: str,
    *,
    paper: str,
    question_number: str,
    question_image_path: str,
    mark_scheme_image_path: str,
    source_pdf: str = "input/question_papers/9709 Mathematics March 2024 Question Paper  12.pdf",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": paper,
        "paper_family": "p1",
        "question_number": question_number,
        "canonical_question_artifact": question_image_path,
        "question_image_path": question_image_path,
        "question_image_paths": [question_image_path] if question_image_path else [],
        "mark_scheme_image_path": mark_scheme_image_path,
        "mark_scheme_image_paths": [mark_scheme_image_path] if mark_scheme_image_path else [],
        "mark_scheme_text": "M1 A1" if mark_scheme_image_path else "",
        "notes": {
            "source_pdf": source_pdf,
            "mark_scheme_source_pdf": (
                "input/mark_schemes/9709 Mathematics March 2024 Mark Scheme  12.pdf" if mark_scheme_image_path else ""
            ),
        },
    }


def _write_integrity_bank(path: Path, records: list[dict[str, object]], *, artifact_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.question_bank",
                "schema_version": 2,
                "record_count": len(records),
                "run_manifest": {"artifact_root": str(artifact_root)},
                "questions": records,
            }
        ),
        encoding="utf-8",
    )


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake png")
