from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from exam_bank.asterion_export import (
    ASTERION_EXPORT_FILENAME,
    ASTERION_SCHEMA_NAME,
    ASTERION_SCHEMA_VERSION,
    CONTENT_LAB_EXPORT_FILENAME,
    CONTENT_LAB_SCHEMA_NAME,
    CONTENT_LAB_SCHEMA_VERSION,
    MARK_EVENT_TOTAL_DISAGREEMENT,
    build_asterion_export,
    build_content_lab_candidates,
    export_asterion_content_lab_candidates,
    export_asterion_question_bank,
    load_skill_mappings,
)


def test_asterion_projection_is_conservative_for_12spring21_fixtures(tmp_path: Path) -> None:
    artifact_root = _write_artifacts(tmp_path)
    bank = _question_bank_fixture()

    payload = build_asterion_export(
        bank,
        artifact_root=artifact_root,
        base_dir=tmp_path,
        skill_mappings={"12spring21_q01_a": ["9709_p1_series_binomial_positive_integer"]},
    )

    assert payload["schema_name"] == ASTERION_SCHEMA_NAME
    assert payload["schema_version"] == ASTERION_SCHEMA_VERSION
    assert payload["record_count"] == 2

    by_id = {record["question_id"]: record for record in payload["questions"]}
    q1 = by_id["12spring21_q01"]
    q2 = by_id["12spring21_q02"]

    assert "notes" not in q1
    assert "ocr_text" not in q1
    assert "review_flags" not in q1
    assert q1["canonical_question_artifact"] == "p1/12spring21/questions/q01.png"
    assert q1["canonical_mark_scheme_artifact"] == "p1/12spring21/mark_scheme/q01.png"
    assert q1["artifact_integrity"]["question_images"][0]["sha256"] == _sha256(b"question-one")
    assert q1["artifact_integrity"]["mark_scheme_images"][0]["sha256"] == _sha256(b"mark-one")
    assert q1["artifact_integrity"]["question_images"][0]["exists"] is True
    assert q1["source_pdf"]["question_paper"]["sha256"] == _sha256(b"question-paper")
    assert q1["source_pdf"]["mark_scheme"]["sha256"] == _sha256(b"mark-scheme")
    assert q1["source_pdf"]["question_paper"]["exists"] is True

    assert q1["quality_gate"]["canonical_assets_ok"] is True
    assert q1["quality_gate"]["question_crop_ok"] is False
    assert q1["quality_gate"]["mark_scheme_crop_ok"] is True
    assert q1["quality_gate"]["marks_consistent"] is True
    assert q1["quality_gate"]["paper_total_consistent"] is True
    assert q1["quality_gate"]["text_only_display_allowed"] is False
    assert q1["quality_gate"]["content_lab_generation_allowed"] is False
    assert "text_only_blocked_untrusted_math_text" in q1["quality_gate"]["reason_codes"]
    assert "text_only_blocked_status_fail" in q1["quality_gate"]["reason_codes"]
    assert "content_lab_blocked_topic_confidence_low" in q1["quality_gate"]["reason_codes"]
    assert "content_lab_blocked_topic_uncertain" in q1["quality_gate"]["reason_codes"]

    assert q1["usage_roles"] == {
        "canonical_practice": "block",
        "field_guide_source": "block_until_reviewed",
        "quick_check_source": "block_until_reviewed",
        "warmup_generator_source": "block_until_reviewed",
        "guardian_candidate": "block",
        "p3_readiness_metric": "exclude",
    }
    assert [subpart["subpart_id"] for subpart in q1["subparts"]] == [
        "12spring21_q01_a",
        "12spring21_q01_b",
        "12spring21_q01_c",
    ]
    assert [subpart["marks"] for subpart in q1["subparts"]] == [1, 2, 2]
    assert q1["subparts"][0]["question_text"]["trust_level"] == "low"
    assert q1["subparts"][0]["question_text"]["role"] == "untrusted_math_text"
    assert q1["subparts"][0]["question_text"]["text_only_display_allowed"] is False
    assert q1["subparts"][0]["question_text"]["text"].startswith("(a) Find the first three terms")
    assert q1["subparts"][0]["mark_scheme_text"]["text"].startswith("(a) 1 + 5x")
    assert q1["subparts"][0]["review_status"] == "review"
    assert q1["subparts"][0]["mark_events"] == [
        {
            "subpart_id": "12spring21_q01_a",
            "mark_code": "B1",
            "mark_type": "independent_fact",
            "student_action": "1 + 5x + 10x^2",
            "answer_target": "1 + 5x + 10x^2",
            "dependency": None,
            "skill_ids": ["9709_p1_series_binomial_positive_integer"],
            "common_errors": [],
            "evidence_text": "(a) 1 + 5x + 10x^2 B1 1",
            "confidence": 0.88,
            "review_status": "machine_candidate",
        }
    ]
    assert q1["subparts"][1]["mark_events"][0]["mark_code"] == "B2"
    assert q1["subparts"][1]["mark_events"][0]["mark_type"] == "independent_fact"
    assert q1["subparts"][2]["mark_events"][0]["mark_code"] == "A1"
    assert q1["subparts"][2]["mark_events"][0]["mark_type"] == "accuracy"

    assert q2["total_marks"] == 4
    assert q2["quality_gate"]["text_only_display_allowed"] is False
    assert q2["quality_gate"]["content_lab_generation_allowed"] is False
    assert len(q2["subparts"]) == 1
    assert q2["subparts"][0]["subpart_id"] == "12spring21_q02_whole"
    assert q2["subparts"][0]["label"] == "whole"
    assert q2["subparts"][0]["marks"] == 4
    assert q2["subparts"][0]["detected_mark_values"] == [4]
    assert [event["mark_code"] for event in q2["subparts"][0]["mark_events"]] == ["M1", "M1", "A1", "A1"]
    assert [event["mark_type"] for event in q2["subparts"][0]["mark_events"]] == [
        "method",
        "method",
        "accuracy",
        "accuracy",
    ]


def test_mark_events_handle_follow_through_subpart_totals_quarantine_and_degraded_text(tmp_path: Path) -> None:
    artifact_root = _write_artifacts(tmp_path)
    bank = _question_bank_fixture()

    q3 = _base_spring21_record("12spring21_q03", "3", 3)
    q3["question_text"] = "3 (a) Differentiate y = x^2 + 3x. [2] (b) Use your gradient to find a tangent. [1]"
    q3["mark_scheme_text"] = "3(a) Differentiate x^2 + 3x correctly M1\nObtain 2x + 3 A1\n3(b) Correct tangent from their gradient A1ft"
    q3["subparts"] = ["a", "b"]
    q3["subparts_solution_marks"] = {"a": None, "b": None}
    q3["notes"]["question_structure_detected"]["subparts"] = ["a", "b"]
    q3["notes"]["question_structure_detected"]["mark_values_detected"] = [2, 1]
    q3["notes"]["mark_scheme_structure_detected"]["subparts"] = ["a", "b"]

    q4 = _base_spring21_record("12spring21_q04", "4", 4)
    q4["question_text"] = "4 Solve the equation. [4]"
    q4["mark_scheme_text"] = "4 Obtain x = 2 M1 A1"
    q4["notes"]["mark_scheme_total_detected"] = 3
    q4["notes"]["mark_scheme_structure_detected"]["mark_scheme_total_detected"] = 3

    q5 = _base_spring21_record("12spring21_q05", "5", 1)
    q5["question_text"] = "5 State the value. [1]"
    q5["mark_scheme_text"] = "5 ??? ||| M1"

    bank["questions"].extend([q3, q4, q5])
    bank["record_count"] = len(bank["questions"])

    payload = build_asterion_export(bank, artifact_root=artifact_root, base_dir=tmp_path)
    by_id = {record["question_id"]: record for record in payload["questions"]}

    q3_out = by_id["12spring21_q03"]
    assert [subpart["marks"] for subpart in q3_out["subparts"]] == [2, 1]
    assert [event["mark_code"] for event in q3_out["subparts"][0]["mark_events"]] == ["M1", "A1"]
    assert q3_out["subparts"][1]["mark_events"][0]["mark_code"] == "A1FT"
    assert q3_out["subparts"][1]["mark_events"][0]["mark_type"] == "follow_through"
    assert q3_out["subparts"][1]["mark_events"][0]["dependency"] == "follow_through_from_previous_work"

    q4_events = by_id["12spring21_q04"]["subparts"][0]["mark_events"]
    assert [event["review_status"] for event in q4_events] == ["quarantined", "quarantined"]
    assert {event["quarantine_reason"] for event in q4_events} == {MARK_EVENT_TOTAL_DISAGREEMENT}

    assert by_id["12spring21_q05"]["subparts"][0]["mark_events"] == []


def test_asterion_export_writes_sidecar_without_mutating_question_bank(tmp_path: Path) -> None:
    artifact_root = _write_artifacts(tmp_path)
    input_path = tmp_path / "output_ocr_candidate" / "json" / "question_bank.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(json.dumps(_question_bank_fixture(), indent=2), encoding="utf-8")
    original = input_path.read_text(encoding="utf-8")

    output_path = export_asterion_question_bank(input_path, artifact_root=artifact_root, base_dir=tmp_path)

    assert output_path == tmp_path / "output_ocr_candidate" / "asterion" / "exports" / "latest" / ASTERION_EXPORT_FILENAME
    assert input_path.read_text(encoding="utf-8") == original
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_name"] == ASTERION_SCHEMA_NAME
    assert payload["record_count"] == 2
    assert [record["question_id"] for record in payload["questions"]] == ["12spring21_q01", "12spring21_q02"]


def test_asterion_export_loads_optional_skill_map_for_mark_events(tmp_path: Path) -> None:
    skill_map_path = tmp_path / "skill_map.json"
    skill_map_path.write_text(
        json.dumps(
            {
                "mappings": [
                    {
                        "subpart_id": "12spring21_q01_a",
                        "primary_skill_ids": ["primary"],
                        "secondary_skill_ids": ["secondary"],
                        "prerequisite_skill_ids": ["primary", "prerequisite"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_skill_mappings(skill_map_path) == {"12spring21_q01_a": ["primary", "secondary", "prerequisite"]}

    artifact_root = _write_artifacts(tmp_path)
    input_path = tmp_path / "output_ocr_candidate" / "json" / "question_bank.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(json.dumps(_question_bank_fixture(), indent=2), encoding="utf-8")
    output_path = tmp_path / "asterion.json"

    export_asterion_question_bank(input_path, output_path, artifact_root=artifact_root, base_dir=tmp_path, skill_map_path=skill_map_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    q1 = payload["questions"][0]
    assert q1["subparts"][0]["mark_events"][0]["skill_ids"] == ["primary", "secondary", "prerequisite"]


def test_content_lab_candidates_emit_roles_and_block_warmup_until_reviewed(tmp_path: Path) -> None:
    artifact_root = _write_artifacts(tmp_path)
    bank = _question_bank_fixture()
    asterion = build_asterion_export(
        bank,
        artifact_root=artifact_root,
        base_dir=tmp_path,
        skill_mappings={"12spring21_q01_a": ["9709_p1_series_binomial_positive_integer"]},
    )

    payload = build_content_lab_candidates(asterion)

    assert payload["schema_name"] == CONTENT_LAB_SCHEMA_NAME
    assert payload["schema_version"] == CONTENT_LAB_SCHEMA_VERSION
    assert payload["policy"] == {
        "no_student_facing_generated_content": True,
        "emits_candidates_and_metadata_only": True,
        "content_lab_generation_requires_reviewed_or_approved_mapping": True,
        "content_lab_generation_requires_reviewed_or_approved_mark_events": True,
    }
    by_subpart = {candidate["subpart_id"]: candidate for candidate in payload["candidates"]}
    candidate = by_subpart["12spring21_q01_a"]
    assert "generated_content" not in candidate
    assert set(candidate["possible_content_lab_roles"]) >= {
        "generated_warmup_pattern_source",
        "prerequisite_repair_source",
        "mixed_review_source",
    }
    assert candidate["role_statuses"]["field_guide_source"] == "block"
    assert candidate["role_statuses"]["quick_check_source"] == "block"
    assert candidate["role_statuses"]["generated_warmup_pattern_source"] == "blocked_until_reviewed"
    assert candidate["generation_gate"]["blocked"] is True
    assert "mark_events_not_reviewed_or_approved" in candidate["generation_gate"]["block_reasons"]
    assert "mapping_or_subpart_not_reviewed_or_approved" in candidate["generation_gate"]["block_reasons"]
    warmup = candidate["generated_warmup_pattern_source"]
    assert warmup["method_pattern_id"] == "9709_p1_series_binomial_positive_integer__independent_fact"
    assert warmup["source_skill_ids"] == ["9709_p1_series_binomial_positive_integer"]
    assert [event["mark_code"] for event in warmup["source_mark_events"]] == ["B1"]
    assert warmup["suggested_generator_family"] == "fact_or_result_check"
    assert "generation_blocked_until_mapping_and_mark_events_reviewed" in warmup["required_parameter_constraints"]
    assert warmup["common_errors_to_target"] == []
    assert warmup["review_status"] == "blocked_until_reviewed"


def test_content_lab_candidates_allow_warmup_only_after_mapping_and_mark_event_approval(tmp_path: Path) -> None:
    artifact_root = _write_artifacts(tmp_path)
    asterion = build_asterion_export(
        _question_bank_fixture(),
        artifact_root=artifact_root,
        base_dir=tmp_path,
        skill_mappings={"12spring21_q02_whole": ["9709_p1_algebra_quadratics"]},
    )
    q2 = next(record for record in asterion["questions"] if record["question_id"] == "12spring21_q02")
    q2["quality_gate"]["content_lab_generation_allowed"] = True
    q2["usage_roles"] = {
        "canonical_practice": "allow",
        "field_guide_source": "allow",
        "quick_check_source": "allow",
        "warmup_generator_source": "allow",
        "guardian_candidate": "allow",
        "p3_readiness_metric": "exclude",
    }
    q2["subparts"][0]["review_status"] = "approved"
    for event in q2["subparts"][0]["mark_events"]:
        event["review_status"] = "approved"

    payload = build_content_lab_candidates(asterion)
    candidate = next(item for item in payload["candidates"] if item["subpart_id"] == "12spring21_q02_whole")

    assert candidate["generation_gate"] == {"status": "allow", "blocked": False, "block_reasons": []}
    assert candidate["role_statuses"]["generated_warmup_pattern_source"] == "allow"
    assert candidate["role_statuses"]["field_guide_source"] == "allow"
    assert candidate["role_statuses"]["guardian_candidate"] == "allow"
    assert candidate["generated_warmup_pattern_source"]["review_status"] == "ready"
    assert candidate["generated_warmup_pattern_source"]["suggested_generator_family"] == "worked_method_variation"
    assert "generation_blocked_until_mapping_and_mark_events_reviewed" not in candidate["generated_warmup_pattern_source"][
        "required_parameter_constraints"
    ]


def test_content_lab_candidates_export_writes_sidecar_from_question_bank(tmp_path: Path) -> None:
    artifact_root = _write_artifacts(tmp_path)
    skill_map_path = tmp_path / "skill_map.json"
    skill_map_path.write_text(
        json.dumps({"mappings": [{"subpart_id": "12spring21_q01_a", "primary_skill_ids": ["skill-one"]}]}),
        encoding="utf-8",
    )
    input_path = tmp_path / "output_ocr_candidate" / "json" / "question_bank.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(json.dumps(_question_bank_fixture(), indent=2), encoding="utf-8")

    output_path = export_asterion_content_lab_candidates(
        input_path,
        artifact_root=artifact_root,
        base_dir=tmp_path,
        skill_map_path=skill_map_path,
    )

    assert output_path == tmp_path / "output_ocr_candidate" / "asterion" / "exports" / "latest" / CONTENT_LAB_EXPORT_FILENAME
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_name"] == CONTENT_LAB_SCHEMA_NAME
    assert payload["record_count"] >= 1
    q1a = next(candidate for candidate in payload["candidates"] if candidate["subpart_id"] == "12spring21_q01_a")
    assert q1a["source_skill_ids"] == ["skill-one"]


def _write_artifacts(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "output_ocr_candidate"
    files = {
        artifact_root / "p1" / "12spring21" / "questions" / "q01.png": b"question-one",
        artifact_root / "p1" / "12spring21" / "questions" / "q02.png": b"question-two",
        artifact_root / "p1" / "12spring21" / "questions" / "q03.png": b"question-three",
        artifact_root / "p1" / "12spring21" / "questions" / "q04.png": b"question-four",
        artifact_root / "p1" / "12spring21" / "questions" / "q05.png": b"question-five",
        artifact_root / "p1" / "12spring21" / "mark_scheme" / "q01.png": b"mark-one",
        artifact_root / "p1" / "12spring21" / "mark_scheme" / "q02.png": b"mark-two",
        artifact_root / "p1" / "12spring21" / "mark_scheme" / "q03.png": b"mark-three",
        artifact_root / "p1" / "12spring21" / "mark_scheme" / "q04.png": b"mark-four",
        artifact_root / "p1" / "12spring21" / "mark_scheme" / "q05.png": b"mark-five",
        tmp_path / "input" / "question_papers" / "9709 Mathematics March 2021 Question paper  12.pdf": b"question-paper",
        tmp_path / "input" / "mark_schemes" / "9709 Mathematics March 2021 Mark Scheme  12.pdf": b"mark-scheme",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return artifact_root


def _question_bank_fixture() -> dict:
    q1 = _base_spring21_record("12spring21_q01", "1", 5)
    q1["question_text"] = (
        "1 (a) Find the first three terms in the expansion of (1 + x)^5. [1] "
        "(b) Find the first three terms in the expansion of (1 - 2x)^6. [2] "
        "(c) Hence find the coefficient of x^2 in the product. [2]"
    )
    q1["mark_scheme_text"] = "1(a) 1 + 5x + 10x^2 B1\n1(b) 1 - 12x + 60x^2 B2\n1(c) 10 A1"
    q1["subparts"] = ["a", "b", "c"]
    q1["subparts_solution_marks"] = {"a": None, "b": None, "c": None}
    q1["notes"]["question_structure_detected"]["subparts"] = ["a", "b", "c"]
    q1["notes"]["question_structure_detected"]["mark_values_detected"] = [1, 2, 2]
    q1["notes"]["mark_scheme_structure_detected"]["subparts"] = ["a", "b", "c"]

    q2 = _base_spring21_record("12spring21_q02", "2", 4)
    q2["question_text"] = "2 By using a suitable substitution, solve the equation (2x - 3)^2 - (2x - 43)^2 - 3 = 0. [4]"
    q2["mark_scheme_text"] = "2 u = 2x - 3 leading to u^4 - 3u^2 - 4 = 0 M1 M1 A1 A1"
    q2["subparts"] = []
    q2["subparts_solution_marks"] = {}
    q2["notes"]["question_structure_detected"]["mark_values_detected"] = [4]

    return {
        "schema_name": "exam_bank.question_bank",
        "schema_version": 2,
        "record_count": 2,
        "questions": [q1, q2],
    }


def _base_spring21_record(question_id: str, question_number: str, marks: int) -> dict:
    path_number = f"q{int(question_number):02d}.png"
    record = {
        "question_id": question_id,
        "paper": "12spring21",
        "paper_family": "p1",
        "question_number": question_number,
        "canonical_question_artifact": f"p1/12spring21/questions/{path_number}",
        "question_image_path": f"p1/12spring21/questions/{path_number}",
        "mark_scheme_image_path": f"p1/12spring21/mark_scheme/{path_number}",
        "question_image_paths": [f"p1/12spring21/questions/{path_number}"],
        "mark_scheme_image_paths": [f"p1/12spring21/mark_scheme/{path_number}"],
        "question_text_role": "untrusted_math_text",
        "question_text_trust": "low",
        "visual_required": True,
        "visual_curation_status": "review",
        "text_only_status": "fail",
        "question_solution_marks": marks,
        "notes": {
            "source_pdf": "input/question_papers/9709 Mathematics March 2021 Question paper  12.pdf",
            "mark_scheme_source_pdf": "input/mark_schemes/9709 Mathematics March 2021 Mark Scheme  12.pdf",
            "mapping_status": "pass",
            "mapping_failure_reason": "",
            "validation_status": "pass",
            "validation_flags": [],
            "scope_quality_status": "clean",
            "question_crop_confidence": "low",
            "mark_scheme_crop_confidence": "high",
            "question_text_role": "untrusted_math_text",
            "question_text_trust": "low",
            "visual_required": True,
            "visual_curation_status": "review",
            "text_only_status": "fail",
            "topic_confidence": "low",
            "topic_uncertain": True,
            "question_total_detected": marks,
            "mark_scheme_total_detected": marks,
            "paper_total_status": "matched",
            "question_structure_detected": {
                "subparts": [],
                "mark_values_detected": [marks],
                "question_total_detected": marks,
                "contamination_detected": False,
                "likely_truncated": False,
            },
            "mark_scheme_structure_detected": {
                "subparts": [],
                "question_total_detected": marks,
                "mark_scheme_total_detected": marks,
            },
        },
    }
    return deepcopy(record)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
