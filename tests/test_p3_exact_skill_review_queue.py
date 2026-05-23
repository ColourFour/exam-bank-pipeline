from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.p3_exact_skill.review_queue import (
    build_p3_exact_skill_review_queue,
    build_review_queue_item,
    classify_review_queue_item,
)


def test_assets_and_candidate_p3_skill_becomes_clean_candidate_not_clean() -> None:
    status, blockers, action = classify_review_queue_item(
        has_question_asset=True,
        has_mark_scheme_asset=True,
        p3_skill_ids=["9709_p3_3_2_log_exponential_equations"],
        non_p3_skill_ids=[],
        topic_route={"confidence": "high", "review_required": False},
        mapping={"evidence": {"topic_uncertain": False, "topic_confidence": "high"}},
        mark_event_record={"mark_events": [{"event_id": "e1"}], "safe_for_marking_use": False},
        quality_gate={},
    )

    assert status == "clean_candidate"
    assert status != "clean"
    assert action == "review_assets_and_skill"
    assert "mark_events_advisory_only" in blockers


def test_missing_question_asset_becomes_blocked_candidate() -> None:
    status, blockers, action = classify_review_queue_item(
        has_question_asset=False,
        has_mark_scheme_asset=True,
        p3_skill_ids=["9709_p3_3_2_log_exponential_equations"],
        non_p3_skill_ids=[],
    )

    assert status == "blocked_candidate"
    assert "missing_question_asset" in blockers
    assert action == "reject_missing_question_asset"


def test_missing_mark_scheme_asset_becomes_blocked_candidate() -> None:
    status, blockers, action = classify_review_queue_item(
        has_question_asset=True,
        has_mark_scheme_asset=False,
        p3_skill_ids=["9709_p3_3_2_log_exponential_equations"],
        non_p3_skill_ids=[],
    )

    assert status == "blocked_candidate"
    assert "missing_mark_scheme_asset" in blockers
    assert action == "reject_missing_mark_scheme_asset"


def test_ambiguous_routing_becomes_ambiguous_candidate() -> None:
    status, blockers, action = classify_review_queue_item(
        has_question_asset=True,
        has_mark_scheme_asset=True,
        p3_skill_ids=["9709_p3_3_2_log_exponential_equations"],
        non_p3_skill_ids=[],
        topic_route={"confidence": "low", "review_required": True},
        mapping={"evidence": {"topic_uncertain": True, "topic_confidence": "low"}},
        mark_event_record={"mark_events": [{"event_id": "e1"}], "safe_for_marking_use": False},
    )

    assert status == "ambiguous_candidate"
    assert "mixed_or_ambiguous_topic" in blockers
    assert action == "defer_ambiguous_skill"


def test_existing_reviewed_decision_scope_is_marked_already_reviewed() -> None:
    item = build_review_queue_item(
        mapping=_mapping(),
        question=_question(),
        mark_event_record=_mark_events(),
        reviewed_decision={
            "evidence_id": "reviewed-1",
            "question_id": "32spring21_q01",
            "subpart_id": "32spring21_q01_whole",
            "route_status": "review_needed",
        },
    )

    assert item["reviewed_decision_status"] == "already_reviewed"
    assert item["recommended_review_action"] == "already_reviewed"
    assert item["reviewed_evidence_ids"] == ["reviewed-1"]


def test_reviewed_skill_conflict_is_reconciled() -> None:
    item = build_review_queue_item(
        mapping=_mapping(),
        question=_question(),
        mark_event_record=_mark_events(),
        reviewed_decision={
            "evidence_id": "reviewed-1",
            "question_id": "32spring21_q01",
            "subpart_id": "32spring21_q01_whole",
            "route_status": "review_needed",
            "reviewed_source_skill_ids": ["9709_p3_3_1_polynomial_division_factor_remainder"],
        },
    )

    assert "reviewed_skill_ids_not_in_current_candidate_mapping" in item["reconciliation_flags"]


def test_p1_support_only_candidate_is_blocked_from_p3_mastery_review() -> None:
    status, blockers, action = classify_review_queue_item(
        has_question_asset=True,
        has_mark_scheme_asset=True,
        p3_skill_ids=[],
        non_p3_skill_ids=["9709_p1_quadratics_discriminant_intersections"],
    )

    assert status == "blocked_candidate"
    assert "p1_or_support_only_candidate_skill" in blockers
    assert action == "reject_p1_prerequisite_only"


def test_advisory_only_mark_events_do_not_produce_clean_reviewed_evidence() -> None:
    item = build_review_queue_item(
        mapping=_mapping(),
        question=_question(),
        mark_event_record=_mark_events(safe_for_marking_use=False),
        topic_route={"confidence": "high", "review_required": False},
    )

    assert item["proposed_route_status"] == "clean_candidate"
    assert item["proposed_route_status"] != "clean"
    assert "mark_events_advisory_only" in item["proposed_blockers"]
    assert item["reviewed_decision_status"] == "not_reviewed"


def test_dydx_alone_does_not_imply_parametric_implicit_differentiation() -> None:
    status, blockers, action = classify_review_queue_item(
        has_question_asset=True,
        has_mark_scheme_asset=True,
        p3_skill_ids=["9709_p3_3_4_parametric_implicit_differentiation"],
        non_p3_skill_ids=[],
        topic_route={"confidence": "high", "review_required": False},
        mapping={
            "evidence": {
                "topic_confidence": "high",
                "topic_uncertain": False,
                "question_text_snippet": "Find dy/dx at the point where x = 1.",
                "mark_scheme_text_snippet": "Differentiate and substitute x = 1.",
            }
        },
        mark_event_record={"mark_events": [{"event_id": "e1"}], "safe_for_marking_use": False},
    )

    assert status == "ambiguous_candidate"
    assert "weak_parametric_implicit_evidence_dydx_only" in blockers
    assert action == "verify_de_vs_implicit_differentiation"


def test_separation_of_variables_context_downgrades_parametric_candidate() -> None:
    status, blockers, action = classify_review_queue_item(
        has_question_asset=True,
        has_mark_scheme_asset=True,
        p3_skill_ids=["9709_p3_3_4_parametric_implicit_differentiation"],
        non_p3_skill_ids=[],
        topic_route={
            "confidence": "high",
            "review_required": False,
            "primary_topic_id": "9709_p3_topic_differential_equations",
        },
        mapping={
            "evidence": {
                "topic_confidence": "high",
                "topic_uncertain": False,
                "question_text_snippet": "Solve the differential equation dy/dx = e^y sin x.",
                "mark_scheme_text_snippet": "Separate variables correctly and integrate both sides.",
            }
        },
        mark_event_record={"mark_events": [{"event_id": "e1"}], "safe_for_marking_use": False},
    )

    assert status == "ambiguous_candidate"
    assert "possible_differential_equation_not_parametric_or_implicit" in blockers
    assert action == "verify_de_vs_implicit_differentiation"


def test_queue_item_with_separation_context_is_not_clean_parametric_candidate() -> None:
    item = build_review_queue_item(
        mapping={
            **_mapping(),
            "primary_skill_ids": ["9709_p3_3_4_parametric_implicit_differentiation"],
            "prerequisite_skill_ids": ["9709_p3_3_4_derivative_rules"],
            "evidence": {
                "topic_confidence": "high",
                "topic_uncertain": False,
                "question_text_snippet": "The variables x and y satisfy the differential equation dy/dx = y sin x.",
                "mark_scheme_text_snippet": "Separate variables correctly, integrate, and use x = 0, y = 1.",
            },
        },
        question=_question(),
        topic_route={"primary_topic_id": "9709_p3_topic_differential_equations", "confidence": "high"},
        mark_event_record=_mark_events(),
    )

    assert item["proposed_route_status"] == "ambiguous_candidate"
    assert item["recommended_review_action"] == "verify_de_vs_implicit_differentiation"
    assert "possible_differential_equation_not_parametric_or_implicit" in item["proposed_blockers"]


def test_valid_implicit_differentiation_candidate_is_not_blocked_by_boundary_rule() -> None:
    status, blockers, action = classify_review_queue_item(
        has_question_asset=True,
        has_mark_scheme_asset=True,
        p3_skill_ids=["9709_p3_3_4_parametric_implicit_differentiation"],
        non_p3_skill_ids=[],
        topic_route={"confidence": "high", "review_required": False},
        mapping={
            "evidence": {
                "topic_confidence": "high",
                "topic_uncertain": False,
                "question_text_snippet": "The curve is defined implicitly by x^2 + xy + y^2 = 7.",
                "mark_scheme_text_snippet": "Differentiate implicitly and collect terms in dy/dx; isolate dy/dx.",
            }
        },
        mark_event_record={"mark_events": [{"event_id": "e1"}], "safe_for_marking_use": False},
    )

    assert status == "clean_candidate"
    assert "possible_differential_equation_not_parametric_or_implicit" not in blockers
    assert "weak_parametric_implicit_evidence_dydx_only" not in blockers
    assert action == "review_assets_and_skill"


def test_duplicate_reviewed_registry_scopes_are_surfaced(tmp_path: Path) -> None:
    paths = _write_queue_fixture(tmp_path, duplicate_reviewed=True)

    queue = build_p3_exact_skill_review_queue(
        question_bank_path=paths["question_bank"],
        topic_routing_path=None,
        asterion_question_bank_path=None,
        content_lab_candidates_path=None,
        mark_events_path=paths["mark_events"],
        p3_skill_mappings_path=paths["skill_mappings"],
        p3_topic_assignments_path=None,
        reviewed_decisions_path=paths["reviewed_decisions"],
        output_path=None,
        report_path=None,
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert queue["summary"]["reviewed_registry_duplicate_scope_count"] == 1
    assert queue["reconciliation"]["duplicate_reviewed_scopes"] == ["32spring21_q01:32spring21_q01_whole"]


def test_build_review_queue_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_p3_exact_skill_review_queue.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _mapping() -> dict[str, object]:
    return {
        "question_id": "32spring21_q01",
        "paper": "32spring21",
        "session": "March",
        "variant": "32",
        "question_number": "1",
        "subpart_id": "32spring21_q01_whole",
        "subpart_label": "whole",
        "primary_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
        "secondary_skill_ids": [],
        "prerequisite_skill_ids": [],
        "evidence": {"topic_confidence": "high", "topic_uncertain": False},
    }


def _question() -> dict[str, object]:
    return {
        "question_id": "32spring21_q01",
        "paper_family": "p3",
        "question_image_path": "p3/32spring21/questions/q01.png",
        "mark_scheme_image_path": "p3/32spring21/mark_scheme/q01.png",
        "question_text_trust": "medium",
    }


def _mark_events(*, safe_for_marking_use: bool = False) -> dict[str, object]:
    return {
        "question_id": "32spring21_q01",
        "source_mark_scheme_image_path": "p3/32spring21/mark_scheme/q01.png",
        "source_text_kind": "native",
        "safe_for_advisory_use": True,
        "safe_for_marking_use": safe_for_marking_use,
        "total_marks_match": True,
        "mark_events": [{"event_id": "32spring21_q01_me0001", "mark_code_raw": "M1", "part_path": []}],
    }


def _write_queue_fixture(tmp_path: Path, *, duplicate_reviewed: bool = False) -> dict[str, Path]:
    paths = {
        "question_bank": tmp_path / "question_bank.json",
        "mark_events": tmp_path / "mark_events.json",
        "skill_mappings": tmp_path / "skill_mappings.json",
        "reviewed_decisions": tmp_path / "reviewed_decisions.json",
    }
    _write_json(paths["question_bank"], {"questions": [_question()]})
    _write_json(paths["mark_events"], {"records": [_mark_events()]})
    _write_json(
        paths["skill_mappings"],
        {
            "mappings": [
                {
                    **_mapping(),
                    "caie_class_or_component": "Paper 3",
                }
            ]
        },
    )
    reviewed = [
        {
            "evidence_id": "reviewed-1",
            "question_id": "32spring21_q01",
            "subpart_id": "32spring21_q01_whole",
            "route_status": "review_needed",
        }
    ]
    if duplicate_reviewed:
        reviewed.append({**reviewed[0], "evidence_id": "reviewed-2"})
    _write_json(paths["reviewed_decisions"], {"records": reviewed})
    return paths


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
