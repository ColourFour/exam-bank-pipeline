from __future__ import annotations

import json
from pathlib import Path

from exam_bank.p3_exact_skill.batch_conclusions import (
    build_manual_review_batch_conclusions,
    render_manual_review_batch_conclusions_report,
)


def test_manual_review_batch_conclusions_summarises_counts_and_sections(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    queue_path = tmp_path / "queue.json"
    reviewed_path = tmp_path / "reviewed.json"
    _write_json(queue_path, {"items": [_queue_item("q1", "aligned"), _queue_item("q2", "supporting_topic")]})
    _write_json(
        batch_dir / "batch_0001_manifest.v1.json",
        {
            "batch_id": "batch_0001",
            "selected_count": 2,
            "cross_topic_summary": {
                "cross_topic_status_counts": {"cross_topic_reviewable": 2},
                "topic_routing_alignment_counts": {"aligned": 1, "supporting_topic": 1},
            },
        },
    )
    _write_json(
        batch_dir / "batch_0001_decision_template.v1.json",
        {
            "record_count": 2,
            "records": [
                {"mark_event_refs": [{"event_id": "q1_me0001", "review_status": "advisory", "advisory_only": True}]},
                {"mark_event_refs": [{"event_id": "q2_me0001", "review_status": "advisory", "advisory_only": True}]},
            ],
        },
    )
    _write_json(
        batch_dir / "batch_0001_review_responses.v1.json",
        {
            "artifact_kind": "human_review_response_draft",
            "batch_id": "batch_0001",
            "responses": [
                _response("q1", route_status="clean", exact_skill_confirmed="yes"),
                _response("q2", route_status="ambiguous", exact_skill_confirmed="no_wrong_skill"),
            ],
        },
    )
    _write_json(reviewed_path, {"records": [{"route_status": "thin"}]})

    payload = build_manual_review_batch_conclusions(
        batch_dir=batch_dir,
        queue_path=queue_path,
        reviewed_registry_path=reviewed_path,
        output_json_path=None,
        output_report_path=None,
        generated_at="2026-05-26T00:00:00Z",
        dry_run=True,
    )
    report = render_manual_review_batch_conclusions_report(payload)

    assert payload["reviewed_item_counts"]["batch_response_count"] == 2
    assert payload["outcome_counts"]["approved_or_clean_draft"] == 1
    assert payload["outcome_counts"]["ambiguous_or_needs_adjustment_draft"] == 1
    assert payload["reviewed_mark_event_and_evidence_status"]["all_template_mark_events_advisory_only"] is True
    assert "## Top Failure Modes" in report
    assert "## Fields Needing Stricter Gating" in report
    assert "## Implemented In This Pass" in report


def test_batch_0003_conclusions_separate_exact_skill_and_mark_event_decisions() -> None:
    payload = json.loads(Path("reports/manual_review_batch_0003_conclusions.v1.json").read_text(encoding="utf-8"))
    records = payload["record_decisions"]

    assert payload["batch_id"] == "batch_0003"
    assert payload["outcome_counts"]["total_reviewed_records"] == 14
    assert payload["outcome_counts"]["promoted_exact_skill_records"] == 0
    assert payload["outcome_counts"]["approved_mark_event_count"] == 0
    assert payload["outcome_counts"]["content_lab_generation_ready_after"] == 0
    assert payload["advisory_only_mark_event_refs"]["all_records_left_advisory_only"] is True
    assert payload["advisory_only_mark_event_refs"]["explicit_approved_mark_event_ids"] == []
    assert any(record["exact_skill_decision"] == "retagged_not_promoted" for record in records)
    assert any(record["exact_skill_decision"] == "blocked" for record in records)
    assert any(record["exact_skill_decision"] == "deferred_thin" for record in records)
    assert all(record["mark_event_decision"] == "left_advisory_only" for record in records)
    assert all(record["content_lab_generation_allowed"] is False for record in records)


def test_batch_0003_seed_report_promotes_no_records() -> None:
    payload = json.loads(Path("reports/p3_exact_skill_registry_seed_0003.v1.json").read_text(encoding="utf-8"))

    assert payload["batch_id"] == "batch_0003"
    assert payload["selected_count"] == 0
    assert payload["selected_record_ids"] == []
    assert payload["selected_records"] == []
    assert len(payload["records_intentionally_skipped"]) == 14
    assert payload["mark_event_policy"]["approved_mark_event_count"] == 0
    assert payload["candidate_generation_policy"]["new_generation_ready_from_this_pass"] == 0


def _queue_item(question_id: str, alignment: str) -> dict[str, object]:
    return {
        "queue_id": f"queue:{question_id}",
        "question_id": question_id,
        "subpart_id": f"{question_id}_whole",
        "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"],
        "primary_candidate_skill_ids": ["9709_p3_3_6_fixed_point_iteration"],
        "supporting_candidate_skill_ids": ["9709_p3_3_6_root_location"] if alignment == "supporting_topic" else [],
        "candidate_region_topic": {"topic_assignment_name": "Numerical solution of equations"},
        "topic_routing": {
            "primary_topic_id": "9709_p3_topic_numerical_solution_of_equations",
            "confidence": "high",
            "evidence_used": ["question_text"],
        },
        "topic_routing_alignment": alignment,
        "recommended_scope": "subpart_level",
        "proposed_route_status": "cross_topic_candidate",
        "source_question_asset_refs": [{"path": "p3/questions/q01.png", "exists": True}],
        "source_mark_scheme_asset_refs": [{"path": "p3/mark_scheme/q01.png", "exists": True}],
        "mark_event_refs": [{"event_id": f"{question_id}_me0001", "review_status": "advisory"}],
        "asterion_candidate": {
            "generation_gate_status": "blocked_until_reviewed",
            "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"],
        },
    }


def _response(question_id: str, *, route_status: str, exact_skill_confirmed: str) -> dict[str, object]:
    return {
        "queue_id": f"queue:{question_id}",
        "question_id": question_id,
        "subpart_id": f"{question_id}_whole",
        "fields": {
            "route_status": route_status,
            "exact_skill_confirmed": exact_skill_confirmed,
            "allowed_use_case_summary": "source_backed_examples" if route_status == "clean" else "none",
            "evidence_basis_status": "drafted" if route_status == "clean" else "needs_more_review",
            "scope_decision": "subpart_level_needed",
            "decomposition_accepted": "yes" if route_status == "clean" else "needs_adjustment",
            "support_material_decision": "supporting_context_only" if route_status == "clean" else "possible_target_skill",
            "inspected_question_image": "yes",
            "inspected_mark_scheme_image": "yes",
            "part_boundary_confirmed": "yes",
            "reviewer_notes": "Fixture note.",
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
