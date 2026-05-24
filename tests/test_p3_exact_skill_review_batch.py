from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.p3_exact_skill.review_batch import (
    build_decision_template,
    build_p3_exact_skill_review_batch,
    select_review_batch_items,
)
from exam_bank.p3_exact_skill.reviewed_decisions import validate_reviewed_decisions_payload


def test_selects_clean_candidate_before_ambiguous_or_fallback() -> None:
    selected, skipped = select_review_batch_items(
        [
            _item("q1", status="ambiguous_candidate"),
            _item("q2"),
            _item("q3", status="fallback_only"),
            _item("q4", status="cross_topic_candidate", cross_topic_status="cross_topic_reviewable"),
        ],
        reviewed_scopes=set(),
        clean_reviewed_counts={},
        limit=10,
        status="clean_candidate",
    )

    assert [item["question_id"] for item in selected] == ["q2"]
    assert skipped["status_filter"] == 3


def test_default_batch_selection_includes_cross_topic_and_excludes_conflicts() -> None:
    selected, skipped = select_review_batch_items(
        [
            _item("q1", status="conflict_candidate"),
            _item("q2", status="cross_topic_candidate", cross_topic_status="cross_topic_reviewable"),
            _item("q3", status="fallback_only"),
            _item("q4", status="ambiguous_candidate"),
        ],
        reviewed_scopes=set(),
        clean_reviewed_counts={},
        limit=10,
    )

    assert [item["question_id"] for item in selected] == ["q2"]
    assert skipped["excluded_status"] == 3


def test_part_decomposition_review_prioritizes_items_with_part_candidates() -> None:
    selected, skipped = select_review_batch_items(
        [
            _item("q1", status="cross_topic_candidate", cross_topic_status="cross_topic_reviewable"),
            _item(
                "q2",
                status="cross_topic_candidate",
                cross_topic_status="cross_topic_reviewable",
                decomposition_candidates=True,
                priority=100,
            ),
            _item("q3", status="conflict_candidate", decomposition_candidates=True),
        ],
        reviewed_scopes=set(),
        clean_reviewed_counts={},
        limit=10,
        batch_purpose="part_decomposition_review",
    )

    assert [item["question_id"] for item in selected] == ["q2"]
    assert skipped["no_part_decomposition_candidates"] == 1
    assert skipped["excluded_status"] == 1


def test_excludes_already_reviewed_scopes_by_default() -> None:
    selected, skipped = select_review_batch_items(
        [_item("q1"), _item("q2", reviewed=True)],
        reviewed_scopes={("q1", "q1_whole")},
        clean_reviewed_counts={},
        limit=10,
    )

    assert selected == []
    assert skipped["already_reviewed"] == 2


def test_requires_question_and_mark_scheme_asset_refs() -> None:
    selected, skipped = select_review_batch_items(
        [_item("q1", question_asset=False), _item("q2", mark_asset=False), _item("q3")],
        reviewed_scopes=set(),
        clean_reviewed_counts={},
        limit=10,
    )

    assert [item["question_id"] for item in selected] == ["q3"]
    assert skipped["missing_question_asset"] == 1
    assert skipped["missing_mark_scheme_asset"] == 1


def test_prioritizes_sparse_skills_when_summary_data_is_available() -> None:
    selected, _ = select_review_batch_items(
        [
            _item("q1", skill_id="9709_p3_skill_dense", priority=220),
            _item("q2", skill_id="9709_p3_skill_sparse", priority=180),
        ],
        reviewed_scopes=set(),
        clean_reviewed_counts={"9709_p3_skill_dense": 3},
        limit=1,
    )

    assert selected[0]["question_id"] == "q2"


def test_avoids_repeated_question_when_unique_question_is_available() -> None:
    selected, _ = select_review_batch_items(
        [
            _item("q1", priority=300),
            {**_item("q1", priority=290), "subpart_id": "q1_b", "queue_id": "queue:q1:q1_b"},
            _item("q2", priority=150),
        ],
        reviewed_scopes=set(),
        clean_reviewed_counts={},
        limit=2,
    )

    assert [item["question_id"] for item in selected] == ["q1", "q2"]


def test_preserves_advisory_mark_event_refs_but_labels_them_advisory_only() -> None:
    template = build_decision_template(
        [_item("q1")],
        batch_id="batch_test",
        generated_at="2026-05-23T00:00:00Z",
        queue_path="queue.json",
        reviewed_path="reviewed.json",
    )

    ref = template["records"][0]["mark_event_refs"][0]
    assert ref["review_status"] == "advisory"
    assert ref["advisory_only"] is True


def test_generated_decision_template_defaults_to_review_needed_and_blocks_use_cases() -> None:
    template = build_decision_template(
        [_item("q1")],
        batch_id="batch_test",
        generated_at="2026-05-23T00:00:00Z",
        queue_path="queue.json",
        reviewed_path="reviewed.json",
    )
    record = template["records"][0]

    assert template["schema"] == "exam_bank.p3_exact_skill.review_batch_template"
    assert record["route_status"] == "review_needed"
    assert record["blockers"] == ["pending_human_review"]
    assert record["reviewed_source_skill_ids"] == []
    assert record["suggested_source_skill_ids"] == ["9709_p3_3_2_log_exponential_equations"]
    assert all(allowed is False for allowed in record["allowed_use_cases"].values())


def test_decision_template_includes_cross_topic_suggestions_without_reviewed_skill_ids() -> None:
    template = build_decision_template(
        [_item("q1", cross_topic_status="cross_topic_reviewable")],
        batch_id="batch_test",
        generated_at="2026-05-23T00:00:00Z",
        queue_path="queue.json",
        reviewed_path="reviewed.json",
    )
    record = template["records"][0]

    assert record["suggested_primary_skill_ids"] == ["9709_p3_3_2_log_exponential_equations"]
    assert record["suggested_supporting_skill_ids"] == ["9709_p3_3_3_trigonometric_equations"]
    assert record["suggested_cross_topic_status"] == "cross_topic_reviewable"
    assert record["suggested_recommended_scope"] == "reviewer_decide"
    assert record["suggested_candidate_status"] == "clean_candidate"
    assert record["suggested_review_priority"] == "1_clean_candidate"
    assert record["suggested_scope_risk"] == "reviewer_decide"
    assert record["suggested_ambiguity_reason"] == "cross_topic_reviewable"
    assert record["reviewed_source_skill_ids"] == []
    assert record["route_status"] == "review_needed"


def test_cross_topic_template_is_not_valid_reviewed_registry() -> None:
    template = build_decision_template(
        [_item("q1", cross_topic_status="cross_topic_reviewable")],
        batch_id="batch_test",
        generated_at="2026-05-23T00:00:00Z",
        queue_path="queue.json",
        reviewed_path="reviewed.json",
    )

    errors, _ = validate_reviewed_decisions_payload(
        template,
        p3_skill_ids={"9709_p3_3_2_log_exponential_equations"},
    )

    assert any("schema_mismatch" in error for error in errors)


def test_review_packet_includes_cross_topic_fields(tmp_path: Path) -> None:
    packet = build_p3_exact_skill_review_batch(
        queue_path=_write_inline_queue(tmp_path, [_item("q1", cross_topic_status="cross_topic_reviewable")]),
        reviewed_path=_write_inline_reviewed(tmp_path),
        batch_id="batch_test",
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )["packet"]

    assert "Cross-topic status: `cross_topic_reviewable`" in packet
    assert "Topic-routing alignment: `supporting_topic`" in packet
    assert "Cross-topic reviewer checklist:" in packet


def test_dry_run_does_not_write_files(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    reviewed_path = tmp_path / "reviewed.json"
    out_dir = tmp_path / "batches"
    _write_json(queue_path, {"items": [_item("q1")]})
    _write_json(reviewed_path, {"records": []})

    result = build_p3_exact_skill_review_batch(
        queue_path=queue_path,
        reviewed_path=reviewed_path,
        batch_id="batch_test",
        out_dir=out_dir,
        dry_run=True,
    )

    assert result["selected_count"] == 1
    assert not out_dir.exists()


def test_build_review_batch_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_p3_exact_skill_review_batch.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _item(
    question_id: str,
    *,
    status: str = "clean_candidate",
    skill_id: str = "9709_p3_3_2_log_exponential_equations",
    priority: int = 200,
    reviewed: bool = False,
    question_asset: bool = True,
    mark_asset: bool = True,
    cross_topic_status: str = "single_skill_candidate",
    decomposition_candidates: bool = False,
) -> dict[str, object]:
    return {
        "queue_id": f"p3_exact_skill_review_queue:v1:{question_id}:{question_id}_whole",
        "question_id": question_id,
        "part_id": "whole",
        "subpart_id": f"{question_id}_whole",
        "paper": "32spring21",
        "session": "March",
        "variant": "32",
        "candidate_source_skill_ids": [skill_id],
        "candidate_p3_skill_ids": [skill_id],
        "primary_candidate_skill_ids": [skill_id],
        "supporting_candidate_skill_ids": (
            ["9709_p3_3_3_trigonometric_equations"] if cross_topic_status == "cross_topic_reviewable" else []
        ),
        "candidate_region_topic": {"subtopic_name": "Logarithmic and exponential equations"},
        "topic_routing": {"confidence": "high", "review_required": False},
        "topic_routing_topic_ids": ["9709_p3_topic_trigonometry"],
        "topic_routing_alignment": "supporting_topic" if cross_topic_status == "cross_topic_reviewable" else "aligned",
        "cross_topic_status": cross_topic_status,
        "cross_topic_notes": ["Supporting candidate skills are review context only."],
        "recommended_scope": "reviewer_decide" if cross_topic_status == "cross_topic_reviewable" else "whole_question",
        "decomposition_status": "part_level_candidate" if decomposition_candidates else "not_decomposable",
        "proposed_part_level_candidates": (
            [{"decomposition_id": f"decomp:{question_id}:a", "decomposition_status": "part_level_candidate"}]
            if decomposition_candidates
            else []
        ),
        "part_signal_summary": {"has_part_labeled_mark_events": decomposition_candidates},
        "part_scope_warning": "Use whole-question images to confirm part boundary." if decomposition_candidates else "",
        "review_priority_group": {
            "clean_candidate": "1_clean_candidate",
            "cross_topic_candidate": "2_cross_topic_candidate",
            "conflict_candidate": "5_conflict_candidate",
            "fallback_only": "6_fallback_only",
            "ambiguous_candidate": "7_ambiguous_candidate",
        }.get(status, "9_review_needed"),
        "ambiguity_reason": "cross_topic_reviewable" if cross_topic_status == "cross_topic_reviewable" else "unknown_ambiguity",
        "reviewer_cross_topic_checklist": ["Identify the main skill being assessed."],
        "asterion_candidate": {"candidate_id": f"content_lab_{question_id}", "generation_gate_block_reasons": []},
        "source_question_asset_refs": [{"path": f"p3/questions/{question_id}.png", "exists": question_asset}],
        "source_mark_scheme_asset_refs": [{"path": f"p3/marks/{question_id}.png", "exists": mark_asset}],
        "mark_event_refs": [{"event_id": f"{question_id}_me0001", "review_status": "advisory"}],
        "proposed_route_status": status,
        "proposed_blockers": ["mark_events_advisory_only"],
        "recommended_review_action": "review_assets_and_skill",
        "reviewed_decision_status": "already_reviewed" if reviewed else "not_reviewed",
        "reconciliation_flags": [],
        "priority_score": priority,
    }


def _write_inline_queue(tmp_path: Path, items: list[dict[str, object]]) -> Path:
    path = tmp_path / "queue.json"
    _write_json(path, {"items": items})
    return path


def _write_inline_reviewed(tmp_path: Path) -> Path:
    path = tmp_path / "reviewed.json"
    _write_json(path, {"records": []})
    return path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
