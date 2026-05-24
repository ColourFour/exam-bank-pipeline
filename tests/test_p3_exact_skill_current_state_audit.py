from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.p3_exact_skill.current_state_audit import build_p3_exact_skill_current_state_audit


def test_current_state_audit_summarizes_readiness_counts(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    ambiguity_path = tmp_path / "ambiguity.json"
    decomposition_path = tmp_path / "decomposition.json"
    reviewed_path = tmp_path / "reviewed.json"
    sidecar_path = tmp_path / "missing_sidecar.json"
    _write_json(
        queue_path,
        {
            "summary": {
                "total_queue_items": 4,
                "status_counts": {
                    "cross_topic_candidate": 2,
                    "split_needed_candidate": 1,
                    "fallback_only": 1,
                },
                "cross_topic_status_counts": {"cross_topic_reviewable": 2},
                "topic_routing_alignment_counts": {"supporting_topic": 2},
                "already_reviewed_records": 1,
                "advisory_only_mark_event_count": 4,
                "missing_question_asset_count": 0,
                "missing_mark_scheme_asset_count": 0,
                "no_candidate_skill_count": 0,
                "fallback_only_candidates": 1,
            }
        },
    )
    _write_json(ambiguity_path, {"summary": {"audit_group_counts": {"cross_topic_reviewable": 2}}})
    _write_json(
        decomposition_path,
        {
            "summary": {
                "decomposition_candidate_count": 2,
                "part_level_candidate_count": 1,
                "subpart_level_candidate_count": 1,
                "needs_manual_split_count": 1,
                "insufficient_part_signal_count": 0,
                "conflict_needs_review_count": 0,
                "decomposition_status_counts": {"already_part_scoped": 2},
            }
        },
    )
    _write_json(
        reviewed_path,
        {
            "records": [
                {"route_status": "review_needed"},
                {"route_status": "thin"},
                {"route_status": "blocked"},
            ]
        },
    )

    audit = build_p3_exact_skill_current_state_audit(
        queue_path=queue_path,
        ambiguity_audit_path=ambiguity_path,
        part_decomposition_path=decomposition_path,
        reviewed_path=reviewed_path,
        sidecar_path=sidecar_path,
        output_path=None,
        report_path=None,
        dry_run=True,
        generated_at="2026-05-25T00:00:00Z",
    )

    assert audit["summary"]["total_queue_items"] == 4
    assert audit["summary"]["clean_reviewed_registry_count"] == 0
    assert audit["summary"]["thin_reviewed_registry_count"] == 1
    assert audit["summary"]["part_level_decomposition_candidate_count"] == 1
    assert audit["summary"]["subpart_level_decomposition_candidate_count"] == 1
    assert audit["summary"]["forbidden_runtime_sidecar_exists"] is False
    assert "READY_FOR_ASTERION_CONTENT_LAB_REVIEW_DIAGNOSTICS" in audit["readiness_verdicts"]
    assert audit["asterion_recommendation"]["allowed_capacity"] == "Content Lab/admin/reviewer diagnostics only"


def test_current_state_audit_dry_run_does_not_write_files(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    ambiguity_path = tmp_path / "ambiguity.json"
    decomposition_path = tmp_path / "decomposition.json"
    reviewed_path = tmp_path / "reviewed.json"
    output_path = tmp_path / "audit.json"
    report_path = tmp_path / "audit.md"
    _write_json(queue_path, {"summary": {"total_queue_items": 0, "status_counts": {}}})
    _write_json(ambiguity_path, {"summary": {"audit_group_counts": {}}})
    _write_json(decomposition_path, {"summary": {"decomposition_status_counts": {}}})
    _write_json(reviewed_path, {"records": []})

    build_p3_exact_skill_current_state_audit(
        queue_path=queue_path,
        ambiguity_audit_path=ambiguity_path,
        part_decomposition_path=decomposition_path,
        reviewed_path=reviewed_path,
        output_path=output_path,
        report_path=report_path,
        dry_run=True,
    )

    assert not output_path.exists()
    assert not report_path.exists()


def test_current_state_audit_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_p3_exact_skill_current_state.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
