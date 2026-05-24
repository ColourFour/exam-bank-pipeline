from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.p3_exact_skill.ambiguity_audit import build_p3_exact_skill_ambiguity_audit


def test_ambiguity_audit_groups_sharper_candidate_statuses(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    _write_json(
        queue_path,
        {
            "items": [
                _item("q1", "cross_topic_candidate", "cross_topic_reviewable"),
                _item("q2", "split_needed_candidate", "cross_topic_split_needed", skill_count=2),
                _item(
                    "q3",
                    "conflict_candidate",
                    "conflict_needs_review",
                    blockers=["mark_events_advisory_only", "possible_differential_equation_not_parametric_or_implicit"],
                    alignment="conflicting",
                ),
                _item("q4", "ambiguous_candidate", "unknown", blockers=[], alignment="unknown"),
            ]
        },
    )

    audit = build_p3_exact_skill_ambiguity_audit(
        queue_path=queue_path,
        output_path=None,
        report_path=None,
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert audit["summary"]["ambiguous_candidate_count"] == 1
    assert audit["groups"]["cross_topic_reviewable"]["count"] == 1
    assert audit["groups"]["cross_topic_split_needed"]["count"] == 1
    assert audit["groups"]["de_vs_parametric_implicit_conflict"]["count"] == 1
    assert audit["groups"]["unknown_ambiguity"]["count"] == 1
    assert audit["groups"]["cross_topic_reviewable"]["can_reclassify_safely"] is True


def test_ambiguity_audit_dry_run_does_not_write_files(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    output_path = tmp_path / "audit.json"
    report_path = tmp_path / "audit.md"
    _write_json(queue_path, {"items": [_item("q1", "cross_topic_candidate", "cross_topic_reviewable")]})

    result = build_p3_exact_skill_ambiguity_audit(
        queue_path=queue_path,
        output_path=output_path,
        report_path=report_path,
        dry_run=True,
    )

    assert result["summary"]["audited_item_count"] == 1
    assert not output_path.exists()
    assert not report_path.exists()


def test_ambiguity_audit_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_p3_exact_skill_ambiguity_audit.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _item(
    question_id: str,
    status: str,
    cross_topic_status: str,
    *,
    blockers: list[str] | None = None,
    alignment: str = "supporting_topic",
    skill_count: int = 1,
) -> dict[str, object]:
    skills = [f"9709_p3_skill_{index}" for index in range(skill_count)]
    return {
        "queue_id": f"queue:{question_id}",
        "question_id": question_id,
        "subpart_id": f"{question_id}_whole",
        "subpart_label": "whole",
        "candidate_p3_skill_ids": skills,
        "topic_routing_topic_ids": ["9709_p3_topic_test"],
        "topic_routing_alignment": alignment,
        "cross_topic_status": cross_topic_status,
        "proposed_route_status": status,
        "proposed_blockers": ["mark_events_advisory_only"] if blockers is None else blockers,
        "reconciliation_flags": [],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
