from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.p3_exact_skill.part_decomposition import build_p3_exact_skill_part_decomposition, decompose_queue_item


def test_whole_question_with_part_labeled_mark_events_proposes_part_candidates(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    whole = _item("q1", "whole", status="split_needed_candidate", mark_parts=["a", "b"])
    part_a = _item("q1", "a", status="cross_topic_candidate", skill_id="9709_p3_skill_a", mark_parts=["a"])
    part_b = _item("q1", "b", status="cross_topic_candidate", skill_id="9709_p3_skill_b", mark_parts=["b"])
    _write_json(queue_path, {"items": [whole, part_a, part_b]})

    report = build_p3_exact_skill_part_decomposition(
        queue_path=queue_path,
        output_path=None,
        report_path=None,
        dry_run=True,
        generated_at="2026-05-24T00:00:00Z",
    )

    whole_candidates = [item for item in report["items"] if item["source_queue_id"] == whole["queue_id"]]
    assert report["summary"]["part_level_candidate_count"] >= 2
    assert {candidate["proposed_part_id"] for candidate in whole_candidates} == {"a", "b"}
    assert any(candidate["candidate_source_skill_ids"] == ["9709_p3_skill_a"] for candidate in whole_candidates)


def test_already_scoped_part_remains_already_part_scoped() -> None:
    item = _item("q1", "a", status="cross_topic_candidate", mark_parts=["a", "b"])

    status, candidates, summary, warning = decompose_queue_item(item, question_items=[item])

    assert status == "already_part_scoped"
    assert candidates[0]["decomposition_status"] == "already_part_scoped"
    assert candidates[0]["matching_mark_event_refs"]
    assert summary["matching_mark_event_count"] == 1
    assert "whole-question images" in warning


def test_no_part_labels_produces_insufficient_part_signal() -> None:
    item = _item("q1", "whole", status="cross_topic_candidate", mark_parts=[])

    status, candidates, _, warning = decompose_queue_item(item, question_items=[item])

    assert status == "insufficient_part_signal"
    assert candidates == []
    assert "No part-labeled mark events" in warning


def test_split_needed_without_part_labels_needs_manual_split() -> None:
    item = _item("q1", "whole", status="split_needed_candidate", mark_parts=[])

    status, candidates, _, _ = decompose_queue_item(item, question_items=[item])

    assert status == "needs_manual_split"
    assert candidates == []


def test_conflict_does_not_become_safe_decomposition() -> None:
    item = _item("q1", "whole", status="conflict_candidate", mark_parts=["a", "b"])

    status, candidates, _, warning = decompose_queue_item(item, question_items=[item])

    assert status == "conflict_needs_review"
    assert candidates == []
    assert "conflict" in warning


def test_part_decomposition_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_p3_exact_skill_part_decomposition.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _item(
    question_id: str,
    part: str,
    *,
    status: str,
    skill_id: str = "9709_p3_skill",
    mark_parts: list[str],
) -> dict[str, object]:
    subpart_id = f"{question_id}_{part}" if part != "whole" else f"{question_id}_whole"
    return {
        "queue_id": f"queue:{question_id}:{subpart_id}",
        "question_id": question_id,
        "part_id": part,
        "subpart_id": subpart_id,
        "subpart_label": part,
        "candidate_p3_skill_ids": [skill_id],
        "supporting_candidate_skill_ids": [],
        "topic_routing_topic_ids": ["9709_p3_topic_test"],
        "cross_topic_status": "cross_topic_split_needed" if status == "split_needed_candidate" else "cross_topic_reviewable",
        "proposed_route_status": status,
        "source_question_asset_refs": [{"path": "p3/q.png", "exists": True}],
        "source_mark_scheme_asset_refs": [{"path": "p3/m.png", "exists": True}],
        "mark_event_refs": [
            {"event_id": f"{question_id}_{mark_part}_me", "part_path": [mark_part], "review_status": "advisory"}
            for mark_part in mark_parts
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
