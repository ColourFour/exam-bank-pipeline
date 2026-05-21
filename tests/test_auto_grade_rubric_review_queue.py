from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from exam_bank.auto_grade.eligibility import build_eligible_items
from exam_bank.auto_grade.reviewed_rubrics import build_rubric_review_queue
from tests.test_auto_grade_eligibility import _write_fixture


def test_review_queue_builder_creates_candidates_from_fixture_bank(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=paths["eligible"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["missing_reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
    )

    queue = build_rubric_review_queue(
        question_bank_path=paths["question_bank"],
        eligible_items_path=paths["eligible"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        output_path=tmp_path / "queue.json",
        report_path=tmp_path / "queue.md",
        generated_at="2026-05-21T00:00:00Z",
    )

    assert queue["candidate_count"] == 4
    assert (tmp_path / "queue.json").exists()
    assert (tmp_path / "queue.md").exists()
    by_id = {candidate["question_id"]: candidate for candidate in queue["candidates"]}
    assert by_id["11summer26_q01"]["advisory_safe_for_use"] is True
    assert by_id["11summer26_q04"]["has_total_mismatch"] is True
    assert "total_mismatch" in by_id["11summer26_q04"]["candidate_blockers"]


def test_review_queue_prioritizes_clean_candidates_above_risky_candidates(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=paths["eligible"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["missing_reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
    )

    queue = build_rubric_review_queue(
        question_bank_path=paths["question_bank"],
        eligible_items_path=paths["eligible"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        output_path=None,
        report_path=None,
        generated_at="2026-05-21T00:00:00Z",
        dry_run=True,
    )

    positions = {candidate["question_id"]: index for index, candidate in enumerate(queue["candidates"])}
    assert positions["11summer26_q01"] < positions["11summer26_q04"]
    assert queue["summary"]["priority_buckets"]


def test_review_queue_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_auto_grade_rubric_review_queue.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
