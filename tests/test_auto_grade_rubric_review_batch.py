from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.auto_grade.eligibility import build_eligible_items
from exam_bank.auto_grade.review_batch import build_rubric_review_batch
from exam_bank.auto_grade.reviewed_rubrics import build_rubric_review_queue, validate_reviewed_rubrics
from tests.test_auto_grade_eligibility import _write_fixture


def test_review_batch_selects_clean_high_priority_candidates_before_risky_ones(tmp_path: Path) -> None:
    paths = _prepared_queue_fixture(tmp_path)

    result = build_rubric_review_batch(
        question_bank_path=paths["question_bank"],
        review_queue_path=paths["queue"],
        mark_events_path=paths["mark_events"],
        eligible_items_path=paths["eligible"],
        output_path=tmp_path / "batch.json",
        report_path=tmp_path / "batch.md",
        draft_output_path=tmp_path / "draft.json",
        max_rubrics=25,
        max_events=75,
        generated_at="2026-05-21T00:00:00Z",
    )

    batch = result["batch"]
    assert [candidate["question_id"] for candidate in batch["candidates"]] == [
        "11summer26_q01",
        "11summer26_q02",
        "11summer26_q03",
    ]
    assert "11summer26_q04" not in {candidate["question_id"] for candidate in batch["candidates"]}
    assert batch["summary"]["excluded_candidate_counts"]["excluded_risk_flags"] == 1
    assert (tmp_path / "batch.json").exists()
    assert (tmp_path / "batch.md").exists()


def test_review_batch_respects_max_rubrics_and_max_events(tmp_path: Path) -> None:
    paths = _prepared_queue_fixture(tmp_path)

    result = build_rubric_review_batch(
        question_bank_path=paths["question_bank"],
        review_queue_path=paths["queue"],
        mark_events_path=paths["mark_events"],
        eligible_items_path=paths["eligible"],
        output_path=None,
        report_path=None,
        draft_output_path=None,
        max_rubrics=2,
        max_events=3,
        generated_at="2026-05-21T00:00:00Z",
        dry_run=True,
    )

    batch = result["batch"]
    assert batch["rubric_count"] == 1
    assert batch["event_count"] == 2


def test_review_batch_and_draft_do_not_mark_candidates_approved(tmp_path: Path) -> None:
    paths = _prepared_queue_fixture(tmp_path)

    result = build_rubric_review_batch(
        question_bank_path=paths["question_bank"],
        review_queue_path=paths["queue"],
        mark_events_path=paths["mark_events"],
        eligible_items_path=paths["eligible"],
        output_path=None,
        report_path=None,
        draft_output_path=None,
        generated_at="2026-05-21T00:00:00Z",
        dry_run=True,
    )

    assert {candidate["review_status"] for candidate in result["batch"]["candidates"]} == {"needs_human_review"}
    draft = result["draft_reviewed_rubrics"]
    assert draft["summary"]["approved_count"] == 0
    assert all(rubric["review_status"] == "needs_human_review" for rubric in draft["rubrics"])
    assert all(rubric["safe_for_teacher_beta"] is False for rubric in draft["rubrics"])

    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=draft_path,
        question_bank_path=paths["question_bank"],
        report_path=None,
    )
    assert report["approved_rubric_count"] == 0


def test_review_batch_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_auto_grade_rubric_review_batch.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def _prepared_queue_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = _write_fixture(tmp_path)
    for relative in [
        "p1/11summer26/questions/q02.png",
        "p1/11summer26/mark_scheme/q03.png",
    ]:
        path = paths["artifact_root"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=paths["eligible"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["missing_reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
    )
    queue_path = tmp_path / "queue.json"
    build_rubric_review_queue(
        question_bank_path=paths["question_bank"],
        eligible_items_path=paths["eligible"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        output_path=queue_path,
        report_path=None,
        generated_at="2026-05-21T00:00:00Z",
    )
    paths["queue"] = queue_path
    return paths
