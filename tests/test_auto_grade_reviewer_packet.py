from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.auto_grade.review_batch import build_rubric_review_batch
from exam_bank.auto_grade.reviewer_packet import (
    NOT_APPROVED_WARNING,
    build_approval_template,
    build_reviewer_packet,
)
from exam_bank.auto_grade.reviewed_rubrics import build_rubric_review_queue, validate_reviewed_rubrics
from exam_bank.auto_grade.eligibility import build_eligible_items
from tests.test_auto_grade_eligibility import _write_fixture


def test_reviewer_packet_generator_creates_index_and_candidate_pages(tmp_path: Path) -> None:
    paths = _review_batch_fixture(tmp_path)
    output_dir = tmp_path / "packet"

    report = build_reviewer_packet(
        review_batch_path=paths["batch"],
        question_bank_path=paths["question_bank"],
        reviewed_rubrics_path=paths["draft"],
        output_dir=output_dir,
    )

    assert report["candidate_page_count"] == 3
    assert (output_dir / "index.md").exists()
    assert (output_dir / "index.html").exists()
    pages = sorted(output_dir.glob("rubric_*_*.md"))
    assert len(pages) == 3
    page_text = pages[0].read_text(encoding="utf-8")
    assert "Question image:" in page_text
    assert "Mark-scheme image:" in page_text
    assert NOT_APPROVED_WARNING in page_text
    assert "advisory" in page_text.lower()


def test_approval_template_helper_emits_unapproved_template_by_default(tmp_path: Path) -> None:
    paths = _review_batch_fixture(tmp_path)
    output = tmp_path / "template.json"

    template = build_approval_template(review_batch_path=paths["batch"], first=True, output_path=output)

    assert output.exists()
    rubric = template["rubric"]
    assert rubric["review_status"] == "needs_human_review"
    assert rubric["safe_for_auto_grade_lab"] is False
    assert rubric["safe_for_teacher_beta"] is False
    assert rubric["safe_for_student_self_check"] is False
    for field in [
        "reviewed_by",
        "reviewed_at",
        "review_status",
        "rubric_total_verified",
        "safe_for_auto_grade_lab",
        "safe_for_teacher_beta",
        "safe_for_student_self_check",
        "events",
        "review_notes",
    ]:
        assert field in rubric
    event = rubric["events"][0]
    for field in [
        "accepted_evidence",
        "dependency",
        "follow_through_policy",
        "alternative_methods",
        "learning_target_ids",
        "review_notes",
    ]:
        assert field in event


def test_incomplete_approval_template_fails_validation_when_marked_approved(tmp_path: Path) -> None:
    paths = _review_batch_fixture(tmp_path)
    template = build_approval_template(review_batch_path=paths["batch"], first=True)
    rubric = template["rubric"]
    rubric["review_status"] = "approved"
    rubric["safe_for_auto_grade_lab"] = True
    rubric["safe_for_teacher_beta"] = True
    rubric["rubric_total_verified"] = True
    for event in rubric["events"]:
        event["review_status"] = "approved"
    reviewed = tmp_path / "reviewed.json"
    _write_reviewed(reviewed, rubric)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=reviewed,
        question_bank_path=paths["question_bank"],
        report_path=None,
        phase="2C",
    )

    assert report["ok"] is False
    assert any("placeholder" in error for error in report["errors"])


def test_reviewer_packet_and_template_cli_help_exit_successfully() -> None:
    for script in [
        "scripts/build_auto_grade_reviewer_packet.py",
        "scripts/extract_auto_grade_review_approval_template.py",
    ]:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout


def _review_batch_fixture(tmp_path: Path) -> dict[str, Path]:
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
    batch_path = tmp_path / "batch.json"
    draft_path = tmp_path / "draft.json"
    build_rubric_review_batch(
        question_bank_path=paths["question_bank"],
        review_queue_path=queue_path,
        mark_events_path=paths["mark_events"],
        eligible_items_path=paths["eligible"],
        output_path=batch_path,
        report_path=None,
        draft_output_path=draft_path,
        generated_at="2026-05-21T00:00:00Z",
    )
    paths["batch"] = batch_path
    paths["draft"] = draft_path
    return paths


def _write_reviewed(path: Path, *rubrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "exam_bank.auto_grade.reviewed_rubrics",
                "schema_version": 1,
                "generated_at": "2026-05-21T00:00:00Z",
                "source_question_bank_path": "fixture/question_bank.json",
                "source_mark_events_path": "fixture/question_bank.mark_events.v1.json",
                "rubric_count": len(rubrics),
                "event_count": sum(len(rubric["events"]) for rubric in rubrics),
                "summary": {},
                "rubrics": list(rubrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
