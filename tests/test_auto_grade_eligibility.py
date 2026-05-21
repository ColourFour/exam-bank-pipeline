from __future__ import annotations

import json
from pathlib import Path

from exam_bank.auto_grade.eligibility import build_eligible_items


def test_builder_classifies_tiny_fixture_and_does_not_mutate_input(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    original = paths["question_bank"].read_text(encoding="utf-8")

    payload = build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=paths["eligible"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
    )

    assert paths["eligible"].exists()
    assert paths["question_bank"].read_text(encoding="utf-8") == original
    assert payload["record_count"] == 4
    by_id = {item["question_id"]: item for item in payload["items"]}
    assert by_id["11summer26_q01"]["eligibility_status"] == "review_only"
    assert "missing_reviewed_rubric" in by_id["11summer26_q01"]["block_reasons"]
    assert by_id["11summer26_q02"]["eligibility_status"] == "blocked"
    assert "missing_canonical_question_image" in by_id["11summer26_q02"]["block_reasons"]
    assert by_id["11summer26_q03"]["eligibility_status"] == "blocked"
    assert "missing_canonical_mark_scheme_image" in by_id["11summer26_q03"]["block_reasons"]
    assert by_id["11summer26_q04"]["eligibility_status"] == "blocked"
    assert "advisory_mark_event_total_mismatch" in by_id["11summer26_q04"]["block_reasons"]


def test_advisory_mark_events_alone_do_not_permit_student_safe_status(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    payload = build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=None,
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["missing_reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
        dry_run=True,
    )

    statuses = {item["eligibility_status"] for item in payload["items"]}
    assert "student_ready" not in statuses
    assert "student_self_check_beta" not in statuses
    assert payload["summary"]["student_ready_count"] == 0
    assert all(
        "advisory_mark_events_not_scoring_contract" in set(item["block_reasons"])
        for item in payload["items"]
    )


def test_reviewed_rubric_can_only_unlock_teacher_beta_not_student_ready_by_default(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    paths["reviewed_rubrics"].write_text(
        json.dumps(
            {
                "schema": "exam_bank.auto_grade.reviewed_rubrics",
                "schema_version": 1,
                "rubric_count": 1,
                "event_count": 2,
                "rubrics": [
                    {
                        "rubric_id": "rubric-1",
                        "source_question_id": "11summer26_q01",
                        "source_question_image_path": "p1/11summer26/questions/q01.png",
                        "source_mark_scheme_image_path": "p1/11summer26/mark_scheme/q01.png",
                        "paper": "11summer26",
                        "paper_family": "p1",
                        "question_number": "01",
                        "part_path": [],
                        "total_marks": 4,
                        "rubric_total_verified": True,
                        "safe_for_auto_grade_lab": True,
                        "safe_for_teacher_beta": True,
                        "safe_for_student_self_check": False,
                        "review_status": "approved",
                        "reviewed_by": "reviewer",
                        "reviewed_at": "2026-05-21T00:00:00Z",
                        "approval_scope": "teacher_beta",
                        "events": [
                            _rubric_event("rubric-1-e1", "M", 2),
                            _rubric_event("rubric-1-e2", "A", 2),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=None,
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
        dry_run=True,
    )

    by_id = {item["question_id"]: item for item in payload["items"]}
    assert by_id["11summer26_q01"]["eligibility_status"] == "teacher_beta"
    assert payload["summary"]["student_ready_count"] == 0


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "output"
    question_bank = artifact_root / "json" / "question_bank.json"
    eligible = artifact_root / "auto_grade" / "eligible_items.v1.json"
    reviewed_rubrics = artifact_root / "auto_grade" / "reviewed_rubrics.v1.json"
    mark_events = artifact_root / "json" / "question_bank.mark_events.v1.json"
    topic_routing = artifact_root / "json" / "question_bank.topic_routing.v1.json"
    reviewed_rubrics.parent.mkdir(parents=True, exist_ok=True)
    for relative in [
        "p1/11summer26/questions/q01.png",
        "p1/11summer26/mark_scheme/q01.png",
        "p1/11summer26/mark_scheme/q02.png",
        "p1/11summer26/questions/q03.png",
        "p1/11summer26/questions/q04.png",
        "p1/11summer26/mark_scheme/q04.png",
    ]:
        path = artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    questions = [
        _question("11summer26_q01", "q01", 4),
        _question("11summer26_q02", "q02", 4),
        _question("11summer26_q03", "q03", 4),
        _question("11summer26_q04", "q04", 4),
    ]
    question_bank.parent.mkdir(parents=True, exist_ok=True)
    question_bank.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.question_bank",
                "schema_version": 2,
                "record_count": len(questions),
                "questions": questions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    mark_events.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.question_bank.mark_events",
                "schema_version": 1,
                "record_count": 4,
                "records": [
                    _mark_event("11summer26_q01", total_match=True),
                    _mark_event("11summer26_q02", total_match=True),
                    _mark_event("11summer26_q03", total_match=True),
                    _mark_event("11summer26_q04", total_match=False),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    topic_routing.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.topic_routing_sidecar",
                "schema_version": 1,
                "record_count": 4,
                "records": {
                    question["question_id"]: {
                        "primary_topic_id": "9709_p1_topic_algebra",
                        "confidence": "high",
                        "review_required": False,
                    }
                    for question in questions
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "artifact_root": artifact_root,
        "question_bank": question_bank,
        "eligible": eligible,
        "reviewed_rubrics": reviewed_rubrics,
        "missing_reviewed_rubrics": tmp_path / "missing.json",
        "mark_events": mark_events,
        "topic_routing": topic_routing,
    }


def _question(question_id: str, q_label: str, marks: int) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "11summer26",
        "paper_family": "p1",
        "question_number": q_label.removeprefix("q"),
        "canonical_question_artifact": f"p1/11summer26/questions/{q_label}.png",
        "question_image_path": f"p1/11summer26/questions/{q_label}.png",
        "mark_scheme_image_path": f"p1/11summer26/mark_scheme/{q_label}.png",
        "question_solution_marks": marks,
        "notes": {"mapping_status": "pass"},
    }


def _mark_event(question_id: str, *, total_match: bool) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper_id": "11summer26",
        "paper": "11summer26",
        "paper_family": "p1",
        "question_number": question_id.rsplit("_q", 1)[-1],
        "part_path": [],
        "source_mark_scheme_image_path": f"p1/11summer26/mark_scheme/q{question_id.rsplit('_q', 1)[-1]}.png",
        "extraction_status": "parsed",
        "safe_for_advisory_use": True,
        "safe_for_marking_use": False,
        "total_marks_detected": 4 if total_match else 3,
        "total_marks_expected": 4,
        "question_total_detected": 4,
        "total_marks_match": total_match,
        "review_flags": [] if total_match else ["total_marks_mismatch"],
        "mark_events": [
            _source_mark_event(question_id, "me0001", "M2", "method", 2),
            _source_mark_event(question_id, "me0002", "A2", "accuracy", 2),
        ],
    }


def _source_mark_event(
    question_id: str,
    event_suffix: str,
    mark_code_raw: str,
    mark_type: str,
    mark_value: int,
) -> dict[str, object]:
    return {
        "event_id": f"{question_id}_{event_suffix}",
        "part_path": [],
        "raw_text": f"{mark_code_raw} fixture mark scheme line",
        "normalized_text": f"{mark_code_raw} fixture mark scheme line",
        "mark_code_raw": mark_code_raw,
        "mark_type": mark_type,
        "mark_value": mark_value,
        "is_follow_through": False,
        "is_dependent": False,
        "depends_on_event_ids": [],
        "alternative_group_id": None,
        "condition_text": "",
        "answer_text": "fixture answer",
        "common_error_text": "",
        "confidence": "high",
        "review_flags": [],
    }


def _rubric_event(event_id: str, mark_code: str, mark_value: int) -> dict[str, object]:
    return {
        "event_id": event_id,
        "part_path": [],
        "mark_code": mark_code,
        "mark_type": mark_code,
        "mark_value": mark_value,
        "dependency": "independent",
        "follow_through_policy": "none",
        "accepted_evidence": ["human-reviewed mark-scheme evidence"],
        "common_errors": [],
        "alternative_methods": [],
        "learning_target_ids": ["9709_p1_topic_algebra"],
        "review_status": "approved",
        "review_notes": "fixture",
    }
