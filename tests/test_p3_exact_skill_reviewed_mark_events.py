from __future__ import annotations

import json
from pathlib import Path

from exam_bank.p3_exact_skill.reviewed_mark_events import (
    validate_reviewed_mark_events,
    validate_reviewed_mark_events_payload,
)


def test_populated_reviewed_mark_event_artifact_validates() -> None:
    report = validate_reviewed_mark_events(
        reviewed_mark_events_path="data/review/p3_exact_skill_reviewed_mark_events.v1.json",
        question_bank_path="output/json/question_bank.json",
        mark_events_path="output/json/question_bank.mark_events.v1.json",
    )

    assert report["ok"] is True
    assert report["decision_count"] == 11


def test_invalid_mark_event_status_fails(tmp_path: Path) -> None:
    errors, _ = validate_reviewed_mark_events_payload(
        _payload(_decision(tmp_path, status="promoted")),
        question_ids={"q1"},
        mark_event_ids={"q1_me0001"},
        base_dir=tmp_path,
        require_event_ids_in_mark_events=True,
    )

    assert any("invalid_status:promoted" in error for error in errors)


def test_duplicate_decision_ids_fail(tmp_path: Path) -> None:
    errors, _ = validate_reviewed_mark_events_payload(
        _payload(
            _decision(tmp_path, decision_id="decision-1", event_id="q1_me0001"),
            _decision(tmp_path, decision_id="decision-1", event_id="q1_me0002"),
        ),
        question_ids={"q1"},
        mark_event_ids={"q1_me0001", "q1_me0002"},
        base_dir=tmp_path,
        require_event_ids_in_mark_events=True,
    )

    assert any("duplicate_decision_id" in error for error in errors)


def test_duplicate_event_ids_fail(tmp_path: Path) -> None:
    errors, _ = validate_reviewed_mark_events_payload(
        _payload(
            _decision(tmp_path, decision_id="decision-1", event_id="q1_me0001"),
            _decision(tmp_path, decision_id="decision-2", event_id="q1_me0001"),
        ),
        question_ids={"q1"},
        mark_event_ids={"q1_me0001"},
        base_dir=tmp_path,
        require_event_ids_in_mark_events=True,
    )

    assert any("duplicate_event_id" in error for error in errors)


def test_missing_required_fields_fail(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    del decision["rationale"]
    del decision["reviewer"]

    errors, _ = validate_reviewed_mark_events_payload(
        _payload(decision),
        question_ids={"q1"},
        mark_event_ids={"q1_me0001"},
        base_dir=tmp_path,
        require_event_ids_in_mark_events=True,
    )

    assert any("missing_required_field:rationale" in error for error in errors)
    assert any("missing_required_field:reviewer" in error for error in errors)


def test_advisory_and_rejected_cannot_satisfy_generation_gate(tmp_path: Path) -> None:
    advisory = _decision(tmp_path, status="advisory")
    advisory["satisfies_generation_gate"] = True
    rejected = _decision(tmp_path, decision_id="decision-2", event_id="q1_me0002", status="rejected")
    rejected["satisfies_generation_gate"] = True

    errors, _ = validate_reviewed_mark_events_payload(
        _payload(advisory, rejected),
        question_ids={"q1"},
        mark_event_ids={"q1_me0001", "q1_me0002"},
        base_dir=tmp_path,
        require_event_ids_in_mark_events=True,
    )

    assert any("advisory_cannot_satisfy_generation_gate" in error for error in errors)
    assert any("rejected_cannot_satisfy_generation_gate" in error for error in errors)


def test_question_and_image_references_are_checked(tmp_path: Path) -> None:
    decision = _decision(tmp_path, source_question_id="missing")
    decision["question_image_path"] = "missing-question.png"

    errors, _ = validate_reviewed_mark_events_payload(
        _payload(decision),
        question_ids={"q1"},
        mark_event_ids={"q1_me0001"},
        base_dir=tmp_path,
        require_event_ids_in_mark_events=True,
    )

    assert any("source_question_id_not_found:missing" in error for error in errors)
    assert any("question_image_path_not_found:missing-question.png" in error for error in errors)


def test_validator_cli_accepts_populated_fixture() -> None:
    report = validate_reviewed_mark_events(
        reviewed_mark_events_path=Path("data/review/p3_exact_skill_reviewed_mark_events.v1.json"),
        question_bank_path=Path("output/json/question_bank.json"),
        mark_events_path=Path("output/json/question_bank.mark_events.v1.json"),
    )

    assert json.dumps(report)
    assert report["ok"] is True
    assert report["decision_count"] == 11


def _payload(*decisions: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "exam_bank.p3_exact_skill.reviewed_mark_events",
        "schema_version": 1,
        "decision_count": len(decisions),
        "decisions": list(decisions),
    }


def _decision(
    tmp_path: Path,
    *,
    decision_id: str = "decision-1",
    event_id: str = "q1_me0001",
    source_question_id: str = "q1",
    status: str = "approved",
) -> dict[str, object]:
    question_image = tmp_path / "question.png"
    mark_scheme_image = tmp_path / "mark.png"
    question_image.write_bytes(b"question")
    mark_scheme_image.write_bytes(b"mark")
    return {
        "schema_version": 1,
        "decision_id": decision_id,
        "event_id": event_id,
        "source_question_id": source_question_id,
        "part_path": ["a"],
        "question_image_path": str(question_image),
        "mark_scheme_image_path": str(mark_scheme_image),
        "reviewer": "fixture_reviewer",
        "reviewed_at": "2026-05-26T00:00:00Z",
        "status": status,
        "rationale": "Fixture rationale.",
        "notes": [],
    }
