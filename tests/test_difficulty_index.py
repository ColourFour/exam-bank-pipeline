from __future__ import annotations

import json
from pathlib import Path

from exam_bank.difficulty_index import ALLOWED_CONFIDENCE, DIFFICULTY_INDEX_SCHEMA_NAME, DIFFICULTY_INDEX_SCHEMA_VERSION
from exam_bank.difficulty_index.sidecar import build_difficulty_index_sidecar


REQUIRED_FIELDS = {
    "question_id",
    "paper",
    "family",
    "component",
    "session",
    "year",
    "question_number",
    "difficulty_index_0_100",
    "paper_relative_percentile",
    "paper_relative_difficulty_band",
    "paper_relative_band_label",
    "confidence",
    "safe_for_teacher_filtering",
    "safe_for_student_sequencing",
    "evidence_used",
    "features",
    "warnings",
    "review_reasons",
    "unsafe_reasons",
}


def test_schema_shape_and_clamped_scores(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    sidecar = build_difficulty_index_sidecar(
        question_bank_path=paths["question_bank"],
        output_path=paths["output"],
        reports_dir=paths["reports"],
        artifact_root=paths["artifact_root"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        advisory_evidence_path=paths["advisory"],
        generated_at="2026-05-19T00:00:00Z",
    )

    assert paths["output"].exists()
    assert sidecar["schema_name"] == DIFFICULTY_INDEX_SCHEMA_NAME
    assert sidecar["schema_version"] == DIFFICULTY_INDEX_SCHEMA_VERSION
    assert sidecar["record_count"] == 15
    for record in sidecar["records"]:
        assert REQUIRED_FIELDS <= set(record)
        assert isinstance(record["difficulty_index_0_100"], int | float)
        assert 0 <= record["difficulty_index_0_100"] <= 100
        assert record["paper_relative_difficulty_band"] in {1, 2, 3, 4, 5}
        assert record["confidence"] in ALLOWED_CONFIDENCE
        assert isinstance(record["safe_for_teacher_filtering"], bool)
        assert isinstance(record["safe_for_student_sequencing"], bool)
        assert isinstance(record["warnings"], list)
        assert isinstance(record["unsafe_reasons"], list)


def test_paper_relative_banding_middle_weighted_remainders_and_tie_handling(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    sidecar = build_difficulty_index_sidecar(
        question_bank_path=paths["question_bank"],
        output_path=None,
        reports_dir=None,
        artifact_root=paths["artifact_root"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        advisory_evidence_path=paths["advisory"],
        generated_at="2026-05-19T00:00:00Z",
        dry_run=True,
    )

    paper_a = [record for record in sidecar["records"] if record["paper"] == "11summer26"]
    band_counts = {band: sum(1 for record in paper_a if record["paper_relative_difficulty_band"] == band) for band in range(1, 6)}
    assert band_counts == {1: 2, 2: 2, 3: 3, 4: 2, 5: 2}

    easiest = min(paper_a, key=lambda record: record["paper_relative_percentile"])
    hardest = max(paper_a, key=lambda record: record["paper_relative_percentile"])
    assert easiest["paper_relative_difficulty_band"] == 1
    assert hardest["paper_relative_difficulty_band"] == 5

    tied = [record for record in sidecar["records"] if record["paper"] == "12summer26"]
    assert [record["question_id"] for record in sorted(tied, key=lambda record: record["paper_relative_percentile"])] == [
        "12summer26_q04",
        "12summer26_q03",
        "12summer26_q02",
        "12summer26_q01",
    ]


def test_safety_gates_and_ambiguous_evidence_do_not_increase_confidence(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    sidecar = build_difficulty_index_sidecar(
        question_bank_path=paths["question_bank"],
        output_path=None,
        reports_dir=None,
        artifact_root=paths["artifact_root"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        advisory_evidence_path=paths["advisory"],
        generated_at="2026-05-19T00:00:00Z",
        dry_run=True,
    )
    records = {record["question_id"]: record for record in sidecar["records"]}

    assert records["11summer26_q10"]["safe_for_student_sequencing"] is False
    assert records["11summer26_q10"]["confidence"] == "unsafe"
    assert "total_marks_mismatch" in records["11summer26_q10"]["unsafe_reasons"]

    assert records["11summer26_q11"]["safe_for_student_sequencing"] is False
    assert records["11summer26_q11"]["confidence"] == "unsafe"
    assert "missing_mark_scheme_image" in records["11summer26_q11"]["unsafe_reasons"]

    assert records["12summer26_q04"]["confidence"] in {"low", "medium"}
    assert "topic_routing_review_or_low_confidence" in records["12summer26_q04"]["review_reasons"]


def test_report_generation_includes_identifying_band_information(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    build_difficulty_index_sidecar(
        question_bank_path=paths["question_bank"],
        output_path=paths["output"],
        reports_dir=paths["reports"],
        artifact_root=paths["artifact_root"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        advisory_evidence_path=paths["advisory"],
        generated_at="2026-05-19T00:00:00Z",
    )

    summary = paths["reports"] / "difficulty_index_summary.md"
    by_paper = paths["reports"] / "difficulty_index_by_paper.md"
    review = paths["reports"] / "difficulty_index_review_queue.md"
    assert summary.exists()
    assert by_paper.exists()
    assert review.exists()
    assert "Records scored: 15" in summary.read_text(encoding="utf-8")
    assert "`11summer26_q01`" in by_paper.read_text(encoding="utf-8")
    assert "Band" in by_paper.read_text(encoding="utf-8")
    assert "`11summer26_q10`" in review.read_text(encoding="utf-8")


def test_generation_is_deterministic_and_dry_run_does_not_write(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    kwargs = {
        "question_bank_path": paths["question_bank"],
        "output_path": paths["output"],
        "reports_dir": paths["reports"],
        "artifact_root": paths["artifact_root"],
        "mark_events_path": paths["mark_events"],
        "topic_routing_path": paths["topic_routing"],
        "advisory_evidence_path": paths["advisory"],
        "generated_at": "2026-05-19T00:00:00Z",
    }
    first = build_difficulty_index_sidecar(**kwargs, dry_run=True)
    second = build_difficulty_index_sidecar(**kwargs, dry_run=True)

    assert first == second
    assert not paths["output"].exists()
    assert not paths["reports"].exists()


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "output"
    question_bank = artifact_root / "json" / "question_bank.json"
    mark_events = artifact_root / "json" / "question_bank.mark_events.v1.json"
    topic_routing = artifact_root / "json" / "question_bank.topic_routing.v1.json"
    advisory = artifact_root / "advisory_evidence" / "question_bank.advisory_evidence.v1.json"
    output = artifact_root / "json" / "question_bank.difficulty_index.v1.json"
    reports = tmp_path / "reports"
    questions = []
    mark_records = []
    topic_records = {}
    advisory_records = []
    for idx in range(1, 12):
        question_id = f"11summer26_q{idx:02d}"
        q_image = f"p1/11summer26/questions/q{idx:02d}.png"
        ms_image = f"p1/11summer26/mark_scheme/q{idx:02d}.png"
        if idx != 11:
            _write_bytes(artifact_root / ms_image)
        _write_bytes(artifact_root / q_image)
        marks = idx if idx < 10 else 2
        questions.append(_question(question_id, "11summer26", idx, marks, q_image, ms_image))
        mark_records.append(
            _mark_event(
                question_id,
                "11summer26",
                idx,
                marks,
                total_match=idx != 10,
                source_exists=idx != 11,
            )
        )
        topic_records[question_id] = _topic(question_id, "11summer26", idx, confidence="high")
        advisory_records.append(_advisory(question_id, signal="hard" if idx == 9 else "neutral"))

    for idx in range(1, 5):
        question_id = f"12summer26_q{idx:02d}"
        q_image = f"p1/12summer26/questions/q{idx:02d}.png"
        ms_image = f"p1/12summer26/mark_scheme/q{idx:02d}.png"
        _write_bytes(artifact_root / q_image)
        _write_bytes(artifact_root / ms_image)
        legacy_scores = {1: 98, 2: 74, 3: 50, 4: 6}
        questions.append(_question(question_id, "12summer26", idx, 4, q_image, ms_image, legacy_score=legacy_scores[idx]))
        mark_records.append(_mark_event(question_id, "12summer26", idx, 4))
        topic_records[question_id] = _topic(
            question_id,
            "12summer26",
            idx,
            confidence="low" if idx == 4 else "high",
            review_required=idx == 4,
        )
        advisory_records.append(_advisory(question_id, signal="neutral"))

    _write_json(
        question_bank,
        {
            "schema_name": "exam_bank.question_bank",
            "schema_version": 2,
            "record_count": len(questions),
            "questions": questions,
        },
    )
    _write_json(mark_events, {"schema_name": "exam_bank.question_bank.mark_events", "schema_version": 1, "records": mark_records})
    _write_json(topic_routing, {"schema_name": "exam_bank.question_bank.topic_routing", "schema_version": 1, "records": topic_records})
    _write_json(advisory, {"schema": "exam_bank.question_bank.advisory_evidence.v1", "records": advisory_records})
    return {
        "artifact_root": artifact_root,
        "question_bank": question_bank,
        "mark_events": mark_events,
        "topic_routing": topic_routing,
        "advisory": advisory,
        "output": output,
        "reports": reports,
    }


def _question(
    question_id: str,
    paper: str,
    question_number: int,
    marks: int,
    question_image: str,
    mark_scheme_image: str,
    *,
    legacy_score: int | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "paper": paper,
        "paper_family": "p1",
        "question_number": str(question_number),
        "question_image_path": question_image,
        "mark_scheme_image_path": mark_scheme_image,
        "question_text": f"{question_number} Solve the problem. [{marks}]",
        "mark_scheme_text": "M1\nA1",
        "question_solution_marks": marks,
        "difficulty_score": legacy_score if legacy_score is not None else min(95, 20 + question_number * 5),
        "subparts": ["a", "b"] if marks > 6 else [],
        "notes": {
            "source_paper_code": paper[:2],
            "mapping_status": "pass",
            "question_structure_detected": {"question_total_detected": marks, "subparts": []},
            "mark_scheme_structure_detected": {"mark_scheme_total_detected": marks},
        },
    }


def _mark_event(
    question_id: str,
    paper: str,
    question_number: int,
    marks: int,
    *,
    total_match: bool = True,
    source_exists: bool = True,
) -> dict:
    return {
        "question_id": question_id,
        "paper": paper,
        "paper_family": "p1",
        "question_number": str(question_number),
        "source_mark_scheme_image_exists": source_exists,
        "safe_for_advisory_use": total_match and source_exists,
        "total_marks_detected": marks + (1 if not total_match else 0),
        "total_marks_expected": marks,
        "question_total_detected": marks,
        "total_marks_match": total_match,
        "review_flags": [] if total_match else ["total_marks_mismatch"],
        "mark_events": [{"mark_type": "method", "mark_value": 1, "is_dependent": False} for _ in range(max(1, marks))],
    }


def _topic(question_id: str, paper: str, question_number: int, *, confidence: str, review_required: bool = False) -> dict:
    return {
        "question_id": question_id,
        "paper": paper,
        "question_number": str(question_number),
        "primary_topic_id": "9709_p1_topic_algebra",
        "confidence": confidence,
        "review_required": review_required,
    }


def _advisory(question_id: str, *, signal: str) -> dict:
    item_signal = "" if signal == "neutral" else signal
    return {
        "question_id": question_id,
        "advisory_evidence": {
            "grade_threshold_context": {
                "available": True,
                "component_context_label": "paper_context_typical",
                "warnings": [],
            }
        },
        "examiner_report_difficulty": {
            "item_signal": item_signal,
            "confidence": "medium",
            "review_required": False,
            "warnings": [],
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_bytes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
