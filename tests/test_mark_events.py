from __future__ import annotations

import json
from pathlib import Path

from exam_bank.mark_events import MARK_EVENTS_SCHEMA_NAME, MARK_EVENTS_SCHEMA_VERSION
from exam_bank.mark_events.parsing import parse_mark_scheme_text
from exam_bank.mark_events.sidecar import build_mark_events_sidecar, sidecar_summary
from exam_bank.mark_events.validation import validate_mark_events


def test_parser_extracts_mark_codes_conditions_dependencies_and_subparts() -> None:
    result = parse_mark_scheme_text(
        """
1(a) Use a valid method M1 oe
Obtain x = 2 A1 cao www isw
1(b)(i) State the reason B1 AG
1(b)(ii) Correct answer from their value A1ft
Alternative method using substitution DM1
Do not allow decimal approximation.
""",
        question_id="12spring21_q01",
    )

    events = result.events
    assert [event["mark_code_raw"] for event in events] == ["M1", "A1", "B1", "A1FT", "DM1"]
    assert [event["mark_type"] for event in events] == [
        "method",
        "accuracy",
        "answer_given",
        "follow_through",
        "dependent_method",
    ]
    assert [event["part_path"] for event in events] == [["a"], ["a"], ["b", "i"], ["b", "ii"], ["b", "ii"]]
    assert events[0]["mark_value"] == 1
    assert events[3]["is_follow_through"] is True
    assert events[4]["is_dependent"] is True
    assert events[4]["depends_on_event_ids"] == [events[3]["event_id"]]
    assert events[4]["alternative_group_id"] == "alt-001"
    assert "oe" in events[0]["condition_text"]
    assert "cao" in events[1]["condition_text"]
    assert "www" in events[1]["condition_text"]
    assert "isw" in events[1]["condition_text"]
    assert "AG" in events[2]["condition_text"]
    assert result.unparsed_evidence[0]["reason"] == "semantic_line_without_deterministic_mark_code"


def test_parser_flags_ambiguous_dependent_and_unknown_codes() -> None:
    result = parse_mark_scheme_text(
        """
1(a) Some result C1
1(b) Depends on previous work DM1
""",
        question_id="12spring21_q02",
    )

    assert result.events[0]["mark_type"] == "unknown"
    assert "unknown_mark_code" in result.events[0]["review_flags"]
    assert result.events[1]["mark_type"] == "dependent_method"
    assert "dependent_mark_without_deterministic_prior_event" in result.events[1]["review_flags"]


def test_sidecar_generation_is_deterministic_and_does_not_mutate_question_bank(tmp_path: Path) -> None:
    question_bank_path, artifact_root = _write_question_bank_fixture(tmp_path)
    original = question_bank_path.read_text(encoding="utf-8")

    first = build_mark_events_sidecar(
        question_bank_path=question_bank_path,
        artifact_root=artifact_root,
        generated_at="2026-05-19T00:00:00Z",
        dry_run=True,
    )
    second = build_mark_events_sidecar(
        question_bank_path=question_bank_path,
        artifact_root=artifact_root,
        generated_at="2026-05-19T00:00:00Z",
        dry_run=True,
    )

    assert first == second
    assert question_bank_path.read_text(encoding="utf-8") == original
    assert first["schema_name"] == MARK_EVENTS_SCHEMA_NAME
    assert first["schema_version"] == MARK_EVENTS_SCHEMA_VERSION
    assert first["record_count"] == 3
    q1, q2, q3 = first["records"]
    assert q1["safe_for_marking_use"] is False
    assert q1["safe_for_advisory_use"] is True
    assert q1["total_marks_match"] is True
    assert [event["mark_code_raw"] for event in q1["mark_events"]] == ["M1", "A1", "B1"]
    assert q2["total_marks_match"] is False
    assert "total_marks_mismatch" in q2["review_flags"]
    assert q2["safe_for_advisory_use"] is False
    assert q3["source_mark_scheme_image_exists"] is False
    assert "mark_scheme_image_file_missing" in q3["review_flags"]
    assert q3["safe_for_advisory_use"] is False


def test_validation_accepts_review_records_but_rejects_marking_claims_and_excess_totals(tmp_path: Path) -> None:
    question_bank_path, artifact_root = _write_question_bank_fixture(tmp_path, include_missing_image=False)
    sidecar = build_mark_events_sidecar(
        question_bank_path=question_bank_path,
        artifact_root=artifact_root,
        generated_at="2026-05-19T00:00:00Z",
        dry_run=True,
    )
    sidecar_path = tmp_path / "output" / "json" / "question_bank.mark_events.v1.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    report = validate_mark_events(
        question_bank_path=question_bank_path,
        mark_events_path=sidecar_path,
        artifact_root=artifact_root,
    )

    assert report["ok"] is True
    assert any("parsed_total_mismatch" in warning for warning in report["warnings"])

    sidecar["records"][0]["safe_for_marking_use"] = True
    sidecar["records"][0]["total_marks_detected"] = 9
    sidecar["records"][0]["total_marks_expected"] = 3
    sidecar["records"][0]["total_marks_match"] = False
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    report = validate_mark_events(
        question_bank_path=question_bank_path,
        mark_events_path=sidecar_path,
        artifact_root=artifact_root,
    )

    assert report["ok"] is False
    assert any(error.startswith("unsafe_marking_claim") for error in report["errors"])
    assert any(error.startswith("parsed_total_exceeds_expected") for error in report["errors"])


def test_targeted_total_mismatch_repairs_and_correction_are_review_safe(tmp_path: Path) -> None:
    question_bank_path, artifact_root = _write_targeted_total_fixture(tmp_path)
    original = question_bank_path.read_text(encoding="utf-8")

    sidecar = build_mark_events_sidecar(
        question_bank_path=question_bank_path,
        artifact_root=artifact_root,
        generated_at="2026-05-19T00:00:00Z",
        dry_run=True,
    )

    records = {record["question_id"]: record for record in sidecar["records"]}
    assert records["51summer23_q04"]["question_total_detected"] == 9
    assert records["51summer23_q04"]["total_marks_detected"] == 9
    assert records["51summer23_q04"]["total_marks_expected"] == 9
    assert records["51summer23_q04"]["question_total_evidence"]["source"] == "ocr_text"

    assert records["51summer24_q02"]["question_total_detected"] == 7
    assert records["51summer24_q02"]["total_marks_detected"] == 7
    assert records["51summer24_q02"]["total_marks_expected"] == 7
    assert records["51summer24_q02"]["question_total_evidence"]["source"] == "ocr_text"

    q10 = records["32autumn25_q10"]
    assert q10["question_total_detected"] == 4
    assert q10["total_marks_detected"] == 4
    assert q10["total_marks_expected"] == 4
    assert q10["human_verified_total_correction"]["status"] == "human_verified"
    assert "human_verified_total_correction" in q10["review_flags"]
    assert q10["safe_for_advisory_use"] is False

    summary = sidecar_summary(sidecar)
    assert summary["question_total_disagreement_count"] == 0
    assert summary["question_total_repair_count"] == 2
    assert summary["human_verified_total_correction_count"] == 1
    assert summary["question_total_disagreement_resolved_count"] == 3
    assert all(record["safe_for_marking_use"] is False for record in sidecar["records"])
    assert all(
        "question_total_mark_scheme_total_disagree" not in set(record["review_flags"]) for record in sidecar["records"]
    )
    assert question_bank_path.read_text(encoding="utf-8") == original


def test_safe_for_advisory_is_not_granted_by_human_total_correction(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    image = artifact_root / "p3" / "32autumn25" / "mark_scheme" / "q10.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"mark-image")
    payload = {
        "schema_name": "exam_bank.question_bank",
        "schema_version": 2,
        "record_count": 1,
        "questions": [
            {
                "question_id": "32autumn25_q10",
                "paper": "32autumn25",
                "paper_family": "p3",
                "question_number": "10",
                "mark_scheme_image_path": "p3/32autumn25/mark_scheme/q10.png",
                "mark_scheme_image_paths": ["p3/32autumn25/mark_scheme/q10.png"],
                "question_text": "10 (a) Show that dh/dt = (500 - h^2)/250. [4]",
                "mark_scheme_text": "10(a) State equation B1\nUse chain rule M1\nObtain k DM1\nObtain result A1 AG",
                "question_solution_marks": 9,
                "notes": {
                    "source_paper_code": "32",
                    "mapping_status": "pass",
                    "mark_scheme_total_detected": 9,
                    "question_total_detected": 4,
                    "question_structure_detected": {"subparts": ["a", "b"], "mark_values_detected": [4]},
                },
            }
        ],
    }
    question_bank_path = tmp_path / "output" / "json" / "question_bank.json"
    question_bank_path.parent.mkdir(parents=True)
    question_bank_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    sidecar = build_mark_events_sidecar(
        question_bank_path=question_bank_path,
        artifact_root=artifact_root,
        generated_at="2026-05-19T00:00:00Z",
        dry_run=True,
    )

    record = sidecar["records"][0]
    assert record["total_marks_match"] is True
    assert record["safe_for_advisory_use"] is False
    assert record["extraction_status"] == "review"


def _write_question_bank_fixture(tmp_path: Path, *, include_missing_image: bool = True) -> tuple[Path, Path]:
    artifact_root = tmp_path / "output"
    good_image = artifact_root / "p1" / "12spring21" / "mark_scheme" / "q01.png"
    good_image.parent.mkdir(parents=True)
    good_image.write_bytes(b"mark-image")
    second_image = artifact_root / "p1" / "12spring21" / "mark_scheme" / "q02.png"
    second_image.write_bytes(b"mark-image-two")
    records = [
        _record(
            "12spring21_q01",
            "1",
            "p1/12spring21/mark_scheme/q01.png",
            "1(a) Valid method M1\nObtain x = 2 A1 cao\n1(b) State reason B1",
            question_solution_marks=3,
            mark_scheme_total_detected=3,
            question_total_detected=3,
        ),
        _record(
            "12spring21_q02",
            "2",
            "p1/12spring21/mark_scheme/q02.png",
            "2 Use method M1\nObtain x = 2 A1",
            question_solution_marks=4,
            mark_scheme_total_detected=3,
            question_total_detected=4,
        ),
    ]
    if include_missing_image:
        records.append(
            _record(
                "12spring21_q03",
                "3",
                "p1/12spring21/mark_scheme/q03.png",
                "3 Explain why E1",
                question_solution_marks=1,
                mark_scheme_total_detected=1,
                question_total_detected=1,
            )
        )
    payload = {
        "schema_name": "exam_bank.question_bank",
        "schema_version": 2,
        "record_count": len(records),
        "questions": records,
    }
    question_bank_path = tmp_path / "output" / "json" / "question_bank.json"
    question_bank_path.parent.mkdir(parents=True)
    question_bank_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return question_bank_path, artifact_root


def _record(
    question_id: str,
    question_number: str,
    mark_scheme_image_path: str,
    mark_scheme_text: str,
    *,
    question_solution_marks: int,
    mark_scheme_total_detected: int,
    question_total_detected: int,
) -> dict:
    return {
        "question_id": question_id,
        "paper": "12spring21",
        "paper_family": "p1",
        "question_number": question_number,
        "mark_scheme_image_path": mark_scheme_image_path,
        "mark_scheme_image_paths": [mark_scheme_image_path],
        "mark_scheme_text": mark_scheme_text,
        "question_solution_marks": question_solution_marks,
        "notes": {
            "source_paper_code": "12",
            "mapping_status": "pass",
            "mapping_failure_reason": "",
            "mark_scheme_total_detected": mark_scheme_total_detected,
            "question_total_detected": question_total_detected,
        },
    }


def _write_targeted_total_fixture(tmp_path: Path) -> tuple[Path, Path]:
    artifact_root = tmp_path / "output"
    for relative in [
        "p5/51summer23/mark_scheme/q04.png",
        "p5/51summer24/mark_scheme/q02.png",
        "p3/32autumn25/mark_scheme/q10.png",
    ]:
        image = artifact_root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"mark-image")
    records = [
        _targeted_record(
            question_id="51summer23_q04",
            paper="51summer23",
            paper_family="p5",
            question_number="4",
            mark_scheme_image_path="p5/51summer23/mark_scheme/q04.png",
            question_text="4 (a) Find the expected count. (b) Find mu and sigma. [5]",
            ocr_text="4 (a) Find the expected count. [4] (b) Find mu and sigma. [5]",
            mark_scheme_text=(
                "4(a) Use standardisation M1\nFind probability M1\nObtain probability A1\n"
                "Special case B1\nObtain count B1 FT\n"
                "4(b) Set equations M1 B1 B1\nSolve equations M1\nObtain values A1"
            ),
            question_solution_marks=9,
            mark_scheme_total_detected=9,
            question_total_detected=5,
        ),
        _targeted_record(
            question_id="51summer24_q02",
            paper="51summer24",
            paper_family="p5",
            question_number="2",
            mark_scheme_image_path="p5/51summer24/mark_scheme/q02.png",
            question_text="2 (a) Find the probability. (b) Find sigma. [3]",
            ocr_text="2 (a) Find the probability. [4] (b) Find sigma. [3]",
            mark_scheme_text=(
                "2(a) Use standardisation M1\nCorrect formula A1\nFind area M1\nObtain answer A1\n"
                "2(b) Critical value B1 cao\nUse standardisation M1\nObtain sigma A1"
            ),
            question_solution_marks=7,
            mark_scheme_total_detected=7,
            question_total_detected=3,
        ),
        _targeted_record(
            question_id="32autumn25_q10",
            paper="32autumn25",
            paper_family="p3",
            question_number="10",
            mark_scheme_image_path="p3/32autumn25/mark_scheme/q10.png",
            question_text="10 (a) Show that dh/dt = (500 - h^2)/250. [4]",
            ocr_text="10 (a) Show that dh/dt = (500 - h^2)/250. [4]",
            mark_scheme_text=(
                "10(a) State dV/dt B1\nUse chain rule M1\nUse h = 20 DM1\nObtain result A1 AG\n"
                "10(b) Separate variables B1\nObtain t term B1\nObtain log term B1\nUse constant M1\n"
                "Obtain t = 16.1 A1"
            ),
            question_solution_marks=9,
            mark_scheme_total_detected=9,
            question_total_detected=4,
        ),
    ]
    payload = {
        "schema_name": "exam_bank.question_bank",
        "schema_version": 2,
        "record_count": len(records),
        "questions": records,
    }
    question_bank_path = tmp_path / "output" / "json" / "question_bank.json"
    question_bank_path.parent.mkdir(parents=True)
    question_bank_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return question_bank_path, artifact_root


def _targeted_record(
    *,
    question_id: str,
    paper: str,
    paper_family: str,
    question_number: str,
    mark_scheme_image_path: str,
    question_text: str,
    ocr_text: str,
    mark_scheme_text: str,
    question_solution_marks: int,
    mark_scheme_total_detected: int,
    question_total_detected: int,
) -> dict:
    return {
        "question_id": question_id,
        "paper": paper,
        "paper_family": paper_family,
        "question_number": question_number,
        "mark_scheme_image_path": mark_scheme_image_path,
        "mark_scheme_image_paths": [mark_scheme_image_path],
        "question_text": question_text,
        "ocr_text": ocr_text,
        "mark_scheme_text": mark_scheme_text,
        "question_solution_marks": question_solution_marks,
        "notes": {
            "source_paper_code": paper[:2],
            "mapping_status": "pass",
            "mapping_failure_reason": "",
            "mark_scheme_total_detected": mark_scheme_total_detected,
            "question_total_detected": question_total_detected,
            "question_structure_detected": {
                "subparts": ["a", "b"],
                "mark_values_detected": [question_total_detected],
                "question_total_detected": question_total_detected,
            },
            "mark_scheme_structure_detected": {
                "question_total_detected": question_total_detected,
                "mark_scheme_total_detected": mark_scheme_total_detected,
            },
        },
    }
