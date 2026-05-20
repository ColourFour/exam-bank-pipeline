import json
from copy import deepcopy
from pathlib import Path

import pytest

from exam_bank.text_review_queue import build_review_queue, load_review_state, score_record


def base_record(**overrides):
    record = {
        "question_id": "sample_q03",
        "paper": "sample",
        "paper_family": "p1",
        "question_number": "3",
        "question_solution_marks": 5,
        "question_text": "3 (a) Find x. [2] (b) Hence find y. [3]",
        "ocr_text": "3 (a) Find x. [2] (b) Hence find y. [3]",
        "subparts": ["a", "b"],
        "subparts_solution_marks": {"a": 2, "b": 3},
        "question_text_trust": "high",
        "question_image_path": "p1/sample/questions/q03.png",
        "mark_scheme_image_path": "p1/sample/mark_scheme/q03.png",
        "notes": {
            "native_text_score": 95,
            "ocr_text_score": 95,
            "selected_text_score": 95,
            "text_candidate_source": "native",
            "ocr_selected": False,
            "text_fidelity_status": "clean",
            "question_crop_confidence": "high",
            "question_structure_detected": {
                "subparts": ["a", "b"],
                "mark_values_detected": [2, 3],
                "text_length": 44,
                "contamination_detected": False,
                "contamination_indicators": {"foreign_question_anchors": []},
            },
            "review_flags": [],
            "validation_flags": [],
            "extraction_quality_flags": [],
            "ocr_rejected_reasons": [],
        },
    }
    record.update(overrides)
    return record


def reason_codes(record, fixtures=None):
    return set(score_record(record, fixtures or set())["reason_codes"])


def test_score_generates_structural_reason_codes() -> None:
    record = base_record(
        question_text="Solve sin 0 = 0.",
        ocr_text="3 (a) Solve sin theta = 0. [2] (b) Hence find y. [3]",
        notes={
            **base_record()["notes"],
            "native_text_score": 20,
            "ocr_text_score": 91,
            "selected_text_score": 20,
            "text_fidelity_status": "degraded",
            "extraction_quality_flags": ["flattened_display_math", "math_corruption_suspected"],
            "ocr_rejected_reasons": ["ocr_lost_greek_symbol"],
        },
    )

    codes = reason_codes(record)

    assert "ocr_native_disagreement" in codes
    assert "missing_question_number" in codes
    assert "missing_expected_question_number" in codes
    assert "missing_marks" in codes
    assert "missing_mark_bracket" in codes
    assert "lost_subpart_labels" in codes
    assert "missing_subpart_labels" in codes
    assert "suspiciously_short_text" in codes
    assert "suspiciously_short_selected_text" in codes
    assert "likely_math_symbol_loss" in codes
    assert "clean_visual_crop_but_degraded_text" in codes


def test_selected_ocr_with_structural_warning_is_prioritized() -> None:
    record = base_record(
        notes={
            **base_record()["notes"],
            "ocr_selected": True,
            "text_candidate_source": "ocr",
            "review_flags": ["weak_question_text"],
        }
    )

    scored = score_record(record)

    assert "selected_ocr_with_structural_warnings" in scored["reason_codes"]
    assert scored["priority_score"] >= 70


def test_next_question_contamination_reason_from_text_and_metadata() -> None:
    text_record = base_record(question_text="3 Find x. [4] 4 Solve the equation. [5]")
    metadata_record = base_record(
        question_text="3 Find x. [4]",
        notes={
            **base_record()["notes"],
            "question_structure_detected": {
                **base_record()["notes"]["question_structure_detected"],
                "contamination_detected": True,
                "contamination_indicators": {"foreign_question_anchors": ["4"]},
            },
        },
    )

    assert "possible_next_question_contamination" in reason_codes(text_record)
    assert "possible_next_question_contamination" in reason_codes(metadata_record)


def test_fixture_membership_boosts_known_bad_records_near_top() -> None:
    fixture = base_record(question_id="fixture_q03", question_text="Find x.", ocr_text="3 Find x. [5]")
    non_fixture = base_record(question_id="ordinary_q04", question_number="4")

    report = build_review_queue([non_fixture, fixture], {"fixture_q03"})

    assert report["entries"][0]["record_id"] == "fixture_q03"
    assert report["entries"][0]["reason_codes"][0] == "known_fixture_membership"
    assert report["fixture_summary"]["known_fixtures_in_top_50"] == 1
    assert report["fixture_summary"]["known_fixtures_in_top_n"] == 1


def test_top_50_entries_are_explainable_by_reason_codes() -> None:
    records = [
        base_record(question_id=f"fixture_q{i:02d}", question_number=str(i), question_text=f"{i} Find x. [1]")
        for i in range(1, 4)
    ]

    report = build_review_queue(records, {"fixture_q01", "fixture_q02"})

    for entry in report["top_50"]:
        if entry["priority_score"] > 0:
            assert entry["reason_codes"]
            assert entry["reasons"]
            assert all(reason["detail"] for reason in entry["reasons"])


def test_queue_entries_include_reviewer_control_surface_fields() -> None:
    record = base_record()

    entry = build_review_queue([record], {"sample_q03"})["entries"][0]

    assert entry["record_id"] == "sample_q03"
    assert entry["paper_id"] == "sample"
    assert entry["question_number"] == "3"
    assert entry["selected_text_source"] == "native"
    assert entry["selected_text_preview"].startswith("3 (a) Find x.")
    assert entry["question_image_path"] == "p1/sample/questions/q03.png"
    assert entry["mark_scheme_image_path"] == "p1/sample/mark_scheme/q03.png"
    assert entry["crop_context_warning_codes"] == ["suspiciously_short_selected_text"]
    assert entry["crop_context_warnings"][0]["code"] == "suspiciously_short_selected_text"
    assert entry["fixture_membership"] is True
    assert entry["review"] == {
        "reviewed": False,
        "status": None,
        "reviewed_at": None,
        "reviewer_note": "",
        "tags": [],
    }


def test_deterministic_ordering_uses_priority_then_record_id() -> None:
    records = [
        base_record(question_id="tie_q03_b"),
        base_record(question_id="tie_q03_a"),
        base_record(question_id="higher_q03", question_text="Find x.", ocr_text="3 Find x. [5]"),
    ]

    first = build_review_queue(records)
    second = build_review_queue(list(reversed(records)))

    assert [entry["record_id"] for entry in first["entries"]] == [entry["record_id"] for entry in second["entries"]]
    assert [entry["record_id"] for entry in first["entries"]][:3] == ["higher_q03", "tie_q03_a", "tie_q03_b"]


def test_known_fixture_recall_is_stable_inside_configured_top_n() -> None:
    fixture_ids = {f"fixture_q{i:02d}" for i in range(1, 37)}
    fixture_records = [
        base_record(question_id=record_id, question_number=str(index), question_text=f"{index} Find x. [1]")
        for index, record_id in enumerate(sorted(fixture_ids), start=1)
    ]
    ordinary_records = [
        base_record(question_id=f"ordinary_q{i:02d}", question_number=str(i), question_text=f"{i} Find x. [1]")
        for i in range(1, 40)
    ]

    report = build_review_queue(ordinary_records + fixture_records, fixture_ids, fixture_top_n=50)

    assert report["fixture_summary"]["known_fixture_count"] == 36
    assert report["fixture_summary"]["known_fixtures_found"] == 36
    assert report["fixture_summary"]["known_fixtures_in_top_n"] == 36
    assert report["fixture_summary"]["known_fixtures_in_top_50"] == 36
    assert report["fixture_summary"]["fixtures_outside_top_50"] == []


def test_fixture_outside_configured_threshold_is_explained() -> None:
    fixture = base_record(question_id="fixture_q03")
    stronger = [
        base_record(
            question_id=f"strong_q{i:02d}",
            question_text="Solve sin 0 = 0.",
            ocr_text="3 (a) Solve sin theta = 0. [2] (b) Hence find y. [3]",
            notes={
                **base_record()["notes"],
                "native_text_score": 20,
                "ocr_text_score": 91,
                "selected_text_score": 20,
                "text_fidelity_status": "degraded",
                "extraction_quality_flags": ["flattened_display_math", "math_corruption_suspected"],
                "ocr_rejected_reasons": ["ocr_lost_greek_symbol"],
            },
        )
        for i in range(3)
    ]

    report = build_review_queue(stronger + [fixture], {"fixture_q03"}, fixture_top_n=2)

    assert report["fixture_summary"]["known_fixtures_in_top_n"] == 0
    assert report["fixture_summary"]["fixtures_outside_top_50"][0]["record_id"] == "fixture_q03"
    assert "ranked below the top 2" in report["fixture_summary"]["fixtures_outside_top_50"][0]["explanation"]


def test_review_state_loading_and_default_visibility(tmp_path: Path) -> None:
    state_path = tmp_path / "review_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.text_fidelity.review_state",
                "schema_version": 1,
                "records": [
                    {
                        "record_id": "sample_q03",
                        "reviewed_at": "2026-05-20T00:00:00Z",
                        "reviewer_note": "Text acceptable for search only.",
                        "status": "accepted_text",
                        "tags": ["checked"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = load_review_state(state_path)
    report = build_review_queue([base_record()], review_state=state)

    assert report["review_state_summary"]["state_records_loaded"] == 1
    assert report["review_state_summary"]["reviewed_records_in_bank"] == 1
    assert report["entries"][0]["record_id"] == "sample_q03"
    assert report["entries"][0]["review"]["reviewed"] is True
    assert report["entries"][0]["review"]["status"] == "accepted_text"


def test_reviewed_records_are_filtered_only_when_requested() -> None:
    state = {
        "sample_q03": {
            "record_id": "sample_q03",
            "reviewed_at": "2026-05-20T00:00:00Z",
            "reviewer_note": "False positive.",
            "status": "false_positive",
            "tags": [],
        }
    }
    records = [base_record(), base_record(question_id="other_q03")]

    default_report = build_review_queue(records, review_state=state)
    filtered_report = build_review_queue(records, review_state=state, include_reviewed=False)

    assert [entry["record_id"] for entry in default_report["entries"]] == ["other_q03", "sample_q03"]
    assert [entry["record_id"] for entry in filtered_report["entries"]] == ["other_q03"]
    assert filtered_report["record_count"] == 2
    assert filtered_report["visible_record_count"] == 1


def test_invalid_review_status_is_rejected(tmp_path: Path) -> None:
    state_path = tmp_path / "review_state.json"
    state_path.write_text(json.dumps({"records": [{"record_id": "sample_q03", "status": "done"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected review status"):
        load_review_state(state_path)


def test_crop_context_low_confidence_feeds_queue_scoring() -> None:
    record = base_record(
        notes={
            **base_record()["notes"],
            "question_crop_confidence": "low",
        }
    )

    entry = score_record(record)

    assert "low_crop_confidence" in entry["crop_context_warning_codes"]
    assert "low_crop_confidence" in entry["reason_codes"]
    assert entry["priority_score"] >= 30


def test_score_record_does_not_mutate_selected_text_or_canonical_images() -> None:
    record = base_record(
        notes={
            **base_record()["notes"],
            "question_crop_confidence": "low",
            "text_fidelity_status": "degraded",
        }
    )
    original = deepcopy(record)

    entry = score_record(record)

    assert record == original
    assert entry["selected_text_preview"].startswith("3 (a) Find x.")
    assert entry["question_image_path"] == original["question_image_path"]
    assert entry["mark_scheme_image_path"] == original["mark_scheme_image_path"]
    assert entry["selected_text_source"] == original["notes"]["text_candidate_source"]


def test_build_queue_does_not_mutate_records() -> None:
    records = [base_record(), base_record(question_id="other_q03")]
    original = deepcopy(records)

    build_review_queue(records)

    assert records == original


def test_queue_output_does_not_mutate_canonical_records() -> None:
    records = [base_record()]
    before = deepcopy(records)

    build_review_queue(records, {"sample_q03"})

    assert records == before
