from exam_bank.text_review_queue import build_review_queue, score_record


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
    assert "missing_marks" in codes
    assert "lost_subpart_labels" in codes
    assert "suspiciously_short_text" in codes
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

    assert "next_question_contamination" in reason_codes(text_record)
    assert "next_question_contamination" in reason_codes(metadata_record)


def test_fixture_membership_boosts_known_bad_records_near_top() -> None:
    fixture = base_record(question_id="fixture_q03", question_text="Find x.", ocr_text="3 Find x. [5]")
    non_fixture = base_record(question_id="ordinary_q04", question_number="4")

    report = build_review_queue([non_fixture, fixture], {"fixture_q03"})

    assert report["entries"][0]["record_id"] == "fixture_q03"
    assert report["entries"][0]["reason_codes"][0] == "known_fixture_membership"
    assert report["fixture_summary"]["known_fixtures_in_top_50"] == 1


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
