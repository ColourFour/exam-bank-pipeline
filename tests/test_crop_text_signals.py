from copy import deepcopy

from exam_bank.crop_text_signals import (
    audit_fixture_record,
    build_crop_text_signal_audit,
    compute_crop_context_warning_codes,
)


def base_record(**overrides):
    record = {
        "record_id": "sample_q03",
        "paper_id": "sample",
        "paper_family": "p1",
        "question_number": "3",
        "currently_selected_text": "3 (a) Find x. [2] (b) Hence find y. [3]",
        "expected_normalized_text_or_structural_expectations": {
            "type": "structural_expectations",
            "expectations": [
                "contains question number 3",
                "contains mark bracket [3]",
                "preserves subpart labels",
            ],
        },
        "question_image_path": "p1/sample/questions/q03.png",
        "mark_scheme_image_path": "p1/sample/mark_scheme/q03.png",
    }
    record.update(overrides)
    return record


def warning_codes(record, qbank=None):
    return {warning["code"] for warning in audit_fixture_record(record, qbank)["warnings"]}


def test_detects_missing_question_number() -> None:
    record = base_record(currently_selected_text="(a) Find x. [2] (b) Hence find y. [3]")

    assert "missing_expected_question_number" in warning_codes(record)


def test_detects_missing_marks() -> None:
    record = base_record(currently_selected_text="3 (a) Find x. (b) Hence find y.")

    codes = warning_codes(record)

    assert "missing_mark_bracket" in codes


def test_detects_missing_subpart_labels() -> None:
    record = base_record(currently_selected_text="3 Find x and hence y. [3]")

    codes = warning_codes(record, {"subparts": ["a", "b"]})

    assert "missing_subpart_labels" in codes


def test_detects_next_question_contamination() -> None:
    record = base_record(currently_selected_text="3 Find x. [4] 4 Solve the equation sin x = 0. [5]")

    codes = warning_codes(record)

    assert "possible_next_question_contamination" in codes


def test_detects_low_crop_confidence() -> None:
    qbank = {"notes": {"question_crop_confidence": "low"}}

    assert "low_crop_confidence" in warning_codes(base_record(), qbank)


def test_detects_selector_warnings() -> None:
    qbank = {
        "notes": {
            "review_flags": ["weak_question_text"],
            "validation_flags": ["text_needs_review"],
        }
    }

    codes = warning_codes(base_record(), qbank)

    assert "selector_warning_present" in codes
    assert "selector_structural_warning_present" in codes


def test_detects_clean_crop_with_degraded_text() -> None:
    qbank = {
        "notes": {
            "question_crop_confidence": "high",
            "text_fidelity_status": "degraded",
            "selected_text_score": 20,
        }
    }

    assert "clean_visual_crop_but_degraded_text" in warning_codes(base_record(), qbank)


def test_detects_selected_ocr_with_structural_warnings() -> None:
    qbank = {
        "notes": {
            "ocr_selected": True,
            "review_flags": ["weak_question_text"],
        }
    }

    assert "selected_ocr_with_structural_warnings" in warning_codes(base_record(), qbank)


def test_warning_codes_are_deterministic_and_do_not_mutate_inputs() -> None:
    record = base_record(currently_selected_text="Find x.")
    qbank = {
        "question_text": "3 (a) Find x. [2] (b) Hence find y. [3]",
        "subparts": ["a", "b"],
        "notes": {"question_crop_confidence": "low", "review_flags": ["weak_question_text"]},
    }
    original_record = deepcopy(record)
    original_qbank = deepcopy(qbank)

    first = compute_crop_context_warning_codes(record, qbank)
    second = compute_crop_context_warning_codes(record, qbank)

    assert first == second
    assert first == sorted(first)
    assert record == original_record
    assert qbank == original_qbank


def test_build_audit_counts_useful_warnings() -> None:
    manifest = {
        "schema_name": "text_fidelity_bad_text_fixture_manifest",
        "schema_version": 1,
        "records": [
            base_record(currently_selected_text="Find x."),
            base_record(record_id="sample_q04", question_number="4", currently_selected_text="4 Find x. [2]"),
        ],
    }

    report = build_crop_text_signal_audit(manifest)

    assert report["schema_name"] == "exam_bank.crop_text_signal_audit"
    assert report["record_count"] == 2
    assert report["records_with_useful_warnings"] >= 1
    assert report["records_caught_by_practical_now_gates"] >= 1
