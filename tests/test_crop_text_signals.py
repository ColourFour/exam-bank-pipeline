from exam_bank.crop_text_signals import audit_fixture_record, build_crop_text_signal_audit


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

    assert "missing_expected_subpart_labels" in codes


def test_detects_next_question_contamination() -> None:
    record = base_record(currently_selected_text="3 Find x. [4] 4 Solve the equation sin x = 0. [5]")

    codes = warning_codes(record)

    assert "next_question_contamination" in codes


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
