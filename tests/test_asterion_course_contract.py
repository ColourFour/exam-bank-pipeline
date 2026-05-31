from __future__ import annotations

from exam_bank.asterion_course_contract import (
    COURSE_EMPTY_STATE_MESSAGE,
    component_counts,
    course_counts,
    course_id_for_record,
    filter_records_by_course,
    filter_records_by_paper_or_component,
    records_from_payload,
    student_runtime_course_records,
    student_runtime_ready_for_record,
    student_runtime_safe_for_record,
    to_exam_bank_course_record,
    validate_exam_bank_course_record,
)


def test_course_ids_map_product_components_without_cross_loading_p3() -> None:
    records = [
        _record("p3-q1", "p3", "31spring24", safe=True),
        _record("p1-q1", "p1", "12spring24", safe=True),
        _record("m1-q1", "p4", "42spring24", safe=True),
        _record("s1-q1", "p5", "52spring24", safe=True),
    ]

    assert course_id_for_record(records[0]) == "p3"
    assert course_id_for_record(records[2]) == "m1"
    assert [record["question_id"] for record in filter_records_by_course(records, "p3")] == ["p3-q1"]
    assert [record["question_id"] for record in filter_records_by_course(records, "m1")] == ["m1-q1"]
    assert [record["question_id"] for record in filter_records_by_course(records, "s1")] == ["s1-q1"]


def test_p3_legacy_runtime_safe_is_preserved_but_other_courses_are_scaffolded() -> None:
    p3 = _record("p3-q1", "p3", "31spring24", safe=True)
    p1 = _record("p1-q1", "p1", "12spring24", safe=True)
    m1 = _record("m1-q1", "p4", "42spring24", safe=True)
    s1 = _record("s1-q1", "p5", "52spring24", safe=True)

    assert student_runtime_safe_for_record(p3) is True
    assert student_runtime_safe_for_record(p1) is False
    assert student_runtime_safe_for_record(m1) is False
    assert student_runtime_safe_for_record(s1) is False
    assert [record["question_id"] for record in filter_records_by_course([p3, p1, m1, s1], "p3", student_runtime_only=True)] == ["p3-q1"]
    assert filter_records_by_course([p3, p1, m1, s1], "p1", student_runtime_only=True) == []
    assert filter_records_by_course([p3, p1, m1, s1], "m1", student_runtime_only=True) == []
    assert filter_records_by_course([p3, p1, m1, s1], "s1", student_runtime_only=True) == []


def test_explicit_reviewed_non_p3_record_can_enter_future_runtime() -> None:
    record = _record("m1-q1", "p4", "42spring24")
    record["student_runtime_safe"] = True
    record["review_status"] = "reviewed"

    course_record = to_exam_bank_course_record(record)

    assert student_runtime_safe_for_record(record) is True
    assert course_record is not None
    assert course_record["course_id"] == "m1"
    assert course_record["component_name"] == "Mechanics 1"
    assert course_record["student_runtime_safe"] is True
    assert course_record["review_status"] == "reviewed"


def test_student_runtime_filter_requires_reviewed_status() -> None:
    reviewed = _record("m1-q1", "p4", "42spring24")
    reviewed["student_runtime_safe"] = True
    reviewed["review_status"] = "reviewed"
    needs_review = _record("m1-q2", "p4", "42spring24")
    needs_review["student_runtime_safe"] = True
    needs_review["review_status"] = "needs_review"

    assert student_runtime_ready_for_record(reviewed) is True
    assert student_runtime_ready_for_record(needs_review) is False
    assert [record["question_id"] for record in filter_records_by_course([reviewed, needs_review], "m1", student_runtime_only=True)] == [
        "m1-q1"
    ]
    assert "student_runtime_safe_review_status_not_reviewed" in validate_exam_bank_course_record(needs_review)


def test_empty_scaffold_courses_return_empty_arrays_and_known_empty_state() -> None:
    payload = {"schema_name": "asterion.question_bank", "questions": [_record("p3-q1", "p3", "31spring24", safe=True)]}

    assert student_runtime_course_records(payload, "p1") == []
    assert student_runtime_course_records(payload, "m1") == []
    assert student_runtime_course_records(payload, "s1") == []
    assert COURSE_EMPTY_STATE_MESSAGE == "No reviewed exam-bank records available yet."


def test_component_and_paper_filters_are_safe_when_missing_or_invalid() -> None:
    records = [
        _record("p3-q1", "p3", "31spring24", safe=True),
        _record("p3-q2", "p3", "32spring24", safe=True),
        _record("m1-q1", "p4", "42spring24", safe=True),
    ]

    assert [record["question_id"] for record in filter_records_by_paper_or_component(records, paper="31spring24")] == ["p3-q1"]
    assert [record["question_id"] for record in filter_records_by_paper_or_component(records, component_name="Mechanics 1")] == ["m1-q1"]
    assert filter_records_by_course(records, "p2") == []
    assert records_from_payload({"schema_name": "not-supported"}) == []


def test_content_lab_candidates_are_not_student_runtime_records() -> None:
    candidate = {
        "candidate_id": "content_lab_31spring24_q01_whole",
        "question_id": "31spring24_q01",
        "paper_family": "p3",
        "student_runtime_safe": True,
        "role_statuses": {"generated_warmup_pattern_source": "allow"},
        "generation_gate": {"status": "allow"},
    }
    payload = {"schema_name": "asterion.content_lab_candidates", "candidates": [candidate]}

    assert records_from_payload(payload) == []
    assert student_runtime_safe_for_record(candidate) is False
    assert "content_lab_candidate_cannot_be_student_runtime" in validate_exam_bank_course_record(candidate)
    assert to_exam_bank_course_record(candidate) is None


def test_invalid_course_ids_and_missing_images_fail_closed() -> None:
    invalid = _record("x-q1", "p3", "31spring24", safe=True)
    invalid["course_id"] = "p2"
    missing_images = _record("p3-q2", "p3", "31spring24", safe=True)
    missing_images.pop("canonical_question_artifact")

    assert course_id_for_record(invalid) is None
    assert "invalid_course_id" in validate_exam_bank_course_record(invalid)
    assert filter_records_by_course([invalid], "p3") == []
    assert "student_runtime_safe_missing_question_image_path" in validate_exam_bank_course_record(missing_images)


def test_course_counts_include_all_courses_even_when_empty() -> None:
    counts = {row["course_id"]: row for row in course_counts([_record("p3-q1", "p3", "31spring24", safe=True)])}

    assert set(counts) == {"p1", "p3", "m1", "s1"}
    assert counts["p3"]["student_runtime_safe_record_count"] == 1
    assert counts["p1"]["record_count"] == 0
    assert counts["p1"]["empty_state_message"] == COURSE_EMPTY_STATE_MESSAGE


def test_component_counts_include_course_components_and_papers() -> None:
    counts = {
        row["course_id"]: row
        for row in component_counts(
            [
                _record("p3-q1", "p3", "31spring24", safe=True),
                _record("p3-q2", "p3", "31spring24", safe=True),
                _record("m1-q1", "p4", "42spring24", safe=True),
            ]
        )
    }

    assert counts["p3"]["component_name"] == "Pure Mathematics 3"
    assert counts["p3"]["record_count"] == 2
    assert counts["p3"]["papers"] == [{"paper": "31spring24", "record_count": 2}]
    assert counts["m1"]["component_name"] == "Mechanics 1"
    assert counts["m1"]["papers"] == [{"paper": "42spring24", "record_count": 1}]
    assert counts["s1"]["record_count"] == 0


def _record(question_id: str, paper_family: str, paper: str, *, safe: bool = False) -> dict:
    return {
        "question_id": question_id,
        "paper_family": paper_family,
        "paper": paper,
        "canonical_question_artifact": f"{paper_family}/{paper}/questions/q01.png",
        "canonical_mark_scheme_artifact": f"{paper_family}/{paper}/mark_scheme/q01.png",
        "usage_roles": {"canonical_practice": "allow" if safe else "block"},
        "quality_gate": {
            "canonical_assets_ok": True,
            "reason_codes": [],
        },
    }
