import json
from pathlib import Path

from exam_bank.triage import (
    compare_iteration,
    create_triage_iteration,
    is_hard_failure,
    issue_counts,
    primary_issue,
    select_sample_records,
)


def test_triage_classifies_primary_issues_and_hard_failures() -> None:
    records = [
        _record("q1", validation_status="fail", validation_flags=["question_scope_contaminated"]),
        _record("q2", mapping_status="fail", mapping_failure_reason="missing_answer"),
        _record("q3", visual_curation_status="fail"),
        _record("q4", validation_status="review", visual_curation_status="review"),
    ]

    assert [is_hard_failure(record) for record in records] == [True, True, True, False]
    assert primary_issue(records[0]) == "question_scope_contaminated"
    assert primary_issue(records[1]) == "mapping_failed:missing_answer"
    assert primary_issue(records[2]) == "visual_curation_failed"
    assert issue_counts(records) == {
        "question_scope_contaminated": 1,
        "mapping_failed:missing_answer": 1,
        "visual_curation_failed": 1,
    }


def test_triage_sampling_is_deterministic_and_targets_largest_issue() -> None:
    records = [
        _record(f"q{index}", validation_status="fail", validation_flags=["question_scope_contaminated"])
        for index in range(1, 7)
    ]
    records.extend(
        [
            _record("q7", validation_status="fail", validation_flags=["paper_total_mismatch"]),
            _record("q8", validation_status="fail", validation_flags=["paper_total_mismatch"]),
        ]
    )

    target_a, sample_a = select_sample_records(records, sample_size=3, seed=42)
    target_b, sample_b = select_sample_records(records, sample_size=3, seed=42)

    assert target_a == "question_scope_contaminated"
    assert target_b == target_a
    assert [record["question_id"] for record in sample_a] == [record["question_id"] for record in sample_b]
    assert len(sample_a) == 3
    assert all(primary_issue(record) == "question_scope_contaminated" for record in sample_a)


def test_create_triage_iteration_writes_baseline_sample_and_escaped_gallery(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    input_path = output_root / "json" / "question_bank.json"
    input_path.parent.mkdir(parents=True)
    records = [
        _record(
            "q1",
            validation_status="fail",
            validation_flags=["question_scope_contaminated"],
            question_text="Bad <img src=x onerror=alert(1)>",
            question_image_path="p1/sample/questions/q01.png",
            mark_scheme_image_path="p1/sample/mark_scheme/q01.png",
        )
    ]
    input_path.write_text(json.dumps({"questions": records}), encoding="utf-8")

    summary = create_triage_iteration(input_path, iteration="iteration_001", sample_size=1)
    iteration_dir = Path(summary["iteration_dir"])

    assert (iteration_dir / "baseline_question_bank.json").exists()
    assert (iteration_dir / "sample.json").exists()
    assert (iteration_dir / "review.jsonl").exists()
    html = (iteration_dir / "index.html").read_text(encoding="utf-8")
    assert "../../p1/sample/questions/q01.png" in html
    assert "Bad &lt;img src=x onerror=alert(1)&gt;" in html
    assert "<pre>Bad <img" not in html


def test_compare_iteration_reports_target_movement_and_status_changes(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "output" / "triage" / "iteration_001"
    iteration_dir.mkdir(parents=True)
    baseline_path = iteration_dir / "baseline_question_bank.json"
    current_path = tmp_path / "output" / "json" / "question_bank.json"
    current_path.parent.mkdir(parents=True)
    baseline = [
        _record("q1", validation_status="fail", validation_flags=["question_scope_contaminated"]),
        _record("q2"),
    ]
    current = [
        _record("q1"),
        _record("q2", mapping_status="fail", mapping_failure_reason="missing_answer"),
    ]
    baseline_path.write_text(json.dumps({"questions": baseline}), encoding="utf-8")
    current_path.write_text(json.dumps({"questions": current}), encoding="utf-8")
    (iteration_dir / "summary.json").write_text(
        json.dumps(
            {
                "baseline_path": str(baseline_path),
                "issue_set": "hard-failures",
                "target": "question_scope_contaminated",
            }
        ),
        encoding="utf-8",
    )

    report = compare_iteration(iteration_dir, current_path=current_path)

    assert report["baseline_target_issue_count"] == 1
    assert report["current_target_issue_count"] == 0
    assert report["target_issue_delta"] == -1
    assert report["hard_failure_delta"] == 0
    assert report["improved_records"] == [{"question_id": "q1", "fields": ["validation_status"]}]
    assert report["worsened_records"] == [{"question_id": "q2", "fields": ["mapping_status"]}]


def _record(
    question_id: str,
    *,
    validation_status: str = "pass",
    mapping_status: str = "pass",
    visual_curation_status: str = "ready",
    text_only_status: str = "ready",
    validation_flags: list[str] | None = None,
    review_flags: list[str] | None = None,
    visual_reason_flags: list[str] | None = None,
    extraction_quality_flags: list[str] | None = None,
    mapping_failure_reason: str = "",
    question_text: str = "Find x.",
    question_image_path: str = "",
    mark_scheme_image_path: str = "",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "12spring24",
        "paper_family": "p1",
        "question_number": question_id.removeprefix("q"),
        "question_text": question_text,
        "question_image_path": question_image_path,
        "mark_scheme_image_path": mark_scheme_image_path,
        "visual_curation_status": visual_curation_status,
        "text_only_status": text_only_status,
        "notes": {
            "validation_status": validation_status,
            "mapping_status": mapping_status,
            "visual_curation_status": visual_curation_status,
            "text_only_status": text_only_status,
            "validation_flags": validation_flags or [],
            "review_flags": review_flags or [],
            "visual_reason_flags": visual_reason_flags or [],
            "extraction_quality_flags": extraction_quality_flags or [],
            "mapping_failure_reason": mapping_failure_reason,
        },
    }
