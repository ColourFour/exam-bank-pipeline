import json
from pathlib import Path

from exam_bank.auto_triage import (
    build_status_report,
    compare_auto_triage_iteration,
    create_auto_triage_plan,
    decide_auto_triage_iteration,
)


def test_auto_triage_status_counts_and_dominant_issue(tmp_path: Path) -> None:
    input_path = tmp_path / "output" / "json" / "question_bank.json"
    _write_bank(
        input_path,
        [
            _record(
                "q1",
                ocr_ran=True,
                validation_status="fail",
                validation_flags=["question_scope_contaminated"],
                text_source_profile="hybrid",
                review_flags=["needs_visual_review"],
                text_fidelity_status="degraded",
                text_fidelity_flags=["math_text_corrupted"],
            ),
            _record(
                "q2",
                ocr_ran=True,
                validation_status="fail",
                validation_flags=["question_scope_contaminated"],
                text_source_profile="hybrid",
            ),
            _record(
                "q3",
                ocr_ran=False,
                mapping_status="fail",
                mapping_failure_reason="missing_answer",
                text_source_profile="native_pdf",
                visual_curation_status="fail",
            ),
        ],
    )

    report = build_status_report(input_path)

    assert report["record_count"] == 3
    assert report["ocr_counts"] == {"true": 2, "false": 1}
    assert report["ocr_profile"]["ocr_mixed"] is True
    assert report["text_source_profile_counts"] == {"hybrid": 2, "native_pdf": 1}
    assert report["validation_status_counts"] == {"fail": 2, "pass": 1}
    assert report["markscheme_mapping_status_counts"] == {"pass": 2, "fail": 1}
    assert report["text_fidelity_status_counts"] == {"clean": 2, "degraded": 1}
    assert report["scope_quality_status_counts"] == {"clean": 3}
    assert report["topic_trust_status_counts"] == {"normal": 3}
    assert report["visual_curation_status_counts"] == {"ready": 2, "fail": 1}
    assert report["validation_flag_counts"] == {"question_scope_contaminated": 2}
    assert report["review_flag_counts"] == {"needs_visual_review": 1}
    assert report["text_fidelity_flag_counts"] == {"math_text_corrupted": 1}
    assert report["hard_failure_count"] == 3
    assert report["dominant_failure_cluster"] == {"issue": "question_scope_contaminated", "count": 2}
    assert report["top_issue_clusters"][1] == {"issue": "mapping_failed:missing_answer", "count": 1}


def test_auto_triage_plan_stops_when_threshold_met(tmp_path: Path) -> None:
    input_path = tmp_path / "output" / "json" / "question_bank.json"
    handoff_root = tmp_path / "agent_handoffs" / "auto_triage"
    _write_bank(input_path, [_record("q1", ocr_ran=True)])

    report = create_auto_triage_plan(
        input_path,
        handoff_root=handoff_root,
        target_max_hard_failures=0,
        sample_size=5,
    )

    assert report["stopped"] is True
    assert report["reason"] == "target_threshold_met"
    assert not list(handoff_root.glob("iteration_*"))


def test_auto_triage_plan_creates_handoff_iteration_and_prompts(tmp_path: Path) -> None:
    input_path = tmp_path / "output" / "json" / "question_bank.json"
    handoff_root = tmp_path / "agent_handoffs" / "auto_triage"
    existing_triage = tmp_path / "output" / "triage" / "iteration_004"
    existing_triage.mkdir(parents=True)
    _write_bank(
        input_path,
        [
            _record("q1", validation_status="fail", validation_flags=["paper_total_mismatch"]),
            _record("q2", validation_status="fail", validation_flags=["paper_total_mismatch"]),
            _record("q3", validation_status="fail", validation_flags=["question_scope_contaminated"]),
        ],
    )

    report = create_auto_triage_plan(
        input_path,
        handoff_root=handoff_root,
        target_max_hard_failures=0,
        sample_size=2,
    )

    iteration_dir = Path(report["iteration_dir"])
    assert report["stopped"] is False
    assert report["iteration"] == "iteration_005"
    assert (iteration_dir / "metrics_before.json").exists()
    assert (iteration_dir / "selected_target.json").exists()
    assert (iteration_dir / "commands.json").exists()
    assert (iteration_dir / "agent1_request.md").exists()
    selected = json.loads((iteration_dir / "selected_target.json").read_text(encoding="utf-8"))
    assert selected["issue"] == "paper_total_mismatch"
    assert selected["count"] == 2
    commands = json.loads((iteration_dir / "commands.json").read_text(encoding="utf-8"))
    assert "--iteration iteration_005" in commands["triage_sample"]
    assert "--enable-ocr" in commands["full_ocr_rerun"]
    prompt_files = {path.name for path in (handoff_root / "Prompt").iterdir()}
    assert prompt_files == {
        "supervisor_prompt.md",
        "agent1_planner_prompt.md",
        "agent2_builder_prompt.md",
        "agent3_tests_prompt.md",
        "agent4_integration_prompt.md",
        "agent5_review_prompt.md",
    }


def test_auto_triage_decision_accepts_improving_ocr_iteration() -> None:
    comparison = {
        "target": "question_scope_contaminated",
        "hard_failure_delta": -1,
        "target_issue_delta": -1,
        "worsened_records": [],
    }
    metrics_before = _metrics(ocr_enabled=True, hard_failures=1)
    metrics_after = _metrics(ocr_enabled=True, hard_failures=0)

    decision = decide_auto_triage_iteration(
        comparison=comparison,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        selected_target={"issue": "question_scope_contaminated"},
        test_status="pass",
    )

    assert decision["decision"] == "accepted"
    assert decision["accepted"] is True


def test_auto_triage_decision_allows_fail_to_review_improvements() -> None:
    comparison = {
        "target": "paper_total_mismatch",
        "hard_failure_delta": -20,
        "target_issue_delta": -21,
        "worsened_records": [],
    }
    metrics_before = _metrics(ocr_enabled=True, hard_failures=20)
    metrics_after = _metrics(ocr_enabled=True, hard_failures=0)
    metrics_before["validation_status_counts"] = {"fail": 20, "pass": 80}
    metrics_after["validation_status_counts"] = {"review": 15, "pass": 85}

    decision = decide_auto_triage_iteration(
        comparison=comparison,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        selected_target={"issue": "paper_total_mismatch"},
        test_status="pass",
    )

    assert decision["decision"] == "accepted"
    assert decision["status_regressions"] == []


def test_auto_triage_decision_rejects_net_bad_status_regression() -> None:
    comparison = {
        "target": "paper_total_mismatch",
        "hard_failure_delta": -1,
        "target_issue_delta": -1,
        "worsened_records": [],
    }
    metrics_before = _metrics(ocr_enabled=True, hard_failures=1)
    metrics_after = _metrics(ocr_enabled=True, hard_failures=0)
    metrics_before["validation_status_counts"] = {"fail": 1, "pass": 99}
    metrics_after["validation_status_counts"] = {"review": 12, "pass": 88}

    decision = decide_auto_triage_iteration(
        comparison=comparison,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        selected_target={"issue": "paper_total_mismatch"},
        test_status="pass",
    )

    assert decision["decision"] == "rejected"
    assert "major_status_regression" in decision["rejected_reasons"]
    assert decision["status_regressions"] == [
        {
            "field": "validation_status_counts",
            "status": "review",
            "delta": 12,
            "bad_status_delta": 11,
        }
    ]


def test_auto_triage_decision_rejects_no_ocr_current_output() -> None:
    decision = decide_auto_triage_iteration(
        comparison={"target": "paper_total_mismatch", "hard_failure_delta": -1, "target_issue_delta": -1},
        metrics_before=_metrics(ocr_enabled=True, hard_failures=2),
        metrics_after=_metrics(ocr_enabled=False, hard_failures=1),
        selected_target={"issue": "paper_total_mismatch"},
        test_status="pass",
    )

    assert decision["decision"] == "rejected"
    assert "current_output_not_ocr_enabled" in decision["rejected_reasons"]


def test_auto_triage_decision_rejects_failed_tests() -> None:
    decision = decide_auto_triage_iteration(
        comparison={"target": "paper_total_mismatch", "hard_failure_delta": -1, "target_issue_delta": -1},
        metrics_before=_metrics(ocr_enabled=True, hard_failures=2),
        metrics_after=_metrics(ocr_enabled=True, hard_failures=1),
        selected_target={"issue": "paper_total_mismatch"},
        test_status="fail",
    )

    assert decision["decision"] == "rejected"
    assert "tests_failed" in decision["rejected_reasons"]


def test_auto_triage_compare_writes_decision_and_preserves_baseline(tmp_path: Path) -> None:
    handoff_iteration = tmp_path / "agent_handoffs" / "auto_triage" / "iteration_001"
    handoff_iteration.mkdir(parents=True)
    (handoff_iteration / "selected_target.json").write_text(
        json.dumps({"issue": "question_scope_contaminated"}),
        encoding="utf-8",
    )
    baseline_triage = tmp_path / "output" / "triage" / "iteration_001"
    baseline_triage.mkdir(parents=True)
    baseline_path = baseline_triage / "baseline_question_bank.json"
    current_path = tmp_path / "output_ocr_candidate" / "json" / "question_bank.json"
    comparison_path = baseline_triage / "comparison.auto-iteration-001.json"
    _write_bank(
        baseline_path,
        [
            _record("q1", ocr_ran=True, validation_status="fail", validation_flags=["question_scope_contaminated"]),
            _record("q2", ocr_ran=True),
        ],
    )
    _write_bank(current_path, [_record("q1", ocr_ran=True), _record("q2", ocr_ran=True)])
    (baseline_triage / "summary.json").write_text(
        json.dumps(
            {
                "baseline_path": str(baseline_path),
                "issue_set": "hard-failures",
                "target": "question_scope_contaminated",
            }
        ),
        encoding="utf-8",
    )
    baseline_before = baseline_path.read_text(encoding="utf-8")

    report = compare_auto_triage_iteration(
        iteration=handoff_iteration,
        baseline_triage=baseline_triage,
        current_path=current_path,
        output_path=comparison_path,
        test_status="pass",
    )

    assert report["decision"]["decision"] == "accepted"
    assert (handoff_iteration / "metrics_after.json").exists()
    assert (handoff_iteration / "decision.json").exists()
    assert comparison_path.exists()
    assert baseline_path.read_text(encoding="utf-8") == baseline_before


def _write_bank(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"record_count": len(records), "questions": records}), encoding="utf-8")


def _record(
    question_id: str,
    *,
    ocr_ran: bool = False,
    validation_status: str = "pass",
    mapping_status: str = "pass",
    visual_curation_status: str = "ready",
    text_only_status: str = "ready",
    scope_quality_status: str = "clean",
    text_fidelity_status: str = "clean",
    topic_trust_status: str = "normal",
    text_source_profile: str = "native_pdf",
    validation_flags: list[str] | None = None,
    review_flags: list[str] | None = None,
    text_fidelity_flags: list[str] | None = None,
    mapping_failure_reason: str = "",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "12spring24",
        "paper_family": "p1",
        "question_number": question_id.removeprefix("q"),
        "question_text": "Find x.",
        "ocr_ran": ocr_ran,
        "visual_curation_status": visual_curation_status,
        "text_only_status": text_only_status,
        "notes": {
            "validation_status": validation_status,
            "mapping_status": mapping_status,
            "visual_curation_status": visual_curation_status,
            "text_only_status": text_only_status,
            "scope_quality_status": scope_quality_status,
            "text_fidelity_status": text_fidelity_status,
            "topic_trust_status": topic_trust_status,
            "text_source_profile": text_source_profile,
            "validation_flags": validation_flags or [],
            "review_flags": review_flags or [],
            "text_fidelity_flags": text_fidelity_flags or [],
            "mapping_failure_reason": mapping_failure_reason,
        },
    }


def _metrics(*, ocr_enabled: bool, hard_failures: int) -> dict[str, object]:
    return {
        "record_count": max(1, hard_failures),
        "hard_failure_count": hard_failures,
        "ocr_profile": {
            "ocr_enabled": ocr_enabled,
            "ocr_disabled": not ocr_enabled,
            "ocr_mixed": False,
            "ocr_missing": 0,
        },
        "validation_status_counts": {"fail": hard_failures, "pass": max(0, 10 - hard_failures)},
        "markscheme_mapping_status_counts": {"pass": 10},
        "scope_quality_status_counts": {"clean": 10},
        "text_fidelity_status_counts": {"clean": 10},
        "topic_trust_status_counts": {"normal": 10},
        "visual_curation_status_counts": {"ready": 10},
        "validation_flag_counts": {"some_flag": hard_failures} if hard_failures else {},
    }
