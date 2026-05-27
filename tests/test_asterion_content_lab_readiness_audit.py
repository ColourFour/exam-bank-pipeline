from __future__ import annotations

import csv
import json
from pathlib import Path

from exam_bank.asterion_content_lab_audit import run_audit


def test_content_lab_readiness_audit_reports_pass_rate_and_blockers(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "audit"

    summary = run_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        artifact_root=paths["artifact_root"],
        out_dir=out_dir,
        sample_size=10,
        sample_seed="unit-seed",
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
    )

    assert summary["total_candidates"] == 5
    assert summary["p3_candidates"] == 5
    assert summary["sample_size"] == 5
    assert summary["sample_passed"] == 1
    assert summary["sample_failed"] == 4
    assert summary["sample_pass_rate"] == 0.2
    assert summary["target_met"] is False
    assert summary["student_runtime_changed"] is False
    assert summary["trust_gates_weakened"] is False
    assert summary["legacy_schema_mismatch_count"] == 1
    assert summary["mapping_validation_contradiction_count"] == 1
    assert set(summary["regions_covered"]) >= {"Algebra Vault", "Vectors Gate"}

    assert (out_dir / "audit_summary.json").exists()
    assert (out_dir / "audit_summary.md").exists()
    assert (out_dir / "p3_sample_frame.csv").exists()
    assert (out_dir / "p3_sample_results.csv").exists()
    assert (out_dir / "blocker_breakdown.csv").exists()
    assert (out_dir / "repair_candidates.csv").exists()

    sample_rows = _read_csv(out_dir / "p3_sample_results.csv")
    by_id = {row["candidate_id"]: row for row in sample_rows}
    assert by_id["content_lab_31spring24_q01_whole"]["passed"] == "True"
    assert by_id["content_lab_31spring24_q01_whole"]["legacy_schema_mismatch"] == "True"
    assert by_id["content_lab_31spring24_q02_whole"]["passed"] == "False"
    assert "blocked_mark_events" in by_id["content_lab_31spring24_q02_whole"]["blocker_classes"]
    assert by_id["content_lab_31spring24_q03_whole"]["passed"] == "False"
    assert "missing_canonical_mark_scheme_image_file" in by_id["content_lab_31spring24_q03_whole"]["blocker_reasons"]
    assert by_id["content_lab_31spring24_q04_whole"]["mapping_validation_contradiction"] == "True"
    assert by_id["content_lab_31spring24_q05_whole"]["passed"] == "False"
    assert "blocked_quarantine_ambiguous" in by_id["content_lab_31spring24_q05_whole"]["blocker_classes"]


def test_content_lab_readiness_sample_is_deterministic_and_stratified(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = run_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        artifact_root=paths["artifact_root"],
        out_dir=first_dir,
        sample_size=3,
        sample_seed="stable",
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
    )
    second = run_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        artifact_root=paths["artifact_root"],
        out_dir=second_dir,
        sample_size=3,
        sample_seed="stable",
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
    )

    assert first["sample_size"] == second["sample_size"] == 3
    assert first["sample_method"] == second["sample_method"]
    assert (first_dir / "p3_sample_frame.csv").read_text(encoding="utf-8") == (
        second_dir / "p3_sample_frame.csv"
    ).read_text(encoding="utf-8")
    assert len(first["regions_covered"]) >= 2


def test_reviewed_evidence_coverage_reports_join_and_human_review_queue(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "coverage"

    summary = run_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        artifact_root=paths["artifact_root"],
        out_dir=out_dir,
        sample_size=10,
        sample_seed="coverage",
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
    )

    coverage = summary["reviewed_evidence_coverage"]
    assert coverage["existing_reviewed_exact_skill_evidence_count"] == 1
    assert coverage["existing_reviewed_source_skill_evidence_count"] == 1
    assert coverage["existing_reviewed_mark_event_subpart_evidence_count"] == 1
    assert coverage["fails_because_reviewed_evidence_absent_count"] == 4
    assert coverage["fails_with_ambiguous_or_quarantined_evidence_count"] == 1
    assert coverage["requires_human_review_no_matter_code_count"] == 4

    join_rows = _read_csv(out_dir / "candidate_evidence_join.csv")
    by_id = {row["candidate_id"]: row for row in join_rows}
    assert by_id["content_lab_31spring24_q01_whole"]["clean_source_skill_evidence_exists"] == "True"
    assert by_id["content_lab_31spring24_q01_whole"]["all_mark_events_reviewed"] == "True"
    assert by_id["content_lab_31spring24_q02_whole"]["all_mark_events_reviewed"] == "False"
    assert by_id["content_lab_31spring24_q02_whole"]["source_mark_event_count"] == "1"
    assert by_id["content_lab_31spring24_q05_whole"]["non_generation_satisfying_exact_skill_record_count"] == "1"

    queue_rows = _read_csv(out_dir / "human_review_queue.csv")
    queue_by_id = {row["candidate_id"]: row for row in queue_rows}
    assert "content_lab_31spring24_q01_whole" not in queue_by_id
    assert "review_mark_events_for_candidate_part" in queue_by_id["content_lab_31spring24_q02_whole"]["human_review_reasons"]
    assert "resolve_non_generation_satisfying_exact_skill_record" in queue_by_id[
        "content_lab_31spring24_q05_whole"
    ]["human_review_reasons"]


def test_full_pool_report_writes_baseline_region_and_blocker_outputs(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "full_pool"

    summary = run_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        artifact_root=paths["artifact_root"],
        out_dir=out_dir,
        sample_size=10,
        sample_seed="full-pool",
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
        full_pool_report=True,
    )

    baseline = json.loads((out_dir / "full_pool_baseline.json").read_text(encoding="utf-8"))
    assert baseline["total_p3_candidates"] == summary["p3_candidates"] == 5
    assert baseline["pass_count"] == 1
    assert baseline["deterministic_sample_guard"]["pass_count"] == summary["sample_passed"]
    assert baseline["additional_approvals_needed"]["for_70_percent"] == 3
    assert (out_dir / "full_pool_candidate_results.csv").exists()
    assert (out_dir / "full_pool_region_summary.csv").exists()
    assert (out_dir / "full_pool_blocker_summary.csv").exists()
    region_rows = _read_csv(out_dir / "full_pool_region_summary.csv")
    assert {row["region"] for row in region_rows} >= {"Algebra Vault", "Vectors Gate"}
    blocker_rows = _read_csv(out_dir / "full_pool_blocker_summary.csv")
    assert any(row["blocker_class"] == "blocked_mark_events" for row in blocker_rows)


def test_new_gate_fields_do_not_pass_without_reviewed_source_files(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    empty_source = tmp_path / "empty_source.json"
    empty_marks = tmp_path / "empty_marks.json"
    _write_json(empty_source, {"records": []})
    _write_json(empty_marks, {"decisions": []})

    summary = run_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "unbacked",
        sample_size=10,
        sample_seed="unbacked",
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        reviewed_source_skills_path=empty_source,
        reviewed_mark_events_path=empty_marks,
    )

    assert summary["sample_passed"] == 0
    rows = _read_csv(tmp_path / "unbacked" / "p3_sample_results.csv")
    row = next(item for item in rows if item["candidate_id"] == "content_lab_31spring24_q01_whole")
    assert "reviewed_source_skill_gate_unbacked_by_reviewed_source_file" in row["blocker_reasons"]
    assert "mark_event_gate_unbacked_by_reviewed_source_file" in row["blocker_reasons"]


def test_loop004_blocker_classification_and_same_sample_outputs() -> None:
    root = Path.cwd()
    loop003_rows = _read_csv(root / "output/audits/asterion_content_lab_loop/iteration_003b/sample_results.csv")
    loop004_rows = _read_csv(root / "output/audits/asterion_content_lab_loop/iteration_004/sample_results.csv")
    classification_rows = _read_csv(root / "output/review/content_lab_p3_auto_loop_004/remaining_sample_blockers.csv")
    corrections = [
        json.loads(line)
        for line in (root / "output/review/content_lab_p3_auto_loop_004/candidate_mapping_corrections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    merged = json.loads(
        (root / "data/review/content_lab_p3_auto_reviewed_decisions_merged_0002.v1.json").read_text(encoding="utf-8")
    )
    summary = json.loads((root / "output/audits/asterion_content_lab_loop/iteration_004/audit_summary.json").read_text())

    assert len(loop003_rows) == len(loop004_rows) == 100
    assert [row["candidate_id"] for row in loop003_rows] == [row["candidate_id"] for row in loop004_rows]
    assert sum(row["passed"] == "True" for row in loop003_rows) == 70
    assert sum(row["passed"] == "True" for row in loop004_rows) == 96
    assert len(classification_rows) == 30
    class_counts = {key: 0 for key in {
        "unreviewed_eligible_for_agentic_review",
        "rejected_but_mapping_correctable",
        "requires_new_candidate_generation",
    }}
    for row in classification_rows:
        class_counts[row["blocker_classification"]] = class_counts.get(row["blocker_classification"], 0) + 1
    assert class_counts["unreviewed_eligible_for_agentic_review"] == 7
    assert class_counts["rejected_but_mapping_correctable"] == 19
    assert class_counts["requires_new_candidate_generation"] == 4
    assert len(corrections) == 19
    assert merged["record_count"] == 93
    assert summary["sample_size"] == 100
    assert summary["sample_passed"] == 96
    assert summary["target_pass_rate"] == 0.9
    assert summary["target_met"] is True
    assert summary["trust_gates_weakened"] is False
    still_blocked = {row["candidate_id"] for row in loop004_rows if row["passed"] == "False"}
    assert still_blocked == {
        "content_lab_32summer22_q06_b",
        "content_lab_32spring25_q05_whole",
        "content_lab_32summer23_q10_b",
        "content_lab_33autumn21_q02_a",
    }


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "output"
    for question_number in range(1, 6):
        (artifact_root / "p3" / "31spring24" / "questions").mkdir(parents=True, exist_ok=True)
        (artifact_root / "p3" / "31spring24" / "mark_scheme").mkdir(parents=True, exist_ok=True)
        (artifact_root / "p3" / "31spring24" / "questions" / f"q{question_number:02d}.png").write_bytes(b"question")
        if question_number != 3:
            (artifact_root / "p3" / "31spring24" / "mark_scheme" / f"q{question_number:02d}.png").write_bytes(b"mark")

    questions = [_question(question_number) for question_number in range(1, 6)]
    questions[3]["notes"]["mapping_status"] = "fail"
    questions[3]["notes"]["validation_status"] = "pass"

    candidates = [
        _candidate(1, generation_status="allow", reviewed_subpart=False),
        _candidate(2, generation_status="blocked_until_reviewed"),
        _candidate(3, generation_status="allow"),
        _candidate(4, generation_status="allow"),
        _candidate(5, generation_status="blocked_until_reviewed", generation_reasons=["mark_events_quarantined"]),
    ]
    candidates[1]["mark_event_review_gate"] = {
        "status": "blocked_until_reviewed",
        "blocked": True,
        "block_reasons": ["reviewed_mark_event_decision_missing"],
    }
    candidates[2]["source_artifacts"]["mark_scheme_crop_path"] = "p3/31spring24/mark_scheme/q03.png"

    skill_map = {
        "skills": [
            {"skill_id": "9709_p3_3_1_algebra", "section": "3.1 Algebra"},
            {"skill_id": "9709_p3_3_7_vectors", "section": "3.7 Vectors"},
        ]
    }
    mappings = {
        "mappings": [
            {
                "question_id": "31spring24_q02",
                "subpart_id": "31spring24_q02_whole",
                "primary_skill_ids": ["9709_p3_3_7_vectors"],
            }
        ]
    }
    topic_routing = {"records": {"31spring24_q02": {"primary_topic_id": "vectors", "paper_family": "p3"}}}

    paths = {
        "artifact_root": artifact_root,
        "candidates": tmp_path / "candidates.json",
        "asterion_bank": tmp_path / "asterion_bank.json",
        "question_bank": tmp_path / "question_bank.json",
        "topic_routing": tmp_path / "topic_routing.json",
        "skill_map": tmp_path / "skill_map.json",
        "question_skill_mappings": tmp_path / "question_skill_mappings.json",
        "reviewed_source_skills": tmp_path / "reviewed_source_skills.json",
        "reviewed_mark_events": tmp_path / "reviewed_mark_events.json",
    }
    _write_json(paths["candidates"], {"schema_name": "asterion.content_lab_candidates", "schema_version": 1, "candidates": candidates})
    _write_json(paths["asterion_bank"], {"schema_name": "asterion.question_bank", "schema_version": 1, "questions": questions})
    _write_json(paths["question_bank"], {"schema_name": "exam_bank.question_bank", "schema_version": 2, "questions": questions})
    _write_json(paths["topic_routing"], topic_routing)
    _write_json(paths["skill_map"], skill_map)
    _write_json(paths["question_skill_mappings"], mappings)
    _write_json(
        paths["reviewed_source_skills"],
        {
            "records": [
                _reviewed_source_skill_record(1, status="approved", route_status="clean"),
                _reviewed_source_skill_record(5, status="review_needed", route_status="review_needed", blockers=["ambiguous"]),
            ]
        },
    )
    _write_json(
        paths["reviewed_mark_events"],
        {
            "decisions": [
                {
                    "event_id": "31spring24_q01_me0001",
                    "status": "approved",
                    "satisfies_generation_gate": True,
                    "source_question_id": "31spring24_q01",
                    "part_path": ["whole"],
                }
            ]
        },
    )
    return paths


def _question(question_number: int) -> dict:
    question_id = f"31spring24_q{question_number:02d}"
    return {
        "question_id": question_id,
        "paper": "31spring24",
        "paper_family": "p3",
        "question_number": str(question_number),
        "canonical_question_artifact": f"p3/31spring24/questions/q{question_number:02d}.png",
        "canonical_mark_scheme_artifact": f"p3/31spring24/mark_scheme/q{question_number:02d}.png",
        "topic": "algebra" if question_number != 2 else "vectors",
        "notes": {"mapping_status": "pass", "validation_status": "pass"},
    }


def _candidate(
    question_number: int,
    *,
    generation_status: str,
    reviewed_subpart: bool = True,
    generation_reasons: list[str] | None = None,
) -> dict:
    question_id = f"31spring24_q{question_number:02d}"
    subpart_id = f"{question_id}_whole"
    blocked = generation_status != "allow"
    skill_id = "9709_p3_3_1_algebra" if question_number != 2 else "9709_p3_3_7_vectors"
    return {
        "candidate_id": f"content_lab_{subpart_id}",
        "question_id": question_id,
        "paper": "31spring24",
        "paper_family": "p3",
        "question_number": str(question_number),
        "subpart_id": subpart_id,
        "candidate_selection": {
            "reviewed_or_approved_subpart": reviewed_subpart,
            "high_confidence_subpart": True,
            "minimum_mark_event_confidence": 0.9,
        },
        "source_artifacts": {
            "question_crop_path": f"p3/31spring24/questions/q{question_number:02d}.png",
            "mark_scheme_crop_path": f"p3/31spring24/mark_scheme/q{question_number:02d}.png",
        },
        "source_skill_ids": [skill_id],
        "source_mark_event_ids": [f"{question_id}_me0001"],
        "reviewed_source_skill_ids": [skill_id],
        "source_skill_review_satisfied": True,
        "source_skill_review_gate": {"status": "allow", "blocked": False, "block_reasons": []},
        "mapping_review_satisfied": True,
        "mapping_review_gate": {"status": "allow", "blocked": False, "block_reasons": []},
        "source_mark_event_count": 1,
        "mark_event_review_gate": {"status": "allow", "blocked": False, "block_reasons": []},
        "generation_gate": {"status": generation_status, "blocked": blocked, "block_reasons": generation_reasons or []},
        "review_status": "ready" if generation_status == "allow" else "blocked_until_reviewed",
    }


def _reviewed_source_skill_record(
    question_number: int,
    *,
    status: str,
    route_status: str,
    blockers: list[str] | None = None,
) -> dict:
    question_id = f"31spring24_q{question_number:02d}"
    subpart_id = f"{question_id}_whole"
    return {
        "evidence_id": f"reviewed:{subpart_id}",
        "question_id": question_id,
        "subpart_id": subpart_id,
        "part_id": "whole",
        "route_status": route_status,
        "reviewer": {"review_status": status},
        "reviewed_source_skill_ids": ["9709_p3_3_1_algebra"],
        "mark_event_refs": [{"event_id": f"{question_id}_me0001"}],
        "blockers": blockers or [],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
