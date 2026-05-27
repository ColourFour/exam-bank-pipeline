from __future__ import annotations

import json
from pathlib import Path

from exam_bank.asterion_student_runtime_safe import (
    PROMOTION_DECISIONS_SCHEMA,
    PROMOTION_DECISIONS_SCHEMA_VERSION,
    RUNTIME_SAFE_CONTRACT_VERSION,
    run_runtime_safe_audit,
    validate_promotion_decisions_payload,
)


def test_runtime_safe_audit_promotes_only_contract_satisfying_candidates(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "audit"
    promotion_path = tmp_path / "data" / "review" / "runtime_safe.json"
    export_decisions = tmp_path / "output" / "asterion" / "exports" / "latest" / "decisions.json"
    export_candidates = tmp_path / "output" / "asterion" / "exports" / "latest" / "safe_candidates.json"

    summary = run_runtime_safe_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        mark_events_path=paths["mark_events"],
        reviewed_decisions_path=paths["reviewed_decisions"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=out_dir,
        target_pass_rate=0.50,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        deterministic_sample_path=paths["sample"],
        regeneration_backlog_path=paths["backlog"],
        write_promotion_decisions_path=promotion_path,
        write_export_decisions_path=export_decisions,
        write_export_candidates_path=export_candidates,
    )

    assert summary["total_p3_candidate_count"] == 7
    assert summary["current_student_runtime_safe_true_count"] == 0
    assert summary["final_student_runtime_safe_true_count"] == 1
    assert summary["target_true_count_for_50_percent"] == 4
    assert summary["additional_candidates_needed_for_50_percent_final"] == 3
    assert summary["target_met"] is False
    assert summary["trust_gates_weakened"] is False
    assert summary["student_runtime_app_changed"] is False
    assert summary["safe_promotion_decision_count"] == 1
    assert summary["region_summary"][0]["region"] == "Algebra Vault"
    assert summary["region_summary"][0]["final_student_runtime_safe_true_count"] == 1
    assert summary["region_summary"][0]["final_student_runtime_safe_percentage"] == 0.1429

    promotion_payload = json.loads(promotion_path.read_text(encoding="utf-8"))
    assert promotion_payload["decision_count"] == 1
    assert promotion_payload["decisions"][0]["candidate_id"] == "content_lab_31spring24_q01_whole"
    assert promotion_payload["decisions"][0]["promotion_decision"] == "student_runtime_safe"

    rows = {
        row["candidate_id"]: row
        for row in json.loads((out_dir / "runtime_safe_classification.json").read_text(encoding="utf-8"))["rows"]
    }
    assert rows["content_lab_31spring24_q01_whole"]["classification"] == "runtime_safe_now"
    assert rows["content_lab_31spring24_q02_whole"]["classification"] == "artifact_or_schema_repairable"
    assert "canonical_question_image" in rows["content_lab_31spring24_q02_whole"]["blocker_classes"]
    assert rows["content_lab_31spring24_q03_whole"]["classification"] == "needs_candidate_regeneration"
    assert "ambiguous_or_quarantined" in rows["content_lab_31spring24_q03_whole"]["blocker_classes"]
    assert rows["content_lab_31spring24_q04_whole"]["classification"] == "generation_seed_only"
    assert rows["content_lab_31spring24_q05_whole"]["classification"] == "teacher_preview_only"
    assert rows["content_lab_31spring24_q06_whole"]["classification"] == "needs_reviewed_evidence"
    assert "reviewed_exact_skill_mapping" in rows["content_lab_31spring24_q06_whole"]["blocker_classes"]
    assert rows["content_lab_31spring24_q07_whole"]["classification"] == "reviewed_but_not_runtime_safe"
    assert "unsafe_advisory_text" in rows["content_lab_31spring24_q07_whole"]["blocker_classes"]

    exported = json.loads(export_candidates.read_text(encoding="utf-8"))
    assert exported["record_count"] == 1
    assert exported["candidates"][0]["student_runtime_safe"] is True
    assert exported["candidates"][0]["candidate_id"] == "content_lab_31spring24_q01_whole"

    rerun = run_runtime_safe_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        mark_events_path=paths["mark_events"],
        reviewed_decisions_path=paths["reviewed_decisions"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "audit_rerun",
        target_pass_rate=0.50,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        deterministic_sample_path=paths["sample"],
        regeneration_backlog_path=paths["backlog"],
        promotion_decisions_path=promotion_path,
    )
    assert rerun["final_student_runtime_safe_true_count"] == 1
    assert rerun["safe_promotion_decision_count"] == 1
    assert rerun["target_true_count_for_50_percent"] == 4


def test_promotion_decisions_validate_provenance_and_reject_conflicts(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    candidates = json.loads(paths["candidates"].read_text(encoding="utf-8"))["candidates"]
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    valid = _promotion_decision("content_lab_31spring24_q01_whole")

    errors, valid_index = validate_promotion_decisions_payload(
        {
            "schema": PROMOTION_DECISIONS_SCHEMA,
            "schema_version": PROMOTION_DECISIONS_SCHEMA_VERSION,
            "decision_count": 1,
            "decisions": [valid],
        },
        candidates_by_id=candidates_by_id,
        artifact_root=paths["artifact_root"],
    )
    assert errors == []
    assert set(valid_index) == {"content_lab_31spring24_q01_whole"}

    missing_provenance = dict(valid)
    missing_provenance.pop("provenance")
    conflict = dict(valid)
    conflict["question_id"] = "31spring24_q99"
    errors, valid_index = validate_promotion_decisions_payload(
        {
            "schema": PROMOTION_DECISIONS_SCHEMA,
            "schema_version": PROMOTION_DECISIONS_SCHEMA_VERSION,
            "decision_count": 3,
            "decisions": [valid, missing_provenance, conflict],
        },
        candidates_by_id=candidates_by_id,
        artifact_root=paths["artifact_root"],
    )
    assert valid_index == {"content_lab_31spring24_q01_whole": valid}
    assert any("missing_required_field:provenance" in error for error in errors)
    assert any("duplicate_conflicting_candidate_decision" in error for error in errors)
    assert any("question_id_mismatch" in error for error in errors)


def test_artifact_schema_diagnosis_reports_subtypes_and_existing_images(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "audit"

    run_runtime_safe_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        mark_events_path=paths["mark_events"],
        reviewed_decisions_path=paths["reviewed_decisions"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=out_dir,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        deterministic_sample_path=paths["sample"],
        regeneration_backlog_path=paths["backlog"],
    )

    diagnosis = json.loads((out_dir / "artifact_schema_blocker_diagnosis.json").read_text(encoding="utf-8"))
    assert diagnosis["candidate_count"] == 1
    assert diagnosis["blocker_subtype_counts"]["missing_question_image"] == 1
    assert diagnosis["blocker_subtype_counts"]["missing_mark_scheme_image"] == 0
    assert diagnosis["blocker_subtype_counts"]["genuine_not_runtime_safe"] == 1
    row = diagnosis["rows"][0]
    assert row["candidate_id"] == "content_lab_31spring24_q02_whole"
    assert row["question_image_exists"] is False
    assert row["mark_scheme_image_exists"] is True
    assert row["safely_promotable_after_repair"] is False


def test_stale_runtime_role_promotes_only_with_validated_runtime_safe_decision(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "audit"
    promotion_path = tmp_path / "runtime_safe_decisions.json"
    export_candidates = tmp_path / "safe_candidates.json"
    asterion = json.loads(paths["asterion_bank"].read_text(encoding="utf-8"))
    q1 = next(question for question in asterion["questions"] if question["question_id"] == "31spring24_q01")
    q1["usage_roles"]["canonical_practice"] = "block"
    q1["usage_roles"]["field_guide_source"] = "block_until_reviewed"
    q1["usage_roles"]["guardian_candidate"] = "block"
    q1["usage_roles"]["warmup_generator_source"] = "block_until_reviewed"
    _write_json(paths["asterion_bank"], asterion)

    summary = run_runtime_safe_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        mark_events_path=paths["mark_events"],
        reviewed_decisions_path=paths["reviewed_decisions"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=out_dir,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        deterministic_sample_path=paths["sample"],
        regeneration_backlog_path=paths["backlog"],
        write_promotion_decisions_path=promotion_path,
        write_export_candidates_path=export_candidates,
    )

    assert summary["final_student_runtime_safe_true_count"] == 1
    rows = {
        row["candidate_id"]: row
        for row in json.loads((out_dir / "runtime_safe_classification.json").read_text(encoding="utf-8"))["rows"]
    }
    assert rows["content_lab_31spring24_q01_whole"]["runtime_role_projection_applied"] is True
    assert rows["content_lab_31spring24_q01_whole"]["classification"] == "runtime_safe_now"
    assert rows["content_lab_31spring24_q04_whole"]["final_student_runtime_safe"] is False
    assert rows["content_lab_31spring24_q05_whole"]["final_student_runtime_safe"] is False
    assert rows["content_lab_31spring24_q03_whole"]["final_student_runtime_safe"] is False

    exported = json.loads(export_candidates.read_text(encoding="utf-8"))
    assert exported["candidates"][0]["runtime_role_projection_applied"] is True
    assert exported["candidates"][0]["provenance"]["contract_version"] == RUNTIME_SAFE_CONTRACT_VERSION
    assert exported["candidates"][0]["reviewed_decision_refs"] == [
        "content_lab_auto_review:content_lab_31spring24_q01_whole"
    ]


def test_runtime_role_is_not_bypassed_without_reviewed_evidence(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "audit"
    promotion_path = tmp_path / "runtime_safe_decisions.json"
    reviewed = json.loads(paths["reviewed_decisions"].read_text(encoding="utf-8"))
    reviewed["records"] = [record for record in reviewed["records"] if record["candidate_id"] != "content_lab_31spring24_q01_whole"]
    _write_json(paths["reviewed_decisions"], reviewed)
    _write_json(
        promotion_path,
        {
            "schema": PROMOTION_DECISIONS_SCHEMA,
            "schema_version": PROMOTION_DECISIONS_SCHEMA_VERSION,
            "decision_count": 1,
            "decisions": [_promotion_decision("content_lab_31spring24_q01_whole")],
        },
    )

    summary = run_runtime_safe_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        mark_events_path=paths["mark_events"],
        reviewed_decisions_path=paths["reviewed_decisions"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=out_dir,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        deterministic_sample_path=paths["sample"],
        regeneration_backlog_path=paths["backlog"],
        promotion_decisions_path=promotion_path,
    )

    assert summary["final_student_runtime_safe_true_count"] == 0
    assert any("reviewed_runtime_evidence_missing" in error for error in summary["promotion_validation_errors"])


def test_projection_rejects_promotion_not_backed_by_reviewed_refs(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "audit"
    promotion_path = tmp_path / "runtime_safe_decisions.json"
    decision = _promotion_decision("content_lab_31spring24_q01_whole")
    decision["exact_source_skill_evidence_refs"] = ["9709_p3_3_1_not_reviewed"]
    _write_json(
        promotion_path,
        {
            "schema": PROMOTION_DECISIONS_SCHEMA,
            "schema_version": PROMOTION_DECISIONS_SCHEMA_VERSION,
            "decision_count": 1,
            "decisions": [decision],
        },
    )

    summary = run_runtime_safe_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=paths["topic_routing"],
        asterion_bank_path=paths["asterion_bank"],
        mark_events_path=paths["mark_events"],
        reviewed_decisions_path=paths["reviewed_decisions"],
        reviewed_source_skills_path=paths["reviewed_source_skills"],
        reviewed_mark_events_path=paths["reviewed_mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=out_dir,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["question_skill_mappings"],
        deterministic_sample_path=paths["sample"],
        regeneration_backlog_path=paths["backlog"],
        promotion_decisions_path=promotion_path,
    )

    assert summary["final_student_runtime_safe_true_count"] == 0
    assert any("exact_source_skill_refs_not_validated_by_review" in error for error in summary["promotion_validation_errors"])


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "output"
    question_dir = artifact_root / "p3" / "31spring24" / "questions"
    mark_dir = artifact_root / "p3" / "31spring24" / "mark_scheme"
    question_dir.mkdir(parents=True)
    mark_dir.mkdir(parents=True)
    for index in range(1, 8):
        if index != 2:
            (question_dir / f"q{index:02d}.png").write_bytes(b"question")
        (mark_dir / f"q{index:02d}.png").write_bytes(b"mark")

    paths = {
        "artifact_root": artifact_root,
        "candidates": tmp_path / "candidates.json",
        "asterion_bank": tmp_path / "asterion_bank.json",
        "question_bank": tmp_path / "question_bank.json",
        "topic_routing": tmp_path / "topic_routing.json",
        "mark_events": tmp_path / "mark_events.json",
        "reviewed_decisions": tmp_path / "reviewed_decisions.json",
        "reviewed_source_skills": tmp_path / "reviewed_source_skills.json",
        "reviewed_mark_events": tmp_path / "reviewed_mark_events.json",
        "skill_map": tmp_path / "skill_map.json",
        "question_skill_mappings": tmp_path / "question_skill_mappings.json",
        "sample": tmp_path / "sample.csv",
        "backlog": tmp_path / "backlog.json",
    }
    questions = [_question(index) for index in range(1, 8)]
    candidates = [_candidate(index) for index in range(1, 8)]
    questions[3]["usage_roles"]["canonical_practice"] = "block"
    questions[3]["usage_roles"]["generated_warmup_pattern_source"] = "allow"
    questions[4]["usage_roles"]["canonical_practice"] = "block"
    questions[4]["usage_roles"]["field_guide_source"] = "allow"
    questions[4]["usage_roles"]["guardian_candidate"] = "allow"
    candidates[2]["generation_gate"]["block_reasons"] = ["mark_events_quarantined"]
    candidates[6]["mark_event_review_gate"]["advisory_event_ids"] = ["31spring24_q07_me0001"]

    reviewed_records = [_review_record(index) for index in [1, 2, 3, 4, 5, 7]]
    _write_json(paths["candidates"], {"schema_name": "asterion.content_lab_candidates", "schema_version": 1, "candidates": candidates})
    _write_json(paths["asterion_bank"], {"schema_name": "asterion.question_bank", "schema_version": 1, "questions": questions})
    _write_json(paths["question_bank"], {"schema_name": "exam_bank.question_bank", "schema_version": 2, "questions": questions})
    _write_json(paths["topic_routing"], {"records": {}})
    _write_json(paths["mark_events"], {"schema_name": "exam_bank.mark_events", "records": [_mark_event_record(index) for index in range(1, 8)]})
    _write_json(
        paths["reviewed_decisions"],
        {
            "schema": "exam_bank.content_lab.auto_reviewed_decisions",
            "schema_version": 1,
            "records": reviewed_records,
            "source_skill_records": [],
            "mark_event_decisions": [],
        },
    )
    _write_json(paths["reviewed_source_skills"], {"records": []})
    _write_json(paths["reviewed_mark_events"], {"decisions": []})
    _write_json(paths["skill_map"], {"skills": [{"skill_id": "9709_p3_3_1_algebra", "section": "3.1 Algebra"}]})
    _write_json(paths["question_skill_mappings"], {"mappings": []})
    paths["sample"].write_text("candidate_id\ncontent_lab_31spring24_q01_whole\n", encoding="utf-8")
    _write_json(paths["backlog"], {"rows": []})
    return paths


def _question(index: int) -> dict:
    question_id = f"31spring24_q{index:02d}"
    return {
        "question_id": question_id,
        "paper": "31spring24",
        "paper_family": "p3",
        "question_number": str(index),
        "canonical_question_artifact": f"p3/31spring24/questions/q{index:02d}.png",
        "canonical_mark_scheme_artifact": f"p3/31spring24/mark_scheme/q{index:02d}.png",
        "topic": "algebra",
        "notes": {"mapping_status": "pass", "validation_status": "pass"},
        "quality_gate": {"canonical_assets_ok": index != 2},
        "usage_roles": {
            "canonical_practice": "allow",
            "field_guide_source": "allow",
            "quick_check_source": "allow",
            "warmup_generator_source": "allow",
            "generated_warmup_pattern_source": "block",
            "guardian_candidate": "allow",
        },
        "subparts": [
            {
                "subpart_id": f"{question_id}_whole",
                "question_crop_path": f"p3/31spring24/questions/q{index:02d}.png",
                "mark_scheme_crop_path": f"p3/31spring24/mark_scheme/q{index:02d}.png",
            }
        ],
    }


def _candidate(index: int) -> dict:
    question_id = f"31spring24_q{index:02d}"
    subpart_id = f"{question_id}_whole"
    return {
        "candidate_id": f"content_lab_{subpart_id}",
        "question_id": question_id,
        "paper": "31spring24",
        "paper_family": "p3",
        "subpart_id": subpart_id,
        "source_artifacts": {
            "question_crop_path": f"p3/31spring24/questions/q{index:02d}.png",
            "mark_scheme_crop_path": f"p3/31spring24/mark_scheme/q{index:02d}.png",
        },
        "source_skill_ids": ["9709_p3_3_1_algebra"],
        "reviewed_source_skill_ids": ["9709_p3_3_1_algebra"],
        "source_mark_event_count": 1,
        "source_mark_event_ids": [f"{question_id}_me0001"],
        "mark_event_review_gate": {"status": "allow", "advisory_event_ids": []},
        "generation_gate": {"status": "allow", "block_reasons": []},
        "marks": 1,
    }


def _review_record(index: int) -> dict:
    question_id = f"31spring24_q{index:02d}"
    subpart_id = f"{question_id}_whole"
    return {
        "decision_id": f"content_lab_auto_review:content_lab_{subpart_id}",
        "candidate_id": f"content_lab_{subpart_id}",
        "question_id": question_id,
        "subpart_id": subpart_id,
        "review_source": "automated_agentic_review",
        "adjudication": {"status": "approved", "reviewer_verifier_agree": True},
        "confidence": 0.95,
        "approved_source_skill_ids": ["9709_p3_3_1_algebra"],
        "approved_exact_skill_ids": ["9709_p3_3_1_algebra"],
        "approved_mark_event_refs": [{"event_id": f"{question_id}_me0001", "part_path": ["whole"]}],
        "canonical_question_image_path": f"p3/31spring24/questions/q{index:02d}.png",
        "canonical_mark_scheme_image_path": f"p3/31spring24/mark_scheme/q{index:02d}.png",
        "risk_flags": [],
    }


def _mark_event_record(index: int) -> dict:
    question_id = f"31spring24_q{index:02d}"
    return {
        "question_id": question_id,
        "mark_events": [{"event_id": f"{question_id}_me0001", "part_path": ["whole"], "review_flags": []}],
    }


def _promotion_decision(candidate_id: str) -> dict:
    question_id = candidate_id.removeprefix("content_lab_").removesuffix("_whole")
    return {
        "schema_version": PROMOTION_DECISIONS_SCHEMA_VERSION,
        "decision_id": f"asterion_student_runtime_safe:{candidate_id}",
        "candidate_id": candidate_id,
        "question_id": question_id,
        "subpart_id": f"{question_id}_whole",
        "region_id": "Algebra Vault",
        "promotion_decision": "student_runtime_safe",
        "review_source": "student_runtime_safe_audit",
        "reviewed_decision_refs": [f"content_lab_auto_review:{candidate_id}"],
        "exact_source_skill_evidence_refs": ["9709_p3_3_1_algebra"],
        "mark_event_evidence_refs": [f"{question_id}_me0001"],
        "canonical_question_image_path": f"p3/31spring24/questions/{question_id[-3:]}.png",
        "canonical_mark_scheme_image_path": f"p3/31spring24/mark_scheme/{question_id[-3:]}.png",
        "runtime_safety_reasons": ["satisfies_strict_runtime_safety_contract"],
        "risk_flags": [],
        "provenance": {
            "contract_version": RUNTIME_SAFE_CONTRACT_VERSION,
            "created_by": "tests/test_asterion_student_runtime_safe.py",
        },
        "created_at": "2026-05-27T00:00:00Z",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
