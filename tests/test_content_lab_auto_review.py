from __future__ import annotations

import csv
import json
from pathlib import Path

from exam_bank.asterion_content_lab_audit import run_audit
from exam_bank.content_lab_auto_review import (
    AUTO_REVIEW_DECISION_VERSION,
    AUTO_REVIEW_SOURCE,
    build_full_pool_auto_review_batch,
    build_full_pool_classification,
    build_auto_review_batch,
    import_auto_review_decisions,
    merge_auto_review_import_files,
    write_codex_agentic_review_decisions,
    validate_auto_review_decision,
    validate_mapping_correction_provenance,
)


def test_auto_review_batch_prioritizes_sample_failures_and_balances_regions(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=8)
    payload = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=4,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )

    rows = payload["rows"]
    assert [row["candidate_id"] for row in rows] == [
        "content_lab_q1_whole",
        "content_lab_q2_whole",
        "content_lab_q3_whole",
        "content_lab_q4_whole",
    ]
    assert all(row["in_deterministic_sample"] for row in rows)
    assert {row["region"] for row in rows} == {"Algebra Vault", "Vectors Gate"}
    assert (tmp_path / "batch" / "auto_review_batch.json").exists()
    assert (tmp_path / "batch" / "auto_review_batch.csv").exists()
    assert (tmp_path / "batch" / "auto_review_batch.md").exists()
    assert (tmp_path / "batch" / "auto_review_manifest.json").exists()


def test_auto_review_batch_skips_missing_canonical_images(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=2)
    (paths["artifact_root"] / "p3" / "paper" / "mark_scheme" / "q01.png").unlink()

    payload = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=2,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )

    assert "content_lab_q1_whole" not in {row["candidate_id"] for row in payload["rows"]}
    skipped = {row["candidate_id"]: row for row in payload["skipped_rows"]}
    assert "missing_canonical_mark_scheme_image" in skipped["content_lab_q1_whole"]["reasons"]


def test_auto_review_schema_rejects_low_confidence_partial_ambiguous_unknowns_and_missing_metadata(
    tmp_path: Path,
) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=1)
    batch = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=1,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    row = batch["rows"][0]
    skill_ids = {"9709_p3_3_1_modulus_equations_inequalities"}
    event_ids = {"q1_me0001"}

    cases = [
        ("confidence_below_threshold", {"confidence": 0.89}),
        ("exact_skill_decision_not_approved", {"exact_skill_decision": "ambiguous"}),
        ("candidate_generation_decision_not_approved", {"candidate_generation_decision": "blocked"}),
        ("blocking_risk_flags_present", {"risk_flags": ["ambiguous_mapping"]}),
        ("reviewer_model_or_provider_missing", {"reviewer": {"provider": "openai", "prompt_version": "v1"}}),
        ("reviewer_verifier_disagreement", {"adjudication": {"status": "approved", "reviewer_verifier_agree": False}}),
        ("unknown_skill_ids:unknown_skill", {"approved_exact_skill_ids": ["unknown_skill"]}),
        ("unknown_mark_event_refs:unknown_event", {"approved_mark_event_refs": [{"event_id": "unknown_event"}]}),
    ]
    for expected_error, overrides in cases:
        decision = _decision(row)
        decision.update(overrides)
        errors = validate_auto_review_decision(
            decision,
            batch_row=row,
            skill_ids=skill_ids,
            mark_event_ids=event_ids,
            artifact_root=paths["artifact_root"],
        )
        assert expected_error in errors


def test_importer_rejects_unknown_candidate_and_accepts_valid_decision(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=1)
    batch = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=1,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    batch_path = tmp_path / "batch" / "auto_review_batch.json"
    decisions_path = tmp_path / "decisions.jsonl"
    unknown = _decision(batch["rows"][0])
    unknown["candidate_id"] = "missing"
    decisions_path.write_text(json.dumps(unknown) + "\n", encoding="utf-8")

    rejected = import_auto_review_decisions(
        decisions_path=decisions_path,
        batch_path=batch_path,
        out_review_file=tmp_path / "reviewed.json",
        dry_run=True,
        skill_map_path=paths["skill_map"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
    )
    assert rejected["ok"] is False
    assert any("unknown_candidate_id" in error for error in rejected["errors"])

    decisions_path.write_text(json.dumps(_decision(batch["rows"][0])) + "\n", encoding="utf-8")
    accepted = import_auto_review_decisions(
        decisions_path=decisions_path,
        batch_path=batch_path,
        out_review_file=tmp_path / "reviewed.json",
        dry_run=False,
        skill_map_path=paths["skill_map"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
    )
    assert accepted["ok"] is True
    payload = json.loads((tmp_path / "reviewed.json").read_text(encoding="utf-8"))
    assert payload["review_source"] == AUTO_REVIEW_SOURCE
    assert payload["record_count"] == 1
    assert payload["source_skill_records"]
    assert payload["mark_event_decisions"]


def test_importer_rejects_invalid_decisions(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=1)
    batch = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=1,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    batch_path = tmp_path / "batch" / "auto_review_batch.json"
    cases = [
        ("confidence_below_threshold", {"confidence": 0.2}),
        ("source_skill_decision_not_approved", {"source_skill_decision": "blocked"}),
        ("blocking_risk_flags_present", {"risk_flags": ["quarantined_mark_scheme"]}),
        ("reviewer_model_or_provider_missing", {"reviewer": {"provider": "openai", "prompt_version": "v1"}}),
        ("reviewer_verifier_disagreement", {"adjudication": {"status": "approved", "reviewer_verifier_agree": False}}),
        ("unknown_skill_ids:unknown_skill", {"approved_source_skill_ids": ["unknown_skill"]}),
        ("unknown_mark_event_refs:unknown_event", {"approved_mark_event_refs": [{"event_id": "unknown_event"}]}),
    ]
    for expected_error, overrides in cases:
        decision = _decision(batch["rows"][0])
        decision.update(overrides)
        decisions_path = tmp_path / f"{expected_error.split(':')[0]}.jsonl"
        decisions_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
        report = import_auto_review_decisions(
            decisions_path=decisions_path,
            batch_path=batch_path,
            out_review_file=tmp_path / "reviewed.json",
            dry_run=True,
            skill_map_path=paths["skill_map"],
            mark_events_path=paths["mark_events"],
            artifact_root=paths["artifact_root"],
        )
        assert report["ok"] is False
        assert any(expected_error in error for error in report["errors"])


def test_corrected_mapping_provenance_rejects_unchanged_rejected_mapping(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=1)
    batch = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=1,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    decision = _decision(batch["rows"][0])
    decision["mapping_correction"] = {
        "source_loop": "loop_004",
        "original_proposed_skill_ids": decision["approved_source_skill_ids"],
        "original_rejection_reason": "Loop 003B rejected this unsupported mapping.",
        "correction_decision": "existing_candidate_can_be_safely_approved_with_corrected_reviewed_metadata",
        "canonical_evidence_refs": decision["evidence_refs"],
    }

    assert "corrected_mapping_matches_rejected_original" in validate_mapping_correction_provenance(decision)

    decision["approved_source_skill_ids"] = ["9709_p3_3_7_vector_lines"]
    decision["approved_exact_skill_ids"] = ["9709_p3_3_7_vector_lines"]
    assert validate_mapping_correction_provenance(decision) == []


def test_corrected_mapping_approval_preserves_rejection_provenance(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=2)
    batch = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=1,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    row = batch["rows"][0]
    decision = _decision(row)
    original = decision["approved_source_skill_ids"]
    corrected = ["9709_p3_3_7_vector_lines"]
    decision["approved_source_skill_ids"] = corrected
    decision["approved_exact_skill_ids"] = corrected
    decision["mapping_correction"] = {
        "source_loop": "loop_004",
        "original_proposed_skill_ids": original,
        "original_rejection_reason": "Loop 003B rejected the original skill as unsupported.",
        "correction_decision": "existing_candidate_can_be_safely_approved_with_corrected_reviewed_metadata",
        "canonical_evidence_refs": decision["evidence_refs"],
    }
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")

    report = import_auto_review_decisions(
        decisions_path=decisions_path,
        batch_path=tmp_path / "batch" / "auto_review_batch.json",
        out_review_file=tmp_path / "reviewed.json",
        dry_run=False,
        skill_map_path=paths["skill_map"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
    )

    assert report["ok"] is True
    payload = json.loads((tmp_path / "reviewed.json").read_text(encoding="utf-8"))
    record = payload["records"][0]
    assert record["mapping_correction"]["original_proposed_skill_ids"] == original
    assert record["mapping_correction"]["corrected_approved_skill_ids"] if "corrected_approved_skill_ids" in record["mapping_correction"] else corrected
    assert payload["source_skill_records"][0]["provenance"]["mapping_correction"]["original_rejection_reason"]


def test_corrected_mapping_approval_requires_canonical_evidence(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=1)
    batch = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=1,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    row = batch["rows"][0]
    decision = _decision(row)
    decision["evidence_refs"] = [{"type": "canonical_question_image", "path": row["canonical_question_image_path"]}]
    errors = validate_auto_review_decision(
        decision,
        batch_row=row,
        skill_ids={"9709_p3_3_1_modulus_equations_inequalities"},
        mark_event_ids={"q1_me0001"},
        artifact_root=paths["artifact_root"],
    )

    assert "canonical_mark_scheme_image_evidence_missing" in errors


def test_merge_reviewed_decisions_preserves_records_and_rejects_conflicts(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=2)
    batch = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=2,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    review_files = []
    for index, row in enumerate(batch["rows"], start=1):
        decisions_path = tmp_path / f"decisions_{index}.jsonl"
        decisions_path.write_text(json.dumps(_decision(row)) + "\n", encoding="utf-8")
        review_file = tmp_path / f"reviewed_{index}.json"
        import_auto_review_decisions(
            decisions_path=decisions_path,
            batch_path=tmp_path / "batch" / "auto_review_batch.json",
            out_review_file=review_file,
            dry_run=False,
            skill_map_path=paths["skill_map"],
            mark_events_path=paths["mark_events"],
            artifact_root=paths["artifact_root"],
        )
        review_files.append(review_file)

    merged = merge_auto_review_import_files(
        reviewed_files=review_files,
        out_review_file=tmp_path / "merged.json",
        dry_run=False,
    )
    assert merged["ok"] is True
    assert json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))["record_count"] == 2

    conflicting = json.loads(review_files[1].read_text(encoding="utf-8"))
    conflicting["records"][0]["confidence"] = 0.91
    conflict_file = tmp_path / "conflict.json"
    _write_json(conflict_file, conflicting)
    rejected = merge_auto_review_import_files(
        reviewed_files=[review_files[1], conflict_file],
        out_review_file=tmp_path / "bad_merge.json",
        dry_run=True,
    )
    assert rejected["ok"] is False
    assert any("duplicate_conflicting_decision" in error for error in rejected["errors"])


def test_audit_consumes_valid_automated_decisions_but_not_raw_mark_values(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=1)
    before = run_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=None,
        asterion_bank_path=paths["question_bank"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "before",
        sample_size=1,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
        reviewed_source_skills_path=tmp_path / "empty_source.json",
        reviewed_mark_events_path=tmp_path / "empty_marks.json",
    )
    assert before["sample_passed"] == 0

    batch = build_auto_review_batch(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        target_pass_count=1,
        buffer_count=0,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(json.dumps(_decision(batch["rows"][0])) + "\n", encoding="utf-8")
    import_auto_review_decisions(
        decisions_path=decisions_path,
        batch_path=tmp_path / "batch" / "auto_review_batch.json",
        out_review_file=tmp_path / "reviewed.json",
        dry_run=False,
        skill_map_path=paths["skill_map"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
    )

    after = run_audit(
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        topic_routing_path=None,
        asterion_bank_path=paths["question_bank"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "after",
        sample_size=1,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
        reviewed_source_skills_path=tmp_path / "empty_source.json",
        reviewed_mark_events_path=tmp_path / "empty_marks.json",
        reviewed_decisions_path=tmp_path / "reviewed.json",
    )
    assert after["sample_passed"] == 1


def test_full_pool_classification_and_batch_exclude_sample_rows(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=6)
    mappings = json.loads(paths["mappings"].read_text(encoding="utf-8"))
    mappings["mappings"][4]["confidence"] = 0.82
    mappings["mappings"][4]["secondary_skill_ids"] = ["9709_p3_3_7_vector_lines"]
    _write_json(paths["mappings"], mappings)
    inventory_rows = list(csv.DictReader((paths["audit_dir"] / "p3_candidate_inventory.csv").open(encoding="utf-8")))
    inventory_rows[0]["passed"] = "True"
    _write_csv(paths["audit_dir"] / "p3_candidate_inventory.csv", inventory_rows)
    backlog_path = tmp_path / "regeneration_backlog.json"
    _write_json(backlog_path, {"rows": [{"original_candidate_id": "content_lab_q6_whole"}]})

    payload = build_full_pool_classification(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "review",
        loop005_regeneration_backlog_path=backlog_path,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    counts = payload["classification_counts"]
    assert counts["eligible_for_direct_agentic_review"] == 3
    assert counts["mapping_correctable"] == 1
    assert counts["regenerate_candidate_required"] == 1
    assert payload["estimated_additional_approvals_needed"]["for_70_percent"] == 4
    assert "content_lab_q1_whole" not in {row["candidate_id"] for row in payload["rows"]}
    assert (tmp_path / "review" / "full_pool_classification.csv").exists()
    assert (tmp_path / "review" / "full_pool_classification.md").exists()

    batch = build_full_pool_auto_review_batch(
        classification_path=tmp_path / "review" / "full_pool_classification.json",
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        limit=10,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    assert batch["manifest"]["selected_sample_count"] == 0
    with (paths["audit_dir"] / "sample_results.csv").open(encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    assert {row["candidate_id"] for row in batch["rows"]}.isdisjoint(
        {row["candidate_id"] for row in sample_rows}
    )
    assert "content_lab_q6_whole" not in {row["candidate_id"] for row in batch["rows"]}


def test_codex_decisions_import_and_improve_only_from_valid_reviewed_evidence(tmp_path: Path) -> None:
    paths = _write_auto_review_fixture(tmp_path, row_count=4)
    classification = build_full_pool_classification(
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "review",
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    batch = build_full_pool_auto_review_batch(
        classification_path=tmp_path / "review" / "full_pool_classification.json",
        audit_dir=paths["audit_dir"],
        candidates_path=paths["candidates"],
        question_bank_path=paths["question_bank"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "batch",
        limit=2,
        skill_map_path=paths["skill_map"],
        question_skill_mappings_path=paths["mappings"],
    )
    summary = write_codex_agentic_review_decisions(
        batch_path=tmp_path / "batch" / "auto_review_batch.json",
        out_dir=tmp_path / "batch",
    )
    assert summary["approved_count"] == len(batch["rows"])
    report = import_auto_review_decisions(
        decisions_path=tmp_path / "batch" / "auto_review_decisions.jsonl",
        batch_path=tmp_path / "batch" / "auto_review_batch.json",
        out_review_file=tmp_path / "reviewed.json",
        dry_run=False,
        skill_map_path=paths["skill_map"],
        mark_events_path=paths["mark_events"],
        artifact_root=paths["artifact_root"],
    )
    assert report["ok"] is True
    assert report["accepted_count"] == summary["approved_count"]
    assert classification["classification_counts"]["eligible_for_direct_agentic_review"] == 4


def _write_auto_review_fixture(tmp_path: Path, *, row_count: int) -> dict[str, Path]:
    artifact_root = tmp_path / "output"
    (artifact_root / "p3" / "paper" / "questions").mkdir(parents=True)
    (artifact_root / "p3" / "paper" / "mark_scheme").mkdir(parents=True)
    candidates = []
    questions = []
    mark_records = []
    mappings = []
    sample_rows = []
    queue_rows = []
    inventory_rows = []
    for index in range(1, row_count + 1):
        question_id = f"q{index}"
        subpart_id = f"{question_id}_whole"
        candidate_id = f"content_lab_{subpart_id}"
        skill_id = (
            "9709_p3_3_1_modulus_equations_inequalities"
            if index % 2
            else "9709_p3_3_7_vector_lines"
        )
        region = "Algebra Vault" if index % 2 else "Vectors Gate"
        (artifact_root / "p3" / "paper" / "questions" / f"q{index:02d}.png").write_bytes(b"question")
        (artifact_root / "p3" / "paper" / "mark_scheme" / f"q{index:02d}.png").write_bytes(b"mark")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "question_id": question_id,
                "paper": "paper",
                "paper_family": "p3",
                "question_number": str(index),
                "subpart_id": subpart_id,
                "subpart_label": "whole",
                "marks": 1,
                "candidate_selection": {"reviewed_or_approved_subpart": False},
                "source_artifacts": {
                    "question_crop_path": f"p3/paper/questions/q{index:02d}.png",
                    "mark_scheme_crop_path": f"p3/paper/mark_scheme/q{index:02d}.png",
                },
                "source_skill_ids": [],
                "reviewed_source_skill_ids": [],
                "source_mark_event_count": 1,
                "source_mark_event_ids": [f"{question_id}_me0001"],
                "source_skill_review_satisfied": False,
                "mapping_review_satisfied": False,
                "source_skill_review_gate": {"status": "blocked_until_reviewed"},
                "mapping_review_gate": {"status": "blocked_until_reviewed"},
                "mark_event_review_gate": {"status": "blocked_until_reviewed"},
                "generation_gate": {"status": "blocked_until_reviewed", "block_reasons": ["reviewed_mark_event_decision_missing"]},
                "review_status": "blocked_until_reviewed",
            }
        )
        questions.append(
            {
                "question_id": question_id,
                "paper": "paper",
                "paper_family": "p3",
                "question_number": str(index),
                "canonical_question_artifact": f"p3/paper/questions/q{index:02d}.png",
                "canonical_mark_scheme_artifact": f"p3/paper/mark_scheme/q{index:02d}.png",
                "topic": "vectors" if index % 2 == 0 else "modulus",
                "notes": {"mapping_status": "pass", "validation_status": "pass"},
            }
        )
        mark_records.append(
            {
                "question_id": question_id,
                "mark_events": [
                    {
                        "event_id": f"{question_id}_me0001",
                        "part_path": ["whole"],
                        "mark_code_raw": "B1",
                        "mark_type": "independent_statement",
                        "mark_value": 1,
                        "confidence": "high",
                        "raw_text": "B1",
                    }
                ],
            }
        )
        mappings.append(
            {
                "mapping_id": f"mapping_{question_id}",
                "question_id": question_id,
                "subpart_id": subpart_id,
                "primary_skill_ids": [skill_id],
                "secondary_skill_ids": [],
                "confidence": 0.95,
                "review_status": "needs_review",
                "evidence": {},
            }
        )
        row = {
            "candidate_id": candidate_id,
            "question_id": question_id,
            "subpart_id": subpart_id,
            "region": region,
            "passed": "False",
            "blocker_classes": "blocked_skill_mapping|blocked_mapping_review_gate|blocked_mark_events|review_required",
            "blocker_reasons": "missing_source_skill_ids|reviewed_mark_event_decision_missing",
        }
        inventory_rows.append(row)
        queue_rows.append(
            {
                **row,
                "human_review_reasons": "review_exact_skill_source_skill_mapping|review_mark_events_for_candidate_part",
            }
        )
        if index <= max(1, row_count // 2):
            sample_rows.append(row)

    skill_map = {
        "skills": [
            {"skill_id": "9709_p3_3_1_modulus_equations_inequalities", "section": "3.1 Algebra"},
            {"skill_id": "9709_p3_3_7_vector_lines", "section": "3.7 Vectors"},
        ]
    }
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    _write_csv(audit_dir / "sample_results.csv", sample_rows)
    _write_csv(audit_dir / "human_review_queue.csv", queue_rows)
    _write_csv(audit_dir / "p3_candidate_inventory.csv", inventory_rows)
    _write_json(audit_dir / "audit_summary.json", {"sample_passed": 0})
    paths = {
        "artifact_root": artifact_root,
        "audit_dir": audit_dir,
        "candidates": tmp_path / "candidates.json",
        "question_bank": tmp_path / "question_bank.json",
        "mark_events": tmp_path / "mark_events.json",
        "skill_map": tmp_path / "skill_map.json",
        "mappings": tmp_path / "mappings.json",
    }
    _write_json(paths["candidates"], {"candidates": candidates})
    _write_json(paths["question_bank"], {"questions": questions})
    _write_json(paths["mark_events"], {"records": mark_records})
    _write_json(paths["skill_map"], skill_map)
    _write_json(paths["mappings"], {"mappings": mappings})
    _write_json(tmp_path / "empty_source.json", {"records": []})
    _write_json(tmp_path / "empty_marks.json", {"decisions": []})
    return paths


def _decision(row: dict) -> dict:
    return {
        "decision_version": AUTO_REVIEW_DECISION_VERSION,
        "review_source": AUTO_REVIEW_SOURCE,
        "candidate_id": row["candidate_id"],
        "question_id": row["question_id"],
        "subpart_id": row["subpart_id"],
        "exact_skill_decision": "approved",
        "source_skill_decision": "approved",
        "mark_event_decision": "approved",
        "candidate_generation_decision": "approved",
        "confidence": 0.95,
        "evidence_refs": [
            {"type": "canonical_question_image", "path": row["canonical_question_image_path"]},
            {"type": "canonical_mark_scheme_image", "path": row["canonical_mark_scheme_image_path"]},
        ],
        "approved_exact_skill_ids": row["proposed_exact_skill_ids"],
        "approved_source_skill_ids": row["proposed_source_skill_ids"],
        "approved_mark_event_refs": [
            {"event_id": event_id, "part_path": ["whole"]} for event_id in row["source_mark_event_ids"]
        ],
        "explanation": "Canonical question and mark-scheme images support the mapping and event refs.",
        "risk_flags": [],
        "reviewer": {"provider": "openai", "model": "gpt-5-mini", "prompt_version": "v1"},
        "verifier": {"provider": "deterministic_validator", "model": "content_lab_auto_review_validator_v1", "prompt_version": "v1"},
        "adjudication": {"status": "approved", "reviewer_verifier_agree": True},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
