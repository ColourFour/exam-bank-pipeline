from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.p3_exact_skill.reviewed_decisions import (
    validate_reviewed_decisions,
    validate_reviewed_decisions_payload,
)

P3_SKILL_IDS = {
    "9709_p3_3_2_log_exponential_equations",
    "9709_p3_3_1_polynomial_division_factor_remainder",
}

SEED_SUBPART_IDS = {
    "33summer23_q11_b",
    "31summer24_q04_b",
    "32summer23_q06_c",
    "32autumn23_q06_c",
    "33summer23_q06_b",
    "33summer23_q09_b",
    "32spring23_q05_b",
}

BATCH_0002_SEED_SUBPART_IDS = {
    "31summer24_q01_whole",
    "33autumn23_q01_whole",
    "33autumn23_q03_whole",
    "31summer21_q04_a",
    "32spring24_q03_a",
    "31summer24_q06_d",
    "31summer24_q09_b",
    "33summer25_q05_a",
}

UNPROMOTED_DRAFT_SUBPART_IDS = {
    "32autumn23_q09_b",
    "31autumn21_q07_c",
    "32summer23_q07_b",
    "33autumn23_q07_b",
    "33autumn22_q10_a",
    "33autumn22_q10_b",
    "31summer23_q02_a",
    "31autumn21_q04_whole",
    "31summer24_q06_c",
    "31summer24_q09_a",
    "32spring24_q05_b",
}

BATCH_0003_NOT_PROMOTED_SUBPART_IDS = {
    "32autumn23_q09_b",
    "31autumn21_q07_c",
    "32summer23_q07_b",
    "33autumn22_q10_a",
    "33autumn22_q10_b",
    "31autumn21_q04_whole",
    "33autumn23_q07_b",
    "31summer23_q02_a",
    "31summer24_q06_c",
    "31summer24_q09_a",
    "32spring24_q05_b",
}


def test_valid_conservative_fixture_passes() -> None:
    report = validate_reviewed_decisions(
        reviewed_decisions_path="data/review/p3_exact_skill_reviewed_decisions.v1.json",
        p3_skill_map_path="exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json",
    )

    assert report["ok"] is True
    assert report["record_count"] == 18
    assert report["warning_count"] == 15
    assert all("clean_with_unreviewed_mark_event_refs" in warning for warning in report["warnings"])


def test_clean_seed_records_validate_as_registry_evidence() -> None:
    payload = _load_registry()
    seed_records = [
        record
        for record in payload["records"]
        if record["subpart_id"] in SEED_SUBPART_IDS
    ]

    assert len(seed_records) == 7
    assert {record["subpart_id"] for record in seed_records} == SEED_SUBPART_IDS
    assert all(record["route_status"] == "clean" for record in seed_records)
    assert all(record["reviewed_source_skill_ids"] for record in seed_records)
    assert all(record["reviewer"]["reviewed_by"] == "codex_registry_seed_review" for record in seed_records)
    assert all(record["provenance"]["batch_id"] == "batch_0001" for record in seed_records)
    assert all(record["provenance"]["draft_response_artifact_kind"] == "human_review_response_draft" for record in seed_records)
    assert all(record["allowed_use_cases"]["source_backed_examples"] is True for record in seed_records)


def test_batch_0002_clean_seed_records_validate_as_registry_evidence() -> None:
    payload = _load_registry()
    seed_records = [
        record
        for record in payload["records"]
        if record["subpart_id"] in BATCH_0002_SEED_SUBPART_IDS
    ]

    assert len(seed_records) == 8
    assert {record["subpart_id"] for record in seed_records} == BATCH_0002_SEED_SUBPART_IDS
    assert all(record["route_status"] == "clean" for record in seed_records)
    assert all(record["reviewed_source_skill_ids"] for record in seed_records)
    assert all(record["reviewer"]["reviewed_by"] == "codex_registry_seed_review" for record in seed_records)
    assert all(record["provenance"]["batch_id"] == "batch_0002" for record in seed_records)
    assert all(record["provenance"]["draft_response_artifact_kind"] == "human_review_response_draft" for record in seed_records)
    assert all(record["allowed_use_cases"]["source_backed_examples"] is True for record in seed_records)


def test_ambiguous_and_blocked_draft_responses_are_not_converted() -> None:
    payload = _load_registry()
    registry_subpart_ids = {record["subpart_id"] for record in payload["records"]}

    assert UNPROMOTED_DRAFT_SUBPART_IDS.isdisjoint(registry_subpart_ids)
    assert BATCH_0003_NOT_PROMOTED_SUBPART_IDS.isdisjoint(registry_subpart_ids)


def test_clean_seed_records_keep_unreviewed_mark_events_generation_blocked() -> None:
    payload = _load_registry()
    seed_records = [
        record
        for record in payload["records"]
        if record["subpart_id"] in SEED_SUBPART_IDS | BATCH_0002_SEED_SUBPART_IDS
    ]

    assert all(record["allowed_use_cases"]["candidate_generation"] is False for record in seed_records)
    assert all(record["evidence_basis"]["candidate_generation_reviewed"] is False for record in seed_records)
    assert all(
        mark_ref["review_status"] == "advisory"
        for record in seed_records
        for mark_ref in record["mark_event_refs"]
    )


def test_batch_0003_mark_event_probes_do_not_change_content_lab_generation_readiness() -> None:
    conclusions = json.loads(Path("reports/manual_review_batch_0003_conclusions.v1.json").read_text(encoding="utf-8"))
    content_lab = json.loads(Path("output/asterion/exports/latest/asterion_content_lab_candidates_v1.json").read_text(encoding="utf-8"))
    ready_count = sum(
        1
        for candidate in content_lab["candidates"]
        if (candidate.get("generation_gate") or {}).get("blocked") is False
        or (candidate.get("generation_gate") or {}).get("status") in {"ready", "generation_ready", "generation-ready"}
    )

    assert conclusions["outcome_counts"]["approved_mark_event_count"] == 0
    assert conclusions["outcome_counts"]["content_lab_generation_ready_after"] == 0
    assert ready_count == 0


def test_clean_seed_records_require_verified_asset_refs_and_provenance() -> None:
    payload = _load_registry()
    seed_records = [
        record
        for record in payload["records"]
        if record["subpart_id"] in SEED_SUBPART_IDS | BATCH_0002_SEED_SUBPART_IDS
    ]

    assert all(
        ref["verified"] is True and ref.get("asset_id")
        for record in seed_records
        for ref in record["source_question_asset_refs"]
    )
    assert all(
        ref["verified"] is True and ref.get("asset_id")
        for record in seed_records
        for ref in record["source_mark_scheme_asset_refs"]
    )
    assert all(
        record["provenance"]["source_draft_response_id"].startswith(("batch_0001:", "batch_0002:"))
        for record in seed_records
    )


def test_duplicate_evidence_id_fails() -> None:
    payload = _payload(_non_clean_record(), _non_clean_record(question_id="32spring21_q02"))

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("duplicate_evidence_id" in error for error in errors)


def test_invalid_route_status_fails() -> None:
    payload = _payload(_non_clean_record(route_status="promoted"))

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("invalid_route_status" in error for error in errors)


def test_clean_without_question_asset_fails() -> None:
    record = _clean_record()
    record["source_question_asset_refs"] = []
    payload = _payload(record)

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("clean_without_question_asset_refs" in error for error in errors)


def test_clean_without_mark_scheme_asset_fails() -> None:
    record = _clean_record()
    record["source_mark_scheme_asset_refs"] = []
    payload = _payload(record)

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("clean_without_mark_scheme_asset_refs" in error for error in errors)


def test_clean_without_reviewed_p3_skill_fails() -> None:
    record = _clean_record()
    record["reviewed_source_skill_ids"] = []
    payload = _payload(record)

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("clean_without_reviewed_p3_skill" in error for error in errors)


def test_mastery_true_on_non_clean_route_fails() -> None:
    record = _non_clean_record(route_status="thin")
    record["allowed_use_cases"]["mastery"] = True
    payload = _payload(record)

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("mastery_true_requires_clean_route" in error for error in errors)


def test_source_backed_examples_true_on_non_clean_route_fails() -> None:
    record = _non_clean_record(route_status="review_needed")
    record["allowed_use_cases"]["source_backed_examples"] = True
    payload = _payload(record)

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("source_backed_examples_true_requires_clean_route" in error for error in errors)


def test_blocked_or_review_needed_without_blocker_or_evidence_explanation_fails() -> None:
    record = _non_clean_record(route_status="blocked")
    record["blockers"] = []
    record["evidence_basis"] = {}
    payload = _payload(record)

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("blocked_without_blockers_or_evidence_basis" in error for error in errors)


def test_p1_support_only_skill_cannot_be_mastery_true() -> None:
    record = _clean_record()
    record["reviewed_source_skill_ids"] = ["9709_p1_quadratics_discriminant_intersections"]
    record["allowed_use_cases"]["mastery"] = True
    payload = _payload(record)

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("p1_or_support_skill_cannot_be_mastery" in error for error in errors)


def test_advisory_only_mark_events_cannot_make_record_clean() -> None:
    record = _clean_record()
    record["evidence_basis"]["basis_type"] = "advisory_mark_events_only"
    payload = _payload(record)

    errors, _ = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert any("advisory_only_evidence_cannot_be_clean" in error for error in errors)


def test_clean_with_unreviewed_mark_events_warns_and_generation_stays_blocked() -> None:
    record = _clean_record()
    payload = _payload(record)

    errors, warnings = validate_reviewed_decisions_payload(payload, p3_skill_ids=P3_SKILL_IDS)

    assert errors == []
    assert any("clean_with_unreviewed_mark_event_refs" in warning for warning in warnings)

    record["allowed_use_cases"]["candidate_generation"] = True
    errors, _ = validate_reviewed_decisions_payload(_payload(record), p3_skill_ids=P3_SKILL_IDS)

    assert any("candidate_generation_true_without_reviewed_generation_basis" in error for error in errors)


def test_clean_with_unverified_asset_refs_fails() -> None:
    record = _clean_record()
    record["source_question_asset_refs"][0]["verified"] = False
    record["source_mark_scheme_asset_refs"][0]["verified"] = False

    errors, warnings = validate_reviewed_decisions_payload(_payload(record), p3_skill_ids=P3_SKILL_IDS)

    assert warnings == ["record:0:fixture-evidence-1:32spring21_q01:clean_with_unreviewed_mark_event_refs:32spring21_q01_me0001"]
    assert any("clean_with_unverified_question_asset_refs" in error for error in errors)
    assert any("clean_with_unverified_mark_scheme_asset_refs" in error for error in errors)


def test_validator_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_p3_exact_skill_reviewed_decisions.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _payload(*records: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "exam_bank.p3_exact_skill.reviewed_decisions",
        "schema_version": 1,
        "artifact_kind": "manual_reviewed_decision_input",
        "generated_at": "2026-05-23T00:00:00Z",
        "record_count": len(records),
        "records": list(records),
    }


def _load_registry() -> dict[str, object]:
    return json.loads(Path("data/review/p3_exact_skill_reviewed_decisions.v1.json").read_text(encoding="utf-8"))


def _clean_record() -> dict[str, object]:
    record = _non_clean_record(route_status="clean")
    record["reviewed_region"] = {
        "component": "p3",
        "section": "3.2 Logarithmic and exponential functions",
        "skill_id": "9709_p3_3_2_log_exponential_equations",
        "asterion_region_id": "asterion:9709:p3:3.2:9709-p3-3-2-log-exponential-equations",
    }
    record["evidence_basis"] = {
        "basis_type": "canonical_images_reviewed",
        "review_notes": "Human reviewer verified the canonical question and mark-scheme images.",
        "candidate_generation_reviewed": False,
    }
    record["blockers"] = []
    record["reviewer"] = {
        "review_status": "approved",
        "reviewed_by": "fixture reviewer",
        "reviewed_at": "2026-05-23T00:00:00Z",
    }
    record["provenance"] = {
        "decision_source": "unit test",
        "timestamp": "2026-05-23T00:00:00Z",
    }
    return record


def _non_clean_record(
    *,
    question_id: str = "32spring21_q01",
    route_status: str = "review_needed",
) -> dict[str, object]:
    suffix = question_id
    return json.loads(
        json.dumps(
            {
                "evidence_id": "fixture-evidence-1",
                "question_id": question_id,
                "part_id": "whole",
                "subpart_id": f"{suffix}_whole",
                "paper": "32spring21",
                "session": "March",
                "variant": "32",
                "reviewed_source_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
                "reviewed_region": None,
                "route_status": route_status,
                "source_question_asset_refs": [
                    {
                        "path": "p3/32spring21/questions/q01.png",
                        "sha256": "sha",
                        "verified": True,
                    }
                ],
                "source_mark_scheme_asset_refs": [
                    {
                        "path": "p3/32spring21/mark_scheme/q01.png",
                        "sha256": "sha",
                        "verified": True,
                    }
                ],
                "mark_event_refs": [
                    {
                        "event_id": "32spring21_q01_me0001",
                        "source": "output/json/question_bank.mark_events.v1.json",
                        "review_status": "advisory",
                    }
                ],
                "evidence_basis": {
                    "basis_type": "review_seed",
                    "review_notes": "Not yet clean.",
                },
                "blockers": ["not_reviewed"],
                "allowed_use_cases": {
                    "mastery": False,
                    "guardian": False,
                    "export": False,
                    "source_backed_examples": False,
                    "candidate_generation": False,
                },
                "reviewer": {
                    "review_status": "review_needed",
                    "reviewed_by": "",
                    "reviewed_at": "",
                },
                "provenance": {
                    "decision_source": "unit test",
                    "timestamp": "2026-05-23T00:00:00Z",
                },
            }
        )
    )
