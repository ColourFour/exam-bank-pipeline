from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.topic_routing_audit import build_topic_routing_baseline_audit, write_json


def test_topic_routing_baseline_audit_counts_and_failure_analysis(tmp_path: Path) -> None:
    question_bank = tmp_path / "question_bank.json"
    topic_routing = tmp_path / "topic_routing.json"
    mark_events = tmp_path / "mark_events.json"
    missing_catalog = tmp_path / "missing_catalog.json"
    missing_runtime = tmp_path / "missing_runtime.json"
    missing_auto_grade = tmp_path / "missing_auto_grade.json"
    _write_json(
        question_bank,
        {
            "questions": [
                {
                    "question_id": "q1",
                    "text_only_status": "ready",
                    "question_crop_confidence": "high",
                    "visual_required": False,
                    "question_text": "Differentiate x^2.",
                    "ocr_text": "Differentiate x^2.",
                    "mark_scheme_text": "2x",
                    "question_image_path": "p1/q1.png",
                },
                {
                    "question_id": "q2",
                    "text_only_status": "review",
                    "question_crop_confidence": "low",
                    "visual_required": True,
                    "question_text": "",
                    "ocr_text": "",
                    "mark_scheme_text": "diagram evidence",
                    "question_image_paths": ["p3/q2.png"],
                },
                {
                    "question_id": "q3",
                    "text_only_status": "fail",
                    "question_crop_confidence": "low",
                    "visual_required": True,
                    "question_text": "",
                    "ocr_text": "",
                    "mark_scheme_text": "",
                },
            ]
        },
    )
    _write_json(
        topic_routing,
        {
            "record_count": 3,
            "records": {
                "q1": {
                    "primary_topic_id": "topic_a",
                    "topic_distribution": [{"topic_id": "topic_a", "fit_percent": 100}],
                    "confidence": "high",
                    "review_required": False,
                    "review_reasons": [],
                    "evidence_used": ["question_text"],
                    "course_id": "p1",
                    "component_name": "Pure Mathematics 1",
                    "evidence_used_repaired": True,
                    "evidence_used_original": ["ocr_text", "question_text"],
                    "evidence_used_dropped": ["ocr_text"],
                },
                "q2": {
                    "primary_topic_id": None,
                    "topic_distribution": [],
                    "confidence": "low",
                    "review_required": True,
                    "review_reasons": ["Visual required but no image provided"],
                    "evidence_used": ["mark_scheme_text"],
                },
                "q3": {
                    "primary_topic_id": None,
                    "topic_distribution": [],
                    "confidence": "low",
                    "review_required": True,
                    "review_reasons": ["schema_validation_error"],
                    "evidence_used": ["ocr_text"],
                    "error": {
                        "type": "schema_validation_error",
                        "message": "records[q3]: evidence_used contains evidence that was not supplied: ['ocr_text'].",
                    },
                    "routing_source": "deepseek_topic_routing_error",
                },
            },
        },
    )
    _write_json(
        mark_events,
        {
            "records": [
                {
                    "question_id": "q1",
                    "extraction_status": "ok",
                    "safe_for_marking_use": True,
                    "safe_for_advisory_use": True,
                },
                {
                    "question_id": "q2",
                    "extraction_status": "review",
                    "safe_for_marking_use": False,
                    "safe_for_advisory_use": True,
                },
            ]
        },
    )

    audit = build_topic_routing_baseline_audit(
        question_bank_path=question_bank,
        topic_routing_path=topic_routing,
        mark_events_path=mark_events,
        asterion_catalog_path=missing_catalog,
        asterion_runtime_path=missing_runtime,
        auto_grade_eligibility_path=missing_auto_grade,
        generated_at="2026-06-11T00:00:00+00:00",
    )

    assert audit["question_bank_baseline"]["text_only_status_counts"] == {"fail": 1, "ready": 1, "review": 1}
    assert audit["question_bank_baseline"]["question_crop_confidence_counts"] == {"high": 1, "low": 2}
    assert audit["question_bank_baseline"]["visual_required_true_count"] == 2
    assert audit["question_bank_baseline"]["usable_question_text_count"] == 1
    assert audit["topic_routing_sidecar_baseline"]["failed_count"] == 1
    assert audit["topic_routing_sidecar_baseline"]["review_required_count"] == 2
    assert audit["topic_routing_sidecar_baseline"]["strict_filter_candidate_count"] == 1
    assert audit["topic_routing_sidecar_baseline"]["missing_course_metadata_count"] == 2
    assert audit["topic_routing_sidecar_baseline"]["missing_evidence_packet_hash_count"] == 3
    assert audit["topic_routing_sidecar_baseline"]["stale_missing_metadata_or_packet_hash_count"] == 3
    assert audit["topic_routing_sidecar_baseline"]["evidence_used_repaired_count"] == 1
    assert audit["topic_routing_sidecar_baseline"]["top_dropped_evidence_fields"] == {"ocr_text": 1}
    assert audit["topic_routing_sidecar_baseline"]["repaired_strict_filter_candidate_count"] == 1
    assert audit["topic_routing_sidecar_baseline"]["repaired_review_required_or_error_count"] == 0
    assert audit["failure_analysis"]["unsupported_evidence_used_failure_count"] == 1
    assert audit["failure_analysis"]["top_unsupported_evidence_fields"] == {"ocr_text": 1}
    assert audit["review_required_analysis"]["visual_required_overlap_count"] == 2
    assert audit["review_required_analysis"]["weak_text_or_crop_readiness_overlap_count"] == 2
    assert audit["review_required_analysis"]["normalized_bucket_counts"] == {
        "schema_validation_error": 1,
        "visual_required_without_sufficient_text_evidence": 1,
    }
    assert audit["mark_events_snapshot"]["record_count"] == 2
    assert audit["mark_events_snapshot"]["extraction_status_counts"] == {"ok": 1, "review": 1}
    assert audit["mark_events_snapshot"]["safe_for_marking_use_true_count"] == 1
    assert audit["auto_grade_readiness_snapshot"]["file_found"] is False
    assert audit["inputs"]["mark_events"]["exists"] is True


def test_topic_routing_baseline_audit_handles_downstream_snapshots(tmp_path: Path) -> None:
    question_bank = tmp_path / "question_bank.json"
    topic_routing = tmp_path / "topic_routing.json"
    catalog = tmp_path / "catalog.json"
    runtime = tmp_path / "runtime.json"
    auto_grade = tmp_path / "eligible_items.json"
    _write_json(question_bank, {"questions": [{"question_id": "q1", "text_only_status": "ready"}]})
    _write_json(
        topic_routing,
        {
            "records": {
                "q1": {
                    "primary_topic_id": "topic_a",
                    "topic_distribution": [{"topic_id": "topic_a", "fit_percent": 100}],
                    "confidence": "medium",
                    "review_required": False,
                    "review_reasons": [],
                    "evidence_used": ["question_text"],
                    "course_id": "p3",
                    "component_name": "Pure Mathematics 3",
                    "evidence_packet_hash": "a" * 64,
                }
            }
        },
    )
    _write_json(
        catalog,
        {
            "questions": [
                {"question_id": "q1", "topic_route": {"filter_ok": True, "review_required": False}},
                {
                    "question_id": "q2",
                    "topic_route": {"filter_ok": False, "review_required": True},
                    "quality_gate": {"reason_codes": ["content_lab_blocked_topic_uncertain"]},
                },
            ]
        },
    )
    _write_json(runtime, {"questions": [{"question_id": "q1", "paper_family": "p3"}]})
    _write_json(
        auto_grade,
        {
            "items": [
                {"question_id": "q1", "eligibility_status": "blocked", "block_reasons": ["rubric_not_reviewed"]},
                {"question_id": "q2", "eligibility_status": "eligible", "block_reasons": []},
            ]
        },
    )

    audit = build_topic_routing_baseline_audit(
        question_bank_path=question_bank,
        topic_routing_path=topic_routing,
        mark_events_path=tmp_path / "mark_events.json",
        asterion_catalog_path=catalog,
        asterion_runtime_path=runtime,
        auto_grade_eligibility_path=auto_grade,
        generated_at="2026-06-11T00:00:00+00:00",
    )

    assert audit["asterion_readiness_snapshot"]["all_course_catalog_count"] == 2
    assert audit["asterion_readiness_snapshot"]["student_runtime_count"] == 1
    assert audit["asterion_readiness_snapshot"]["p3_runtime_count"] == 1
    assert audit["asterion_readiness_snapshot"]["topic_route_filter_ok_true_count"] == 1
    assert audit["asterion_readiness_snapshot"]["route_review_error_or_stale_blocked_count"] == 1
    assert audit["topic_routing_sidecar_baseline"]["missing_evidence_packet_hash_count"] == 0
    assert audit["auto_grade_readiness_snapshot"]["status_counts"] == {"blocked": 1, "eligible": 1}
    assert audit["auto_grade_readiness_snapshot"]["top_blocker_counts"] == {"rubric_not_reviewed": 1}


def test_topic_routing_baseline_json_output_is_stable(tmp_path: Path) -> None:
    question_bank = tmp_path / "question_bank.json"
    topic_routing = tmp_path / "topic_routing.json"
    output_a = tmp_path / "a.json"
    output_b = tmp_path / "b.json"
    _write_json(question_bank, {"questions": [{"question_id": "q1"}]})
    _write_json(topic_routing, {"records": {"q1": {"confidence": "low", "review_required": True}}})

    kwargs = {
        "question_bank_path": question_bank,
        "topic_routing_path": topic_routing,
        "mark_events_path": tmp_path / "mark_events.json",
        "asterion_catalog_path": tmp_path / "catalog.json",
        "asterion_runtime_path": tmp_path / "runtime.json",
        "auto_grade_eligibility_path": tmp_path / "auto_grade.json",
        "generated_at": "2026-06-11T00:00:00+00:00",
    }
    write_json(output_a, build_topic_routing_baseline_audit(**kwargs))
    write_json(output_b, build_topic_routing_baseline_audit(**kwargs))

    assert output_a.read_text(encoding="utf-8") == output_b.read_text(encoding="utf-8")


def test_topic_routing_baseline_audit_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "exam_bank.topic_routing_audit", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
