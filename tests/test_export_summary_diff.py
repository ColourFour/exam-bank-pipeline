import json
from pathlib import Path

from exam_bank.cli import main as cli_main
from exam_bank.export_summary_diff import compare_export_summaries, render_export_summary_diff, summarize_export


def test_question_bank_summary_diff_reports_readiness_images_topics_and_reasons(tmp_path: Path) -> None:
    before = tmp_path / "before_question_bank.json"
    after = tmp_path / "after_question_bank.json"
    _write_json(
        before,
        {
            "schema_name": "exam_bank.question_bank",
            "schema_version": 2,
            "record_count": 2,
            "run_manifest": {"generated_at": "2026-05-13T00:00:00+00:00", "run_id": "run-before"},
            "questions": [
                _question_record("q1", text_only_status="ready", visual_curation_status="ready", topic_confidence="high"),
                _question_record(
                    "q2",
                    text_only_status="fail",
                    visual_curation_status="review",
                    topic_confidence="low",
                    mark_scheme_image_paths=[],
                    visual_reason_flags=["diagram_required"],
                    mapping_failure_reason="missing_mark_scheme",
                ),
            ],
        },
    )
    _write_json(
        after,
        {
            "schema_name": "exam_bank.question_bank",
            "schema_version": 2,
            "record_count": 3,
            "run_manifest": {"generated_at": "2026-05-14T00:00:00+00:00", "run_id": "run-after"},
            "questions": [
                _question_record("q1", text_only_status="ready", visual_curation_status="ready", topic_confidence="high"),
                _question_record("q2", text_only_status="ready", visual_curation_status="ready", topic_confidence="medium"),
                _question_record(
                    "q3",
                    text_only_status="fail",
                    visual_curation_status="review",
                    topic_confidence="low",
                    question_image_paths=[],
                    ocr_rejected_reasons=["low_confidence"],
                ),
            ],
        },
    )

    report = compare_export_summaries(before, after)
    text = render_export_summary_diff(report)

    assert report["before"]["record_count"] == 2
    assert report["after"]["record_count"] == 3
    assert report["deltas"]["readiness_counts"]["text_only_status.ready"] == {"before": 1, "after": 2}
    assert report["deltas"]["missing_image_counts"]["missing_mark_scheme_image_path"] == {"before": 1, "after": 0}
    assert report["deltas"]["topic_safety_counts"]["topic_confidence.medium"] == {"before": 0, "after": 1}
    assert report["deltas"]["reason_code_counts"]["visual_reason_flags.diagram_required"] == {"before": 1, "after": 0}
    assert "Record count: 2 -> 3 (+1)" in text
    assert "Run ID: run-before -> run-after" in text


def test_asterion_summary_diff_reports_role_gates_and_quality_reasons(tmp_path: Path) -> None:
    before = tmp_path / "before_asterion.json"
    after = tmp_path / "after_asterion.json"
    _write_json(
        before,
        {
            "schema_name": "asterion.question_bank",
            "schema_version": 1,
            "record_count": 1,
            "questions": [
                _asterion_record(
                    "q1",
                    role_status="block",
                    text_only_display_allowed=False,
                    reason_codes=["missing_mark_scheme_image_path", "text_only_blocked_status_fail"],
                )
            ],
        },
    )
    _write_json(
        after,
        {
            "schema_name": "asterion.question_bank",
            "schema_version": 1,
            "record_count": 1,
            "questions": [
                _asterion_record(
                    "q1",
                    role_status="allow",
                    text_only_display_allowed=True,
                    reason_codes=[],
                )
            ],
        },
    )

    report = compare_export_summaries(before, after)

    assert report["deltas"]["role_gate_counts"]["usage_roles.canonical_practice.allow"] == {"before": 0, "after": 1}
    assert report["deltas"]["role_gate_counts"]["usage_roles.canonical_practice.block"] == {"before": 1, "after": 0}
    assert report["deltas"]["readiness_counts"]["quality_gate.text_only_display_allowed.true"] == {"before": 0, "after": 1}
    assert report["deltas"]["missing_image_counts"]["missing_mark_scheme_image_path"] == {"before": 1, "after": 0}
    assert report["deltas"]["reason_code_counts"]["quality_gate.text_only_blocked_status_fail"] == {"before": 1, "after": 0}


def test_topic_routing_summary_diff_reports_safety_metadata_and_review_reasons(tmp_path: Path) -> None:
    before = tmp_path / "before_topics.json"
    after = tmp_path / "after_topics.json"
    _write_json(before, _topic_sidecar(failed_records=1, safe=False, review_required=True))
    _write_json(after, _topic_sidecar(failed_records=0, safe=True, review_required=False))

    report = compare_export_summaries(before, after)

    assert report["deltas"]["topic_safety_counts"]["failed_records"] == {"before": 1, "after": 0}
    assert report["deltas"]["topic_safety_counts"]["safe_for_strict_filters.false"] == {"before": 1, "after": 0}
    assert report["deltas"]["topic_safety_counts"]["safe_for_strict_filters.true"] == {"before": 0, "after": 1}
    assert report["deltas"]["reason_code_counts"]["review_reason.weak_or_missing_text_evidence"] == {"before": 1, "after": 0}
    assert report["deltas"]["reason_code_counts"]["error.schema_validation_error"] == {"before": 1, "after": 0}


def test_export_summary_diff_cli_returns_zero_for_normal_count_changes(tmp_path: Path, capsys) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_json(
        before,
        {
            "schema_name": "exam_bank.question_bank",
            "schema_version": 2,
            "record_count": 1,
            "questions": [_question_record("q1", text_only_status="fail")],
        },
    )
    _write_json(
        after,
        {
            "schema_name": "exam_bank.question_bank",
            "schema_version": 2,
            "record_count": 2,
            "questions": [_question_record("q1", text_only_status="ready"), _question_record("q2", text_only_status="ready")],
        },
    )

    assert cli_main(["export-summary-diff", str(before), str(after)]) == 0
    output = capsys.readouterr().out

    assert "Export summary diff" in output
    assert "Record count: 1 -> 2 (+1)" in output


def test_export_summary_diff_cli_returns_nonzero_for_invalid_comparisons(tmp_path: Path, capsys) -> None:
    question_bank = tmp_path / "question_bank.json"
    asterion = tmp_path / "asterion.json"
    _write_json(
        question_bank,
        {
            "schema_name": "exam_bank.question_bank",
            "schema_version": 2,
            "record_count": 1,
            "questions": [_question_record("q1")],
        },
    )
    _write_json(
        asterion,
        {
            "schema_name": "asterion.question_bank",
            "schema_version": 1,
            "record_count": 1,
            "questions": [_asterion_record("q1")],
        },
    )

    assert cli_main(["export-summary-diff", str(question_bank), str(asterion)]) == 2
    assert "Cannot compare different schema_name values" in capsys.readouterr().out


def test_summarize_content_lab_candidates_reports_generation_and_role_statuses(tmp_path: Path) -> None:
    path = tmp_path / "content_lab.json"
    _write_json(
        path,
        {
            "schema_name": "asterion.content_lab_candidates",
            "schema_version": 1,
            "record_count": 1,
            "candidates": [
                {
                    "review_status": "blocked_until_reviewed",
                    "role_statuses": {"warmup_generator_source": "block_until_reviewed"},
                    "generation_gate": {
                        "status": "blocked_until_reviewed",
                        "blockers": ["missing_source_skill_ids"],
                    },
                }
            ],
        },
    )

    summary = summarize_export(path)

    assert summary["readiness_counts"]["review_status.blocked_until_reviewed"] == 1
    assert summary["readiness_counts"]["generation_gate.status.blocked_until_reviewed"] == 1
    assert summary["role_gate_counts"]["role_statuses.warmup_generator_source.block_until_reviewed"] == 1
    assert summary["reason_code_counts"]["generation_gate.blocker.missing_source_skill_ids"] == 1


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _question_record(
    question_id: str,
    *,
    text_only_status: str = "ready",
    visual_curation_status: str = "ready",
    topic_confidence: str = "high",
    question_image_paths: list[str] | None = None,
    mark_scheme_image_paths: list[str] | None = None,
    visual_reason_flags: list[str] | None = None,
    mapping_failure_reason: str = "",
    ocr_rejected_reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_image_paths": ["questions/q01.png"] if question_image_paths is None else question_image_paths,
        "mark_scheme_image_paths": ["mark_scheme/q01.png"] if mark_scheme_image_paths is None else mark_scheme_image_paths,
        "visual_required": False,
        "notes": {
            "validation_status": "pass",
            "mapping_status": "pass" if not mapping_failure_reason else "fail",
            "scope_quality_status": "clean",
            "text_fidelity_status": "clean",
            "visual_curation_status": visual_curation_status,
            "text_only_status": text_only_status,
            "question_text_role": "readable_text",
            "question_text_trust": "high",
            "topic_confidence": topic_confidence,
            "topic_uncertain": topic_confidence == "low",
            "topic_trust_status": "normal",
            "visual_reason_flags": visual_reason_flags or [],
            "mapping_failure_reason": mapping_failure_reason,
            "ocr_rejected_reasons": ocr_rejected_reasons or [],
        },
    }


def _asterion_record(
    question_id: str,
    *,
    role_status: str = "allow",
    text_only_display_allowed: bool = True,
    reason_codes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "quality_gate": {
            "canonical_assets_ok": True,
            "text_only_display_allowed": text_only_display_allowed,
            "reason_codes": reason_codes or [],
        },
        "usage_roles": {
            "canonical_practice": role_status,
            "quick_check_source": role_status,
        },
        "subparts": [
            {
                "subpart_id": f"{question_id}_whole",
                "review_status": "machine_candidate",
                "mark_events": [{"review_status": "machine_candidate"}],
            }
        ],
    }


def _topic_sidecar(*, failed_records: int, safe: bool, review_required: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "primary_topic_id": None if review_required else "9709_p1_topic_quadratics",
        "topic_distribution": [] if review_required else [{"topic_id": "9709_p1_topic_quadratics", "fit_percent": 100}],
        "confidence": "low" if review_required else "high",
        "review_required": review_required,
        "review_reasons": ["weak_or_missing_text_evidence"] if review_required else [],
    }
    if failed_records:
        record["error"] = {"type": "schema_validation_error", "message": "bad response"}
    return {
        "schema_name": "exam_bank.topic_routing_sidecar",
        "schema_version": 1,
        "generated_at": "2026-05-14T00:00:00+00:00",
        "record_count": 1,
        "records": {"q1": record},
        "metadata": {
            "run_summary": {
                "successful_records": 1 - failed_records,
                "failed_records": failed_records,
                "review_required_records": 1 if review_required else 0,
                "strict_filter_records": 0 if review_required else 1,
                "failures_by_reason": {"schema_validation_error": failed_records} if failed_records else {},
                "safe_for_strict_filters": safe,
            }
        },
    }
