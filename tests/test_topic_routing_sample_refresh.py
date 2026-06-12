from __future__ import annotations

import json
from pathlib import Path

from exam_bank.topic_routing_sample_refresh import (
    build_parser,
    build_topic_routing_sample_delta,
    build_topic_routing_sample_triage,
    build_visual_required_evidence_audit,
    render_delta_markdown,
    select_topic_routing_sample,
)


def test_sample_selection_includes_failed_review_and_strict_ids(tmp_path: Path) -> None:
    sidecar = tmp_path / "topic_sidecar.json"
    _write_json(
        sidecar,
        {
            "records": {
                "failed": _route(error=True),
                "review": _route(review_required=True, confidence="low"),
                "strict": _route(),
                "strict2": _route(),
            }
        },
    )

    report = select_topic_routing_sample(sidecar_path=sidecar, sample_size=3)

    assert report["sample_ids"] == ["failed", "review", "strict"]
    assert report["summary"]["failed_ids_included"] == 1
    assert report["summary"]["review_required_ids_included"] == 1
    assert report["summary"]["strict_filter_ids_included"] == 1


def test_sample_refresh_defaults_do_not_target_production_sidecar() -> None:
    parser = build_parser()
    select_args = parser.parse_args(["select"])
    delta_args = parser.parse_args(["delta", "--after-sidecar", "/tmp/sample.json"])

    assert str(select_args.ids_out).startswith("/tmp/")
    assert str(select_args.json_out).startswith("/tmp/")
    assert str(delta_args.json_out).startswith("/tmp/")
    assert str(delta_args.markdown_out).startswith("/tmp/")


def test_sample_delta_counts_hashes_repairs_and_batch_salvage(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_json(
        before,
        {
            "records": {
                "q1": _route(error=True),
                "q2": _route(review_required=True, confidence="low"),
                "q3": _route(),
            }
        },
    )
    _write_json(
        after,
        {
            "records": {
                "q1": _route(hash_value="a" * 64, repaired=True),
                "q2": _route(review_required=True, confidence="low", hash_value="b" * 64),
                "q3": _route(hash_value="c" * 64),
            },
            "metadata": {
                "run_manifest": {
                    "batches": [
                        {
                            "valid_records": 2,
                            "invalid_records": 1,
                            "missing_records": 0,
                            "duplicate_records": 0,
                            "unknown_returned_records": 1,
                            "batch_salvaged": True,
                        }
                    ]
                }
            },
        },
    )

    report = build_topic_routing_sample_delta(
        before_sidecar_path=before,
        after_sidecar_path=after,
        sample_ids=["q1", "q2", "q3"],
    )

    assert report["summary"]["failed_before"] == 1
    assert report["summary"]["failed_after"] == 0
    assert report["summary"]["missing_evidence_packet_hash_before"] == 3
    assert report["summary"]["missing_evidence_packet_hash_after"] == 0
    assert report["summary"]["evidence_used_repaired_after"] == 1
    assert report["deltas"]["failed_count"] == -1
    assert report["deltas"]["missing_evidence_packet_hash_count"] == -3
    assert report["after"]["top_dropped_evidence_fields"] == {"ocr_text": 1}
    assert report["summary"]["valid_sibling_records_preserved_after_failure"] is True
    assert report["summary"]["unknown_returned_records_after"] == 1


def test_sample_delta_handles_missing_after_sidecar(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    _write_json(before, {"records": {"q1": _route()}})

    report = build_topic_routing_sample_delta(
        before_sidecar_path=before,
        after_sidecar_path=tmp_path / "missing.json",
        sample_ids=["q1"],
    )

    assert report["summary"]["after_file_found"] is False
    assert report["summary"]["after_status"] == "unavailable"
    assert report["summary"]["behavioral_conclusion"] == "not_computed_provider_backed_refresh_did_not_run"
    assert report["summary"]["failed_before"] == 0
    assert report["summary"]["failed_after"] is None
    assert report["summary"]["evidence_used_repaired_after"] is None
    assert report["summary"]["valid_sibling_records_preserved_after_failure"] is None
    assert report["after"]["record_count"] is None
    assert report["deltas"]["failed_count"] == "not_computed"


def test_missing_after_sidecar_markdown_makes_no_improvement_claim(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    _write_json(before, {"records": {"q1": _route(error=True)}})

    report = build_topic_routing_sample_delta(
        before_sidecar_path=before,
        after_sidecar_path=tmp_path / "missing.json",
        sample_ids=["q1"],
    )
    markdown = render_delta_markdown(report)

    assert "Provider-backed refresh did not run" in markdown
    assert "no behavioral conclusion or improvement claim can be made" in markdown
    assert "`unavailable`" in markdown


def test_sample_triage_groups_review_required_records_by_bucket_and_joins_qbank(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar.json"
    qbank = tmp_path / "question_bank.json"
    _write_json(
        sidecar,
        {
            "records": {
                "q1": _route(
                    review_required=True,
                    confidence="low",
                    hash_value="a" * 64,
                    review_reasons=["Visual-dependent question without image; text insufficient for full routing."],
                ),
                "q2": _route(
                    review_required=True,
                    confidence="low",
                    hash_value="b" * 64,
                    review_reasons=["Multiple topics involved"],
                ),
                "q3": _route(hash_value="c" * 64, repaired=True),
            }
        },
    )
    _write_json(
        qbank,
        {
            "questions": [
                {
                    "question_id": "q1",
                    "text_only_status": "review",
                    "visual_required": True,
                    "question_crop_confidence": "low",
                    "question_text": "The diagram shows a region.",
                    "mark_scheme_text": "Use integration.",
                    "paper_family": "p3",
                },
                {
                    "question_id": "q2",
                    "text_only_status": "ready",
                    "visual_required": False,
                    "notes": {"question_crop_confidence": "high"},
                    "question_text": "Find the equation.",
                    "ocr_text": "Find the equation.",
                    "mark_scheme_text": "Differentiate then integrate.",
                    "paper_family": "p1",
                },
            ]
        },
    )

    report = build_topic_routing_sample_triage(
        sidecar_path=sidecar,
        sample_ids=["q1", "q2", "q3"],
        question_bank_path=qbank,
    )

    assert report["summary"]["review_required_count"] == 2
    assert report["summary"]["evidence_used_repaired_count"] == 1
    assert report["summary"]["missing_evidence_packet_hash_count"] == 0
    assert report["summary"]["review_required_bucket_counts"] == {
        "ambiguous_multi_topic_fit": 1,
        "visual_required_without_sufficient_text_evidence": 1,
    }
    assert report["review_required_records"][0]["question_id"] == "q1"
    assert report["review_required_records"][0]["available_evidence_fields"] == ["question_text", "mark_scheme_text"]
    assert report["review_required_records"][1]["question_crop_confidence"] == "high"


def test_sample_triage_handles_missing_qbank_fields_gracefully(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar.json"
    qbank = tmp_path / "question_bank.json"
    _write_json(
        sidecar,
        {
            "records": {
                "q1": _route(
                    review_required=True,
                    confidence="low",
                    review_reasons=["Unable to determine topic from context"],
                )
            }
        },
    )
    _write_json(qbank, {"questions": [{"question_id": "q1"}]})

    report = build_topic_routing_sample_triage(
        sidecar_path=sidecar,
        sample_ids=["q1"],
        question_bank_path=qbank,
    )

    row = report["review_required_records"][0]
    assert row["text_only_status"] is None
    assert row["question_crop_confidence"] is None
    assert row["available_evidence_fields"] == []
    assert row["inferred_bucket"] == "insufficient_question_context"


def test_visual_evidence_audit_counts_buckets_and_packet_gaps(tmp_path: Path) -> None:
    qbank = tmp_path / "question_bank.json"
    sidecar = tmp_path / "sidecar.json"
    _write_json(
        qbank,
        {
            "questions": [
                _qbank_record(
                    "q1",
                    text_only_status="review",
                    question_text="A diagram shows a curve.",
                    question_text_role="readable_text",
                    question_text_trust="high",
                    ocr_text="A diagram shows a curve.",
                    mark_scheme_text="Differentiate.",
                    question_crop_confidence="low",
                ),
                _qbank_record(
                    "q2",
                    text_only_status="ready",
                    question_text="Search hint only",
                    question_text_role="search_hint",
                    question_text_trust="medium",
                    ocr_text="",
                    mark_scheme_text="",
                    question_crop_confidence="high",
                ),
                _qbank_record(
                    "q3",
                    text_only_status="fail",
                    question_text="",
                    ocr_text="",
                    mark_scheme_text="",
                    question_crop_confidence="low",
                ),
                _qbank_record("q4", visual_required=False, text_only_status="ready"),
            ]
        },
    )
    _write_json(
        sidecar,
        {
            "records": {
                "q1": _route(review_required=True, confidence="low", review_reasons=["Visual required but no image provided"]),
                "q2": _route(hash_value="a" * 64),
                "q3": _route(review_required=True, confidence="low", review_reasons=["weak_or_missing_text_evidence"]),
            }
        },
    )

    report = build_visual_required_evidence_audit(
        question_bank_path=qbank,
        production_sidecar_path=sidecar,
        sample_sidecar_path=tmp_path / "missing_sample.json",
        triage_path=tmp_path / "missing_triage.json",
        taxonomy_root=Path("exam_bank_taxonomy/canonical"),
    )

    assert report["summary"]["total_question_bank_records"] == 4
    assert report["summary"]["visual_required_count"] == 3
    assert report["summary"]["visual_text_only_status_counts"] == {"fail": 1, "ready": 1, "review": 1}
    assert report["summary"]["visual_question_crop_confidence_counts"] == {"high": 1, "low": 2}
    assert report["packet_evidence_gap"]["evidence_exists_but_withheld_count"] == 1
    assert report["packet_evidence_gap"]["packet_no_meaningful_text_evidence_count"] == 1
    assert report["packet_evidence_gap"]["packet_only_mark_scheme_text_count"] == 0
    assert report["packet_evidence_gap"]["ocr_fallback_supplied_count"] == 1
    assert report["packet_evidence_gap"]["search_hint_fallback_supplied_count"] == 1
    assert report["review_required_overlap"]["production_sidecar"]["visual_required_count"] == 2
    assert report["review_required_overlap"]["sample_triage"]["review_required_count"] is None
    categories = {item["category"]: item["impact_count"] for item in report["candidate_fix_categories"]}
    assert categories["packet can include existing safe text currently withheld"] == 1
    assert categories["crop exists but OCR/text is missing"] == 1


def test_visual_evidence_audit_uses_optional_sample_and_triage_files(tmp_path: Path) -> None:
    qbank = tmp_path / "question_bank.json"
    sidecar = tmp_path / "sidecar.json"
    sample = tmp_path / "sample.json"
    triage = tmp_path / "triage.json"
    _write_json(
        qbank,
        {"questions": [_qbank_record("q1", text_only_status="ready", question_crop_confidence="high")]},
    )
    _write_json(sidecar, {"records": {"q1": _route(hash_value="a" * 64)}})
    _write_json(sample, {"records": {"q1": _route(review_required=True, confidence="low", review_reasons=["Visual-dependent question; no image provided."])}})
    _write_json(triage, {"review_required_records": [{"question_id": "q1"}]})

    report = build_visual_required_evidence_audit(
        question_bank_path=qbank,
        production_sidecar_path=sidecar,
        sample_sidecar_path=sample,
        triage_path=triage,
        taxonomy_root=Path("exam_bank_taxonomy/canonical"),
    )

    assert report["inputs"]["sample_sidecar"]["exists"] is True
    assert report["inputs"]["triage"]["exists"] is True
    assert report["review_required_overlap"]["sample_triage"]["review_required_count"] == 1
    assert report["representative_examples"][0]["current_route_review_required"] is True


def _route(
    *,
    review_required: bool = False,
    confidence: str = "high",
    error: bool = False,
    hash_value: str | None = None,
    repaired: bool = False,
    review_reasons: list[str] | None = None,
) -> dict[str, object]:
    route: dict[str, object] = {
        "primary_topic_id": None if error else "topic_a",
        "topic_distribution": [] if error else [{"topic_id": "topic_a", "fit_percent": 100}],
        "confidence": "low" if error else confidence,
        "review_required": True if error else review_required,
        "review_reasons": review_reasons if review_reasons is not None else (["schema_validation_error"] if error else []),
        "evidence_used": ["question_text"],
        "course_id": "p1",
        "component_name": "Pure Mathematics 1",
    }
    if error:
        route["error"] = {"type": "schema_validation_error", "message": "broken"}
    if hash_value:
        route["evidence_packet_hash"] = hash_value
    if repaired:
        route["evidence_used_repaired"] = True
        route["evidence_used_original"] = ["ocr_text", "question_text"]
        route["evidence_used_dropped"] = ["ocr_text"]
    return route


def _qbank_record(
    question_id: str,
    *,
    visual_required: bool = True,
    text_only_status: str = "ready",
    question_text: str = "A diagram shows a curve.",
    question_text_role: str = "readable_text",
    question_text_trust: str = "high",
    ocr_text: str = "OCR curve text",
    ocr_text_trust: str = "high",
    mark_scheme_text: str = "Use calculus.",
    question_crop_confidence: str = "high",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "12spring24",
        "paper_family": "p1",
        "question_number": "1",
        "visual_required": visual_required,
        "text_only_status": text_only_status,
        "question_text": question_text,
        "question_text_role": question_text_role,
        "question_text_trust": question_text_trust,
        "ocr_text": ocr_text,
        "ocr_text_trust": ocr_text_trust,
        "mark_scheme_text": mark_scheme_text,
        "question_image_path": f"p1/12spring24/questions/{question_id}.png",
        "mark_scheme_image_path": f"p1/12spring24/mark_scheme/{question_id}.png",
        "question_crop_confidence": question_crop_confidence,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
