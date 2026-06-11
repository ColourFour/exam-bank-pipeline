from __future__ import annotations

import json
from pathlib import Path

from exam_bank.topic_routing_sample_refresh import (
    build_parser,
    build_topic_routing_sample_delta,
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


def _route(
    *,
    review_required: bool = False,
    confidence: str = "high",
    error: bool = False,
    hash_value: str | None = None,
    repaired: bool = False,
) -> dict[str, object]:
    route: dict[str, object] = {
        "primary_topic_id": None if error else "topic_a",
        "topic_distribution": [] if error else [{"topic_id": "topic_a", "fit_percent": 100}],
        "confidence": "low" if error else confidence,
        "review_required": True if error else review_required,
        "review_reasons": ["schema_validation_error"] if error else [],
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
