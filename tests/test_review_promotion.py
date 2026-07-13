from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from exam_bank.review_promotion import (
    ReviewPromotionError,
    promote_review_artifact,
    validate_promoted_review_artifact,
)
from exam_bank.topic_packet_contracts import packet_projection_fingerprint, path_provenance


def test_promotes_review_artifact_with_provenance(tmp_path: Path) -> None:
    source = tmp_path / "run" / "decisions.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"schema_name": "fixture", "records": [{"id": "q1"}]}), encoding="utf-8")

    report = promote_review_artifact(
        source,
        "topic/decisions.v1.json",
        authority="human",
        source_run_id="review-run-1",
        reviewed_by="reviewer@example.invalid",
        reviewed_at="2026-07-13T00:00:00Z",
        canonical_root=tmp_path / "canonical",
    )

    target = Path(report["target"])
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["records"] == [{"id": "q1"}]
    assert payload["promotion"]["decision_authority"] == "human"
    assert payload["promotion"]["source_run_id"] == "review-run-1"
    assert validate_promoted_review_artifact(target) == []


def test_promoted_review_keeps_packet_projection_fingerprint_compatible(tmp_path: Path) -> None:
    source = tmp_path / "run" / "decisions.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"schema_name": "fixture", "records": [{"id": "q1"}]}), encoding="utf-8")
    report = promote_review_artifact(
        source,
        "topic/decisions.v1.json",
        authority="human",
        source_run_id="review-run-1",
        reviewed_by="reviewer@example.invalid",
        reviewed_at="2026-07-13T00:00:00Z",
        canonical_root=tmp_path / "canonical",
    )
    source_provenance = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    promoted_provenance = path_provenance(Path(report["target"]))
    base_manifest = {
        "schema_version": 1,
        "paper_family": "p1",
        "topic_id": "quadratics",
        "subtopic_id": None,
        "included_records": [
            {
                "question_id": "q1",
                "primary_topic_id": "quadratics",
                "secondary_topic_ids": [],
                "section": "approved",
            }
        ],
    }
    legacy_manifest = base_manifest | {
        "routing_provenance": {
            "reviewed_decisions": source_provenance,
            "generator_schema_version": 1,
        }
    }
    promoted_manifest = base_manifest | {
        "routing_provenance": {
            "reviewed_decisions": promoted_provenance,
            "generator_schema_version": 1,
        }
    }

    assert promoted_provenance["path"] == report["target"]
    assert promoted_provenance["promoted_from"] == source_provenance
    assert packet_projection_fingerprint(promoted_manifest) == packet_projection_fingerprint(legacy_manifest)


def test_dry_run_does_not_write_target(tmp_path: Path) -> None:
    source = tmp_path / "decisions.json"
    source.write_text("{}", encoding="utf-8")

    report = promote_review_artifact(
        source,
        "decisions.json",
        authority="automated_review",
        source_run_id="run-1",
        reviewed_by="review-agent",
        canonical_root=tmp_path / "canonical",
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert not Path(report["target"]).exists()


@pytest.mark.parametrize("target", ["../escape.json", "/tmp/escape.json", "not-json.txt"])
def test_rejects_targets_outside_canonical_json_root(tmp_path: Path, target: str) -> None:
    source = tmp_path / "decisions.json"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(ReviewPromotionError):
        promote_review_artifact(
            source,
            target,
            authority="human",
            source_run_id="run-1",
            reviewed_by="reviewer",
            canonical_root=tmp_path / "canonical",
        )
