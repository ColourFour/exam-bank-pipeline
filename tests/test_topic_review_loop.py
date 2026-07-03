from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from exam_bank.topic_review_loop import (
    build_topic_review_batch,
    import_topic_review_decisions,
    merge_topic_review_decision_files,
)


def test_build_topic_review_batch_selects_review_required_and_skips_existing(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.topic_bank_reviewed_decisions",
                "schema_version": 1,
                "records": [
                    {
                        "question_id": "q_existing",
                        "action": "keep",
                        "reviewed_topic": "integration",
                        "reviewed_subtopic": "",
                        "reviewed_skill": "",
                        "reason": "Already checked.",
                        "reviewer": "test",
                        "reviewed_at": "2026-06-29T00:00:00Z",
                        "source": "manual_review",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    batch = build_topic_review_batch(
        question_bank_path=paths["bank"],
        topic_routing_path=paths["routing"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=reviewed,
        out_dir=tmp_path / "batch",
        dry_run=True,
    )

    assert batch["manifest"]["review_required_input_count"] == 2
    assert batch["manifest"]["selected_count"] == 1
    assert batch["manifest"]["skipped_reason_counts"]["already_reviewed"] == 1
    row = batch["rows"][0]
    assert row["question_id"] == "q_review"
    assert row["image_evidence_available"] is True
    assert row["allowed_topics"][0]["topic_id"] == "integration"


def test_import_topic_review_decisions_writes_accepted_auto_decision(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    batch = build_topic_review_batch(
        question_bank_path=paths["bank"],
        topic_routing_path=paths["routing"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=None,
        out_dir=tmp_path / "batch",
        dry_run=False,
    )
    batch_path = tmp_path / "batch" / "topic_review_batch.json"
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(_decision(batch["rows"][0], paths["artifact_root"])) + "\n", encoding="utf-8")
    out = tmp_path / "auto_reviewed.json"

    report = import_topic_review_decisions(
        decisions_path=decisions,
        batch_path=batch_path,
        out_review_file=out,
        artifact_root=paths["artifact_root"],
        taxonomy_path=paths["taxonomy"],
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["accepted_count"] == 1
    assert payload["records"][0]["source"] == "automated_agentic_review"
    assert payload["records"][0]["release_override"] is True
    assert payload["records"][0]["current_syllabus_status"] == "current_relevant"


def test_import_topic_review_decisions_rejects_low_confidence_approval(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    batch = build_topic_review_batch(
        question_bank_path=paths["bank"],
        topic_routing_path=paths["routing"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=None,
        out_dir=tmp_path / "batch",
        dry_run=False,
    )
    batch_path = tmp_path / "batch" / "topic_review_batch.json"
    decision = _decision(batch["rows"][0], paths["artifact_root"])
    decision["confidence"] = 0.5
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(decision) + "\n", encoding="utf-8")

    report = import_topic_review_decisions(
        decisions_path=decisions,
        batch_path=batch_path,
        out_review_file=tmp_path / "auto_reviewed.json",
        artifact_root=paths["artifact_root"],
        taxonomy_path=paths["taxonomy"],
        dry_run=True,
    )

    assert report["ok"] is False
    assert any("confidence_below_threshold" in error for error in report["errors"])


def test_pending_topic_review_is_reported_but_not_imported(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    batch = build_topic_review_batch(
        question_bank_path=paths["bank"],
        topic_routing_path=paths["routing"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=None,
        out_dir=tmp_path / "batch",
        dry_run=False,
        limit=1,
    )
    batch_path = tmp_path / "batch" / "topic_review_batch.json"
    decision = _decision(batch["rows"][0], paths["artifact_root"])
    decision["decision_action"] = "pending"
    decision["confidence"] = 0.2
    decision["current_syllabus_status"] = "ambiguous"
    decision["release_override"] = False
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(decision) + "\n", encoding="utf-8")

    report = import_topic_review_decisions(
        decisions_path=decisions,
        batch_path=batch_path,
        out_review_file=tmp_path / "auto_reviewed.json",
        artifact_root=paths["artifact_root"],
        taxonomy_path=paths["taxonomy"],
        dry_run=True,
    )

    assert report["ok"] is True
    assert report["accepted_count"] == 0
    assert report["pending_count"] == 1


def test_merge_topic_review_decisions_detects_conflicting_duplicates(tmp_path: Path) -> None:
    file_a = _review_file(tmp_path / "a.json", "q1", "keep")
    file_b = _review_file(tmp_path / "b.json", "q1", "exclude")

    report = merge_topic_review_decision_files(
        reviewed_files=[file_a, file_b],
        out_review_file=tmp_path / "merged.json",
        dry_run=True,
    )

    assert report["ok"] is False
    assert report["error_count"] == 1
    assert "duplicate_conflicting_decision:q1" in report["errors"]


def _fixture(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "output"
    for qid in ("q_review", "q_existing"):
        _png(artifact_root / "p3" / "paper" / "questions" / f"{qid}.png")
        _png(artifact_root / "p3" / "paper" / "mark_scheme" / f"{qid}.png")
    bank = tmp_path / "question_bank.json"
    bank.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.question_bank",
                "schema_version": 2,
                "questions": [_record("q_review"), _record("q_existing")],
            }
        ),
        encoding="utf-8",
    )
    routing = tmp_path / "routing.json"
    routing.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.topic_routing_sidecar",
                "schema_version": 1,
                "records": {
                    "q_review": _route(),
                    "q_existing": _route(),
                    "q_strict": {**_route(), "review_required": False},
                },
            }
        ),
        encoding="utf-8",
    )
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    taxonomy = _write_taxonomy(tmp_path)
    return {
        "artifact_root": artifact_root,
        "bank": bank,
        "routing": routing,
        "canonical_root": canonical_root,
        "taxonomy": taxonomy,
    }


def _record(question_id: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "31summer24",
        "paper_family": "p3",
        "question_number": "1",
        "topic": "integration",
        "question_text": "Integrate x squared.",
        "mark_scheme_text": "Uses integration.",
        "question_image_path": f"p3/paper/questions/{question_id}.png",
        "mark_scheme_image_path": f"p3/paper/mark_scheme/{question_id}.png",
    }


def _route() -> dict[str, object]:
    return {
        "primary_topic_id": "9709_p3_topic_integration",
        "topic_distribution": [{"topic_id": "9709_p3_topic_integration", "fit_percent": 100}],
        "confidence": "medium",
        "review_required": True,
        "review_reasons": ["visual_curation_status_not_ready"],
        "paper_family": "p3",
        "packet_topic_id": "integration",
        "routing_source": "deterministic_topic_packet_normalization",
    }


def _decision(row: dict[str, object], artifact_root: Path) -> dict[str, object]:
    question_id = str(row["question_id"])
    return {
        "decision_version": "topic_review_auto_decision_v1",
        "review_source": "automated_agentic_review",
        "question_id": question_id,
        "decision_action": "keep",
        "reviewed_topic": "integration",
        "reviewed_subtopic": "",
        "reviewed_skill": "",
        "current_syllabus_status": "current_relevant",
        "release_override": True,
        "confidence": 0.95,
        "evidence_refs": [
            {
                "type": "canonical_question_image",
                "path": str(artifact_root / "p3" / "paper" / "questions" / f"{question_id}.png"),
            },
            {
                "type": "canonical_mark_scheme_image",
                "path": str(artifact_root / "p3" / "paper" / "mark_scheme" / f"{question_id}.png"),
            },
            {
                "type": "syllabus_reference",
                "path": "https://www.cambridgeinternational.org/Images/697427-2026-2027-syllabus.pdf",
            },
        ],
        "risk_flags": [],
        "explanation": "The inspected images assess integration and the topic is current in the 2026-2027 syllabus.",
        "reviewer_model": "gpt-5-mini",
        "prompt_version": "topic_review_9709_2026_2027_v1",
    }


def _review_file(path: Path, question_id: str, action: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.topic_bank_reviewed_decisions",
                "schema_version": 1,
                "records": [
                    {
                        "question_id": question_id,
                        "action": action,
                        "reviewed_topic": "integration" if action != "exclude" else "",
                        "reviewed_subtopic": "",
                        "reviewed_skill": "",
                        "reason": "fixture",
                        "reviewer": "test",
                        "reviewed_at": "2026-06-29T00:00:00Z",
                        "source": "manual_review",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_taxonomy(tmp_path: Path) -> Path:
    path = tmp_path / "taxonomy.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.caie_9709_syllabus_topics",
                "schema_version": 1,
                "components": [
                    {
                        "paper_family": "p3",
                        "component_key": "p3",
                        "paper_code_group": "Paper 3",
                        "topics": [
                            {
                                "topic_id": "integration",
                                "topic_label": "Integration",
                                "canonical_topic_id": "9709_p3_topic_integration",
                                "subtopics": [],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color="white").save(path)

