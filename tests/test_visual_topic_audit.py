from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from exam_bank.topic_packets import generate_topic_packets, load_packet_taxonomy
from exam_bank.visual_topic_audit import (
    VISUAL_AUDIT_DECISION_VERSION,
    build_visual_topic_audit_batch,
    import_visual_topic_audit_decisions,
    validate_visual_topic_audit_decision,
)


def test_build_visual_topic_audit_batch_parses_queues_and_maps_p4_mechanics_records(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, family="mechanics", component="41", paper="41summer17", topic="forces_newtons_second_law")
    audit = _audit_report(
        tmp_path,
        missing=[{"paper_family": "p4", "paper": "41summer17", "missing_topics": ["momentum"]}],
        ge3=[{"paper_family": "p4", "paper": "41summer17", "topics_count_ge_3": {"forces_and_equilibrium": 3}}],
        counts={"newtons_laws_of_motion": 1, "momentum": 0, "forces_and_equilibrium": 3},
    )

    batch = build_visual_topic_audit_batch(
        question_bank_path=paths["bank"],
        packet_audit_path=audit,
        packet_summary_path=tmp_path / "missing_summary.json",
        taxonomy_path=paths["taxonomy"],
        artifact_root=paths["artifact_root"],
        existing_overlap_review_path=None,
        queue="both",
    )

    assert batch["selection"]["selected_paper_count"] == 1
    row = batch["rows"][0]
    assert row["paper_family"] == "p4"
    assert row["raw_question_bank_family"] == "p4"
    assert row["source_component_family"] == "p4"
    assert row["current_topic"] == "newtons_laws_of_motion"
    assert row["missing_topics"] == ["momentum"]
    assert row["high_count_topics"] == {"forces_and_equilibrium": 3}
    assert row["allowed_packet_topics"] == row["allowed_topics"]
    assert {topic["topic_id"] for topic in row["allowed_packet_topics"]} >= {
        "forces_and_equilibrium",
        "momentum",
    }
    assert row["canonical_question_image_path"]
    assert row["canonical_mark_scheme_image_path"]
    assert row["image_evidence_available"] is True


def test_visual_topic_audit_decision_validation_rejects_bad_topics_and_missing_evidence(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, family="mechanics", component="41", paper="41summer17", topic="forces_newtons_second_law")
    batch = _build_single_batch(paths, tmp_path, "p4", "41summer17", ["momentum"])
    rows = {batch["rows"][0]["question_id"]: batch["rows"][0]}
    base = _decision(batch["rows"][0], primary_topic="invented", coverage_topics=["invented"])
    errors = validate_visual_topic_audit_decision(base, batch_rows=rows, taxonomy=paths["taxonomy_payload"], artifact_root=paths["artifact_root"])
    assert "unknown_primary_topic:invented" in errors
    assert "unknown_coverage_topic:invented" in errors

    missing_evidence = _decision(batch["rows"][0], primary_topic="momentum", coverage_topics=["momentum"])
    missing_evidence["evidence_refs"] = []
    errors = validate_visual_topic_audit_decision(
        missing_evidence,
        batch_rows=rows,
        taxonomy=paths["taxonomy_payload"],
        artifact_root=paths["artifact_root"],
    )
    assert "missing_evidence_refs" in errors

    pending = {
        "decision_version": VISUAL_AUDIT_DECISION_VERSION,
        "question_id": batch["rows"][0]["question_id"],
        "status": "pending",
    }
    assert validate_visual_topic_audit_decision(
        pending,
        batch_rows=rows,
        taxonomy=paths["taxonomy_payload"],
        artifact_root=paths["artifact_root"],
    ) == []


def test_import_visual_topic_audit_decisions_merges_p4_and_skips_pending(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, family="mechanics", component="41", paper="41summer17", topic="forces_newtons_second_law")
    batch = _build_single_batch(paths, tmp_path, "p4", "41summer17", ["momentum"])
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, batch)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(_decision(batch["rows"][0], primary_topic="newtons_laws_of_motion", secondary_topics=["momentum"], coverage_topics=["newtons_laws_of_motion", "momentum"]))
        + "\n"
        + json.dumps({"decision_version": VISUAL_AUDIT_DECISION_VERSION, "question_id": "pending_q", "status": "pending"})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "merged.json"

    report = import_visual_topic_audit_decisions(
        batch_path=batch_path,
        decisions_path=decisions,
        base_overlap_review_path=None,
        out_overlap_review_path=out,
        taxonomy_path=paths["taxonomy"],
        artifact_root=paths["artifact_root"],
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["imported_count"] == 1
    assert report["pending_count"] == 1
    assert payload["paper_families"] == ["p4"]
    assert payload["records"][0]["paper_family"] == "p4"
    assert payload["records"][0]["coverage_topics"] == ["newtons_laws_of_motion", "momentum"]


def test_visual_topic_audit_secondary_coverage_does_not_duplicate_pdf_records(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, family="mechanics", component="41", paper="41summer17", topic="forces_newtons_second_law")
    batch = _build_single_batch(paths, tmp_path, "p4", "41summer17", ["momentum"])
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, batch)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(_decision(batch["rows"][0], primary_topic="newtons_laws_of_motion", secondary_topics=["momentum"], coverage_topics=["newtons_laws_of_motion", "momentum"]))
        + "\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "merged.json"
    import_visual_topic_audit_decisions(
        batch_path=batch_path,
        decisions_path=decisions,
        base_overlap_review_path=None,
        out_overlap_review_path=sidecar,
        taxonomy_path=paths["taxonomy"],
        artifact_root=paths["artifact_root"],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        output_root=tmp_path / "packets",
        artifact_root=paths["artifact_root"],
        topic_overlap_review_path=sidecar,
    )

    assert summary["total_unique_questions_included"] == 1
    assert summary["total_topic_coverage_placements"] == 2
    assert summary["topic_overlap_reviews_applied"] == 1


def test_visual_topic_audit_can_exclude_p3_routing_artifact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, family="pm1", component="13", paper="13summer15", topic="algebra")
    audit = _audit_report(
        tmp_path,
        missing=[
            {
                "paper_family": "p3",
                "paper": "13summer15",
                "missing_topics": ["complex_numbers", "differentiation"],
            }
        ],
        ge3=[],
        counts={"algebra": 1, "complex_numbers": 0, "differentiation": 0},
    )
    batch = build_visual_topic_audit_batch(
        question_bank_path=paths["bank"],
        packet_audit_path=audit,
        packet_summary_path=tmp_path / "missing_summary.json",
        taxonomy_path=paths["taxonomy"],
        artifact_root=paths["artifact_root"],
        existing_overlap_review_path=None,
        queue="missing",
    )
    assert batch["rows"][0]["identity_warning"] == "source_component_family_mismatch"
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, batch)
    decision = _decision(batch["rows"][0], status="exclude_current_syllabus", primary_topic="", coverage_topics=[])
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(decision) + "\n", encoding="utf-8")
    sidecar = tmp_path / "merged.json"
    import_visual_topic_audit_decisions(
        batch_path=batch_path,
        decisions_path=decisions,
        base_overlap_review_path=None,
        out_overlap_review_path=sidecar,
        taxonomy_path=paths["taxonomy"],
        artifact_root=paths["artifact_root"],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        output_root=tmp_path / "packets",
        artifact_root=paths["artifact_root"],
        topic_overlap_review_path=sidecar,
        paper_family="p3",
    )

    assert summary["total_included"] == 0
    assert summary["skipped_records"][0]["reason"] == "topic_overlap_current_syllabus_exclude"


def _fixture(tmp_path: Path, *, family: str, component: str, paper: str, topic: str) -> dict[str, object]:
    artifact_root = tmp_path / "output"
    _png(artifact_root / "questions" / f"{paper}_q01.png")
    _png(artifact_root / "marks" / f"{paper}_q01.png")
    bank = tmp_path / "question_bank.json"
    _write_json(
        bank,
        {
            "schema_name": "exam_bank.question_bank",
            "schema_version": 2,
            "questions": [
                {
                    "question_id": f"{paper}_q01",
                    "paper": paper,
                    "paper_family": family,
                    "question_number": "1",
                    "topic": topic,
                    "question_text": "A visual mechanics or algebra question.",
                    "mark_scheme_text": "Award method marks for the topic.",
                    "question_image_path": f"questions/{paper}_q01.png",
                    "mark_scheme_image_path": f"marks/{paper}_q01.png",
                    "notes": {"source_paper_code": component, "topic_trust_status": "normal", "topic_confidence": "high"},
                }
            ],
        },
    )
    taxonomy_payload = _taxonomy()
    taxonomy = tmp_path / "taxonomy.json"
    _write_json(taxonomy, taxonomy_payload)
    return {
        "artifact_root": artifact_root,
        "bank": bank,
        "taxonomy": taxonomy,
        "taxonomy_payload": load_packet_taxonomy(taxonomy),
    }


def _build_single_batch(paths: dict[str, object], tmp_path: Path, family: str, paper: str, missing_topics: list[str]) -> dict[str, object]:
    audit = _audit_report(
        tmp_path,
        missing=[{"paper_family": family, "paper": paper, "missing_topics": missing_topics}],
        ge3=[],
        counts={topic: 0 for topic in missing_topics},
    )
    return build_visual_topic_audit_batch(
        question_bank_path=paths["bank"],
        packet_audit_path=audit,
        packet_summary_path=tmp_path / "missing_summary.json",
        taxonomy_path=paths["taxonomy"],
        artifact_root=paths["artifact_root"],
        existing_overlap_review_path=None,
        queue="missing",
    )


def _audit_report(tmp_path: Path, *, missing: list[dict[str, object]], ge3: list[dict[str, object]], counts: dict[str, int]) -> Path:
    paper = str((missing or ge3)[0]["paper"])
    family = str((missing or ge3)[0]["paper_family"])
    path = tmp_path / f"audit_{paper}.json"
    _write_json(
        path,
        {
            "schema_name": "exam_bank.topic_packet_paper_topic_audit",
            "schema_version": 1,
            "papers_lacking_at_least_one_topic": missing,
            "papers_with_topic_count_ge_3": ge3,
            "paper_topic_counts": [
                {
                    "paper_family": family,
                    "paper": paper,
                    "topic_coverage_counts": counts,
                    "unique_question_count": 1,
                }
            ],
        },
    )
    return path


def _decision(
    row: dict[str, object],
    *,
    status: str = "relabel_primary_add_secondary",
    primary_topic: str,
    secondary_topics: list[str] | None = None,
    coverage_topics: list[str],
) -> dict[str, object]:
    return {
        "decision_version": VISUAL_AUDIT_DECISION_VERSION,
        "question_id": row["question_id"],
        "paper": row["paper"],
        "paper_family": row["paper_family"],
        "status": status,
        "primary_topic": primary_topic,
        "secondary_topics": secondary_topics or [],
        "coverage_topics": coverage_topics,
        "rationale": "The canonical question and mark-scheme images support this topic path.",
        "evidence_refs": row["evidence_refs"],
        "source": "manual_visual_audit:test",
    }


def _taxonomy() -> dict[str, object]:
    return {
        "schema_name": "exam_bank.caie_9709_syllabus_topics",
        "schema_version": 1,
        "components": [
            {
                "paper_family": "p1",
                "component_key": "p1",
                "topics": [
                    {"topic_id": "quadratics", "topic_label": "Quadratics", "canonical_topic_id": "9709_p1_topic_quadratics", "subtopics": []}
                ],
            },
            {
                "paper_family": "p3",
                "component_key": "p3",
                "topics": [
                    {"topic_id": "algebra", "topic_label": "Algebra", "canonical_topic_id": "9709_p3_topic_algebra", "subtopics": []},
                    {
                        "topic_id": "complex_numbers",
                        "topic_label": "Complex numbers",
                        "canonical_topic_id": "9709_p3_topic_complex_numbers",
                        "subtopics": [],
                    },
                    {
                        "topic_id": "differentiation",
                        "topic_label": "Differentiation",
                        "canonical_topic_id": "9709_p3_topic_differentiation",
                        "subtopics": [],
                    },
                ],
            },
            {
                "paper_family": "p4",
                "component_key": "m1",
                "topics": [
                    {
                        "topic_id": "newtons_laws_of_motion",
                        "topic_label": "Newton's laws of motion",
                        "canonical_topic_id": "9709_m1_topic_newtons_laws_of_motion",
                        "deprecated_aliases": ["forces_newtons_second_law"],
                        "subtopics": [],
                    },
                    {
                        "topic_id": "momentum",
                        "topic_label": "Momentum",
                        "canonical_topic_id": "9709_m1_topic_momentum",
                        "subtopics": [],
                    },
                    {
                        "topic_id": "forces_and_equilibrium",
                        "topic_label": "Forces and equilibrium",
                        "canonical_topic_id": "9709_m1_topic_forces_and_equilibrium",
                        "subtopics": [],
                    },
                ],
            },
        ],
    }


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 80), color="white").save(path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
