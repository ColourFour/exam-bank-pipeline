from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from exam_bank.topic_difficulty_review import (
    TOPIC_DIFFICULTY_DECISION_VERSION,
    TOPIC_DIFFICULTY_PROMPT_VERSION,
    build_topic_difficulty_batch,
    import_topic_difficulty_decisions,
    run_topic_difficulty_reviews,
)


def test_build_topic_difficulty_batch_from_manifest_resolves_images(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    batch = build_topic_difficulty_batch(
        manifest_path=paths["manifest"],
        artifact_root=paths["artifact_root"],
        out_dir=tmp_path / "review",
        dry_run=False,
    )

    assert batch["selection"]["question_count"] == 3
    assert batch["selection"]["missing_question_image_count"] == 0
    assert batch["selection"]["missing_mark_scheme_image_count"] == 0
    assert batch["packet"]["topic_id"] == "integration"
    assert batch["rows"][0]["question_id"] == "q1"
    assert batch["rows"][0]["canonical_question_image_path"].endswith("q1.png")
    assert batch["rows"][0]["canonical_mark_scheme_image_path"].endswith("q1_ms.png")
    assert batch["rows"][0]["image_evidence_available"] is True
    packet_dir = tmp_path / "review" / batch["packet_id"]
    assert (packet_dir / "topic_packet_difficulty_batch.json").is_file()
    assert (packet_dir / "topic_packet_difficulty_batch.md").is_file()


def test_topic_difficulty_runner_dry_run_resumes_existing_jsonl(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    batch = build_topic_difficulty_batch(manifest_path=paths["manifest"], artifact_root=paths["artifact_root"], dry_run=True)
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, batch)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(_decision(batch["rows"][0], 70)) + "\n", encoding="utf-8")

    manifest = run_topic_difficulty_reviews(
        batch_path=batch_path,
        out_path=decisions,
        max_records=1,
        dry_run=True,
    )

    assert manifest["resumed_count"] == 1
    assert manifest["pending_count"] == 1


def test_import_rejects_invalid_decisions_and_missing_full_coverage(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    batch = build_topic_difficulty_batch(manifest_path=paths["manifest"], artifact_root=paths["artifact_root"], dry_run=True)
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, batch)
    bad = [
        _decision(batch["rows"][0], 101),
        _decision(batch["rows"][0], 50),
        {**_decision(batch["rows"][1], 40), "question_id": "unknown"},
        {**_decision(batch["rows"][1], 40), "status": "pending", "evidence_refs": []},
    ]
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("\n".join(json.dumps(row) for row in bad) + "\n", encoding="utf-8")

    report = import_topic_difficulty_decisions(
        batch_path=batch_path,
        decisions_path=decisions,
        out_path=tmp_path / "sidecar.json",
        artifact_root=paths["artifact_root"],
        reports_dir=tmp_path / "reports",
        dry_run=True,
    )

    assert report["ok"] is False
    assert any("invalid_visual_difficulty_score_0_100" in error for error in report["errors"])
    assert any("duplicate_decision" in error for error in report["errors"])
    assert any("unknown_question_id" in error for error in report["errors"])
    assert any("pending_decision" in error for error in report["errors"])
    assert any("missing_decision:q2" in error for error in report["errors"])
    assert any("missing_decision:q3" in error for error in report["errors"])


def test_complete_import_assigns_unique_packet_ranks_and_hardness_percentiles(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    before_hash = _sha256(paths["manifest"])
    batch = build_topic_difficulty_batch(manifest_path=paths["manifest"], artifact_root=paths["artifact_root"], dry_run=True)
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, batch)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        "\n".join(
            [
                json.dumps(_decision(batch["rows"][0], 80, confidence="high")),
                json.dumps(_decision(batch["rows"][1], 80, confidence="high")),
                json.dumps(_decision(batch["rows"][2], 25, confidence="medium")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sidecar.json"

    report = import_topic_difficulty_decisions(
        batch_path=batch_path,
        decisions_path=decisions,
        out_path=out,
        artifact_root=paths["artifact_root"],
        reports_dir=tmp_path / "reports",
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    records = {record["question_id"]: record for record in payload["records"]}
    assert report["ok"] is True
    assert payload["complete"] is True
    assert payload["safe_for_teacher_filtering"] is True
    assert payload["safe_for_student_sequencing"] is False
    assert [record["packet_rank"] for record in payload["records"]] == [1, 2, 3]
    assert records["q2"]["packet_rank"] == 1
    assert records["q2"]["difficulty_percentile_0_100"] == 100.0
    assert records["q1"]["packet_rank"] == 2
    assert records["q1"]["difficulty_percentile_0_100"] == 50.0
    assert records["q3"]["packet_rank"] == 3
    assert records["q3"]["difficulty_percentile_0_100"] == 0.0
    assert _sha256(paths["manifest"]) == before_hash
    assert (tmp_path / "reports" / batch["packet_id"] / "ranking.md").is_file()


def test_allow_incomplete_writes_draft_unsafe_sidecar(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    batch = build_topic_difficulty_batch(manifest_path=paths["manifest"], artifact_root=paths["artifact_root"], dry_run=True)
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, batch)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(_decision(batch["rows"][0], 70)) + "\n", encoding="utf-8")
    out = tmp_path / "draft.json"

    report = import_topic_difficulty_decisions(
        batch_path=batch_path,
        decisions_path=decisions,
        out_path=out,
        artifact_root=paths["artifact_root"],
        reports_dir=tmp_path / "reports",
        allow_incomplete=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["complete"] is False
    assert payload["draft"] is True
    assert payload["safe_for_teacher_filtering"] is False
    assert payload["safe_for_student_sequencing"] is False
    assert payload["record_count"] == 1


def _fixture(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "output"
    for question_id in ("q1", "q2", "q3"):
        _png(artifact_root / "questions" / f"{question_id}.png")
        _png(artifact_root / "marks" / f"{question_id}_ms.png")
    manifest = tmp_path / "packet" / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_name": "exam_bank.topic_packets",
            "schema_version": 1,
            "paper_family": "p3",
            "topic_id": "integration",
            "topic_label": "Integration",
            "packet_level": "major_topic",
            "packet_mode": "combined",
            "pdf_path": str(tmp_path / "packet" / "p3_integration_packet.pdf"),
            "total_questions": 3,
            "approved_count": 2,
            "review_required_count": 1,
            "included_records": [
                _record("q1", 1, marks=4, section="approved"),
                _record("q2", 2, marks=6, section="approved"),
                _record("q3", 3, marks=3, section="review_required"),
            ],
        },
    )
    return {"artifact_root": artifact_root, "manifest": manifest}


def _record(question_id: str, problem_number: int, *, marks: int, section: str) -> dict[str, object]:
    return {
        "problem_number": problem_number,
        "question_id": question_id,
        "source_label": f"Question {problem_number}",
        "paper": "31summer24",
        "question_number": str(problem_number),
        "marks": marks,
        "question_image_paths": [f"questions/{question_id}.png"],
        "mark_scheme_image_paths": [f"marks/{question_id}_ms.png"],
        "answer_available": True,
        "warnings": [],
        "section": section,
        "primary_topic_id": "integration",
        "secondary_topic_ids": [],
        "coverage_topic_ids": ["integration"],
        "review_reasons": ["needs_review"] if section == "review_required" else [],
    }


def _decision(row: dict[str, object], score: int, *, confidence: str = "high") -> dict[str, object]:
    return {
        "decision_version": TOPIC_DIFFICULTY_DECISION_VERSION,
        "question_id": row["question_id"],
        "status": "accepted",
        "visual_difficulty_score_0_100": score,
        "confidence": confidence,
        "rationale": "Visual evidence supports this packet-relative score.",
        "difficulty_factors": ["multi-step method"],
        "risk_flags": [],
        "evidence_refs": row["evidence_refs"],
        "source": "test",
        "reviewer_model": "test-model",
        "prompt_version": TOPIC_DIFFICULTY_PROMPT_VERSION,
    }


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=(255, 255, 255)).save(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
