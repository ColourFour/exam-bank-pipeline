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
    reconcile_topic_difficulty,
    run_topic_difficulty_reviews,
)


def test_reconcile_preserves_same_packet_and_flags_moved_new_and_removed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    packets_root = tmp_path / "packets"
    manifest_path = packets_root / "p3" / "integration" / "manifest.json"
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    _write_json(manifest_path, manifest)
    difficulty_root = tmp_path / "difficulty"
    _write_json(
        difficulty_root / "p3_integration_old" / "topic_packet_difficulty_review.v1.json",
        _legacy_sidecar(
            "p3_integration_old",
            "integration",
            [
                _legacy_record("q1", score=80, percentile=80),
                _legacy_record("q_removed", score=55, percentile=40),
            ],
        ),
    )
    _write_json(
        difficulty_root / "p3_differentiation_old" / "topic_packet_difficulty_review.v1.json",
        _legacy_sidecar(
            "p3_differentiation_old",
            "differentiation",
            [_legacy_record("q2", score=65, percentile=60)],
        ),
    )
    difficulty_index = tmp_path / "difficulty_index.json"
    _write_json(
        difficulty_index,
        {"records": [{"question_id": "q3", "paper_relative_percentile": 25.0}]},
    )

    report = reconcile_topic_difficulty(
        packets_root=packets_root,
        difficulty_root=difficulty_root,
        difficulty_index_path=difficulty_index,
        artifact_root=paths["artifact_root"],
        reports_dir=tmp_path / "reports",
        auto_review=False,
    )

    packet_report = report["packets"][0]
    sidecar_path = next(difficulty_root.glob("*/topic_packet_difficulty_review.v2.json"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    records = {row["question_id"]: row for row in sidecar["records"]}
    assert packet_report["removed_question_ids"] == ["q_removed"]
    assert records["q1"]["difficulty_status"] == "reviewed"
    assert records["q1"]["evidence_refs"] == [
        {
            "type": "canonical_question_image",
            "path": str(paths["artifact_root"] / "questions" / "q1.png"),
        },
        {
            "type": "canonical_mark_scheme_image",
            "path": str(paths["artifact_root"] / "marks" / "q1_ms.png"),
        },
    ]
    assert records["q2"]["difficulty_status"] == "provisional_topic_changed"
    assert records["q2"]["provisional_percentile_0_100"] == 60.0
    assert records["q2"]["source_packet_id"] == "p3_differentiation_old"
    assert records["q3"]["difficulty_status"] == "provisional_new_member"
    assert records["q3"]["provisional_percentile_0_100"] == 25.0
    assert [row["question_id"] for row in sidecar["records"]] == ["q1", "q2", "q3"]
    assert sidecar["difficulty_ranking_complete"] is False
    assert sidecar["pending_question_ids"] == ["q2", "q3"]
    assert sidecar["projection_fingerprint"]


def test_reconcile_automatic_review_replaces_provisional_record(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    packets_root = tmp_path / "packets"
    manifest_path = packets_root / "p3" / "integration" / "manifest.json"
    _write_json(manifest_path, json.loads(paths["manifest"].read_text(encoding="utf-8")))
    difficulty_root = tmp_path / "difficulty"
    _write_json(
        difficulty_root / "p3_integration_old" / "topic_packet_difficulty_review.v1.json",
        _legacy_sidecar("p3_integration_old", "integration", [_legacy_record("q1", score=80, percentile=80)]),
    )
    difficulty_index = tmp_path / "difficulty_index.json"
    _write_json(
        difficulty_index,
        {
            "records": [
                {"question_id": "q2", "paper_relative_percentile": 50.0},
                {"question_id": "q3", "paper_relative_percentile": 20.0},
            ]
        },
    )

    def fake_runner(*, batch_path, out_path, **_kwargs):
        batch = json.loads(Path(batch_path).read_text(encoding="utf-8"))
        decisions = [_decision(row, 70 - index * 10) for index, row in enumerate(batch["rows"])]
        Path(out_path).write_text("\n".join(json.dumps(row) for row in decisions) + "\n", encoding="utf-8")
        return {"pending_count": len(decisions)}

    # Patch the exact globals dictionary used by the imported callable. Some
    # CLI safety tests intentionally reload modules, so a string-based patch can
    # otherwise target a newer module object than this collected test uses.
    monkeypatch.setitem(reconcile_topic_difficulty.__globals__, "run_topic_difficulty_reviews", fake_runner)
    report = reconcile_topic_difficulty(
        packets_root=packets_root,
        difficulty_root=difficulty_root,
        difficulty_index_path=difficulty_index,
        artifact_root=paths["artifact_root"],
        reports_dir=tmp_path / "reports",
    )

    packet_report = report["packets"][0]
    sidecar_path = difficulty_root / packet_report["packet_id"] / "topic_packet_difficulty_review.v2.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert report["pending_count"] == 0, report
    assert sidecar["difficulty_ranking_complete"] is True
    assert sidecar["pending_question_ids"] == []
    assert {row["difficulty_status"] for row in sidecar["records"]} == {"reviewed"}


def _legacy_sidecar(packet_id: str, topic_id: str, records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_name": "exam_bank.topic_packet_difficulty_review",
        "schema_version": 1,
        "generated_at": "2026-07-06T00:00:00+00:00",
        "packet_id": packet_id,
        "complete": True,
        "packet": {"paper_family": "p3", "topic_id": topic_id, "subtopic_id": ""},
        "expected_record_count": len(records),
        "records": records,
    }


def _legacy_record(question_id: str, *, score: int, percentile: int) -> dict[str, object]:
    return {
        "question_id": question_id,
        "visual_difficulty_score_0_100": score,
        "difficulty_percentile_0_100": percentile,
        "packet_rank": 1,
        "confidence": "high",
        "rationale": "Reviewed legacy packet evidence.",
        "difficulty_factors": [],
        "risk_flags": [],
        "evidence_refs": [],
        "source": "legacy-test",
    }


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
