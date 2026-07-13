from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw

from exam_bank.topic_packet_visual_audit import (
    TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION,
    TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION,
    TopicPacketVisualAuditError,
    build_topic_packet_visual_audit_batch,
    import_topic_packet_visual_audit_decisions,
    run_topic_packet_visual_audit_reviews,
)


def test_build_topic_packet_visual_audit_batch_maps_pages_to_problems_and_seed_bugs(tmp_path: Path) -> None:
    paths = _packet_fixture(tmp_path)

    batch = build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
        seed_bugs=[
            {"type": "problem", "user_label": "Fixture Q1", "question_id": "q1"},
            {"type": "mark_scheme", "user_label": "Fixture Q2 mark scheme", "question_id": "q2"},
        ],
    )

    assert batch["selection"]["packet_count"] == 1
    assert batch["selection"]["page_count"] == 3
    assert batch["selection"]["seed_bug_page_count"] == 2
    assert (paths["review_dir"] / "topic_packet_visual_audit_batch.json").is_file()
    assert (paths["review_dir"] / "topic_packet_visual_audit_batch.md").is_file()

    question_page = batch["rows"][0]
    assert question_page["row_id"] == "p1_circular_measure_page_0001"
    assert question_page["page_section"] == "Questions"
    assert question_page["related_problem_numbers"] == [1]
    assert question_page["related_question_ids"] == ["q1"]
    assert question_page["seed_bug_refs"][0]["user_label"] == "Fixture Q1"
    assert Path(question_page["page_image_path"]).is_file()

    answer_page = batch["rows"][2]
    assert answer_page["page_section"] == "Answers / Mark Schemes"
    assert answer_page["related_question_ids"] == ["q1", "q2"]
    assert answer_page["seed_bug_refs"][0]["user_label"] == "Fixture Q2 mark scheme"
    assert "oversized_block_scaled_below_legibility:answer:1:scale=0.31" in answer_page["layout_warnings"]


def test_build_topic_packet_visual_audit_maps_split_answer_continuation_page(tmp_path: Path) -> None:
    paths = _packet_fixture(tmp_path)
    manifest_path = paths["packet_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["page_count"] = 4
    manifest["answers_section_page_range"] = [3, 4]
    manifest["included_records"][0]["answer_end_page"] = 4
    _pdf(Path(manifest["pdf_path"]), ["Problem 1", "Problem 2", "Answers", "Answer 1 continued"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    batch = build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
        seed_bugs=[],
    )

    continuation_page = batch["rows"][3]
    assert continuation_page["page_section"] == "Answers / Mark Schemes"
    assert continuation_page["related_question_ids"] == ["q1"]
    assert continuation_page["related_records"][0]["kind"] == "answer"


def test_import_topic_packet_visual_audit_decisions_tracks_seed_bugs_as_fixed(tmp_path: Path) -> None:
    paths = _packet_fixture(tmp_path)
    batch = build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
        seed_bugs=[{"type": "problem", "user_label": "Fixture Q1", "question_id": "q1"}],
    )
    batch_path = paths["review_dir"] / "topic_packet_visual_audit_batch.json"
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(
        "\n".join(json.dumps(_decision(row, status="pass")) for row in batch["rows"]) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "registry.json"

    report = import_topic_packet_visual_audit_decisions(
        batch_path=batch_path,
        decisions_path=decisions_path,
        out_path=out_path,
        markdown_out_path=tmp_path / "registry.md",
    )

    registry = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["complete"] is True
    assert report["bug_record_count"] == 1
    assert registry["bug_records"][0]["resolution_status"] == "fixed"
    assert registry["bug_records"][0]["seed_bug_refs"][0]["question_id"] == "q1"
    assert registry["seed_bug_status"][0]["resolution_status"] == "fixed"


def test_import_topic_packet_visual_audit_decisions_keeps_absent_seed_open_after_complete_import(tmp_path: Path) -> None:
    paths = _packet_fixture(tmp_path)
    batch = build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
        seed_bugs=[
            {"type": "problem", "user_label": "Fixture Q1", "question_id": "q1"},
            {"type": "problem", "user_label": "Missing fixture", "question_id": "missing"},
        ],
    )
    batch_path = paths["review_dir"] / "topic_packet_visual_audit_batch.json"
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(
        "\n".join(json.dumps(_decision(row, status="pass")) for row in batch["rows"]) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "registry.json"

    report = import_topic_packet_visual_audit_decisions(
        batch_path=batch_path,
        decisions_path=decisions_path,
        out_path=out_path,
    )

    registry = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    missing_seed = next(status for status in registry["seed_bug_status"] if status["question_id"] == "missing")
    assert missing_seed["resolution_status"] == "open"
    assert missing_seed["record_count"] == 0
    assert "No rendered page" in missing_seed["rationale"]


def test_import_topic_packet_visual_audit_decisions_supports_mixed_seed_page_resolutions(tmp_path: Path) -> None:
    paths = _packet_fixture(tmp_path)
    batch = build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
        seed_bugs=[
            {"type": "mark_scheme", "user_label": "Fixture Q1 mark scheme", "question_id": "q1"},
            {"type": "mark_scheme", "user_label": "Fixture Q2 mark scheme", "question_id": "q2"},
        ],
    )
    batch_path = paths["review_dir"] / "topic_packet_visual_audit_batch.json"
    decisions = []
    for row in batch["rows"]:
        decision = _decision(row, status="pass")
        if row["page_section"] == "Answers / Mark Schemes":
            decision = _decision(row, status="bug", categories=["mark_scheme_crop"])
            decision["resolution_status"] = "waived_with_reason"
            decision["seed_resolution_status_by_question_id"] = {"q1": "fixed", "q2": "waived_with_reason"}
            decision["rationale"] = "One seed is fixed while the other remains waived on the same rendered page."
        decisions.append(decision)
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text("\n".join(json.dumps(decision) for decision in decisions) + "\n", encoding="utf-8")
    out_path = tmp_path / "registry.json"

    report = import_topic_packet_visual_audit_decisions(
        batch_path=batch_path,
        decisions_path=decisions_path,
        out_path=out_path,
    )

    registry = json.loads(out_path.read_text(encoding="utf-8"))
    statuses = {status["question_id"]: status["resolution_status"] for status in registry["seed_bug_status"]}
    assert report["ok"] is True
    assert statuses == {"q1": "fixed", "q2": "waived_with_reason"}


def test_import_topic_packet_visual_audit_decisions_rejects_duplicate_and_unknown_rows(tmp_path: Path) -> None:
    paths = _packet_fixture(tmp_path)
    batch = build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
        limit_pages=1,
    )
    row = batch["rows"][0]
    batch_path = paths["review_dir"] / "topic_packet_visual_audit_batch.json"
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(
        json.dumps(_decision(row, status="pass"))
        + "\n"
        + json.dumps(_decision(row, status="bug", categories=["question_crop"]))
        + "\n"
        + json.dumps({**_decision(row, status="pass"), "row_id": "unknown_row"})
        + "\n",
        encoding="utf-8",
    )

    report = import_topic_packet_visual_audit_decisions(
        batch_path=batch_path,
        decisions_path=decisions_path,
        out_path=tmp_path / "registry.json",
    )

    assert report["ok"] is False
    assert any("duplicate_decision" in error for error in report["errors"])
    assert any("unknown_row_id" in error for error in report["errors"])


def test_import_topic_packet_visual_audit_decisions_is_fail_closed_when_incomplete(tmp_path: Path) -> None:
    paths = _packet_fixture(tmp_path)
    batch = build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
    )
    batch_path = paths["review_dir"] / "topic_packet_visual_audit_batch.json"
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(json.dumps(_decision(batch["rows"][0], status="pass")) + "\n", encoding="utf-8")

    report = import_topic_packet_visual_audit_decisions(
        batch_path=batch_path,
        decisions_path=decisions_path,
        out_path=tmp_path / "registry.json",
    )

    assert report["ok"] is False
    assert report["complete"] is False
    assert report["missing_count"] == 2
    assert any(error.startswith("missing_decision:") for error in report["errors"])


def test_codex_review_runner_writes_decisions_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    paths = _packet_fixture(tmp_path)
    build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
    )
    codex_bin = _fake_codex_bin(tmp_path)
    decisions_path = tmp_path / "decisions.jsonl"

    report = run_topic_packet_visual_audit_reviews(
        batch_path=paths["review_dir"] / "topic_packet_visual_audit_batch.json",
        out_path=decisions_path,
        max_records=1,
        provider="codex",
        codex_bin=codex_bin,
    )

    decisions = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()]
    assert report["provider"] == "codex"
    assert report["pending_count"] == 1
    assert report["codex_bin"] == str(codex_bin)
    assert decisions[0]["row_id"] == "p1_circular_measure_page_0001"
    assert decisions[0]["status"] == "pass"
    assert (decisions_path.parent / "topic_packet_visual_audit_decision_schema.json").is_file()


def test_openai_review_runner_requires_api_key_only_when_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    paths = _packet_fixture(tmp_path)
    build_topic_packet_visual_audit_batch(
        packets_root=paths["packets_root"],
        artifact_root=paths["artifact_root"],
        render_root=paths["render_root"],
        out_dir=paths["review_dir"],
    )

    with pytest.raises(TopicPacketVisualAuditError, match="OPENAI_API_KEY"):
        run_topic_packet_visual_audit_reviews(
            batch_path=paths["review_dir"] / "topic_packet_visual_audit_batch.json",
            out_path=tmp_path / "decisions.jsonl",
            max_records=1,
            provider="openai",
        )


def test_topic_packet_visual_audit_decision_schema_is_strict_for_codex() -> None:
    from exam_bank.topic_packet_visual_audit import topic_packet_visual_audit_decision_schema

    schema = topic_packet_visual_audit_decision_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(schema["properties"])
    for definition in schema["properties"].values():
        assert "type" in definition
    assert schema["properties"]["decision_version"] == {
        "type": "string",
        "const": TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION,
    }
    assert schema["properties"]["prompt_version"] == {
        "type": "string",
        "const": TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION,
    }
    assert schema["properties"]["status"]["type"] == "string"
    assert schema["properties"]["categories"]["items"]["type"] == "string"
    assert "verification_result" not in schema["properties"]


def _packet_fixture(tmp_path: Path) -> dict[str, Path]:
    packets_root = tmp_path / "packets"
    packet_dir = packets_root / "p1" / "circular_measure"
    packet_dir.mkdir(parents=True)
    artifact_root = tmp_path / "output"
    _png(artifact_root / "pm1" / "q1_question.png", "q1 question")
    _png(artifact_root / "pm1" / "q2_question.png", "q2 question")
    _png(artifact_root / "pm1" / "q1_ms.png", "q1 mark scheme")
    _png(artifact_root / "pm1" / "q2_ms.png", "q2 mark scheme")

    pdf_path = packet_dir / "p1_circular_measure_packet.pdf"
    _pdf(pdf_path, ["Problem 1", "Problem 2", "Answers"])
    manifest = {
        "schema_name": "exam_bank.topic_packets",
        "schema_version": 1,
        "paper_family": "p1",
        "topic_id": "circular_measure",
        "packet_level": "major_topic",
        "pdf_path": str(pdf_path),
        "page_count": 3,
        "page_sections": ["Questions", "Questions", "Answers / Mark Schemes"],
        "questions_section_page_range": [1, 2],
        "answers_section_page_range": [3, 3],
        "oversized_block_warnings": ["oversized_block_scaled_below_legibility:answer:1:scale=0.31"],
        "pdf_outputs": {
            "topic_packet": {
                "path": str(pdf_path),
                "warnings": ["image_downsampled_heavily:q2_ms.png:2000x8000->800x3200"],
            }
        },
        "included_records": [
            _record(1, "q1", "Fixture Q1", question_page=1, answer_page=3),
            _record(2, "q2", "Fixture Q2", question_page=2, answer_page=3),
        ],
    }
    (packet_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "packets_root": packets_root,
        "packet_dir": packet_dir,
        "artifact_root": artifact_root,
        "render_root": tmp_path / "rendered",
        "review_dir": tmp_path / "review",
    }


def _record(problem: int, question_id: str, label: str, *, question_page: int, answer_page: int) -> dict[str, object]:
    return {
        "problem_number": problem,
        "question_id": question_id,
        "source_label": label,
        "question_number": str(problem),
        "section": "review_required",
        "answer_available": True,
        "question_image_paths": [f"pm1/{question_id}_question.png"],
        "mark_scheme_image_paths": [f"pm1/{question_id}_ms.png"],
        "warnings": ["visual_review"],
        "review_reasons": ["visual_review"],
        "question_start_page": question_page,
        "answer_start_page": answer_page,
        "question_block_height_estimate": 120,
        "answer_block_height_estimate": 1800 if question_id == "q1" else 120,
    }


def _decision(row: dict[str, object], *, status: str, categories: list[str] | None = None) -> dict[str, object]:
    return {
        "decision_version": TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION,
        "prompt_version": TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION,
        "row_id": row["row_id"],
        "packet_id": row["packet_id"],
        "page_number": row["page_number"],
        "status": status,
        "categories": [] if categories is None else categories,
        "likely_root_cause": "none",
        "fix_owner_area": "unknown",
        "generalization_decision": "not_applicable",
        "rationale": "fixture decision",
    }


def _png(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 60), label, fill="black")
    image.save(path)


def _pdf(path: Path, labels: list[str]) -> None:
    doc = fitz.open()
    for label in labels:
        page = doc.new_page(width=320, height=240)
        page.insert_text((36, 80), label, fontsize=18)
    doc.save(path)
    doc.close()


def _fake_codex_bin(tmp_path: Path) -> Path:
    path = tmp_path / "codex"
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys

if "--ask-for-approval" in sys.argv:
    print("unexpected approval flag", file=sys.stderr)
    raise SystemExit(2)

out_path = sys.argv[sys.argv.index("--output-last-message") + 1]
decision = {
    "decision_version": "topic_packet_visual_audit_decision_v1",
    "prompt_version": "topic_packet_visual_audit_9709_v1",
    "row_id": "p1_circular_measure_page_0001",
    "packet_id": "p1_circular_measure",
    "page_number": 1,
    "status": "pass",
    "categories": [],
    "likely_root_cause": "none",
    "fix_owner_area": "unknown",
    "generalization_decision": "not_applicable",
    "rationale": "fixture codex decision",
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(decision, handle)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path
