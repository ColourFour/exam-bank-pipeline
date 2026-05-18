from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import fitz
import pytest
from PIL import Image

from exam_bank.topic_packets import (
    PacketKey,
    TopicPacketError,
    generate_topic_packets,
    load_packet_taxonomy,
    validate_packet_key,
)


ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_loading_and_validation(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    loaded = load_packet_taxonomy(taxonomy)

    assert validate_packet_key(PacketKey("release", "p3", "integration", "integration_by_parts"), loaded)
    assert not validate_packet_key(PacketKey("release", "p3", "integration", "invented"), loaded)


def test_invalid_topic_subtopic_ids_rejected_in_strict_mode(tmp_path: Path) -> None:
    taxonomy = load_packet_taxonomy(_write_taxonomy(tmp_path))

    with pytest.raises(TopicPacketError):
        if not validate_packet_key(PacketKey("release", "p3", "invented", "integration_by_parts"), taxonomy):
            raise TopicPacketError("Packet key outside allowed taxonomy")


def test_grouping_by_paper_family_topic_subtopic_and_manifest_shape(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    packet_dir = paths["output"] / "p3" / "integration" / "integration_by_parts"
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    assert summary["total_included_in_release_packets"] == 2
    assert manifest["paper_family"] == "p3"
    assert manifest["topic_id"] == "integration"
    assert manifest["subtopic_id"] == "integration_by_parts"
    assert manifest["packet_mode"] == "release"
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert manifest["question_count"] == 2
    assert set(manifest) >= {
        "source_image_paths",
        "source_mark_scheme_image_paths",
        "topic_assignment_source",
        "topic_assignment_confidence_trust_status",
        "warnings",
    }


def test_broad_topic_only_records_follow_configured_mode(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, include_q3=True)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert "q3" in summary["records_with_broad_topic_only_assignment"]
    assert any(item["question_id"] == "q3" and item["reason"] == "broad_topic_only" for item in summary["skipped_records"])


def test_unsafe_topic_records_excluded_from_release_packets(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, ambiguous=True)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert "q2" in summary["records_with_unsafe_topic_assignment"]
    assert summary["total_included_in_release_packets"] == 1


def test_dry_run_writes_no_pdfs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert not list(paths["output"].glob("**/*.pdf"))
    assert not (paths["output"] / "topic_packet_summary.json").exists()


def test_question_and_answer_order_match(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    packet_dir = paths["output"] / "p3" / "integration" / "integration_by_parts"
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    questions = fitz.open(packet_dir / "questions.pdf")
    answers = fitz.open(packet_dir / "answers.pdf")
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert questions.page_count == 2
    assert answers.page_count == 2
    assert "q1" in questions[0].get_text()
    assert "q2" in questions[1].get_text()
    assert "q1" in answers[0].get_text()
    assert "q2" in answers[1].get_text()
    answers.close()
    questions.close()


def test_missing_mark_scheme_reported_without_breaking_question_pdf(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, missing_answer=True)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    packet_dir = paths["output"] / "p3" / "integration" / "integration_by_parts"
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    questions = fitz.open(packet_dir / "questions.pdf")
    answers = fitz.open(packet_dir / "answers.pdf")
    assert "q2" in summary["missing_mark_scheme_images"]
    assert manifest["missing_answer_ids"] == ["q2"]
    assert questions.page_count == 2
    assert answers.page_count == 1
    answers.close()
    questions.close()


def test_cli_help_exits_cleanly() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "exam_bank.cli", "topic-packets", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _fixture(
    tmp_path: Path,
    *,
    include_q3: bool = False,
    ambiguous: bool = False,
    missing_answer: bool = False,
) -> dict[str, Path]:
    taxonomy = _write_taxonomy(tmp_path)
    artifact_root = tmp_path / "output"
    for name in ["q1", "q2", "q3"]:
        _png(artifact_root / "p3" / "paper" / "questions" / f"{name}.png")
        if not (missing_answer and name == "q2"):
            _png(artifact_root / "p3" / "paper" / "mark_scheme" / f"{name}.png")

    questions = [_record("q1"), _record("q2")]
    if include_q3:
        questions.append(_record("q3", topic="integration", answer=True))

    bank = tmp_path / "question_bank.json"
    bank.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.question_bank",
                "schema_version": 2,
                "run_manifest": {"artifact_root": str(artifact_root)},
                "questions": questions,
            }
        ),
        encoding="utf-8",
    )

    canonical_root = tmp_path / "canonical"
    assignment_dir = canonical_root / "question_topic_assignments"
    assignment_dir.mkdir(parents=True)
    assignments = [_assignment("q1"), _assignment("q2")]
    if ambiguous:
        assignments.append(_assignment("q2", subtopic="9709_p3_subtopic_standard_integration"))
    (assignment_dir / "question_topic_assignments_9709_p3_v1.json").write_text(
        json.dumps({"assignments": assignments}),
        encoding="utf-8",
    )
    return {
        "taxonomy": taxonomy,
        "artifact_root": artifact_root,
        "bank": bank,
        "canonical_root": canonical_root,
        "output": tmp_path / "packets",
    }


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
                                "subtopics": [
                                    {
                                        "subtopic_id": "integration_by_parts",
                                        "subtopic_label": "Integration by parts",
                                        "canonical_subtopic_id": "9709_p3_subtopic_integration_by_parts",
                                        "packet_eligible": True,
                                    },
                                    {
                                        "subtopic_id": "standard_integration",
                                        "subtopic_label": "Standard integration",
                                        "canonical_subtopic_id": "9709_p3_subtopic_standard_integration",
                                        "packet_eligible": True,
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _record(question_id: str, *, topic: str = "unknown", answer: bool = True) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "31summer24",
        "paper_family": "p3",
        "question_number": question_id[-1],
        "question_solution_marks": 4,
        "topic": topic,
        "question_image_path": f"p3/paper/questions/{question_id}.png",
        "mark_scheme_image_path": f"p3/paper/mark_scheme/{question_id}.png" if answer else "",
        "notes": {"source_pdf": "qp.pdf"},
    }


def _assignment(question_id: str, *, subtopic: str = "9709_p3_subtopic_integration_by_parts") -> dict[str, object]:
    return {
        "question_id": question_id,
        "topic_assignments": [
            {
                "topic_id": "9709_p3_topic_integration",
                "topic_name": "Integration",
                "subtopic_id": subtopic,
                "subtopic_name": "Integration by parts",
                "assignment_type": "primary_assessed",
                "confidence": 0.91,
                "strict_filter_eligible": True,
                "review_status": "high-confidence machine_candidate",
            }
        ],
    }


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color=(255, 255, 255)).save(path)
