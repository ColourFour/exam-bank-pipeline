from __future__ import annotations

import hashlib
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


def test_major_topic_grouping_ignores_subtopic_and_manifest_shape(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    packet_dir = paths["output"] / "p3" / "integration"
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    assert summary["total_included_in_release_packets"] == 2
    assert summary["total_included"] == 2
    assert summary["total_major_topic_packets_generated"] == 1
    assert manifest["paper_family"] == "p3"
    assert manifest["topic_id"] == "integration"
    assert manifest["packet_level"] == "major_topic"
    assert manifest["subtopic_id"] is None
    assert manifest["packet_mode"] == "release"
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert manifest["problem_count"] == 2
    assert manifest["question_count"] == 2
    assert (packet_dir / "topic_packet.pdf").is_file()
    assert not (packet_dir / "questions.pdf").exists()
    assert not (packet_dir / "answers.pdf").exists()
    assert set(manifest) >= {
        "pdf_path",
        "pdf_file_size_bytes",
        "pdf_profile",
        "page_count",
        "included_records",
        "source_image_paths",
        "source_mark_scheme_image_paths",
        "pdf_image_optimization",
        "pdf_outputs",
        "topic_assignment_source",
        "topic_assignment_confidence_trust_status",
        "warnings",
    }
    assert manifest["included_records"][0]["problem_number"] == 1
    assert manifest["included_records"][0]["source_label"] == "2024 June P31 Question 1"
    assert manifest["pdf_outputs"]["topic_packet"]["file_size_bytes"] > 0


def test_pdf_optimization_does_not_modify_canonical_source_images(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source = paths["artifact_root"] / "p3" / "paper" / "questions" / "q1.png"
    before = _sha256(source)

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        pdf_profile="screen",
        max_image_width=40,
    )

    assert _sha256(source) == before


def test_no_image_optimization_records_previous_embedding_mode(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        image_optimization=False,
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pdf_image_optimization"]["enabled"] is False
    assert manifest["pdf_outputs"]["topic_packet"]["image_count"] == 4


def test_pdf_optimization_profile_recorded_in_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        pdf_profile="screen",
        jpeg_quality=77,
        max_image_width=60,
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pdf_image_optimization"] == {
        "enabled": True,
        "profile": "screen",
        "image_dpi": 144,
        "jpeg_quality": 77,
        "max_image_width": 60,
        "max_image_height": 2400,
    }
    assert summary["packets_generated"][0]["pdf_image_optimization"]["profile"] == "screen"


def test_general_subtopic_record_included_when_major_topic_valid(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, include_q3=True)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert summary["total_included"] == 3
    assert "q3" not in summary["records_with_broad_topic_only_assignment"]
    assert not any(item["question_id"] == "q3" for item in summary["skipped_records"])


def test_mapping_status_fail_excluded_by_default(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"mapping_status": "fail"}}})

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert "q2" in summary["records_with_mapping_failures"]
    assert summary["skipped_by_reason"]["mapping_status_fail"] == 1
    assert summary["total_included_in_release_packets"] == 1


def test_validation_status_fail_excluded_by_default(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"validation_status": "fail"}}})

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert "q2" in summary["records_with_validation_failures"]
    assert summary["skipped_by_reason"]["validation_status_fail"] == 1
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
    assert summary["packets_generated"][0]["pdf_image_optimization"]["profile"] == "print"


@pytest.mark.parametrize(
    ("note_key", "note_value", "warning"),
    [
        ("topic_confidence", "low", "low_topic_confidence"),
        ("topic_uncertain", True, "topic_uncertain"),
        ("topic_trust_status", "degraded_text", "degraded_text"),
        ("question_crop_confidence", "low", "low_question_crop_confidence"),
    ],
)
def test_quality_signals_are_warnings_not_exclusions(
    tmp_path: Path,
    note_key: str,
    note_value: object,
    warning: str,
) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {note_key: note_value}}})

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["total_included"] == 2
    assert summary["warnings_by_type"][warning] == 1
    assert manifest["warning_counts"][warning] == 1


def test_text_and_visual_review_statuses_are_warnings_not_exclusions(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={
            "q1": {"notes": {"visual_curation_status": "review", "text_only_status": "review"}},
            "q2": {"notes": {"text_only_status": "fail"}},
        },
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert summary["total_included"] == 2
    assert summary["warnings_by_type"]["visual_review"] == 1
    assert summary["warnings_by_type"]["text_only_review"] == 1
    assert summary["warnings_by_type"]["text_only_fail"] == 1


def test_invalid_major_topic_excluded_in_strict_syllabus_mode(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"topic": "invented"}})

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        strict_syllabus=True,
    )

    assert summary["total_included"] == 1
    assert "q2" in summary["records_with_invalid_topics"]
    assert summary["skipped_by_reason"]["invalid_major_topic"] == 1


def test_question_and_answer_order_match(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    packet_dir = paths["output"] / "p3" / "integration"
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    packet = fitz.open(packet_dir / "topic_packet.pdf")
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert packet.page_count == 4
    assert "Problem 1" in packet[0].get_text()
    assert "2024 June P31 Question 1" in packet[0].get_text()
    assert "Answer to Problem 1" in packet[1].get_text()
    assert "2024 June P31 Question 1" in packet[1].get_text()
    assert "Problem 2" in packet[2].get_text()
    assert "Answer to Problem 2" in packet[3].get_text()
    assert "Page 1 of 4" in packet[0].get_text()
    assert "Page 4 of 4" in packet[3].get_text()
    packet.close()


def test_missing_mark_scheme_reported_without_breaking_question_pdf(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, missing_answer=True)

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    packet_dir = paths["output"] / "p3" / "integration"
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    packet = fitz.open(packet_dir / "topic_packet.pdf")
    assert "q2" in summary["missing_mark_scheme_images"]
    assert "q2" in summary["records_with_missing_mark_schemes"]
    assert manifest["missing_answer_ids"] == ["q2"]
    assert manifest["missing_answer_count"] == 1
    assert manifest["included_records"][1]["answer_available"] is False
    assert "missing_mark_scheme_image" in manifest["included_records"][1]["warnings"]
    assert packet.page_count == 4
    assert "Answer unavailable: missing mark-scheme image" in packet[3].get_text()
    packet.close()


def test_source_label_maps_compact_seasons(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={
            "q1": {"paper": "11spring23", "notes": {"source_paper_code": "11"}},
            "q2": {"paper": "52autumn21", "question_number": "7", "notes": {"source_paper_code": "52"}},
        },
    )

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    labels = {item["question_id"]: item["source_label"] for item in manifest["included_records"]}
    assert labels["q1"] == "2023 March P11 Question 1"
    assert labels["q2"] == "2021 November P52 Question 7"


def test_multiple_question_and_mark_scheme_images_do_not_increment_problem_number(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={
            "q1": {
                "question_image_paths": [
                    "p3/paper/questions/q1.png",
                    "p3/paper/questions/q1_cont.png",
                ],
                "question_image_path": "",
                "mark_scheme_image_paths": [
                    "p3/paper/mark_scheme/q1.png",
                    "p3/paper/mark_scheme/q1_cont.png",
                ],
                "mark_scheme_image_path": "",
            }
        },
    )
    _png(paths["artifact_root"] / "p3" / "paper" / "questions" / "q1_cont.png")
    _png(paths["artifact_root"] / "p3" / "paper" / "mark_scheme" / "q1_cont.png")

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    packet = fitz.open(paths["output"] / "p3" / "integration" / "topic_packet.pdf")
    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert [item["problem_number"] for item in manifest["included_records"]] == [1, 2]
    assert packet.page_count == 6
    assert "Problem 1" in packet[0].get_text()
    assert "Problem 1" in packet[1].get_text()
    assert "Answer to Problem 1" in packet[2].get_text()
    assert "Answer to Problem 1" in packet[3].get_text()
    assert "Problem 2" in packet[4].get_text()
    packet.close()


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
    record_overrides: dict[str, dict[str, object]] | None = None,
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
    if record_overrides:
        for record in questions:
            override = record_overrides.get(str(record["question_id"]))
            if override:
                _deep_update(record, override)

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


def _record(question_id: str, *, topic: str = "integration", answer: bool = True) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "31summer24",
        "paper_family": "p3",
        "question_number": question_id[-1],
        "question_solution_marks": 4,
        "topic": topic,
        "question_image_path": f"p3/paper/questions/{question_id}.png",
        "mark_scheme_image_path": f"p3/paper/mark_scheme/{question_id}.png" if answer else "",
        "notes": {"source_pdf": "qp.pdf", "subtopic": "general"},
    }


def _deep_update(target: dict[str, object], update: dict[str, object]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            nested = target[key]
            assert isinstance(nested, dict)
            nested.update(value)
        else:
            target[key] = value


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
