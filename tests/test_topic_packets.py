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
    choose_major_topic_assignment,
    generate_topic_packets,
    load_packet_taxonomy,
    normalize_packet_topic,
    resolve_answer_image_paths,
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
    assert manifest["packet_mode"] == "combined"
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert manifest["approved_question_ids"] == ["q1", "q2"]
    assert manifest["review_required_question_ids"] == []
    assert manifest["approved_count"] == 2
    assert manifest["review_required_count"] == 0
    assert manifest["total_questions"] == 2
    assert manifest["problem_count"] == 2
    assert manifest["question_count"] == 2
    packet_pdf = packet_dir / "p3_integration_packet.pdf"
    assert packet_pdf.is_file()
    assert not (packet_dir / "topic_packet.pdf").exists()
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
    assert manifest["pdf_path"] == str(packet_pdf)
    assert manifest["pdf_outputs"]["topic_packet"]["path"] == str(packet_pdf)
    assert manifest["pdf_outputs"]["topic_packet"]["file_size_bytes"] > 0
    assert summary["packets_generated"][0]["pdf_path"] == str(packet_pdf)
    assert summary["generated_pdfs"] == [str(packet_pdf)]


def test_legacy_question_bank_family_aliases_are_packet_eligible(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={
            "q1": {"paper_family": "pm3"},
            "q2": {"paper_family": "pm3"},
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
    assert summary["skipped_by_reason"].get("invalid_paper_family", 0) == 0
    assert (paths["output"] / "p3" / "integration" / "p3_integration_packet.pdf").is_file()


@pytest.mark.parametrize(
    ("question_id", "component", "stored_family", "raw_topic", "expected_family", "expected_topic"),
    [
        ("04summer08_q01", "04", "stats", "forces_newtons_second_law", "p4", "newtons_laws_of_motion"),
        ("04summer08_q02", "04", "stats", "work_energy_power", "p4", "energy_work_and_power"),
        ("04winter08_q02", "04", "stats", "friction_rough_plane", "p4", "forces_and_equilibrium"),
        ("41winter09_q01", "41", "stats", "work_energy_power", "p4", "energy_work_and_power"),
        ("06summer09_q02", "06", "stats", "probability", "p5", "probability"),
        ("06summer08_q04", "06", "stats", "normal_distribution", "p5", "the_normal_distribution"),
        ("61winter09_q03", "61", "stats", "normal_distribution", "p5", "the_normal_distribution"),
        ("41winter20_q06", "41", "stats", "connected_particles_energy", "p4", "energy_work_and_power"),
    ],
)
def test_packet_topic_normalization_uses_component_before_routing(
    question_id: str,
    component: str,
    stored_family: str,
    raw_topic: str,
    expected_family: str,
    expected_topic: str,
) -> None:
    taxonomy = load_packet_taxonomy(ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json")

    normalization = normalize_packet_topic(
        component_code=component,
        current_family=stored_family,
        raw_topic=raw_topic,
        taxonomy=taxonomy,
    )

    assert question_id
    assert normalization.status == "resolved"
    assert normalization.expected_family == expected_family
    assert normalization.expected_topic == expected_topic


def test_normal_distribution_alias_resolves_only_to_p5() -> None:
    taxonomy = load_packet_taxonomy(ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json")

    for component, stored_family in [("04", "p4"), ("11", "p1"), ("31", "p3"), ("61", "stats")]:
        normalization = normalize_packet_topic(
            component_code=component,
            current_family=stored_family,
            raw_topic="normal_distribution",
            taxonomy=taxonomy,
        )

        assert normalization.status == "resolved"
        assert normalization.expected_family == "p5"
        assert normalization.expected_topic == "the_normal_distribution"


def test_major_topic_assignment_uses_normalized_packet_family_for_audit_sample() -> None:
    taxonomy = load_packet_taxonomy(ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json")

    assignment, reasons = choose_major_topic_assignment(
        {
            "question_id": "04summer08_q01",
            "paper": "04summer08",
            "paper_family": "stats",
            "topic": "forces_newtons_second_law",
            "notes": {"source_paper_code": "04"},
        },
        taxonomy,
    )

    assert reasons == []
    assert assignment is not None
    assert assignment.paper_family == "p4"
    assert assignment.topic_id == "newtons_laws_of_motion"


def test_connected_particles_energy_resolves_to_energy_work_and_power() -> None:
    taxonomy = load_packet_taxonomy(ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json")

    normalization = normalize_packet_topic(
        component_code="41",
        current_family="stats",
        raw_topic="connected_particles_energy",
        taxonomy=taxonomy,
    )

    assert normalization.status == "resolved"
    assert normalization.expected_family == "p4"
    assert normalization.expected_topic == "energy_work_and_power"


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


def test_answer_image_resolver_uses_top_level_mark_scheme_image_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    _png(artifact_root / "p3" / "paper" / "mark_scheme" / "q1a.png")
    _png(artifact_root / "p3" / "paper" / "mark_scheme" / "q1b.png")
    record = _record(
        "q1",
        answer=False,
    )
    record["mark_scheme_image_paths"] = [
        "p3/paper/mark_scheme/q1a.png",
        "p3/paper/mark_scheme/q1b.png",
    ]

    assert resolve_answer_image_paths(record, artifact_root=artifact_root, fallback_root=tmp_path) == [
        "p3/paper/mark_scheme/q1a.png",
        "p3/paper/mark_scheme/q1b.png",
    ]


def test_answer_image_resolver_uses_top_level_mark_scheme_image_path(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    _png(artifact_root / "p3" / "paper" / "mark_scheme" / "q1.png")
    record = _record("q1")

    assert resolve_answer_image_paths(record, artifact_root=artifact_root, fallback_root=tmp_path) == [
        "p3/paper/mark_scheme/q1.png"
    ]


def test_answer_image_resolver_uses_nested_canonical_path_when_top_level_empty(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    _png(artifact_root / "pm3" / "pm3_2024_m24_33_ms_q01_markscheme.png")
    record = _record("q1", answer=False)
    record["notes"]["mark_scheme_structure_detected"] = {
        "asset_identity": {"canonical_path": "pm3/pm3_2024_m24_33_ms_q01_markscheme.png"}
    }

    assert resolve_answer_image_paths(record, artifact_root=artifact_root, fallback_root=tmp_path) == [
        "pm3/pm3_2024_m24_33_ms_q01_markscheme.png"
    ]


def test_answer_image_resolver_ignores_missing_nested_canonical_path(tmp_path: Path) -> None:
    record = _record("q1", answer=False)
    record["notes"]["mark_scheme_structure_detected"] = {
        "asset_identity": {"canonical_path": "pm3/missing_markscheme.png"}
    }

    assert resolve_answer_image_paths(record, artifact_root=tmp_path / "output", fallback_root=tmp_path) == []


def test_answer_image_resolver_does_not_fuzzy_match_other_question_ids(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    _png(artifact_root / "pm3" / "pm3_2024_m24_33_ms_q02_markscheme.png")
    record = _record("q1", answer=False)

    assert resolve_answer_image_paths(record, artifact_root=artifact_root, fallback_root=tmp_path) == []


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


def test_reviewed_keep_preserves_existing_topic_label(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "q2",
                "action": "keep",
                "reviewed_topic": "",
                "reviewed_subtopic": "",
                "reason": "Confirmed generated topic.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    q2 = manifest["included_records"][1]
    assert summary["reviewed_decisions_loaded"] == 1
    assert summary["reviewed_decision_counts"]["keep"] == 1
    assert manifest["topic_id"] == "integration"
    assert q2["question_id"] == "q2"
    assert q2["review_status_marker"] == "Reviewed"
    assert q2["review_decision_action"] == "keep"


def test_reviewed_relabel_changes_topic_used_in_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "q2",
                "action": "relabel",
                "reviewed_topic": "vectors",
                "reviewed_subtopic": "vector_lines",
                "reason": "Synthetic relabel fixture.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    integration_manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    vectors_manifest = json.loads((paths["output"] / "p3" / "vectors" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["reviewed_decision_counts"]["relabel"] == 1
    assert integration_manifest["included_question_ids"] == ["q1"]
    assert vectors_manifest["included_question_ids"] == ["q2"]
    assert vectors_manifest["included_records"][0]["review_status_marker"] == "Relabeled"
    assert vectors_manifest["topic_assignment_source"]["reviewed_topic_bank_decision"] == 1


def test_reviewed_relabel_accepts_subtopic_label(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "q2",
                "action": "relabel",
                "reviewed_topic": "vectors",
                "reviewed_subtopic": "Vector lines",
                "reason": "Synthetic relabel fixture.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    vectors_manifest = json.loads((paths["output"] / "p3" / "vectors" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["reviewed_decision_counts"]["relabel"] == 1
    assert vectors_manifest["included_question_ids"] == ["q2"]
    assert vectors_manifest["included_records"][0]["reviewed_subtopic"] == "Vector lines"


def test_reviewed_relabel_overrides_invalid_generated_major_topic(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"topic": "invented"}})
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "q2",
                "action": "relabel",
                "reviewed_topic": "vectors",
                "reviewed_subtopic": "vector_lines",
                "reason": "Question belongs to Vectors, not the invalid generated topic.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
        strict_syllabus=True,
    )

    vectors_manifest = json.loads((paths["output"] / "p3" / "vectors" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["total_included"] == 2
    assert summary["skipped_by_reason"] == {}
    assert summary["reviewed_decision_counts"]["relabel"] == 1
    assert vectors_manifest["included_question_ids"] == ["q2"]
    assert vectors_manifest["included_records"][0]["review_status_marker"] == "Relabeled"


def test_automated_relabel_uses_source_paper_family_for_legacy_misroute(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={
            "q2": {
                "paper": "11winter16",
                "paper_family": "pm1",
                "topic": "vectors",
                "notes": {"source_paper_code": "11"},
            }
        },
    )
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            _automated_decision(
                "q2",
                action="relabel",
                reviewed_topic="quadratics",
                release_override=True,
            )
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
        strict_syllabus=True,
    )

    quadratics_manifest = json.loads((paths["output"] / "p1" / "quadratics" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["reviewed_decision_counts"]["relabel"] == 1
    assert summary["applied_reviewed_decision_counts"]["relabel"] == 1
    assert quadratics_manifest["paper_family"] == "p1"
    assert quadratics_manifest["included_question_ids"] == ["q2"]
    q2 = quadratics_manifest["included_records"][0]
    assert q2["review_status_marker"] == "Relabeled"
    assert q2["reviewed_topic"] == "quadratics"
    assert q2["release_override"] is True


def test_reviewed_exclude_removes_question_from_student_facing_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "q2",
                "action": "exclude",
                "reviewed_topic": "",
                "reviewed_subtopic": "",
                "reason": "Not suitable for student-facing topic bank.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["total_included_in_release_packets"] == 1
    assert summary["total_included_in_review_packets"] == 0
    assert summary["total_excluded_questions"] == 1
    assert summary["skipped_by_reason"]["topic_bank_reviewed_exclude"] == 1
    assert manifest["included_question_ids"] == ["q1"]


def test_reviewed_decision_unknown_question_id_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "unknown_q99",
                "action": "keep",
                "reviewed_topic": "",
                "reviewed_subtopic": "",
                "reason": "Bad fixture.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    with pytest.raises(TopicPacketError, match="unknown question_id"):
        generate_topic_packets(
            question_bank_path=paths["bank"],
            taxonomy_path=paths["taxonomy"],
            canonical_taxonomy_root=paths["canonical_root"],
            output_root=paths["output"],
            artifact_root=paths["artifact_root"],
            reviewed_decisions_path=decisions,
        )


def test_reviewed_relabel_unknown_topic_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "q2",
                "action": "relabel",
                "reviewed_topic": "invented",
                "reviewed_subtopic": "",
                "reason": "Bad fixture.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    with pytest.raises(TopicPacketError, match="unknown topic"):
        generate_topic_packets(
            question_bank_path=paths["bank"],
            taxonomy_path=paths["taxonomy"],
            canonical_taxonomy_root=paths["canonical_root"],
            output_root=paths["output"],
            artifact_root=paths["artifact_root"],
            reviewed_decisions_path=decisions,
        )


def test_generation_works_with_empty_reviewed_decisions_file(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decisions = _write_reviewed_decisions(tmp_path, [])

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["reviewed_decisions_loaded"] == 0
    assert summary["total_included_in_release_packets"] == 2
    assert manifest["included_question_ids"] == ["q1", "q2"]


def test_topic_overlap_sidecar_relabels_primary_and_counts_secondary_coverage(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    overlap = _write_topic_overlap_review(
        tmp_path,
        [
            {
                "question_id": "q2",
                "paper": "31summer24",
                "primary": "vectors",
                "secondary": ["integration"],
                "rationale": "Synthetic mixed-topic question.",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        topic_overlap_review_path=overlap,
    )

    integration_manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    vectors_manifest = json.loads((paths["output"] / "p3" / "vectors" / "manifest.json").read_text(encoding="utf-8"))
    q2 = vectors_manifest["included_records"][0]
    assert integration_manifest["included_question_ids"] == ["q1"]
    assert vectors_manifest["included_question_ids"] == ["q2"]
    assert q2["primary_topic_id"] == "vectors"
    assert q2["secondary_topic_ids"] == ["integration"]
    assert q2["coverage_topic_ids"] == ["vectors", "integration"]
    assert vectors_manifest["topic_placement_count"] == 2
    assert vectors_manifest["coverage_topic_counts"] == {"integration": 1, "vectors": 1}
    assert summary["per_topic_included_counts"] == {"p3/integration": 1, "p3/vectors": 1}
    assert summary["per_topic_coverage_counts"] == {"p3/integration": 2, "p3/vectors": 1}
    assert summary["total_unique_questions_included"] == 2
    assert summary["total_topic_coverage_placements"] == 3
    assert summary["topic_overlap_reviews_loaded"] == 1
    assert summary["topic_overlap_reviews_applied"] == 1


def test_topic_overlap_family_override_participates_in_paper_family_filter(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={
            "q1": {
                "paper": "11summer24",
                "paper_family": "pm1",
                "topic": "integration",
                "notes": {"source_paper_code": "11"},
            }
        },
    )
    overlap = _write_topic_overlap_review(
        tmp_path,
        [
            {
                "question_id": "q1",
                "paper": "11summer24",
                "primary": "quadratics",
                "rationale": "Synthetic P1 rescue for a raw topic that otherwise routes outside P1.",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        topic_overlap_review_path=overlap,
        paper_family="p1",
    )

    manifest = json.loads((paths["output"] / "p1" / "quadratics" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_question_ids"] == ["q1"]
    assert manifest["included_records"][0]["primary_topic_id"] == "quadratics"
    assert summary["per_topic_included_counts"] == {"p1/quadratics": 1}
    assert summary["topic_overlap_reviews_loaded"] == 1
    assert summary["topic_overlap_reviews_applied"] == 1


def test_topic_overlap_sidecar_restored_split_question_counts_as_coverage_only(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    overlap = _write_topic_overlap_review(
        tmp_path,
        [
            {
                "question_id": "31summer24_q03",
                "paper": "31summer24",
                "source_paper_code": "31",
                "status": "restore_split_question",
                "primary": "vectors",
                "rationale": "Question is visible in a neighboring crop but not extracted as a record.",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        topic_overlap_review_path=overlap,
    )

    assert summary["total_included"] == 2
    assert summary["topic_overlap_coverage_only_count"] == 1
    assert summary["topic_overlap_coverage_only_records"][0]["question_id"] == "31summer24_q03"
    assert summary["per_topic_included_counts"] == {"p3/integration": 2}
    assert summary["per_topic_coverage_counts"] == {"p3/integration": 2, "p3/vectors": 1}
    audit_row = summary["paper_topic_coverage_audit"]["papers"][0]
    assert audit_row["paper"] == "31summer24"
    assert audit_row["unique_question_count"] == 3
    assert audit_row["topic_coverage_counts"]["integration"] == 2
    assert audit_row["topic_coverage_counts"]["vectors"] == 1


def test_reviewed_keep_overrides_release_quality_downgrade(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"visual_curation_status": "review"}}})
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "q2",
                "action": "keep",
                "reviewed_topic": "integration",
                "reviewed_subtopic": "",
                "reason": "Correctly labeled after manual review.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    release_manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["total_included_in_release_packets"] == 2
    assert summary["total_included_in_review_packets"] == 0
    assert release_manifest["included_question_ids"] == ["q1", "q2"]
    assert release_manifest["included_records"][1]["review_status_marker"] == "Reviewed"


def test_reviewed_keep_overrides_mapping_and_validation_failures(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={"q2": {"notes": {"mapping_status": "fail", "validation_status": "fail"}}},
    )
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            {
                "question_id": "q2",
                "action": "keep",
                "reviewed_topic": "integration",
                "reviewed_subtopic": "",
                "reason": "Correctly labeled after manual review.",
                "reviewer": "test",
                "reviewed_at": "2026-06-12T00:00:00Z",
                "source": "manual_review",
            }
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["skipped_by_reason"].get("mapping_status_fail", 0) == 0
    assert summary["skipped_by_reason"].get("validation_status_fail", 0) == 0
    assert manifest["included_question_ids"] == ["q1", "q2"]


def test_automated_reviewed_keep_without_release_override_stays_review_required(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"visual_curation_status": "review"}}})
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            _automated_decision(
                "q2",
                action="keep",
                reviewed_topic="integration",
                release_override=False,
            )
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    q2 = next(record for record in manifest["included_records"] if record["question_id"] == "q2")
    assert summary["reviewed_decision_counts"]["keep"] == 1
    assert summary["total_included_in_release_packets"] == 1
    assert summary["total_included_in_review_packets"] == 1
    assert q2["section"] == "review_required"
    assert q2["review_decision_source"] == "automated_agentic_review"
    assert q2["current_syllabus_status"] == "current_relevant"
    assert q2["release_override"] is False


def test_automated_reviewed_keep_with_release_override_overrides_quality_downgrade(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"visual_curation_status": "review"}}})
    decisions = _write_reviewed_decisions(
        tmp_path,
        [
            _automated_decision(
                "q2",
                action="keep",
                reviewed_topic="integration",
                release_override=True,
            )
        ],
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        reviewed_decisions_path=decisions,
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    q2 = next(record for record in manifest["included_records"] if record["question_id"] == "q2")
    assert summary["total_included_in_release_packets"] == 2
    assert summary["total_included_in_review_packets"] == 0
    assert q2["section"] == "approved"
    assert q2["release_override"] is True


def test_automated_reviewed_decision_low_confidence_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decision = _automated_decision("q2", action="keep", reviewed_topic="integration", release_override=True)
    decision["confidence"] = 0.89
    decisions = _write_reviewed_decisions(tmp_path, [decision])

    with pytest.raises(TopicPacketError, match="confidence >= 0.90"):
        generate_topic_packets(
            question_bank_path=paths["bank"],
            taxonomy_path=paths["taxonomy"],
            canonical_taxonomy_root=paths["canonical_root"],
            output_root=paths["output"],
            artifact_root=paths["artifact_root"],
            reviewed_decisions_path=decisions,
        )


def test_mapping_status_fail_enters_review_required_section_by_default(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"mapping_status": "fail"}}})

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert "q2" in summary["records_with_mapping_failures"]
    assert summary["skipped_by_reason"].get("mapping_status_fail", 0) == 0
    assert summary["total_included_in_release_packets"] == 1
    assert summary["total_included_in_review_packets"] == 1
    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert manifest["approved_question_ids"] == ["q1"]
    assert manifest["review_required_question_ids"] == ["q2"]
    assert "mapping_status_not_pass" in manifest["review_required_reasons_by_question_id"]["q2"]


def test_validation_status_fail_enters_review_required_section_by_default(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"validation_status": "fail"}}})

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert "q2" in summary["records_with_validation_failures"]
    assert summary["skipped_by_reason"].get("validation_status_fail", 0) == 0
    assert summary["total_included_in_release_packets"] == 1
    assert summary["total_included_in_review_packets"] == 1
    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert manifest["approved_question_ids"] == ["q1"]
    assert manifest["review_required_question_ids"] == ["q2"]
    assert "validation_status_not_pass" in manifest["review_required_reasons_by_question_id"]["q2"]


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


def test_low_crop_confidence_downgrades_release_record_to_review_section(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"question_crop_confidence": "low"}}})

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    packet = fitz.open(paths["output"] / "p3" / "integration" / "p3_integration_packet.pdf")
    text = "\n".join(page.get_text() for page in packet)
    assert summary["total_included"] == 2
    assert summary["total_included_in_release_packets"] == 1
    assert summary["total_included_in_review_packets"] == 1
    assert summary["total_approved_questions"] == 1
    assert summary["total_review_required_questions"] == 1
    assert summary["release_quality_downgrade_count"] == 1
    assert summary["release_quality_downgrade_reason_counts"]["question_crop_confidence_not_high"] == 1
    assert manifest["packet_mode"] == "combined"
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert manifest["approved_question_ids"] == ["q1"]
    assert manifest["review_required_question_ids"] == ["q2"]
    assert manifest["included_records"][0]["section"] == "approved"
    assert manifest["included_records"][1]["section"] == "review_required"
    assert "question_crop_confidence_not_high" in manifest["included_records"][1]["review_reasons"]
    assert "Approved Questions" in text
    assert "Review Required Questions" in text
    assert "Review Required - verify topic/routing before classroom use." in text
    assert text.index("Approved Questions") < text.index("Review Required Questions")
    assert not (paths["output"] / "review_required" / "p3" / "integration" / "manifest.json").exists()
    packet.close()


def test_visual_review_and_text_fail_downgrade_to_review_section(tmp_path: Path) -> None:
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
    assert summary["total_included_in_release_packets"] == 0
    assert summary["total_included_in_review_packets"] == 2
    assert summary["release_quality_downgrade_count"] == 2
    assert summary["warnings_by_type"]["visual_review"] == 1
    assert summary["warnings_by_type"]["text_only_review"] == 1
    assert summary["warnings_by_type"]["text_only_fail"] == 1
    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["approved_question_ids"] == []
    assert manifest["review_required_question_ids"] == ["q1", "q2"]
    assert "visual_curation_status_not_ready" in manifest["included_records"][0]["review_reasons"]
    assert "text_only_status_fail" in manifest["included_records"][1]["review_reasons"]


def test_full_generation_replaces_stale_release_packet_when_all_records_downgrade(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={
            "q1": {"notes": {"visual_curation_status": "review"}},
            "q2": {"notes": {"visual_curation_status": "review"}},
        },
    )
    stale_dir = paths["output"] / "p3" / "integration"
    stale_dir.mkdir(parents=True)
    (stale_dir / "manifest.json").write_text(
        json.dumps({"packet_mode": "release", "included_records": [{"question_id": "stale"}]}),
        encoding="utf-8",
    )

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    assert summary["total_included_in_release_packets"] == 0
    assert summary["total_included_in_review_packets"] == 2
    manifest = json.loads((stale_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert manifest["approved_count"] == 0
    assert manifest["review_required_count"] == 2


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
    packet = fitz.open(packet_dir / "p3_integration_packet.pdf")
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert packet.page_count == 2
    assert manifest["page_size"] == "a4"
    assert manifest["orientation"] == "portrait"
    assert manifest["layout"] == "compact"
    assert manifest["answer_placement"] == "end"
    assert manifest["questions_section_page_range"] == [1, 1]
    assert manifest["answers_section_page_range"] == [2, 2]
    assert manifest["problems_per_page_summary"]["questions"]["1"] == 2
    assert manifest["problems_per_page_summary"]["answers"]["2"] == 2
    assert "Problem 1" in packet[0].get_text()
    assert "2024 June P31 Question 1" in packet[0].get_text()
    assert "Problem 2" in packet[0].get_text()
    assert "Answer to Problem 1" not in packet[0].get_text()
    assert "Answer to Problem 1" in packet[1].get_text()
    assert "2024 June P31 Question 1" in packet[1].get_text()
    assert "Answer to Problem 2" in packet[1].get_text()
    assert "Page 1 of 2" in packet[0].get_text()
    assert "Page 2 of 2" in packet[1].get_text()
    assert manifest["included_records"][0]["question_start_page"] == 1
    assert manifest["included_records"][0]["answer_start_page"] == 2
    packet.close()


def test_missing_mark_scheme_excluded_from_packet(tmp_path: Path) -> None:
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
    packet = fitz.open(packet_dir / "p3_integration_packet.pdf")
    assert "q2" in summary["missing_mark_scheme_images"]
    assert "q2" in summary["records_with_missing_mark_schemes"]
    assert summary["skipped_by_reason"]["missing_answer_image"] == 1
    assert manifest["included_question_ids"] == ["q1"]
    assert manifest["missing_answer_ids"] == []
    assert manifest["missing_answer_count"] == 0
    assert packet.page_count == 2
    assert "Answer to Problem 2" not in packet[1].get_text()
    packet.close()


def test_packet_builder_includes_record_with_nested_mark_scheme_canonical_path(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, missing_answer=True)
    _png(paths["artifact_root"] / "pm3" / "pm3_2024_m24_31_ms_q02_markscheme.png")
    question_bank = json.loads(paths["bank"].read_text(encoding="utf-8"))
    for record in question_bank["questions"]:
        if record["question_id"] == "q2":
            record["notes"]["mark_scheme_structure_detected"] = {
                "asset_identity": {"canonical_path": "pm3/pm3_2024_m24_31_ms_q02_markscheme.png"}
            }
    paths["bank"].write_text(json.dumps(question_bank), encoding="utf-8")

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    packet_dir = paths["output"] / "p3" / "integration"
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    packet = fitz.open(packet_dir / "p3_integration_packet.pdf")
    assert summary["total_included"] == 2
    assert summary["skipped_by_reason"].get("missing_answer_image", 0) == 0
    assert manifest["included_question_ids"] == ["q1", "q2"]
    assert manifest["included_records"][1]["mark_scheme_image_paths"] == [
        "pm3/pm3_2024_m24_31_ms_q02_markscheme.png"
    ]
    assert "Answer to Problem 2" in packet[1].get_text()
    packet.close()


def test_packet_builder_excludes_record_with_no_readable_answer_path(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, missing_answer=True)
    question_bank = json.loads(paths["bank"].read_text(encoding="utf-8"))
    for record in question_bank["questions"]:
        if record["question_id"] == "q2":
            record["notes"]["mark_scheme_structure_detected"] = {
                "asset_identity": {"canonical_path": "pm3/missing_markscheme.png"}
            }
    paths["bank"].write_text(json.dumps(question_bank), encoding="utf-8")

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["skipped_by_reason"]["missing_answer_image"] == 1
    assert manifest["included_question_ids"] == ["q1"]


def test_source_label_maps_compact_seasons(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        record_overrides={
            "q1": {"paper": "11spring23", "notes": {"source_paper_code": "11"}},
            "q2": {"paper": "52winter21", "question_number": "7", "notes": {"source_paper_code": "52"}},
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

    packet = fitz.open(paths["output"] / "p3" / "integration" / "p3_integration_packet.pdf")
    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert [item["problem_number"] for item in manifest["included_records"]] == [1, 2]
    assert packet.page_count == 2
    assert "Problem 1" in packet[0].get_text()
    assert "Problem 2" in packet[0].get_text()
    assert "Problem 3" not in packet[0].get_text()
    assert "Answer to Problem 1" in packet[1].get_text()
    assert "Answer to Problem 2" in packet[1].get_text()
    packet.close()


def test_compact_layout_starts_new_page_when_block_does_not_fit(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _png(paths["artifact_root"] / "p3" / "paper" / "questions" / "q2.png", size=(800, 1200))

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_records"][0]["question_start_page"] == 1
    assert manifest["included_records"][1]["question_start_page"] == 2
    assert manifest["answers_section_page_range"][0] > manifest["questions_section_page_range"][1]


def test_long_review_required_problem_header_is_rendered(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, record_overrides={"q2": {"notes": {"visual_curation_status": "review"}}})
    taxonomy = json.loads(paths["taxonomy"].read_text(encoding="utf-8"))
    for component in taxonomy["components"]:
        if component["paper_family"] == "p3":
            for topic in component["topics"]:
                if topic["topic_id"] == "integration":
                    topic["topic_label"] = "Permutations and combinations with extended syllabus routing review marker"
    paths["taxonomy"].write_text(json.dumps(taxonomy), encoding="utf-8")

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    q2 = next(record for record in manifest["included_records"] if record["question_id"] == "q2")
    packet = fitz.open(paths["output"] / "p3" / "integration" / "p3_integration_packet.pdf")
    page_text = packet[int(q2["question_start_page"]) - 1].get_text()
    assert "Problem 2 - 2024 June P31 Question 2" in page_text
    assert "Review Required - verify topic/routing before classroom use." in page_text
    assert q2["question_block_height_estimate"] > 18
    packet.close()


def test_answer_placement_inline_preserves_paired_order(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
        layout="one-per-page",
        answer_placement="inline",
    )

    packet = fitz.open(paths["output"] / "p3" / "integration" / "p3_integration_packet.pdf")
    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert packet.page_count == 5
    assert manifest["layout"] == "one-per-page"
    assert manifest["answer_placement"] == "inline"
    assert "Approved Questions" in packet[0].get_text()
    assert "Problem 1" in packet[1].get_text()
    assert "Answer to Problem 1" in packet[2].get_text()
    assert "Problem 2" in packet[3].get_text()
    assert "Answer to Problem 2" in packet[4].get_text()
    packet.close()


def test_oversized_image_warning_is_recorded(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _png(paths["artifact_root"] / "p3" / "paper" / "questions" / "q1.png", size=(800, 3000))

    summary = generate_topic_packets(
        question_bank_path=paths["bank"],
        taxonomy_path=paths["taxonomy"],
        canonical_taxonomy_root=paths["canonical_root"],
        output_root=paths["output"],
        artifact_root=paths["artifact_root"],
    )

    manifest = json.loads((paths["output"] / "p3" / "integration" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["blocks_scaled_to_fit_count"] >= 1
    assert manifest["oversized_block_warnings"]
    assert summary["oversized_block_warning_count"] >= 1


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
                        "paper_family": "p1",
                        "component_key": "p1",
                        "paper_code_group": "Paper 1",
                        "topics": [
                            {
                                "topic_id": "quadratics",
                                "topic_label": "Quadratics",
                                "canonical_topic_id": "9709_p1_topic_quadratics",
                                "subtopics": [
                                    {
                                        "subtopic_id": "quadratics_completed_square",
                                        "subtopic_label": "Completing the square",
                                        "canonical_subtopic_id": "9709_p1_subtopic_quadratics_completed_square",
                                        "packet_eligible": True,
                                    }
                                ],
                            }
                        ],
                    },
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
                            },
                            {
                                "topic_id": "vectors",
                                "topic_label": "Vectors",
                                "canonical_topic_id": "9709_p3_topic_vectors",
                                "subtopics": [
                                    {
                                        "subtopic_id": "vector_lines",
                                        "subtopic_label": "Vector lines",
                                        "canonical_subtopic_id": "9709_p3_subtopic_vector_lines",
                                        "packet_eligible": True,
                                    }
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


def _write_reviewed_decisions(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "topic_bank_reviewed_decisions.v1.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.topic_bank_reviewed_decisions",
                "schema_version": 1,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_topic_overlap_review(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "topic_overlap_review.v1.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.p3_topic_overlap_review_batch",
                "schema_version": 1,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def _automated_decision(
    question_id: str,
    *,
    action: str,
    reviewed_topic: str,
    release_override: bool,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "action": action,
        "reviewed_topic": reviewed_topic,
        "reviewed_subtopic": "",
        "reviewed_skill": "",
        "reason": "Automated image and syllabus review confirmed the topic.",
        "reviewer": "automated_agentic_review:gpt-5-mini",
        "reviewed_at": "2026-06-29T00:00:00Z",
        "source": "automated_agentic_review",
        "current_syllabus_status": "current_relevant",
        "release_override": release_override,
        "confidence": 0.95,
        "evidence_refs": [
            {"type": "canonical_question_image", "path": "p3/paper/questions/q2.png"},
            {"type": "canonical_mark_scheme_image", "path": "p3/paper/mark_scheme/q2.png"},
            {
                "type": "syllabus_reference",
                "path": "https://www.cambridgeinternational.org/Images/697427-2026-2027-syllabus.pdf",
            },
        ],
        "risk_flags": [],
        "reviewer_model": "gpt-5-mini",
        "prompt_version": "topic_review_9709_2026_2027_v1",
    }


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
        "notes": {
            "source_pdf": "qp.pdf",
            "subtopic": "general",
            "mapping_status": "pass",
            "validation_status": "pass",
            "scope_quality_status": "clean",
            "question_crop_confidence": "high",
            "visual_curation_status": "ready",
            "text_only_status": "ready",
        },
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


def _png(path: Path, *, size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
