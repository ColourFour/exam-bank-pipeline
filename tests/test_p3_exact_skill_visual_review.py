from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.p3_exact_skill.visual_review import build_p3_exact_skill_visual_review_packet


def test_html_generation_includes_selected_item_metadata(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    html = result["html"]
    assert "P3 Exact-Skill Visual Review Packet: batch_test" in html
    assert "p3_exact_skill_review_queue:v1:q1:q1_whole" in html
    assert "9709_p3_3_2_log_exponential_equations" in html
    assert "review_assets_and_skill" in html


def test_existing_image_refs_become_relative_image_links(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        output_path=paths["batch_dir"] / "batch_test_visual_review.html",
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    html = result["html"]
    assert '<img src="../../../output/p3/test/questions/q1.png"' in html
    assert '<img src="../../../output/p3/test/mark_scheme/q1.png"' in html
    assert "Open original" in html


def test_missing_image_refs_produce_warning_text(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path, write_images=False)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert "Asset path is missing or cannot be resolved" in result["html"]


def test_advisory_mark_event_warning_is_present(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert "advisory-only review context" in result["html"]
    assert "q1_me0001" in result["html"]


def test_generated_html_does_not_call_candidates_clean(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert "clean_candidate" not in result["html"]
    assert "route_status: clean" not in result["html"]


def test_visual_review_surfaces_de_vs_implicit_warning_context(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path, de_vs_implicit_warning=True)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert "possible_differential_equation_not_parametric_or_implicit" in result["html"]
    assert "verify_de_vs_implicit_differentiation" in result["html"]


def test_visual_review_renders_cross_topic_review_section(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path, cross_topic_status="cross_topic_reviewable")

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert "Cross-topic Review" in result["html"]
    assert "cross_topic_reviewable" in result["html"]
    assert "Supporting skills are not automatically reviewed source evidence" in result["html"]


def test_visual_review_renders_sharper_status_warnings(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path, candidate_status="split_needed_candidate")

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert "Candidate status" in result["html"]
    assert "split_needed_candidate" in result["html"]
    assert "Do not approve whole-question evidence" in result["html"]


def test_visual_review_renders_conflict_status_warning(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path, candidate_status="conflict_candidate", de_vs_implicit_warning=True)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert "Known-risk conflict" in result["html"]
    assert "Treat as ambiguous or blocked" in result["html"]


def test_visual_review_renders_part_decomposition_section(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path, with_part_decomposition=True)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    assert "Part-level Decomposition" in result["html"]
    assert "Is this part actually testing one specific skill?" in result["html"]
    assert "data-field=\"part_boundary_confirmed\"" in result["html"]
    assert "data-field=\"decomposition_accepted\"" in result["html"]
    assert "data-field=\"reviewed_part_subpart\"" in result["html"]


def test_visual_review_includes_response_capture_controls(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        dry_run=True,
        generated_at="2026-05-23T00:00:00Z",
    )

    html = result["html"]
    assert "Your Review Response" in html
    assert "Have you inspected the question image?" in html
    assert "Does this confirm the suggested exact P3 skill?" in html
    assert "data-field=\"route_status\"" in html
    assert "data-field=\"allowed_use_case_summary\"" in html
    assert "data-field=\"reviewer_notes\"" in html
    assert "allowed_mastery" not in html
    assert "data-field=\"evidence_basis\"" not in html
    assert "p3ExactSkillReviewResponses:batch_test" in html
    assert "/p3-exact-skill-review-responses" in html
    assert "Download JSON" in html
    assert "Write repo JSON (server required)" in html


def test_dry_run_does_not_write_files(tmp_path: Path) -> None:
    paths = _write_visual_fixture(tmp_path)
    output = paths["batch_dir"] / "batch_test_visual_review.html"

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=paths["batch_dir"],
        batch_id="batch_test",
        repo_root=paths["repo_root"],
        output_path=output,
        dry_run=True,
    )

    assert result["selected_count"] == 1
    assert not output.exists()


def test_visual_review_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_p3_exact_skill_visual_review_packet.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_visual_review_save_server_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/serve_p3_exact_skill_visual_review.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def _write_visual_fixture(
    tmp_path: Path,
    *,
    write_images: bool = True,
    de_vs_implicit_warning: bool = False,
    cross_topic_status: str = "single_skill_candidate",
    candidate_status: str = "cross_topic_candidate",
    with_part_decomposition: bool = False,
) -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    batch_dir = repo_root / "data" / "review" / "p3_exact_skill_batches"
    queue_path = repo_root / "reports" / "p3_exact_skill_review_queue.v1.json"
    image_dir = repo_root / "output" / "p3" / "test"
    if write_images:
        _write_bytes(image_dir / "questions" / "q1.png", b"fake-question-image")
        _write_bytes(image_dir / "mark_scheme" / "q1.png", b"fake-mark-image")
    batch_dir.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch_test_review_packet.md").write_text("# packet\n", encoding="utf-8")
    _write_json(
        batch_dir / "batch_test_manifest.v1.json",
        {
            "schema": "exam_bank.p3_exact_skill.review_batch_manifest",
            "batch_id": "batch_test",
            "selected_count": 1,
            "generated_at": "2026-05-23T00:00:00Z",
            "source_queue_path": "reports/p3_exact_skill_review_queue.v1.json",
            "selected_queue_ids": ["p3_exact_skill_review_queue:v1:q1:q1_whole"],
        },
    )
    _write_json(
        batch_dir / "batch_test_decision_template.v1.json",
        {
            "schema": "exam_bank.p3_exact_skill.review_batch_template",
            "batch_id": "batch_test",
            "records": [
                {
                    "queue_id": "p3_exact_skill_review_queue:v1:q1:q1_whole",
                    "question_id": "q1",
                    "part_id": "whole",
                    "subpart_id": "q1_whole",
                    "paper": "test",
                    "session": "June",
                    "variant": "1",
                    "suggested_source_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
                    "suggested_primary_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
                    "suggested_supporting_skill_ids": (
                        ["9709_p3_3_3_trigonometric_equations"]
                        if cross_topic_status == "cross_topic_reviewable"
                        else []
                    ),
                    "suggested_cross_topic_status": cross_topic_status,
                    "suggested_candidate_status": candidate_status,
                    "suggested_review_priority": "2_cross_topic_candidate",
                    "suggested_scope_risk": "reviewer_decide",
                    "suggested_ambiguity_reason": "cross_topic_reviewable",
                    "suggested_decomposition_status": "part_level_candidate" if with_part_decomposition else "not_decomposable",
                    "suggested_part_level_candidates": (
                        [
                            {
                                "decomposition_id": "p3_part_decomp:v1:q1:a",
                                "proposed_part_id": "a",
                                "decomposition_status": "part_level_candidate",
                                "candidate_source_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
                                "candidate_topic_ids": ["9709_p3_topic_test"],
                                "matching_mark_event_refs": [{"event_id": "q1_me0001", "part_path": ["a"]}],
                                "other_part_mark_event_refs": [],
                                "evidence_signals": {"mark_event_part_match": True},
                            }
                        ]
                        if with_part_decomposition
                        else []
                    ),
                    "suggested_recommended_scope": (
                        "reviewer_decide" if cross_topic_status == "cross_topic_reviewable" else "whole_question"
                    ),
                    "source_question_asset_refs": [{"path": "p3/test/questions/q1.png"}],
                    "source_mark_scheme_asset_refs": [{"path": "p3/test/mark_scheme/q1.png"}],
                    "mark_event_refs": [{"event_id": "q1_me0001", "review_status": "advisory", "advisory_only": True}],
                }
            ],
        },
    )
    _write_json(
        queue_path,
        {
            "items": [
                {
                    "queue_id": "p3_exact_skill_review_queue:v1:q1:q1_whole",
                    "candidate_p3_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
                    "primary_candidate_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
                    "supporting_candidate_skill_ids": (
                        ["9709_p3_3_3_trigonometric_equations"]
                        if cross_topic_status == "cross_topic_reviewable"
                        else []
                    ),
                    "candidate_region_topic": {"topic_assignment_name": "Logarithmic and exponential functions"},
                    "cross_topic_status": "conflict_needs_review" if de_vs_implicit_warning else cross_topic_status,
                    "topic_routing_topic_ids": ["9709_p3_topic_trigonometry"],
                    "topic_routing_alignment": (
                        "conflicting"
                        if de_vs_implicit_warning
                        else "supporting_topic"
                        if cross_topic_status == "cross_topic_reviewable"
                        else "aligned"
                    ),
                    "cross_topic_notes": ["Supporting candidate skills are review context only."],
                    "recommended_scope": "reviewer_decide",
                    "reviewer_cross_topic_checklist": ["Identify the main skill being assessed."],
                    "recommended_review_action": (
                        "verify_de_vs_implicit_differentiation" if de_vs_implicit_warning else "review_assets_and_skill"
                    ),
                    "proposed_route_status": candidate_status,
                    "review_priority_group": "2_cross_topic_candidate",
                    "ambiguity_reason": "cross_topic_reviewable",
                    "decomposition_status": "part_level_candidate" if with_part_decomposition else "not_decomposable",
                    "proposed_part_level_candidates": (
                        [
                            {
                                "decomposition_id": "p3_part_decomp:v1:q1:a",
                                "proposed_part_id": "a",
                                "decomposition_status": "part_level_candidate",
                                "candidate_source_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
                                "candidate_topic_ids": ["9709_p3_topic_test"],
                                "matching_mark_event_refs": [{"event_id": "q1_me0001", "part_path": ["a"]}],
                                "other_part_mark_event_refs": [],
                                "evidence_signals": {"mark_event_part_match": True},
                            }
                        ]
                        if with_part_decomposition
                        else []
                    ),
                    "part_signal_summary": {"has_part_labeled_mark_events": with_part_decomposition},
                    "part_scope_warning": "Use whole-question images to confirm part boundary.",
                    "proposed_blockers": [
                        "possible_differential_equation_not_parametric_or_implicit"
                        if de_vs_implicit_warning
                        else "mark_events_advisory_only"
                    ],
                    "asterion_candidate": {"generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved"]},
                    "topic_routing": {"confidence": "high"},
                }
            ]
        },
    )
    return {"repo_root": repo_root, "batch_dir": batch_dir}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
