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


def _write_visual_fixture(
    tmp_path: Path,
    *,
    write_images: bool = True,
    de_vs_implicit_warning: bool = False,
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
                    "candidate_region_topic": {"topic_assignment_name": "Logarithmic and exponential functions"},
                    "recommended_review_action": (
                        "verify_de_vs_implicit_differentiation" if de_vs_implicit_warning else "review_assets_and_skill"
                    ),
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
