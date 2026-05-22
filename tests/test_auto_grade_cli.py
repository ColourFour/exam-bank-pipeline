from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.test_auto_grade_eligibility import _write_fixture


def test_scripts_help_exits_successfully() -> None:
    for script in [
        "scripts/build_auto_grade_eligible_items.py",
        "scripts/validate_auto_grade_eligible_items.py",
        "scripts/build_auto_grade_rubric_review_batch.py",
        "scripts/check_auto_grade_rubric_review_completion.py",
        "scripts/promote_auto_grade_reviewed_rubrics.py",
    ]:
        result = subprocess.run([sys.executable, script, "--help"], cwd=Path.cwd(), text=True, capture_output=True)
        assert result.returncode == 0
        assert "usage:" in result.stdout


def test_build_and_validate_scripts_run_on_fixture_without_mutating_input(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    _write_bytes(paths["artifact_root"] / "p1/11summer26/questions/q02.png")
    _write_bytes(paths["artifact_root"] / "p1/11summer26/mark_scheme/q03.png")
    report = tmp_path / "reports" / "eligible_items_summary.md"
    original = paths["question_bank"].read_text(encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}

    build = subprocess.run(
        [
            sys.executable,
            "scripts/build_auto_grade_eligible_items.py",
            "--question-bank",
            str(paths["question_bank"]),
            "--output",
            str(paths["eligible"]),
            "--artifact-root",
            str(paths["artifact_root"]),
            "--reviewed-rubrics",
            str(paths["reviewed_rubrics"]),
            "--mark-events",
            str(paths["mark_events"]),
            "--topic-routing",
            str(paths["topic_routing"]),
            "--report",
            str(report),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        env=env,
    )
    assert build.returncode == 0, build.stderr
    assert paths["eligible"].exists()
    assert report.exists()
    assert paths["question_bank"].read_text(encoding="utf-8") == original

    validate = subprocess.run(
        [
            sys.executable,
            "scripts/validate_auto_grade_eligible_items.py",
            "--eligible-items",
            str(paths["eligible"]),
            "--question-bank",
            str(paths["question_bank"]),
            "--artifact-root",
            str(paths["artifact_root"]),
            "--reviewed-rubrics",
            str(paths["reviewed_rubrics"]),
            "--output",
            str(tmp_path / "validation.json"),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        env=env,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr


def _write_bytes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
