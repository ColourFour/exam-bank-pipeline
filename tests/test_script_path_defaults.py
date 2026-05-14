from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_script_module(script_name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{script_name}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_paths(paths: list[str | Path]) -> set[str]:
    return {Path(path).as_posix() for path in paths}


def test_taxonomy_audit_defaults_use_current_generated_layout() -> None:
    module = load_script_module("audit_and_organize_taxonomy_files")

    args = module.build_parser().parse_args(["--phase", "audit"])
    paths = normalize_paths(module.selected_reference_json_files(args))

    assert "output/json/question_bank.json" in paths
    assert "output/asterion/exports/latest/asterion_question_bank_v1.json" in paths
    assert "output/asterion/exports/latest/asterion_content_lab_candidates_v1.json" in paths
    assert "output_ocr_candidate/json/question_bank.json" not in paths
    assert "output_ocr_candidate/json/asterion_question_bank_v1.json" not in paths
    assert "output_ocr_candidate/json/asterion_content_lab_candidates_v1.json" not in paths


def test_taxonomy_audit_historical_references_must_be_explicit() -> None:
    module = load_script_module("audit_and_organize_taxonomy_files")

    args = module.build_parser().parse_args(
        [
            "--phase",
            "audit",
            "--historical-reference-json",
            "output_ocr_candidate/json/question_bank.json",
        ]
    )
    paths = normalize_paths(module.selected_reference_json_files(args))

    assert "output_ocr_candidate/json/question_bank.json" in paths
