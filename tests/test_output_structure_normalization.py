import json
from pathlib import Path

from exam_bank.output_structure_normalization import (
    build_normalization_plan,
    normalize_output_structure,
    validate_normalized_output,
)


def test_normalizes_legacy_output_folders_filenames_and_metadata_refs(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write(root / "p1" / "12spring21" / "questions" / "q01.png", b"question")
    _write(root / "p3" / "31summer08" / "mark_scheme" / "q04.png", b"mark")
    _write(root / "p4" / "42autumn23" / "questions" / "q12.png", b"stats")
    _write(root / "p5" / "51summer25" / "mark_scheme" / "q02.png", b"mechanics")
    _write_json(
        root / "json" / "question_bank.json",
        {
            "questions": [
                {
                    "question_image_path": "p1/12spring21/questions/q01.png",
                    "mark_scheme_image_path": "p3/31summer08/mark_scheme/q04.png",
                }
            ]
        },
    )

    report = normalize_output_structure(root)

    assert report["files_renamed"] == 4
    assert report["conflicts_resolved"] == 0
    assert {Path(item["new_path"]).name for item in report["folders_renamed"]} == {"pm1", "pm3", "stats", "mechanics"}
    assert (root / "pm1" / "pm1_2021_m21_qp_q01_question.png").read_bytes() == b"question"
    assert (root / "pm3" / "pm3_2008_s08_ms_q04_markscheme.png").read_bytes() == b"mark"
    assert (root / "stats" / "stats_2023_w23_qp_q12_question.png").read_bytes() == b"stats"
    assert (root / "mechanics" / "mechanics_2025_s25_ms_q02_markscheme.png").read_bytes() == b"mechanics"
    assert not (root / "p1").exists()
    payload = json.loads((root / "json" / "question_bank.json").read_text(encoding="utf-8"))
    assert payload["questions"][0]["question_image_path"] == "pm1/pm1_2021_m21_qp_q01_question.png"
    assert payload["questions"][0]["mark_scheme_image_path"] == "pm3/pm3_2008_s08_ms_q04_markscheme.png"
    assert json.loads((root / "migration" / "output_structure_normalization.json").read_text(encoding="utf-8"))[
        "files_renamed"
    ] == 4
    assert validate_normalized_output(root)["ok"] is True


def test_normalization_conflict_uses_v2_suffix_and_second_run_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write(root / "p1" / "12spring21" / "questions" / "q01.png", b"legacy")
    _write(root / "pm1" / "pm1_2021_m21_qp_q01_question.png", b"existing")

    report = normalize_output_structure(root)
    second = normalize_output_structure(root, dry_run=True)

    assert report["files_renamed"] == 1
    assert report["conflicts_resolved"] == 1
    assert (root / "pm1" / "pm1_2021_m21_qp_q01_question_v2.png").read_bytes() == b"legacy"
    assert (root / "pm1" / "pm1_2021_m21_qp_q01_question.png").read_bytes() == b"existing"
    assert build_normalization_plan(root) == []
    assert second["files_renamed"] == 0
    assert second["validation"]["ok"] is True


def test_validation_flags_legacy_and_schema_violations(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write(root / "p1" / "12spring21" / "questions" / "q01.png", b"legacy")
    _write(root / "pm1" / "bad.png", b"bad")
    _write(root / "pm3" / "pm1_2021_m21_qp_q01_question.png", b"mixed")

    report = validate_normalized_output(root)

    assert report["ok"] is False
    assert report["legacy_path_count"] > 0
    assert report["invalid_png_count"] >= 2
    assert report["mixed_subject_path_count"] == 1


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
