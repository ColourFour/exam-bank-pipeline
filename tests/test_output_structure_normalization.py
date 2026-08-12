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
    _write(root / "p4" / "42winter23" / "questions" / "q12.png", b"mechanics")
    _write(root / "p5" / "51summer25" / "mark_scheme" / "q02.png", b"stats")
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
    assert (root / "pm1" / "pm1_2021_m21_12_qp_q01_question.png").read_bytes() == b"question"
    assert (root / "pm3" / "pm3_2008_s08_31_ms_q04_markscheme.png").read_bytes() == b"mark"
    assert (root / "mechanics" / "mechanics_2023_w23_42_qp_q12_question.png").read_bytes() == b"mechanics"
    assert (root / "stats" / "stats_2025_s25_51_ms_q02_markscheme.png").read_bytes() == b"stats"
    assert not (root / "p1").exists()
    payload = json.loads((root / "json" / "question_bank.json").read_text(encoding="utf-8"))
    assert payload["questions"][0]["question_image_path"] == "pm1/pm1_2021_m21_12_qp_q01_question.png"
    assert payload["questions"][0]["mark_scheme_image_path"] == "pm3/pm3_2008_s08_31_ms_q04_markscheme.png"
    assert json.loads((root / "migration" / "output_structure_normalization.json").read_text(encoding="utf-8"))[
        "files_renamed"
    ] == 4
    assert validate_normalized_output(root)["ok"] is True


def test_normalization_conflict_uses_v2_suffix_and_second_run_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write(root / "p1" / "12spring21" / "questions" / "q01.png", b"legacy")
    _write(root / "pm1" / "pm1_2021_m21_12_qp_q01_question.png", b"existing")

    report = normalize_output_structure(root)
    second = normalize_output_structure(root, dry_run=True)

    assert report["files_renamed"] == 1
    assert report["conflicts_resolved"] == 1
    assert (root / "pm1" / "pm1_2021_m21_12_qp_q01_question_v2.png").read_bytes() == b"legacy"
    assert (root / "pm1" / "pm1_2021_m21_12_qp_q01_question.png").read_bytes() == b"existing"
    assert build_normalization_plan(root) == []
    assert second["files_renamed"] == 0
    assert second["validation"]["ok"] is True


def test_validation_flags_legacy_and_schema_violations(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write(root / "p1" / "12spring21" / "questions" / "q01.png", b"legacy")
    _write(root / "pm1" / "bad.png", b"bad")
    _write(root / "pm3" / "pm1_2021_m21_12_qp_q01_question.png", b"mixed")
    _write(root / "stats" / "stats_2023_w23_42_qp_q01_question.png", b"wrong-family")

    report = validate_normalized_output(root)

    assert report["ok"] is False
    assert report["legacy_path_count"] > 0
    assert report["invalid_png_count"] >= 2
    assert report["mixed_subject_path_count"] == 1
    assert report["component_subject_mismatch_count"] == 1


def test_normalization_ignores_taxonomy_family_dirs_and_noncanonical_png_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write(root / "topic_packets" / "p4" / "forces" / "page_0001.png", b"packet")
    _write(root / "audits" / "p1_sample" / "page_0001.png", b"audit")

    report = normalize_output_structure(root, dry_run=True)

    assert report["folders_renamed"] == []
    assert report["files_renamed"] == 0
    assert report["validation"]["ok"] is True


def test_normalizes_reversed_canonical_component_family_and_json_metadata(tmp_path: Path) -> None:
    root = tmp_path / "output"
    old_question = root / "stats" / "stats_2023_w23_42_qp_q01_question.png"
    old_mark_scheme = root / "mechanics" / "mechanics_2024_s24_51_ms_q02_markscheme.png"
    _write(old_question, b"mechanics-question")
    _write(old_mark_scheme, b"statistics-mark-scheme")
    _write_json(
        root / "json" / "question_bank.json",
        {
            "questions": [
                {
                    "question_id": "42winter23_q01",
                    "paper": "42winter23",
                    "paper_family": "stats",
                    "question_image_path": old_question.relative_to(root).as_posix(),
                },
                {
                    "question_id": "51summer24_q02",
                    "paper": "51summer24",
                    "paper_family": "mechanics",
                    "mark_scheme_image_path": old_mark_scheme.relative_to(root).as_posix(),
                },
            ]
        },
    )

    report = normalize_output_structure(root)

    assert report["files_renamed"] == 2
    assert not old_question.exists()
    assert not old_mark_scheme.exists()
    assert (root / "mechanics" / "mechanics_2023_w23_42_qp_q01_question.png").read_bytes() == b"mechanics-question"
    assert (root / "stats" / "stats_2024_s24_51_ms_q02_markscheme.png").read_bytes() == b"statistics-mark-scheme"
    payload = json.loads((root / "json" / "question_bank.json").read_text(encoding="utf-8"))
    assert payload["questions"][0]["paper_family"] == "mechanics"
    assert payload["questions"][0]["question_image_path"].startswith("mechanics/mechanics_")
    assert payload["questions"][1]["paper_family"] == "stats"
    assert payload["questions"][1]["mark_scheme_image_path"].startswith("stats/stats_")
    assert report["validation"]["ok"] is True
    assert validate_normalized_output(root)["ok"] is True


def test_repairs_stale_json_path_after_canonical_file_was_already_moved(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write(root / "mechanics" / "mechanics_2023_w23_42_qp_q01_question.png", b"question")
    _write_json(
        root / "json" / "question_bank.json",
        {
            "questions": [
                {
                    "question_id": "42winter23_q01",
                    "paper_family": "stats",
                    "question_image_path": "stats/stats_2023_w23_42_qp_q01_question.png",
                }
            ]
        },
    )

    report = normalize_output_structure(root)

    assert report["files_renamed"] == 0
    payload = json.loads((root / "json" / "question_bank.json").read_text(encoding="utf-8"))
    assert payload["questions"][0]["paper_family"] == "mechanics"
    assert payload["questions"][0]["question_image_path"] == "mechanics/mechanics_2023_w23_42_qp_q01_question.png"
    assert validate_normalized_output(root)["ok"] is True


def test_pre_2020_p5_mechanics_2_is_not_silently_migrated_to_stats(tmp_path: Path) -> None:
    root = tmp_path / "output"
    legacy = root / "p5" / "51summer19" / "questions" / "q01.png"
    _write(legacy, b"unsupported-m2")

    report = normalize_output_structure(root, dry_run=True)

    assert report["files_renamed"] == 0
    assert report["folders_renamed"] == []
    assert report["validation"]["ok"] is False
    assert report["validation"]["legacy_path_count"] > 0


def test_validation_rejects_canonical_path_for_unsupported_pre_2020_p5(tmp_path: Path) -> None:
    root = tmp_path / "output"
    path = root / "stats" / "stats_2019_s19_51_qp_q01_question.png"
    _write(path, b"unsupported-m2")

    report = validate_normalized_output(root)

    assert report["ok"] is False
    assert report["unsupported_component_era_count"] == 1
    assert report["unsupported_component_era_paths"] == [str(path)]


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
