import json
from pathlib import Path

from exam_bank.asset_manifest import (
    MARK_SCHEME_IMAGE_KIND,
    QUESTION_IMAGE_KIND,
    asset_id_for_record,
    build_asset_manifest,
    validate_asset_references,
    write_asset_manifest,
)
from exam_bank.asterion_export import build_asterion_export, build_content_lab_candidates
from exam_bank.storage_audit import (
    CACHE_ACTION,
    apply_delete_plan,
    apply_quarantine_plan,
    build_delete_manifest,
    build_storage_audit,
)


def test_asset_manifest_indexes_canonical_images_and_exports_resolve(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _write_bytes(output / "p1" / "12spring21" / "questions" / "q01.png", b"question-one")
    _write_bytes(output / "p1" / "12spring21" / "mark_scheme" / "q01.png", b"mark-one")
    bank_path = output / "json" / "question_bank.json"
    bank = _question_bank()
    _write_json(bank_path, bank)

    manifest_path = write_asset_manifest(bank_path, artifact_root=output, base_dir=tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    question_asset_id = asset_id_for_record(QUESTION_IMAGE_KIND, bank["questions"][0], "p1/12spring21/questions/q01.png")
    mark_asset_id = asset_id_for_record(MARK_SCHEME_IMAGE_KIND, bank["questions"][0], "p1/12spring21/mark_scheme/q01.png")
    by_id = {asset["asset_id"]: asset for asset in manifest["assets"]}
    assert by_id[question_asset_id]["canonical_path"] == "p1/12spring21/questions/q01.png"
    assert by_id[mark_asset_id]["canonical_path"] == "p1/12spring21/mark_scheme/q01.png"

    asterion = build_asterion_export(bank, artifact_root=output, base_dir=tmp_path)
    content_lab = build_content_lab_candidates(asterion)
    _write_json(output / "asterion" / "exports" / "latest" / "asterion_question_bank_v1.json", asterion)
    _write_json(output / "asterion" / "exports" / "latest" / "asterion_content_lab_candidates_v1.json", content_lab)
    _write_json(
        output / "json" / "question_bank.topic_routing.v1.json",
        {
            "schema_name": "exam_bank.topic_routing_sidecar",
            "schema_version": 1,
            "record_count": 1,
            "records": {"12spring21_q01": {"review_required": True}},
        },
    )

    report = validate_asset_references(
        question_bank_path=bank_path,
        asset_manifest_path=manifest_path,
        asterion_path=output / "asterion" / "exports" / "latest" / "asterion_question_bank_v1.json",
        content_lab_path=output / "asterion" / "exports" / "latest" / "asterion_content_lab_candidates_v1.json",
        topic_routing_path=output / "json" / "question_bank.topic_routing.v1.json",
        artifact_root=output,
        project_root=tmp_path,
    )

    assert report["ok"] is True


def test_storage_audit_is_deterministic_and_quarantines_noncanonical_duplicates(tmp_path: Path) -> None:
    canonical = tmp_path / "output" / "p1" / "12spring21" / "questions" / "q01.png"
    duplicate = tmp_path / "output" / "candidates" / "ocr" / "run1" / "p1" / "12spring21" / "questions" / "q01.png"
    _write_bytes(canonical, b"same-image")
    _write_bytes(duplicate, b"same-image")
    bank_path = tmp_path / "output" / "json" / "question_bank.json"
    _write_json(bank_path, _question_bank())

    kwargs = {
        "scan_roots": [Path("output")],
        "reference_json_files": [Path("output/json/question_bank.json")],
        "project_root": tmp_path,
    }
    first = build_storage_audit(**kwargs)
    second = build_storage_audit(**kwargs)

    assert first == second
    image_groups = first["image_duplicate_groups"]
    assert len(image_groups) == 1
    assert image_groups[0]["canonical_file"] == "output/p1/12spring21/questions/q01.png"
    assert any(item["suggested_action"] == CACHE_ACTION for item in first["cleanup_candidates"])

    result = apply_quarantine_plan(first, quarantine_dir=Path("output/_quarantine_test"), project_root=tmp_path)
    assert result["applied"] is True
    assert not duplicate.exists()
    assert (tmp_path / "output" / "_quarantine_test" / "output" / "candidates" / "ocr" / "run1" / "p1" / "12spring21" / "questions" / "q01.png").is_file()
    assert canonical.is_file()


def test_delete_plan_hard_deletes_only_allowlisted_noncanonical_exact_duplicates(tmp_path: Path) -> None:
    canonical = tmp_path / "output" / "p1" / "12spring21" / "questions" / "q01.png"
    candidate_duplicate = tmp_path / "output" / "candidates" / "ocr" / "run1" / "p1" / "12spring21" / "questions" / "q01.png"
    targeted_duplicate = tmp_path / "output" / "codex_text_extraction_targeted" / "p1" / "12spring21" / "questions" / "q01.png"
    topic_packet_duplicate = tmp_path / "output" / "topic_packets" / "p1" / "series" / "copied.png"
    other_canonical_duplicate = tmp_path / "output" / "p1" / "12spring21" / "questions" / "q99.png"
    _write_bytes(canonical, b"same-image")
    _write_bytes(candidate_duplicate, b"same-image")
    _write_bytes(targeted_duplicate, b"same-image")
    _write_bytes(topic_packet_duplicate, b"same-image")
    _write_bytes(other_canonical_duplicate, b"same-image")
    _write_json(tmp_path / "output" / "json" / "question_bank.json", _question_bank())

    audit = build_storage_audit(
        scan_roots=[Path("output")],
        reference_json_files=[Path("output/json/question_bank.json")],
        project_root=tmp_path,
    )
    manifest = build_delete_manifest(audit, project_root=tmp_path)

    paths = {entry["path"] for entry in manifest["entries"]}
    assert paths == {
        "output/candidates/ocr/run1/p1/12spring21/questions/q01.png",
        "output/codex_text_extraction_targeted/p1/12spring21/questions/q01.png",
    }
    assert manifest["delete_file_count"] == 2

    result = apply_delete_plan(
        audit,
        manifest_path=tmp_path / "reports" / "output_storage_delete_manifest.v1.json",
        project_root=tmp_path,
    )

    assert result["deleted_file_count"] == 2
    assert not candidate_duplicate.exists()
    assert not targeted_duplicate.exists()
    assert canonical.is_file()
    assert other_canonical_duplicate.is_file()
    assert topic_packet_duplicate.is_file()
    written = json.loads((tmp_path / "reports" / "output_storage_delete_manifest.v1.json").read_text(encoding="utf-8"))
    assert written["entries"][0]["retained_path"] == "output/p1/12spring21/questions/q01.png"


def _question_bank() -> dict:
    return {
        "schema_name": "exam_bank.question_bank",
        "schema_version": 2,
        "record_count": 1,
        "questions": [
            {
                "question_id": "12spring21_q01",
                "paper": "12spring21",
                "paper_family": "p1",
                "question_number": "1",
                "canonical_question_artifact": "p1/12spring21/questions/q01.png",
                "canonical_mark_scheme_artifact": "p1/12spring21/mark_scheme/q01.png",
                "question_image_path": "p1/12spring21/questions/q01.png",
                "mark_scheme_image_path": "p1/12spring21/mark_scheme/q01.png",
                "question_image_paths": ["p1/12spring21/questions/q01.png"],
                "mark_scheme_image_paths": ["p1/12spring21/mark_scheme/q01.png"],
                "question_text": "1 Find x. [1]",
                "mark_scheme_text": "1 x = 2 B1",
                "question_text_role": "readable_text",
                "question_text_trust": "high",
                "text_only_status": "ready",
                "visual_required": False,
                "visual_curation_status": "ready",
                "question_solution_marks": 1,
                "subparts": [],
                "subparts_solution_marks": {},
                "notes": {
                    "mapping_status": "pass",
                    "validation_status": "pass",
                    "scope_quality_status": "clean",
                    "question_crop_confidence": "high",
                    "mark_scheme_crop_confidence": "high",
                    "paper_total_status": "matched",
                    "topic_confidence": "high",
                    "topic_uncertain": False,
                    "question_total_detected": 1,
                    "mark_scheme_total_detected": 1,
                    "question_structure_detected": {"mark_values_detected": [1]},
                    "mark_scheme_structure_detected": {},
                },
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
