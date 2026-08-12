from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_question_text_gold_registry import SCHEMA_NAME, SCHEMA_VERSION, main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _gold_record(
    question_id: str,
    image_path: Path,
    *,
    review_status: str = "verified",
    image_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper_family": "pure_mathematics_1",
        "question_text": f"Exact text for {question_id}",
        "source_image_path": str(image_path),
        "source_image_sha256": image_sha256 or _sha256(image_path),
        "review_status": review_status,
        "notes": None,
    }


def test_main_builds_canonical_registry_in_cohort_order_with_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    q1_image = images / "q1.png"
    q2_image = images / "q2.png"
    q1_image.write_bytes(b"question-one-image")
    q2_image.write_bytes(b"question-two-image")

    cohort_path = tmp_path / "cohort.json"
    _write_json(
        cohort_path,
        {"questions": [{"question_id": "q2"}, {"question_id": "q1"}]},
    )
    batch_dir = tmp_path / "batches"
    first_batch = batch_dir / "batch_001.json"
    second_batch = batch_dir / "batch_002.json"
    _write_json(first_batch, {"records": [_gold_record("q1", q1_image)]})
    _write_json(second_batch, {"records": [_gold_record("q2", q2_image)]})
    output_path = tmp_path / "canonical" / "gold.json"

    result = main(
        [
            "--batch-dir",
            str(batch_dir),
            "--cohort",
            str(cohort_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_name"] == SCHEMA_NAME
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["source_cohort"] == str(cohort_path)
    assert payload["question_count"] == 2
    assert payload["all_records_verified"] is True
    assert [record["question_id"] for record in payload["records"]] == ["q2", "q1"]
    assert payload["records"][0]["source_image_path"] == str(q2_image)
    assert payload["records"][0]["notes"] == ""
    assert payload["source_batches"] == [
        {
            "path": str(first_batch),
            "sha256": _sha256(first_batch),
            "record_count": 1,
        },
        {
            "path": str(second_batch),
            "sha256": _sha256(second_batch),
            "record_count": 1,
        },
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "output": str(output_path),
        "question_count": 2,
        "source_batch_count": 2,
    }


def test_main_rejects_unverified_gold_record(tmp_path: Path) -> None:
    image_path = tmp_path / "q1.png"
    image_path.write_bytes(b"question-image")
    cohort_path = tmp_path / "cohort.json"
    batch_dir = tmp_path / "batches"
    _write_json(cohort_path, {"questions": [{"question_id": "q1"}]})
    _write_json(
        batch_dir / "batch_001.json",
        {"records": [_gold_record("q1", image_path, review_status="needs_adjudication")]},
    )

    with pytest.raises(ValueError, match="gold record q1 is not verified"):
        main(
            [
                "--batch-dir",
                str(batch_dir),
                "--cohort",
                str(cohort_path),
                "--output",
                str(tmp_path / "gold.json"),
            ]
        )


def test_main_rejects_source_image_hash_mismatch(tmp_path: Path) -> None:
    image_path = tmp_path / "q1.png"
    image_path.write_bytes(b"question-image")
    cohort_path = tmp_path / "cohort.json"
    batch_dir = tmp_path / "batches"
    _write_json(cohort_path, {"questions": [{"question_id": "q1"}]})
    _write_json(
        batch_dir / "batch_001.json",
        {"records": [_gold_record("q1", image_path, image_sha256="0" * 64)]},
    )

    with pytest.raises(ValueError, match="source image hash mismatch for q1"):
        main(
            [
                "--batch-dir",
                str(batch_dir),
                "--cohort",
                str(cohort_path),
                "--output",
                str(tmp_path / "gold.json"),
            ]
        )


def test_main_rejects_gold_cohort_coverage_mismatch(tmp_path: Path) -> None:
    q1_image = tmp_path / "q1.png"
    extra_image = tmp_path / "extra.png"
    q1_image.write_bytes(b"question-one-image")
    extra_image.write_bytes(b"extra-question-image")
    cohort_path = tmp_path / "cohort.json"
    batch_dir = tmp_path / "batches"
    _write_json(
        cohort_path,
        {"questions": [{"question_id": "q1"}, {"question_id": "missing"}]},
    )
    _write_json(
        batch_dir / "batch_001.json",
        {"records": [_gold_record("q1", q1_image), _gold_record("extra", extra_image)]},
    )

    with pytest.raises(
        ValueError,
        match=r"gold/cohort coverage mismatch; missing=\['missing'\], extra=\['extra'\]",
    ):
        main(
            [
                "--batch-dir",
                str(batch_dir),
                "--cohort",
                str(cohort_path),
                "--output",
                str(tmp_path / "gold.json"),
            ]
        )
