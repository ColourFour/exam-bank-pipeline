from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_NAME = "exam_bank.question_text_exact_gold"
SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote reviewed question-text batches into one exact-match gold registry."
    )
    parser.add_argument(
        "--batch-dir",
        default="data/review/runs/question_text_gold",
        help="Directory containing batch_*.json review artifacts.",
    )
    parser.add_argument(
        "--cohort",
        default="reports/random_visual_accuracy_audit_20260723/sample.json",
        help="Fixed cohort JSON whose question order defines the registry order.",
    )
    parser.add_argument(
        "--output",
        default="data/review/canonical/text_fidelity/question_text_gold.v1.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_dir = Path(args.batch_dir)
    cohort_path = Path(args.cohort)
    output_path = Path(args.output)

    cohort = _load_object(cohort_path)
    cohort_rows = cohort.get("questions")
    if not isinstance(cohort_rows, list) or not all(isinstance(row, dict) for row in cohort_rows):
        raise ValueError(f"{cohort_path} must contain a questions list of objects")
    cohort_ids = [_required_text(row, "question_id", source=str(cohort_path)) for row in cohort_rows]
    if len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("cohort contains duplicate question_id values")

    batch_paths = sorted(batch_dir.glob("batch_*.json"))
    if not batch_paths:
        raise ValueError(f"no batch_*.json files found in {batch_dir}")

    records_by_id: dict[str, dict[str, Any]] = {}
    source_batches: list[dict[str, Any]] = []
    for batch_path in batch_paths:
        batch = _load_object(batch_path)
        rows = batch.get("records")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{batch_path} must contain a records list of objects")
        source_batches.append(
            {
                "path": str(batch_path),
                "sha256": _sha256(batch_path),
                "record_count": len(rows),
            }
        )
        for row in rows:
            question_id = _required_text(row, "question_id", source=str(batch_path))
            if question_id in records_by_id:
                raise ValueError(f"duplicate gold question_id: {question_id}")
            if row.get("review_status") != "verified":
                raise ValueError(f"gold record {question_id} is not verified")
            question_text = _required_text(row, "question_text", source=question_id)
            paper_family = _required_text(row, "paper_family", source=question_id)
            image_path = _resolved_image_path(
                _required_text(row, "source_image_path", source=question_id)
            )
            expected_hash = _required_text(row, "source_image_sha256", source=question_id)
            actual_hash = _sha256(image_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"source image hash mismatch for {question_id}: {actual_hash} != {expected_hash}"
                )
            records_by_id[question_id] = {
                "question_id": question_id,
                "paper_family": paper_family,
                "question_text": question_text,
                "source_image_path": str(image_path),
                "source_image_sha256": expected_hash,
                "review_status": "verified",
                "notes": str(row.get("notes") or ""),
            }

    missing = [question_id for question_id in cohort_ids if question_id not in records_by_id]
    extra = sorted(set(records_by_id) - set(cohort_ids))
    if missing or extra:
        raise ValueError(f"gold/cohort coverage mismatch; missing={missing}, extra={extra}")

    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source_cohort": str(cohort_path),
        "review_method": (
            "Manual transcription and verification at original PNG resolution; incomplete crops "
            "were adjudicated against rendered source-PDF pages."
        ),
        "question_count": len(cohort_ids),
        "all_records_verified": True,
        "source_batches": source_batches,
        "records": [records_by_id[question_id] for question_id in cohort_ids],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "question_count": len(cohort_ids),
                "source_batch_count": len(source_batches),
            },
            indent=2,
        )
    )
    return 0


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _required_text(row: dict[str, Any], field: str, *, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} must contain a non-empty {field}")
    return value


def _resolved_image_path(value: str) -> Path:
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute() and not raw.parts[:1] == ("output",):
        candidates.append(Path("output") / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"source image is missing: tried {[str(path) for path in candidates]}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
