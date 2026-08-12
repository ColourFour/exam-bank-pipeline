from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .atomic_json import write_atomic_json
from .review_asset_binding import question_asset_index


SCHEMA_NAME = "exam_bank.question_text_gold_asset_binding"
SCHEMA_VERSION = 1
GOLD_SCHEMA_NAME = "exam_bank.question_text_exact_gold"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind question-text gold records to byte-identical images in the current canonical bank."
        )
    )
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/review/canonical/text_fidelity/question_text_gold.v1.json"),
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("manifests/validations/question_text_gold_asset_binding.v1.json"),
    )
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_bank = _read_object(args.question_bank)
    gold = _read_object(args.gold)
    source_gold_sha256 = _file_sha256(args.gold)
    result = rebind_question_text_gold_registry(
        question_bank,
        gold,
        artifact_root=args.artifact_root,
    )
    report = dict(result["report"])
    report.update(
        {
            "question_bank": str(args.question_bank),
            "question_bank_sha256": _file_sha256(args.question_bank),
            "gold_registry": str(args.gold),
            "source_gold_registry_sha256": source_gold_sha256,
            "write": bool(args.write),
        }
    )
    if args.write:
        write_atomic_json(result["registry"], args.gold)
        report["current_gold_registry_sha256"] = _file_sha256(args.gold)
    write_atomic_json(report, args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["ok"] else 1


def rebind_question_text_gold_registry(
    question_bank: dict[str, Any],
    gold_registry: dict[str, Any],
    *,
    artifact_root: str | Path,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Bind verified transcriptions to byte-identical current question images."""

    rebound = copy.deepcopy(gold_registry)
    rows = rebound.get("records")
    if not isinstance(rows, list):
        raise ValueError("question-text gold registry must contain a records list")
    questions = question_bank.get("questions")
    if not isinstance(questions, list):
        raise ValueError("question bank must contain a questions list")
    by_id = {
        _text(row.get("question_id")): row
        for row in questions
        if isinstance(row, dict) and _text(row.get("question_id"))
    }
    assets = question_asset_index(
        question_bank,
        artifact_root=artifact_root,
        base_dir=base_dir,
    )
    assets_by_hash: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for question_id, kinds in assets.items():
        for asset in kinds.get("question", []):
            digest = _text(asset.get("sha256"))
            if digest:
                assets_by_hash.setdefault(digest, []).append((question_id, asset))

    rebound_ids: list[str] = []
    unchanged_ids: list[str] = []
    re_review_ids: list[str] = []
    missing_question_ids: list[str] = []
    missing_asset_ids: list[str] = []
    reidentified: list[dict[str, str]] = []
    assigned_question_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"question-text gold record {index} must be an object")
        question_id = _text(row.get("question_id"))
        expected_hash = _text(row.get("source_image_sha256"))
        hash_matches = assets_by_hash.get(expected_hash, [])
        matching_ids = sorted({match_question_id for match_question_id, _ in hash_matches})
        target_id = question_id if question_id in matching_ids else ""
        if not target_id and len(matching_ids) == 1:
            target_id = matching_ids[0]
        if not target_id:
            failure_id = question_id or f"index:{index}"
            if question_id not in by_id:
                missing_question_ids.append(failure_id)
            elif not assets.get(question_id, {}).get("question"):
                missing_asset_ids.append(failure_id)
            else:
                re_review_ids.append(failure_id)
            _mark_for_re_review(row, "reviewed_image_hash_not_current")
            continue
        if target_id in assigned_question_ids:
            re_review_ids.append(question_id or f"index:{index}")
            _mark_for_re_review(row, "reviewed_image_identity_collision")
            continue
        assigned_question_ids.add(target_id)
        match = next(asset for match_id, asset in hash_matches if match_id == target_id)
        if row.get("review_status") != "verified":
            re_review_ids.append(question_id or target_id)
            _mark_for_re_review(row, "record_was_not_verified")
            continue
        if target_id != question_id:
            reidentified.append(
                {
                    "legacy_question_id": question_id,
                    "canonical_question_id": target_id,
                    "source_image_sha256": expected_hash,
                }
            )
            row["question_id"] = target_id
        current_path = match["path"]
        if row.get("source_image_path") == current_path:
            unchanged_ids.append(target_id)
        else:
            row["source_image_path"] = current_path
            rebound_ids.append(target_id)
        family = _text(by_id[target_id].get("paper_family"))
        if family:
            row["paper_family"] = family

    invalid_ids = [*missing_question_ids, *missing_asset_ids, *re_review_ids]
    rebound["question_count"] = len(rows)
    rebound["all_records_verified"] = not invalid_ids
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "ok": not invalid_ids,
        "record_count": len(rows),
        "rebound_count": len(rebound_ids),
        "reidentified_count": len(reidentified),
        "unchanged_count": len(unchanged_ids),
        "re_review_required_count": len(invalid_ids),
        "rebound_question_ids": rebound_ids,
        "reidentified_questions": reidentified,
        "re_review_required_question_ids": invalid_ids,
        "missing_question_ids": missing_question_ids,
        "missing_asset_question_ids": missing_asset_ids,
    }
    rebound["current_asset_binding"] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "ok": report["ok"],
        "re_review_required_count": report["re_review_required_count"],
    }
    return {"registry": rebound, "report": report}


def _mark_for_re_review(row: dict[str, Any], reason: str) -> None:
    row["review_status"] = "re_review_required"
    row["asset_binding_error"] = reason


def _text(value: Any) -> str:
    return str(value or "").strip()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
