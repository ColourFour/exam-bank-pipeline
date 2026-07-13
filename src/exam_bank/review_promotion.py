from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_json import write_atomic_json
from .corpus import sha256_file

REVIEW_PROMOTION_SCHEMA = "exam_bank.review_promotion"
REVIEW_PROMOTION_VERSION = 1
DEFAULT_CANONICAL_REVIEW_ROOT = Path("data/review/canonical")
ALLOWED_AUTHORITIES = {"human", "automated_review"}


class ReviewPromotionError(ValueError):
    pass


def promote_review_artifact(
    source_path: str | Path,
    target: str | Path,
    *,
    authority: str,
    source_run_id: str,
    reviewed_by: str,
    reviewed_at: str | None = None,
    canonical_root: str | Path = DEFAULT_CANONICAL_REVIEW_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_path)
    canonical_root = Path(canonical_root)
    if authority not in ALLOWED_AUTHORITIES:
        raise ReviewPromotionError(f"Invalid decision authority: {authority}")
    if not source_run_id.strip():
        raise ReviewPromotionError("source_run_id is required")
    if not reviewed_by.strip():
        raise ReviewPromotionError("reviewed_by is required")
    reviewed_at = reviewed_at or _utc_now_iso()
    _validate_timestamp(reviewed_at)
    if not source_path.is_file():
        raise ReviewPromotionError(f"Review source does not exist: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReviewPromotionError("Review source must contain a JSON object")
    target_path = _target_path(canonical_root, target)
    promoted = dict(payload)
    promoted["promotion"] = {
        "schema_name": REVIEW_PROMOTION_SCHEMA,
        "schema_version": REVIEW_PROMOTION_VERSION,
        "decision_authority": authority,
        "source_run_id": source_run_id,
        "source_artifact_path": str(source_path),
        "source_artifact_sha256": sha256_file(source_path),
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
    }
    if not dry_run:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_json(promoted, target_path, sort_keys=True)
    return {
        "schema_name": "exam_bank.review_promotion_report",
        "schema_version": 1,
        "ok": True,
        "dry_run": dry_run,
        "source": str(source_path),
        "source_sha256": promoted["promotion"]["source_artifact_sha256"],
        "target": str(target_path),
        "authority": authority,
        "source_run_id": source_run_id,
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
    }


def validate_promoted_review_artifact(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return [f"artifact_missing:{path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid_json:{exc}"]
    if not isinstance(payload, dict):
        return ["artifact_not_object"]
    promotion = payload.get("promotion")
    if not isinstance(promotion, dict):
        return ["promotion_missing"]
    errors: list[str] = []
    if promotion.get("schema_name") != REVIEW_PROMOTION_SCHEMA:
        errors.append("promotion_schema_mismatch")
    if promotion.get("schema_version") != REVIEW_PROMOTION_VERSION:
        errors.append("promotion_schema_version_mismatch")
    if promotion.get("decision_authority") not in ALLOWED_AUTHORITIES:
        errors.append("invalid_decision_authority")
    for field in ("source_run_id", "source_artifact_path", "source_artifact_sha256", "reviewed_by", "reviewed_at"):
        if not str(promotion.get(field) or "").strip():
            errors.append(f"promotion_field_missing:{field}")
    try:
        _validate_timestamp(str(promotion.get("reviewed_at") or ""))
    except ReviewPromotionError:
        errors.append("invalid_reviewed_at")
    return errors


def run_promote(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a validated review artifact into canonical review storage.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True, help="Path relative to the canonical review root.")
    parser.add_argument("--authority", required=True, choices=sorted(ALLOWED_AUTHORITIES))
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--reviewed-at", default=None)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_REVIEW_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = promote_review_artifact(
        args.input,
        args.target,
        authority=args.authority,
        source_run_id=args.source_run_id,
        reviewed_by=args.reviewed_by,
        reviewed_at=args.reviewed_at,
        canonical_root=args.canonical_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _target_path(root: Path, target: str | Path) -> Path:
    target = Path(target)
    if target.is_absolute() or ".." in target.parts:
        raise ReviewPromotionError(f"Target must be relative to {root}: {target}")
    root_resolved = root.resolve()
    resolved = (root / target).resolve()
    if root_resolved not in resolved.parents:
        raise ReviewPromotionError(f"Target escapes canonical review root: {target}")
    if resolved.suffix.lower() != ".json":
        raise ReviewPromotionError("Canonical review artifacts must be JSON files")
    return resolved


def _validate_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewPromotionError(f"Invalid reviewed_at timestamp: {value}") from exc


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
