from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .topic_routing_audit import (
    has_evidence_packet_hash,
    is_failed_route,
    is_strict_filter_candidate,
    route_records_from_payload,
)


DEFAULT_DURABLE_SIDECAR_PATH = Path("data/topic_routing/question_bank.topic_routing.v1.json")
DEFAULT_DURABLE_SHA256_PATH = Path("data/topic_routing/question_bank.topic_routing.v1.sha256")
DEFAULT_LOCAL_SIDECAR_PATH = Path("output/json/question_bank.topic_routing.v1.json")
DEFAULT_QUESTION_BANK_PATH = Path("output/json/question_bank.json")

EXPECTED_RECORD_COUNT = 1301
EXPECTED_FAILED_COUNT = 0
EXPECTED_REVIEW_REQUIRED_COUNT = 42
EXPECTED_STRICT_FILTER_COUNT = 1259
EXPECTED_MISSING_HASH_COUNT = 0


class TopicRoutingArtifactError(RuntimeError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256_file(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise TopicRoutingArtifactError(f"SHA-256 file is empty: {path}")
    value = text.split()[0]
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise TopicRoutingArtifactError(f"Invalid SHA-256 value in {path}: {value!r}")
    return value


def build_topic_routing_artifact_report(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    local_sidecar_path: str | Path = DEFAULT_LOCAL_SIDECAR_PATH,
    durable_sidecar_path: str | Path = DEFAULT_DURABLE_SIDECAR_PATH,
    durable_sha256_path: str | Path = DEFAULT_DURABLE_SHA256_PATH,
    question_bank_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    local_sidecar_path = Path(local_sidecar_path)
    durable_sidecar_path = Path(durable_sidecar_path)
    durable_sha256_path = Path(durable_sha256_path)

    errors: list[str] = []
    for label, path in (
        ("question_bank", question_bank_path),
        ("local_sidecar", local_sidecar_path),
        ("durable_sidecar", durable_sidecar_path),
        ("durable_sha256", durable_sha256_path),
    ):
        if label == "question_bank" and question_bank_payload is not None:
            continue
        if not path.exists():
            errors.append(f"Missing {label}: {path}")

    expected_sha = None
    local_sha = None
    durable_sha = None
    if durable_sha256_path.exists():
        try:
            expected_sha = read_sha256_file(durable_sha256_path)
        except TopicRoutingArtifactError as exc:
            errors.append(str(exc))
    if local_sidecar_path.exists():
        local_sha = file_sha256(local_sidecar_path)
    if durable_sidecar_path.exists():
        durable_sha = file_sha256(durable_sidecar_path)
    if expected_sha and durable_sha and durable_sha != expected_sha:
        errors.append("Durable sidecar SHA-256 does not match the checked-in .sha256 file.")
    if local_sha and durable_sha and local_sha != durable_sha:
        errors.append("Local output/json topic-routing sidecar does not match the durable sidecar artifact.")

    question_bank = question_bank_payload or _read_json_object(question_bank_path)
    local_sidecar = _read_json_object(local_sidecar_path) if local_sidecar_path.exists() else {}
    rows = route_records_from_payload(local_sidecar)
    questions = question_bank.get("questions") if isinstance(question_bank.get("questions"), list) else []
    question_ids = {
        str(row.get("question_id") or "").strip()
        for row in questions
        if isinstance(row, dict) and str(row.get("question_id") or "").strip()
    }
    route_ids = [str(row.get("question_id") or "").strip() for row in rows]
    route_id_counts = Counter(route_ids)
    missing_ids = sorted(question_ids - set(route_ids))
    extra_ids = sorted(set(route_ids) - question_ids)
    duplicate_ids = sorted(question_id for question_id, count in route_id_counts.items() if question_id and count > 1)
    failed_count = sum(1 for row in rows if is_failed_route(row))
    review_required_count = sum(1 for row in rows if row.get("review_required") is True)
    strict_filter_count = sum(1 for row in rows if is_strict_filter_candidate(row))
    missing_hash_count = sum(1 for row in rows if not has_evidence_packet_hash(row))
    safe_for_strict_filters = failed_count == 0 and strict_filter_count > 0

    expected_counts = {
        "records": EXPECTED_RECORD_COUNT,
        "failed": EXPECTED_FAILED_COUNT,
        "review_required": EXPECTED_REVIEW_REQUIRED_COUNT,
        "strict_filter_candidates": EXPECTED_STRICT_FILTER_COUNT,
        "missing_evidence_packet_hash": EXPECTED_MISSING_HASH_COUNT,
    }
    observed_counts = {
        "question_bank_records": len(questions),
        "records": len(rows),
        "unique_ids": len(set(route_ids)),
        "failed": failed_count,
        "review_required": review_required_count,
        "strict_filter_candidates": strict_filter_count,
        "missing_evidence_packet_hash": missing_hash_count,
    }
    for key, expected in expected_counts.items():
        if observed_counts[key] != expected:
            errors.append(f"Unexpected {key}: {observed_counts[key]} != {expected}")
    if len(questions) != EXPECTED_RECORD_COUNT:
        errors.append(f"Unexpected question-bank records: {len(questions)} != {EXPECTED_RECORD_COUNT}")
    if missing_ids:
        errors.append(f"Missing question-bank IDs in sidecar: {len(missing_ids)}")
    if extra_ids:
        errors.append(f"Extra sidecar IDs not in question bank: {len(extra_ids)}")
    if duplicate_ids:
        errors.append(f"Duplicate sidecar IDs: {len(duplicate_ids)}")
    if not safe_for_strict_filters:
        errors.append("Computed safe_for_strict_filters is false.")

    return {
        "ok": not errors,
        "errors": errors,
        "paths": {
            "question_bank": str(question_bank_path),
            "local_sidecar": str(local_sidecar_path),
            "durable_sidecar": str(durable_sidecar_path),
            "durable_sha256": str(durable_sha256_path),
        },
        "sha256": {
            "expected": expected_sha,
            "local_sidecar": local_sha,
            "durable_sidecar": durable_sha,
            "local_matches_durable": bool(local_sha and durable_sha and local_sha == durable_sha),
            "durable_matches_expected": bool(expected_sha and durable_sha and durable_sha == expected_sha),
        },
        "counts": observed_counts,
        "id_coverage": {
            "missing_count": len(missing_ids),
            "extra_count": len(extra_ids),
            "duplicate_count": len(duplicate_ids),
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
            "duplicate_ids": duplicate_ids,
        },
        "safe_for_strict_filters": safe_for_strict_filters,
    }


def verify_topic_routing_artifact(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    local_sidecar_path: str | Path = DEFAULT_LOCAL_SIDECAR_PATH,
    durable_sidecar_path: str | Path = DEFAULT_DURABLE_SIDECAR_PATH,
    durable_sha256_path: str | Path = DEFAULT_DURABLE_SHA256_PATH,
    question_bank_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = build_topic_routing_artifact_report(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha256_path,
        question_bank_payload=question_bank_payload,
    )
    if not report["ok"]:
        raise TopicRoutingArtifactError("; ".join(report["errors"]))
    return report


def restore_topic_routing_sidecar(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    local_sidecar_path: str | Path = DEFAULT_LOCAL_SIDECAR_PATH,
    durable_sidecar_path: str | Path = DEFAULT_DURABLE_SIDECAR_PATH,
    durable_sha256_path: str | Path = DEFAULT_DURABLE_SHA256_PATH,
) -> dict[str, Any]:
    local_sidecar_path = Path(local_sidecar_path)
    durable_sidecar_path = Path(durable_sidecar_path)
    durable_sha256_path = Path(durable_sha256_path)

    expected_sha = read_sha256_file(durable_sha256_path)
    durable_sha = file_sha256(durable_sidecar_path)
    if durable_sha != expected_sha:
        raise TopicRoutingArtifactError("Durable sidecar SHA-256 does not match the checked-in .sha256 file.")
    local_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(durable_sidecar_path, local_sidecar_path)
    return verify_topic_routing_artifact(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha256_path,
    )


def should_enforce_production_topic_routing_provenance(path: str | Path | None) -> bool:
    if path is None:
        return False
    candidate = Path(path)
    if candidate == DEFAULT_LOCAL_SIDECAR_PATH:
        return True
    try:
        return candidate.resolve() == DEFAULT_LOCAL_SIDECAR_PATH.resolve()
    except FileNotFoundError:
        return False


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TopicRoutingArtifactError(f"Expected JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore or verify the durable topic-routing sidecar artifact.")
    parser.add_argument("action", choices=["verify", "restore"])
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK_PATH)
    parser.add_argument("--local-sidecar", type=Path, default=DEFAULT_LOCAL_SIDECAR_PATH)
    parser.add_argument("--durable-sidecar", type=Path, default=DEFAULT_DURABLE_SIDECAR_PATH)
    parser.add_argument("--durable-sha256", type=Path, default=DEFAULT_DURABLE_SHA256_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "restore":
            report = restore_topic_routing_sidecar(
                question_bank_path=args.question_bank,
                local_sidecar_path=args.local_sidecar,
                durable_sidecar_path=args.durable_sidecar,
                durable_sha256_path=args.durable_sha256,
            )
        else:
            report = verify_topic_routing_artifact(
                question_bank_path=args.question_bank,
                local_sidecar_path=args.local_sidecar,
                durable_sidecar_path=args.durable_sidecar,
                durable_sha256_path=args.durable_sha256,
            )
    except TopicRoutingArtifactError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
