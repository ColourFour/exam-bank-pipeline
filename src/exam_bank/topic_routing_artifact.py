from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .release_manifest import (
    DEFAULT_RELEASE_MANIFEST_PATH,
    QUESTION_BANK_ROLE,
    TOPIC_ROUTING_ROLE,
    ReleaseManifestError,
    build_release_manifest,
    file_sha256,
    resolve_release_artifact,
    verify_release_manifest,
)

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

class TopicRoutingArtifactError(RuntimeError):
    pass


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
    release_manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST_PATH,
    question_bank_payload: dict[str, Any] | None = None,
    check_local_sidecar: bool = True,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    local_sidecar_path = Path(local_sidecar_path)
    durable_sidecar_path = Path(durable_sidecar_path)
    durable_sha256_path = Path(durable_sha256_path)
    release_manifest_path = Path(release_manifest_path)

    errors: list[str] = []
    for label, path in (
        ("question_bank", question_bank_path),
        ("durable_sidecar", durable_sidecar_path),
        ("durable_sha256", durable_sha256_path),
        ("release_manifest", release_manifest_path),
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
    if check_local_sidecar and local_sha and durable_sha and local_sha != durable_sha:
        errors.append("Local output/json topic-routing sidecar does not match the durable sidecar artifact.")

    release_report: dict[str, Any] | None = None
    if release_manifest_path.exists():
        try:
            release_report = verify_release_manifest(
                release_manifest_path,
                required_roles=(QUESTION_BANK_ROLE, TOPIC_ROUTING_ROLE),
            )
        except ReleaseManifestError as exc:
            errors.append(str(exc))
        else:
            released_bank = release_report["artifacts"][QUESTION_BANK_ROLE]["path"]
            released_routes = release_report["artifacts"][TOPIC_ROUTING_ROLE]["path"]
            if released_bank.resolve() != question_bank_path.resolve():
                errors.append("Release manifest question_bank path does not match the requested question bank.")
            if released_routes.resolve() != durable_sidecar_path.resolve():
                errors.append("Release manifest topic_routing path does not match the durable sidecar.")

    question_bank = question_bank_payload or _read_json_object(question_bank_path)
    durable_sidecar = _read_json_object(durable_sidecar_path) if durable_sidecar_path.exists() else {}
    if question_bank and question_bank.get("schema_name") != "exam_bank.question_bank":
        errors.append(f"Unexpected question-bank schema: {question_bank.get('schema_name')!r}")
    if durable_sidecar and durable_sidecar.get("schema_name") != "exam_bank.topic_routing_sidecar":
        errors.append(f"Unexpected topic-routing schema: {durable_sidecar.get('schema_name')!r}")
    if durable_sidecar and durable_sidecar.get("schema_version") != 1:
        errors.append(f"Unsupported topic-routing schema version: {durable_sidecar.get('schema_version')!r}")
    rows = route_records_from_payload(durable_sidecar)
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

    observed_counts = {
        "question_bank_records": len(questions),
        "records": len(rows),
        "unique_ids": len(set(route_ids)),
        "failed": failed_count,
        "review_required": review_required_count,
        "strict_filter_candidates": strict_filter_count,
        "missing_evidence_packet_hash": missing_hash_count,
    }
    declared_bank_count = question_bank.get("record_count")
    if declared_bank_count is not None and declared_bank_count != len(questions):
        errors.append(f"Question-bank declared record_count mismatch: {declared_bank_count} != {len(questions)}")
    declared_route_count = durable_sidecar.get("record_count")
    if declared_route_count is not None and declared_route_count != len(rows):
        errors.append(f"Topic-routing declared record_count mismatch: {declared_route_count} != {len(rows)}")
    if len(rows) != len(questions):
        errors.append(f"Topic-routing/question-bank record-count mismatch: {len(rows)} != {len(questions)}")
    if failed_count:
        errors.append(f"Topic-routing contains failed records: {failed_count}")
    if missing_hash_count:
        errors.append(f"Topic-routing records missing evidence-packet hashes: {missing_hash_count}")
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
            "release_manifest": str(release_manifest_path),
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
        "release": {
            "verified": release_report is not None,
            "release_id": release_report.get("release_id") if release_report else None,
        },
    }


def verify_topic_routing_artifact(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    local_sidecar_path: str | Path = DEFAULT_LOCAL_SIDECAR_PATH,
    durable_sidecar_path: str | Path = DEFAULT_DURABLE_SIDECAR_PATH,
    durable_sha256_path: str | Path = DEFAULT_DURABLE_SHA256_PATH,
    release_manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST_PATH,
    question_bank_payload: dict[str, Any] | None = None,
    check_local_sidecar: bool = True,
) -> dict[str, Any]:
    report = build_topic_routing_artifact_report(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha256_path,
        release_manifest_path=release_manifest_path,
        question_bank_payload=question_bank_payload,
        check_local_sidecar=check_local_sidecar,
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
    release_manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST_PATH,
) -> dict[str, Any]:
    local_sidecar_path = Path(local_sidecar_path)
    durable_sidecar_path = Path(durable_sidecar_path)
    durable_sha256_path = Path(durable_sha256_path)

    verify_topic_routing_artifact(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha256_path,
        release_manifest_path=release_manifest_path,
        check_local_sidecar=False,
    )
    local_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(durable_sidecar_path, local_sidecar_path)
    return verify_topic_routing_artifact(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha256_path,
        release_manifest_path=release_manifest_path,
    )


def build_topic_routing_release_manifest(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    durable_sidecar_path: str | Path = DEFAULT_DURABLE_SIDECAR_PATH,
    release_manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST_PATH,
    base_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    effective_base_dir = base_dir or Path(
        os.path.commonpath(
            [
                Path(question_bank_path).resolve(),
                Path(durable_sidecar_path).resolve(),
                Path(release_manifest_path).resolve(),
            ]
        )
    )
    try:
        return build_release_manifest(
            question_bank_path=question_bank_path,
            artifacts={TOPIC_ROUTING_ROLE: durable_sidecar_path},
            output_path=release_manifest_path,
            base_dir=effective_base_dir,
            generated_at=generated_at,
        )
    except ReleaseManifestError as exc:
        raise TopicRoutingArtifactError(str(exc)) from exc


def resolve_topic_routing_sidecar(
    *,
    question_bank_path: str | Path,
    requested_path: str | Path | None = None,
    release_manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST_PATH,
) -> Path | None:
    question_bank_path = Path(question_bank_path)
    release_manifest_path = Path(release_manifest_path)
    if requested_path is not None:
        candidate = Path(requested_path)
        if not candidate.is_file():
            raise TopicRoutingArtifactError(f"Missing requested topic-routing sidecar: {candidate}")
        if _is_production_artifact_path(candidate):
            if not release_manifest_path.is_file():
                raise TopicRoutingArtifactError(f"Missing release manifest: {release_manifest_path}")
            report = verify_topic_routing_artifact(
                question_bank_path=question_bank_path,
                local_sidecar_path=candidate if candidate == DEFAULT_LOCAL_SIDECAR_PATH else DEFAULT_LOCAL_SIDECAR_PATH,
                durable_sidecar_path=DEFAULT_DURABLE_SIDECAR_PATH,
                release_manifest_path=release_manifest_path,
            )
            if candidate == DEFAULT_LOCAL_SIDECAR_PATH and not report["sha256"]["local_matches_durable"]:
                raise TopicRoutingArtifactError("Requested local topic-routing sidecar is not the released artifact.")
        return candidate
    production_question_bank = question_bank_path.resolve() == DEFAULT_QUESTION_BANK_PATH.resolve()
    explicit_release_manifest = release_manifest_path.resolve() != DEFAULT_RELEASE_MANIFEST_PATH.resolve()
    if release_manifest_path.is_file() and (production_question_bank or explicit_release_manifest):
        try:
            released_bank = resolve_release_artifact(QUESTION_BANK_ROLE, manifest_path=release_manifest_path)
            released_routes = resolve_release_artifact(TOPIC_ROUTING_ROLE, manifest_path=release_manifest_path)
        except ReleaseManifestError as exc:
            raise TopicRoutingArtifactError(str(exc)) from exc
        if released_bank.resolve() != question_bank_path.resolve():
            raise TopicRoutingArtifactError("Release manifest is bound to a different question bank.")
        return released_routes
    sibling = question_bank_path.parent / DEFAULT_LOCAL_SIDECAR_PATH.name
    if not production_question_bank:
        return sibling if sibling.is_file() else None
    raise TopicRoutingArtifactError(f"Missing release manifest: {release_manifest_path}")


def should_enforce_production_topic_routing_provenance(path: str | Path | None) -> bool:
    if path is None:
        return False
    candidate = Path(path)
    if candidate in {DEFAULT_LOCAL_SIDECAR_PATH, DEFAULT_DURABLE_SIDECAR_PATH}:
        return True
    try:
        return candidate.resolve() in {
            DEFAULT_LOCAL_SIDECAR_PATH.resolve(),
            DEFAULT_DURABLE_SIDECAR_PATH.resolve(),
        }
    except FileNotFoundError:
        return False


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TopicRoutingArtifactError(f"Expected JSON object: {path}")
    return payload


def _is_production_artifact_path(path: Path) -> bool:
    return should_enforce_production_topic_routing_provenance(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore or verify the durable topic-routing sidecar artifact.")
    parser.add_argument("action", choices=["verify", "restore", "manifest"])
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK_PATH)
    parser.add_argument("--local-sidecar", type=Path, default=DEFAULT_LOCAL_SIDECAR_PATH)
    parser.add_argument("--durable-sidecar", type=Path, default=DEFAULT_DURABLE_SIDECAR_PATH)
    parser.add_argument("--durable-sha256", type=Path, default=DEFAULT_DURABLE_SHA256_PATH)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "manifest":
            report = build_topic_routing_release_manifest(
                question_bank_path=args.question_bank,
                durable_sidecar_path=args.durable_sidecar,
                release_manifest_path=args.release_manifest,
            )
        elif args.action == "restore":
            report = restore_topic_routing_sidecar(
                question_bank_path=args.question_bank,
                local_sidecar_path=args.local_sidecar,
                durable_sidecar_path=args.durable_sidecar,
                durable_sha256_path=args.durable_sha256,
                release_manifest_path=args.release_manifest,
            )
        else:
            report = verify_topic_routing_artifact(
                question_bank_path=args.question_bank,
                local_sidecar_path=args.local_sidecar,
                durable_sidecar_path=args.durable_sidecar,
                durable_sha256_path=args.durable_sha256,
                release_manifest_path=args.release_manifest,
                check_local_sidecar=False,
            )
    except TopicRoutingArtifactError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def manifest_main(argv: list[str] | None = None) -> int:
    return main(["manifest", *(argv or [])])


def verify_main(argv: list[str] | None = None) -> int:
    return main(["verify", *(argv or [])])


def restore_main(argv: list[str] | None = None) -> int:
    return main(["restore", *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main())
