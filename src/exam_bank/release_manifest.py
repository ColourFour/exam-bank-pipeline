from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .atomic_json import write_atomic_json
from .publication_safety import publication_read_guard


RELEASE_MANIFEST_SCHEMA_NAME = "exam_bank.question_bank_release_manifest"
RELEASE_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_RELEASE_MANIFEST_PATH = Path("manifests/releases/question_bank_release_manifest.v1.json")
QUESTION_BANK_ROLE = "question_bank"
TOPIC_ROUTING_ROLE = "topic_routing"
ASSET_MANIFEST_ROLE = "asset_manifest"
MARK_EVENTS_ROLE = "mark_events"
DIFFICULTY_INDEX_ROLE = "difficulty_index"
ASTERION_CATALOG_ROLE = "asterion_catalog"
ASTERION_RUNTIME_ROLE = "asterion_runtime"
ASTERION_CONTENT_LAB_ROLE = "asterion_content_lab"
PROMOTION_DECISIONS_ROLE = "promotion_decisions"
CORPUS_MANIFEST_ROLE = "corpus_manifest"

QUESTION_ID_COVERAGE_EXACT = "exact"
QUESTION_ID_COVERAGE_EXACT_SET = "exact_set"
QUESTION_ID_COVERAGE_SUBSET = "subset"
QUESTION_ID_COVERAGE_NONE = "none"
QUESTION_ID_COVERAGE_POLICIES = {
    QUESTION_ID_COVERAGE_EXACT,
    QUESTION_ID_COVERAGE_EXACT_SET,
    QUESTION_ID_COVERAGE_SUBSET,
    QUESTION_ID_COVERAGE_NONE,
}

_RECORD_COLLECTIONS = (
    "questions",
    "records",
    "items",
    "candidates",
    "decisions",
    "rubrics",
    "rows",
    "assets",
    "artifacts",
)
_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class ReleaseManifestError(RuntimeError):
    pass


class _StrictJsonError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_manifest(
    *,
    question_bank_path: str | Path,
    artifacts: Mapping[str, str | Path],
    output_path: str | Path = DEFAULT_RELEASE_MANIFEST_PATH,
    base_dir: str | Path | None = None,
    exact_question_id_roles: Iterable[str] = (TOPIC_ROUTING_ROLE,),
    question_id_coverage_by_role: Mapping[str, str] | None = None,
    dependencies_by_role: Mapping[str, Iterable[str]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    root = Path(base_dir or Path.cwd()).resolve()
    artifact_roles = list(artifacts)
    for role in artifact_roles:
        _validate_role(role, context="artifact")
        if role == QUESTION_BANK_ROLE:
            raise ReleaseManifestError(f"Invalid release artifact role: {role!r}")
    coverage_overrides = question_id_coverage_by_role or {}
    for role in coverage_overrides:
        _validate_role(role, context="coverage")
    with publication_read_guard(question_bank_path):
        question_bank_entry, question_bank_ids, question_bank_id_references = _build_artifact_entry(
            question_bank_path,
            root=root,
        )
    _validate_question_bank_entry(question_bank_entry)
    if question_bank_id_references != question_bank_entry["record_count"]:
        raise ReleaseManifestError(
            "question_bank must contain one non-empty question ID per record "
            f"({question_bank_id_references} references for {question_bank_entry['record_count']} records)"
        )
    if len(question_bank_ids) != question_bank_entry["record_count"]:
        raise ReleaseManifestError(
            "question_bank must contain one unique question ID per record "
            f"({len(question_bank_ids)} unique IDs for {question_bank_entry['record_count']} records)"
        )
    entries: dict[str, dict[str, Any]] = {QUESTION_BANK_ROLE: question_bank_entry}
    exact_roles = tuple(exact_question_id_roles)
    for role in exact_roles:
        _validate_role(role, context="exact-coverage")
    coverage_by_role = {
        role: QUESTION_ID_COVERAGE_EXACT for role in exact_roles
    }
    coverage_by_role.update(
        {str(role): str(policy) for role, policy in (question_id_coverage_by_role or {}).items()}
    )
    unknown_coverage_roles = sorted(set(coverage_overrides) - set(artifacts))
    if unknown_coverage_roles:
        raise ReleaseManifestError(
            f"Question-ID coverage refers to roles missing from the release: {unknown_coverage_roles}"
        )
    invalid_policies = {
        role: policy
        for role, policy in coverage_by_role.items()
        if policy not in QUESTION_ID_COVERAGE_POLICIES
    }
    if invalid_policies:
        raise ReleaseManifestError(f"Invalid question-ID coverage policies: {invalid_policies}")

    seen_paths = {
        (root / question_bank_entry["path"]).resolve(): QUESTION_BANK_ROLE,
    }
    for role, path in sorted(artifacts.items()):
        entry, question_ids, question_id_references = _build_artifact_entry(path, root=root)
        resolved_path = (root / entry["path"]).resolve()
        duplicate_role = seen_paths.get(resolved_path)
        if duplicate_role is not None:
            raise ReleaseManifestError(
                f"Release artifact path is assigned to multiple roles: {duplicate_role}, {role}"
            )
        seen_paths[resolved_path] = role
        entry["bound_question_bank_sha256"] = question_bank_entry["sha256"]
        declared_source_sha = entry.get("declared_source_question_bank_sha256")
        if declared_source_sha is not None and declared_source_sha != question_bank_entry["sha256"]:
            raise ReleaseManifestError(
                f"{role} declares a different source question-bank SHA-256: "
                f"{declared_source_sha} != {question_bank_entry['sha256']}"
            )
        coverage = coverage_by_role.get(role, QUESTION_ID_COVERAGE_NONE)
        if coverage == QUESTION_ID_COVERAGE_EXACT:
            if question_id_references != entry["record_count"]:
                raise ReleaseManifestError(
                    f"{role} must contain one non-empty question ID per record "
                    f"({question_id_references} references for {entry['record_count']} records)"
                )
            if len(question_ids) != entry["record_count"]:
                raise ReleaseManifestError(
                    f"{role} must contain one unique question ID per record "
                    f"({len(question_ids)} unique IDs for {entry['record_count']} records)"
                )
            if question_ids != question_bank_ids:
                missing = sorted(question_bank_ids - question_ids)
                extra = sorted(question_ids - question_bank_ids)
                raise ReleaseManifestError(
                    f"{role} question-ID coverage does not match question_bank "
                    f"(missing={len(missing)}, extra={len(extra)})"
                )
        elif coverage == QUESTION_ID_COVERAGE_EXACT_SET:
            if question_id_references != entry["record_count"]:
                raise ReleaseManifestError(
                    f"{role} must contain a non-empty question ID on every record "
                    f"({question_id_references} references for {entry['record_count']} records)"
                )
            if question_ids != question_bank_ids:
                missing = sorted(question_bank_ids - question_ids)
                extra = sorted(question_ids - question_bank_ids)
                raise ReleaseManifestError(
                    f"{role} question-ID set does not match question_bank "
                    f"(missing={len(missing)}, extra={len(extra)})"
                )
        elif coverage == QUESTION_ID_COVERAGE_SUBSET:
            if question_id_references != entry["record_count"]:
                raise ReleaseManifestError(
                    f"{role} must contain a non-empty question ID on every record "
                    f"({question_id_references} references for {entry['record_count']} records)"
                )
            extra = sorted(question_ids - question_bank_ids)
            if extra:
                raise ReleaseManifestError(
                    f"{role} question-ID coverage is not a subset of question_bank "
                    f"(extra={len(extra)})"
                )
        entry["question_id_coverage"] = coverage
        entries[role] = entry

    dependencies: dict[str, tuple[str, ...]] = {}
    for role, role_dependencies in (dependencies_by_role or {}).items():
        _validate_role(role, context="dependency owner")
        if isinstance(role_dependencies, (str, bytes)):
            raise ReleaseManifestError(f"{role} dependencies must be an iterable of roles")
        try:
            dependencies[role] = tuple(role_dependencies)
        except TypeError as exc:
            raise ReleaseManifestError(
                f"{role} dependencies must be an iterable of roles"
            ) from exc
    for role, role_dependencies in sorted(dependencies.items()):
        for dependency in role_dependencies:
            _validate_role(dependency, context=f"{role} dependency")
        if role not in entries or role == QUESTION_BANK_ROLE:
            raise ReleaseManifestError(f"Dependency owner is not a release artifact role: {role!r}")
        if not role_dependencies:
            raise ReleaseManifestError(f"{role} has an empty dependency list")
        if len(set(role_dependencies)) != len(role_dependencies):
            raise ReleaseManifestError(f"{role} contains duplicate dependency roles")
        missing_dependencies = sorted(set(role_dependencies) - set(entries))
        if missing_dependencies:
            raise ReleaseManifestError(
                f"{role} depends on roles missing from the release: {missing_dependencies}"
            )
        if role in role_dependencies:
            raise ReleaseManifestError(f"Release artifact cannot depend on itself: {role}")
        entries[role]["bound_artifact_sha256"] = {
            dependency: entries[dependency]["sha256"]
            for dependency in sorted(set(role_dependencies))
        }

    for role, entry in entries.items():
        if role != QUESTION_BANK_ROLE:
            _verify_declared_dependency_hashes(role, entry, entries)
    _verify_corpus_manifest_binding(entries)

    manifest = {
        "schema_name": RELEASE_MANIFEST_SCHEMA_NAME,
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "artifact_root": os.path.relpath(root, output_path.parent.resolve()),
        "artifacts": entries,
    }
    manifest["release_id"] = _canonical_sha256(entries)[:16]
    output_resolved = output_path.resolve()
    colliding_role = seen_paths.get(output_resolved)
    if colliding_role is not None:
        raise ReleaseManifestError(
            f"Release manifest output would overwrite the {colliding_role} artifact: {output_resolved}"
        )
    write_atomic_json(manifest, output_path, sort_keys=True)
    return manifest


def verify_release_manifest(
    manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST_PATH,
    *,
    required_roles: Iterable[str] = (),
    require_validation_ok: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    required_roles = tuple(required_roles)
    if not manifest_path.is_file():
        raise ReleaseManifestError(f"Missing release manifest: {manifest_path}")
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_name") != RELEASE_MANIFEST_SCHEMA_NAME:
        raise ReleaseManifestError(f"Unexpected release manifest schema: {manifest.get('schema_name')!r}")
    if manifest.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ReleaseManifestError(f"Unsupported release manifest version: {manifest.get('schema_version')!r}")
    entries = manifest.get("artifacts")
    if not isinstance(entries, dict):
        raise ReleaseManifestError("Release manifest is missing artifacts.")
    for role in entries:
        _validate_role(role, context="manifest artifact")
    for role in required_roles:
        _validate_role(role, context="required")
    missing_roles = sorted(set(required_roles) - set(entries))
    if missing_roles:
        raise ReleaseManifestError(f"Release manifest is missing required roles: {missing_roles}")

    root = _release_root(manifest_path, manifest)
    verified: dict[str, dict[str, Any]] = {}
    id_sets: dict[str, set[str]] = {}
    id_reference_counts: dict[str, int] = {}
    declared_ok_by_role: dict[str, bool | None] = {}
    for role, raw_entry in sorted(entries.items()):
        if not isinstance(raw_entry, dict):
            raise ReleaseManifestError(f"Invalid release artifact entry: {role}")
        path_value = raw_entry.get("path")
        expected_sha = raw_entry.get("sha256")
        expected_size = raw_entry.get("size_bytes")
        expected_count = raw_entry.get("record_count")
        if not isinstance(path_value, str) or not path_value or Path(path_value).is_absolute():
            raise ReleaseManifestError(f"Release artifact path must be non-empty and relative: {role}")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
        ):
            raise ReleaseManifestError(f"Invalid release artifact SHA-256: {role}")
        if not _is_non_negative_int(expected_size):
            raise ReleaseManifestError(f"Invalid release artifact size: {role}")
        if not _is_non_negative_int(expected_count):
            raise ReleaseManifestError(f"Invalid release artifact record count: {role}")
        for count_field in ("question_id_count", "question_id_reference_count"):
            if count_field not in raw_entry or not _is_non_negative_int(raw_entry.get(count_field)):
                raise ReleaseManifestError(f"Invalid release artifact {count_field}: {role}")
        path = (root / path_value).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ReleaseManifestError(f"Release artifact escapes artifact_root: {role}") from exc
        if not path.is_file():
            raise ReleaseManifestError(f"Missing release artifact for {role}: {path}")
        if role == QUESTION_BANK_ROLE:
            with publication_read_guard(path):
                payload, actual_sha, actual_size = _read_json_snapshot(path)
        else:
            payload, actual_sha, actual_size = _read_json_snapshot(path)
        if actual_sha != expected_sha:
            raise ReleaseManifestError(f"{role} SHA-256 mismatch: {actual_sha} != {expected_sha}")
        if actual_size != expected_size:
            raise ReleaseManifestError(f"{role} size mismatch: {actual_size} != {expected_size}")
        actual_count, question_ids, question_id_references, collection_name = _payload_record_info(payload)
        declared_count = _declared_record_count(payload, collection_name)
        _validate_declared_record_count(
            declared_count,
            actual_count,
            path=path,
        )
        actual_schema_name, actual_schema_version = _schema_info(payload, path=path)
        expected_schema_name = raw_entry.get("schema_name")
        expected_schema_version = raw_entry.get("schema_version")
        if (
            type(expected_schema_name) is not type(actual_schema_name)
            or expected_schema_name != actual_schema_name
        ):
            raise ReleaseManifestError(
                f"{role} schema-name mismatch: {actual_schema_name!r} != {expected_schema_name!r}"
            )
        if (
            type(expected_schema_version) is not type(actual_schema_version)
            or expected_schema_version != actual_schema_version
        ):
            raise ReleaseManifestError(
                f"{role} schema-version mismatch: {actual_schema_version!r} != {expected_schema_version!r}"
            )
        if expected_count != actual_count:
            raise ReleaseManifestError(
                f"{role} record-count mismatch: {actual_count} != {expected_count}"
            )
        actual_id_sha = _question_id_set_sha256(question_ids)
        if raw_entry.get("question_id_set_sha256") != actual_id_sha:
            raise ReleaseManifestError(f"{role} question-ID set SHA-256 mismatch")
        if raw_entry.get("question_id_count") != len(question_ids):
            raise ReleaseManifestError(f"{role} question-ID count mismatch")
        if (
            raw_entry.get("question_id_reference_count") != question_id_references
        ):
            raise ReleaseManifestError(f"{role} question-ID reference count mismatch")
        if "record_collection" not in raw_entry:
            raise ReleaseManifestError(f"{role} is missing record-collection metadata")
        if raw_entry.get("record_collection") != collection_name:
            raise ReleaseManifestError(f"{role} record-collection mismatch")
        declared_source_sha = _declared_source_question_bank_sha256(payload)
        if raw_entry.get("declared_source_question_bank_sha256") != declared_source_sha:
            raise ReleaseManifestError(f"{role} declared source question-bank SHA-256 mismatch")
        declared_artifact_sha = _declared_artifact_sha256(payload)
        if raw_entry.get("declared_artifact_sha256", {}) != declared_artifact_sha:
            raise ReleaseManifestError(f"{role} declared dependency SHA-256 metadata mismatch")
        declared_corpus_sha = _declared_corpus_manifest_sha256(payload)
        if raw_entry.get("declared_corpus_manifest_sha256") != declared_corpus_sha:
            raise ReleaseManifestError(f"{role} declared corpus-manifest SHA-256 mismatch")
        verified[role] = {
            "path": path,
            "sha256": actual_sha,
            "size_bytes": actual_size,
            "record_count": actual_count,
            "schema_name": actual_schema_name,
            "schema_version": actual_schema_version,
            "record_collection": collection_name,
            "question_id_count": len(question_ids),
        }
        if "ok" in payload:
            declared_ok = payload.get("ok")
            declared_ok_by_role[role] = declared_ok if isinstance(declared_ok, bool) else None
            verified[role]["declared_ok"] = declared_ok_by_role[role]
        id_sets[role] = question_ids
        id_reference_counts[role] = question_id_references

    question_bank_entry = entries.get(QUESTION_BANK_ROLE)
    if not isinstance(question_bank_entry, dict):
        raise ReleaseManifestError("Release manifest is missing question_bank.")
    _validate_question_bank_entry(question_bank_entry)
    question_bank_sha = question_bank_entry.get("sha256")
    question_bank_ids = id_sets.get(QUESTION_BANK_ROLE, set())
    question_bank_record_count = question_bank_entry.get("record_count")
    if (
        id_reference_counts.get(QUESTION_BANK_ROLE, 0) != question_bank_record_count
        or len(question_bank_ids) != question_bank_record_count
    ):
        raise ReleaseManifestError("question_bank does not contain one unique question ID per record")
    for role, raw_entry in entries.items():
        if role == QUESTION_BANK_ROLE or not isinstance(raw_entry, dict):
            continue
        if raw_entry.get("bound_question_bank_sha256") != question_bank_sha:
            raise ReleaseManifestError(f"{role} is not bound to the manifest question_bank SHA-256")
        declared_source_sha = raw_entry.get("declared_source_question_bank_sha256")
        if declared_source_sha is not None and declared_source_sha != question_bank_sha:
            raise ReleaseManifestError(f"{role} declares a different source question-bank SHA-256")
        if "question_id_coverage" not in raw_entry:
            raise ReleaseManifestError(f"{role} is missing question-ID coverage metadata")
        coverage = raw_entry.get("question_id_coverage")
        if coverage not in QUESTION_ID_COVERAGE_POLICIES:
            raise ReleaseManifestError(f"{role} has invalid question-ID coverage: {coverage!r}")
        if coverage == QUESTION_ID_COVERAGE_EXACT:
            if id_sets.get(role, set()) != question_bank_ids:
                raise ReleaseManifestError(f"{role} no longer has exact question-ID coverage")
            if (
                id_reference_counts.get(role, 0) != raw_entry.get("record_count")
                or len(id_sets.get(role, set())) != raw_entry.get("record_count")
            ):
                raise ReleaseManifestError(f"{role} no longer has one unique question ID per record")
        elif coverage == QUESTION_ID_COVERAGE_EXACT_SET:
            if id_sets.get(role, set()) != question_bank_ids:
                raise ReleaseManifestError(f"{role} no longer has exact question-ID-set coverage")
            if id_reference_counts.get(role, 0) != raw_entry.get("record_count"):
                raise ReleaseManifestError(f"{role} no longer has a question ID on every record")
        elif coverage == QUESTION_ID_COVERAGE_SUBSET:
            if id_reference_counts.get(role, 0) != raw_entry.get("record_count"):
                raise ReleaseManifestError(f"{role} no longer has a question ID on every record")
            if not id_sets.get(role, set()).issubset(question_bank_ids):
                raise ReleaseManifestError(f"{role} no longer has subset question-ID coverage")
        bound_artifacts = raw_entry.get("bound_artifact_sha256", {})
        if not isinstance(bound_artifacts, dict):
            raise ReleaseManifestError(f"{role} has invalid bound_artifact_sha256 metadata")
        for dependency, expected_sha in sorted(bound_artifacts.items()):
            _validate_role(dependency, context=f"{role} dependency")
            dependency_entry = entries.get(dependency)
            if not isinstance(dependency_entry, dict):
                raise ReleaseManifestError(f"{role} depends on missing release role: {dependency}")
            if dependency == role:
                raise ReleaseManifestError(f"Release artifact cannot depend on itself: {role}")
            if expected_sha != dependency_entry.get("sha256"):
                raise ReleaseManifestError(f"{role} dependency hash does not match release role: {dependency}")
        _verify_declared_dependency_hashes(role, raw_entry, entries)

    _verify_corpus_manifest_binding(entries)

    expected_release_id = _canonical_sha256(entries)[:16]
    if manifest.get("release_id") != expected_release_id:
        raise ReleaseManifestError("Release manifest release_id does not match its artifact entries")
    validation_roles = sorted(role for role in entries if role.endswith("_validation"))
    validation_statuses = {
        role: declared_ok_by_role.get(role)
        for role in validation_roles
    }
    blocking_validation_roles = [
        role for role, ok in validation_statuses.items() if ok is not True
    ]
    policy_ok = not blocking_validation_roles
    if require_validation_ok and not policy_ok:
        raise ReleaseManifestError(
            "Release validation artifacts are not all ok:true: "
            f"{blocking_validation_roles}"
        )
    return {
        "ok": True,
        "provenance_ok": True,
        "policy_ok": policy_ok,
        "validation_statuses": validation_statuses,
        "blocking_validation_roles": blocking_validation_roles,
        "manifest_path": manifest_path,
        "artifact_root": root,
        "release_id": expected_release_id,
        "artifacts": verified,
    }


def resolve_release_artifact(
    role: str,
    *,
    manifest_path: str | Path = DEFAULT_RELEASE_MANIFEST_PATH,
) -> Path:
    report = verify_release_manifest(manifest_path, required_roles=(QUESTION_BANK_ROLE, role))
    return report["artifacts"][role]["path"]


def _build_artifact_entry(path_value: str | Path, *, root: Path) -> tuple[dict[str, Any], set[str], int]:
    path = Path(path_value).resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        raise ReleaseManifestError(f"Release artifacts must be inside artifact_root {root}: {path}") from exc
    if not path.is_file():
        raise ReleaseManifestError(f"Missing release artifact: {path}")
    payload, artifact_sha, artifact_size = _read_json_snapshot(path)
    record_count, question_ids, question_id_references, collection_name = _payload_record_info(payload)
    declared_count = _declared_record_count(payload, collection_name)
    _validate_declared_record_count(declared_count, record_count, path=path)
    schema_name, schema_version = _schema_info(payload, path=path)
    return (
        {
            "path": relative_path.as_posix(),
            "sha256": artifact_sha,
            "size_bytes": artifact_size,
            "schema_name": schema_name,
            "schema_version": schema_version,
            "record_collection": collection_name,
            "record_count": record_count,
            "question_id_count": len(question_ids),
            "question_id_reference_count": question_id_references,
            "question_id_set_sha256": _question_id_set_sha256(question_ids),
            "declared_source_question_bank_sha256": _declared_source_question_bank_sha256(payload),
            "declared_artifact_sha256": _declared_artifact_sha256(payload),
            "declared_corpus_manifest_sha256": _declared_corpus_manifest_sha256(payload),
        },
        question_ids,
        question_id_references,
    )


def _payload_record_info(payload: dict[str, Any]) -> tuple[int, set[str], int, str | None]:
    present_collections = [name for name in _RECORD_COLLECTIONS if name in payload]
    if len(present_collections) > 1:
        raise ReleaseManifestError(
            f"Release artifact has multiple record collections: {present_collections}"
        )
    if not present_collections:
        declared_count = payload.get("record_count")
        metadata_count = declared_count if _is_non_negative_int(declared_count) else 0
        return metadata_count, set(), 0, None
    collection_name = present_collections[0]
    if not isinstance(payload[collection_name], (list, dict)):
        raise ReleaseManifestError(
            f"Release artifact record collection must be a list or object: {collection_name}"
        )
    records: Any = payload[collection_name]
    if isinstance(records, dict):
        question_id_values = [
            question_id
            for key, record in records.items()
            if isinstance(record, dict)
            and (question_id := _record_question_id(record, fallback=key, allow_id=False)) is not None
        ]
        return len(records), set(question_id_values), len(question_id_values), collection_name
    if isinstance(records, list):
        allow_id = collection_name in {"records", "items", "rows"}
        question_id_values = [
            question_id
            for record in records
            if isinstance(record, dict)
            and (question_id := _record_question_id(record, allow_id=allow_id)) is not None
        ]
        return len(records), set(question_id_values), len(question_id_values), collection_name
    raise AssertionError("record collection selection returned an unsupported value")


def _declared_record_count(payload: dict[str, Any], collection_name: str | None) -> Any:
    declared = payload.get("record_count")
    if declared is None and collection_name == "decisions":
        declared = payload.get("decision_count")
    if declared is None and collection_name == "candidates":
        declared = payload.get("candidate_count")
    if declared is None and collection_name == "assets":
        declared = payload.get("asset_count")
    return declared


def _record_question_id(
    record: Mapping[str, Any],
    *,
    fallback: Any = None,
    allow_id: bool,
) -> str | None:
    if "question_id" in record:
        value = record.get("question_id")
    elif "source_question_id" in record:
        value = record.get("source_question_id")
    elif allow_id and "id" in record:
        value = record.get("id")
    else:
        value = fallback
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return value


def _schema_info(payload: Mapping[str, Any], *, path: Path) -> tuple[str | None, int | None]:
    schema_name = payload.get("schema_name")
    schema_alias = payload.get("schema")
    for field, value in (("schema_name", schema_name), ("schema", schema_alias)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ReleaseManifestError(f"Invalid {field} at {path}")
    if schema_name is not None and schema_alias is not None and schema_name != schema_alias:
        raise ReleaseManifestError(
            f"Conflicting schema_name and schema declarations at {path}: "
            f"{schema_name!r} != {schema_alias!r}"
        )
    schema_version = payload.get("schema_version")
    if schema_version is not None and (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version <= 0
    ):
        raise ReleaseManifestError(f"Invalid schema_version at {path}: {schema_version!r}")
    return schema_name or schema_alias, schema_version


def _validate_declared_record_count(declared: Any, actual: int, *, path: Path) -> None:
    if declared is None:
        return
    if not _is_non_negative_int(declared):
        raise ReleaseManifestError(f"Invalid declared record count at {path}: {declared!r}")
    if declared != actual:
        raise ReleaseManifestError(
            f"Declared record count does not match payload at {path}: {declared} != {actual}"
        )


def _validate_question_bank_entry(entry: Mapping[str, Any]) -> None:
    if entry.get("schema_name") != "exam_bank.question_bank":
        raise ReleaseManifestError(
            f"Unexpected question_bank schema: {entry.get('schema_name')!r}"
        )
    if entry.get("schema_version") != 2:
        raise ReleaseManifestError(
            f"Unsupported question_bank schema version: {entry.get('schema_version')!r}"
        )
    if entry.get("record_collection") != "questions":
        raise ReleaseManifestError("question_bank must use the questions record collection")
    if not _is_non_negative_int(entry.get("record_count")) or entry.get("record_count") == 0:
        raise ReleaseManifestError("question_bank must contain at least one record")


def _declared_corpus_manifest_sha256(payload: Mapping[str, Any]) -> str | None:
    if payload.get("schema_name") != "exam_bank.question_bank":
        return None
    run_manifest = payload.get("run_manifest")
    if run_manifest is None:
        return None
    if not isinstance(run_manifest, dict):
        raise ReleaseManifestError("question_bank run_manifest must be an object")
    value = run_manifest.get("corpus_manifest_sha256")
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ReleaseManifestError("question_bank declares an invalid corpus-manifest SHA-256")
    return value


def _verify_corpus_manifest_binding(entries: Mapping[str, Mapping[str, Any]]) -> None:
    corpus_entry = entries.get(CORPUS_MANIFEST_ROLE)
    if corpus_entry is None:
        return
    question_bank_entry = entries.get(QUESTION_BANK_ROLE)
    if not isinstance(question_bank_entry, Mapping):
        raise ReleaseManifestError("Release manifest is missing question_bank.")
    declared = question_bank_entry.get("declared_corpus_manifest_sha256")
    if declared is None:
        raise ReleaseManifestError(
            "question_bank must declare run_manifest.corpus_manifest_sha256 when corpus_manifest is released"
        )
    if declared != corpus_entry.get("sha256"):
        raise ReleaseManifestError(
            "question_bank corpus-manifest SHA-256 does not match the released corpus_manifest"
        )


def _validate_role(role: Any, *, context: str) -> None:
    if not isinstance(role, str) or _ROLE_PATTERN.fullmatch(role) is None:
        raise ReleaseManifestError(f"Invalid {context} role: {role!r}")


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _declared_source_question_bank_sha256(payload: dict[str, Any]) -> str | None:
    value = payload.get("source_question_bank_sha256")
    if value is None:
        return None
    value = str(value).strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReleaseManifestError("Invalid declared source_question_bank_sha256")
    return value


def _declared_artifact_sha256(payload: dict[str, Any]) -> dict[str, str]:
    declared: dict[str, str] = {}
    for field, value in payload.items():
        if not field.startswith("source_") or not field.endswith("_sha256"):
            continue
        role = field.removeprefix("source_").removesuffix("_sha256")
        if role == QUESTION_BANK_ROLE or value in (None, ""):
            continue
        _add_declared_artifact_sha256(declared, role, value, field=field)

    source_sidecars = payload.get("source_sidecars")
    if isinstance(source_sidecars, dict):
        for field, value in source_sidecars.items():
            if not field.endswith("_sha256") or value in (None, ""):
                continue
            role = field.removesuffix("_sha256")
            _add_declared_artifact_sha256(
                declared,
                role,
                value,
                field=f"source_sidecars.{field}",
            )

    release_inputs = payload.get("release_inputs")
    if isinstance(release_inputs, dict):
        durable_sidecar = release_inputs.get("durable_sidecar")
        if isinstance(durable_sidecar, dict) and durable_sidecar.get("sha256") not in (None, ""):
            _add_declared_artifact_sha256(
                declared,
                TOPIC_ROUTING_ROLE,
                durable_sidecar["sha256"],
                field="release_inputs.durable_sidecar.sha256",
            )

    export_artifacts = payload.get("export_artifacts")
    if isinstance(export_artifacts, dict):
        aliases = {
            "catalog": ASTERION_CATALOG_ROLE,
            "student_runtime": ASTERION_RUNTIME_ROLE,
            "content_lab_candidates": ASTERION_CONTENT_LAB_ROLE,
        }
        for artifact_name, role in aliases.items():
            artifact = export_artifacts.get(artifact_name)
            if isinstance(artifact, dict) and artifact.get("sha256") not in (None, ""):
                _add_declared_artifact_sha256(
                    declared,
                    role,
                    artifact["sha256"],
                    field=f"export_artifacts.{artifact_name}.sha256",
                )
    return dict(sorted(declared.items()))


def _add_declared_artifact_sha256(
    declared: dict[str, str],
    role: str,
    value: Any,
    *,
    field: str,
) -> None:
    _validate_role(role, context="declared dependency")
    normalized = _normalized_sha256(value, field=field)
    previous = declared.get(role)
    if previous is not None and previous != normalized:
        raise ReleaseManifestError(
            f"Conflicting declared SHA-256 values for {role}: {previous} != {normalized}"
        )
    declared[role] = normalized


def _verify_declared_dependency_hashes(
    role: str,
    entry: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    declared = entry.get("declared_artifact_sha256", {})
    if not isinstance(declared, dict):
        raise ReleaseManifestError(f"{role} has invalid declared_artifact_sha256 metadata")
    for dependency, declared_sha in sorted(declared.items()):
        dependency_entry = entries.get(dependency)
        if dependency_entry is None:
            continue
        if declared_sha != dependency_entry.get("sha256"):
            raise ReleaseManifestError(
                f"{role} declares a stale SHA-256 for release role {dependency}: "
                f"{declared_sha} != {dependency_entry.get('sha256')}"
            )


def _normalized_sha256(value: Any, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ReleaseManifestError(f"Invalid declared SHA-256 in {field}")
    return normalized


def _release_root(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    root_value = manifest.get("artifact_root")
    if not isinstance(root_value, str) or not root_value or Path(root_value).is_absolute():
        raise ReleaseManifestError("Release manifest artifact_root must be non-empty and relative")
    return (manifest_path.parent.resolve() / root_value).resolve()


def _question_id_set_sha256(question_ids: set[str]) -> str:
    return _canonical_sha256(sorted(question_ids))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    return _read_json_snapshot(path)[0]


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str, int]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _StrictJsonError) as exc:
        raise ReleaseManifestError(f"Cannot read JSON release artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseManifestError(f"Expected JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest(), len(raw)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise _StrictJsonError(f"duplicate object key {key!r}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise _StrictJsonError(f"non-standard JSON constant {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify a hash-bound canonical question-bank release bundle."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser("build", help="Build a release manifest from named JSON artifacts.")
    build.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    build.add_argument("--output", type=Path, default=DEFAULT_RELEASE_MANIFEST_PATH)
    build.add_argument("--base-dir", type=Path, default=Path.cwd())
    build.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Add a JSON release artifact. Repeat for each role.",
    )
    build.add_argument(
        "--coverage",
        action="append",
        default=[],
        metavar="ROLE=POLICY",
        help="Set question-ID coverage to exact, exact_set, subset, or none. Repeat as needed.",
    )
    build.add_argument(
        "--depends-on",
        action="append",
        default=[],
        metavar="ROLE=ROLE[,ROLE...]",
        help="Bind one artifact to the hashes of other roles in the same release.",
    )
    build.add_argument("--generated-at", default=None)

    verify = subparsers.add_parser("verify", help="Verify every artifact and dependency in a release manifest.")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST_PATH)
    verify.add_argument("--require-role", action="append", default=[])
    verify.add_argument(
        "--require-validation-ok",
        action="store_true",
        help="Fail unless every role ending in _validation contains ok:true.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "build":
            artifacts = _parse_role_assignments(args.artifact, option="--artifact")
            coverage = _parse_role_assignments(args.coverage, option="--coverage")
            dependencies = {
                role: tuple(dependency.strip() for dependency in value.split(",") if dependency.strip())
                for role, value in _parse_role_assignments(
                    args.depends_on,
                    option="--depends-on",
                ).items()
            }
            report = build_release_manifest(
                question_bank_path=args.question_bank,
                artifacts=artifacts,
                output_path=args.output,
                base_dir=args.base_dir,
                question_id_coverage_by_role=coverage,
                dependencies_by_role=dependencies,
                generated_at=args.generated_at,
            )
        else:
            report = verify_release_manifest(
                args.manifest,
                required_roles=args.require_role,
                require_validation_ok=args.require_validation_ok,
            )
    except ReleaseManifestError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


def run_build(argv: list[str] | None = None) -> int:
    """Run release-manifest build through the unified command surface."""

    return main(["build", *(argv or [])])


def run_verify(argv: list[str] | None = None) -> int:
    """Run release-manifest verification through the unified command surface."""

    return main(["verify", *(argv or [])])


def _parse_role_assignments(values: Iterable[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_value in values:
        role, separator, value = str(raw_value).partition("=")
        role = role.strip()
        value = value.strip()
        if not separator or not role or not value:
            raise ReleaseManifestError(f"{option} requires ROLE=VALUE, got {raw_value!r}")
        if role in parsed:
            raise ReleaseManifestError(f"Duplicate {option} role: {role}")
        parsed[role] = value
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
