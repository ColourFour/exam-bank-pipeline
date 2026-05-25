from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .atomic_json import write_atomic_json
from .topic_routing import audit_topic_routing_sidecar_payload


ASSET_MANIFEST_SCHEMA_NAME = "exam_bank.asset_manifest"
ASSET_MANIFEST_SCHEMA_VERSION = 1
ASSET_MANIFEST_FILENAME = "asset_manifest.v1.json"

QUESTION_IMAGE_KIND = "question_image"
MARK_SCHEME_IMAGE_KIND = "mark_scheme_image"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def asset_id_for_record(kind: str, record: dict[str, Any], path: str, *, occurrence: int = 0) -> str:
    paper = _slug(str(record.get("paper") or "unknown_paper"))
    question_id = _slug(str(record.get("question_id") or "unknown_question"))
    base = f"{kind}:{paper}:{question_id}"
    if occurrence:
        return f"{base}:{occurrence + 1}"
    if paper == "unknown_paper" or question_id == "unknown_question":
        return f"{base}:{hashlib.sha256(path.encode('utf-8')).hexdigest()[:12]}"
    return base


def build_asset_manifest(
    question_bank_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(question_bank_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(artifact_root) if artifact_root is not None else _infer_artifact_root(path)
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    return build_asset_manifest_payload(
        payload,
        artifact_root=root,
        base_dir=base,
        question_bank_path=path,
    )


def build_asset_manifest_payload(
    question_bank: dict[str, Any],
    *,
    artifact_root: str | Path,
    base_dir: str | Path | None = None,
    question_bank_path: str | Path | None = None,
) -> dict[str, Any]:
    if question_bank.get("schema_name") != "exam_bank.question_bank":
        raise ValueError("Asset manifest requires exam_bank.question_bank input")
    root = Path(artifact_root)
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    source_path = _display_path(Path(question_bank_path), base) if question_bank_path is not None else None

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for question in question_bank.get("questions", []):
        if not isinstance(question, dict):
            continue
        records.extend(
            _asset_records_for_question(
                question,
                kind=QUESTION_IMAGE_KIND,
                paths=_question_image_paths(question),
                artifact_root=root,
                base_dir=base,
                source_path=source_path,
                source_fields=["canonical_question_artifact", "question_image_path", "question_image_paths"],
                seen_ids=seen_ids,
            )
        )
        records.extend(
            _asset_records_for_question(
                question,
                kind=MARK_SCHEME_IMAGE_KIND,
                paths=_mark_scheme_image_paths(question),
                artifact_root=root,
                base_dir=base,
                source_path=source_path,
                source_fields=["canonical_mark_scheme_artifact", "mark_scheme_image_path", "mark_scheme_image_paths"],
                seen_ids=seen_ids,
            )
        )

    records.sort(key=lambda item: item["asset_id"])
    return {
        "schema_name": ASSET_MANIFEST_SCHEMA_NAME,
        "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "source_schema": {
            "schema_name": question_bank.get("schema_name"),
            "schema_version": question_bank.get("schema_version"),
            "record_count": question_bank.get("record_count"),
        },
        "source_question_bank_path": source_path,
        "artifact_root": _display_path(root, base),
        "asset_count": len(records),
        "assets": records,
    }


def write_asset_manifest(
    question_bank_path: str | Path,
    output_path: str | Path | None = None,
    *,
    artifact_root: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    input_path = Path(question_bank_path)
    output = Path(output_path) if output_path is not None else input_path.parent / ASSET_MANIFEST_FILENAME
    payload = build_asset_manifest(input_path, artifact_root=artifact_root, base_dir=base_dir)
    return write_atomic_json(payload, output, sort_keys=True)


def validate_asset_references(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    asset_manifest_path: str | Path = "output/json/asset_manifest.v1.json",
    asterion_path: str | Path = "output/asterion/exports/latest/asterion_question_bank_v1.json",
    content_lab_path: str | Path = "output/asterion/exports/latest/asterion_content_lab_candidates_v1.json",
    topic_routing_path: str | Path = "output/json/question_bank.topic_routing.v1.json",
    artifact_root: str | Path = "output",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(project_root) if project_root is not None else Path.cwd()
    root = Path(artifact_root)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    question_bank = _read_json_if_exists(Path(question_bank_path), errors, "question_bank")
    manifest = _read_json_if_exists(Path(asset_manifest_path), errors, "asset_manifest")
    asterion = _read_json_if_exists(Path(asterion_path), errors, "asterion_question_bank")
    content_lab = _read_json_if_exists(Path(content_lab_path), errors, "content_lab_candidates")
    topic_routing = _read_json_if_exists(Path(topic_routing_path), errors, "topic_routing")

    manifest_ids: set[str] = set()
    manifest_paths: set[str] = set()
    manifest_sha_by_path: dict[str, str] = {}
    if manifest:
        if manifest.get("schema_name") != ASSET_MANIFEST_SCHEMA_NAME:
            errors.append(_issue("asset_manifest_schema", str(asset_manifest_path), "Unexpected asset manifest schema."))
        assets = manifest.get("assets") if isinstance(manifest.get("assets"), list) else []
        manifest_id_values = [str(asset.get("asset_id")) for asset in assets if isinstance(asset, dict) and asset.get("asset_id")]
        manifest_ids = set(manifest_id_values)
        manifest_paths = {str(asset.get("canonical_path")) for asset in assets if isinstance(asset, dict) and asset.get("canonical_path")}
        manifest_sha_by_path = {
            str(asset.get("canonical_path")): str(asset.get("sha256") or "")
            for asset in assets
            if isinstance(asset, dict) and asset.get("canonical_path")
        }
        declared = manifest.get("asset_count")
        if isinstance(declared, int) and declared != len(assets):
            errors.append(_issue("asset_manifest_count", str(asset_manifest_path), "asset_count does not match assets length."))

    if question_bank:
        if question_bank.get("schema_name") != "exam_bank.question_bank":
            errors.append(_issue("question_bank_schema", str(question_bank_path), "Unexpected question-bank schema."))
        questions = question_bank.get("questions") if isinstance(question_bank.get("questions"), list) else []
        _check_record_count(question_bank, questions, errors, str(question_bank_path))
        for question in questions:
            if not isinstance(question, dict):
                continue
            for field, value in _question_bank_image_field_values(question):
                _check_path_exists(
                    value,
                    root=root,
                    base=base,
                    errors=errors,
                    code="missing_question_bank_image_path",
                    source=str(question_bank_path),
                    detail={"question_id": question.get("question_id"), "field": field},
                )

    if asterion:
        if asterion.get("schema_name") != "asterion.question_bank":
            errors.append(_issue("asterion_schema", str(asterion_path), "Unexpected Asterion question-bank schema."))
        questions = asterion.get("questions") if isinstance(asterion.get("questions"), list) else []
        _check_record_count(asterion, questions, errors, str(asterion_path))
        for question in questions:
            if not isinstance(question, dict):
                continue
            for field in ["canonical_question_artifact", "canonical_mark_scheme_artifact"]:
                _check_path_exists(
                    question.get(field),
                    root=root,
                    base=base,
                    errors=errors,
                    code="missing_asterion_image_path",
                    source=str(asterion_path),
                    detail={"question_id": question.get("question_id"), "field": field},
                )
            for field in ["canonical_question_asset_id", "canonical_mark_scheme_asset_id"]:
                _check_asset_id(question.get(field), manifest_ids, errors, str(asterion_path), question.get("question_id"), field)
            for subpart in question.get("subparts", []):
                if not isinstance(subpart, dict):
                    continue
                for field in ["question_asset_id", "mark_scheme_asset_id"]:
                    _check_asset_id(subpart.get(field), manifest_ids, errors, str(asterion_path), subpart.get("subpart_id"), field)

    if content_lab:
        if content_lab.get("schema_name") != "asterion.content_lab_candidates":
            errors.append(_issue("content_lab_schema", str(content_lab_path), "Unexpected Content Lab schema."))
        candidates = content_lab.get("candidates") if isinstance(content_lab.get("candidates"), list) else []
        _check_record_count(content_lab, candidates, errors, str(content_lab_path))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            artifacts = candidate.get("source_artifacts") if isinstance(candidate.get("source_artifacts"), dict) else {}
            for field in ["question_crop_path", "mark_scheme_crop_path"]:
                _check_path_exists(
                    artifacts.get(field),
                    root=root,
                    base=base,
                    errors=errors,
                    code="missing_content_lab_image_path",
                    source=str(content_lab_path),
                    detail={"candidate_id": candidate.get("candidate_id"), "field": field},
                )
            for field in ["question_asset_id", "mark_scheme_asset_id"]:
                _check_asset_id(artifacts.get(field), manifest_ids, errors, str(content_lab_path), candidate.get("candidate_id"), field)

    if topic_routing:
        if topic_routing.get("schema_name") != "exam_bank.topic_routing_sidecar":
            errors.append(_issue("topic_routing_schema", str(topic_routing_path), "Unexpected topic-routing schema."))
        records = topic_routing.get("records") if isinstance(topic_routing.get("records"), dict) else {}
        declared = topic_routing.get("record_count")
        if isinstance(declared, int) and declared != len(records):
            errors.append(_issue("topic_routing_count", str(topic_routing_path), "record_count does not match records length."))
        topic_summary = audit_topic_routing_sidecar_payload(topic_routing)
    else:
        topic_summary = {}

    missing_manifest_paths = [
        path
        for path in sorted(manifest_paths)
        if not _resolve_existing_path(path, root=root, base=base)
    ]
    for path in missing_manifest_paths:
        errors.append(_issue("missing_manifest_asset_file", str(asset_manifest_path), "Manifest canonical_path does not resolve.", {"path": path}))

    duplicate_ids = [asset_id for asset_id, count in Counter(manifest_id_values if manifest else []).items() if count > 1]
    for asset_id in sorted(duplicate_ids):
        errors.append(_issue("duplicate_manifest_asset_id", str(asset_manifest_path), "Duplicate asset_id in manifest.", {"asset_id": asset_id}))

    export_image_duplicates = _noncanonical_export_image_duplicates(
        root=root,
        base=base,
        canonical_sha_by_path=manifest_sha_by_path,
    )
    for item in export_image_duplicates:
        errors.append(_issue("duplicate_image_in_export_folder", "output/asterion/exports", "Copied image exists in an export folder.", item))

    return {
        "schema_name": "exam_bank.asset_reference_validation",
        "schema_version": 1,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "manifest_asset_count": len(manifest_ids),
            "question_bank_records": len(question_bank.get("questions", [])) if question_bank else 0,
            "asterion_records": len(asterion.get("questions", [])) if asterion else 0,
            "content_lab_candidates": len(content_lab.get("candidates", [])) if content_lab else 0,
            "topic_routing": topic_summary,
        },
    }


def _asset_records_for_question(
    question: dict[str, Any],
    *,
    kind: str,
    paths: list[str],
    artifact_root: Path,
    base_dir: Path,
    source_path: str | None,
    source_fields: list[str],
    seen_ids: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        asset_id = asset_id_for_record(kind, question, path, occurrence=index)
        while asset_id in seen_ids:
            index += 1
            asset_id = asset_id_for_record(kind, question, path, occurrence=index)
        seen_ids.add(asset_id)
        resolved = _resolve_existing_path(path, root=artifact_root, base=base_dir)
        width, height = _image_dimensions(resolved)
        record = {
            "asset_id": asset_id,
            "kind": kind,
            "paper_id": question.get("paper"),
            "paper_family": question.get("paper_family"),
            "question_id": question.get("question_id"),
            "question_number": question.get("question_number"),
            "canonical_path": path,
            "sha256": _sha256(resolved) if resolved else None,
            "size_bytes": resolved.stat().st_size if resolved else None,
            "width": width,
            "height": height,
            "exists": resolved is not None,
            "source": {
                "question_bank_path": source_path,
                "source_fields": source_fields,
                "source_pdf": _note_or_top(question, "source_pdf"),
                "mark_scheme_source_pdf": _note_or_top(question, "mark_scheme_source_pdf"),
                "page_refs": question.get("page_refs"),
            },
        }
        records.append(record)
    return records


def _question_bank_image_field_values(question: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for field in ["canonical_question_artifact", "question_image_path", "canonical_mark_scheme_artifact", "mark_scheme_image_path"]:
        value = question.get(field)
        if isinstance(value, str) and value.strip():
            yield field, value.strip()
    for field in ["question_image_paths", "mark_scheme_image_paths"]:
        for value in _list_paths(question.get(field)):
            yield field, value


def _question_image_paths(question: dict[str, Any]) -> list[str]:
    return _unique_paths(
        _list_paths(question.get("canonical_question_artifact"))
        + _list_paths(question.get("question_image_path"))
        + _list_paths(question.get("question_image_paths"))
    )


def _mark_scheme_image_paths(question: dict[str, Any]) -> list[str]:
    return _unique_paths(
        _list_paths(question.get("canonical_mark_scheme_artifact"))
        + _list_paths(question.get("mark_scheme_image_path"))
        + _list_paths(question.get("mark_scheme_image_paths"))
    )


def _list_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _unique_paths(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _infer_artifact_root(path: Path) -> Path:
    if path.parent.name == "json":
        return path.parent.parent
    return path.parent


def _resolve_existing_path(value: Any, *, root: Path, base: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [root / path, base / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_dimensions(path: Path | None) -> tuple[int | None, int | None]:
    if path is None or path.suffix.lower() not in IMAGE_SUFFIXES:
        return None, None
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def _note_or_top(record: dict[str, Any], field: str) -> Any:
    if field in record and record[field] not in (None, ""):
        return record[field]
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(field)
    return None


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug or "unknown"


def _read_json_if_exists(path: Path, errors: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not path.is_file():
        errors.append(_issue("missing_validation_input", str(path), f"Missing {label} input."))
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(_issue("invalid_json", str(path), f"Invalid JSON: {exc}"))
        return {}
    if not isinstance(payload, dict):
        errors.append(_issue("invalid_json_shape", str(path), f"{label} must be a JSON object."))
        return {}
    return payload


def _check_record_count(payload: dict[str, Any], records: list[Any], errors: list[dict[str, Any]], source: str) -> None:
    declared = payload.get("record_count")
    if isinstance(declared, int) and declared != len(records):
        errors.append(_issue("record_count_mismatch", source, "record_count does not match records length."))


def _check_path_exists(
    value: Any,
    *,
    root: Path,
    base: Path,
    errors: list[dict[str, Any]],
    code: str,
    source: str,
    detail: dict[str, Any],
) -> None:
    if not value:
        return
    if _resolve_existing_path(value, root=root, base=base) is None:
        payload = dict(detail)
        payload["path"] = value
        errors.append(_issue(code, source, "Referenced image path does not resolve.", payload))


def _check_asset_id(value: Any, manifest_ids: set[str], errors: list[dict[str, Any]], source: str, owner_id: Any, field: str) -> None:
    asset_id = str(value or "").strip()
    if not asset_id:
        errors.append(_issue("missing_export_asset_id", source, "Export record is missing an asset_id reference.", {"owner_id": owner_id, "field": field}))
        return
    if asset_id not in manifest_ids:
        errors.append(_issue("unresolved_export_asset_id", source, "Export asset_id does not resolve through asset manifest.", {"owner_id": owner_id, "field": field, "asset_id": asset_id}))


def _noncanonical_export_image_duplicates(*, root: Path, base: Path, canonical_sha_by_path: dict[str, str]) -> list[dict[str, Any]]:
    canonical_hashes = {sha for sha in canonical_sha_by_path.values() if sha}
    export_root = base / root / "asterion" / "exports" if not root.is_absolute() else root / "asterion" / "exports"
    if not export_root.exists():
        return []
    duplicates: list[dict[str, Any]] = []
    for path in sorted(export_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digest = _sha256(path)
        if digest in canonical_hashes:
            duplicates.append({"path": str(path), "sha256": digest})
    return duplicates


def _issue(code: str, source: str, message: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "source": source, "message": message}
    if detail:
        payload["detail"] = detail
    return payload


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
