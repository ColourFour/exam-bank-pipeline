from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .atomic_json import write_atomic_json

SCHEMA_NAME = "exam_bank.review_asset_binding"
SCHEMA_VERSION = 1
_GENERATION_STATUSES = {"approved", "reviewed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate that release-affecting review decisions still bind to current canonical image bytes."
    )
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--source-skills",
        type=Path,
        default=Path("data/review/canonical/p3_exact_skill/reviewed_decisions.v1.json"),
    )
    parser.add_argument(
        "--mark-events",
        type=Path,
        default=Path("data/review/canonical/p3_exact_skill/reviewed_mark_events.v1.json"),
    )
    parser.add_argument(
        "--content-lab",
        type=Path,
        default=Path("data/review/canonical/asterion/content_lab_reviewed_decisions.v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/validations/review_asset_binding_validation.v1.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = bind_review_evidence_to_question_bank(
        _read_object(args.question_bank),
        artifact_root=args.artifact_root,
        source_skill_payload=_read_optional_object(args.source_skills),
        mark_event_payload=_read_optional_object(args.mark_events),
        content_lab_payload=_read_optional_object(args.content_lab),
    )
    report = dict(result["report"])
    report.update(
        {
            "ok": report["review_provenance_ok"],
            "question_bank": str(args.question_bank),
            "artifact_root": str(args.artifact_root),
            "source_skill_decisions_path": str(args.source_skills),
            "mark_event_decisions_path": str(args.mark_events),
            "content_lab_decisions_path": str(args.content_lab),
        }
    )
    write_atomic_json(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["ok"] else 1


def question_asset_index(
    question_bank: dict[str, Any],
    *,
    artifact_root: str | Path,
    base_dir: str | Path = ".",
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Return current canonical question/mark-scheme paths and byte hashes by ID."""

    return _question_assets(
        question_bank,
        artifact_root=Path(artifact_root),
        base_dir=Path(base_dir),
    )


def bind_review_evidence_to_question_bank(
    question_bank: dict[str, Any],
    *,
    artifact_root: str | Path,
    base_dir: str | Path = ".",
    source_skill_payload: dict[str, Any] | None = None,
    mark_event_payload: dict[str, Any] | None = None,
    content_lab_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebind review evidence to the current canonical image set.

    Review decisions are copied before they are changed. Evidence whose reviewed
    image bytes no longer match the current question-bank assets is demoted in
    the returned payloads, so stale files cannot silently unlock generation or
    student-runtime promotion.
    """

    root = Path(artifact_root)
    base = Path(base_dir)
    assets = _question_assets(question_bank, artifact_root=root, base_dir=base)

    source = copy.deepcopy(source_skill_payload or {})
    marks = copy.deepcopy(mark_event_payload or {})
    content_lab = copy.deepcopy(content_lab_payload or {})

    source_report = _bind_source_skill_records(source, assets)
    mark_report = _bind_mark_event_decisions(marks, assets)
    content_lab_report = _bind_content_lab_records(
        content_lab,
        assets,
        artifact_root=root,
        base_dir=base,
    )
    reports = {
        "source_skill_records": source_report,
        "mark_event_decisions": mark_report,
        "content_lab_records": content_lab_report,
    }
    active_invalid_count = sum(report["active_invalid_count"] for report in reports.values())
    return {
        "source_skill_payload": source,
        "mark_event_payload": marks,
        "content_lab_payload": content_lab,
        "report": {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "review_provenance_ok": active_invalid_count == 0,
            "fail_closed_applied": active_invalid_count > 0,
            "active_invalid_count": active_invalid_count,
            "question_bank_record_count": len(assets),
            **reports,
        },
    }


def _bind_source_skill_records(
    payload: dict[str, Any],
    assets: dict[str, dict[str, list[dict[str, str]]]],
) -> dict[str, Any]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        return _empty_report()
    invalid_ids: list[str] = []
    active_invalid_ids: list[str] = []
    rebound_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question_id = _text(row.get("question_id"))
        evidence_id = _text(row.get("evidence_id")) or f"index:{index}"
        question_ok, question_rebound = _bind_hashed_refs(
            row.get("source_question_asset_refs"),
            assets.get(question_id, {}).get("question", []),
        )
        marks_ok, marks_rebound = _bind_hashed_refs(
            row.get("source_mark_scheme_asset_refs"),
            assets.get(question_id, {}).get("mark_scheme", []),
        )
        rebound_count += question_rebound + marks_rebound
        valid = bool(question_id) and question_ok and marks_ok
        active = _source_record_is_active(row)
        row["review_asset_binding"] = {
            "status": "current" if valid else "stale",
            "question_assets_match": question_ok,
            "mark_scheme_assets_match": marks_ok,
        }
        if valid:
            continue
        invalid_ids.append(evidence_id)
        if active:
            active_invalid_ids.append(evidence_id)
            row["route_status"] = "blocked"
            blockers = _text_list(row.get("blockers"))
            if "stale_review_asset_binding" not in blockers:
                blockers.append("stale_review_asset_binding")
            row["blockers"] = blockers
            allowed = row.get("allowed_use_cases")
            if isinstance(allowed, dict):
                row["allowed_use_cases"] = {key: False for key in allowed}
    return _report(len(rows), invalid_ids, active_invalid_ids, rebound_count)


def _bind_mark_event_decisions(
    payload: dict[str, Any],
    assets: dict[str, dict[str, list[dict[str, str]]]],
) -> dict[str, Any]:
    rows = payload.get("decisions")
    if not isinstance(rows, list):
        return _empty_report()
    invalid_ids: list[str] = []
    active_invalid_ids: list[str] = []
    rebound_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question_id = _text(row.get("source_question_id"))
        decision_id = _text(row.get("decision_id")) or f"index:{index}"
        question_ok, question_rebound = _bind_single_hashed_ref(
            row,
            ref_field="question_image_ref",
            path_field="question_image_path",
            current_assets=assets.get(question_id, {}).get("question", []),
        )
        marks_ok, marks_rebound = _bind_single_hashed_ref(
            row,
            ref_field="mark_scheme_image_ref",
            path_field="mark_scheme_image_path",
            current_assets=assets.get(question_id, {}).get("mark_scheme", []),
        )
        rebound_count += question_rebound + marks_rebound
        valid = bool(question_id) and question_ok and marks_ok
        active = _text(row.get("status")).lower() in _GENERATION_STATUSES and row.get(
            "satisfies_generation_gate"
        ) is not False
        row["review_asset_binding"] = {
            "status": "current" if valid else "stale",
            "question_assets_match": question_ok,
            "mark_scheme_assets_match": marks_ok,
        }
        if valid:
            continue
        invalid_ids.append(decision_id)
        if active:
            active_invalid_ids.append(decision_id)
            row["status"] = "advisory"
            row["satisfies_generation_gate"] = False
            warnings = _text_list(row.get("warnings"))
            if "Demoted because reviewed assets do not match the current canonical bank." not in warnings:
                warnings.append(
                    "Demoted because reviewed assets do not match the current canonical bank."
                )
            row["warnings"] = warnings
    return _report(len(rows), invalid_ids, active_invalid_ids, rebound_count)


def _bind_content_lab_records(
    payload: dict[str, Any],
    assets: dict[str, dict[str, list[dict[str, str]]]],
    *,
    artifact_root: Path,
    base_dir: Path,
) -> dict[str, Any]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        return _empty_report()
    invalid_ids: list[str] = []
    active_invalid_ids: list[str] = []
    rebound_count = 0
    reviewed_digest_cache: dict[Path, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question_id = _text(row.get("question_id"))
        decision_id = _text(row.get("decision_id")) or f"index:{index}"
        question_ok, question_path = _match_reviewed_path(
            row.get("canonical_question_image_path"),
            assets.get(question_id, {}).get("question", []),
            artifact_root=artifact_root,
            base_dir=base_dir,
            digest_cache=reviewed_digest_cache,
        )
        marks_ok, marks_path = _match_reviewed_path(
            row.get("canonical_mark_scheme_image_path"),
            assets.get(question_id, {}).get("mark_scheme", []),
            artifact_root=artifact_root,
            base_dir=base_dir,
            digest_cache=reviewed_digest_cache,
        )
        if question_ok and question_path and row.get("canonical_question_image_path") != question_path:
            row["canonical_question_image_path"] = question_path
            rebound_count += 1
        if marks_ok and marks_path and row.get("canonical_mark_scheme_image_path") != marks_path:
            row["canonical_mark_scheme_image_path"] = marks_path
            rebound_count += 1
        valid = bool(question_id) and question_ok and marks_ok
        adjudication = row.get("adjudication") if isinstance(row.get("adjudication"), dict) else {}
        active = _text(adjudication.get("status")).lower() == "approved"
        row["review_asset_binding"] = {
            "status": "current" if valid else "stale",
            "question_assets_match": question_ok,
            "mark_scheme_assets_match": marks_ok,
        }
        if valid:
            continue
        invalid_ids.append(decision_id)
        if active:
            active_invalid_ids.append(decision_id)
            adjudication = dict(adjudication)
            adjudication["status"] = "blocked"
            adjudication["asset_binding_error"] = "stale_review_asset_binding"
            row["adjudication"] = adjudication
            risks = _text_list(row.get("risk_flags"))
            if "stale_review_asset_binding" not in risks:
                risks.append("stale_review_asset_binding")
            row["risk_flags"] = risks
    return _report(len(rows), invalid_ids, active_invalid_ids, rebound_count)


def _question_assets(
    question_bank: dict[str, Any],
    *,
    artifact_root: Path,
    base_dir: Path,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    rows = question_bank.get("questions")
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, list[dict[str, str]]]] = {}
    digest_cache: dict[Path, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = _text(row.get("question_id"))
        if not question_id:
            continue
        result[question_id] = {
            "question": _current_assets(
                _record_paths(
                    row,
                    "question_image_paths",
                    "question_image_path",
                    "canonical_question_artifact",
                ),
                artifact_root=artifact_root,
                base_dir=base_dir,
                digest_cache=digest_cache,
            ),
            "mark_scheme": _current_assets(
                _record_paths(
                    row,
                    "mark_scheme_image_paths",
                    "mark_scheme_image_path",
                    "canonical_mark_scheme_artifact",
                ),
                artifact_root=artifact_root,
                base_dir=base_dir,
                digest_cache=digest_cache,
            ),
        }
    return result


def _record_paths(
    row: dict[str, Any],
    plural: str,
    singular: str,
    canonical: str,
) -> list[str]:
    value = row.get(plural)
    if isinstance(value, list):
        paths = [_text(item) for item in value if _text(item)]
        if paths:
            return paths
    path = _text(row.get(singular) or row.get(canonical))
    return [path] if path else []


def _current_assets(
    paths: Iterable[str],
    *,
    artifact_root: Path,
    base_dir: Path,
    digest_cache: dict[Path, str],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[Path] = set()
    for raw in paths:
        resolved = _resolve_current_path(raw, artifact_root=artifact_root, base_dir=base_dir)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        digest = digest_cache.get(resolved)
        if digest is None:
            digest = _sha256(resolved)
            digest_cache[resolved] = digest
        result.append(
            {
                "path": _portable_path(resolved, base_dir=base_dir),
                "sha256": digest,
            }
        )
    return result


def _resolve_current_path(raw: str, *, artifact_root: Path, base_dir: Path) -> Path | None:
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [artifact_root / path, base_dir / path, path]
    if not path.is_absolute() and path.parts[:1] == (artifact_root.name,):
        candidates.insert(0, base_dir / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_reviewed_path(raw: Any, *, artifact_root: Path, base_dir: Path) -> Path | None:
    value = _text(raw)
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [base_dir / path, path, artifact_root / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _portable_path(path: Path, *, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _bind_hashed_refs(refs: Any, current_assets: list[dict[str, str]]) -> tuple[bool, int]:
    if not isinstance(refs, list) or not refs or not current_assets:
        return False, 0
    by_hash = {item["sha256"]: item for item in current_assets}
    rebound = 0
    for ref in refs:
        if not isinstance(ref, dict):
            return False, rebound
        if ref.get("verified") is not True:
            return False, rebound
        match = by_hash.get(_text(ref.get("sha256")))
        if match is None:
            return False, rebound
        current_path = match["path"]
        if ref.get("path") != current_path:
            ref["path"] = current_path
            rebound += 1
    return True, rebound


def _bind_single_hashed_ref(
    row: dict[str, Any],
    *,
    ref_field: str,
    path_field: str,
    current_assets: list[dict[str, str]],
) -> tuple[bool, int]:
    ref = row.get(ref_field)
    if not isinstance(ref, dict):
        return False, 0
    valid, rebound = _bind_hashed_refs([ref], current_assets)
    if valid:
        current_path = ref["path"]
        if row.get(path_field) != current_path:
            row[path_field] = current_path
            rebound += 1
    return valid, rebound


def _match_reviewed_path(
    raw: Any,
    current_assets: list[dict[str, str]],
    *,
    artifact_root: Path,
    base_dir: Path,
    digest_cache: dict[Path, str],
) -> tuple[bool, str | None]:
    reviewed = _resolve_reviewed_path(raw, artifact_root=artifact_root, base_dir=base_dir)
    if reviewed is None or not current_assets:
        return False, None
    reviewed_hash = digest_cache.get(reviewed)
    if reviewed_hash is None:
        reviewed_hash = _sha256(reviewed)
        digest_cache[reviewed] = reviewed_hash
    for asset in current_assets:
        if asset["sha256"] == reviewed_hash:
            return True, asset["path"]
    return False, None


def _source_record_is_active(row: dict[str, Any]) -> bool:
    if _text(row.get("route_status")).lower() == "clean":
        return True
    allowed = row.get("allowed_use_cases")
    return isinstance(allowed, dict) and any(value is True for value in allowed.values())


def _report(
    count: int,
    invalid_ids: list[str],
    active_invalid_ids: list[str],
    rebound_count: int,
) -> dict[str, Any]:
    return {
        "record_count": count,
        "invalid_count": len(invalid_ids),
        "active_invalid_count": len(active_invalid_ids),
        "rebound_path_count": rebound_count,
        "invalid_ids": invalid_ids,
        "active_invalid_ids": active_invalid_ids,
    }


def _empty_report() -> dict[str, Any]:
    return _report(0, [], [], 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_optional_object(path: Path) -> dict[str, Any]:
    return _read_object(path) if path.is_file() else {}
