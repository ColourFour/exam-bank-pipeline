from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .asset_manifest import IMAGE_SUFFIXES, write_asset_manifest
from .atomic_json import write_atomic_json


STORAGE_AUDIT_SCHEMA_NAME = "exam_bank.output_storage_duplicate_audit"
STORAGE_AUDIT_SCHEMA_VERSION = 1
DELETE_MANIFEST_SCHEMA_NAME = "exam_bank.output_storage_delete_manifest"
DELETE_MANIFEST_SCHEMA_VERSION = 1

DEFAULT_SCAN_ROOTS = (Path("output"), Path("reports"))
DEFAULT_REFERENCE_JSON_FILES = (
    Path("output/json/question_bank.json"),
    Path("output/json/question_bank.topic_routing.v1.json"),
    Path("output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json"),
    Path("output/asterion/exports/latest/asterion_question_bank_v1.json"),
    Path("output/asterion/exports/latest/asterion_content_lab_candidates_v1.json"),
    Path("output/json/asset_manifest.v1.json"),
)
SELF_REPORT_PATHS = {
    "reports/output_storage_duplicate_audit.v1.json",
    "reports/output_storage_duplicate_audit.md",
    "reports/output_storage_optimization_plan.md",
    "reports/output_storage_delete_manifest.v1.json",
}

JSON_SUFFIXES = {".json", ".jsonl"}
REPORT_SUFFIXES = {".md", ".csv", ".txt"}
PDF_SUFFIXES = {".pdf"}

CANONICAL_ACTION = "keep canonical"
REMAP_ACTION = "remap reference"
SAFE_DELETE_ACTION = "safe duplicate delete candidate"
CACHE_ACTION = "generated cache candidate"
QUARANTINE_ACTION = "quarantine only"
DO_NOT_TOUCH_ACTION = "do not touch"


@dataclass(frozen=True)
class FileInfo:
    path: Path
    rel_path: str
    size_bytes: int
    suffix: str
    media_type: str
    sha256: str


def build_storage_audit(
    *,
    scan_roots: Iterable[str | Path] = DEFAULT_SCAN_ROOTS,
    reference_json_files: Iterable[str | Path] = DEFAULT_REFERENCE_JSON_FILES,
    project_root: str | Path | None = None,
    largest_limit: int = 30,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    roots = [Path(path) for path in scan_roots]
    existing_roots = [path for path in roots if (root / path).exists() or path.exists()]
    files = _scan_files(existing_roots, project_root=root)
    references = collect_json_references(reference_json_files, project_root=root)
    referenced_paths = set(references["referenced_files"])

    by_sha: dict[str, list[FileInfo]] = defaultdict(list)
    by_size_type: dict[tuple[int, str], list[FileInfo]] = defaultdict(list)
    for info in files:
        by_sha[info.sha256].append(info)
        by_size_type[(info.size_bytes, info.media_type)].append(info)

    duplicate_groups = [
        _duplicate_group(sha, group, referenced_paths)
        for sha, group in by_sha.items()
        if len(group) > 1
    ]
    duplicate_groups.sort(key=lambda item: (-int(item["wasted_bytes"]), str(item["canonical_file"] or ""), item["sha256"]))

    size_type_groups = [
        _size_type_group(size, media_type, group)
        for (size, media_type), group in by_size_type.items()
        if len(group) > 1
    ]
    size_type_groups.sort(key=lambda item: (-int(item["total_bytes"]), item["media_type"], int(item["size_bytes"])))

    cleanup_candidates = _cleanup_candidates(duplicate_groups)
    largest_dirs = _largest_directories(files, limit=largest_limit)
    largest_files = [
        _file_summary(info, referenced_paths=referenced_paths)
        for info in sorted(files, key=lambda item: (-item.size_bytes, item.rel_path))[:largest_limit]
    ]

    total_bytes = sum(info.size_bytes for info in files)
    file_paths = {info.rel_path for info in files}
    referenced_files = sorted(file_paths & referenced_paths)
    unreferenced_files = sorted(file_paths - referenced_paths)

    return {
        "schema_name": STORAGE_AUDIT_SCHEMA_NAME,
        "schema_version": STORAGE_AUDIT_SCHEMA_VERSION,
        "scan_roots": [str(path) for path in roots],
        "scan_roots_found": [str(path) for path in existing_roots],
        "reference_json_files": [str(path) for path in reference_json_files],
        "summary": {
            "total_file_count": len(files),
            "total_size_bytes": total_bytes,
            "total_size_human": _human_bytes(total_bytes),
            "referenced_file_count": len(referenced_files),
            "unreferenced_file_count": len(unreferenced_files),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_file_count": sum(int(group["file_count"]) for group in duplicate_groups),
            "duplicate_wasted_bytes": sum(int(group["wasted_bytes"]) for group in duplicate_groups),
            "duplicate_wasted_human": _human_bytes(sum(int(group["wasted_bytes"]) for group in duplicate_groups)),
            "estimated_reclaimable_bytes": sum(int(item["size_bytes"]) for item in cleanup_candidates),
            "estimated_reclaimable_human": _human_bytes(sum(int(item["size_bytes"]) for item in cleanup_candidates)),
        },
        "largest_directories": largest_dirs,
        "largest_files": largest_files,
        "duplicate_groups_by_sha256": duplicate_groups,
        "duplicate_groups_by_size_type": size_type_groups,
        "image_duplicate_groups": [_group_brief(group) for group in duplicate_groups if group["media_type"] == "image"],
        "json_duplicate_groups": [_group_brief(group) for group in duplicate_groups if group["media_type"] == "json"],
        "pdf_or_report_duplicate_groups": [
            _group_brief(group) for group in duplicate_groups if group["media_type"] in {"pdf", "report"}
        ],
        "json_references": references,
        "files_referenced_by_current_json_exports": referenced_files,
        "files_not_referenced_by_current_json_exports": unreferenced_files,
        "cleanup_candidates": cleanup_candidates,
        "policy_notes": [
            "Canonical image evidence is under output/p*/<paper>/questions/*.png and output/p*/<paper>/mark_scheme/*.png.",
            "Cleanup candidates are exact SHA-256 duplicates only; default mode is dry-run.",
            "Apply mode moves candidates to a quarantine directory and never permanently deletes files.",
            "Delete mode writes a deletion manifest before hard-deleting allowlisted non-canonical exact duplicates.",
        ],
    }


def collect_json_references(
    reference_json_files: Iterable[str | Path],
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    references_by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    unresolved: list[dict[str, str]] = []
    inspected: list[str] = []

    for raw_path in reference_json_files:
        json_path = Path(raw_path)
        resolved_json_path = json_path if json_path.is_absolute() else root / json_path
        if not resolved_json_path.is_file():
            continue
        inspected.append(str(json_path))
        try:
            payload = json.loads(resolved_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            unresolved.append({"source_json": str(json_path), "path": str(json_path), "reason": f"invalid_json:{exc}"})
            continue
        artifact_root = _artifact_root_for_json(json_path)
        for json_pointer, key, value in _walk_json_paths(payload):
            for candidate in _candidate_path_values(key, value):
                resolved = _resolve_reference(candidate, source_json=json_path, artifact_root=artifact_root, project_root=root)
                if resolved is None:
                    unresolved.append({"source_json": str(json_path), "json_pointer": json_pointer, "path": candidate})
                    continue
                rel = _relpath(resolved, root)
                references_by_file[rel].append(
                    {
                        "source_json": str(json_path),
                        "json_pointer": json_pointer,
                        "field": key,
                        "value": candidate,
                    }
                )

    return {
        "json_files_inspected": inspected,
        "referenced_files": sorted(references_by_file),
        "references_by_file": {
            path: {"reference_count": len(refs), "references_sample": refs[:5]}
            for path, refs in sorted(references_by_file.items())
        },
        "unresolved_references": sorted(unresolved, key=lambda item: (item.get("source_json", ""), item.get("path", ""))),
    }


def write_storage_audit_outputs(
    audit: dict[str, Any],
    *,
    json_path: str | Path = "reports/output_storage_duplicate_audit.v1.json",
    markdown_path: str | Path = "reports/output_storage_duplicate_audit.md",
    plan_path: str | Path = "reports/output_storage_optimization_plan.md",
) -> None:
    write_atomic_json(audit, json_path, sort_keys=True)
    _write_text(Path(markdown_path), render_storage_audit_markdown(audit))
    _write_text(Path(plan_path), render_storage_optimization_plan(audit))


def run_storage_audit(
    *,
    scan_roots: Iterable[str | Path] = DEFAULT_SCAN_ROOTS,
    reference_json_files: Iterable[str | Path] = DEFAULT_REFERENCE_JSON_FILES,
    question_bank_path: str | Path = "output/json/question_bank.json",
    asset_manifest_path: str | Path = "output/json/asset_manifest.v1.json",
    write_reports: bool = True,
    json_path: str | Path = "reports/output_storage_duplicate_audit.v1.json",
    markdown_path: str | Path = "reports/output_storage_duplicate_audit.md",
    plan_path: str | Path = "reports/output_storage_optimization_plan.md",
    dry_run: bool = True,
    apply: bool = False,
    apply_delete: bool = False,
    delete_manifest_path: str | Path = "reports/output_storage_delete_manifest.v1.json",
    quarantine_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    manifest_input = Path(question_bank_path)
    manifest_output = Path(asset_manifest_path)
    if (root / manifest_input).is_file() or manifest_input.is_file():
        write_asset_manifest(root / manifest_input if not manifest_input.is_absolute() else manifest_input, root / manifest_output if not manifest_output.is_absolute() else manifest_output)

    audit = build_storage_audit(scan_roots=scan_roots, reference_json_files=reference_json_files, project_root=root)
    cleanup_result = {
        "dry_run": dry_run or not apply,
        "applied": False,
        "quarantine_dir": str(quarantine_dir) if quarantine_dir is not None else None,
        "moved_files": [],
    }
    if apply:
        cleanup_result = apply_quarantine_plan(audit, quarantine_dir=quarantine_dir, project_root=root)
    audit["cleanup_result"] = cleanup_result
    delete_result = {
        "dry_run": dry_run or not apply_delete,
        "applied": False,
        "delete_manifest_path": str(delete_manifest_path),
        "deleted_files": [],
        "deleted_file_count": 0,
        "deleted_bytes": 0,
        "deleted_human": "0 B",
    }
    if apply_delete:
        delete_result = apply_delete_plan(audit, manifest_path=delete_manifest_path, project_root=root)
    audit["delete_result"] = delete_result
    if write_reports:
        write_storage_audit_outputs(audit, json_path=json_path, markdown_path=markdown_path, plan_path=plan_path)
    return audit


def apply_quarantine_plan(
    audit: dict[str, Any],
    *,
    quarantine_dir: str | Path | None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    target_root = Path(quarantine_dir) if quarantine_dir is not None else Path("output/_quarantine_storage_cleanup")
    target_root = target_root if target_root.is_absolute() else root / target_root
    moved: list[dict[str, Any]] = []
    for candidate in audit.get("cleanup_candidates", []):
        if not isinstance(candidate, dict):
            continue
        action = str(candidate.get("suggested_action") or "")
        if action not in {SAFE_DELETE_ACTION, CACHE_ACTION}:
            continue
        rel = str(candidate.get("path") or "")
        source = root / rel
        if not source.is_file() or _is_canonical_asset_path(rel):
            continue
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append({"from": rel, "to": _relpath(target, root), "suggested_action": action})
    return {
        "dry_run": False,
        "applied": True,
        "quarantine_dir": _relpath(target_root, root),
        "moved_files": moved,
    }


def build_delete_manifest(
    audit: dict[str, Any],
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for group in audit.get("duplicate_groups_by_sha256", []):
        if not isinstance(group, dict):
            continue
        retained_path = str(group.get("canonical_file") or "")
        group_sha = str(group.get("sha256") or "")
        if not retained_path or not group_sha:
            skipped.append(_delete_skip("missing_retained_file", group))
            continue
        retained = root / retained_path
        if not retained.is_file():
            skipped.append(_delete_skip("retained_file_missing", group, retained_path=retained_path))
            continue
        if _sha256(retained) != group_sha:
            skipped.append(_delete_skip("retained_hash_mismatch", group, retained_path=retained_path))
            continue
        if _retained_entry_for_group(group, retained_path) is None:
            skipped.append(_delete_skip("no_retained_keeper_entry", group, retained_path=retained_path))
            continue

        for file_entry in group.get("files", []):
            if not isinstance(file_entry, dict):
                continue
            path = str(file_entry.get("path") or "")
            if not path or path == retained_path:
                continue
            if not _delete_allowed_duplicate_path(path):
                continue
            if _is_canonical_asset_path(path):
                skipped.append(_delete_skip("canonical_asset_protected", group, path=path, retained_path=retained_path))
                continue
            action = str(file_entry.get("suggested_action") or "")
            if action not in {SAFE_DELETE_ACTION, CACHE_ACTION}:
                continue
            candidate = root / path
            if not candidate.is_file():
                continue
            expected_size = int(group.get("size_bytes") or file_entry.get("size_bytes") or 0)
            if expected_size and candidate.stat().st_size != expected_size:
                skipped.append(_delete_skip("candidate_size_mismatch", group, path=path, retained_path=retained_path))
                continue
            if _sha256(candidate) != group_sha:
                skipped.append(_delete_skip("candidate_hash_mismatch", group, path=path, retained_path=retained_path))
                continue
            entries.append(
                {
                    "path": path,
                    "retained_path": retained_path,
                    "sha256": group_sha,
                    "size_bytes": candidate.stat().st_size,
                    "size_human": _human_bytes(candidate.stat().st_size),
                    "reason": "exact SHA-256 duplicate of retained file in an allowlisted generated/candidate/archive/cache path",
                    "safety_classification": _delete_safety_classification(path),
                    "suggested_action": action,
                }
            )

    entries.sort(key=lambda item: (-int(item["size_bytes"]), item["path"]))
    total_bytes = sum(int(item["size_bytes"]) for item in entries)
    return {
        "schema_name": DELETE_MANIFEST_SCHEMA_NAME,
        "schema_version": DELETE_MANIFEST_SCHEMA_VERSION,
        "dry_run": True,
        "delete_file_count": len(entries),
        "delete_bytes": total_bytes,
        "delete_human": _human_bytes(total_bytes),
        "entries": entries,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:100],
        "policy": {
            "hard_delete_only_allowlisted_noncanonical_exact_duplicates": True,
            "canonical_image_paths_protected": True,
            "topic_packet_pdfs_excluded": True,
        },
    }


def write_delete_manifest(
    audit: dict[str, Any],
    *,
    manifest_path: str | Path = "reports/output_storage_delete_manifest.v1.json",
    project_root: str | Path | None = None,
    merge_existing: bool = False,
) -> dict[str, Any]:
    manifest = build_delete_manifest(audit, project_root=project_root)
    if merge_existing:
        existing = _read_delete_manifest(manifest_path, project_root=project_root)
        if existing is not None:
            manifest = _merge_delete_manifests(existing, manifest)
    write_atomic_json(manifest, manifest_path, sort_keys=True)
    return manifest


def apply_delete_plan(
    audit: dict[str, Any],
    *,
    manifest_path: str | Path = "reports/output_storage_delete_manifest.v1.json",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    manifest = write_delete_manifest(audit, manifest_path=manifest_path, project_root=root, merge_existing=True)
    deleted: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        rel = str(entry.get("path") or "")
        retained_rel = str(entry.get("retained_path") or "")
        expected_sha = str(entry.get("sha256") or "")
        candidate = root / rel
        retained = root / retained_rel
        if not rel or not candidate.is_file() or _is_canonical_asset_path(rel) or not _delete_allowed_duplicate_path(rel):
            continue
        if not retained.is_file():
            continue
        if _sha256(retained) != expected_sha or _sha256(candidate) != expected_sha:
            continue
        size = candidate.stat().st_size
        candidate.unlink()
        deleted.append(
            {
                "path": rel,
                "retained_path": retained_rel,
                "sha256": expected_sha,
                "size_bytes": size,
                "size_human": _human_bytes(size),
                "safety_classification": entry.get("safety_classification"),
            }
        )
    total_bytes = sum(int(item["size_bytes"]) for item in deleted)
    return {
        "dry_run": False,
        "applied": True,
        "delete_manifest_path": str(manifest_path),
        "deleted_files": deleted,
        "deleted_file_count": len(deleted),
        "deleted_bytes": total_bytes,
        "deleted_human": _human_bytes(total_bytes),
    }


def render_storage_audit_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    lines = [
        "# Output Storage Duplicate Audit",
        "",
        f"- Files scanned: `{summary.get('total_file_count', 0)}`",
        f"- Total size: `{summary.get('total_size_human', '0 B')}` (`{summary.get('total_size_bytes', 0)}` bytes)",
        f"- Exact duplicate groups: `{summary.get('duplicate_group_count', 0)}`",
        f"- Duplicate wasted size: `{summary.get('duplicate_wasted_human', '0 B')}`",
        f"- Estimated reclaimable size: `{summary.get('estimated_reclaimable_human', '0 B')}`",
        f"- Files referenced by current JSON exports: `{summary.get('referenced_file_count', 0)}`",
        f"- Files not referenced by current JSON exports: `{summary.get('unreferenced_file_count', 0)}`",
        "",
        "## Largest Directories",
    ]
    for item in audit.get("largest_directories", [])[:20]:
        lines.append(f"- `{item['path']}`: `{item['size_human']}`")
    lines.extend(["", "## Largest Files"])
    for item in audit.get("largest_files", [])[:20]:
        lines.append(f"- `{item['path']}`: `{item['size_human']}`")
    lines.extend(["", "## Exact Duplicate Groups"])
    groups = audit.get("duplicate_groups_by_sha256", [])
    if not groups:
        lines.append("- None")
    for group in groups[:50]:
        lines.append(
            f"- `{group['sha256'][:12]}` `{group['media_type']}`: "
            f"`{group['file_count']}` files, `{group['wasted_human']}` wasted, canonical `{group.get('canonical_file')}`"
        )
        for file_entry in group.get("files", [])[:8]:
            lines.append(f"  - `{file_entry['suggested_action']}` `{file_entry['path']}`")
    if len(groups) > 50:
        lines.append(f"- Additional duplicate groups omitted from Markdown: `{len(groups) - 50}`. See JSON report.")
    lines.extend(["", "## Cleanup Candidates"])
    candidates = audit.get("cleanup_candidates", [])
    if not candidates:
        lines.append("- None")
    for candidate in candidates[:100]:
        lines.append(
            f"- `{candidate['suggested_action']}` `{candidate['path']}` "
            f"(`{candidate['size_human']}`, canonical `{candidate.get('canonical_file')}`)"
        )
    if len(candidates) > 100:
        lines.append(f"- Additional cleanup candidates omitted from Markdown: `{len(candidates) - 100}`. See JSON report.")
    lines.extend(["", "## Unresolved JSON References"])
    unresolved = audit.get("json_references", {}).get("unresolved_references", [])
    if not unresolved:
        lines.append("- None")
    for item in unresolved[:50]:
        lines.append(f"- `{item.get('source_json')}` -> `{item.get('path')}`")
    return "\n".join(lines).rstrip() + "\n"


def render_storage_optimization_plan(audit: dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    lines = [
        "# Output Storage Optimization Plan",
        "",
        "## Current Size",
        f"- Total scanned size: `{summary.get('total_size_human', '0 B')}` (`{summary.get('total_size_bytes', 0)}` bytes).",
        f"- Exact duplicate wasted size: `{summary.get('duplicate_wasted_human', '0 B')}`.",
        f"- Conservative reclaimable size: `{summary.get('estimated_reclaimable_human', '0 B')}`.",
        "",
        "## Largest Duplicate Sources",
    ]
    for item in audit.get("largest_directories", [])[:15]:
        lines.append(f"- `{item['path']}`: `{item['size_human']}`")
    lines.extend(
        [
            "",
            "## Canonical Directories To Keep",
            "- `output/json/question_bank.json`",
            "- `output/json/asset_manifest.v1.json`",
            "- `output/p*/<paper>/questions/*.png`",
            "- `output/p*/<paper>/mark_scheme/*.png`",
            "- `output/asterion/exports/latest/*.json` as lightweight references, not copied images",
            "",
            "## Generated Or Rebuildable Candidates",
        ]
    )
    for path in [
        "output/candidates/ocr/*",
        "output/codex_text_extraction_candidate*",
        "output/codex_text_extraction_targeted/*",
        "output/archive/generated_cleanup_*",
        "output/topic_packets/*/topic_packet.pdf",
        "output/audits/*",
        "output/run_status/*",
    ]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Exact Duplicate Groups"])
    groups = audit.get("duplicate_groups_by_sha256", [])
    if not groups:
        lines.append("- None")
    for group in groups[:75]:
        lines.append(
            f"- `{group['sha256'][:12]}` `{group['media_type']}`: `{group['file_count']}` files, "
            f"`{group['wasted_human']}` wasted, canonical `{group.get('canonical_file')}`"
        )
    if len(groups) > 75:
        lines.append(f"- Additional exact duplicate groups: `{len(groups) - 75}`. See `reports/output_storage_duplicate_audit.v1.json`.")
    lines.extend(
        [
            "",
            "## Recommended Implementation Steps",
            "1. Keep canonical images under `output/p*/...` as the source of truth.",
            "2. Keep downstream JSON exports path-compatible but prefer `*_asset_id` fields and canonical relative paths.",
            "3. Use `output/json/asset_manifest.v1.json` as an index for asset lookup and integrity checks.",
            "4. Use hard-delete mode only for exact duplicate non-canonical files after reviewing `reports/output_storage_delete_manifest.v1.json`.",
            "5. Regenerate topic packets and candidate outputs instead of storing copied image trees long term.",
            "",
            "## Risks",
            "- Some archive and candidate folders may still carry historical comparison evidence.",
            "- Standalone PDFs intentionally embed images and should not be rewritten as path references.",
            "- Canonical image duplicates can be real duplicate evidence across papers; do not remove them without explicit remap review.",
            "",
            "## Regeneration Commands",
            "- `.venv/bin/python -m exam_bank.cli asterion-export --input output/json/question_bank.json --artifact-root output`",
            "- `.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --input output/json/question_bank.json --artifact-root output`",
            "- `.venv/bin/python -m exam_bank.cli topic-packets --input output/json/question_bank.json --artifact-root output`",
            "- `.venv/bin/python scripts/audit_output_storage.py --dry-run`",
            "- `.venv/bin/python scripts/audit_output_storage.py --apply-delete`",
            "- `.venv/bin/python scripts/validate_asset_references.py`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _scan_files(roots: Iterable[Path], *, project_root: Path) -> list[FileInfo]:
    infos: list[FileInfo] = []
    for raw_root in roots:
        root = raw_root if raw_root.is_absolute() else project_root / raw_root
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = _relpath(path, project_root)
            if rel in SELF_REPORT_PATHS:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            suffix = path.suffix.lower()
            infos.append(
                FileInfo(
                    path=path,
                    rel_path=rel,
                    size_bytes=size,
                    suffix=suffix,
                    media_type=_media_type(suffix),
                    sha256=_sha256(path),
                )
            )
    return infos


def _duplicate_group(sha: str, group: list[FileInfo], referenced_paths: set[str]) -> dict[str, Any]:
    files = sorted(group, key=lambda item: _canonical_rank(item.rel_path, referenced_paths))
    canonical = files[0]
    group_has_canonical = any(_is_canonical_asset_path(info.rel_path) for info in files)
    entries = []
    for info in files:
        entries.append(
            {
                **_file_summary(info, referenced_paths=referenced_paths),
                "suggested_action": _suggested_action(info.rel_path, canonical.rel_path, referenced_paths, group_has_canonical),
                "canonical_file": canonical.rel_path,
            }
        )
    return {
        "sha256": sha,
        "size_bytes": canonical.size_bytes,
        "size_human": _human_bytes(canonical.size_bytes),
        "file_count": len(files),
        "total_bytes": sum(info.size_bytes for info in files),
        "wasted_bytes": canonical.size_bytes * (len(files) - 1),
        "wasted_human": _human_bytes(canonical.size_bytes * (len(files) - 1)),
        "media_type": canonical.media_type,
        "candidate_canonical_file": canonical.rel_path,
        "canonical_file": canonical.rel_path,
        "files": entries,
    }


def _size_type_group(size: int, media_type: str, group: list[FileInfo]) -> dict[str, Any]:
    sha_count = len({info.sha256 for info in group})
    return {
        "size_bytes": size,
        "size_human": _human_bytes(size),
        "media_type": media_type,
        "file_count": len(group),
        "distinct_sha256_count": sha_count,
        "total_bytes": size * len(group),
        "paths_sample": [info.rel_path for info in sorted(group, key=lambda item: item.rel_path)[:25]],
    }


def _group_brief(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "sha256": group["sha256"],
        "size_bytes": group["size_bytes"],
        "size_human": group["size_human"],
        "file_count": group["file_count"],
        "wasted_bytes": group["wasted_bytes"],
        "wasted_human": group["wasted_human"],
        "media_type": group["media_type"],
        "canonical_file": group.get("canonical_file"),
        "paths": [str(item.get("path")) for item in group.get("files", []) if isinstance(item, dict)],
    }


def _cleanup_candidates(duplicate_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group in duplicate_groups:
        for file_entry in group.get("files", []):
            action = file_entry.get("suggested_action")
            if action in {SAFE_DELETE_ACTION, CACHE_ACTION}:
                candidates.append(file_entry)
    candidates.sort(key=lambda item: (-int(item.get("size_bytes") or 0), str(item.get("path") or "")))
    return candidates


def _largest_directories(files: list[FileInfo], *, limit: int) -> list[dict[str, Any]]:
    sizes: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for info in files:
        path = Path(info.rel_path)
        parents = list(path.parents)
        for parent in parents:
            if str(parent) in {"", "."}:
                continue
            key = str(parent)
            sizes[key] += info.size_bytes
            counts[key] += 1
    rows = [
        {"path": path, "size_bytes": size, "size_human": _human_bytes(size), "file_count": counts[path]}
        for path, size in sizes.items()
    ]
    return sorted(rows, key=lambda item: (-int(item["size_bytes"]), str(item["path"])))[:limit]


def _file_summary(info: FileInfo, *, referenced_paths: set[str]) -> dict[str, Any]:
    return {
        "path": info.rel_path,
        "size_bytes": info.size_bytes,
        "size_human": _human_bytes(info.size_bytes),
        "media_type": info.media_type,
        "suffix": info.suffix,
        "sha256": info.sha256,
        "referenced_by_current_json_exports": info.rel_path in referenced_paths,
        "canonical_asset": _is_canonical_asset_path(info.rel_path),
    }


def _suggested_action(path: str, canonical_path: str, referenced_paths: set[str], group_has_canonical: bool) -> str:
    if path == canonical_path:
        return CANONICAL_ACTION if _is_canonical_asset_path(path) else DO_NOT_TOUCH_ACTION
    if _is_canonical_asset_path(path):
        return DO_NOT_TOUCH_ACTION
    if path in referenced_paths:
        return REMAP_ACTION if group_has_canonical else DO_NOT_TOUCH_ACTION
    if _is_generated_cache_path(path):
        return CACHE_ACTION
    if group_has_canonical:
        return SAFE_DELETE_ACTION
    if _is_archive_path(path):
        return QUARANTINE_ACTION
    return DO_NOT_TOUCH_ACTION


def _canonical_rank(path: str, referenced_paths: set[str]) -> tuple[int, int, int, str]:
    return (
        0 if _is_canonical_asset_path(path) else 1,
        0 if path in referenced_paths else 1,
        0 if path.startswith("output/") else 1,
        path,
    )


def _is_canonical_asset_path(path: str) -> bool:
    parts = Path(path).parts
    return (
        len(parts) >= 5
        and parts[0] == "output"
        and parts[1].startswith("p")
        and parts[2]
        and parts[3] in {"questions", "mark_scheme"}
        and Path(path).suffix.lower() in IMAGE_SUFFIXES
    )


def _is_generated_cache_path(path: str) -> bool:
    parts = set(Path(path).parts)
    return bool(parts & {"candidates", "audits", "run_status"}) or "codex_text_extraction_candidate" in path


def _delete_allowed_duplicate_path(path: str) -> bool:
    parts = Path(path).parts
    if _is_canonical_asset_path(path):
        return False
    if len(parts) >= 4 and parts[:3] == ("output", "candidates", "ocr"):
        return True
    if len(parts) >= 2 and parts[0] == "output" and parts[1].startswith("codex_text_extraction_candidate"):
        return True
    if len(parts) >= 2 and parts[0] == "output" and parts[1] == "codex_text_extraction_targeted":
        return True
    if len(parts) >= 3 and parts[0] == "output" and parts[1] == "archive" and parts[2].startswith("generated_cleanup_"):
        return True
    if len(parts) >= 2 and parts[0] == "output" and parts[1] in {"audits", "run_status"}:
        return True
    return False


def _delete_safety_classification(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 4 and parts[:3] == ("output", "candidates", "ocr"):
        return "ocr_candidate_duplicate"
    if len(parts) >= 2 and parts[0] == "output" and parts[1].startswith("codex_text_extraction_candidate"):
        return "codex_text_extraction_candidate_duplicate"
    if len(parts) >= 2 and parts[0] == "output" and parts[1] == "codex_text_extraction_targeted":
        return "codex_text_extraction_targeted_duplicate"
    if len(parts) >= 3 and parts[0] == "output" and parts[1] == "archive" and parts[2].startswith("generated_cleanup_"):
        return "archive_generated_cleanup_duplicate"
    if len(parts) >= 2 and parts[0] == "output" and parts[1] == "audits":
        return "generated_audit_duplicate"
    if len(parts) >= 2 and parts[0] == "output" and parts[1] == "run_status":
        return "generated_run_status_duplicate"
    return "not_allowlisted"


def _retained_entry_for_group(group: dict[str, Any], retained_path: str) -> dict[str, Any] | None:
    for file_entry in group.get("files", []):
        if isinstance(file_entry, dict) and str(file_entry.get("path") or "") == retained_path:
            action = str(file_entry.get("suggested_action") or "")
            if action in {CANONICAL_ACTION, DO_NOT_TOUCH_ACTION, REMAP_ACTION}:
                return file_entry
    return None


def _delete_skip(reason: str, group: dict[str, Any], **detail: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reason": reason,
        "sha256": str(group.get("sha256") or ""),
        "canonical_file": str(group.get("canonical_file") or ""),
    }
    payload.update({key: value for key, value in detail.items() if value})
    return payload


def _read_delete_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any] | None:
    root = Path(project_root) if project_root is not None else Path.cwd()
    path = Path(manifest_path)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_name") != DELETE_MANIFEST_SCHEMA_NAME:
        return None
    return payload


def _merge_delete_manifests(existing: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    entries_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest in (existing, current):
        for entry in manifest.get("entries", []):
            if not isinstance(entry, dict):
                continue
            key = (str(entry.get("path") or ""), str(entry.get("sha256") or ""))
            if not key[0] or not key[1]:
                continue
            entries_by_key[key] = entry
    entries = sorted(entries_by_key.values(), key=lambda item: (-int(item["size_bytes"]), item["path"]))
    total_bytes = sum(int(item["size_bytes"]) for item in entries)
    merged = dict(current)
    merged["delete_file_count"] = len(entries)
    merged["delete_bytes"] = total_bytes
    merged["delete_human"] = _human_bytes(total_bytes)
    merged["entries"] = entries
    merged["skipped_count"] = int(existing.get("skipped_count") or 0) + int(current.get("skipped_count") or 0)
    merged["skipped_sample"] = list(existing.get("skipped_sample") or [])[:50] + list(current.get("skipped_sample") or [])[:50]
    merged["manifest_note"] = "Entries may include files deleted by previous --apply-delete runs with the same manifest path."
    return merged


def _is_archive_path(path: str) -> bool:
    return "archive" in Path(path).parts


def _media_type(suffix: str) -> str:
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in JSON_SUFFIXES:
        return "json"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in REPORT_SUFFIXES:
        return "report"
    return suffix.lstrip(".") or "other"


def _walk_json_paths(payload: Any, pointer: str = "", key: str = "") -> Iterable[tuple[str, str, Any]]:
    if isinstance(payload, dict):
        for child_key in sorted(payload):
            child_pointer = f"{pointer}/{_escape_json_pointer(str(child_key))}"
            yield from _walk_json_paths(payload[child_key], child_pointer, str(child_key))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            yield from _walk_json_paths(item, f"{pointer}/{index}", key)
    else:
        yield pointer or "/", key, payload


def _candidate_path_values(key: str, value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    key_lower = key.lower()
    suffix = Path(text).suffix.lower()
    looks_like_key = (
        key_lower in {"path", "paths", "file", "filename"}
        or key_lower.endswith("_path")
        or key_lower.endswith("_paths")
        or key_lower.endswith("_artifact")
        or key_lower.endswith("_pdf")
        or key_lower.endswith("_file")
        or key_lower in {"canonical_question_artifact", "canonical_mark_scheme_artifact", "question_crop_path", "mark_scheme_crop_path"}
    )
    looks_like_value = "/" in text and suffix in IMAGE_SUFFIXES | JSON_SUFFIXES | PDF_SUFFIXES | REPORT_SUFFIXES
    if (looks_like_key and suffix in IMAGE_SUFFIXES | JSON_SUFFIXES | PDF_SUFFIXES | REPORT_SUFFIXES) or looks_like_value:
        return [text]
    return []


def _resolve_reference(
    value: str,
    *,
    source_json: Path,
    artifact_root: Path,
    project_root: Path,
) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        return path if path.is_file() else None
    candidates = [
        project_root / value,
        project_root / artifact_root / value,
        project_root / source_json.parent / value,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _artifact_root_for_json(path: Path) -> Path:
    parts = path.parts
    if "output" in parts:
        return Path("output")
    if path.parent.name == "json":
        return path.parent.parent
    return Path(".")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _escape_json_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
