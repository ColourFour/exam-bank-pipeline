from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .atomic_json import write_atomic_json
from .output_layout import (
    CANONICAL_SUBJECTS,
    LEGACY_SUBJECT_DIRS,
    canonical_asset_filename,
)


LEGACY_IMAGE_RE = re.compile(
    r"^(?P<family>p1|p3|p4|p5|p6)/(?P<paper>(?P<component>\d{1,2})(?P<session>[a-z]+)(?P<yy>\d{2}))/"
    r"(?P<role>questions|mark_scheme)/q(?P<question>\d+)\.png$"
)
CANONICAL_IMAGE_RE = re.compile(
    r"^(?P<subject>pm1|pm3|stats|mechanics)/"
    r"(?P=subject)_(?P<year>\d{4})_(?P<session>[msw]\d{2})_(?P<component>\d{2})_(?P<paper_type>qp|ms)_q\d{2}_"
    r"(?P<asset_type>question|markscheme)(?:_v\d+)?\.png$"
)
LEGACY_PATH_PARTS = {"p1", "p3", "p4", "p5", "p6"}
MIGRATION_LOG_PATH = Path("output/migration/output_structure_normalization.json")


@dataclass(frozen=True)
class RenamePlanEntry:
    old_path: Path
    new_path: Path
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "old_path": str(self.old_path),
            "new_path": str(self.new_path),
            "reason": self.reason,
        }


def normalize_output_structure(root: str | Path = "output", *, dry_run: bool = False) -> dict[str, Any]:
    root_path = Path(root)
    plan = build_normalization_plan(root_path)
    json_updates = _json_update_plan(root_path, plan)
    dir_plan = build_legacy_directory_plan(root_path, exclude_paths={entry.old_path for entry in plan})
    log_path = root_path / "migration" / "output_structure_normalization.json"
    if not dry_run and not plan and not dir_plan and not json_updates and log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8"))
    report = _report(root_path, dry_run=dry_run, plan=plan, dir_plan=dir_plan, json_updates=json_updates)
    if dry_run:
        return report

    for entry in plan:
        entry.new_path.parent.mkdir(parents=True, exist_ok=True)
        entry.old_path.rename(entry.new_path)
        _prune_empty_parents(entry.old_path.parent, stop=root_path)

    for path, payload in json_updates.items():
        write_atomic_json(payload, path)

    _remove_legacy_ds_store_files(root_path)
    _prune_empty_legacy_dirs(root_path)
    for old_path, new_path in dir_plan:
        if not old_path.exists():
            continue
        new_path.parent.mkdir(parents=True, exist_ok=True)
        target = _non_conflicting_directory(new_path)
        old_path.rename(target)
        _prune_empty_parents(old_path.parent, stop=root_path)

    write_atomic_json(_report(root_path, dry_run=dry_run, plan=plan, dir_plan=dir_plan, json_updates=json_updates), log_path)
    return report


def build_normalization_plan(root: str | Path = "output") -> list[RenamePlanEntry]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    reserved = {path.resolve() for path in root_path.rglob("*")}
    plan: list[RenamePlanEntry] = []
    for path in sorted(root_path.rglob("*.png")):
        relative = path.relative_to(root_path).as_posix()
        if CANONICAL_IMAGE_RE.fullmatch(relative):
            continue
        match = LEGACY_IMAGE_RE.fullmatch(relative)
        if not match:
            continue
        canonical_relative = legacy_image_path_to_canonical(relative)
        if canonical_relative is None:
            continue
        filename = Path(canonical_relative).name
        subject = Path(canonical_relative).parts[0]
        target = _non_conflicting_path(root_path / subject / filename, reserved)
        reserved.add(target.resolve())
        reason = "legacy_folder_and_filename"
        if target.name != filename:
            reason = "legacy_folder_and_filename_conflict_v2"
        plan.append(RenamePlanEntry(old_path=path, new_path=target, reason=reason))
    return plan


def build_legacy_directory_plan(root: str | Path = "output", *, exclude_paths: set[Path] | None = None) -> list[tuple[Path, Path]]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    excluded = {path.resolve() for path in (exclude_paths or set())}
    plan: list[tuple[Path, Path]] = []
    for path in sorted((item for item in root_path.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        if path.name not in LEGACY_SUBJECT_DIRS:
            continue
        if path.resolve() in excluded:
            continue
        plan.append((path, path.with_name(LEGACY_SUBJECT_DIRS[path.name])))
    return plan


def legacy_image_path_to_canonical(path_value: str) -> str | None:
    path = Path(path_value)
    parts = path.parts
    candidates = [Path(*parts[index:]).as_posix() for index, part in enumerate(parts) if part in LEGACY_SUBJECT_DIRS]
    for candidate in candidates:
        match = LEGACY_IMAGE_RE.fullmatch(candidate)
        if not match:
            continue
        subject = LEGACY_SUBJECT_DIRS[match.group("family")]
        year = f"20{match.group('yy')}"
        session = f"{_session_letter(match.group('session'))}{match.group('yy')}"
        paper_type = "qp" if match.group("role") == "questions" else "ms"
        asset_type = "question" if paper_type == "qp" else "markscheme"
        filename = canonical_asset_filename(
            subject=subject,
            year=year,
            session=session,
            paper_type=paper_type,
            question_number=match.group("question"),
            asset_type=asset_type,
            component=match.group("component"),
        )
        return f"{subject}/{filename}"
    return None


def validate_normalized_output(root: str | Path = "output") -> dict[str, Any]:
    root_path = Path(root)
    legacy_paths = []
    invalid_pngs = []
    mixed_subject_paths = []
    for path in sorted(root_path.rglob("*")) if root_path.exists() else []:
        if any(part in LEGACY_PATH_PARTS for part in path.relative_to(root_path).parts):
            legacy_paths.append(str(path))
        if path.suffix.lower() == ".png":
            relative = path.relative_to(root_path).as_posix()
            parts = path.relative_to(root_path).parts
            filename_subject = path.name.split("_", 1)[0]
            if parts and parts[0] in CANONICAL_SUBJECTS and filename_subject in CANONICAL_SUBJECTS:
                if parts[0] != filename_subject:
                    mixed_subject_paths.append(str(path))
            if not CANONICAL_IMAGE_RE.fullmatch(relative):
                invalid_pngs.append(str(path))
    report = {
        "ok": not legacy_paths and not invalid_pngs and not mixed_subject_paths,
        "legacy_path_count": len(legacy_paths),
        "invalid_png_count": len(invalid_pngs),
        "mixed_subject_path_count": len(mixed_subject_paths),
        "legacy_paths": legacy_paths[:25],
        "invalid_pngs": invalid_pngs[:25],
        "mixed_subject_paths": mixed_subject_paths[:25],
    }
    return report


def _json_update_plan(root: Path, plan: list[RenamePlanEntry]) -> dict[Path, Any]:
    replacements: dict[str, str] = {}
    for entry in plan:
        old_rel = entry.old_path.relative_to(root).as_posix()
        new_rel = entry.new_path.relative_to(root).as_posix()
        replacements[old_rel] = new_rel
        replacements[str(entry.old_path)] = str(entry.new_path)
        try:
            replacements[str(entry.old_path.resolve())] = str(entry.new_path.resolve())
        except OSError:
            pass

    updates: dict[Path, Any] = {}
    if not replacements:
        return updates
    for path in sorted(root.rglob("*.json")):
        if path.parts[-3:] == ("migration", "output_structure_normalization.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        updated = _replace_strings(payload, replacements)
        if updated != payload:
            updates[path] = updated
    return updates


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def _report(
    root: Path,
    *,
    dry_run: bool,
    plan: list[RenamePlanEntry],
    dir_plan: list[tuple[Path, Path]],
    json_updates: dict[Path, Any],
) -> dict[str, Any]:
    validation = validate_normalized_output(root)
    return {
        "schema_name": "exam_bank.output_structure_normalization",
        "schema_version": 1,
        "dry_run": dry_run,
        "root": str(root),
        "folders_renamed": _folders_renamed(plan, dir_plan, root),
        "files_renamed": len(plan),
        "conflicts_resolved": sum(1 for entry in plan if "_conflict_" in entry.reason),
        "metadata_outputs_updated": [str(path) for path in sorted(json_updates)],
        "remaining_legacy_references": 0 if plan else validation["legacy_path_count"],
        "renames": [entry.as_dict() for entry in plan],
        "validation": validation,
    }


def _folders_renamed(plan: list[RenamePlanEntry], dir_plan: list[tuple[Path, Path]], root: Path) -> list[dict[str, str]]:
    pairs = {
        (entry.old_path.relative_to(root).parts[0], entry.new_path.relative_to(root).parts[0])
        for entry in plan
        if entry.old_path.relative_to(root).parts
    }
    pairs.update((old.relative_to(root).as_posix(), new.relative_to(root).as_posix()) for old, new in dir_plan)
    return [{"old_path": str(root / old), "new_path": str(root / new)} for old, new in sorted(pairs)]


def _session_letter(session: str) -> str:
    normalized = session.lower()
    if normalized.startswith(("spring", "march", "m")):
        return "m"
    if normalized.startswith(("summer", "june", "s")):
        return "s"
    if normalized.startswith(("autumn", "winter", "nov", "w")):
        return "w"
    return "x"


def _non_conflicting_path(target: Path, reserved: set[Path]) -> Path:
    if target.resolve() not in reserved:
        return target
    stem = target.stem
    suffix = target.suffix
    version = 2
    while True:
        candidate = target.with_name(f"{stem}_v{version}{suffix}")
        if candidate.resolve() not in reserved:
            return candidate
        version += 1


def _non_conflicting_directory(target: Path) -> Path:
    if not target.exists():
        return target
    version = 2
    while True:
        candidate = target.with_name(f"{target.name}_v{version}")
        if not candidate.exists():
            return candidate
        version += 1


def _remove_legacy_ds_store_files(root: Path) -> None:
    for path in root.rglob(".DS_Store"):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in LEGACY_PATH_PARTS for part in relative_parts):
            path.unlink()


def _prune_empty_legacy_dirs(root: Path) -> None:
    legacy_dirs = [
        path
        for path in root.rglob("*")
        if path.is_dir() and any(part in LEGACY_PATH_PARTS for part in path.relative_to(root).parts)
    ]
    for path in sorted(legacy_dirs, key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue


def _prune_empty_parents(path: Path, *, stop: Path) -> None:
    stop = stop.resolve()
    current = path
    while current.exists() and current.resolve() != stop:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent
