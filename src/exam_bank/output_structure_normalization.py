from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .atomic_json import write_atomic_json
from .core.subject_contract import paper_number_from_component, subject_family_for_component
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

    final_report = _report(root_path, dry_run=dry_run, plan=plan, dir_plan=dir_plan, json_updates=json_updates)
    write_atomic_json(final_report, log_path)
    return final_report


def build_normalization_plan(root: str | Path = "output") -> list[RenamePlanEntry]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    reserved = {path.resolve() for path in root_path.rglob("*")}
    plan: list[RenamePlanEntry] = []
    for path in sorted(root_path.rglob("*.png")):
        relative = path.relative_to(root_path).as_posix()
        canonical_match = CANONICAL_IMAGE_RE.fullmatch(relative)
        if canonical_match:
            current_subject = canonical_match.group("subject")
            expected_subject = subject_family_for_component(
                canonical_match.group("component"),
                year=canonical_match.group("year"),
            )
            if not expected_subject or current_subject == expected_subject:
                continue
            filename = path.name.replace(f"{current_subject}_", f"{expected_subject}_", 1)
            target = _non_conflicting_path(root_path / expected_subject / filename, reserved)
            reserved.add(target.resolve())
            plan.append(
                RenamePlanEntry(
                    old_path=path,
                    new_path=target,
                    reason="component_subject_family_mismatch",
                )
            )
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
    # Only the direct children of the generated-output root are legacy image
    # family directories.  Nested p1/p3/p4/p5/p6 directories are used by
    # taxonomy-facing artifacts (for example topic_packets/p4) and must retain
    # their syllabus component names.
    for path in sorted((item for item in root_path.iterdir() if item.is_dir()), key=lambda item: item.name):
        if path.name not in LEGACY_SUBJECT_DIRS:
            continue
        if path.resolve() in excluded:
            continue
        unplanned_files = [
            item
            for item in path.rglob("*")
            if item.is_file() and item.name != ".DS_Store" and item.resolve() not in excluded
        ]
        if unplanned_files:
            # Unsupported or unrecognized legacy artifacts must remain visible
            # to validation instead of being wholesale relabeled by directory.
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
        year = f"20{match.group('yy')}"
        subject = subject_family_for_component(match.group("component"), year=year)
        if not subject:
            return None
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
    component_subject_mismatches = []
    unsupported_component_era_paths = []
    for path in sorted(root_path.rglob("*")) if root_path.exists() else []:
        relative_parts = path.relative_to(root_path).parts
        top_level = relative_parts[0] if relative_parts else ""
        if top_level in LEGACY_PATH_PARTS:
            legacy_paths.append(str(path))
        # Audit screenshots and rendered topic-packet pages are valid generated
        # PNGs too.  The canonical filename contract applies only inside the
        # top-level image-family namespaces.
        if path.suffix.lower() == ".png" and top_level in CANONICAL_SUBJECTS:
            relative = path.relative_to(root_path).as_posix()
            canonical_match = CANONICAL_IMAGE_RE.fullmatch(relative)
            parts = relative_parts
            filename_subject = path.name.split("_", 1)[0]
            if parts and parts[0] in CANONICAL_SUBJECTS and filename_subject in CANONICAL_SUBJECTS:
                if parts[0] != filename_subject:
                    mixed_subject_paths.append(str(path))
            if canonical_match:
                expected_subject = subject_family_for_component(
                    canonical_match.group("component"),
                    year=canonical_match.group("year"),
                )
                if not expected_subject and paper_number_from_component(canonical_match.group("component")) == "5":
                    unsupported_component_era_paths.append(str(path))
                elif expected_subject and canonical_match.group("subject") != expected_subject:
                    component_subject_mismatches.append(str(path))
            else:
                invalid_pngs.append(str(path))
    report = {
        "ok": not legacy_paths
        and not invalid_pngs
        and not mixed_subject_paths
        and not component_subject_mismatches
        and not unsupported_component_era_paths,
        "legacy_path_count": len(legacy_paths),
        "invalid_png_count": len(invalid_pngs),
        "mixed_subject_path_count": len(mixed_subject_paths),
        "component_subject_mismatch_count": len(component_subject_mismatches),
        "unsupported_component_era_count": len(unsupported_component_era_paths),
        "legacy_paths": legacy_paths[:25],
        "invalid_pngs": invalid_pngs[:25],
        "mixed_subject_paths": mixed_subject_paths[:25],
        "component_subject_mismatches": component_subject_mismatches[:25],
        "unsupported_component_era_paths": unsupported_component_era_paths[:25],
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
    for path in sorted(root.rglob("*.json")):
        if path.parts[-3:] == ("migration", "output_structure_normalization.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        updated = _normalize_component_subject_families(_replace_strings(payload, replacements))
        if updated != payload:
            updates[path] = updated
    return updates


def _normalize_component_subject_families(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_component_subject_families(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {key: _normalize_component_subject_families(item) for key, item in value.items()}
    component = _component_from_mapping(normalized)
    expected_subject = subject_family_for_component(
        component,
        year=normalized.get("year") or normalized.get("canonical_year_folder"),
        session=normalized.get("session") or normalized.get("canonical_session"),
        paper=normalized.get("paper") or normalized.get("question_id") or normalized.get("canonical_paper_id"),
    )
    if expected_subject:
        for field_name in ("paper_family", "subject_family"):
            current = normalized.get(field_name)
            if current in CANONICAL_SUBJECTS and current != expected_subject:
                normalized[field_name] = expected_subject
    return normalized


def _component_from_mapping(value: dict[str, Any]) -> str:
    for field_name in ("component", "source_component"):
        candidate = str(value.get(field_name) or "").strip()
        if re.fullmatch(r"\d{1,2}", candidate):
            return candidate.zfill(2)
    for field_name in ("question_id", "paper", "canonical_paper_id"):
        match = re.match(r"^(\d{1,2})", str(value.get(field_name) or "").strip())
        if match:
            return match.group(1).zfill(2)
    return ""


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, _normalize_canonical_asset_path(value))
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def _normalize_canonical_asset_path(value: str) -> str:
    """Normalize a reversed canonical path even after its file was moved.

    This makes an interrupted migration resumable: a second run can still
    repair JSON references for files that were renamed before the first run
    stopped and therefore no longer appear in the current rename plan.
    """

    path = Path(value)
    if path.parent.name not in CANONICAL_SUBJECTS:
        return value
    relative = f"{path.parent.name}/{path.name}"
    match = CANONICAL_IMAGE_RE.fullmatch(relative)
    if not match:
        return value
    current_subject = match.group("subject")
    expected_subject = subject_family_for_component(
        match.group("component"),
        year=match.group("year"),
    )
    if not expected_subject or current_subject == expected_subject:
        return value
    filename = path.name.replace(f"{current_subject}_", f"{expected_subject}_", 1)
    return str(path.parent.with_name(expected_subject) / filename)


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
