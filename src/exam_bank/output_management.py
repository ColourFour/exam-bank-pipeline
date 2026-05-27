from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .output_layout import (
    ITERATION_DIR_RE,
    PAPER_ARTIFACT_DIR_RE,
    QUESTION_BANK_FILENAME,
    TRIAGE_BASELINE_FILENAME,
)
from .run_status import RUN_MANIFEST_FILENAME, RUN_STATUS_FILENAME


REPORT_FILENAMES = {
    "output_inventory.json",
    "output_inventory.md",
    "output_cleanup_plan.md",
    "generated_output_inventory.txt",
}


@dataclass(frozen=True)
class PathEntry:
    path: Path
    kind: str
    root: Path
    size_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": str(self.path),
            "kind": self.kind,
            "root": str(self.root),
        }
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        return payload


def build_output_inventory(
    roots: Iterable[str | Path],
    *,
    include_size: bool = False,
    max_depth: int = 6,
) -> dict[str, Any]:
    root_paths = [Path(root) for root in roots]
    existing_roots = [root for root in root_paths if root.exists()]
    entries = _scan_entries(existing_roots, include_size=include_size, max_depth=max_depth)
    by_kind: dict[str, list[PathEntry]] = {}
    for entry in entries:
        by_kind.setdefault(entry.kind, []).append(entry)

    artifact_trees = _artifact_trees(existing_roots, include_size=include_size)
    run_ids = sorted(
        {
            entry.path.parent.name
            for entry in by_kind.get("run_status", [])
            if entry.path.name == RUN_STATUS_FILENAME
        }
        | {
            path.name
            for root in existing_roots
            for path in (root / "runs").glob("*")
            if path.is_dir()
        }
    )
    frozen_baselines = by_kind.get("frozen_baseline", [])
    comparison_entries = by_kind.get("triage_comparison", [])
    audit_entries = by_kind.get("audit_folder", [])
    asterion_entries = by_kind.get("asterion_output", [])
    question_banks = by_kind.get("question_bank", [])
    likely_stale = _likely_stale_entries(existing_roots, by_kind, artifact_trees)

    return {
        "schema_name": "exam_bank.output_inventory",
        "schema_version": 1,
        "roots_requested": [str(root) for root in root_paths],
        "output_roots_found": [str(root) for root in existing_roots],
        "run_ids_found": run_ids,
        "question_bank_files": _entry_dicts(question_banks),
        "artifact_trees": _entry_dicts(artifact_trees),
        "triage_iterations": _entry_dicts(by_kind.get("triage_iteration", [])),
        "auto_triage_comparisons": _entry_dicts(
            [entry for entry in comparison_entries if entry.path.name.startswith("comparison.auto-")]
        ),
        "triage_comparisons": _entry_dicts(comparison_entries),
        "asterion_outputs": _asterion_entry_dicts(asterion_entries),
        "audit_folders": _entry_dicts(audit_entries),
        "run_status_files": _entry_dicts(by_kind.get("run_status", [])),
        "frozen_baselines": _entry_dicts(frozen_baselines),
        "likely_stale_folders": _entry_dicts(likely_stale),
        "likely_safe_to_delete_generated_reports": _entry_dicts(_safe_report_entries(by_kind)),
    }


def build_cleanup_plan(
    roots: Iterable[str | Path],
    *,
    include_size: bool = False,
    max_depth: int = 6,
) -> dict[str, Any]:
    inventory = build_output_inventory(roots, include_size=include_size, max_depth=max_depth)
    root_paths = [Path(root) for root in inventory["output_roots_found"]]
    plans: list[dict[str, Any]] = []

    for root in root_paths:
        canonical_paths = _canonical_keep_paths(root)
        for path in canonical_paths:
            if path.exists():
                plans.append(_plan_entry(path, root, "keep: canonical/current", include_size))

        for baseline in root.glob("**/" + TRIAGE_BASELINE_FILENAME):
            plans.append(_plan_entry(baseline, root, "keep: frozen baseline", include_size))

        latest_candidate = _latest_candidate_root(root)
        if latest_candidate is not None:
            plans.append(_plan_entry(latest_candidate, root, "keep: latest candidate", include_size))

        for candidate in _candidate_roots(root):
            if latest_candidate is not None and candidate == latest_candidate:
                continue
            plans.append(_plan_entry(candidate, root, "archive candidate", include_size))

        audit_dirs = sorted(
            [path for path in (root / "audits").glob("*") if path.is_dir()],
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        for audit_dir in audit_dirs[:-1]:
            plans.append(_plan_entry(audit_dir, root, "archive old audit/report", include_size))
        if audit_dirs:
            plans.append(_plan_entry(audit_dirs[-1], root, "keep: latest audit/report", include_size))

        for report in _report_paths(root):
            plans.append(_plan_entry(report, root, "safe delete generated report", include_size))

        for unknown in _unknown_top_level_paths(root):
            plans.append(_plan_entry(unknown, root, "unknown/manual review", include_size))

    return {
        "schema_name": "exam_bank.output_cleanup_plan",
        "schema_version": 1,
        "dry_run": True,
        "roots": [str(root) for root in root_paths],
        "entries": _dedupe_plan(plans),
        "notes": [
            "This is a dry-run plan only. It never deletes or moves files.",
            "Frozen triage baselines are always classified as keep.",
            "Archive/delete classifications are suggestions for human review.",
        ],
    }


def write_inventory_outputs(
    report: dict[str, Any],
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> None:
    if json_path is not None:
        _write_json(Path(json_path), report)
    if markdown_path is not None:
        _write_text(Path(markdown_path), render_inventory_markdown(report))


def write_cleanup_plan_outputs(
    plan: dict[str, Any],
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> None:
    if json_path is not None:
        _write_json(Path(json_path), plan)
    if markdown_path is not None:
        _write_text(Path(markdown_path), render_cleanup_plan_markdown(plan))


def render_inventory_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Generated Output Inventory",
        "",
        f"- Roots found: {len(report.get('output_roots_found', []))}",
        f"- Run IDs found: {len(report.get('run_ids_found', []))}",
        f"- Question banks: {len(report.get('question_bank_files', []))}",
        f"- Artifact trees: {len(report.get('artifact_trees', []))}",
        f"- Triage iterations: {len(report.get('triage_iterations', []))}",
        f"- Frozen baselines: {len(report.get('frozen_baselines', []))}",
        "",
    ]
    for title, key in [
        ("Output Roots", "output_roots_found"),
        ("Question Banks", "question_bank_files"),
        ("Artifact Trees", "artifact_trees"),
        ("Frozen Baselines", "frozen_baselines"),
        ("Asterion Outputs", "asterion_outputs"),
        ("Likely Stale Folders", "likely_stale_folders"),
        ("Safe-To-Delete Generated Reports", "likely_safe_to_delete_generated_reports"),
    ]:
        lines.extend(_markdown_section(title, report.get(key, [])))
    return "\n".join(lines).rstrip() + "\n"


def render_cleanup_plan_markdown(plan: dict[str, Any]) -> str:
    lines = ["# Generated Output Cleanup Plan", "", "Dry run only. No files were deleted or moved.", ""]
    entries = plan.get("entries") if isinstance(plan.get("entries"), list) else []
    by_action: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if isinstance(entry, dict):
            by_action.setdefault(str(entry.get("action") or "unknown/manual review"), []).append(entry)
    for action in sorted(by_action):
        lines.append(f"## {action}")
        for entry in sorted(by_action[action], key=lambda item: str(item.get("path") or "")):
            size = f" ({entry['size_bytes']} bytes)" if "size_bytes" in entry else ""
            lines.append(f"- `{entry.get('path')}`{size}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _scan_entries(roots: list[Path], *, include_size: bool, max_depth: int) -> list[PathEntry]:
    entries: list[PathEntry] = []
    for root in roots:
        for path in _walk_limited(root, max_depth=max_depth):
            kind = _classify_path(path)
            if kind:
                entries.append(PathEntry(path=path, kind=kind, root=root, size_bytes=_size(path) if include_size else None))
    return entries


def _walk_limited(root: Path, *, max_depth: int) -> Iterable[Path]:
    if not root.exists():
        return
    yield root
    for path in root.rglob("*"):
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if depth <= max_depth:
            yield path


def _classify_path(path: Path) -> str:
    if path.name == QUESTION_BANK_FILENAME and path.is_file():
        return "question_bank"
    if path.name == TRIAGE_BASELINE_FILENAME and path.is_file():
        return "frozen_baseline"
    if path.is_dir() and ITERATION_DIR_RE.fullmatch(path.name) and path.parent.name == "triage":
        return "triage_iteration"
    if path.is_file() and path.name.startswith("comparison.") and path.suffix == ".json":
        return "triage_comparison"
    if path.is_dir() and path.parent.name == "audits":
        return "audit_folder"
    if path.is_file() and path.name in {RUN_STATUS_FILENAME, RUN_MANIFEST_FILENAME, "batch_status.jsonl"}:
        return "run_status"
    if path.is_file() and "asterion" in path.parts and path.suffix in {".json", ".md", ".csv"}:
        return "asterion_output"
    return ""


def _artifact_trees(roots: list[Path], *, include_size: bool) -> list[PathEntry]:
    entries: list[PathEntry] = []
    for root in roots:
        for path in _walk_limited(root, max_depth=5):
            if path.is_dir() and PAPER_ARTIFACT_DIR_RE.fullmatch(path.name):
                entries.append(
                    PathEntry(
                        path=path,
                        kind="artifact_tree",
                        root=root,
                        size_bytes=_size(path) if include_size else None,
                    )
                )
    return sorted(set(entries), key=lambda entry: str(entry.path))


def _likely_stale_entries(
    roots: list[Path],
    by_kind: dict[str, list[PathEntry]],
    artifact_trees: list[PathEntry],
) -> list[PathEntry]:
    stale: list[PathEntry] = []
    known_top_level = {
        "json",
        "debug",
        "run_status",
        "triage",
        "auto_triage",
        "audits",
        "asterion",
        "runs",
        "current",
        "candidates",
        "logs",
        "artifacts",
    }
    known_top_level.update({entry.path.name for entry in artifact_trees})
    for root in roots:
        for child in root.iterdir() if root.exists() else []:
            if child.name in known_top_level:
                continue
            if child.is_dir() and (child / "json" / QUESTION_BANK_FILENAME).exists():
                stale.append(PathEntry(path=child, kind="likely_stale_folder", root=root))
    return stale


def _safe_report_entries(by_kind: dict[str, list[PathEntry]]) -> list[PathEntry]:
    entries: list[PathEntry] = []
    entries.extend(by_kind.get("triage_comparison", []))
    for audit in by_kind.get("audit_folder", []):
        entries.append(audit)
    return entries


def _canonical_keep_paths(root: Path) -> list[Path]:
    return [
        root / "json" / QUESTION_BANK_FILENAME,
        root / "current" / QUESTION_BANK_FILENAME,
        root / "current" / RUN_MANIFEST_FILENAME,
    ] + [root / f"p{number}" for number in range(1, 7)]


def _candidate_roots(root: Path) -> list[Path]:
    candidates: list[Path] = []
    if root.name == "output_ocr_candidate":
        candidates.append(root)
    candidates.extend(path for path in (root / "candidates" / "ocr").glob("*") if path.is_dir())
    candidates.extend(
        path
        for path in root.glob("*candidate*")
        if path.is_dir() and (path / "json" / QUESTION_BANK_FILENAME).exists()
    )
    return sorted(set(candidates), key=lambda path: str(path))


def _latest_candidate_root(root: Path) -> Path | None:
    candidates = _candidate_roots(root)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _report_paths(root: Path) -> list[Path]:
    paths = [path for path in root.glob("*") if path.is_file() and path.name in REPORT_FILENAMES]
    paths.extend(path for path in root.glob("**/comparison.*.json") if path.is_file())
    return sorted(set(paths), key=lambda path: str(path))


def _unknown_top_level_paths(root: Path) -> list[Path]:
    known = {
        ".gitkeep",
        "json",
        "debug",
        "run_status",
        "triage",
        "audits",
        "asterion",
        "runs",
        "current",
        "candidates",
        "logs",
        "artifacts",
        *[f"p{number}" for number in range(1, 7)],
    }
    unknown: list[Path] = []
    for child in root.iterdir() if root.exists() else []:
        if child.name in known or child.name in REPORT_FILENAMES:
            continue
        if child.is_dir() and ITERATION_DIR_RE.fullmatch(child.name):
            unknown.append(child)
        elif child.is_dir() and not (child / "json" / QUESTION_BANK_FILENAME).exists():
            unknown.append(child)
    return unknown


def _plan_entry(path: Path, root: Path, action: str, include_size: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": str(path), "root": str(root), "action": action}
    if include_size:
        entry["size_bytes"] = _size(path)
    return entry


def _dedupe_plan(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "keep: canonical/current": 0,
        "keep: frozen baseline": 0,
        "keep: latest candidate": 1,
        "keep: latest audit/report": 2,
        "archive candidate": 3,
        "archive old audit/report": 4,
        "safe delete generated report": 5,
        "unknown/manual review": 6,
    }
    by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        path = str(entry.get("path") or "")
        existing = by_path.get(path)
        if existing is None or priority.get(str(entry.get("action")), 99) < priority.get(str(existing.get("action")), 99):
            by_path[path] = entry
    return [by_path[path] for path in sorted(by_path)]


def _entry_dicts(entries: list[PathEntry]) -> list[dict[str, Any]]:
    return [entry.as_dict() for entry in sorted(entries, key=lambda item: str(item.path))]


def _asterion_entry_dicts(entries: list[PathEntry]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: str(item.path)):
        payload = entry.as_dict()
        if entry.path.suffix == ".json":
            last_run_at = _json_last_run_at(entry.path)
            if last_run_at:
                payload["last_run_at"] = last_run_at
        values.append(payload)
    return values


def _json_last_run_at(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    value = payload.get("last_run_at") or payload.get("generated_at")
    return str(value) if value else ""


def _markdown_section(title: str, value: Any) -> list[str]:
    lines = [f"## {title}"]
    if not value:
        return lines + ["- None", ""]
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                size = f" ({item['size_bytes']} bytes)" if "size_bytes" in item else ""
                last_run = f" - last run: `{item['last_run_at']}`" if "last_run_at" in item else ""
                lines.append(f"- `{item.get('path')}`{size}{last_run}")
            else:
                lines.append(f"- `{item}`")
    else:
        lines.append(f"- `{value}`")
    lines.append("")
    return lines


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
