from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import fitz

from .atomic_json import write_atomic_json


CORPUS_SESSION_IDENTITY_SCHEMA = "exam_bank.corpus_session_identity_validation"
CORPUS_SESSION_IDENTITY_VERSION = 1
DEFAULT_CORPUS_ROOT = Path("input")
DEFAULT_REPORT_PATH = Path(
    "manifests/migrations/caie_9709_corpus_session_identity.v1.json"
)

_PDF_NAME = re.compile(
    r"^(?P<syllabus>\d{4})_(?P<session>[msw])(?P<yy>\d{2})_"
    r"(?P<document_type>qp|ms|er|gt)_(?P<component>\d{1,2})\.pdf$",
    re.IGNORECASE,
)
_SESSION_PATTERNS = {
    "m": (
        re.compile(r"\bfebruary\s*/\s*march\b", re.IGNORECASE),
        re.compile(r"\bmarch\s+20\d{2}\b", re.IGNORECASE),
    ),
    "s": (
        re.compile(r"\bmay\s*/\s*june\b", re.IGNORECASE),
        re.compile(r"\bjune\s+20\d{2}\b", re.IGNORECASE),
    ),
    "w": (
        re.compile(r"\boctober\s*/\s*november\b", re.IGNORECASE),
        re.compile(r"\bnovember\s+20\d{2}\b", re.IGNORECASE),
    ),
}
_SESSION_NAMES = {"m": "March", "s": "June", "w": "November"}


class CorpusSessionIdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class PdfSessionEvidence:
    session_code: str | None
    status: str
    matched_session_codes: tuple[str, ...]
    first_page_text_sha256: str | None
    first_page_character_count: int
    error: str | None = None


@dataclass(frozen=True)
class _ScannedDocument:
    path: Path
    relative_path: str
    syllabus: str
    raw_session_code: str
    yy: str
    document_type: str
    component: str
    sha256: str
    size_bytes: int
    evidence: PdfSessionEvidence

    @property
    def pair_key(self) -> tuple[str, str, str, str]:
        return self.syllabus, self.raw_session_code, self.yy, self.component


def detect_pdf_session(path: str | Path) -> PdfSessionEvidence:
    """Detect the examination session from first-page publisher text only."""

    path = Path(path)
    display_errors = bool(fitz.TOOLS.mupdf_display_errors())
    fitz.TOOLS.mupdf_display_errors(False)
    try:
        with fitz.open(path) as document:
            if document.page_count < 1:
                return PdfSessionEvidence(None, "unreadable", (), None, 0, "pdf_has_no_pages")
            text = " ".join(document[0].get_text("text").split())
    except Exception as exc:
        return PdfSessionEvidence(
            None,
            "unreadable",
            (),
            None,
            0,
            f"pdf_unreadable:{exc.__class__.__name__}",
        )
    finally:
        fitz.TOOLS.mupdf_display_errors(display_errors)

    # Session wording is part of the publisher header. Limiting the evidence window
    # prevents incidental month names later in examiner-report prose from influencing it.
    evidence_text = text[:2500]
    matched = tuple(
        code
        for code, patterns in _SESSION_PATTERNS.items()
        if any(pattern.search(evidence_text) for pattern in patterns)
    )
    digest = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
    if len(matched) == 1:
        return PdfSessionEvidence(matched[0], "resolved", matched, digest, len(evidence_text))
    if not matched:
        return PdfSessionEvidence(None, "unknown", (), digest, len(evidence_text), "session_not_found")
    return PdfSessionEvidence(None, "ambiguous", matched, digest, len(evidence_text), "multiple_sessions_found")


def normalize_corpus_session_identity(
    *,
    root: str | Path = DEFAULT_CORPUS_ROOT,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    apply: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Plan or transactionally apply content-evidenced corpus filename corrections."""

    root = Path(root).resolve()
    report_path = Path(report_path)
    _validate_report_path(root, report_path)
    scan = _scan_corpus(root)
    plans, unresolved, blockers = _build_plan(root, scan)
    operation_id = _operation_id(plans)
    generated = generated_at or _utc_now_iso()
    report = _build_report(
        root=root,
        report_path=report_path,
        generated_at=generated,
        operation_id=operation_id,
        mode="apply" if apply else "dry_run",
        scan=scan,
        plans=plans,
        unresolved=unresolved,
        blockers=blockers,
        operation_state="planned",
        operation_ok=not blockers and not unresolved,
        post_apply_mismatch_count=None,
    )
    if not apply:
        write_atomic_json(report, report_path, sort_keys=True)
        return report
    if blockers or unresolved:
        write_atomic_json(report, report_path, sort_keys=True)
        return report
    if not plans:
        report["operation_state"] = "completed"
        report["post_apply_mismatch_count"] = 0
        report["ok"] = True
        write_atomic_json(report, report_path, sort_keys=True)
        return report

    stage_root = _stage_root(root, operation_id)
    if stage_root.exists():
        raise CorpusSessionIdentityError(f"session-normalization stage already exists: {stage_root}")
    stage_root.mkdir(parents=True)
    staged: list[tuple[dict[str, Any], Path]] = []
    in_progress = dict(report)
    in_progress["operation_state"] = "apply_in_progress"
    in_progress["stage_root"] = str(stage_root)
    write_atomic_json(in_progress, report_path, sort_keys=True)
    try:
        for index, entry in enumerate(plans):
            source = _contained_path(root, entry["source_path"])
            _verify_source(source, entry)
            stage = stage_root / f"{index:04d}.pdf"
            os.replace(source, stage)
            staged.append((entry, stage))
        _fsync_directories({Path(entry["source_absolute_path"]).parent for entry in plans} | {stage_root})

        for entry, stage in staged:
            destination = _contained_path(root, entry["destination_path"], require_missing=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination = _contained_path(root, entry["destination_path"], require_missing=True)
            os.replace(stage, destination)
            entry["status"] = "renamed"
        _fsync_directories({Path(entry["destination_absolute_path"]).parent for entry in plans} | {stage_root})
    except BaseException as exc:
        rollback_errors = _rollback_moves(root, staged)
        failed = _build_report(
            root=root,
            report_path=report_path,
            generated_at=generated,
            operation_id=operation_id,
            mode="apply",
            scan=scan,
            plans=plans,
            unresolved=unresolved,
            blockers=blockers,
            operation_state="rolled_back" if not rollback_errors else "rollback_incomplete",
            operation_ok=False,
            post_apply_mismatch_count=None,
        )
        failed["operation_error"] = f"{type(exc).__name__}: {exc}"
        failed["rollback_errors"] = rollback_errors
        write_atomic_json(failed, report_path, sort_keys=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            stage_root.rmdir()
        with contextlib.suppress(OSError):
            stage_root.parent.rmdir()

    post_scan = _scan_corpus(root)
    post_plans, post_unresolved, post_blockers = _build_plan(root, post_scan)
    completed = _build_report(
        root=root,
        report_path=report_path,
        generated_at=generated,
        operation_id=operation_id,
        mode="apply",
        scan=post_scan,
        plans=plans,
        unresolved=post_unresolved,
        blockers=post_blockers,
        operation_state="completed",
        operation_ok=not post_unresolved and not post_blockers and not post_plans,
        post_apply_mismatch_count=len(post_plans),
    )
    completed["renamed_count"] = len(plans)
    completed["ok"] = not post_unresolved and not post_blockers and not post_plans
    write_atomic_json(completed, report_path, sort_keys=True)
    if not completed["ok"]:
        raise CorpusSessionIdentityError("corpus session normalization failed post-apply verification")
    return completed


def update_corpus_manifest_from_session_report(
    *,
    manifest_path: str | Path,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    root: str | Path = DEFAULT_CORPUS_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Rebind a corpus manifest to an already-applied, hash-verified rename ledger.

    Source URLs deliberately remain unchanged: they identify the original mirror
    payload whose filename was wrong, and hydration still verifies that payload by hash.
    """

    from .corpus import CORPUS_MANIFEST_SCHEMA, CORPUS_MANIFEST_VERSION, validate_corpus_manifest

    manifest_path = Path(manifest_path)
    report_path = Path(report_path)
    root = Path(root).resolve()
    manifest = _read_json_object(manifest_path, label="corpus manifest")
    report = _read_json_object(report_path, label="session-identity report")
    if manifest.get("schema_name") != CORPUS_MANIFEST_SCHEMA:
        raise CorpusSessionIdentityError("unexpected corpus manifest schema")
    if manifest.get("schema_version") != CORPUS_MANIFEST_VERSION:
        raise CorpusSessionIdentityError("unsupported corpus manifest version")
    if report.get("schema_name") != CORPUS_SESSION_IDENTITY_SCHEMA:
        raise CorpusSessionIdentityError("unexpected session-identity report schema")
    if (
        report.get("mode") != "apply"
        or report.get("operation_state") != "completed"
        or report.get("ok") is not True
        or report.get("post_apply_mismatch_count") != 0
    ):
        raise CorpusSessionIdentityError("session-identity report is not a successful applied migration")
    if Path(str(report.get("root") or "")).resolve() != root:
        raise CorpusSessionIdentityError("session-identity report root does not match corpus root")
    documents = manifest.get("documents")
    entries = report.get("entries")
    if not isinstance(documents, list) or not all(isinstance(item, dict) for item in documents):
        raise CorpusSessionIdentityError("corpus manifest documents must be objects")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise CorpusSessionIdentityError("session-identity report entries must be objects")

    entries_by_source: dict[str, dict[str, Any]] = {}
    for entry in entries:
        source_path = str(entry.get("source_path") or "")
        if not source_path or source_path in entries_by_source:
            raise CorpusSessionIdentityError(f"duplicate or empty report source path: {source_path!r}")
        if entry.get("status") != "renamed":
            raise CorpusSessionIdentityError(f"report entry was not applied: {source_path}")
        entries_by_source[source_path] = entry

    original_manifest_sha256 = _file_sha256(manifest_path)
    seen_report_sources: set[str] = set()
    rebound_documents: list[dict[str, Any]] = []
    for document in documents:
        rebound = dict(document)
        local_path = str(document.get("local_path") or "")
        entry = entries_by_source.get(local_path)
        if entry is not None:
            if document.get("sha256") != entry.get("sha256"):
                raise CorpusSessionIdentityError(f"manifest/report hash mismatch: {local_path}")
            if document.get("size_bytes") != entry.get("size_bytes"):
                raise CorpusSessionIdentityError(f"manifest/report size mismatch: {local_path}")
            destination_path = str(entry.get("destination_path") or "")
            destination = _contained_path(root, destination_path)
            _verify_source(destination, entry)
            destination_match = _PDF_NAME.fullmatch(destination.name)
            if destination_match is None:
                raise CorpusSessionIdentityError(f"invalid destination filename: {destination_path}")
            original_source_url = str(document.get("source_url") or "")
            rebound.update(
                {
                    "document_id": destination.stem.lower(),
                    "session_code": entry["internal_session_code"],
                    "session": entry["internal_session"],
                    "local_path": destination_path,
                    "source_payload_filename": Path(local_path).name,
                    "session_identity_evidence_sha256": entry["evidence"][
                        "first_page_text_sha256"
                    ],
                    "source_url": original_source_url,
                }
            )
            seen_report_sources.add(local_path)
        rebound_documents.append(rebound)
    missing = sorted(set(entries_by_source) - seen_report_sources)
    if missing:
        raise CorpusSessionIdentityError(
            f"session-identity report sources are absent from corpus manifest: {missing[:10]}"
        )
    rebound_documents.sort(key=lambda item: str(item.get("local_path") or ""))
    updated = dict(manifest)
    updated["generated_at"] = generated_at or _utc_now_iso()
    updated["record_count"] = len(rebound_documents)
    updated["documents"] = rebound_documents
    updated["documents_sha256"] = _documents_sha256(rebound_documents)
    updated["derivation"] = {
        "operation": "content_evidenced_session_filename_normalization",
        "source_manifest_sha256": original_manifest_sha256,
        "session_identity_report": str(report_path),
        "session_identity_report_sha256": _file_sha256(report_path),
        "renamed_document_count": len(entries),
        "source_url_policy": "preserved_original_mirror_payload_url",
    }
    errors = validate_corpus_manifest(updated)
    if errors:
        raise CorpusSessionIdentityError(f"updated corpus manifest is invalid: {errors}")
    write_atomic_json(updated, manifest_path, sort_keys=True)
    return {
        "schema_name": "exam_bank.corpus_manifest_session_rebind",
        "schema_version": 1,
        "ok": True,
        "manifest": str(manifest_path),
        "source_manifest_sha256": original_manifest_sha256,
        "updated_manifest_sha256": _file_sha256(manifest_path),
        "documents_sha256": updated["documents_sha256"],
        "record_count": len(rebound_documents),
        "renamed_document_count": len(entries),
    }


def _scan_corpus(root: Path) -> list[_ScannedDocument]:
    documents: list[_ScannedDocument] = []
    for path in sorted(root.glob("pastpapers/9709/**/*.pdf")):
        match = _PDF_NAME.fullmatch(path.name)
        if match is None:
            continue
        relative = path.relative_to(root).as_posix()
        year = 2000 + int(match.group("yy"))
        if len(path.parts) < 3 or path.parts[-3] != str(year):
            raise CorpusSessionIdentityError(f"filename/path year mismatch: {relative}")
        documents.append(
            _ScannedDocument(
                path=path,
                relative_path=relative,
                syllabus=match.group("syllabus"),
                raw_session_code=match.group("session").lower(),
                yy=match.group("yy"),
                document_type=match.group("document_type").lower(),
                component=match.group("component").zfill(2),
                sha256=_file_sha256(path),
                size_bytes=path.stat().st_size,
                evidence=detect_pdf_session(path),
            )
        )
    if not documents:
        raise CorpusSessionIdentityError(f"no recognized corpus PDFs found under {root}")
    return documents


def _build_plan(
    root: Path,
    documents: Sequence[_ScannedDocument],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str, str, str], list[_ScannedDocument]] = {}
    for document in documents:
        by_pair.setdefault(document.pair_key, []).append(document)

    resolved: dict[str, tuple[str | None, str]] = {}
    unresolved: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for document in documents:
        direct = document.evidence.session_code
        counterpart_codes = {
            item.evidence.session_code
            for item in by_pair[document.pair_key]
            if item.evidence.session_code is not None
        }
        if direct is not None:
            if counterpart_codes - {direct}:
                blockers.append(
                    {
                        "path": document.relative_path,
                        "reason": "paired_documents_disagree",
                        "direct_session_code": direct,
                        "paired_session_codes": sorted(counterpart_codes),
                    }
                )
            resolved[document.relative_path] = (direct, "first_page")
        elif len(counterpart_codes) == 1:
            resolved[document.relative_path] = (next(iter(counterpart_codes)), "paired_document")
        else:
            resolved[document.relative_path] = (None, "unresolved")
            unresolved.append(
                {
                    "path": document.relative_path,
                    "raw_session_code": document.raw_session_code,
                    "document_type": document.document_type,
                    "component": document.component,
                    "evidence": asdict(document.evidence),
                    "paired_session_codes": sorted(counterpart_codes),
                }
            )

    plans: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    destination_owners: dict[Path, str] = {}
    for document in documents:
        internal_code, evidence_source = resolved[document.relative_path]
        if internal_code is None or internal_code == document.raw_session_code:
            continue
        destination_name = _renamed_filename(document.path.name, internal_code)
        destination = document.path.with_name(destination_name)
        destination_relative = destination.relative_to(root).as_posix()
        entry = {
            "status": "planned",
            "source_path": document.relative_path,
            "source_absolute_path": str(document.path),
            "destination_path": destination_relative,
            "destination_absolute_path": str(destination),
            "sha256": document.sha256,
            "size_bytes": document.size_bytes,
            "document_type": document.document_type,
            "component": document.component,
            "year": 2000 + int(document.yy),
            "raw_session_code": document.raw_session_code,
            "internal_session_code": internal_code,
            "internal_session": _SESSION_NAMES[internal_code],
            "evidence_source": evidence_source,
            "evidence": asdict(document.evidence),
        }
        plans.append(entry)
        source_paths.add(document.path)
        previous = destination_owners.get(destination)
        if previous is not None:
            blockers.append(
                {
                    "path": document.relative_path,
                    "reason": "duplicate_destination",
                    "destination_path": destination_relative,
                    "other_source_path": previous,
                }
            )
        destination_owners[destination] = document.relative_path

    for entry in plans:
        destination = Path(entry["destination_absolute_path"])
        if destination.exists() and destination not in source_paths:
            blockers.append(
                {
                    "path": entry["source_path"],
                    "reason": "destination_conflict",
                    "destination_path": entry["destination_path"],
                }
            )
    plans.sort(key=lambda item: item["source_path"])
    unresolved.sort(key=lambda item: item["path"])
    blockers.sort(key=lambda item: (item.get("path", ""), item["reason"]))
    return plans, unresolved, blockers


def _build_report(
    *,
    root: Path,
    report_path: Path,
    generated_at: str,
    operation_id: str,
    mode: str,
    scan: Sequence[_ScannedDocument],
    plans: Sequence[dict[str, Any]],
    unresolved: Sequence[dict[str, Any]],
    blockers: Sequence[dict[str, Any]],
    operation_state: str,
    operation_ok: bool,
    post_apply_mismatch_count: int | None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for document in scan:
        key = document.document_type
        counts[key] = counts.get(key, 0) + 1
    already_correct = len(scan) - len(plans) - len(unresolved)
    ok = not blockers and not unresolved and not plans
    if mode == "apply" and operation_state == "completed":
        ok = operation_ok and post_apply_mismatch_count == 0
    return {
        "schema_name": CORPUS_SESSION_IDENTITY_SCHEMA,
        "schema_version": CORPUS_SESSION_IDENTITY_VERSION,
        "generated_at": generated_at,
        "mode": mode,
        "operation_id": operation_id,
        "operation_state": operation_state,
        "operation_ok": operation_ok,
        "ok": ok,
        "root": str(root),
        "report": str(report_path),
        "identity_contract": {"m": "March", "s": "June", "w": "November"},
        "scanned_count": len(scan),
        "document_type_counts": dict(sorted(counts.items())),
        "already_correct_count": already_correct,
        "mismatch_count": len(plans),
        "unresolved_count": len(unresolved),
        "blocking_count": len(blockers),
        "post_apply_mismatch_count": post_apply_mismatch_count,
        "entries": list(plans),
        "unresolved": list(unresolved),
        "blockers": list(blockers),
    }


def _rollback_moves(
    root: Path,
    staged: Sequence[tuple[dict[str, Any], Path]],
) -> list[str]:
    errors: list[str] = []
    # First clear every destination back into its unique stage slot. This is
    # necessary for swaps, where one destination is another record's source.
    for entry, stage in reversed(staged):
        destination = _contained_path(root, entry["destination_path"])
        if not stage.exists() and destination.is_file():
            try:
                if _file_sha256(destination) != entry["sha256"]:
                    raise CorpusSessionIdentityError(
                        f"rollback destination hash mismatch: {entry['destination_path']}"
                    )
                os.replace(destination, stage)
            except Exception as exc:
                errors.append(f"{entry['destination_path']}: {type(exc).__name__}: {exc}")
    for entry, stage in staged:
        source = _contained_path(root, entry["source_path"])
        if stage.is_file() and not source.exists():
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage, source)
            except Exception as exc:
                errors.append(f"{entry['source_path']}: {type(exc).__name__}: {exc}")
    return errors


def _verify_source(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file():
        raise CorpusSessionIdentityError(f"source disappeared after preflight: {entry['source_path']}")
    if path.stat().st_size != entry["size_bytes"] or _file_sha256(path) != entry["sha256"]:
        raise CorpusSessionIdentityError(f"source changed after preflight: {entry['source_path']}")


def _contained_path(root: Path, relative: str, *, require_missing: bool = False) -> Path:
    path_value = Path(relative)
    if path_value.is_absolute() or not relative or ".." in path_value.parts:
        raise CorpusSessionIdentityError(f"unsafe corpus path: {relative}")
    current = root
    for part in path_value.parts:
        current = current / part
        if current.is_symlink():
            raise CorpusSessionIdentityError(f"corpus path contains a symbolic link: {relative}")
    resolved = current.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise CorpusSessionIdentityError(f"corpus path escapes root: {relative}")
    if require_missing and current.exists():
        raise CorpusSessionIdentityError(f"destination already exists: {relative}")
    return current


def _renamed_filename(filename: str, session_code: str) -> str:
    match = _PDF_NAME.fullmatch(filename)
    if match is None:
        raise CorpusSessionIdentityError(f"unrecognized corpus filename: {filename}")
    start, end = match.span("session")
    return f"{filename[:start]}{session_code}{filename[end:]}"


def _stage_root(root: Path, operation_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", operation_id):
        raise CorpusSessionIdentityError(f"unsafe operation ID: {operation_id}")
    return root / ".corpus_session_identity_stage" / operation_id


def _operation_id(plans: Iterable[dict[str, Any]]) -> str:
    canonical = [
        {
            "source_path": item["source_path"],
            "destination_path": item["destination_path"],
            "sha256": item["sha256"],
        }
        for item in plans
    ]
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"session-identity-{hashlib.sha256(raw).hexdigest()[:16]}"


def _validate_report_path(root: Path, report_path: Path) -> None:
    report = report_path.resolve()
    if report == root or root in report.parents:
        raise CorpusSessionIdentityError("session-identity report must be outside the corpus root")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _documents_sha256(documents: Iterable[dict[str, Any]]) -> str:
    raw = json.dumps(list(documents), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusSessionIdentityError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusSessionIdentityError(f"{label} root must be an object: {path}")
    return payload


def _fsync_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or transactionally normalize corpus session filenames from publisher text."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--apply", action="store_true", help="Apply the validated rename plan; default is dry-run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = normalize_corpus_session_identity(
        root=args.root,
        report_path=args.report,
        apply=bool(args.apply),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["operation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
