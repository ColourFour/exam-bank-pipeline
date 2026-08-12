from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import fitz

from .atomic_json import write_atomic_json

CORPUS_MANIFEST_SCHEMA = "exam_bank.corpus_manifest"
CORPUS_MANIFEST_VERSION = 1
CORPUS_QUARANTINE_VALIDATION_SCHEMA = "exam_bank.corpus_quarantine_validation"
CORPUS_QUARANTINE_VALIDATION_VERSION = 1
DEFAULT_CORPUS_MANIFEST = Path("manifests/corpora/caie_9709.v1.json")
DEFAULT_CORPUS_ROOT = Path("input")
DEFAULT_CORPUS_QUARANTINE_REPORT = Path("reports/corpus/corpus_quarantine_validation.v1.json")
DEFAULT_SOURCE_BASE_URL = "https://pastpapers.co/caie/a-level/mathematics-9709"

_PDF_NAME = re.compile(
    r"^(?P<syllabus>\d{4})_(?P<session>[msw])(?P<yy>\d{2})_"
    r"(?P<document_type>qp|ms|er|gt)_(?P<component>\d{1,2})\.pdf$",
    re.IGNORECASE,
)
_SESSION_NAMES = {"m": "March", "s": "June", "w": "November"}
_SOURCE_SESSION_SLUGS = {"m": "mar", "s": "jun", "w": "nov"}
_DOCUMENT_TYPES = {
    "qp": "question_paper",
    "ms": "mark_scheme",
    "er": "examiner_report",
    "gt": "grade_threshold",
}


class CorpusManifestError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    document_type: str
    syllabus: str
    year: int
    session: str
    session_code: str
    component: str
    local_path: str
    source_url: str
    mirror_urls: tuple[str, ...]
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "syllabus": self.syllabus,
            "year": self.year,
            "session": self.session,
            "session_code": self.session_code,
            "component": self.component,
            "local_path": self.local_path,
            "source_url": self.source_url,
            "mirror_urls": list(self.mirror_urls),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def build_corpus_manifest(
    root: str | Path = DEFAULT_CORPUS_ROOT,
    *,
    corpus_id: str = "caie-9709-2008-2025",
    generated_at: str | None = None,
    source_base_url: str = DEFAULT_SOURCE_BASE_URL,
) -> dict[str, Any]:
    root = Path(root)
    documents: list[CorpusDocument] = []
    for path in sorted(root.glob("pastpapers/9709/**/*.pdf")):
        structural_error = _pdf_structural_error(path)
        if structural_error:
            raise CorpusManifestError(
                f"Structurally invalid corpus PDF ({structural_error}): {path}"
            )
        match = _PDF_NAME.match(path.name)
        if match is None:
            raise CorpusManifestError(f"Unrecognized corpus PDF filename: {path}")
        session_code = match.group("session").lower()
        year = 2000 + int(match.group("yy"))
        if year != int(path.parts[-3]):
            raise CorpusManifestError(f"Filename/path year mismatch: {path}")
        source_url = (
            f"{source_base_url.rstrip('/')}/{year}/{year}-{_SOURCE_SESSION_SLUGS[session_code]}/{path.name}"
        )
        documents.append(
            CorpusDocument(
                document_id=path.stem.lower(),
                document_type=_DOCUMENT_TYPES[match.group("document_type").lower()],
                syllabus=match.group("syllabus"),
                year=year,
                session=_SESSION_NAMES[session_code],
                session_code=session_code,
                component=match.group("component").zfill(2),
                local_path=path.relative_to(root).as_posix(),
                source_url=source_url,
                mirror_urls=(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    document_payloads = [document.as_dict() for document in documents]
    return {
        "schema_name": CORPUS_MANIFEST_SCHEMA,
        "schema_version": CORPUS_MANIFEST_VERSION,
        "corpus_id": corpus_id,
        "generated_at": generated_at or _utc_now_iso(),
        "record_count": len(document_payloads),
        "documents_sha256": _documents_sha256(document_payloads),
        "documents": document_payloads,
    }


def load_corpus_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_corpus_manifest(manifest)
    if errors:
        raise CorpusManifestError("; ".join(errors))
    return manifest


def validate_corpus_manifest(manifest: Any) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest_not_object"]
    errors: list[str] = []
    if manifest.get("schema_name") != CORPUS_MANIFEST_SCHEMA:
        errors.append("schema_name_mismatch")
    if manifest.get("schema_version") != CORPUS_MANIFEST_VERSION:
        errors.append("schema_version_mismatch")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        return errors + ["documents_not_list"]
    if manifest.get("record_count") != len(documents):
        errors.append("record_count_mismatch")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    required = {
        "document_id",
        "document_type",
        "syllabus",
        "year",
        "session",
        "session_code",
        "component",
        "local_path",
        "source_url",
        "mirror_urls",
        "sha256",
        "size_bytes",
    }
    for index, document in enumerate(documents):
        prefix = f"document:{index}"
        if not isinstance(document, dict):
            errors.append(f"{prefix}:not_object")
            continue
        missing = sorted(required - set(document))
        errors.extend(f"{prefix}:missing:{field}" for field in missing)
        document_id = str(document.get("document_id") or "")
        local_path = str(document.get("local_path") or "")
        if not document_id:
            errors.append(f"{prefix}:empty_document_id")
        elif document_id in seen_ids:
            errors.append(f"{prefix}:duplicate_document_id:{document_id}")
        seen_ids.add(document_id)
        if not _safe_relative_path(local_path):
            errors.append(f"{prefix}:unsafe_local_path:{local_path}")
        elif local_path in seen_paths:
            errors.append(f"{prefix}:duplicate_local_path:{local_path}")
        seen_paths.add(local_path)
        digest = str(document.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{prefix}:invalid_sha256")
        if not isinstance(document.get("size_bytes"), int) or int(document.get("size_bytes") or 0) < 0:
            errors.append(f"{prefix}:invalid_size_bytes")
        mirrors = document.get("mirror_urls")
        if not isinstance(mirrors, list) or not all(isinstance(url, str) for url in mirrors):
            errors.append(f"{prefix}:invalid_mirror_urls")
    if isinstance(manifest.get("documents_sha256"), str):
        if manifest["documents_sha256"] != _documents_sha256(documents):
            errors.append("documents_sha256_mismatch")
    else:
        errors.append("documents_sha256_missing")
    return errors


def verify_corpus(
    manifest_path: str | Path = DEFAULT_CORPUS_MANIFEST,
    *,
    root: str | Path = DEFAULT_CORPUS_ROOT,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(manifest_path)
    root = Path(root)
    missing: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    checksum_mismatches: list[dict[str, Any]] = []
    structural_failures: list[dict[str, str]] = []
    verified: list[str] = []
    for document in manifest["documents"]:
        path = _document_path(root, document["local_path"])
        if not path.is_file():
            missing.append(document["local_path"])
            continue
        actual_size = path.stat().st_size
        if actual_size != document["size_bytes"]:
            size_mismatches.append(
                {"local_path": document["local_path"], "expected": document["size_bytes"], "actual": actual_size}
            )
            continue
        actual_sha256 = sha256_file(path)
        if actual_sha256 != document["sha256"]:
            checksum_mismatches.append(
                {"local_path": document["local_path"], "expected": document["sha256"], "actual": actual_sha256}
            )
            continue
        structural_error = _pdf_structural_error(path)
        if structural_error:
            structural_failures.append(
                {"local_path": document["local_path"], "reason": structural_error}
            )
            continue
        verified.append(document["local_path"])
    return {
        "schema_name": "exam_bank.corpus_verification",
        "schema_version": 1,
        "manifest": str(manifest_path),
        "corpus_id": manifest["corpus_id"],
        "root": str(root),
        "ok": not missing and not size_mismatches and not checksum_mismatches and not structural_failures,
        "record_count": manifest["record_count"],
        "verified_count": len(verified),
        "missing_count": len(missing),
        "size_mismatch_count": len(size_mismatches),
        "checksum_mismatch_count": len(checksum_mismatches),
        "structural_failure_count": len(structural_failures),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "checksum_mismatches": checksum_mismatches,
        "structural_failures": structural_failures,
    }


def _pdf_structural_error(path: Path) -> str | None:
    """Return a stable reason when a PDF has no usable pages or visible content."""

    display_errors = bool(fitz.TOOLS.mupdf_display_errors())
    fitz.TOOLS.mupdf_display_errors(False)
    try:
        with fitz.open(path) as document:
            if document.page_count < 1:
                return "pdf_has_no_pages"
            for page in document:
                if page.get_text("text").strip():
                    return None
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(0.2, 0.2),
                    colorspace=fitz.csGRAY,
                    alpha=False,
                )
                if pixmap.samples and min(pixmap.samples) < 250:
                    return None
    except Exception as exc:
        return f"pdf_unreadable:{exc.__class__.__name__}"
    finally:
        fitz.TOOLS.mupdf_display_errors(display_errors)
    return "pdf_has_no_renderable_content"


def hydrate_corpus(
    manifest_path: str | Path = DEFAULT_CORPUS_MANIFEST,
    *,
    root: str | Path = DEFAULT_CORPUS_ROOT,
    repair: bool = False,
    offline: bool = False,
    timeout: float = 60.0,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(manifest_path)
    root = Path(root)
    hydrated: list[str] = []
    already_verified: list[str] = []
    failed: list[dict[str, Any]] = []
    quarantined: list[str] = []
    for document in manifest["documents"]:
        destination = _document_path(root, document["local_path"])
        state = _file_state(destination, document)
        if state == "verified":
            already_verified.append(document["local_path"])
            continue
        if state != "missing" and not repair:
            failed.append({"local_path": document["local_path"], "reason": state, "repair_required": True})
            continue
        if offline:
            failed.append({"local_path": document["local_path"], "reason": f"offline_{state}"})
            continue
        urls = _download_urls(document)
        if not urls:
            failed.append({"local_path": document["local_path"], "reason": "no_download_url"})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f".{destination.name}.partial")
        error_messages: list[str] = []
        downloaded = False
        for url in urls:
            try:
                _download(url, temp_path, timeout=timeout)
                if _file_state(temp_path, document) != "verified":
                    raise CorpusManifestError("downloaded payload failed checksum or size verification")
                downloaded = True
                break
            except Exception as exc:  # network/provider errors are reported per document
                error_messages.append(f"{url}: {type(exc).__name__}: {exc}")
                temp_path.unlink(missing_ok=True)
        if not downloaded:
            failed.append(
                {"local_path": document["local_path"], "reason": "download_failed", "errors": error_messages}
            )
            continue
        if destination.exists():
            quarantine_path = _quarantine_path(root, document["local_path"])
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, quarantine_path)
            quarantined.append(str(quarantine_path))
        os.replace(temp_path, destination)
        hydrated.append(document["local_path"])
    verification = verify_corpus(manifest_path, root=root)
    return {
        "schema_name": "exam_bank.corpus_hydration",
        "schema_version": 1,
        "manifest": str(manifest_path),
        "root": str(root),
        "ok": not failed and verification["ok"],
        "hydrated_count": len(hydrated),
        "already_verified_count": len(already_verified),
        "failed_count": len(failed),
        "quarantined_count": len(quarantined),
        "hydrated": hydrated,
        "failed": failed,
        "quarantined": quarantined,
        "verification": verification,
    }


def write_corpus_manifest(
    path: str | Path,
    *,
    root: str | Path = DEFAULT_CORPUS_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    manifest = build_corpus_manifest(root, generated_at=generated_at)
    write_atomic_json(manifest, path, sort_keys=True)
    return manifest


def quarantine_structural_failures(
    manifest_path: str | Path = DEFAULT_CORPUS_MANIFEST,
    *,
    root: str | Path = DEFAULT_CORPUS_ROOT,
    report_path: str | Path = DEFAULT_CORPUS_QUARANTINE_REPORT,
    apply: bool = False,
    active_manifest_path: str | Path | None = None,
    quarantine_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Plan or apply recoverable quarantine of checksum-valid, structurally invalid PDFs.

    The authoritative manifest is never changed. Applying the plan requires a separate
    active-manifest destination; that derived manifest explicitly records every omitted
    document and remains partial until a valid replacement satisfies the source contract.
    """

    manifest_path = Path(manifest_path)
    report_path = Path(report_path)
    manifest = load_corpus_manifest(manifest_path)
    root = Path(root).resolve()
    operation_id = quarantine_id or f"structural-{manifest['documents_sha256'][:16]}"
    _validate_quarantine_id(operation_id)
    active_path = Path(active_manifest_path) if active_manifest_path is not None else None
    if apply and active_path is None:
        raise CorpusManifestError("--apply requires a separate --active-manifest path")
    _validate_quarantine_output_paths(
        root=root,
        source_manifest=manifest_path,
        report_path=report_path,
        active_manifest_path=active_path,
    )

    generated = generated_at or _utc_now_iso()
    candidates: list[dict[str, Any]] = []
    already_quarantined: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    verified_count = 0
    for document in manifest["documents"]:
        local_path = document["local_path"]
        original = _contained_path_without_symlinks(root, local_path)
        quarantine_relative = (Path("quarantine") / operation_id / local_path).as_posix()
        quarantine = _contained_path_without_symlinks(root, quarantine_relative)
        original_state = _file_state(original, document)
        if original_state == "verified":
            structural_reason = _pdf_structural_error(original)
            if structural_reason is None:
                verified_count += 1
                continue
            entry = _quarantine_entry(
                document,
                structural_reason=structural_reason,
                status="planned",
                root=root,
                original=original,
                quarantine=quarantine,
            )
            if quarantine.exists():
                entry["status"] = "destination_conflict"
                entry["blocking_reason"] = "quarantine_destination_already_exists"
                blockers.append(entry)
            else:
                candidates.append(entry)
            continue

        if original_state == "missing" and quarantine.is_file():
            quarantine_state = _file_state(quarantine, document)
            structural_reason = (
                _pdf_structural_error(quarantine) if quarantine_state == "verified" else None
            )
            if quarantine_state == "verified" and structural_reason is not None:
                already_quarantined.append(
                    _quarantine_entry(
                        document,
                        structural_reason=structural_reason,
                        status="already_quarantined",
                        root=root,
                        original=original,
                        quarantine=quarantine,
                    )
                )
                continue
            blockers.append(
                _quarantine_entry(
                    document,
                    structural_reason=structural_reason,
                    status="destination_conflict",
                    root=root,
                    original=original,
                    quarantine=quarantine,
                    blocking_reason=f"quarantine_{quarantine_state}",
                )
            )
            continue

        blockers.append(
            _quarantine_entry(
                document,
                structural_reason=None,
                status="not_eligible",
                root=root,
                original=original,
                quarantine=quarantine,
                blocking_reason=original_state,
            )
        )

    excluded_entries = [*already_quarantined, *candidates]
    active_manifest = _build_active_corpus_manifest(
        manifest,
        excluded_entries=excluded_entries,
        source_manifest_path=manifest_path,
        quarantine_report_path=report_path,
        generated_at=generated,
    )
    report = _quarantine_report(
        manifest=manifest,
        manifest_path=manifest_path,
        root=root,
        report_path=report_path,
        active_manifest_path=active_path,
        active_manifest=active_manifest,
        operation_id=operation_id,
        mode="apply" if apply else "dry_run",
        generated_at=generated,
        verified_count=verified_count,
        candidates=candidates,
        already_quarantined=already_quarantined,
        blockers=blockers,
        active_manifest_written=False,
        operation_ok=not blockers,
    )

    if not apply or blockers:
        write_atomic_json(report, report_path, sort_keys=True)
        return report

    # Persist the fail-closed plan before the first move so an interrupted run
    # still leaves the exact recovery locations on disk.
    in_progress = dict(report)
    in_progress["operation_state"] = "apply_in_progress"
    write_atomic_json(in_progress, report_path, sort_keys=True)
    try:
        for entry in candidates:
            original = Path(entry["original_absolute_path"])
            quarantine = Path(entry["quarantine_absolute_path"])
            document = entry["manifest_entry"]
            # Revalidate immediately before mutation. A changed source is never moved.
            if _file_state(original, document) != "verified":
                raise CorpusManifestError(
                    f"Corpus document changed after quarantine preflight: {entry['original_path']}"
                )
            structural_reason = _pdf_structural_error(original)
            if structural_reason is None or structural_reason != entry["structural_reason"]:
                raise CorpusManifestError(
                    f"Corpus document structure changed after quarantine preflight: {entry['original_path']}"
                )
            quarantine = _contained_path_without_symlinks(
                root, entry["quarantine_path"], require_missing=True
            )
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            quarantine = _contained_path_without_symlinks(
                root, entry["quarantine_path"], require_missing=True
            )
            os.replace(original, quarantine)
            entry["status"] = "quarantined"

        # Rebuild after the moves so only recoverably quarantined documents are omitted.
        excluded_entries = [*already_quarantined, *candidates]
        active_manifest = _build_active_corpus_manifest(
            manifest,
            excluded_entries=excluded_entries,
            source_manifest_path=manifest_path,
            quarantine_report_path=report_path,
            generated_at=generated,
        )
        assert active_path is not None
        write_atomic_json(active_manifest, active_path, sort_keys=True)
    except BaseException as exc:
        failed_report = _quarantine_report(
            manifest=manifest,
            manifest_path=manifest_path,
            root=root,
            report_path=report_path,
            active_manifest_path=active_path,
            active_manifest=active_manifest,
            operation_id=operation_id,
            mode="apply",
            generated_at=generated,
            verified_count=verified_count,
            candidates=candidates,
            already_quarantined=already_quarantined,
            blockers=blockers,
            active_manifest_written=False,
            operation_ok=False,
            operation_error=f"{type(exc).__name__}: {exc}",
        )
        failed_report["operation_state"] = "apply_interrupted"
        write_atomic_json(failed_report, report_path, sort_keys=True)
        raise

    report = _quarantine_report(
        manifest=manifest,
        manifest_path=manifest_path,
        root=root,
        report_path=report_path,
        active_manifest_path=active_path,
        active_manifest=active_manifest,
        operation_id=operation_id,
        mode="apply",
        generated_at=generated,
        verified_count=verified_count,
        candidates=candidates,
        already_quarantined=already_quarantined,
        blockers=blockers,
        active_manifest_written=True,
        operation_ok=True,
    )
    write_atomic_json(report, report_path, sort_keys=True)
    return report


def _quarantine_entry(
    document: dict[str, Any],
    *,
    structural_reason: str | None,
    status: str,
    root: Path,
    original: Path,
    quarantine: Path,
    blocking_reason: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "document_id": document["document_id"],
        "status": status,
        "structural_reason": structural_reason,
        "source_url": document["source_url"],
        "source_sha256": document["sha256"],
        "source_size_bytes": document["size_bytes"],
        "original_path": original.relative_to(root).as_posix(),
        "original_absolute_path": str(original),
        "quarantine_path": quarantine.relative_to(root).as_posix(),
        "quarantine_absolute_path": str(quarantine),
        "manifest_entry": dict(document),
    }
    if blocking_reason is not None:
        entry["blocking_reason"] = blocking_reason
    return entry


def _build_active_corpus_manifest(
    source_manifest: dict[str, Any],
    *,
    excluded_entries: Iterable[dict[str, Any]],
    source_manifest_path: Path,
    quarantine_report_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    excluded_entries = list(excluded_entries)
    excluded_paths = {entry["original_path"] for entry in excluded_entries}
    documents = [
        dict(document)
        for document in source_manifest["documents"]
        if document["local_path"] not in excluded_paths
    ]
    partial = bool(excluded_entries)
    active_documents_sha256 = _documents_sha256(documents)
    corpus_id = source_manifest["corpus_id"]
    if partial:
        corpus_id = f"{corpus_id}-partial-quarantine-{active_documents_sha256[:12]}"
    excluded = [
        {
            "document_id": entry["document_id"],
            "local_path": entry["original_path"],
            "quarantine_path": entry["quarantine_path"],
            "structural_reason": entry["structural_reason"],
            "sha256": entry["source_sha256"],
            "size_bytes": entry["source_size_bytes"],
        }
        for entry in excluded_entries
    ]
    return {
        "schema_name": CORPUS_MANIFEST_SCHEMA,
        "schema_version": CORPUS_MANIFEST_VERSION,
        "corpus_id": corpus_id,
        "generated_at": generated_at,
        "record_count": len(documents),
        "documents_sha256": active_documents_sha256,
        "documents": documents,
        "corpus_state": "partial_quarantined" if partial else "complete",
        "derivation": {
            "source_corpus_id": source_manifest["corpus_id"],
            "source_manifest": str(source_manifest_path),
            "source_documents_sha256": source_manifest["documents_sha256"],
            "quarantine_validation": str(quarantine_report_path),
            "excluded_document_count": len(excluded),
            "excluded_documents": excluded,
        },
    }


def _quarantine_report(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    root: Path,
    report_path: Path,
    active_manifest_path: Path | None,
    active_manifest: dict[str, Any],
    operation_id: str,
    mode: str,
    generated_at: str,
    verified_count: int,
    candidates: list[dict[str, Any]],
    already_quarantined: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    active_manifest_written: bool,
    operation_ok: bool,
    operation_error: str | None = None,
) -> dict[str, Any]:
    unresolved = [*already_quarantined, *candidates, *blockers]
    report: dict[str, Any] = {
        "schema_name": CORPUS_QUARANTINE_VALIDATION_SCHEMA,
        "schema_version": CORPUS_QUARANTINE_VALIDATION_VERSION,
        "generated_at": generated_at,
        "operation_id": operation_id,
        "mode": mode,
        "operation_state": "completed" if mode == "apply" and operation_ok else "planned",
        "operation_ok": operation_ok,
        # This is deliberately false for a successful quarantine: the authoritative
        # corpus remains incomplete until valid replacements satisfy its checksums.
        "ok": not unresolved,
        "authoritative_manifest": str(manifest_path),
        "authoritative_manifest_sha256": sha256_file(manifest_path),
        "source_corpus_id": manifest["corpus_id"],
        "source_record_count": manifest["record_count"],
        "root": str(root),
        "report": str(report_path),
        "verified_count": verified_count,
        "planned_count": sum(entry["status"] == "planned" for entry in candidates),
        "quarantined_count": sum(entry["status"] == "quarantined" for entry in candidates),
        "already_quarantined_count": len(already_quarantined),
        "blocking_count": len(blockers),
        "unresolved_count": len(unresolved),
        "entries": [*already_quarantined, *candidates],
        "blockers": blockers,
        "active_manifest": str(active_manifest_path) if active_manifest_path is not None else None,
        "active_manifest_written": active_manifest_written,
        "active_corpus_id": active_manifest["corpus_id"],
        "active_record_count": active_manifest["record_count"],
        "active_documents_sha256": active_manifest["documents_sha256"],
        "excluded_document_count": active_manifest["derivation"]["excluded_document_count"],
    }
    if operation_error is not None:
        report["operation_error"] = operation_error
    return report


def _validate_quarantine_id(value: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", value) or value in {".", ".."}:
        raise CorpusManifestError(
            "Unsafe quarantine ID; use 1-128 lowercase letters, digits, dots, underscores, or hyphens"
        )


def _validate_quarantine_output_paths(
    *,
    root: Path,
    source_manifest: Path,
    report_path: Path,
    active_manifest_path: Path | None,
) -> None:
    source = source_manifest.resolve()
    report = report_path.resolve()
    active = active_manifest_path.resolve() if active_manifest_path is not None else None
    if report == source:
        raise CorpusManifestError("Quarantine report cannot overwrite the authoritative manifest")
    if active is not None and active in {source, report}:
        raise CorpusManifestError("Active manifest must be separate from the source manifest and report")
    for label, path in (("report", report), ("active manifest", active)):
        if path is not None and (path == root or root in path.parents):
            raise CorpusManifestError(f"Quarantine {label} must be outside the corpus root")


def _contained_path_without_symlinks(
    root: Path,
    relative: str,
    *,
    require_missing: bool = False,
) -> Path:
    if not _safe_relative_path(relative):
        raise CorpusManifestError(f"Unsafe corpus path: {relative}")
    root = root.resolve()
    path = root.joinpath(*Path(relative).parts)
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise CorpusManifestError(f"Corpus path contains a symbolic link: {relative}")
    resolved = path.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise CorpusManifestError(f"Corpus path escapes root: {relative}")
    if require_missing and path.exists():
        raise CorpusManifestError(f"Quarantine destination already exists: {relative}")
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _documents_sha256(documents: Iterable[dict[str, Any]]) -> str:
    raw = json.dumps(list(documents), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _document_path(root: Path, relative: str) -> Path:
    if not _safe_relative_path(relative):
        raise CorpusManifestError(f"Unsafe corpus path: {relative}")
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise CorpusManifestError(f"Corpus path escapes root: {relative}")
    return path


def _file_state(path: Path, document: dict[str, Any]) -> str:
    if not path.is_file():
        return "missing"
    if path.stat().st_size != document["size_bytes"]:
        return "size_mismatch"
    if sha256_file(path) != document["sha256"]:
        return "checksum_mismatch"
    return "verified"


def _download_urls(document: dict[str, Any]) -> list[str]:
    values = [str(document.get("source_url") or ""), *[str(url) for url in document.get("mirror_urls") or []]]
    return list(dict.fromkeys(url for url in values if url))


def _download(url: str, destination: Path, *, timeout: float) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise CorpusManifestError(f"Unsupported corpus URL scheme: {parsed.scheme}")
    request = Request(url, headers={"User-Agent": "exam-bank-pipeline/1"})
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:  # nosec B310
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _quarantine_path(root: Path, relative: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "_quarantine_corpus" / timestamp / relative


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
