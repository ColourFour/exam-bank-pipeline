from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import fcntl

from . import __version__
from .atomic_json import write_atomic_json
from .classification import (
    _explicit_primary_topic_from_text,
    classify_question,
    classify_question_parts,
    infer_source_paper_code,
)
from .config import AppConfig
from .core.paper_identity import IdentityError, PaperIdentity, paper_identity_from_parts, session_for_source_path
from .document_metadata import (
    DocumentMetadata,
    parse_filename_metadata,
    parse_internal_document_metadata,
    reconcile_document_metadata,
)
from .document_registry import DocumentRegistry, build_document_registry, build_document_registry_from_paths
from .examiner_reports import examiner_report_topic_evidence
from .exporters import export_records, records_to_output_questions, write_json as write_question_bank_records
from .extraction_structure import build_structured_question_text
from .image_rendering import render_question_image
from .mark_schemes import (
    MarkSchemeImageResult,
    analyze_mark_scheme,
    extract_mark_scheme_answers,
    find_mark_scheme,
    render_mark_scheme_images,
)
from .models import ClassificationResult, PageLayout, QuestionRecord, QuestionSpan
from .ocr import select_text_candidate
from .output_layout import paper_family_dir_name, paper_instance_id
from .pdf_extract import extract_pdf_layout
from .publication_safety import (
    PIPELINE_LOCK_FILENAME,
    publication_committed_prefix,
    publication_journal_prefix,
)
from .question_detection import (
    detect_question_anchor_candidates,
    detect_question_spans,
    extract_question_total_from_text,
)
from .trust import (
    CropConfidence,
    MappingStatus,
    PaperTotalStatus,
    RescanResult,
    ValidationStatus,
)
from .trust import (
    assess_text_fidelity as _assess_text_fidelity,
)
from .trust import (
    derive_question_text_semantics as _derive_question_text_semantics,
)
from .trust import (
    derive_scope_quality_status as _derive_scope_quality_status,
)
from .trust import (
    derive_text_only_status as _derive_text_only_status,
)
from .trust import (
    derive_topic_trust_status as _derive_topic_trust_status,
)
from .trust import (
    derive_visual_curation_status as _derive_visual_curation_status,
)
from .trust import (
    references_source_visual as _references_source_visual,
)
from .trust import (
    refine_validation_status as _refine_validation_status,
)
from .trust import (
    text_source_profile as _text_source_profile,
)
from .trust import (
    visual_reason_flags as _visual_reason_flags,
)


@dataclass(frozen=True)
class PipelineResult:
    records: list[QuestionRecord]
    json_path: Path
    output_root: Path
    question_count: int | None = None
    paper_count: int | None = None


class EmptyQuestionPaperInputError(ValueError):
    pass


class EmptyPaperExtractionError(RuntimeError):
    pass


class PipelineOutputLockedError(RuntimeError):
    pass


class PipelinePublicationRecoveryError(RuntimeError):
    pass


EXTRACTION_CACHE_SCHEMA_VERSION = 3
_PUBLICATION_JOURNAL_SCHEMA_VERSION = 1
_PUBLICATION_JOURNAL_FILENAME = "publication_journal.json"
_PUBLICATION_BACKUP_DIRNAME = "backups"


def process_inputs(
    input_path: str | Path,
    config: AppConfig,
    *,
    progress: Any | None = None,
    resume_completed_batch_ids: set[str] | None = None,
    force_rerun: bool = False,
    workers: int = 1,
    allow_empty: bool = False,
) -> PipelineResult:
    config.ensure_output_dirs()
    if progress:
        progress.update_phase("scanning_inputs", force_render=True)
    registry = build_document_registry(input_path, allowed_document_types=set(config.runtime.input_document_types))
    return _process_registry_entries_transactionally(
        registry,
        config,
        progress=progress,
        resume_completed_batch_ids=resume_completed_batch_ids,
        force_rerun=force_rerun,
        workers=workers,
        allow_empty=allow_empty,
    )


def process_batch(config: AppConfig, *, workers: int = 1, allow_empty: bool = False) -> PipelineResult:
    config.ensure_output_dirs()
    active_document_types = set(config.runtime.input_document_types)
    source_paths: list[Path] = []
    if config.runtime.supports_input_document_type("question_paper"):
        source_paths.append(config.input.question_papers_dir)
    if config.runtime.supports_input_document_type("mark_scheme"):
        source_paths.append(config.input.mark_schemes_dir)
    registry = build_document_registry_from_paths(
        source_paths,
        allowed_document_types=active_document_types,
    )
    return _process_registry_entries_transactionally(
        registry,
        config,
        workers=workers,
        allow_empty=allow_empty,
    )


def process_folder(
    folder: str | Path,
    config: AppConfig,
    *,
    workers: int = 1,
    allow_empty: bool = False,
) -> PipelineResult:
    config.ensure_output_dirs()
    registry = build_document_registry(folder, allowed_document_types=set(config.runtime.input_document_types))
    return _process_registry_entries_transactionally(
        registry,
        config,
        workers=workers,
        allow_empty=allow_empty,
    )


def _process_registry_entries_transactionally(
    registry: DocumentRegistry,
    config: AppConfig,
    *,
    progress: Any | None = None,
    resume_completed_batch_ids: set[str] | None = None,
    force_rerun: bool = False,
    workers: int = 1,
    allow_empty: bool = False,
) -> PipelineResult:
    output_root = config.output.root_dir()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with _output_root_lock(output_root):
        stage_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.run-",
                dir=output_root.parent,
            )
        )
        stage_config = deepcopy(config)
        stage_config.output.apply_root(stage_root)
        stage_config.ensure_output_dirs()
        try:
            result = _process_registry_entries(
                registry,
                stage_config,
                progress=progress,
                resume_completed_batch_ids=resume_completed_batch_ids,
                force_rerun=force_rerun,
                workers=workers,
                allow_empty=allow_empty,
                publication_config=config,
            )
            staged_json_relative = result.json_path.relative_to(stage_root)
            _promote_run_artifacts_transactionally(
                stage_root,
                output_root,
                final_json_relative=staged_json_relative,
            )
            return PipelineResult(
                records=result.records,
                json_path=output_root / staged_json_relative,
                output_root=output_root,
                question_count=result.question_count,
                paper_count=result.paper_count,
            )
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)


@contextmanager
def _output_root_lock(output_root: Path) -> Iterator[None]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / PIPELINE_LOCK_FILENAME
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise PipelineOutputLockedError(
                f"Another pipeline run is already publishing to {output_root}."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "output_root": str(output_root.resolve())}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        _recover_interrupted_publications(output_root)
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _promote_run_artifacts_transactionally(
    stage_root: Path,
    output_root: Path,
    *,
    final_json_relative: Path,
) -> None:
    _validate_publication_relative_path(final_json_relative)
    if final_json_relative.name == PIPELINE_LOCK_FILENAME:
        raise ValueError("The final JSON path cannot be a pipeline lock file")
    staged_final_json = stage_root / final_json_relative
    if not staged_final_json.is_file() or staged_final_json.is_symlink():
        raise FileNotFoundError(f"Staged final JSON is missing: {staged_final_json}")

    files = [
        path
        for path in stage_root.rglob("*")
        if path.is_file() and path.name != PIPELINE_LOCK_FILENAME
    ]
    symlinks = [path for path in files if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Staged publication contains a symbolic link: {symlinks[0]}")
    files.sort(
        key=lambda path: (
            path.relative_to(stage_root) == final_json_relative,
            str(path.relative_to(stage_root)),
        )
    )
    if not files or files[-1] != staged_final_json:
        raise FileNotFoundError(f"Staged final JSON is not publishable: {staged_final_json}")

    journal_root, entries = _create_publication_journal(
        files,
        stage_root=stage_root,
        output_root=output_root,
        final_json_relative=final_json_relative,
    )
    try:
        touched_directories: set[Path] = set()
        for source, entry in zip(files, entries, strict=True):
            touched_directories.update(
                _promote_publication_file(
                    source,
                    output_root=output_root,
                    journal_root=journal_root,
                    entry=entry,
                )
            )
        _fsync_publication_directories(touched_directories)
        committed_root = _mark_publication_committed(journal_root, output_root=output_root)
    except BaseException as publication_error:
        try:
            _rollback_publication_journal(
                journal_root,
                output_root=output_root,
                entries=entries,
            )
        except BaseException as recovery_error:
            raise PipelinePublicationRecoveryError(
                "Publication failed and automatic rollback was interrupted; "
                f"recovery journal retained at {journal_root}"
            ) from recovery_error
        _cleanup_publication_journal(journal_root)
        raise publication_error
    else:
        _cleanup_publication_journal(committed_root)


def _create_publication_journal(
    files: list[Path],
    *,
    stage_root: Path,
    output_root: Path,
    final_json_relative: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    for source in files:
        relative = source.relative_to(stage_root)
        destination = _publication_destination(output_root, relative)
        preexisting = _path_exists(destination)
        if preexisting and destination.is_dir() and not destination.is_symlink():
            raise IsADirectoryError(f"Publication destination is a directory: {destination}")
        entries.append({"path": relative.as_posix(), "preexisting": preexisting})

    journal_root = Path(
        tempfile.mkdtemp(prefix=_publication_journal_prefix(output_root), dir=output_root.parent)
    )
    payload = {
        "schema_name": "exam_bank.output_publication_journal",
        "schema_version": _PUBLICATION_JOURNAL_SCHEMA_VERSION,
        "output_root": str(output_root.resolve()),
        "final_json_relative": final_json_relative.as_posix(),
        "entries": entries,
    }
    try:
        write_atomic_json(payload, journal_root / _PUBLICATION_JOURNAL_FILENAME, sort_keys=True)
        _fsync_publication_directory(journal_root)
        _fsync_publication_directory(output_root.parent)
    except BaseException:
        shutil.rmtree(journal_root, ignore_errors=True)
        raise
    return journal_root, entries


def _promote_publication_file(
    source: Path,
    *,
    output_root: Path,
    journal_root: Path,
    entry: dict[str, Any],
) -> set[Path]:
    relative = _journal_entry_relative_path(entry)
    destination = _publication_destination(output_root, relative)
    if source.name == PIPELINE_LOCK_FILENAME or destination.name == PIPELINE_LOCK_FILENAME:
        raise ValueError("Pipeline lock files cannot be promoted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".jsonl" and destination.is_file():
        _merge_jsonl_into_source(destination, source)
    if bool(entry["preexisting"]):
        backup = _publication_backup_path(journal_root, relative)
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.replace(destination, backup)
        touched_directories = _publication_directory_chain(destination.parent, output_root)
        touched_directories.update(_publication_directory_chain(backup.parent, journal_root))
    else:
        touched_directories = _publication_directory_chain(destination.parent, output_root)
    os.replace(source, destination)
    return touched_directories


def _mark_publication_committed(journal_root: Path, *, output_root: Path) -> Path:
    active_prefix = _publication_journal_prefix(output_root)
    if not journal_root.name.startswith(active_prefix):
        raise PipelinePublicationRecoveryError(f"Unexpected publication journal name: {journal_root}")
    suffix = journal_root.name[len(active_prefix) :]
    committed_root = journal_root.with_name(f"{_publication_committed_prefix(output_root)}{suffix}")
    os.replace(journal_root, committed_root)
    _fsync_publication_directory(output_root.parent)
    return committed_root


def _recover_interrupted_publications(output_root: Path) -> None:
    parent = output_root.parent
    if not parent.exists():
        return
    active_prefix = _publication_journal_prefix(output_root)
    committed_prefix = _publication_committed_prefix(output_root)
    siblings = sorted(parent.iterdir(), key=lambda path: path.name)
    for committed_root in (
        path for path in siblings if path.name.startswith(committed_prefix)
    ):
        if committed_root.is_symlink() or not committed_root.is_dir():
            raise PipelinePublicationRecoveryError(
                f"Unsafe committed publication marker: {committed_root}"
            )
        manifest_path = committed_root / _PUBLICATION_JOURNAL_FILENAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PipelinePublicationRecoveryError(
                f"Committed publication marker has no valid manifest: {committed_root}"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelinePublicationRecoveryError(
                f"Committed publication manifest is unreadable: {manifest_path}"
            ) from exc
        _validate_publication_journal(payload, output_root=output_root)
        _cleanup_publication_journal(committed_root)

    for journal_root in (
        path for path in siblings if path.name.startswith(active_prefix)
    ):
        if journal_root.is_symlink() or not journal_root.is_dir():
            raise PipelinePublicationRecoveryError(f"Unsafe publication recovery journal: {journal_root}")
        manifest_path = journal_root / _PUBLICATION_JOURNAL_FILENAME
        if not manifest_path.is_file():
            if any(journal_root.iterdir()):
                raise PipelinePublicationRecoveryError(
                    f"Publication recovery journal has no valid manifest: {journal_root}"
                )
            _cleanup_publication_journal(journal_root)
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelinePublicationRecoveryError(
                f"Publication recovery journal is unreadable: {manifest_path}"
            ) from exc
        entries = _validate_publication_journal(payload, output_root=output_root)
        _rollback_publication_journal(
            journal_root,
            output_root=output_root,
            entries=entries,
        )
        _cleanup_publication_journal(journal_root)


def _validate_publication_journal(
    payload: Any,
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_name") != "exam_bank.output_publication_journal"
        or payload.get("schema_version") != _PUBLICATION_JOURNAL_SCHEMA_VERSION
    ):
        raise PipelinePublicationRecoveryError("Unsupported publication recovery journal schema")
    try:
        journal_output_root = Path(str(payload.get("output_root") or "")).resolve()
    except (OSError, RuntimeError) as exc:
        raise PipelinePublicationRecoveryError("Invalid output root in publication recovery journal") from exc
    if journal_output_root != output_root.resolve():
        raise PipelinePublicationRecoveryError(
            "Publication recovery journal targets a different output root"
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise PipelinePublicationRecoveryError("Publication recovery journal has no entries")
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or not isinstance(raw_entry.get("preexisting"), bool):
            raise PipelinePublicationRecoveryError("Invalid publication recovery journal entry")
        relative = _journal_entry_relative_path(raw_entry)
        if relative in seen:
            raise PipelinePublicationRecoveryError(
                f"Duplicate path in publication recovery journal: {relative}"
            )
        seen.add(relative)
        _publication_destination(output_root, relative)
        entries.append({"path": relative.as_posix(), "preexisting": raw_entry["preexisting"]})
    final_json_relative = Path(str(payload.get("final_json_relative") or ""))
    _validate_publication_relative_path(final_json_relative)
    if final_json_relative not in seen or Path(entries[-1]["path"]) != final_json_relative:
        raise PipelinePublicationRecoveryError(
            "Publication recovery journal does not end with the final JSON barrier"
        )
    return entries


def _rollback_publication_journal(
    journal_root: Path,
    *,
    output_root: Path,
    entries: list[dict[str, Any]],
) -> None:
    for entry in reversed(entries):
        relative = _journal_entry_relative_path(entry)
        destination = _publication_destination(output_root, relative)
        backup = _publication_backup_path(journal_root, relative)
        if bool(entry["preexisting"]):
            if _path_exists(backup):
                if _path_exists(destination):
                    if destination.is_dir() and not destination.is_symlink():
                        raise PipelinePublicationRecoveryError(
                            f"Cannot replace directory while recovering publication: {destination}"
                        )
                    destination.unlink()
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
                _fsync_publication_directory(destination.parent)
                _fsync_publication_directory(backup.parent)
            elif not _path_exists(destination):
                raise PipelinePublicationRecoveryError(
                    f"Both live file and rollback backup are missing: {destination}"
                )
        elif _path_exists(destination):
            if destination.is_dir() and not destination.is_symlink():
                raise PipelinePublicationRecoveryError(
                    f"Cannot remove directory while recovering publication: {destination}"
                )
            destination.unlink()
            _fsync_publication_directory(destination.parent)


def _publication_destination(output_root: Path, relative: Path) -> Path:
    _validate_publication_relative_path(relative)
    if relative.name == PIPELINE_LOCK_FILENAME:
        raise PipelinePublicationRecoveryError("Publication journal cannot target a pipeline lock")
    destination = output_root / relative
    resolved_root = output_root.resolve()
    resolved_parent = destination.parent.resolve()
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise PipelinePublicationRecoveryError(
            f"Publication destination escapes the output root: {relative}"
        )
    return destination


def _publication_backup_path(journal_root: Path, relative: Path) -> Path:
    _validate_publication_relative_path(relative)
    backup_root = journal_root / _PUBLICATION_BACKUP_DIRNAME
    backup = backup_root / relative
    resolved_journal_root = journal_root.resolve()
    resolved_backup_root = backup_root.resolve()
    resolved_parent = backup.parent.resolve()
    if resolved_backup_root.parent != resolved_journal_root:
        raise PipelinePublicationRecoveryError(
            f"Publication backup root escapes the journal: {backup_root}"
        )
    if resolved_parent != resolved_backup_root and resolved_backup_root not in resolved_parent.parents:
        raise PipelinePublicationRecoveryError(
            f"Publication backup path escapes the journal: {relative}"
        )
    return backup


def _journal_entry_relative_path(entry: dict[str, Any]) -> Path:
    relative = Path(str(entry.get("path") or ""))
    _validate_publication_relative_path(relative)
    return relative


def _validate_publication_relative_path(relative: Path) -> None:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Publication path must be a safe relative path: {relative}")


def _publication_journal_prefix(output_root: Path) -> str:
    return publication_journal_prefix(output_root)


def _publication_committed_prefix(output_root: Path) -> str:
    return publication_committed_prefix(output_root)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _cleanup_publication_journal(journal_root: Path) -> None:
    if journal_root.is_symlink():
        raise PipelinePublicationRecoveryError(f"Refusing to remove symlinked journal: {journal_root}")
    shutil.rmtree(journal_root, ignore_errors=True)
    _fsync_publication_directory(journal_root.parent)


def _fsync_publication_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_publication_directories(paths: set[Path]) -> None:
    for path in sorted(paths, key=lambda item: (-len(item.parts), str(item))):
        _fsync_publication_directory(path)


def _publication_directory_chain(directory: Path, boundary: Path) -> set[Path]:
    if directory != boundary and boundary not in directory.parents:
        raise PipelinePublicationRecoveryError(
            f"Publication directory escapes its durability boundary: {directory}"
        )
    paths = {directory}
    current = directory
    while current != boundary:
        current = current.parent
        paths.add(current)
    return paths


def _merge_jsonl_into_source(existing: Path, staged: Path) -> None:
    lines = {
        line
        for path in (existing, staged)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    staged.write_text("".join(f"{line}\n" for line in sorted(lines)), encoding="utf-8")


def _process_registry_entries(
    registry: DocumentRegistry,
    config: AppConfig,
    *,
    progress: Any | None = None,
    resume_completed_batch_ids: set[str] | None = None,
    force_rerun: bool = False,
    workers: int = 1,
    allow_empty: bool = False,
    publication_config: AppConfig | None = None,
) -> PipelineResult:
    publication_config = publication_config or config
    publication_root = publication_config.output.root_dir()
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers > 1 and config.debug.enabled:
        raise ValueError("workers > 1 is incompatible with debug mode until debug streams are isolated")
    entries = registry.question_paper_entries()
    if not entries and not allow_empty:
        unclassified = ", ".join(str(path) for path in registry.unclassified[:10]) or "none"
        raise EmptyQuestionPaperInputError(
            "No classified question-paper PDFs were found; refusing to replace the question bank with an empty output. "
            f"Unclassified PDFs ({len(registry.unclassified)}): {unclassified}. "
            "Pass allow_empty=True (CLI: --allow-empty) only when an empty bank is intentional."
        )
    if workers > 1:
        return _process_registry_entries_parallel(
            registry,
            config,
            progress=progress,
            resume_completed_batch_ids=resume_completed_batch_ids,
            force_rerun=force_rerun,
            workers=workers,
            publication_config=publication_config,
        )

    records: list[QuestionRecord] = []
    resume_completed_batch_ids = resume_completed_batch_ids or set()
    if progress:
        progress.set_totals(total_batches=len(entries))
    for entry in entries:
        assert entry.question_paper is not None
        progress_context = _entry_progress_context(entry)
        batch_id = progress_context["batch_id"]
        paper = progress_context["paper"]
        paper_family = progress_context["paper_family"]
        cache_key = _extraction_batch_cache_key(entry, publication_config)
        if progress and not force_rerun and batch_id in resume_completed_batch_ids:
            cached_records = _load_valid_cached_batch_records(
                progress,
                batch_id=batch_id,
                expected_cache_key=cache_key,
                publication_root=publication_root,
            )
            if cached_records:
                records.extend(cached_records)
                progress.skip_batch(
                    batch_id=batch_id,
                    paper=paper,
                    paper_family=paper_family,
                    record_count=len(cached_records),
                )
                continue

        question_metadata = entry.metadata_by_path.get(str(entry.question_paper))
        if progress:
            progress.start_batch(
                batch_id=batch_id,
                paper=paper,
                paper_family=paper_family,
                phase="parsing_pdfs",
                session=progress_context["session"],
                component=progress_context["component"],
            )
        try:
            paper_records = build_records_for_pdf(
                entry.question_paper,
                config,
                mark_scheme_pdf=entry.mark_scheme,
                examiner_report_paths=entry.examiner_reports,
                filename_metadata=question_metadata,
                registry_warnings=entry.warnings,
                progress=progress,
                progress_paper=paper,
                progress_paper_family=paper_family,
                progress_session=progress_context["session"],
                progress_component=progress_context["component"],
            )
            _require_paper_records(
                paper_records,
                source_path=entry.question_paper,
                batch_id=batch_id,
            )
        except Exception as exc:
            if progress:
                progress.fail_batch(
                    batch_id=batch_id,
                    paper=paper,
                    paper_family=paper_family,
                    error_message=f"{exc.__class__.__name__}: {exc}",
                )
            raise
        if config.output.root_dir().resolve() != publication_root.resolve():
            for record in paper_records:
                _rebase_record_artifact_paths(record, config.output.root_dir(), publication_root)
        records.extend(paper_records)
        if progress:
            paper_payload = records_to_output_questions(paper_records, publication_root)
            _write_batch_cache(
                progress,
                batch_id=batch_id,
                cache_key=cache_key,
                records=paper_records,
                question_payload=paper_payload,
                rendered_root=config.output.root_dir(),
                publication_root=publication_root,
            )
            progress.complete_batch(
                batch_id=batch_id,
                paper=paper,
                paper_family=paper_family,
                record_count=len(paper_records),
                successful_records=len(paper_records),
            )
    return _finalize_registry_result(
        records,
        config,
        progress=progress,
        publication_config=publication_config,
    )


def _process_registry_entries_parallel(
    registry: DocumentRegistry,
    config: AppConfig,
    *,
    progress: Any | None,
    resume_completed_batch_ids: set[str] | None,
    force_rerun: bool,
    workers: int,
    publication_config: AppConfig,
) -> PipelineResult:
    entries = registry.question_paper_entries()
    resume_completed_batch_ids = resume_completed_batch_ids or set()
    publication_root = publication_config.output.root_dir()
    results_by_index: dict[int, list[QuestionRecord]] = {}
    stages_by_index: dict[int, Path] = {}
    if progress:
        progress.set_totals(total_batches=len(entries))

    config.output.root_dir().mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".workers-", dir=config.output.root_dir()))
    futures: dict[Future[tuple[list[QuestionRecord], Path]], tuple[int, Any, dict[str, str], str]] = {}
    try:
        with (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exam-bank-paper") as executor,
            _cancel_pending_futures_on_error(futures),
        ):
            entry_iterator = iter(enumerate(entries))

            def fill_submission_window() -> None:
                while len(futures) < workers:
                    try:
                        index, entry = next(entry_iterator)
                    except StopIteration:
                        return
                    assert entry.question_paper is not None
                    context = _entry_progress_context(entry)
                    batch_id = context["batch_id"]
                    cache_key = _extraction_batch_cache_key(entry, publication_config)
                    if progress and not force_rerun and batch_id in resume_completed_batch_ids:
                        cached_records = _load_valid_cached_batch_records(
                            progress,
                            batch_id=batch_id,
                            expected_cache_key=cache_key,
                            publication_root=publication_root,
                        )
                        if cached_records:
                            results_by_index[index] = cached_records
                            progress.skip_batch(
                                batch_id=batch_id,
                                paper=context["paper"],
                                paper_family=context["paper_family"],
                                record_count=len(cached_records),
                            )
                            continue
                    if progress:
                        progress.start_batch(
                            batch_id=batch_id,
                            paper=context["paper"],
                            paper_family=context["paper_family"],
                            phase="parsing_pdfs",
                            session=context["session"],
                            component=context["component"],
                        )
                    future = executor.submit(_build_registry_entry_in_stage, entry, config, stage_parent, context)
                    futures[future] = (index, entry, context, cache_key)

            fill_submission_window()
            while futures:
                future = next(as_completed(tuple(futures)))
                index, _entry, context, cache_key = futures[future]
                try:
                    paper_records, stage_root = future.result()
                    _require_paper_records(
                        paper_records,
                        source_path=_entry.question_paper,
                        batch_id=context["batch_id"],
                    )
                except Exception as exc:
                    if progress:
                        progress.fail_batch(
                            batch_id=context["batch_id"],
                            paper=context["paper"],
                            paper_family=context["paper_family"],
                            error_message=f"{exc.__class__.__name__}: {exc}",
                        )
                    raise
                results_by_index[index] = paper_records
                stages_by_index[index] = stage_root
                if stage_root.resolve() != publication_root.resolve():
                    for record in paper_records:
                        _rebase_record_artifact_paths(record, stage_root, publication_root)
                if progress:
                    paper_payload = records_to_output_questions(paper_records, publication_root)
                    _write_batch_cache(
                        progress,
                        batch_id=context["batch_id"],
                        cache_key=cache_key,
                        records=paper_records,
                        question_payload=paper_payload,
                        rendered_root=stage_root,
                        publication_root=publication_root,
                    )
                    progress.complete_batch(
                        batch_id=context["batch_id"],
                        paper=context["paper"],
                        paper_family=context["paper_family"],
                        record_count=len(paper_records),
                        successful_records=len(paper_records),
                    )
                del futures[future]
                fill_submission_window()
            for index in sorted(results_by_index):
                future_stage = stages_by_index.get(index)
                if future_stage is not None and future_stage.exists():
                    _promote_worker_artifacts(future_stage, config.output.root_dir(), results_by_index[index])
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)

    records = [record for index in sorted(results_by_index) for record in results_by_index[index]]
    return _finalize_registry_result(
        records,
        config,
        progress=progress,
        publication_config=publication_config,
    )


def _require_paper_records(
    records: list[QuestionRecord],
    *,
    source_path: Path,
    batch_id: str,
) -> None:
    if records:
        return
    raise EmptyPaperExtractionError(
        "Classified question paper produced zero records; refusing to mark the batch complete. "
        f"source={source_path} batch_id={batch_id}. "
        "Check whether the PDF is corrupt or unreadable and inspect question-boundary detection."
    )


@contextmanager
def _cancel_pending_futures_on_error(
    futures: dict[Future[Any], Any],
) -> Iterator[None]:
    """Cancel queued work before executor shutdown handles an interruption."""

    try:
        yield
    except BaseException:
        for future in futures:
            future.cancel()
        raise


def _build_registry_entry_in_stage(
    entry: Any,
    config: AppConfig,
    stage_parent: Path,
    context: dict[str, str],
) -> tuple[list[QuestionRecord], Path]:
    stage_root = Path(tempfile.mkdtemp(prefix=f"{context['batch_id']}-", dir=stage_parent))
    worker_config = deepcopy(config)
    worker_config.output.apply_root(stage_root)
    worker_config.ensure_output_dirs()
    question_metadata = entry.metadata_by_path.get(str(entry.question_paper))
    try:
        records = build_records_for_pdf(
            entry.question_paper,
            worker_config,
            mark_scheme_pdf=entry.mark_scheme,
            examiner_report_paths=entry.examiner_reports,
            filename_metadata=question_metadata,
            registry_warnings=entry.warnings,
        )
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    return records, stage_root


def _promote_worker_artifacts(stage_root: Path, output_root: Path, records: list[QuestionRecord]) -> None:
    for source in sorted(path for path in stage_root.rglob("*") if path.is_file()):
        destination = output_root / source.relative_to(stage_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".jsonl" and destination.exists():
            _merge_jsonl_artifacts(destination, source)
            continue
        staged_destination = destination.with_name(f".{destination.name}.worker-{stage_root.name}")
        os.replace(source, staged_destination)
        os.replace(staged_destination, destination)
    for record in records:
        _rebase_record_artifact_paths(record, stage_root, output_root)
    shutil.rmtree(stage_root, ignore_errors=True)


def _merge_jsonl_artifacts(destination: Path, source: Path) -> None:
    lines = {
        line
        for path in (destination, source)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    staged = destination.with_name(f".{destination.name}.merge")
    staged.write_text("".join(f"{line}\n" for line in sorted(lines)), encoding="utf-8")
    os.replace(staged, destination)


def _normalize_debug_jsonl_files(debug_dir: Path) -> None:
    if not debug_dir.is_dir():
        return
    for path in sorted(debug_dir.rglob("*.jsonl")):
        lines = {line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        staged = path.with_name(f".{path.name}.normalize")
        staged.write_text("".join(f"{line}\n" for line in sorted(lines)), encoding="utf-8")
        os.replace(staged, path)


def _rebase_record_artifact_paths(record: QuestionRecord, old_root: Path, new_root: Path) -> None:
    if not is_dataclass(record):
        return
    replacements = [
        (str(old_root.resolve()), str(new_root.resolve())),
        (_display_path(old_root), _display_path(new_root)),
    ]
    for field_info in fields(record):
        value = getattr(record, field_info.name)
        setattr(record, field_info.name, _rebase_artifact_value(value, replacements))


def _rebase_artifact_value(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in replacements:
            if value == old:
                return new
            prefix = old.rstrip("/") + "/"
            if value.startswith(prefix):
                return new.rstrip("/") + "/" + value[len(prefix) :]
        return value
    if isinstance(value, list):
        return [_rebase_artifact_value(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_rebase_artifact_value(item, replacements) for item in value)
    if isinstance(value, dict):
        return {key: _rebase_artifact_value(item, replacements) for key, item in value.items()}
    return value


def _finalize_registry_result(
    records: list[QuestionRecord],
    config: AppConfig,
    *,
    progress: Any | None,
    publication_config: AppConfig | None = None,
) -> PipelineResult:
    publication_config = publication_config or config
    publication_root = publication_config.output.root_dir()
    records = sorted(records, key=_canonical_record_sort_key)
    _normalize_debug_jsonl_files(config.output.debug_dir)
    if progress:
        published_json_path = publication_config.output.json_dir / publication_config.naming.json_name
        progress.update_phase(
            "writing_outputs", output_path=published_json_path, force_render=True
        )
    if publication_config is config:
        json_path = export_records(records, config)
    else:
        json_path = config.output.json_dir / config.naming.json_name
        write_question_bank_records(
            records,
            json_path,
            output_root=publication_root,
            config=publication_config,
        )
    _write_missing_image_repair_report(records, config)
    if config.debug.enabled:
        _write_batch_diagnostic(records, config)
    if progress:
        progress.update_phase("writing_reports", output_path=published_json_path, force_render=True)
    papers = {record.paper_name for record in records}
    question_count = len(records)
    return PipelineResult(records, json_path, config.output.root_dir(), question_count=question_count, paper_count=len(papers))


def _canonical_record_sort_key(record: QuestionRecord) -> tuple[str, str]:
    try:
        identity = paper_identity_from_parts(
            syllabus=record.syllabus_code or "9709",
            subject_family=record.paper_family,
            year=record.year,
            session=record.session,
            component=record.component or record.source_paper_code,
            question_number=record.question_number,
        )
        return identity.question_id, record.full_question_label
    except (IdentityError, ValueError):
        return f"{record.paper_name}:{record.question_number}", record.full_question_label


def process_sample(
    question_pdf: str | Path,
    config: AppConfig,
    mark_scheme_pdf: str | Path | None = None,
) -> PipelineResult:
    question_pdf = Path(question_pdf)
    output_root = config.output.root_dir()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    basename = _safe_basename(question_pdf.stem)
    with _output_root_lock(output_root):
        stage_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.sample-",
                dir=output_root.parent,
            )
        )
        stage_config = deepcopy(config)
        stage_config.output.apply_root(stage_root)
        stage_config.ensure_output_dirs()
        try:
            records = build_records_for_pdf(
                question_pdf,
                stage_config,
                mark_scheme_pdf=mark_scheme_pdf,
            )
            _require_paper_records(
                records,
                source_path=question_pdf,
                batch_id=f"sample:{basename}",
            )
            for record in records:
                _rebase_record_artifact_paths(record, stage_root, output_root)
            staged_json_path = stage_config.output.json_dir / f"{basename}_sample.json"
            write_question_bank_records(
                records,
                staged_json_path,
                output_root=output_root,
                config=config,
            )
            _write_missing_image_repair_report(records, stage_config)
            if stage_config.debug.enabled:
                _write_batch_diagnostic(
                    records,
                    stage_config,
                    basename=f"{basename}_sample",
                )
            final_json_relative = staged_json_path.relative_to(stage_root)
            _promote_run_artifacts_transactionally(
                stage_root,
                output_root,
                final_json_relative=final_json_relative,
            )
            return PipelineResult(
                records,
                output_root / final_json_relative,
                output_root,
            )
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)


def _entry_progress_identity(entry) -> tuple[str, str, str]:
    context = _entry_progress_context(entry)
    return context["batch_id"], context["paper"], context["paper_family"]


def _extraction_batch_cache_key(entry: Any, config: AppConfig) -> str:
    source_paths = [entry.question_paper, entry.mark_scheme, *(entry.examiner_reports or [])]
    sources = []
    for path_value in sorted({str(path) for path in source_paths if path is not None}):
        path = Path(path_value)
        sources.append(
            {
                "path": path_value,
                "sha256": _file_sha256(path) if path.is_file() else "missing",
            }
        )
    payload = {
        "cache_schema_version": EXTRACTION_CACHE_SCHEMA_VERSION,
        "pipeline_version": __version__,
        "algorithm_fingerprint": _pipeline_code_fingerprint(),
        "configuration": asdict(config),
        "ocr_profile": asdict(config.ocr),
        "sources": sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _pipeline_code_fingerprint() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_batch_cache(
    progress: Any,
    *,
    batch_id: str,
    cache_key: str,
    records: list[QuestionRecord],
    question_payload: list[dict[str, Any]],
    rendered_root: Path,
    publication_root: Path,
) -> None:
    records_payload = [asdict(record) for record in records]
    assets_payload = _build_batch_asset_manifest(
        question_payload,
        rendered_root=rendered_root,
        publication_root=publication_root,
    )
    progress.write_batch_artifact(batch_id, "questions.json", question_payload)
    progress.write_batch_artifact(batch_id, "records.json", records_payload)
    progress.write_batch_artifact(batch_id, "assets.json", assets_payload)
    progress.write_batch_artifact(
        batch_id,
        "cache_key.json",
        {
            "cache_key": cache_key,
            "cache_schema_version": EXTRACTION_CACHE_SCHEMA_VERSION,
            "algorithm_fingerprint": _pipeline_code_fingerprint(),
            "artifact_sha256": {
                "questions.json": _json_payload_sha256(question_payload),
                "records.json": _json_payload_sha256(records_payload),
                "assets.json": _json_payload_sha256(assets_payload),
            },
        },
    )


def _load_valid_cached_batch_records(
    progress: Any,
    *,
    batch_id: str,
    expected_cache_key: str,
    publication_root: Path,
) -> list[QuestionRecord] | None:
    cached_key = progress.read_batch_artifact(batch_id, "cache_key.json")
    if not isinstance(cached_key, dict):
        return None
    if cached_key.get("cache_key") != expected_cache_key:
        return None
    if cached_key.get("cache_schema_version") != EXTRACTION_CACHE_SCHEMA_VERSION:
        return None
    cached_questions = progress.read_batch_artifact(batch_id, "questions.json")
    cached_records = progress.read_batch_artifact(batch_id, "records.json")
    cached_assets = progress.read_batch_artifact(batch_id, "assets.json")
    if not isinstance(cached_questions, list) or not isinstance(cached_records, list) or not cached_records:
        return None
    expected_artifact_sha256 = cached_key.get("artifact_sha256")
    if not isinstance(expected_artifact_sha256, dict):
        return None
    actual_artifact_sha256 = {
        "questions.json": _json_payload_sha256(cached_questions),
        "records.json": _json_payload_sha256(cached_records),
        "assets.json": _json_payload_sha256(cached_assets),
    }
    if actual_artifact_sha256 != expected_artifact_sha256:
        return None
    if not _cached_asset_manifest_is_current(cached_assets, publication_root=publication_root):
        return None
    try:
        allowed_fields = {field_info.name for field_info in fields(QuestionRecord)}
        records = [
            QuestionRecord(**{key: value for key, value in payload.items() if key in allowed_fields})
            for payload in cached_records
            if isinstance(payload, dict)
        ]
    except (TypeError, ValueError):
        return None
    if len(records) != len(cached_records):
        return None
    expected_paths = _question_payload_asset_paths(records_to_output_questions(records, publication_root))
    cached_paths = {
        str(item.get("path") or "")
        for item in cached_assets.get("artifacts", [])
        if isinstance(item, dict)
    }
    if expected_paths != cached_paths:
        return None
    return records


def _json_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_batch_asset_manifest(
    question_payload: list[dict[str, Any]],
    *,
    rendered_root: Path,
    publication_root: Path,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for relative in sorted(_question_payload_asset_paths(question_payload)):
        rendered_path = rendered_root / relative
        if not rendered_path.is_file():
            raise FileNotFoundError(f"Rendered cache artifact is missing: {rendered_path}")
        artifacts.append(
            {
                "path": relative,
                "size_bytes": rendered_path.stat().st_size,
                "sha256": _file_sha256(rendered_path),
            }
        )
    return {
        "schema_name": "exam_bank.extraction_batch_assets",
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _cached_asset_manifest_is_current(payload: Any, *, publication_root: Path) -> bool:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_name") != "exam_bank.extraction_batch_assets"
        or payload.get("schema_version") != 1
    ):
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or payload.get("artifact_count") != len(artifacts):
        return False
    resolved_root = publication_root.resolve()
    for item in artifacts:
        if not isinstance(item, dict):
            return False
        relative = Path(str(item.get("path") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            return False
        path = (publication_root / relative).resolve()
        if resolved_root not in path.parents or not path.is_file():
            return False
        if path.stat().st_size != item.get("size_bytes"):
            return False
        if _file_sha256(path) != item.get("sha256"):
            return False
    return True


def _question_payload_asset_paths(question_payload: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for question in question_payload:
        if not isinstance(question, dict):
            continue
        for field_name in (
            "question_image_path",
            "mark_scheme_image_path",
            "question_image_paths",
            "mark_scheme_image_paths",
        ):
            value = question.get(field_name)
            values = value if isinstance(value, list) else [value]
            for item in values:
                text = str(item or "").strip()
                path = Path(text)
                if not text:
                    continue
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Cached artifact path is not a safe relative path: {text}")
                paths.add(path.as_posix())
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_progress_context(entry) -> dict[str, str]:
    metadata = None
    if entry.question_paper is not None:
        metadata = entry.metadata_by_path.get(str(entry.question_paper))
    component = (metadata.component if metadata else entry.component) or entry.component
    session = (metadata.normalized_session_key if metadata else entry.normalized_session_key) or entry.normalized_session_key
    year = (metadata.year if metadata else entry.year) or entry.year
    paper = paper_instance_id(component, session, year)
    paper_family = paper_family_dir_name(metadata.paper_family if metadata else f"P{component[:1]}")
    batch_id = f"{paper_family}_{paper}"
    return {
        "batch_id": batch_id,
        "paper": paper,
        "paper_family": paper_family,
        "session": session,
        "component": component,
    }


def build_records_for_pdf(
    question_pdf: str | Path,
    config: AppConfig,
    mark_scheme_pdf: str | Path | None = None,
    examiner_report_paths: list[Path] | None = None,
    filename_metadata: DocumentMetadata | None = None,
    registry_warnings: list[str] | None = None,
    progress: Any | None = None,
    progress_paper: str | None = None,
    progress_paper_family: str | None = None,
    progress_session: str | None = None,
    progress_component: str | None = None,
) -> list[QuestionRecord]:
    question_pdf = Path(question_pdf)
    config.output.root_dir().mkdir(parents=True, exist_ok=True)
    pass_parent = Path(tempfile.mkdtemp(prefix=".detection-passes-", dir=config.output.root_dir()))
    initial_root = pass_parent / "initial"
    initial_config = deepcopy(config)
    initial_config.output.apply_root(initial_root)
    initial_config.ensure_output_dirs()
    try:
        if progress:
            progress.update_phase(
                "parsing_pdfs",
                current_paper=progress_paper,
                current_paper_family=progress_paper_family,
                current_session=progress_session,
                current_component=progress_component,
            )
        layouts = extract_pdf_layout(question_pdf, initial_config)
        parsed_filename_metadata = filename_metadata or parse_filename_metadata(question_pdf)
        internal_metadata = parse_internal_document_metadata(layouts)
        document_metadata = reconcile_document_metadata(parsed_filename_metadata, internal_metadata)
        if progress:
            progress.update_phase(
                "detecting_question_spans",
                current_paper=progress_paper,
                current_paper_family=progress_paper_family,
                current_session=progress_session,
                current_component=progress_component,
            )
        initial_spans = detect_question_spans(layouts, question_pdf, initial_config)
        source_paper_code, _source_paper_code_confidence = infer_source_paper_code(question_pdf.name)
        source_paper_code = document_metadata.component or source_paper_code
        initial_records = _build_records_from_spans(
            question_pdf=question_pdf,
            layouts=layouts,
            spans=initial_spans,
            config=initial_config,
            mark_scheme_pdf=mark_scheme_pdf,
            examiner_report_paths=examiner_report_paths,
            document_metadata=document_metadata,
            registry_warnings=registry_warnings or [],
            source_paper_code=source_paper_code,
            progress=progress,
            progress_paper=progress_paper,
            progress_paper_family=progress_paper_family,
            progress_session=progress_session,
            progress_component=progress_component,
        )
        initial_total_check = _paper_total_check(
            initial_records,
            component=document_metadata.component or source_paper_code,
            paper_family=initial_records[0].paper_family if initial_records else "",
            syllabus_code=document_metadata.syllabus,
        )

        final_spans = initial_spans
        final_records = initial_records
        final_total_check = initial_total_check
        final_layouts = layouts
        final_artifact_root = initial_root
        rescan_triggered = False
        rescan_result = RescanResult.NOT_TRIGGERED

        if _should_trigger_paper_total_rescan(initial_total_check):
            rescan_triggered = True
            rescan_root = pass_parent / "rescan"
            rescan_config = deepcopy(config)
            rescan_config.output.apply_root(rescan_root)
            rescan_config.ensure_output_dirs()
            broader_config = _broadened_detection_config(config)
            broader_config.output.apply_root(rescan_root)
            broader_config.ensure_output_dirs()
            if progress:
                progress.update_phase(
                    "detecting_question_spans",
                    current_paper=progress_paper,
                    current_paper_family=progress_paper_family,
                    current_session=progress_session,
                    current_component=progress_component,
                )
            rescanned_layouts = extract_pdf_layout(question_pdf, broader_config)
            rescanned_spans = detect_question_spans(rescanned_layouts, question_pdf, broader_config)
            rescanned_records = _build_records_from_spans(
                question_pdf=question_pdf,
                layouts=rescanned_layouts,
                spans=rescanned_spans,
                config=rescan_config,
                mark_scheme_pdf=mark_scheme_pdf,
                examiner_report_paths=examiner_report_paths,
                document_metadata=document_metadata,
                registry_warnings=registry_warnings or [],
                source_paper_code=source_paper_code,
                progress=progress,
                progress_paper=progress_paper,
                progress_paper_family=progress_paper_family,
                progress_session=progress_session,
                progress_component=progress_component,
            )
            rescanned_total_check = _paper_total_check(
                rescanned_records,
                component=document_metadata.component or source_paper_code,
                paper_family=rescanned_records[0].paper_family if rescanned_records else "",
                syllabus_code=document_metadata.syllabus,
            )
            (
                final_spans,
                final_records,
                final_total_check,
                rescan_result,
            ) = _select_preferred_detection_pass(
                initial_spans=initial_spans,
                initial_records=initial_records,
                initial_total_check=initial_total_check,
                rescanned_spans=rescanned_spans,
                rescanned_records=rescanned_records,
                rescanned_total_check=rescanned_total_check,
            )
            if final_spans is rescanned_spans:
                final_layouts = rescanned_layouts
                final_artifact_root = rescan_root

        _promote_worker_artifacts(final_artifact_root, config.output.root_dir(), final_records)
        _reconcile_paper_topics(final_records, config)
        if progress:
            progress.update_phase(
                "validating_records",
                current_paper=progress_paper,
                current_paper_family=progress_paper_family,
                current_session=progress_session,
                current_component=progress_component,
            )
        _apply_paper_total_metadata(
            final_records,
            initial_total_check=initial_total_check,
            total_check=final_total_check,
            rescan_triggered=rescan_triggered,
            rescan_result=rescan_result,
            focus=_paper_total_focus(final_records),
        )
        _reconcile_question_mark_total_mismatches(final_records)
        if config.debug.enabled:
            _write_pdf_diagnostic(question_pdf, final_layouts, final_spans, final_records, config)
        return final_records
    finally:
        shutil.rmtree(pass_parent, ignore_errors=True)


def _build_records_from_spans(
    *,
    question_pdf: Path,
    layouts: list[PageLayout],
    spans: list[QuestionSpan],
    config: AppConfig,
    mark_scheme_pdf: str | Path | None,
    examiner_report_paths: list[Path] | None,
    document_metadata: DocumentMetadata,
    registry_warnings: list[str],
    source_paper_code: str,
    progress: Any | None = None,
    progress_paper: str | None = None,
    progress_paper_family: str | None = None,
    progress_session: str | None = None,
    progress_component: str | None = None,
) -> list[QuestionRecord]:
    expected_numbers = [span.question_number for span in spans if span.question_number.isdigit()]
    expected_marks = {
        span.question_number: span.question_total_detected if span.question_total_detected is not None else extract_question_total_from_text(span.combined_text)
        for span in spans
        if span.question_number.isdigit()
    }
    expected_subparts = {span.question_number: _question_subparts_from_span(span) for span in spans if span.question_number.isdigit()}
    expected_validation_flags = {span.question_number: list(span.validation_flags) for span in spans if span.question_number.isdigit()}
    paper_identity, paper_identity_flags = _paper_identity_for_metadata(question_pdf, document_metadata, source_paper_code)
    question_identities, question_identity_flags = _question_identities_for_spans(spans, question_pdf, document_metadata, source_paper_code)

    matched_mark_scheme = Path(mark_scheme_pdf) if mark_scheme_pdf else None
    explicit_mark_scheme_identity_mismatch = False
    if matched_mark_scheme is not None and paper_identity is not None and not _mark_scheme_matches_identity(matched_mark_scheme, paper_identity):
        matched_mark_scheme = None
        explicit_mark_scheme_identity_mismatch = True
    if matched_mark_scheme is None and mark_scheme_pdf is None:
        matched_mark_scheme = find_mark_scheme(
            question_pdf,
            config.input.mark_schemes_dir,
            config.input.mappings_dir,
        )
    answers: dict[str, str] = {}
    mark_scheme_images: dict[str, MarkSchemeImageResult] = {}
    mark_scheme_flags: list[str] = list(paper_identity_flags)
    if explicit_mark_scheme_identity_mismatch:
        mark_scheme_flags.append("mark_scheme_identity_mismatch")
    if matched_mark_scheme and matched_mark_scheme.exists():
        mark_scheme_analysis = None
        try:
            if progress:
                progress.update_phase(
                    "pairing_mark_schemes",
                    current_paper=progress_paper,
                    current_paper_family=progress_paper_family,
                    current_session=progress_session,
                    current_component=progress_component,
                    total_current_records=len(spans),
                )
            mark_scheme_analysis = analyze_mark_scheme(matched_mark_scheme, config)
            answers = extract_mark_scheme_answers(
                matched_mark_scheme,
                config,
                expected_numbers,
                analysis=mark_scheme_analysis,
            )
        except Exception as exc:
            mark_scheme_flags.append(f"mark_scheme_extract_failed:{exc.__class__.__name__}")
        try:
            if progress:
                progress.update_phase(
                    "rendering_mark_scheme_images",
                    current_paper=progress_paper,
                    current_paper_family=progress_paper_family,
                    current_session=progress_session,
                    current_component=progress_component,
                    total_current_records=len(spans),
                )
            mark_scheme_images = render_mark_scheme_images(
                matched_mark_scheme,
                config,
                expected_numbers,
                question_marks=expected_marks,
                question_subparts=expected_subparts,
                question_validation_flags=expected_validation_flags,
                question_identities=question_identities,
                analysis=mark_scheme_analysis,
            )
        except Exception as exc:
            mark_scheme_flags.append(f"markscheme_image_export_failed:{exc.__class__.__name__}")
    else:
        mark_scheme_flags.append("unmatched_mark_scheme")

    records: list[QuestionRecord] = []
    for record_index, span in enumerate(spans, start=1):
        if span.question_number not in question_identities:
            continue
        question_subparts = _question_subparts_from_span(span)
        if progress:
            progress.update_phase(
                "running_ocr" if config.ocr.enabled else "rendering_question_images",
                current_paper=progress_paper,
                current_paper_family=progress_paper_family,
                current_session=progress_session,
                current_component=progress_component,
                current_record_id=span.question_number,
                current_record_index=record_index,
                total_current_records=len(spans),
            )
        render_result = render_question_image(
            question_pdf,
            span,
            layouts,
            config,
            identity=question_identities.get(span.question_number),
        )
        if progress:
            progress.update_phase(
                "extracting_text",
                current_paper=progress_paper,
                current_paper_family=progress_paper_family,
                current_session=progress_session,
                current_component=progress_component,
                current_record_id=span.question_number,
                current_record_index=record_index,
                total_current_records=len(spans),
            )
        structured_text = build_structured_question_text(span, layouts, config)
        question_text = structured_text.combined_question_text or render_result.extracted_text or span.combined_text
        marks = span.question_total_detected if span.question_total_detected is not None else extract_question_total_from_text(question_text)
        answer_text = answers.get(span.question_number, "")
        mark_scheme_image = mark_scheme_images.get(span.question_number)
        examiner_evidence = None
        if config.runtime.supports_input_document_type("examiner_report"):
            examiner_evidence = examiner_report_topic_evidence(
                question_pdf,
                config.input.examiner_reports_dir,
                span.question_number,
                config,
                report_paths=examiner_report_paths,
            )
        examiner_text = examiner_evidence.classification_text if examiner_evidence else ""
        records.append(
            _build_question_record(
                question_pdf=question_pdf,
                span=span,
                question_text=question_text,
                marks=marks,
                answer_text=answer_text,
                render_result=render_result,
                structured_text=structured_text,
                question_subparts=question_subparts,
                mark_scheme_image=mark_scheme_image,
                mark_scheme_flags=mark_scheme_flags,
                identity_flags=question_identity_flags.get(span.question_number, []),
                matched_mark_scheme=matched_mark_scheme,
                document_metadata=document_metadata,
                registry_warnings=registry_warnings or [],
                config=config,
                source_paper_code=source_paper_code,
                examiner_evidence=examiner_evidence,
                examiner_text=examiner_text,
            )
        )
    return records


def _paper_identity_for_metadata(
    source_pdf: Path,
    document_metadata: DocumentMetadata,
    source_paper_code: str,
) -> tuple[PaperIdentity | None, list[str]]:
    try:
        return (
            paper_identity_from_parts(
                syllabus=document_metadata.syllabus or "9709",
                subject_family=document_metadata.paper_family,
                year=document_metadata.year,
                session=session_for_source_path(
                    source_pdf,
                    year=document_metadata.year,
                    fallback_session=document_metadata.normalized_session_key or document_metadata.session,
                ),
                component=document_metadata.component or source_paper_code,
            ),
            [],
        )
    except IdentityError as exc:
        return None, [f"paper_identity_unresolved:{exc}"]


def _question_identities_for_spans(
    spans: list[QuestionSpan],
    source_pdf: Path,
    document_metadata: DocumentMetadata,
    source_paper_code: str,
) -> tuple[dict[str, PaperIdentity], dict[str, list[str]]]:
    identities: dict[str, PaperIdentity] = {}
    flags: dict[str, list[str]] = {}
    for span in spans:
        try:
            identities[span.question_number] = paper_identity_from_parts(
                syllabus=document_metadata.syllabus or "9709",
                subject_family=document_metadata.paper_family,
                year=document_metadata.year,
                session=session_for_source_path(
                    source_pdf,
                    year=document_metadata.year,
                    fallback_session=document_metadata.normalized_session_key or document_metadata.session,
                ),
                component=document_metadata.component or source_paper_code,
                question_number=span.question_number,
            )
        except IdentityError as exc:
            flags[span.question_number] = [f"question_identity_unresolved:{exc}", "question_asset_not_emitted"]
    return identities, flags


def _mark_scheme_matches_identity(mark_scheme_pdf: Path, paper_identity: PaperIdentity) -> bool:
    metadata = parse_filename_metadata(mark_scheme_pdf)
    try:
        mark_scheme_identity = paper_identity_from_parts(
            syllabus=metadata.syllabus or "9709",
            subject_family=metadata.paper_family,
            year=metadata.year,
            session=session_for_source_path(
                mark_scheme_pdf,
                year=metadata.year,
                fallback_session=metadata.normalized_session_key or metadata.session,
            ),
            component=metadata.component,
        )
    except IdentityError:
        return False
    return mark_scheme_identity.paper_id == paper_identity.paper_id


def _build_question_record(
    *,
    question_pdf: Path,
    span: QuestionSpan,
    question_text: str,
    marks: int | None,
    answer_text: str,
    render_result,
    structured_text,
    question_subparts: list[str],
    mark_scheme_image: MarkSchemeImageResult | None,
    mark_scheme_flags: list[str],
    identity_flags: list[str],
    matched_mark_scheme: Path | None,
    document_metadata: DocumentMetadata,
    registry_warnings: list[str],
    config: AppConfig,
    source_paper_code: str,
    examiner_evidence,
    examiner_text: str,
) -> QuestionRecord:
        flags = list(span.review_flags)
        flags.extend(mark_scheme_flags)
        flags.extend(identity_flags)
        flags.extend(document_metadata.warnings)
        flags.extend(registry_warnings)
        if matched_mark_scheme and matched_mark_scheme.exists() and not answer_text:
            flags.append("unmatched_answer")
        if matched_mark_scheme and matched_mark_scheme.exists():
            if mark_scheme_image is None or not mark_scheme_image.image_path:
                flags.append("markscheme_image_missing")
            elif mark_scheme_image.crop_confidence != "high":
                flags.append("markscheme_image_uncertain")
        if mark_scheme_image:
            flags.extend(mark_scheme_image.review_flags)

        if not render_result.screenshot_path:
            flags.append("missing_question_image")
        if render_result.crop_uncertain:
            flags.append("low_confidence_question_crop")
        flags.extend(render_result.review_flags)
        preliminary_validation_status, preliminary_validation_flags = _refine_validation_status(
            base_status=span.validation_status,
            base_validation_flags=span.validation_flags,
            mapping_status=mark_scheme_image.mapping_status if mark_scheme_image else MappingStatus.FAIL,
            mapping_failure_reason=mark_scheme_image.failure_reason if mark_scheme_image else "",
            crop_uncertain=render_result.crop_uncertain,
            extraction_quality_flags=structured_text.extraction_quality_flags,
            review_flags=flags,
            question_structure_detected=span.structure_detected,
        )
        preliminary_scope_quality_status = _derive_scope_quality_status(
            validation_flags=preliminary_validation_flags,
            review_flags=flags,
            question_structure_detected=span.structure_detected,
        )
        text_candidate_decision = select_text_candidate(
            native_text=question_text,
            ocr_text=render_result.ocr_text,
            expected_question_number=span.question_number,
            expected_subparts=question_subparts,
            scope_quality_status=preliminary_scope_quality_status,
            mapping_status=mark_scheme_image.mapping_status if mark_scheme_image else MappingStatus.FAIL,
            validation_status=preliminary_validation_status,
        )
        question_text = text_candidate_decision.selected_text
        if text_candidate_decision.ocr_selected:
            flags.extend(["ocr_question_text", "ocr_selected_for_question_text"])
            if marks is None:
                marks = extract_question_total_from_text(question_text)
        elif "ocr_large_margin_blocked_by_structural_rejection" in text_candidate_decision.ocr_rejected_reasons:
            flags.append("ocr_large_margin_blocked_by_structural_rejection")
        classification = classify_question(
            question_text,
            marks,
            config,
            context_flags=flags + list(structured_text.extraction_quality_flags),
            source_name=question_pdf.name,
            examiner_report_text=examiner_text,
            mark_scheme_text=answer_text,
            question_ocr_text=render_result.ocr_text,
            body_text_normalized=structured_text.body_text_normalized,
            part_texts=structured_text.part_texts,
            body_text_raw=structured_text.body_text_raw,
            math_lines=structured_text.math_lines,
        )
        part_level_topics = classify_question_parts(
            question_text,
            span.question_number,
            config,
            context_flags=flags + list(structured_text.extraction_quality_flags),
            source_name=question_pdf.name,
            examiner_report_text=examiner_text,
            mark_scheme_text=answer_text,
            question_ocr_text=render_result.ocr_text,
            structured_part_texts=structured_text.part_texts,
        )
        question_topic = _question_topic_from_parts(classification, part_level_topics)
        flags.extend(question_topic["review_flags"])
        flags = sorted(set(flags))
        confidence = _record_confidence(float(question_topic["confidence"]), flags)
        validation_status, validation_flags = _refine_validation_status(
            base_status=span.validation_status,
            base_validation_flags=span.validation_flags,
            mapping_status=mark_scheme_image.mapping_status if mark_scheme_image else MappingStatus.FAIL,
            mapping_failure_reason=mark_scheme_image.failure_reason if mark_scheme_image else "",
            crop_uncertain=render_result.crop_uncertain,
            extraction_quality_flags=structured_text.extraction_quality_flags,
            review_flags=flags,
            question_structure_detected=span.structure_detected,
        )
        scope_quality_status = _derive_scope_quality_status(
            validation_flags=validation_flags,
            review_flags=flags,
            question_structure_detected=span.structure_detected,
        )
        text_source_profile = _text_source_profile(flags)
        text_fidelity_status, text_fidelity_flags = _assess_text_fidelity(
            question_text=question_text,
            extraction_quality_flags=structured_text.extraction_quality_flags,
            review_flags=flags,
            validation_flags=validation_flags,
            question_structure_detected=span.structure_detected,
            mapping_failure_reason=mark_scheme_image.failure_reason if mark_scheme_image else "",
            text_source_profile=text_source_profile,
        )
        visual_reason_flags = _visual_reason_flags(
            question_text=question_text,
            extraction_quality_flags=structured_text.extraction_quality_flags,
            review_flags=flags,
            question_structure_detected=span.structure_detected,
            text_source_profile=text_source_profile,
        )
        question_text_role, question_text_trust, visual_required = _derive_question_text_semantics(
            question_text=question_text,
            text_fidelity_status=text_fidelity_status,
            visual_reason_flags=visual_reason_flags,
        )
        missing_image_reason = _missing_question_image_reason(
            question_text=question_text,
            visual_required=visual_required,
            visual_reason_flags=visual_reason_flags,
            crop_diagnostics=render_result.crop_diagnostics,
        )
        if missing_image_reason:
            validation_status = ValidationStatus.FAIL
            validation_flags = sorted(set(validation_flags) | {"missing_image_detection_failure"})
            flags = sorted(set(flags) | {"missing_image_detection_failure"})
            render_result.crop_diagnostics["missing_image_reason"] = missing_image_reason
            render_result.crop_diagnostics["missing_image_failure_metadata"] = {
                **dict(render_result.crop_diagnostics.get("missing_image_failure_metadata") or {}),
                "reason": missing_image_reason,
                "hard_failure": True,
                "visual_reason_flags": list(visual_reason_flags),
            }
        question_crop_confidence = CropConfidence.LOW if render_result.crop_uncertain else CropConfidence.HIGH
        mark_scheme_image_path = _display_path(mark_scheme_image.image_path) if mark_scheme_image and mark_scheme_image.image_path else ""
        mark_scheme_crop_confidence = mark_scheme_image.crop_confidence if mark_scheme_image else ""
        missing_mark_scheme_reason = ""
        if mark_scheme_image and mark_scheme_image.missing_mark_scheme_reason:
            missing_mark_scheme_reason = mark_scheme_image.missing_mark_scheme_reason
        elif matched_mark_scheme and matched_mark_scheme.exists() and not mark_scheme_image_path:
            missing_mark_scheme_reason = "segmentation_failure"
        elif not matched_mark_scheme:
            missing_mark_scheme_reason = "unmatched_mark_scheme"
        visual_curation_status = _derive_visual_curation_status(
            validation_status=validation_status,
            scope_quality_status=scope_quality_status,
            question_image_path=_display_path(render_result.screenshot_path) if render_result.screenshot_path else "",
            question_crop_confidence=question_crop_confidence,
            mark_scheme_image_path=mark_scheme_image_path,
            mark_scheme_crop_confidence=mark_scheme_crop_confidence,
        )
        text_only_status = _derive_text_only_status(
            validation_status=validation_status,
            scope_quality_status=scope_quality_status,
            question_text_role=question_text_role,
            question_text_trust=question_text_trust,
        )
        topic_trust_status = _derive_topic_trust_status(
            topic_confidence=str(question_topic["topic_confidence"]),
            topic_uncertain=bool(question_topic["topic_uncertain"]),
            text_fidelity_status=text_fidelity_status,
            validation_status=validation_status,
            scope_quality_status=scope_quality_status,
            question_text_role=question_text_role,
            visual_required=visual_required,
        )

        return QuestionRecord(
                source_pdf=_display_path(question_pdf),
                paper_name=span.paper_name,
                question_number=span.question_number,
                full_question_label=span.full_question_label,
                screenshot_path=_display_path(render_result.screenshot_path) if render_result.screenshot_path else "",
                combined_question_text=question_text,
                body_text_raw=structured_text.body_text_raw,
                body_text_normalized=structured_text.body_text_normalized,
                math_lines=structured_text.math_lines,
                diagram_text=structured_text.diagram_text,
                extraction_quality_score=structured_text.extraction_quality_score,
                extraction_quality_flags=structured_text.extraction_quality_flags,
                part_texts=structured_text.part_texts,
                answer_text=answer_text,
                paper_family=str(question_topic["paper_family"]),
                source_paper_code=source_paper_code,
                syllabus_code=document_metadata.syllabus,
                session=document_metadata.session,
                year=document_metadata.year,
                document_type=document_metadata.document_type or "question_paper",
                component=document_metadata.component,
                document_key=document_metadata.canonical_key,
                metadata_source=document_metadata.source,
                mark_scheme_source_pdf=_display_path(matched_mark_scheme) if matched_mark_scheme else "",
                source_paper_family=classification.source_paper_family,
                inferred_paper_family=classification.inferred_paper_family,
                paper_family_confidence=classification.paper_family_confidence,
                question_level_paper_family=str(question_topic["paper_family"]),
                question_level_topic=str(question_topic["topic"]),
                question_level_subtopic=str(question_topic["subtopic"]),
                part_level_topics=part_level_topics,
                topic=str(question_topic["topic"]),
                subtopic=str(question_topic["subtopic"]),
                topic_confidence=str(question_topic["topic_confidence"]),
                topic_evidence=classification.topic_evidence,
                topic_evidence_details={
                    **classification.topic_evidence_details,
                    **({"examiner_report_structured": examiner_evidence.to_dict()} if examiner_evidence else {}),
                },
                examiner_report_evidence=examiner_evidence.to_dict() if examiner_evidence else {},
                secondary_topics=list(question_topic["secondary_topics"]),
                topic_uncertain=bool(question_topic["topic_uncertain"]),
                difficulty=classification.difficulty,
                difficulty_confidence=classification.difficulty_confidence,
                difficulty_evidence=classification.difficulty_evidence,
                difficulty_uncertain=classification.difficulty_uncertain,
                difficulty_score=classification.difficulty_score,
                difficulty_band=classification.difficulty_band,
                difficulty_score_scale=classification.difficulty_score_scale,
                difficulty_features=classification.difficulty_features,
                difficulty_review_flags=classification.difficulty_review_flags,
                difficulty_model_version=classification.difficulty_model_version,
                marks=marks,
                marks_if_available=marks,
                page_numbers=span.page_numbers,
                review_flags=flags,
                confidence=confidence,
                crop_uncertain=render_result.crop_uncertain,
                question_crop_confidence=question_crop_confidence,
                crop_debug_paths=render_result.debug_paths,
                question_crop_diagnostics=render_result.crop_diagnostics,
                topic_alternatives=classification.alternative_topics,
                markscheme_image=mark_scheme_image_path,
                markscheme_pages=mark_scheme_image.page_numbers if mark_scheme_image else [],
                markscheme_question_number=mark_scheme_image.markscheme_question_number if mark_scheme_image else "",
                markscheme_crop_confidence=mark_scheme_crop_confidence,
                markscheme_mapping_method=mark_scheme_image.mapping_method if mark_scheme_image else "",
                markscheme_table_detected=mark_scheme_image.table_detected if mark_scheme_image else False,
                markscheme_table_header_detected=mark_scheme_image.table_header_detected if mark_scheme_image else [],
                markscheme_nearby_anchors=mark_scheme_image.nearby_anchors if mark_scheme_image else [],
                markscheme_debug_paths=mark_scheme_image.debug_paths if mark_scheme_image else [],
                markscheme_table_header_ok=mark_scheme_image.table_header_ok if mark_scheme_image else False,
                markscheme_continuation_rows_included=mark_scheme_image.continuation_rows_included if mark_scheme_image else False,
                question_subparts=question_subparts,
                markscheme_subparts=mark_scheme_image.markscheme_subparts if mark_scheme_image else [],
                question_marks_total=mark_scheme_image.question_marks_total if mark_scheme_image else marks,
                markscheme_marks_total=mark_scheme_image.markscheme_marks_total if mark_scheme_image else None,
                markscheme_mapping_status=mark_scheme_image.mapping_status if mark_scheme_image else MappingStatus.FAIL,
                markscheme_failure_reason=mark_scheme_image.failure_reason if mark_scheme_image else "segmentation_failure",
                markscheme_block_ids=mark_scheme_image.block_ids if mark_scheme_image else [],
                markscheme_confidence_score=mark_scheme_image.confidence_score if mark_scheme_image else 0.0,
                missing_mark_scheme_reason=missing_mark_scheme_reason,
                validation_status=validation_status,
                validation_flags=validation_flags,
                scope_quality_status=scope_quality_status,
                text_source_profile=text_source_profile,
                text_fidelity_status=text_fidelity_status,
                text_fidelity_flags=text_fidelity_flags,
                question_text_role=question_text_role,
                question_text_trust=question_text_trust,
                visual_required=visual_required,
                visual_reason_flags=visual_reason_flags,
                visual_curation_status=visual_curation_status,
                text_only_status=text_only_status,
                topic_trust_status=topic_trust_status,
                recovery_attempted=span.recovery_attempted,
                recovery_result=span.recovery_result,
                ocr_ran=render_result.ocr_ran,
                ocr_engine=render_result.ocr_engine,
                ocr_text=render_result.ocr_text,
                ocr_text_trust=render_result.ocr_text_trust,
                ocr_failure_reason=render_result.ocr_failure_reason,
                ocr_text_role=render_result.ocr_text_role,
                text_candidate_source=text_candidate_decision.text_candidate_source,
                native_text_score=text_candidate_decision.native_text_score,
                ocr_text_score=text_candidate_decision.ocr_text_score,
                selected_text_score=text_candidate_decision.selected_text_score,
                text_candidate_decision=text_candidate_decision.text_candidate_decision,
                text_candidate_decision_reasons=text_candidate_decision.text_candidate_decision_reasons,
                ocr_selected=text_candidate_decision.ocr_selected,
                ocr_rejected_reasons=text_candidate_decision.ocr_rejected_reasons,
                question_structure_detected=span.structure_detected,
                mark_scheme_structure_detected={
                    "subparts": mark_scheme_image.markscheme_subparts if mark_scheme_image else [],
                    "question_subparts": mark_scheme_image.question_subparts if mark_scheme_image else [],
                    "question_total_detected": mark_scheme_image.question_marks_total if mark_scheme_image else None,
                    "mark_scheme_total_detected": mark_scheme_image.markscheme_marks_total if mark_scheme_image else None,
                    "mark_scheme_block_ids": mark_scheme_image.block_ids if mark_scheme_image else [],
                    "mark_scheme_confidence_score": mark_scheme_image.confidence_score if mark_scheme_image else 0.0,
                    "missing_mark_scheme_reason": missing_mark_scheme_reason,
                    "asset_identity": {
                        "question_id": mark_scheme_image.question_id if mark_scheme_image else "",
                        "paper_id": mark_scheme_image.paper_id if mark_scheme_image else "",
                        "component": mark_scheme_image.component if mark_scheme_image else "",
                        "canonical_path": mark_scheme_image.canonical_path if mark_scheme_image else "",
                    },
                },
                question_total_detected=span.question_total_detected,
                mark_scheme_total_detected=mark_scheme_image.markscheme_marks_total if mark_scheme_image else None,
                question_format_profile=span.format_profile,
            )


_QUESTION_SUBPART_LABEL_RE = re.compile(
    r"^\s*(?:\d+\s*)?(?P<labels>\((?:a|b|c|d|e|f|g|h|viii|vii|vi|iv|ix|iii|ii|i|v|x)\)(?:\s*\((?:a|b|c|d|e|f|g|h|viii|vii|vi|iv|ix|iii|ii|i|v|x)\))*)",
    re.IGNORECASE,
)


def _question_subparts_from_text(text: str) -> list[str]:
    subparts: list[str] = []
    for line in text.splitlines():
        match = _QUESTION_SUBPART_LABEL_RE.match(line.strip())
        if not match:
            continue
        for label in re.findall(r"\((a|b|c|d|e|f|g|h|viii|vii|vi|iv|ix|iii|ii|i|v|x)\)", match.group("labels"), re.IGNORECASE):
            normalized = label.lower()
            if normalized not in subparts:
                subparts.append(normalized)
    if subparts and "a" not in subparts and _embedded_alpha_subpart_label_present(text, "a"):
        subparts.insert(0, "a")
    alpha_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]
    roman_labels = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
    if any(label in alpha_labels for label in subparts):
        return sorted({label for label in subparts if label in alpha_labels}, key=alpha_labels.index)
    if any(label in roman_labels for label in subparts):
        return sorted({label for label in subparts if label in roman_labels}, key=roman_labels.index)
    return subparts


def _embedded_alpha_subpart_label_present(text: str, label: str) -> bool:
    for match in re.finditer(rf"\({re.escape(label)}\)", text):
        after = text[match.end() : match.end() + 240]
        if re.search(r"\[\d{1,2}\]", after):
            return True
    return False


def _question_subparts_from_span(span: QuestionSpan) -> list[str]:
    lines: list[str] = []
    for block in span.blocks:
        control_stripped = "".join(char if ord(char) >= 32 or char in "\n\t\r" else " " for char in block.text)
        for raw_line in control_stripped.replace("\u00a0", " ").splitlines():
            normalized_line = " ".join(raw_line.split())
            if normalized_line:
                lines.append(normalized_line)
    return _question_subparts_from_text("\n".join(lines))


def _broadened_detection_config(config: AppConfig) -> AppConfig:
    broader = deepcopy(config)
    broader.detection.anchor_min_confidence = max(0.45, config.detection.anchor_min_confidence - 0.08)
    broader.detection.anchor_y_tolerance = config.detection.anchor_y_tolerance + 8
    broader.detection.prompt_region_max_gap = config.detection.prompt_region_max_gap + 30
    broader.detection.crop_padding = config.detection.crop_padding + 8
    broader.detection.crop_bottom_margin = max(18, config.detection.crop_bottom_margin - 12)
    return broader


def _record_solution_marks(record: QuestionRecord) -> int | None:
    for value in [record.markscheme_marks_total, record.question_marks_total, record.marks_if_available, record.marks]:
        if value is not None:
            return int(value)
    return None


_PAPER_TOTALS_BY_SYLLABUS: dict[str, dict[str, int]] = {
    "9709": {"P1": 75, "P3": 75, "P4": 50, "P5": 50, "P6": 50},
}


def _normalize_syllabus_code(syllabus_code: str = "") -> str:
    match = re.search(r"\b\d{4}\b", str(syllabus_code or ""))
    return match.group(0) if match else str(syllabus_code or "").strip()


def _expected_paper_total(component: str, paper_family: str = "", syllabus_code: str = "") -> int | None:
    syllabus = _normalize_syllabus_code(syllabus_code)
    if syllabus and syllabus not in _PAPER_TOTALS_BY_SYLLABUS:
        return None

    totals = _PAPER_TOTALS_BY_SYLLABUS["9709"]
    family = (paper_family or "").strip().upper()
    if family in totals:
        return totals[family]

    digits = "".join(char for char in str(component) if char.isdigit())
    if digits:
        return {family[-1]: total for family, total in totals.items()}.get(digits[0])
    return None


def _paper_total_syllabus_code(records: list[QuestionRecord], syllabus_code: str = "") -> str:
    explicit_syllabus = _normalize_syllabus_code(syllabus_code)
    if explicit_syllabus:
        return explicit_syllabus

    record_syllabuses = {
        normalized
        for record in records
        if (normalized := _normalize_syllabus_code(getattr(record, "syllabus_code", "")))
    }
    if len(record_syllabuses) == 1:
        return next(iter(record_syllabuses))
    if record_syllabuses:
        return "unsupported"
    return ""


def _paper_total_check(
    records: list[QuestionRecord],
    *,
    component: str,
    paper_family: str,
    syllabus_code: str = "",
) -> dict[str, int | str | bool | None]:
    expected_total = _expected_paper_total(
        component,
        paper_family,
        syllabus_code=_paper_total_syllabus_code(records, syllabus_code),
    )
    detected_total = sum(mark for record in records if (mark := _record_solution_marks(record)) is not None)
    status = PaperTotalStatus.UNKNOWN_EXPECTED_TOTAL
    if expected_total is not None:
        status = PaperTotalStatus.MATCHED if detected_total == expected_total else PaperTotalStatus.MISMATCH
    return {
        "expected_total": expected_total,
        "detected_total": detected_total,
        "status": status,
        "difference": None if expected_total is None else detected_total - expected_total,
    }


def _should_trigger_paper_total_rescan(total_check: dict[str, int | str | bool | None]) -> bool:
    return total_check.get("expected_total") is not None and total_check.get("status") == PaperTotalStatus.MISMATCH


def _structural_failure_count(records: list[QuestionRecord]) -> int:
    structural_flags = {
        "question_subparts_incomplete",
        "question_scope_contaminated",
        "missing_terminal_mark_total",
        "question_mark_total_mismatch",
        "question_mark_total_missing",
        "likely_truncated_question_crop",
    }
    count = 0
    for record in records:
        if any(flag in structural_flags for flag in record.validation_flags):
            count += 1
            continue
        if record.markscheme_failure_reason in structural_flags:
            count += 1
    return count


_PAPER_TOTAL_RECORD_HARD_FLAGS = frozenset(
    {
        "question_subparts_incomplete",
        "question_scope_contaminated",
        "missing_terminal_mark_total",
        "question_mark_total_mismatch",
        "question_mark_total_missing",
        "likely_truncated_question_crop",
        "polluted_pass_requires_review",
    }
)


def _paper_total_mismatch_should_fail_record(record: QuestionRecord) -> bool:
    if record.validation_status == ValidationStatus.FAIL:
        return True
    if record.markscheme_mapping_status == MappingStatus.FAIL:
        return True
    if record.visual_curation_status == "fail":
        return True
    if set(record.validation_flags) & _PAPER_TOTAL_RECORD_HARD_FLAGS:
        return True
    if record.markscheme_failure_reason in _PAPER_TOTAL_RECORD_HARD_FLAGS:
        return True
    return False


def _reconcile_question_mark_total_mismatches(records: list[QuestionRecord]) -> None:
    for record in records:
        if not _should_trust_markscheme_total_for_record(record):
            continue

        validation_flags = set(record.validation_flags)
        validation_flags.discard("question_mark_total_mismatch")
        record.validation_flags = sorted(validation_flags)
        record.markscheme_mapping_status = MappingStatus.PASS
        record.markscheme_failure_reason = ""

        review_flags = set(record.review_flags)
        review_flags.add("question_mark_total_mismatch")
        review_flags.add("question_mark_total_review_only")
        record.review_flags = sorted(review_flags)
        record.validation_status = ValidationStatus.REVIEW
        _refresh_validation_derivatives(record)


def _should_trust_markscheme_total_for_record(record: QuestionRecord) -> bool:
    if record.paper_total_status not in {PaperTotalStatus.MATCHED, PaperTotalStatus.RECOVERED_AFTER_RESCAN}:
        return False
    if record.paper_total_expected is None or record.paper_total_detected != record.paper_total_expected:
        return False
    if record.markscheme_mapping_status != MappingStatus.FAIL:
        return False
    if record.markscheme_failure_reason != "question_mark_total_mismatch":
        return False
    if record.question_marks_total is None or record.markscheme_marks_total is None:
        return False
    remaining_hard_flags = set(record.validation_flags) - {"question_mark_total_mismatch"}
    if remaining_hard_flags & (_PAPER_TOTAL_RECORD_HARD_FLAGS - {"question_mark_total_mismatch"}):
        return False
    if not record.markscheme_image:
        return False
    return True


def _refresh_validation_derivatives(record: QuestionRecord) -> None:
    if not record.scope_quality_status:
        record.scope_quality_status = _derive_scope_quality_status(
            validation_flags=record.validation_flags,
            review_flags=record.review_flags,
            question_structure_detected=record.question_structure_detected,
        )
    record.visual_curation_status = _derive_visual_curation_status(
        validation_status=record.validation_status,
        scope_quality_status=record.scope_quality_status,
        question_image_path=record.screenshot_path,
        question_crop_confidence=record.question_crop_confidence,
        mark_scheme_image_path=record.markscheme_image,
        mark_scheme_crop_confidence=record.markscheme_crop_confidence,
    )
    record.text_only_status = _derive_text_only_status(
        validation_status=record.validation_status,
        scope_quality_status=record.scope_quality_status,
        question_text_role=record.question_text_role,
        question_text_trust=record.question_text_trust,
    )
    record.topic_trust_status = _derive_topic_trust_status(
        topic_confidence=record.topic_confidence,
        topic_uncertain=record.topic_uncertain,
        text_fidelity_status=record.text_fidelity_status,
        validation_status=record.validation_status,
        scope_quality_status=record.scope_quality_status,
        question_text_role=record.question_text_role,
        visual_required=record.visual_required,
    )


def _paper_total_preference_key(
    total_check: dict[str, int | str | bool | None],
    records: list[QuestionRecord],
) -> tuple[int, int, int]:
    expected_total = total_check.get("expected_total")
    detected_total = int(total_check.get("detected_total") or 0)
    if expected_total is None:
        distance = 10**6
    else:
        distance = abs(detected_total - int(expected_total))
    return (-distance, -_structural_failure_count(records), len(records))


def _select_preferred_detection_pass(
    *,
    initial_spans: list[QuestionSpan],
    initial_records: list[QuestionRecord],
    initial_total_check: dict[str, int | str | bool | None],
    rescanned_spans: list[QuestionSpan],
    rescanned_records: list[QuestionRecord],
    rescanned_total_check: dict[str, int | str | bool | None],
) -> tuple[
    list[QuestionSpan],
    list[QuestionRecord],
    dict[str, int | str | bool | None],
    str,
]:
    if rescanned_total_check.get("status") == PaperTotalStatus.MATCHED and initial_total_check.get("status") != PaperTotalStatus.MATCHED:
        return rescanned_spans, rescanned_records, rescanned_total_check, RescanResult.RECOVERED_EXACT_TOTAL

    if _paper_total_preference_key(rescanned_total_check, rescanned_records) > _paper_total_preference_key(initial_total_check, initial_records):
        result = RescanResult.IMPROVED_BUT_STILL_MISMATCH
        if rescanned_total_check.get("status") == PaperTotalStatus.MATCHED:
            result = RescanResult.RECOVERED_EXACT_TOTAL
        return rescanned_spans, rescanned_records, rescanned_total_check, result

    return initial_spans, initial_records, initial_total_check, RescanResult.NO_IMPROVEMENT


def _apply_paper_total_metadata(
    records: list[QuestionRecord],
    *,
    initial_total_check: dict[str, int | str | bool | None],
    total_check: dict[str, int | str | bool | None],
    rescan_triggered: bool,
    rescan_result: str,
    focus: dict[str, object],
) -> None:
    before_total = initial_total_check.get("detected_total")
    expected_total = total_check.get("expected_total")
    detected_total = total_check.get("detected_total")
    status = str(total_check.get("status") or "")
    focus_questions = [str(question) for question in focus.get("question_numbers", [])]
    focus_pages = [int(page) for page in focus.get("pages", [])]
    reasons_by_question = dict(focus.get("reasons_by_question", {}))
    if status == PaperTotalStatus.MATCHED and rescan_triggered:
        status = PaperTotalStatus.RECOVERED_AFTER_RESCAN if rescan_result == RescanResult.RECOVERED_EXACT_TOTAL else PaperTotalStatus.MATCHED
    elif status == PaperTotalStatus.MISMATCH and rescan_triggered:
        status = PaperTotalStatus.MISMATCH_AFTER_RESCAN

    for record in records:
        record.paper_total_expected = int(expected_total) if expected_total is not None else None
        record.paper_total_detected = int(detected_total) if detected_total is not None else None
        record.paper_total_status = status
        record.rescan_triggered = rescan_triggered
        record.rescan_result = rescan_result
        record.paper_total_before_rescan = int(before_total) if before_total is not None else None
        record.paper_total_after_rescan = int(detected_total) if detected_total is not None else None
        record.paper_total_focus_questions = list(focus_questions)
        record.paper_total_focus_pages = list(focus_pages)
        record.paper_total_focus_reason = str(reasons_by_question.get(record.question_number, ""))

        review_flags = set(record.review_flags)
        if rescan_triggered:
            review_flags.add("paper_total_rescan_triggered")
        if rescan_result == RescanResult.RECOVERED_EXACT_TOTAL:
            review_flags.add("paper_total_rescan_recovered")
        if record.question_number in focus_questions:
            review_flags.add("paper_total_focus_candidate")
        if status == PaperTotalStatus.MISMATCH_AFTER_RESCAN:
            review_flags.add("paper_total_mismatch")
        record.review_flags = sorted(review_flags)

        validation_flags = set(record.validation_flags)
        if status == PaperTotalStatus.MISMATCH_AFTER_RESCAN:
            if _paper_total_mismatch_should_fail_record(record):
                validation_flags.add("paper_total_mismatch")
                record.validation_status = ValidationStatus.FAIL
            elif record.validation_status in {"", ValidationStatus.PASS}:
                record.validation_status = ValidationStatus.REVIEW
        record.validation_flags = sorted(validation_flags)


def _paper_total_focus(records: list[QuestionRecord]) -> dict[str, object]:
    scored: list[tuple[int, str, list[int], list[str]]] = []
    for record in records:
        reasons: list[str] = []
        structural_reason = record.markscheme_failure_reason
        if structural_reason in {
            "question_scope_contaminated",
            "question_subparts_incomplete",
            "question_mark_total_mismatch",
            "question_mark_total_missing",
        }:
            reasons.append(structural_reason)
        if "weak_question_anchor" in record.validation_flags or "question_start_uncertain" in record.review_flags:
            reasons.append("anchor_or_boundary")
        if "possible_next_question_contamination" in record.review_flags:
            reasons.append("adjacent_boundary_contamination")
        if record.recovery_attempted and record.recovery_result == "no_change":
            reasons.append("recovery_stalled")
        if len(record.page_numbers) > 1:
            reasons.append("cross_page_scope")
        if not reasons:
            continue
        score = 0
        priority = {
            "question_scope_contaminated": 5,
            "question_subparts_incomplete": 5,
            "question_mark_total_mismatch": 4,
            "question_mark_total_missing": 4,
            "anchor_or_boundary": 3,
            "adjacent_boundary_contamination": 3,
            "recovery_stalled": 2,
            "cross_page_scope": 1,
        }
        for reason in reasons:
            score += priority.get(reason, 1)
        scored.append((score, record.question_number, list(record.page_numbers), reasons))

    scored.sort(key=lambda item: (-item[0], int(item[1]) if item[1].isdigit() else 999, item[1]))
    top = scored[:3]
    pages: list[int] = []
    question_numbers: list[str] = []
    reasons_by_question: dict[str, str] = {}
    for _score, question_number, record_pages, reasons in top:
        question_numbers.append(question_number)
        for page in record_pages:
            if page not in pages:
                pages.append(page)
        reasons_by_question[question_number] = ", ".join(reasons)
    return {
        "question_numbers": question_numbers,
        "pages": pages,
        "reasons_by_question": reasons_by_question,
    }


def _missing_question_image_reason(
    *,
    question_text: str = "",
    visual_required: bool,
    visual_reason_flags: list[str],
    crop_diagnostics: dict[str, Any],
) -> str:
    if not visual_required:
        return ""
    source_visual_referenced = _references_source_visual(question_text)
    existing_reason = str(crop_diagnostics.get("missing_image_reason") or "")
    if existing_reason and (source_visual_referenced or not str(question_text or "").strip()):
        return existing_reason
    if not source_visual_referenced:
        return ""
    figure_like_flags = {
        "contains_graph_or_diagram_prompt",
        "contains_table_or_data_prompt",
        "contains_inequality_or_region_prompt",
    }
    if not (set(visual_reason_flags) & figure_like_flags):
        return ""
    if int(crop_diagnostics.get("detected_figure_count") or 0) > 0:
        return ""
    return "detection_failure"


def _write_missing_image_repair_report(records: list[QuestionRecord], config: AppConfig) -> Path:
    report = build_missing_image_repair_report(records)
    output_path = config.output.root_dir() / "audits" / "missing_image_repair_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return write_atomic_json(report, output_path, sort_keys=True)


def build_missing_image_repair_report(records: list[QuestionRecord]) -> dict[str, Any]:
    initial_missing = sum(1 for record in records if _record_expected_figure(record) and not _record_has_detected_figure(record))
    fallback_records = [
        record
        for record in records
        if _record_has_detected_figure(record)
        and (
            "question_context_figure_inference_used" in set(record.review_flags)
            or "ocr_hint_figure_regions" in _layout_warning_text(record)
            or _diagnostic_flag(record, "question_context_figure_inference_used")
        )
    ]
    remaining_missing = sum(1 for record in records if _record_missing_image_reason(record) == "detection_failure")
    method_breakdown: dict[str, int] = {
        "embedded_or_vector_graphic_regions": sum(1 for record in records if _record_has_detected_figure(record)),
        "question_context_inference": sum(1 for record in records if _diagnostic_flag(record, "question_context_figure_inference_used")),
        "ocr_hint_signals": sum(1 for record in records if _diagnostic_flag(record, "ocr_hint_figure_regions")),
        "legacy_fallback": sum(1 for record in fallback_records if _is_legacy_record(record)),
    }
    legacy_records = [record for record in records if _is_legacy_record(record)]
    modern_records = [record for record in records if not _is_legacy_record(record)]
    return {
        "schema_name": "exam_bank.missing_image_repair_report",
        "schema_version": 1,
        "initial_missing_images": initial_missing + len(fallback_records),
        "final_missing_images": remaining_missing,
        "detected_images_added_via_fallback": len(fallback_records),
        "detection_success_rate_improvement": round(len(fallback_records) / max(1, initial_missing + len(fallback_records)), 6),
        "detection_method_breakdown": method_breakdown,
        "legacy_vs_modern_breakdown": {
            "legacy": _missing_image_breakdown(legacy_records),
            "modern": _missing_image_breakdown(modern_records),
        },
        "visual_coverage_acceptable": remaining_missing == 0,
        "remaining_missing_images": [
            {
                "paper": record.paper_name,
                "question_number": record.question_number,
                "reason": _record_missing_image_reason(record),
            }
            for record in records
            if _record_missing_image_reason(record)
        ],
    }


def _missing_image_breakdown(records: list[QuestionRecord]) -> dict[str, int]:
    expected = sum(1 for record in records if _record_expected_figure(record))
    detected = sum(1 for record in records if _record_expected_figure(record) and _record_has_detected_figure(record))
    missing = sum(1 for record in records if _record_missing_image_reason(record) == "detection_failure")
    fallback = sum(1 for record in records if _diagnostic_flag(record, "question_context_figure_inference_used"))
    return {
        "expected_figure_questions": expected,
        "detected_figure_questions": detected,
        "fallback_detected_questions": fallback,
        "remaining_missing_images": missing,
        "improvement": fallback,
    }


def _record_expected_figure(record: QuestionRecord) -> bool:
    return bool(record.visual_required and set(record.visual_reason_flags) & {"contains_graph_or_diagram_prompt", "contains_table_or_data_prompt", "contains_inequality_or_region_prompt"})


def _record_has_detected_figure(record: QuestionRecord) -> bool:
    return int(record.question_crop_diagnostics.get("detected_figure_count") or 0) > 0


def _record_missing_image_reason(record: QuestionRecord) -> str:
    return str(record.question_crop_diagnostics.get("missing_image_reason") or "")


def _diagnostic_flag(record: QuestionRecord, flag: str) -> bool:
    return flag in set(record.review_flags) or flag in set(record.question_crop_diagnostics.get("flags") or [])


def _layout_warning_text(record: QuestionRecord) -> str:
    return " ".join(str(flag) for flag in record.question_crop_diagnostics.get("flags") or [])


def _is_legacy_record(record: QuestionRecord) -> bool:
    try:
        return int(str(record.year)[-2:]) < 17
    except (TypeError, ValueError):
        return bool(re.search(r"_(?:[msw])(?:0\d|1[0-6])_", str(record.source_pdf).lower()))


def _record_confidence(classification_confidence: float, flags: list[str]) -> float:
    penalty = min(0.45, len(set(flags)) * 0.04)
    return max(0.05, min(0.98, classification_confidence - penalty))


def _question_topic_from_parts(
    classification: ClassificationResult,
    part_level_topics: list[dict[str, object]],
) -> dict[str, object]:
    review_flags = list(classification.review_flags)
    part_topics = []
    for part in part_level_topics:
        part_topic = str(part.get("topic", ""))
        if not part_topic or part_topic == classification.topic:
            continue
        if bool(part.get("topic_uncertain")) or str(part.get("topic_confidence", "")) == "low":
            continue
        part_topics.append(part_topic)
    secondary_topics = []
    for topic in part_topics:
        if topic not in secondary_topics:
            secondary_topics.append(topic)
    topic_confidence = classification.topic_confidence
    topic_uncertain = classification.topic_uncertain
    confidence = classification.confidence
    paper_family = classification.paper_family

    part_families = sorted(
        {
            str(part.get("paper_family", ""))
            for part in part_level_topics
            if part.get("paper_family") and part.get("paper_family") != "unknown"
        }
    )

    if len(part_families) == 1 and paper_family == "unknown":
        paper_family = part_families[0]
    elif len(part_families) > 1:
        paper_family = "unknown"
        review_flags.append("paper_family_uncertain")
    elif paper_family == "unknown":
        review_flags.append("paper_family_uncertain")

    if any(part.get("topic_uncertain") or part.get("topic_confidence") == "low" for part in part_level_topics):
        review_flags.append("part_topic_uncertain")
    if secondary_topics:
        review_flags.append("mixed_topic_possible")
    if len(secondary_topics) >= 2 and classification.topic_confidence != "high":
        review_flags.append("topic_uncertain_mixed_major_topics")
        topic_uncertain = True

    return {
        "paper_family": paper_family,
        "topic": classification.topic,
        "subtopic": classification.subtopic,
        "topic_confidence": topic_confidence,
        "topic_uncertain": topic_uncertain,
        "secondary_topics": secondary_topics,
        "review_flags": sorted(set(review_flags)),
        "confidence": confidence,
    }


def _secondary_main_topics(labels: list[str], primary_topic: str) -> list[str]:
    topics: list[str] = []
    for label in labels:
        topic = str(label).split(":", 1)[0]
        if topic and topic != primary_topic and topic not in topics:
            topics.append(topic)
    return topics


def _clear_resolved_mixed_topic_flags(flags: list[str]) -> list[str]:
    cleaned = [flag for flag in flags if flag != "topic_uncertain_mixed_major_topics"]
    remaining_topic_uncertainty = any(flag.startswith("topic_uncertain_") for flag in cleaned)
    if not remaining_topic_uncertainty:
        cleaned = [flag for flag in cleaned if flag != "topic_uncertain"]
    return cleaned


def _topic_uncertain_from_flags(flags: list[str]) -> bool:
    return "topic_uncertain" in flags or any(flag.startswith("topic_uncertain_") for flag in flags)


def _reconcile_paper_topics(records: list[QuestionRecord], config: AppConfig) -> None:
    if not records:
        return
    paper_family = records[0].paper_family
    if paper_family not in {"P1", "P3", "P4", "P5"}:
        return

    allowed_topics = set(config.paper_family_taxonomy.get(paper_family, {}))
    coverage = _paper_topic_coverage_summary(records, allowed_topics)
    missing_topics = sorted(topic for topic, counts in coverage.items() if counts["high"] == 0 and counts["medium"] == 0)

    for record in records:
        candidate_topics = _candidate_topics_for_reconciliation(record, allowed_topics)
        if not _record_is_reconciliation_candidate(record, candidate_topics):
            record.paper_repair_considered = False
            record.paper_repair_changed_topic = False
            record.paper_repair_candidates = []
            record.paper_repair_missing_topics = missing_topics
            record.paper_repair_reason = ""
            record.paper_repair_note = _paper_repair_note(missing_topics, changed=False, considered=False)
            record.paper_repair_supporting_evidence = {}
            continue

        record.paper_repair_considered = True
        record.paper_repair_candidates = candidate_topics
        record.paper_repair_missing_topics = missing_topics
        if len(candidate_topics) <= 1:
            record.paper_repair_changed_topic = False
            record.paper_repair_reason = ""
            record.paper_repair_note = _paper_repair_note(missing_topics, changed=False, considered=True)
            record.paper_repair_supporting_evidence = {
                "paper_family": paper_family,
                "coverage_summary": coverage,
                "eligible_for_repair": True,
                "candidate_topics": candidate_topics,
                "decision": "insufficient_local_alternatives",
            }
            continue

        current_scores = _reconciliation_topic_scores(record, record.topic, paper_family, missing_topics, coverage)
        best_topic = record.topic
        best_scores = current_scores
        all_scores = [current_scores]
        for topic in candidate_topics:
            scores = _reconciliation_topic_scores(record, topic, paper_family, missing_topics, coverage)
            all_scores.append(scores)
            if _is_better_reconciliation_candidate(scores, best_scores):
                best_topic = topic
                best_scores = scores

        record.paper_repair_supporting_evidence = {
            "paper_family": paper_family,
            "coverage_summary": coverage,
            "missing_topics": missing_topics,
            "eligible_for_repair": True,
            "candidate_topics": candidate_topics,
            "current_topic_scores": current_scores,
            "repair_candidate_scores": {item["topic"]: item for item in all_scores},
            "selected_topic": best_topic,
        }

        if best_topic == record.topic:
            record.reconciliation_changed_topic = False
            record.reconciliation_reason = ""
            record.reconciliation_note = _reconciliation_note(record, missing_topics, changed=False)
            record.paper_repair_changed_topic = False
            record.paper_repair_reason = ""
            record.paper_repair_note = _paper_repair_note(missing_topics, changed=False, considered=True)
            record.paper_repair_from_topic = record.topic
            record.paper_repair_to_topic = record.topic
            continue

        previous_topic = record.topic
        record.topic = best_topic
        record.question_level_topic = best_topic
        record.subtopic = "general"
        record.question_level_subtopic = "general"
        record.topic_confidence = "medium" if best_scores["local_support"] >= 3 else "low"
        record.topic_uncertain = record.topic_confidence == "low"
        record.confidence = min(0.72, max(record.confidence, 0.58)) if record.topic_confidence == "medium" else min(record.confidence, 0.5)
        record.review_flags = _update_reconciliation_flags(record.review_flags, record.topic_uncertain)
        record.topic_alternatives = [f"{paper_family}:{previous_topic}:general"] if previous_topic != best_topic else record.topic_alternatives[:1]
        record.secondary_topics = [previous_topic] if previous_topic != best_topic and previous_topic in allowed_topics else []
        record.reconciliation_changed_topic = True
        record.reconciliation_reason = (
            f"paper-level reconciliation reranked `{previous_topic}` to `{best_topic}` because `{best_topic}` "
            f"had genuine local support and better fit missing paper coverage"
        )
        record.reconciliation_note = _reconciliation_note(record, missing_topics, changed=True)
        record.paper_repair_changed_topic = True
        record.paper_repair_reason = (
            f"paper-level fallback repair reranked `{previous_topic}` to `{best_topic}` because the local label was weak "
            f"and `{best_topic}` had stronger local plausibility plus better missing-topic fit"
        )
        record.paper_repair_note = _paper_repair_note(missing_topics, changed=True, considered=True)
        record.paper_repair_from_topic = previous_topic
        record.paper_repair_to_topic = best_topic


def _record_is_reconciliation_candidate(record: QuestionRecord, candidate_topics: list[str]) -> bool:
    trigger_flags = {
        "low_classification_confidence",
        "topic_forced_no_rule_match",
        "topic_forced_low_confidence",
        "mixed_topic_possible",
        "weak_question_text",
        "weak_markscheme_signal",
        "likely_needs_visual_review",
        "part_topic_continuity_applied",
        "object_cue_conflict_with_method_scoring",
    }
    details = record.topic_evidence_details or {}
    score_breakdown = details.get("topic_score_breakdown", {})
    current_breakdown = score_breakdown.get(record.topic, {})
    extraction_weak = record.extraction_quality_score < 0.68 or "likely_needs_visual_review" in record.extraction_quality_flags
    meaningful_alternatives = [topic for topic in candidate_topics if topic and topic != record.topic]
    has_meaningful_alternative = bool(meaningful_alternatives)
    weak_signal = bool(
        record.topic_confidence != "high"
        or record.topic_uncertain
        or any(flag in trigger_flags for flag in record.review_flags)
        or any("uncertain" in flag for flag in record.review_flags)
        or bool(details.get("object_cue_conflict_with_method_scoring"))
        or bool(current_breakdown.get("object_protection_penalty"))
        or (
            extraction_weak
            and (
                record.topic_confidence != "high"
                or record.topic_uncertain
                or bool(details.get("object_cue_conflict_with_method_scoring"))
                or len(meaningful_alternatives) > 0
                or bool(record.secondary_topics)
                or _has_meaningful_part_tension(record)
                or _score_breakdown_is_close(record.topic, score_breakdown)
            )
        )
    )

    if _is_protected_local_win(record, candidate_topics):
        return False

    if not weak_signal:
        return False
    if not has_meaningful_alternative:
        return False
    return True


def _candidate_topics_for_reconciliation(record: QuestionRecord, allowed_topics: set[str]) -> list[str]:
    candidates: list[str] = []
    if record.topic in allowed_topics:
        candidates.append(record.topic)
    for topic in _topics_from_alternatives(record.topic_alternatives):
        if topic in allowed_topics and topic not in candidates:
            candidates.append(topic)
    for part in record.part_level_topics:
        topic = str(part.get("topic", ""))
        if topic in allowed_topics and str(part.get("topic_confidence", "")) in {"medium", "high"} and topic not in candidates:
            candidates.append(topic)
    details = record.topic_evidence_details or {}
    object_topic = str(details.get("object_cue_primary_topic", ""))
    if object_topic in allowed_topics and object_topic not in candidates:
        candidates.append(object_topic)
    for topic in record.secondary_topics:
        if topic in allowed_topics and topic not in candidates:
            candidates.append(topic)
    for topic in _close_runner_up_topics(details.get("topic_score_breakdown", {}), record.topic):
        if topic in allowed_topics and topic not in candidates:
            candidates.append(topic)
    return candidates[:5]


def _topics_from_alternatives(alternatives: list[str]) -> list[str]:
    topics: list[str] = []
    for label in alternatives:
        parts = str(label).split(":")
        if len(parts) >= 2 and parts[1] and parts[1] not in topics:
            topics.append(parts[1])
    return topics


def _reconciliation_topic_scores(
    record: QuestionRecord,
    topic: str,
    paper_family: str,
    missing_topics: list[str],
    coverage: dict[str, dict[str, int]],
) -> dict[str, float]:
    question_text = (record.body_text_normalized or record.combined_question_text).lower()
    markscheme_text = record.answer_text.lower()
    details = record.topic_evidence_details or {}
    object_scores = details.get("object_cue_topic_scores", {})
    object_primary = str(details.get("object_cue_primary_topic", ""))
    score_breakdown = details.get("topic_score_breakdown", {})
    current_breakdown = score_breakdown.get(record.topic, {})
    candidate_breakdown = score_breakdown.get(topic, {})
    explicit_question = 1.0 if _explicit_primary_topic_from_text(question_text, paper_family) == topic else 0.0
    explicit_markscheme = 1.0 if markscheme_text and _explicit_primary_topic_from_text(markscheme_text, paper_family) == topic else 0.0
    part_support = sum(
        1.0
        for part in record.part_level_topics
        if str(part.get("topic", "")) == topic and str(part.get("topic_confidence", "")) in {"medium", "high"}
    )
    alternative_support = 1.0 if topic in _topics_from_alternatives(record.topic_alternatives) else 0.0
    current_bonus = 0.8 if topic == record.topic else 0.0
    object_support = min(3.0, float(object_scores.get(topic, 0.0)) / 6.0)
    object_alignment = 1.5 if object_primary == topic and topic != record.topic else 0.0
    secondary_support = 0.8 if topic in record.secondary_topics else 0.0
    close_runner_bonus = _close_score_bonus(current_breakdown, candidate_breakdown, topic != record.topic)
    extraction_bonus = 1.2 if (record.extraction_quality_score < 0.68 or "likely_needs_visual_review" in record.extraction_quality_flags) else 0.0
    drift_bonus = 1.2 if _looks_like_incidental_algebra_drift(record, topic) else 0.0
    paper_fit = _paper_repair_bonus(topic, missing_topics, coverage)
    local_support = (
        explicit_question * 3.0
        + explicit_markscheme * 1.8
        + part_support * 1.4
        + alternative_support
        + current_bonus
        + object_support
        + object_alignment
        + secondary_support
        + close_runner_bonus
    )
    repair_bonus = paper_fit + extraction_bonus + drift_bonus
    return {
        "topic": topic,
        "local_support": local_support,
        "paper_fit": paper_fit,
        "repair_bonus": repair_bonus,
        "total": local_support + repair_bonus,
        "explicit_question": explicit_question,
        "explicit_markscheme": explicit_markscheme,
        "object_support": object_support,
        "object_alignment": object_alignment,
        "close_runner_bonus": close_runner_bonus,
        "extraction_bonus": extraction_bonus,
        "drift_bonus": drift_bonus,
        "current_final_score": float(current_breakdown.get("final_score", 0.0)),
        "candidate_final_score": float(candidate_breakdown.get("final_score", 0.0)),
    }


def _is_better_reconciliation_candidate(candidate: dict[str, float], current: dict[str, float]) -> bool:
    if candidate["topic"] == current["topic"]:
        return False
    if candidate["local_support"] < 2.6:
        return False
    if candidate["repair_bonus"] < 0.9:
        return False
    if candidate["total"] < current["total"] + 1.25:
        return False
    if candidate["local_support"] + 0.5 < current["local_support"]:
        return False
    return True


def _update_reconciliation_flags(flags: list[str], topic_uncertain: bool) -> list[str]:
    cleaned = [
        flag
        for flag in flags
        if flag not in {"mixed_topic_possible", "topic_forced_no_rule_match", "topic_forced_low_confidence"}
    ]
    cleaned.append("paper_level_topic_reconciled")
    if not topic_uncertain:
        cleaned = [flag for flag in cleaned if flag != "low_classification_confidence"]
    return sorted(set(cleaned))


def _reconciliation_note(record: QuestionRecord, missing_topics: list[str], changed: bool) -> str:
    if changed:
        return f"soft paper-level coverage prior considered missing topics {missing_topics} and reranked this weak label"
    if missing_topics:
        return f"soft paper-level coverage prior considered missing topics {missing_topics} but local evidence remained stronger"
    return "soft paper-level coverage prior found no meaningful missing-topic repair"


def _paper_repair_note(missing_topics: list[str], changed: bool, considered: bool) -> str:
    if not considered:
        return "paper-level fallback repair did not consider this question because the local label was protected"
    if changed:
        return f"paper-level fallback repair used missing-topic pressure from {missing_topics} to rerank a weak label"
    if missing_topics:
        return f"paper-level fallback repair considered missing topics {missing_topics} but did not find enough local support"
    return "paper-level fallback repair found no underrepresented topics worth using"


def _paper_topic_coverage_summary(records: list[QuestionRecord], allowed_topics: set[str]) -> dict[str, dict[str, int]]:
    coverage = {topic: {"high": 0, "medium": 0, "weak": 0} for topic in sorted(allowed_topics)}
    for record in records:
        topic = record.topic
        if topic not in coverage:
            continue
        bucket = "weak"
        if record.topic_confidence == "high" and not record.topic_uncertain:
            bucket = "high"
        elif record.topic_confidence == "medium" and not record.topic_uncertain:
            bucket = "medium"
        coverage[topic][bucket] += 1
    return coverage


def _close_runner_up_topics(score_breakdown: dict[str, dict[str, Any]], current_topic: str) -> list[str]:
    current_score = float(score_breakdown.get(current_topic, {}).get("final_score", 0.0))
    close_topics: list[str] = []
    for topic, details in sorted(score_breakdown.items(), key=lambda item: float(item[1].get("final_score", 0.0)), reverse=True):
        if topic == current_topic:
            continue
        score = float(details.get("final_score", 0.0))
        if current_score and current_score - score > 6.5:
            continue
        if score <= 0:
            continue
        close_topics.append(topic)
        if len(close_topics) >= 2:
            break
    return close_topics


def _score_gap_is_clear_winner(current_topic: str, score_breakdown: dict[str, dict[str, Any]]) -> bool:
    if current_topic not in score_breakdown:
        return False
    ordered = sorted((float(details.get("final_score", 0.0)), topic) for topic, details in score_breakdown.items())
    if len(ordered) < 2:
        return True
    top_score, top_topic = ordered[-1]
    runner_up_score = ordered[-2][0]
    return top_topic == current_topic and top_score - runner_up_score >= 8.0


def _score_breakdown_is_close(current_topic: str, score_breakdown: dict[str, dict[str, Any]]) -> bool:
    if current_topic not in score_breakdown or len(score_breakdown) < 2:
        return False
    ordered = sorted((float(details.get("final_score", 0.0)), topic) for topic, details in score_breakdown.items())
    top_score, top_topic = ordered[-1]
    runner_up_score = ordered[-2][0]
    return top_topic == current_topic and top_score - runner_up_score <= 6.5


def _has_meaningful_part_tension(record: QuestionRecord) -> bool:
    for part in record.part_level_topics:
        part_topic = str(part.get("topic", ""))
        if not part_topic or part_topic == record.topic:
            continue
        if str(part.get("topic_confidence", "")) in {"medium", "high"} and not bool(part.get("topic_uncertain")):
            return True
    return False


def _is_protected_local_win(record: QuestionRecord, candidate_topics: list[str]) -> bool:
    details = record.topic_evidence_details or {}
    score_breakdown = details.get("topic_score_breakdown", {})
    object_primary = str(details.get("object_cue_primary_topic", ""))
    object_scores = details.get("object_cue_topic_scores", {})
    has_object_conflict = bool(details.get("object_cue_conflict_with_method_scoring"))
    meaningful_alternatives = [topic for topic in candidate_topics if topic and topic != record.topic]
    strong_object_anchor = object_primary == record.topic and float(object_scores.get(record.topic, 0.0)) >= 8.0
    clear_local_win = _score_gap_is_clear_winner(record.topic, score_breakdown)
    no_part_tension = not _has_meaningful_part_tension(record)
    return bool(
        record.topic_confidence == "high"
        and not record.topic_uncertain
        and strong_object_anchor
        and clear_local_win
        and not has_object_conflict
        and not meaningful_alternatives
        and no_part_tension
    )


def _close_score_bonus(
    current_breakdown: dict[str, Any],
    candidate_breakdown: dict[str, Any],
    is_alternative: bool,
) -> float:
    if not is_alternative:
        return 0.0
    current_score = float(current_breakdown.get("final_score", 0.0))
    candidate_score = float(candidate_breakdown.get("final_score", 0.0))
    if candidate_score <= 0:
        return 0.0
    gap = current_score - candidate_score
    if gap <= 3.0:
        return 1.4
    if gap <= 7.0:
        return 0.8
    return 0.0


def _paper_repair_bonus(topic: str, missing_topics: list[str], coverage: dict[str, dict[str, int]]) -> float:
    if topic in missing_topics:
        return 1.5
    counts = coverage.get(topic, {})
    if counts and counts.get("high", 0) == 0 and counts.get("medium", 0) == 0 and counts.get("weak", 0) > 0:
        return 0.7
    return 0.0


def _looks_like_incidental_algebra_drift(record: QuestionRecord, alternative_topic: str) -> bool:
    details = record.topic_evidence_details or {}
    object_primary = str(details.get("object_cue_primary_topic", ""))
    if record.topic not in {"algebra", "quadratics", "polynomials"}:
        return False
    if alternative_topic != object_primary:
        return False
    return float(details.get("object_cue_topic_scores", {}).get(alternative_topic, 0.0)) >= 6.0


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _safe_basename(stem: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in stem).strip("_") or "paper"


def _write_pdf_diagnostic(
    question_pdf: Path,
    layouts: list[PageLayout],
    spans: list[QuestionSpan],
    records: list[QuestionRecord],
    config: AppConfig,
) -> Path:
    config.ensure_output_dirs()
    paper_name = _safe_basename(question_pdf.stem)
    anchors = detect_question_anchor_candidates(layouts, config)
    uncertain_records = [
        record
        for record in records
        if record.crop_uncertain
        or any("uncertain" in flag or "contamination" in flag or "sequence_gap" in flag for flag in record.review_flags)
    ]
    ocr_pages = [
        layout.page_number
        for layout in layouts
        if layout.text_source == "ocr" or str(layout.extraction_warning or "").startswith("ocr")
    ]
    footer_contamination = [
        record.question_number
        for record in records
        if any("header_footer_contamination" in flag or "crop_reaches_page_margin" in flag for flag in record.review_flags)
    ]
    payload = {
        "source_pdf": _display_path(question_pdf),
        "paper_name": paper_name,
        "detected_top_level_questions": len(records),
        "detected_question_numbers": [record.question_number for record in records],
        "candidate_question_anchors": len(anchors),
        "accepted_question_anchors": len(spans),
        "uncertain_splits": len(uncertain_records),
        "ocr_fallback_pages": len(ocr_pages),
        "ocr_page_numbers": ocr_pages,
        "footer_header_contamination_count": len(footer_contamination),
        "footer_header_contamination_questions": footer_contamination,
        "crop_uncertain_count": sum(1 for record in records if record.crop_uncertain),
        "topic_counts_by_paper_family": _topic_counts_by_paper_family(records),
        "difficulty_counts_by_paper_family": _difficulty_counts_by_paper_family(records),
        "markscheme_image_count": sum(1 for record in records if record.markscheme_image),
        "markscheme_image_missing_count": sum(1 for record in records if "markscheme_image_missing" in record.review_flags),
        "review_flag_counts": _flag_counts(records),
    }
    path = config.output.debug_dir / f"{paper_name}_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_batch_diagnostic(records: list[QuestionRecord], config: AppConfig, basename: str | None = None) -> Path:
    config.ensure_output_dirs()
    name = f"{basename}_diagnostics.json" if basename else "batch_diagnostics.json"
    payload = {
        "record_count": len(records),
        "paper_family_counts": _paper_family_counts(records),
        "topic_counts_by_paper_family": _topic_counts_by_paper_family(records),
        "difficulty_counts_by_paper_family": _difficulty_counts_by_paper_family(records),
        "markscheme_image_count": sum(1 for record in records if record.markscheme_image),
        "markscheme_image_missing_count": sum(1 for record in records if "markscheme_image_missing" in record.review_flags),
        "review_flag_counts": _flag_counts(records),
    }
    path = config.output.debug_dir / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _paper_family_counts(records: list[QuestionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        family = record.paper_family or "unknown"
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _topic_counts_by_paper_family(records: list[QuestionRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        family = record.paper_family or "unknown"
        topic = record.question_level_topic or record.topic or "unknown"
        family_counts = counts.setdefault(family, {})
        family_counts[topic] = family_counts.get(topic, 0) + 1
    return {family: dict(sorted(topic_counts.items())) for family, topic_counts in sorted(counts.items())}


def _difficulty_counts_by_paper_family(records: list[QuestionRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        family = record.paper_family or "unknown"
        difficulty = record.difficulty or "unknown"
        family_counts = counts.setdefault(family, {})
        family_counts[difficulty] = family_counts.get(difficulty, 0) + 1
    return {family: dict(sorted(difficulty_counts.items())) for family, difficulty_counts in sorted(counts.items())}


def _flag_counts(records: list[QuestionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for flag in record.review_flags:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def _write_topic_debug_report(question_pdf: Path, records: list[QuestionRecord], config: AppConfig) -> Path:
    config.ensure_output_dirs()
    paper_name = _safe_basename(question_pdf.stem)
    allowed_topics = set(config.paper_family_taxonomy.get(records[0].paper_family, {})) if records else set()
    payload = {
        "source_pdf": _display_path(question_pdf),
        "paper_name": paper_name,
        "paper_repair_summary": _paper_repair_debug_summary(records, allowed_topics),
        "questions": [
            {
                "question_number": record.question_number,
                "text_snippet": record.combined_question_text[:500],
                "body_text_raw": record.body_text_raw[:500],
                "body_text_normalized": record.body_text_normalized[:500],
                "math_lines": record.math_lines,
                "diagram_text": record.diagram_text,
                "extraction_quality_score": record.extraction_quality_score,
                "extraction_quality_flags": record.extraction_quality_flags,
                "paper_family": record.paper_family,
                "source_paper_family": record.source_paper_family,
                "inferred_paper_family": record.inferred_paper_family,
                "paper_family_confidence": record.paper_family_confidence,
                "question_level_paper_family": record.question_level_paper_family or record.paper_family,
                "question_level_topic": record.question_level_topic or record.topic,
                "question_level_subtopic": record.question_level_subtopic or record.subtopic,
                "topic": record.topic,
                "subtopic": record.subtopic,
                "topic_confidence": record.topic_confidence,
                "record_confidence": record.confidence,
                "topic_uncertain": record.topic_uncertain,
                "topic_evidence": record.topic_evidence,
                "detected_object_cues": record.topic_evidence_details.get("detected_object_cues", []),
                "object_cue_topic_scores": record.topic_evidence_details.get("object_cue_topic_scores", {}),
                "object_cue_source_topic_scores": record.topic_evidence_details.get("object_cue_source_topic_scores", {}),
                "object_cue_primary_topic": record.topic_evidence_details.get("object_cue_primary_topic", ""),
                "object_cue_conflict_with_method_scoring": record.topic_evidence_details.get(
                    "object_cue_conflict_with_method_scoring", False
                ),
                "object_cue_protection_applied": record.topic_evidence_details.get("object_cue_protection_applied", False),
                "object_cue_protection_topics": record.topic_evidence_details.get("object_cue_protection_topics", []),
                "object_cue_resisted_override": record.topic_evidence_details.get("object_cue_resisted_override", False),
                "source_method_stage_top_topic": record.topic_evidence_details.get("source_method_stage_top_topic", ""),
                "source_method_stage_top_score": record.topic_evidence_details.get("source_method_stage_top_score", 0),
                "object_cue_override_stage": record.topic_evidence_details.get("object_cue_override_stage", ""),
                "object_cue_override_topic": record.topic_evidence_details.get("object_cue_override_topic", ""),
                "topic_score_breakdown": record.topic_evidence_details.get("topic_score_breakdown", {}),
                "secondary_topics": record.secondary_topics,
                "part_level_topics": record.part_level_topics,
                "alternative_candidate_topics": record.topic_alternatives if record.topic_confidence != "high" else [],
                "difficulty": record.difficulty,
                "difficulty_confidence": record.difficulty_confidence,
                "difficulty_evidence": record.difficulty_evidence,
                "difficulty_uncertain": record.difficulty_uncertain,
                "reconciliation_changed_topic": record.reconciliation_changed_topic,
                "reconciliation_reason": record.reconciliation_reason,
                "reconciliation_note": record.reconciliation_note,
                "paper_repair_considered": record.paper_repair_considered,
                "paper_repair_changed_topic": record.paper_repair_changed_topic,
                "paper_repair_reason": record.paper_repair_reason,
                "paper_repair_note": record.paper_repair_note,
                "paper_repair_from_topic": record.paper_repair_from_topic,
                "paper_repair_to_topic": record.paper_repair_to_topic,
                "paper_repair_candidates": record.paper_repair_candidates,
                "paper_repair_missing_topics": record.paper_repair_missing_topics,
                "paper_repair_supporting_evidence": record.paper_repair_supporting_evidence,
                "markscheme_image_found": bool(record.markscheme_image),
                "markscheme_question_number": record.markscheme_question_number,
                "markscheme_crop_confidence": record.markscheme_crop_confidence,
                "markscheme_mapping_method": record.markscheme_mapping_method,
                "markscheme_table_detected": record.markscheme_table_detected,
                "classification_restricted_by_paper_family": record.paper_family not in {"", "unknown"},
            }
            for record in records
        ],
    }
    path = config.output.debug_dir / f"{paper_name}_topic_debug.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _paper_repair_debug_summary(records: list[QuestionRecord], allowed_topics: set[str]) -> dict[str, Any]:
    if not records:
        return {}
    coverage = _paper_topic_coverage_summary(records, allowed_topics)
    missing_topics = sorted(topic for topic, counts in coverage.items() if counts["high"] == 0 and counts["medium"] == 0)
    eligible_questions = [record.question_number for record in records if record.paper_repair_considered]
    changed_questions = [record.question_number for record in records if record.paper_repair_changed_topic]
    return {
        "paper_family": records[0].paper_family,
        "topic_coverage_summary": coverage,
        "missing_or_underrepresented_topics": missing_topics,
        "repair_eligible_questions": eligible_questions,
        "repair_changed_questions": changed_questions,
    }
