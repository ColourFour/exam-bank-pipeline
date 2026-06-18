from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from . import __version__
from .atomic_json import write_atomic_json
from .config import AppConfig
from .core.paper_identity import paper_identity_from_parts
from .models import QuestionRecord
from .ocr import OCR_ENGINE
from .output_layout import (
    CANONICAL_SUBJECTS,
    component_code_from_values,
    OUTPUT_LAYOUT_VERSION,
    output_profile_for_root,
)
from .output_structure_normalization import legacy_image_path_to_canonical
from .trust import CropConfidence


QUESTION_BANK_SCHEMA_NAME = "exam_bank.question_bank"
QUESTION_BANK_SCHEMA_VERSION = 2
QUESTION_BANK_RUN_MANIFEST_VERSION = 1


def export_records(records: list[QuestionRecord], config: AppConfig, basename: str | None = None) -> Path:
    config.ensure_output_dirs()
    json_name = f"{basename}.json" if basename else config.naming.json_name
    json_path = config.output.json_dir / json_name
    write_json(records, json_path, output_root=config.output.root_dir(), config=config)
    return json_path


def records_to_output_questions(records: list[QuestionRecord], output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(output_root) if output_root is not None else None
    return [_record_to_output_dict(record, root) for record in records]


def write_question_bank_payload(
    question_payload: list[dict[str, Any]],
    output_path: str | Path,
    *,
    run_manifest: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    write_atomic_json(
        {
            "schema_name": QUESTION_BANK_SCHEMA_NAME,
            "schema_version": QUESTION_BANK_SCHEMA_VERSION,
            "record_count": len(question_payload),
            "run_manifest": run_manifest or _build_payload_run_manifest(question_payload, output_path=output_path),
            "questions": question_payload,
        },
        output_path,
    )
    return output_path


def write_json(
    records: list[QuestionRecord],
    output_path: str | Path,
    *,
    output_root: str | Path | None = None,
    config: AppConfig | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(output_root) if output_root is not None else None
    question_payload = records_to_output_questions(records, root)
    run_manifest = _build_run_manifest(
        records,
        question_payload,
        output_path=output_path,
        output_root=root,
        config=config,
    )
    return write_question_bank_payload(question_payload, output_path, run_manifest=run_manifest)


def _build_run_manifest(
    records: list[QuestionRecord],
    question_payload: list[dict[str, Any]],
    *,
    output_path: Path,
    output_root: Path | None,
    config: AppConfig | None,
) -> dict[str, Any]:
    generated_at_dt = datetime.now(timezone.utc)
    generated_at = generated_at_dt.isoformat()
    input_manifest_sha256 = _input_manifest_sha256(records)
    return {
        "schema_version": QUESTION_BANK_RUN_MANIFEST_VERSION,
        "generated_at": generated_at,
        "run_id": f"{generated_at_dt.strftime('%Y%m%dT%H%M%SZ')}-{input_manifest_sha256[:12]}",
        "pipeline_version": __version__,
        "git_commit": _git_commit(),
        "model_versions": _model_versions(records, config),
        "ocr_engine_version": _ocr_engine_version(records, config),
        "input_manifest_sha256": input_manifest_sha256,
        "artifact_root": _artifact_root_value(output_root, output_path),
        "output_layout": {
            "version": OUTPUT_LAYOUT_VERSION,
            "profile": output_profile_for_root(output_root or output_path.parent.parent),
        },
        "qa_summary": _qa_summary(records, question_payload),
    }


def _input_manifest_sha256(records: list[QuestionRecord]) -> str:
    manifest = {
        "schema_version": 1,
        "sources": _input_sources(records),
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_sources(records: list[QuestionRecord]) -> list[dict[str, Any]]:
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for role, path_value in [
            ("question_paper", record.extraction.source_pdf),
            ("mark_scheme", record.mark_scheme.source_pdf),
        ]:
            normalized_path = str(path_value or "").strip()
            if not normalized_path:
                continue
            key = (role, normalized_path)
            if key in sources:
                continue
            path = Path(normalized_path)
            exists = path.is_file()
            sources[key] = {
                "role": role,
                "path": normalized_path,
                "exists": exists,
                "sha256": _sha256_file(path) if exists else "",
            }
    return [sources[key] for key in sorted(sources)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _model_versions(records: list[QuestionRecord], config: AppConfig | None) -> dict[str, Any]:
    difficulty_versions = sorted(
        {
            record.classification.difficulty_model_version
            for record in records
            if record.classification.difficulty_model_version
        }
    )
    openai_model = ""
    if config is not None and config.classification.enable_openai:
        openai_model = config.classification.openai_model
    return {
        "topic_classifier": "local-topic-heuristics-v1",
        "difficulty_classifier": difficulty_versions,
        "openai_classifier": openai_model,
    }


def _ocr_engine_version(records: list[QuestionRecord], config: AppConfig | None) -> str:
    engines = {
        record.extraction.ocr_engine
        for record in records
        if record.extraction.ocr_engine
    }
    if config is not None and config.ocr.enabled:
        engines.add(OCR_ENGINE)
    if OCR_ENGINE not in engines:
        return ""
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception:
        tesseract_cmd = config.ocr.tesseract_cmd if config is not None and config.ocr.tesseract_cmd else OCR_ENGINE
        try:
            result = subprocess.run(
                [tesseract_cmd, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return "unavailable"
        return (result.stdout.splitlines() or [""])[0].strip() or "unavailable"


def _artifact_root_value(output_root: Path | None, output_path: Path) -> str:
    root = output_root if output_root is not None else output_path.parent.parent
    return str(root)


def _qa_summary(records: list[QuestionRecord], question_payload: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "paper_family_counts": _counts(question.get("paper_family", "") for question in question_payload),
        "validation_status_counts": _counts(record.validation.validation_status for record in records),
        "mapping_status_counts": _counts(record.mark_scheme.mapping_status for record in records),
        "scope_quality_status_counts": _counts(record.validation.scope_quality_status for record in records),
        "text_fidelity_status_counts": _counts(record.validation.text_fidelity_status for record in records),
        "visual_curation_status_counts": _counts(record.validation.visual_curation_status for record in records),
        "text_only_status_counts": _counts(record.validation.text_only_status for record in records),
        "question_crop_confidence_counts": _counts(
            record.images.question_crop_confidence
            or (CropConfidence.LOW if record.images.crop_uncertain else CropConfidence.HIGH)
            for record in records
        ),
        "mark_scheme_crop_confidence_counts": _counts(record.mark_scheme.crop_confidence for record in records),
        "ocr_summary": {
            "ran_count": sum(1 for record in records if record.extraction.ocr_ran),
            "selected_count": sum(1 for record in records if record.extraction.ocr_selected),
            "engine_counts": _counts(record.extraction.ocr_engine for record in records if record.extraction.ocr_engine),
        },
        "artifact_path_counts": {
            "missing_question_image_path": sum(1 for record in records if not record.images.screenshot_path),
            "missing_mark_scheme_image_path": sum(1 for record in records if not record.mark_scheme.image_path),
        },
    }


def _counts(values: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        key = str(value or "unknown")
        counter[key] += 1
    return {key: counter[key] for key in sorted(counter)}


def _build_payload_run_manifest(question_payload: list[dict[str, Any]], *, output_path: Path) -> dict[str, Any]:
    generated_at_dt = datetime.now(timezone.utc)
    generated_at = generated_at_dt.isoformat()
    payload_hash = hashlib.sha256(
        json.dumps(question_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": QUESTION_BANK_RUN_MANIFEST_VERSION,
        "generated_at": generated_at,
        "run_id": f"{generated_at_dt.strftime('%Y%m%dT%H%M%SZ')}-{payload_hash[:12]}",
        "pipeline_version": __version__,
        "git_commit": _git_commit(),
        "model_versions": {},
        "ocr_engine_version": "",
        "input_manifest_sha256": payload_hash,
        "artifact_root": _artifact_root_value(None, output_path),
        "output_layout": {
            "version": OUTPUT_LAYOUT_VERSION,
            "profile": output_profile_for_root(output_path.parent.parent),
        },
        "qa_summary": {
            "record_count": len(question_payload),
            "paper_family_counts": _counts(question.get("paper_family", "") for question in question_payload),
            "validation_status_counts": _counts(
                (question.get("notes") or {}).get("validation_status", "unknown")
                for question in question_payload
                if isinstance(question.get("notes") or {}, dict)
            ),
        },
    }


def _record_to_output_dict(record: QuestionRecord, output_root: Path | None) -> dict[str, Any]:
    extraction = record.extraction
    classification = record.classification
    images = record.images
    mark_scheme = record.mark_scheme
    validation = record.validation
    metadata = record.paper_metadata
    paper_total = record.paper_total

    identity = paper_identity_from_parts(
        syllabus=metadata.syllabus_code or "9709",
        subject_family=classification.paper_family or classification.question_level_paper_family,
        year=metadata.year,
        session=metadata.session,
        component=metadata.component or metadata.source_paper_code,
        question_number=extraction.question_number,
    )
    paper = identity.paper_id
    canonical_session = identity.canonical_session
    canonical_year_folder = str(identity.year)
    canonical_paper_id = identity.paper_id
    family = identity.subject_family
    subparts = list(extraction.question_subparts or mark_scheme.markscheme_subparts)
    question_image_paths = _path_list(images.screenshot_path, output_root)
    mark_scheme_image_paths = _path_list(mark_scheme.image_path, output_root)
    question_image_path = question_image_paths[0] if question_image_paths else ""
    mark_scheme_image_path = mark_scheme_image_paths[0] if mark_scheme_image_paths else ""
    question_solution_marks = _question_solution_marks(record)

    _validate_canonical_artifact_contract(
        paper=canonical_paper_id,
        canonical_session=canonical_session,
        canonical_year_folder=canonical_year_folder,
        canonical_subject=family,
        artifact_paths=[question_image_path, mark_scheme_image_path, *question_image_paths, *mark_scheme_image_paths],
    )

    return {
        "question_id": identity.question_id,
        "paper": paper,
        "canonical_paper_id": canonical_paper_id,
        "canonical_session": canonical_session,
        "canonical_year_folder": canonical_year_folder,
        "paper_family": family,
        "question_number": extraction.question_number,
        "canonical_question_artifact": question_image_path,
        "canonical_mark_scheme_artifact": mark_scheme_image_path,
        "question_image_path": question_image_path,
        "mark_scheme_image_path": mark_scheme_image_path,
        "question_text": extraction.combined_question_text,
        "question_text_role": validation.question_text_role,
        "question_text_trust": validation.question_text_trust,
        "ocr_ran": extraction.ocr_ran,
        "ocr_engine": extraction.ocr_engine,
        "ocr_text": extraction.ocr_text,
        "ocr_text_trust": extraction.ocr_text_trust,
        "ocr_failure_reason": extraction.ocr_failure_reason,
        "visual_required": validation.visual_required,
        "visual_reason_flags": validation.visual_reason_flags,
        "visual_curation_status": validation.visual_curation_status,
        "text_only_status": validation.text_only_status,
        "mark_scheme_text": mark_scheme.answer_text,
        "question_solution_marks": question_solution_marks,
        "difficulty": classification.difficulty,
        "difficulty_score": classification.difficulty_score,
        "difficulty_band": classification.difficulty_band or classification.difficulty,
        "subparts": subparts,
        "subparts_solution_marks": {label: None for label in subparts},
        "question_image_paths": question_image_paths,
        "mark_scheme_image_paths": mark_scheme_image_paths,
        "page_refs": {
            "question": extraction.page_numbers,
            "mark_scheme": mark_scheme.pages,
        },
        "topic": classification.question_level_topic or classification.topic,
        "notes": {
            "subtopic": classification.question_level_subtopic or classification.subtopic,
            "source_pdf": extraction.source_pdf,
            "mark_scheme_source_pdf": mark_scheme.source_pdf,
            "source_paper_code": component_code_from_values(metadata.component, metadata.source_paper_code),
            "full_question_label": extraction.full_question_label,
            "topic_confidence": classification.topic_confidence,
            "topic_uncertain": classification.topic_uncertain,
            "topic_trust_status": validation.topic_trust_status,
            "difficulty": classification.difficulty,
            "difficulty_confidence": classification.difficulty_confidence,
            "difficulty_evidence": classification.difficulty_evidence,
            "difficulty_uncertain": classification.difficulty_uncertain,
            "difficulty_score": classification.difficulty_score,
            "difficulty_score_scale": classification.difficulty_score_scale,
            "difficulty_features": classification.difficulty_features,
            "difficulty_review_flags": classification.difficulty_review_flags,
            "difficulty_model_version": classification.difficulty_model_version,
            "mapping_status": mark_scheme.mapping_status,
            "mapping_failure_reason": mark_scheme.failure_reason,
            "scope_quality_status": validation.scope_quality_status,
            "question_crop_confidence": images.question_crop_confidence or (CropConfidence.LOW if images.crop_uncertain else CropConfidence.HIGH),
            "question_crop_diagnostics": images.question_crop_diagnostics,
            "text_source_profile": validation.text_source_profile,
            "text_fidelity_status": validation.text_fidelity_status,
            "text_fidelity_flags": validation.text_fidelity_flags,
            "question_text_role": validation.question_text_role,
            "question_text_trust": validation.question_text_trust,
            "visual_required": validation.visual_required,
            "visual_reason_flags": validation.visual_reason_flags,
            "visual_curation_status": validation.visual_curation_status,
            "text_only_status": validation.text_only_status,
            "mark_scheme_crop_confidence": mark_scheme.crop_confidence,
            "review_flags": validation.review_flags,
            "extraction_quality_score": round(extraction.extraction_quality_score, 3),
            "extraction_quality_flags": extraction.extraction_quality_flags,
            "validation_status": validation.validation_status,
            "validation_flags": validation.validation_flags,
            "recovery_attempted": extraction.recovery_attempted,
            "recovery_result": extraction.recovery_result,
            "ocr_ran": extraction.ocr_ran,
            "ocr_engine": extraction.ocr_engine,
            "ocr_text_trust": extraction.ocr_text_trust,
            "ocr_failure_reason": extraction.ocr_failure_reason,
            "ocr_text_role": extraction.ocr_text_role,
            "text_candidate_source": extraction.text_candidate_source,
            "native_text_score": extraction.native_text_score,
            "ocr_text_score": extraction.ocr_text_score,
            "selected_text_score": extraction.selected_text_score,
            "text_candidate_decision": extraction.text_candidate_decision,
            "text_candidate_decision_reasons": extraction.text_candidate_decision_reasons,
            "ocr_selected": extraction.ocr_selected,
            "ocr_rejected_reasons": extraction.ocr_rejected_reasons,
            "question_structure_detected": extraction.question_structure_detected,
            "mark_scheme_structure_detected": mark_scheme.structure_detected,
            "question_total_detected": extraction.question_total_detected,
            "mark_scheme_total_detected": mark_scheme.total_detected,
            "question_format_profile": extraction.question_format_profile,
            "paper_total_expected": paper_total.expected,
            "paper_total_detected": paper_total.detected,
            "paper_total_status": paper_total.status,
            "rescan_triggered": paper_total.rescan_triggered,
            "rescan_result": paper_total.rescan_result,
            "paper_total_before_rescan": paper_total.before_rescan,
            "paper_total_after_rescan": paper_total.after_rescan,
            "paper_total_focus_questions": paper_total.focus_questions,
            "paper_total_focus_pages": paper_total.focus_pages,
            "paper_total_focus_reason": paper_total.focus_reason,
        },
    }


def _question_solution_marks(record: QuestionRecord) -> int | None:
    extraction = record.extraction
    mark_scheme = record.mark_scheme
    for value in [mark_scheme.markscheme_marks_total, extraction.question_marks_total, record.marks_if_available, record.marks]:
        if value is not None:
            return int(value)
    return None


def _path_list(path_value: str, output_root: Path | None) -> list[str]:
    if not path_value:
        return []
    return [_relative_output_path(path_value, output_root)]


def _relative_output_path(path_value: str, output_root: Path | None) -> str:
    path = Path(path_value)
    if output_root is None:
        return legacy_image_path_to_canonical(str(path)) or str(path)
    try:
        relative = str(path.relative_to(output_root))
    except ValueError:
        relative = str(path)
    return legacy_image_path_to_canonical(relative) or relative


def _validate_canonical_artifact_contract(
    *,
    paper: str,
    canonical_session: str,
    canonical_year_folder: str,
    canonical_subject: str,
    artifact_paths: list[str],
) -> None:
    expected_year_suffix = canonical_year_folder[-2:]
    if not paper.endswith(canonical_session):
        raise ValueError(f"canonical paper/session mismatch: paper={paper} canonical_session={canonical_session}")
    if not canonical_session.endswith(expected_year_suffix):
        raise ValueError(
            "canonical session/year mismatch: "
            f"canonical_session={canonical_session} canonical_year_folder={canonical_year_folder}"
        )
    for artifact_path in sorted({path for path in artifact_paths if path}):
        subject, year, session, component = _artifact_subject_year_session(artifact_path)
        if subject != canonical_subject:
            raise ValueError(f"artifact path subject mismatch: subject={canonical_subject} artifact_path={artifact_path}")
        if year != canonical_year_folder:
            raise ValueError(f"artifact path year mismatch: year={canonical_year_folder} artifact_path={artifact_path}")
        if session not in _allowed_compact_session_codes(canonical_session, canonical_year_folder):
            raise ValueError(
                "artifact path session mismatch: "
                f"canonical_session={canonical_session} artifact_path={artifact_path}"
            )
        if not paper.startswith(component):
            raise ValueError(f"artifact path component mismatch: paper={paper} artifact_path={artifact_path}")


def _artifact_subject_year_session(artifact_path: str) -> tuple[str, str, str, str]:
    parts = Path(artifact_path).parts
    for index, part in enumerate(parts):
        if part in CANONICAL_SUBJECTS:
            if index != len(parts) - 2:
                raise ValueError(f"artifact path must be directly under a canonical subject folder: {artifact_path}")
            match = re.fullmatch(
                rf"(?P<subject>{part})_(?P<year>\d{{4}})_(?P<session>[msw]\d{{2}})_(?P<component>\d{{2}})_(?:qp|ms)_q\d{{2}}_(?:question|markscheme)(?:_v\d+)?\.png",
                parts[-1],
            )
            if not match:
                raise ValueError(f"artifact path filename does not match canonical schema: {artifact_path}")
            return match.group("subject"), match.group("year"), match.group("session"), match.group("component")
    raise ValueError(f"artifact path is not under a canonical subject folder: {artifact_path}")


def _compact_session_code(canonical_session: str, canonical_year_folder: str) -> str:
    yy = canonical_year_folder[-2:]
    session = canonical_session.lower()
    if session.startswith(("spring", "march", "m")):
        return f"m{yy}"
    if session.startswith(("summer", "june", "s")):
        return f"s{yy}"
    if session.startswith(("autumn", "winter", "nov", "w")):
        return f"w{yy}"
    return f"{session[:1] or 'x'}{yy}"


def _allowed_compact_session_codes(canonical_session: str, canonical_year_folder: str) -> set[str]:
    code = _compact_session_code(canonical_session, canonical_year_folder)
    yy = canonical_year_folder[-2:]
    if code == f"s{yy}":
        return {code, f"m{yy}"}
    return {code}
