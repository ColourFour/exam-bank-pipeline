from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .models import QuestionRecord
from .output_layout import component_code_from_values, paper_family_dir_name, paper_instance_id, question_id
from .trust import CropConfidence


QUESTION_BANK_SCHEMA_NAME = "exam_bank.question_bank"
QUESTION_BANK_SCHEMA_VERSION = 1


def export_records(records: list[QuestionRecord], config: AppConfig, basename: str | None = None) -> Path:
    config.ensure_output_dirs()
    json_name = f"{basename}.json" if basename else config.naming.json_name
    json_path = config.output.json_dir / json_name
    write_json(records, json_path, output_root=config.output.root_dir())
    return json_path


def write_json(records: list[QuestionRecord], output_path: str | Path, *, output_root: str | Path | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(output_root) if output_root is not None else None
    question_payload = [_record_to_output_dict(record, root) for record in records]
    output_path.write_text(
        json.dumps(
            {
                "schema_name": QUESTION_BANK_SCHEMA_NAME,
                "schema_version": QUESTION_BANK_SCHEMA_VERSION,
                "record_count": len(question_payload),
                "questions": question_payload,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path


def _record_to_output_dict(record: QuestionRecord, output_root: Path | None) -> dict[str, Any]:
    extraction = record.extraction
    classification = record.classification
    images = record.images
    mark_scheme = record.mark_scheme
    validation = record.validation
    metadata = record.paper_metadata
    paper_total = record.paper_total

    paper = paper_instance_id(
        metadata.component or metadata.source_paper_code,
        metadata.session,
        metadata.year,
    )
    family = paper_family_dir_name(classification.paper_family or classification.question_level_paper_family)
    subparts = list(extraction.question_subparts or mark_scheme.markscheme_subparts)
    question_image_paths = _path_list(images.screenshot_path, output_root)
    mark_scheme_image_paths = _path_list(mark_scheme.image_path, output_root)
    question_solution_marks = _question_solution_marks(record)

    return {
        "question_id": question_id(paper, extraction.question_number),
        "paper": paper,
        "paper_family": family,
        "question_number": extraction.question_number,
        "question_text": extraction.combined_question_text,
        "mark_scheme_text": mark_scheme.answer_text,
        "question_solution_marks": question_solution_marks,
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
            "mapping_status": mark_scheme.mapping_status,
            "mapping_failure_reason": mark_scheme.failure_reason,
            "scope_quality_status": validation.scope_quality_status,
            "question_crop_confidence": images.question_crop_confidence or (CropConfidence.LOW if images.crop_uncertain else CropConfidence.HIGH),
            "text_source_profile": validation.text_source_profile,
            "text_fidelity_status": validation.text_fidelity_status,
            "text_fidelity_flags": validation.text_fidelity_flags,
            "mark_scheme_crop_confidence": mark_scheme.crop_confidence,
            "review_flags": validation.review_flags,
            "extraction_quality_score": round(extraction.extraction_quality_score, 3),
            "extraction_quality_flags": extraction.extraction_quality_flags,
            "validation_status": validation.validation_status,
            "validation_flags": validation.validation_flags,
            "recovery_attempted": extraction.recovery_attempted,
            "recovery_result": extraction.recovery_result,
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
        return str(path)
    try:
        return str(path.relative_to(output_root))
    except ValueError:
        return str(path)
