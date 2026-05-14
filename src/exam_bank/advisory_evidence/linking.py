from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from exam_bank.advisory_evidence.common import load_json, records_from_question_bank, rel_path, utc_now_iso
from exam_bank.advisory_evidence.constants import (
    EXAMINER_LINKS_PATH,
    EXAMINER_LINKS_SCHEMA,
    EXAMINER_PARSED_DIR,
    GRADE_THRESHOLD_LINKS_PATH,
    GRADE_THRESHOLD_LINKS_SCHEMA,
    GRADE_THRESHOLD_PARSED_DIR,
)
from exam_bank.atomic_json import write_atomic_json
from exam_bank.document_metadata import parse_filename_metadata


def build_all_links(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    parsed_root: str | Path = "output/advisory_evidence/parsed",
    output_root: str | Path = "output/advisory_evidence",
    dry_run: bool = False,
) -> dict[str, Any]:
    records = records_from_question_bank(load_json(question_bank_path, default={"questions": []}))
    index = build_question_index(records)
    parsed_root = Path(parsed_root)
    output_root = Path(output_root)
    examiner = build_examiner_report_links(parsed_root / "examiner_reports", records=index.records, question_index=index.by_question)
    thresholds = build_grade_threshold_links(parsed_root / "grade_thresholds", records=index.records, component_index=index.by_component)
    examiner_path = output_root / "linking" / EXAMINER_LINKS_PATH.name
    thresholds_path = output_root / "linking" / GRADE_THRESHOLD_LINKS_PATH.name
    if not dry_run:
        write_atomic_json(examiner, examiner_path)
        write_atomic_json(thresholds, thresholds_path)
    return {
        "dry_run": dry_run,
        "outputs": {"examiner_report": rel_path(examiner_path), "grade_thresholds": rel_path(thresholds_path)},
        "examiner_report": examiner,
        "grade_thresholds": thresholds,
    }


class QuestionIndex:
    def __init__(self, records: dict[str, dict[str, Any]], by_question: dict[tuple[str, str, str, str, int], list[str]], by_component: dict[tuple[str, str, str, str], list[str]]) -> None:
        self.records = records
        self.by_question = by_question
        self.by_component = by_component


def build_question_index(records: list[dict[str, Any]]) -> QuestionIndex:
    record_by_id: dict[str, dict[str, Any]] = {}
    by_question: dict[tuple[str, str, str, str, int], list[str]] = defaultdict(list)
    by_component: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for record in records:
        question_id = str(record.get("question_id") or "")
        if not question_id:
            continue
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        source_pdf = str(notes.get("source_pdf") or record.get("source_pdf") or "")
        metadata = parse_filename_metadata(source_pdf)
        component = str(notes.get("source_paper_code") or metadata.component or "")
        question_number = _question_number_int(record.get("question_number"))
        record_by_id[question_id] = record
        if metadata.syllabus and metadata.year and metadata.session and component:
            component_key = (metadata.syllabus, metadata.year, metadata.session, component)
            by_component[component_key].append(question_id)
            if question_number is not None:
                by_question[(*component_key, question_number)].append(question_id)
    for values in by_question.values():
        values.sort()
    for values in by_component.values():
        values.sort()
    return QuestionIndex(record_by_id, by_question, by_component)


def build_examiner_report_links(
    parsed_dir: str | Path = EXAMINER_PARSED_DIR,
    *,
    records: dict[str, dict[str, Any]],
    question_index: dict[tuple[str, str, str, str, int], list[str]],
) -> dict[str, Any]:
    links: list[dict[str, Any]] = []
    warnings: list[str] = []
    parsed_dir = Path(parsed_dir)
    for path in sorted(parsed_dir.glob("*.json")) if parsed_dir.exists() else []:
        parsed = load_json(path)
        for component in parsed.get("components", []):
            component_code = str(component.get("component") or "")
            for question in component.get("questions", []):
                question_number = int(question.get("question_number"))
                key = (
                    str(parsed.get("syllabus") or ""),
                    str(parsed.get("year") or ""),
                    str(parsed.get("session") or ""),
                    component_code,
                    question_number,
                )
                candidates = list(question_index.get(key, []))
                links.append(
                    {
                        "normalized_key": normalized_question_key(*key),
                        "source_path": parsed.get("source_path", ""),
                        "syllabus": key[0],
                        "year": key[1],
                        "session": key[2],
                        "session_key": parsed.get("session_key", ""),
                        "component": component_code,
                        "question_number": question_number,
                        "candidate_question_ids": candidates,
                        "match_status": _match_status(candidates),
                        "evidence_level": question.get("evidence_level", "normal"),
                        "warnings": list(question.get("warnings", [])),
                    }
                )
    if not parsed_dir.exists():
        warnings.append(f"missing_parsed_dir:{parsed_dir.as_posix()}")
    return {
        "schema": EXAMINER_LINKS_SCHEMA,
        "generated_at": utc_now_iso(),
        "links": links,
        "summary": _link_summary(links),
        "warnings": warnings,
    }


def build_grade_threshold_links(
    parsed_dir: str | Path = GRADE_THRESHOLD_PARSED_DIR,
    *,
    records: dict[str, dict[str, Any]],
    component_index: dict[tuple[str, str, str, str], list[str]],
) -> dict[str, Any]:
    links: list[dict[str, Any]] = []
    warnings: list[str] = []
    parsed_dir = Path(parsed_dir)
    for path in sorted(parsed_dir.glob("*.json")) if parsed_dir.exists() else []:
        parsed = load_json(path)
        for component in parsed.get("components", []):
            component_code = str(component.get("component") or "")
            key = (
                str(parsed.get("syllabus") or ""),
                str(parsed.get("year") or ""),
                str(parsed.get("session") or ""),
                component_code,
            )
            candidates = list(component_index.get(key, []))
            links.append(
                {
                    "normalized_key": normalized_component_key(*key),
                    "source_path": parsed.get("source_path", ""),
                    "syllabus": key[0],
                    "year": key[1],
                    "session": key[2],
                    "session_key": parsed.get("session_key", ""),
                    "component": component_code,
                    "candidate_question_ids": candidates,
                    "match_status": "linked" if candidates else "unlinked",
                    "max_raw_mark": component.get("max_raw_mark"),
                    "thresholds": component.get("thresholds", {}),
                    "warnings": list(component.get("warnings", [])),
                }
            )
    if not parsed_dir.exists():
        warnings.append(f"missing_parsed_dir:{parsed_dir.as_posix()}")
    return {
        "schema": GRADE_THRESHOLD_LINKS_SCHEMA,
        "generated_at": utc_now_iso(),
        "links": links,
        "summary": _link_summary(links),
        "warnings": warnings,
    }


def normalized_question_key(syllabus: str, year: str, session: str, component: str, question_number: int) -> str:
    return f"{syllabus}_{year}_{session}_{component}_q{question_number:02d}"


def normalized_component_key(syllabus: str, year: str, session: str, component: str) -> str:
    return f"{syllabus}_{year}_{session}_{component}"


def _question_number_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def _match_status(candidates: list[str]) -> str:
    if not candidates:
        return "unlinked"
    if len(candidates) == 1:
        return "linked"
    return "ambiguous"


def _link_summary(links: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in ["linked", "ambiguous", "unlinked", "not_applicable"]}
    for link in links:
        status = str(link.get("match_status") or "")
        if status in counts:
            counts[status] += 1
    counts["total"] = len(links)
    return counts

