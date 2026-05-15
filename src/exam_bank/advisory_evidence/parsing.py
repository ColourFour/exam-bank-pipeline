from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from exam_bank.advisory_evidence.common import first_ints_after, load_json, rel_path, utc_now_iso
from exam_bank.advisory_evidence.constants import (
    EXAMINER_PARSED_DIR,
    EXAMINER_PARSED_SCHEMA,
    EXAMINER_TEXT_DIR,
    GRADE_THRESHOLD_PARSED_DIR,
    GRADE_THRESHOLD_PARSED_SCHEMA,
    GRADE_THRESHOLD_TEXT_DIR,
)
from exam_bank.atomic_json import write_atomic_json


NO_EVIDENCE_RE = re.compile(
    r"\b(?:too few|not enough|insufficient)\s+candidates\b|"
    r"\b(?:too few|not enough|insufficient)\s+responses?\s+(?:to|for).{0,40}\bmeaningful report\b|"
    r"\btoo few candidates for a meaningful report\b|"
    r"\bnot enough candidates for a meaningful report\b",
    re.IGNORECASE,
)
LOW_EVIDENCE_RE = re.compile(
    r"\bsmall number of candidates\b|"
    r"\bonly a small number of candidates\b|"
    r"\blimited candidate evidence\b",
    re.IGNORECASE,
)


def parse_all_examiner_reports(
    *,
    extracted_dir: str | Path = EXAMINER_TEXT_DIR,
    output_dir: str | Path = EXAMINER_PARSED_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _parse_all(extracted_dir, output_dir, parse_examiner_report_text, dry_run=dry_run)


def parse_all_grade_thresholds(
    *,
    extracted_dir: str | Path = GRADE_THRESHOLD_TEXT_DIR,
    output_dir: str | Path = GRADE_THRESHOLD_PARSED_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _parse_all(extracted_dir, output_dir, parse_grade_threshold_text, dry_run=dry_run)


def parse_examiner_report_text(extracted: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(extracted.get("raw_text") or "")
    components = [_parse_component_section(section) for section in _component_sections(raw_text)]
    warnings = list(extracted.get("warnings", []))
    if not components:
        warnings.append("no_component_sections_detected")
    question_count = sum(len(component["questions"]) for component in components)
    if question_count == 0:
        warnings.append("no_question_comments_detected")
    return {
        "schema": EXAMINER_PARSED_SCHEMA,
        "generated_at": utc_now_iso(),
        "source_path": extracted.get("source_path", ""),
        "syllabus": extracted.get("syllabus", ""),
        "year": extracted.get("year", ""),
        "session": extracted.get("session", ""),
        "session_key": extracted.get("session_key", ""),
        "session_slug": extracted.get("session_slug", ""),
        "document_type": "examiner_report",
        "components": components,
        "parse_quality": {
            "component_count": len(components),
            "question_comment_count": question_count,
            "low_or_no_evidence_count": sum(
                1
                for component in components
                for question in component["questions"]
                if question["evidence_level"] in {"low", "none"}
            ),
        },
        "warnings": warnings,
    }


def parse_grade_threshold_text(extracted: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(extracted.get("raw_text") or "")
    lines = _nonempty_lines(raw_text)
    components = _parse_component_threshold_rows(lines)
    options = _parse_option_threshold_rows(lines)
    warnings = list(extracted.get("warnings", []))
    if not components:
        warnings.append("no_component_threshold_rows_detected")
    if not options:
        warnings.append("no_option_threshold_rows_detected")
    return {
        "schema": GRADE_THRESHOLD_PARSED_SCHEMA,
        "generated_at": utc_now_iso(),
        "source_path": extracted.get("source_path", ""),
        "syllabus": extracted.get("syllabus", ""),
        "year": extracted.get("year", ""),
        "session": extracted.get("session", ""),
        "session_key": extracted.get("session_key", ""),
        "session_slug": extracted.get("session_slug", ""),
        "document_type": "grade_thresholds",
        "components": components,
        "options": options,
        "parse_quality": {
            "component_row_count": len(components),
            "option_row_count": len(options),
        },
        "warnings": warnings,
    }


def _parse_all(
    extracted_dir: str | Path,
    output_dir: str | Path,
    parser,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    extracted_dir = Path(extracted_dir)
    output_dir = Path(output_dir)
    paths = sorted(extracted_dir.glob("*.json")) if extracted_dir.exists() else []
    outputs: list[str] = []
    for path in paths:
        payload = parser(load_json(path))
        output_path = output_dir / path.name
        outputs.append(rel_path(output_path))
        if not dry_run:
            write_atomic_json(payload, output_path)
    return {
        "dry_run": dry_run,
        "input_dir": rel_path(extracted_dir),
        "output_dir": rel_path(output_dir),
        "input_count": len(paths),
        "outputs": outputs,
        "warnings": [] if extracted_dir.exists() else [f"missing_extracted_dir:{extracted_dir.as_posix()}"],
    }


def _component_sections(text: str) -> list[dict[str, str]]:
    marker = re.compile(r"(?im)^\s*Paper\s+9709\s*/\s*([1-6][0-9])\s*$")
    matches = list(marker.finditer(text))
    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append({"component": match.group(1), "text": text[start:end]})
    return sections


def _parse_component_section(section: dict[str, str]) -> dict[str, Any]:
    text = section["text"]
    paper_title = _paper_title(text)
    key_messages = _section_paragraphs(text, "Key messages", ["General comments", "Comments on specific questions"])
    general_comments = "\n\n".join(_section_paragraphs(text, "General comments", ["Comments on specific questions"]))
    question_text = _section_text(text, "Comments on specific questions", [])
    questions = _parse_question_comments(question_text)
    evidence_level = _evidence_level(text)
    warnings: list[str] = []
    if evidence_level in {"low", "none"} and not questions:
        warnings.append("component_low_or_no_evidence")
    if not questions:
        warnings.append("no_question_comments_detected")
    return {
        "component": section["component"],
        "paper_title": paper_title,
        "section_headers": _section_headers(text),
        "key_messages": key_messages,
        "general_comments": general_comments,
        "questions": questions,
        "evidence_level": evidence_level,
        "warnings": warnings,
    }


def _parse_question_comments(text: str) -> list[dict[str, Any]]:
    marker = re.compile(r"(?im)^\s*Questions?\s+(\d{1,2})\s*[:.\-–]?\s*$")
    matches = list(marker.finditer(text))
    questions: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        comment = text[start:end].strip()
        question_number = int(match.group(1))
        evidence_level = _evidence_level(comment)
        questions.append(
            {
                "question_number": question_number,
                "parts": _part_labels(comment),
                "comment_text": comment,
                "evidence_level": evidence_level,
                "warnings": ["low_or_no_evidence_text"] if evidence_level in {"low", "none"} else [],
            }
        )
    return questions


def _paper_title(text: str) -> str:
    for line in _nonempty_lines(text):
        if re.fullmatch(r"Key messages|General comments|Comments on specific questions", line, re.IGNORECASE):
            return ""
        return line
    return ""


def _section_headers(text: str) -> list[str]:
    headers: list[str] = []
    for line in _nonempty_lines(text):
        if re.fullmatch(r"Key messages|General comments|Comments on specific questions", line, re.IGNORECASE):
            headers.append(line)
    return headers


def _section_paragraphs(text: str, header: str, stop_headers: list[str]) -> list[str]:
    section = _section_text(text, header, stop_headers)
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", section) if paragraph.strip()]


def _section_text(text: str, header: str, stop_headers: list[str]) -> str:
    start_match = re.search(rf"(?im)^\s*{re.escape(header)}\s*$", text)
    if not start_match:
        return ""
    start = start_match.end()
    stop_positions = []
    for stop in stop_headers:
        match = re.search(rf"(?im)^\s*{re.escape(stop)}\s*$", text[start:])
        if match:
            stop_positions.append(start + match.start())
    end = min(stop_positions) if stop_positions else len(text)
    return text[start:end].strip()


def _part_labels(text: str) -> list[str]:
    labels = []
    pattern = re.compile(r"^\s*\(([a-hivx]+)\)\s*$|^\s*([a-h])\s*\)\s*$", re.IGNORECASE | re.MULTILINE)
    for match in pattern.finditer(text):
        label = match.group(1) or match.group(2)
        labels.append(label.lower())
    return list(dict.fromkeys(labels))


def _evidence_level(text: str) -> str:
    if NO_EVIDENCE_RE.search(text):
        return "none"
    if LOW_EVIDENCE_RE.search(text):
        return "low"
    return "normal"


def _nonempty_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def _parse_component_threshold_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = re.fullmatch(r"Component\s+([1-6][0-9])", lines[index], re.IGNORECASE)
        if not match:
            index += 1
            continue
        values, next_index = first_ints_after(lines, index + 1, 6)
        warnings: list[str] = []
        if len(values) < 6:
            warnings.append("malformed_component_threshold_row")
            rows.append({"component": match.group(1), "max_raw_mark": None, "thresholds": {}, "warnings": warnings})
        else:
            rows.append(
                {
                    "component": match.group(1),
                    "max_raw_mark": values[0],
                    "thresholds": dict(zip(["A", "B", "C", "D", "E"], values[1:6])),
                    "warnings": warnings,
                }
            )
        index = next_index
    return rows


def _parse_option_threshold_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        code = lines[index]
        if not _looks_like_option_code(code):
            index += 1
            continue
        if index + 2 >= len(lines) or not re.fullmatch(r"\d+", lines[index + 1]):
            index += 1
            continue
        components = re.findall(r"\b[1-9][0-9]\b", lines[index + 2])
        if not components:
            index += 1
            continue
        values, next_index = first_ints_after(lines, index + 3, 6)
        warnings: list[str] = []
        thresholds: dict[str, int] = {}
        if len(values) >= 6:
            thresholds = dict(zip(["A*", "A", "B", "C", "D", "E"], values[:6]))
        elif len(values) == 5:
            thresholds = dict(zip(["A", "B", "C", "D", "E"], values))
        else:
            warnings.append("malformed_option_threshold_row")
        rows.append(
            {
                "option": code,
                "max_weighted_mark": int(lines[index + 1]),
                "components": components,
                "thresholds": thresholds,
                "warnings": warnings,
            }
        )
        index = next_index
    return rows


def _looks_like_option_code(value: str) -> bool:
    if re.fullmatch(r"S[1-9]", value):
        return True
    if re.fullmatch(r"[A-Z][XYZ]", value):
        return True
    return False
