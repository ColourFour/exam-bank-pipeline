from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_NAME = "exam_bank.crop_text_signal_audit"
SCHEMA_VERSION = 1

QUESTION_NUMBER_RE = re.compile(r"^\s*(?:[A-Za-z]\s+)?(?P<number>\d{1,2})\b")
QUESTION_ANCHOR_RE = re.compile(r"(?:^|\n)\s*(?:[A-Za-z]\s+)?(?P<number>\d{1,2})\b")
MARK_BRACKET_RE = re.compile(r"\[(?P<marks>\d{1,2})\]")
SUBPART_RE = re.compile(r"\(([a-z])\)")
WORD_RE = re.compile(r"[a-z0-9]+")

PAGE_FURNITURE_PATTERNS = (
    "answer all questions",
    "additional materials",
    "read these instructions first",
    "cambridge international",
    "permission to reproduce",
    "blank page",
    "turn over",
)
ANSWER_SPACE_PATTERNS = (
    "working space",
    "answer space",
    "space for working",
    "write your answer",
    "do not write",
)

PRACTICAL_NOW_CODES = {
    "missing_expected_question_number",
    "missing_mark_bracket",
    "missing_specific_mark_bracket",
    "missing_expected_subpart_labels",
    "next_question_contamination",
    "page_furniture_or_answer_space_dominated",
    "suspiciously_short_selected_text",
    "selector_warning_present",
    "selector_structural_warning_present",
    "low_crop_confidence",
    "detected_structure_contamination",
    "question_mark_scheme_metadata_mismatch",
}

REQUIRES_METADATA_CODES = {
    "missing_crop_pixel_dimensions",
    "missing_text_line_bboxes",
    "missing_crop_page_position",
    "missing_raw_candidate_windows",
    "missing_selector_warning_metadata",
}

RISKY_CODES = {
    "math_expression_semantics",
    "short_single_part_question",
    "answer_space_language",
    "visual_crop_scope_from_text_only",
}

STRUCTURAL_REVIEW_FLAGS = {
    "weak_question_text",
    "ocr_merged_sparse_lower_region",
    "paper_total_focus_candidate",
    "question_scope_contaminated",
    "question_subparts_incomplete",
    "weak_question_anchor",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_question_bank_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    records = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return {}
    return {str(record.get("question_id") or record.get("record_id")): record for record in records}


def build_crop_text_signal_audit(
    fixture_manifest: dict[str, Any],
    question_bank_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    question_bank_index = question_bank_index or {}
    records = [
        audit_fixture_record(record, question_bank_index.get(str(record.get("record_id"))))
        for record in fixture_manifest.get("records", [])
    ]

    warning_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    caught_by_practical_now = 0
    useful_warning_records = 0
    for record in records:
        practical = False
        useful = False
        for warning in record["warnings"]:
            warning_counts[warning["code"]] += 1
            category_counts[warning["category"]] += 1
            if warning["category"] == "available_now":
                useful = True
            if warning["gate_candidate"] == "practical_now":
                practical = True
        if useful:
            useful_warning_records += 1
        if practical:
            caught_by_practical_now += 1

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "fixture_schema_name": fixture_manifest.get("schema_name"),
        "fixture_schema_version": fixture_manifest.get("schema_version"),
        "record_count": len(records),
        "question_bank_context_records": sum(1 for record in records if record["question_bank_context_available"]),
        "records_with_useful_warnings": useful_warning_records,
        "records_caught_by_practical_now_gates": caught_by_practical_now,
        "warning_counts": dict(sorted(warning_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "signal_assessment": signal_assessment(),
        "records": records,
    }


def audit_fixture_record(record: dict[str, Any], question_bank_record: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = text_value(record.get("currently_selected_text"))
    question_number = str(record.get("question_number") or "").strip()
    expectations = list(
        (record.get("expected_normalized_text_or_structural_expectations") or {}).get("expectations") or []
    )
    qbank = question_bank_record or {}
    notes = qbank.get("notes") if isinstance(qbank.get("notes"), dict) else {}
    structure = notes.get("question_structure_detected") if isinstance(notes.get("question_structure_detected"), dict) else {}
    warnings: list[dict[str, Any]] = []

    add_question_number_warning(warnings, selected, question_number)
    add_mark_warning(warnings, selected, expectations, qbank)
    add_subpart_warning(warnings, selected, expectations, qbank, structure)
    add_next_question_warning(warnings, selected, question_number, structure)
    add_short_text_warning(warnings, selected, qbank, structure)
    add_furniture_warning(warnings, selected)
    add_mark_scheme_mismatch_warning(warnings, record, qbank, notes)
    add_selector_warning_metadata(warnings, record, qbank, notes)
    add_missing_metadata_warnings(warnings, record, qbank, notes)
    add_risky_signal_notes(warnings, selected, expectations, qbank)

    warnings = dedupe_warnings(warnings)
    return {
        "record_id": record.get("record_id"),
        "paper_id": record.get("paper_id"),
        "paper_family": record.get("paper_family"),
        "question_number": question_number,
        "question_bank_context_available": bool(qbank),
        "selected_text_length": len(selected),
        "selected_word_count": len(tokens(selected)),
        "question_image_path": record.get("question_image_path"),
        "mark_scheme_image_path": record.get("mark_scheme_image_path"),
        "warnings": warnings,
        "practical_gate_codes": sorted(
            {warning["code"] for warning in warnings if warning["gate_candidate"] == "practical_now"}
        ),
        "metadata_required_codes": sorted(
            {warning["code"] for warning in warnings if warning["category"] == "requires_new_metadata"}
        ),
        "risky_or_ambiguous_codes": sorted(
            {warning["code"] for warning in warnings if warning["category"] == "too_risky_or_ambiguous"}
        ),
    }


def add_question_number_warning(warnings: list[dict[str, Any]], selected: str, question_number: str) -> None:
    if not question_number:
        return
    if not contains_number_token(selected, question_number):
        warnings.append(
            warning(
                "missing_expected_question_number",
                "available_now",
                "practical_now",
                f"Selected text does not contain expected question number {question_number}.",
            )
        )


def add_mark_warning(
    warnings: list[dict[str, Any]],
    selected: str,
    expectations: list[str],
    qbank: dict[str, Any],
) -> None:
    expected_marks = expected_mark_brackets(expectations)
    qbank_marks = qbank.get("question_solution_marks")
    qbank_subparts = qbank.get("subparts") or []
    if not expected_marks and isinstance(qbank_marks, int) and not qbank_subparts:
        expected_marks.append(str(qbank_marks))
    expects_any_mark = bool(expected_marks) or any("mark bracket" in expectation.lower() for expectation in expectations)
    if not expects_any_mark:
        return
    present_marks = set(MARK_BRACKET_RE.findall(selected))
    if not present_marks:
        warnings.append(
            warning(
                "missing_mark_bracket",
                "available_now",
                "practical_now",
                "Selected text has no Cambridge-style [n] mark bracket.",
            )
        )
        return
    for expected in sorted(set(expected_marks)):
        if expected not in present_marks:
            warnings.append(
                warning(
                    "missing_specific_mark_bracket",
                    "available_now",
                    "practical_now",
                    f"Selected text has mark brackets {sorted(present_marks)} but expected [{expected}].",
                )
            )


def add_subpart_warning(
    warnings: list[dict[str, Any]],
    selected: str,
    expectations: list[str],
    qbank: dict[str, Any],
    structure: dict[str, Any],
) -> None:
    expected = expected_subparts(expectations, qbank, structure)
    if not expected:
        return
    present = set(SUBPART_RE.findall(selected.lower()))
    missing = sorted(set(expected) - present)
    if missing:
        warnings.append(
            warning(
                "missing_expected_subpart_labels",
                "available_now",
                "practical_now",
                f"Selected text is missing expected subpart labels: {', '.join(missing)}.",
            )
        )


def add_next_question_warning(
    warnings: list[dict[str, Any]],
    selected: str,
    question_number: str,
    structure: dict[str, Any],
) -> None:
    next_number = str(int(question_number) + 1) if question_number.isdigit() else ""
    if next_number and re.search(rf"\[\d+\]\s+{re.escape(next_number)}\s+[A-Z][A-Za-z]", selected):
        warnings.append(
            warning(
                "next_question_contamination",
                "available_now",
                "practical_now",
                f"Selected text appears to continue into question {next_number}.",
            )
        )
    indicators = structure.get("contamination_indicators") if isinstance(structure.get("contamination_indicators"), dict) else {}
    foreign_anchors = indicators.get("foreign_question_anchors") or []
    if structure.get("contamination_detected") or foreign_anchors:
        warnings.append(
            warning(
                "detected_structure_contamination",
                "available_now",
                "practical_now",
                "Existing structure detector reported contamination in the selected question text.",
                evidence={"foreign_question_anchors": foreign_anchors[:5]},
            )
        )


def add_short_text_warning(
    warnings: list[dict[str, Any]],
    selected: str,
    qbank: dict[str, Any],
    structure: dict[str, Any],
) -> None:
    word_count = len(tokens(selected))
    detected_length = structure.get("text_length")
    qbank_text = text_value(qbank.get("question_text"))
    if word_count < 8:
        warnings.append(
            warning(
                "suspiciously_short_selected_text",
                "available_now",
                "practical_now",
                f"Selected text has only {word_count} words.",
            )
        )
    elif qbank_text and len(selected) < len(qbank_text) * 0.55:
        warnings.append(
            warning(
                "suspiciously_short_selected_text",
                "available_now",
                "practical_now",
                f"Fixture selected text length {len(selected)} is much shorter than question-bank text length {len(qbank_text)}.",
            )
        )
    elif isinstance(detected_length, int) and detected_length < 60:
        warnings.append(
            warning(
                "suspiciously_short_selected_text",
                "available_now",
                "practical_now",
                f"Existing structure detector text length is {detected_length}.",
            )
        )


def add_furniture_warning(warnings: list[dict[str, Any]], selected: str) -> None:
    lower = selected.lower()
    furniture_hits = [pattern for pattern in PAGE_FURNITURE_PATTERNS if pattern in lower]
    answer_space_hits = [pattern for pattern in ANSWER_SPACE_PATTERNS if pattern in lower]
    if not furniture_hits and not answer_space_hits:
        return
    text_words = tokens(selected)
    hit_count = sum(lower.count(hit) for hit in furniture_hits + answer_space_hits)
    if hit_count >= 2 or (text_words and hit_count / len(text_words) > 0.08):
        warnings.append(
            warning(
                "page_furniture_or_answer_space_dominated",
                "available_now",
                "practical_now",
                "Selected text is dominated by page furniture or answer-space language.",
                evidence={"furniture_hits": furniture_hits, "answer_space_hits": answer_space_hits},
            )
        )


def add_mark_scheme_mismatch_warning(
    warnings: list[dict[str, Any]],
    fixture_record: dict[str, Any],
    qbank: dict[str, Any],
    notes: dict[str, Any],
) -> None:
    session = fixture_record.get("session") if isinstance(fixture_record.get("session"), dict) else {}
    source_pdf = str(session.get("source_pdf") or notes.get("source_pdf") or "")
    mark_scheme_pdf = str(notes.get("mark_scheme_source_pdf") or "")
    source_component = str(session.get("component") or notes.get("source_paper_code") or "")
    mark_component = component_from_path(mark_scheme_pdf)
    if source_component and mark_component and source_component != mark_component:
        warnings.append(
            warning(
                "question_mark_scheme_metadata_mismatch",
                "available_now",
                "practical_now",
                f"Question component {source_component} does not match mark-scheme component {mark_component}.",
            )
        )
    if source_pdf and mark_scheme_pdf:
        source_session = normalized_session_token(source_pdf)
        mark_session = normalized_session_token(mark_scheme_pdf)
        if source_session and mark_session and source_session != mark_session:
            warnings.append(
                warning(
                    "question_mark_scheme_metadata_mismatch",
                    "available_now",
                    "practical_now",
                    f"Question session {source_session} does not match mark-scheme session {mark_session}.",
                )
            )
    mapping_status = notes.get("mapping_status")
    if mapping_status in {"fail", "review"}:
        warnings.append(
            warning(
                "question_mark_scheme_metadata_mismatch",
                "available_now",
                "practical_now",
                f"Existing mark-scheme mapping status is {mapping_status}.",
                evidence={"mapping_failure_reason": notes.get("mapping_failure_reason") or ""},
            )
        )
    elif qbank and not qbank.get("mark_scheme_image_path"):
        warnings.append(
            warning(
                "question_mark_scheme_metadata_mismatch",
                "available_now",
                "practical_now",
                "Question-bank record has no mark-scheme image path.",
            )
        )


def add_selector_warning_metadata(
    warnings: list[dict[str, Any]],
    fixture_record: dict[str, Any],
    qbank: dict[str, Any],
    notes: dict[str, Any],
) -> None:
    existing_flags = [str(flag) for flag in fixture_record.get("text_fidelity_flags") or []]
    review_flags = [str(flag) for flag in notes.get("review_flags") or []]
    validation_flags = [str(flag) for flag in notes.get("validation_flags") or []]
    extraction_flags = [str(flag) for flag in notes.get("extraction_quality_flags") or []]
    reasons = [str(reason) for reason in notes.get("text_candidate_decision_reasons") or []]
    rejected = [str(reason) for reason in notes.get("ocr_rejected_reasons") or []]

    if existing_flags or validation_flags or extraction_flags:
        warnings.append(
            warning(
                "selector_warning_present",
                "available_now",
                "practical_now",
                "Existing selector or validation warnings are available for this record.",
                evidence={
                    "text_fidelity_flags": existing_flags,
                    "validation_flags": validation_flags,
                    "extraction_quality_flags": extraction_flags,
                },
            )
        )

    structural_flags = sorted(set(review_flags) & STRUCTURAL_REVIEW_FLAGS)
    structural_reasons = [
        reason for reason in reasons + rejected if "suspiciously_short" in reason or "merged" in reason or "lost" in reason
    ]
    if structural_flags or structural_reasons:
        warnings.append(
            warning(
                "selector_structural_warning_present",
                "available_now",
                "practical_now",
                "Existing selector metadata already contains structural text warnings.",
                evidence={"review_flags": structural_flags, "decision_reasons": structural_reasons},
            )
        )

    crop_confidence = notes.get("question_crop_confidence")
    if crop_confidence and crop_confidence != "high":
        warnings.append(
            warning(
                "low_crop_confidence",
                "available_now",
                "practical_now",
                f"Existing question crop confidence is {crop_confidence}.",
            )
        )
    if qbank and not any(key in notes for key in ("review_flags", "text_candidate_decision_reasons", "ocr_rejected_reasons")):
        warnings.append(
            warning(
                "missing_selector_warning_metadata",
                "requires_new_metadata",
                "requires_new_metadata",
                "Question-bank context exists but selector warning metadata is absent.",
            )
        )


def add_missing_metadata_warnings(
    warnings: list[dict[str, Any]],
    fixture_record: dict[str, Any],
    qbank: dict[str, Any],
    notes: dict[str, Any],
) -> None:
    diagnostics = notes.get("question_crop_diagnostics") if isinstance(notes.get("question_crop_diagnostics"), dict) else {}
    regions = diagnostics.get("regions") if isinstance(diagnostics.get("regions"), list) else []
    if not qbank:
        warnings.append(
            warning(
                "missing_crop_page_position",
                "requires_new_metadata",
                "requires_new_metadata",
                "No question-bank context was available to inspect crop diagnostics.",
            )
        )
        return
    if not qbank.get("question_image_path") and not fixture_record.get("question_image_path"):
        warnings.append(
            warning(
                "missing_crop_pixel_dimensions",
                "requires_new_metadata",
                "requires_new_metadata",
                "No question image path is available.",
            )
        )
    warnings.append(
        warning(
            "missing_crop_pixel_dimensions",
            "requires_new_metadata",
            "requires_new_metadata",
            "Image pixel dimensions are not exported with the fixture, so text gates cannot normalize crop size.",
        )
    )
    if not regions:
        warnings.append(
            warning(
                "missing_crop_page_position",
                "requires_new_metadata",
                "requires_new_metadata",
                "No crop region bboxes were available for this record.",
            )
        )
    elif not all(isinstance(region.get("text_bbox"), dict) for region in regions if isinstance(region, dict)):
        warnings.append(
            warning(
                "missing_text_line_bboxes",
                "requires_new_metadata",
                "requires_new_metadata",
                "Region diagnostics lack text bboxes for at least one crop region.",
            )
        )
    warnings.append(
        warning(
            "missing_raw_candidate_windows",
            "requires_new_metadata",
            "requires_new_metadata",
            "Rejected candidate text windows and their crop positions are not exported for fixture replay.",
        )
    )


def add_risky_signal_notes(
    warnings: list[dict[str, Any]],
    selected: str,
    expectations: list[str],
    qbank: dict[str, Any],
) -> None:
    expectation_text = " ".join(expectations).lower()
    if any(term in expectation_text for term in ("theta", "integral", "fraction", "radical", "vector", "inequality")):
        warnings.append(
            warning(
                "math_expression_semantics",
                "too_risky_or_ambiguous",
                "too_risky_or_ambiguous",
                "Text and crop metadata can flag risk, but cannot safely verify math expression semantics.",
            )
        )
    if len(tokens(selected)) < 15 and not (qbank.get("subparts") or []):
        warnings.append(
            warning(
                "short_single_part_question",
                "too_risky_or_ambiguous",
                "too_risky_or_ambiguous",
                "Very short single-part questions can be valid, so length should be review-only.",
            )
        )
    if any(pattern in selected.lower() for pattern in ANSWER_SPACE_PATTERNS):
        warnings.append(
            warning(
                "answer_space_language",
                "too_risky_or_ambiguous",
                "too_risky_or_ambiguous",
                "Answer-space language is only safe when it dominates text or combines with other warnings.",
            )
        )
    warnings.append(
        warning(
            "visual_crop_scope_from_text_only",
            "too_risky_or_ambiguous",
            "too_risky_or_ambiguous",
            "Text-only clues cannot prove the canonical crop is visually wrong.",
        )
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Crop Context Signal Audit",
        "",
        "This is a non-mutating audit of advisory text selection for frozen bad-text fixtures.",
        "Canonical question and mark-scheme images remain the source of truth.",
        "",
        "## Summary",
        "",
        f"- Records: {report['record_count']}",
        f"- Question-bank context joined: {report['question_bank_context_records']}",
        f"- Records with useful warnings: {report['records_with_useful_warnings']}",
        f"- Records caught by practical-now gates: {report['records_caught_by_practical_now_gates']}",
        f"- Warning categories: {format_counts(report['category_counts'])}",
        "",
        "## Signals Available Now",
        "",
    ]
    for item in report["signal_assessment"]["available_now"]:
        lines.append(f"- {item['signal']}: {item['assessment']}")
    lines.extend(["", "## Signals Requiring New Metadata", ""])
    for item in report["signal_assessment"]["requires_new_metadata"]:
        lines.append(f"- {item['signal']}: {item['assessment']}")
    lines.extend(["", "## Signals Too Risky Or Ambiguous", ""])
    for item in report["signal_assessment"]["too_risky_or_ambiguous"]:
        lines.append(f"- {item['signal']}: {item['assessment']}")

    lines.extend(["", "## Warning Counts", "", "| Warning | Count |", "| --- | ---: |"])
    for code, count in report["warning_counts"].items():
        lines.append(f"| {code} | {count} |")

    lines.extend(
        [
            "",
            "## Per-Fixture Warnings",
            "",
            "| Record | Practical now | Requires metadata | Risky/ambiguous |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in report["records"]:
        practical = ", ".join(record["practical_gate_codes"]) or "none"
        metadata = ", ".join(record["metadata_required_codes"]) or "none"
        risky = ", ".join(record["risky_or_ambiguous_codes"]) or "none"
        lines.append(f"| {record['record_id']} | {practical} | {metadata} | {risky} |")

    lines.extend(
        [
            "",
            "## Fixture Warning Threshold",
            "",
            threshold_note(report["records_with_useful_warnings"]),
            "",
        ]
    )
    return "\n".join(lines)


def signal_assessment() -> dict[str, list[dict[str, str]]]:
    return {
        "available_now": [
            {
                "signal": "expected question number appears in selected text",
                "assessment": "Safe as a review gate when the expected number is known.",
            },
            {
                "signal": "mark bracket appears in selected text",
                "assessment": "Safe as a review gate for CAIE-style prompts with known mark totals.",
            },
            {
                "signal": "expected subpart labels appear",
                "assessment": "Safe when subparts are present in fixture expectations or detected structure.",
            },
            {
                "signal": "selected text likely includes next question",
                "assessment": "Safe as a high-severity warning when a following question anchor appears after marks.",
            },
            {
                "signal": "suspiciously short selected text",
                "assessment": "Useful warning, but should combine with marks/subparts/selector warnings before blocking.",
            },
            {
                "signal": "page furniture dominates selected text",
                "assessment": "Safe when boilerplate or answer-space phrases dominate rather than merely appear.",
            },
            {
                "signal": "mark-scheme/question mismatch risk",
                "assessment": "Safe when source metadata, mapping status, or missing mark-scheme image contradicts the question.",
            },
            {
                "signal": "current selector warnings",
                "assessment": "Available in question-bank notes and useful as non-mutating warning evidence.",
            },
        ],
        "requires_new_metadata": [
            {
                "signal": "crop pixel dimensions and normalized crop area",
                "assessment": "Needed to judge whether a text crop is implausibly small or too page-like across papers.",
            },
            {
                "signal": "text line bboxes linked to selected text spans",
                "assessment": "Needed to distinguish prompt text from headers, footers, diagrams, and answer spaces reliably.",
            },
            {
                "signal": "raw candidate windows with rejected reasons",
                "assessment": "Needed to replay selector choices and test alternative advisory text gates without mutation.",
            },
        ],
        "too_risky_or_ambiguous": [
            {
                "signal": "math expression correctness from text alone",
                "assessment": "Requires visual/manual or math-aware comparison; crop metadata can only flag risk.",
            },
            {
                "signal": "short text alone",
                "assessment": "Some valid prompts are very short, especially single-part questions.",
            },
            {
                "signal": "answer-space language alone",
                "assessment": "Can appear legitimately in instructions; only dominant or combined evidence is safe.",
            },
            {
                "signal": "visual crop correctness from selected text only",
                "assessment": "Text anomalies do not prove the canonical image crop is wrong.",
            },
        ],
    }


def expected_mark_brackets(expectations: list[str]) -> list[str]:
    marks: list[str] = []
    for expectation in expectations:
        match = MARK_BRACKET_RE.search(expectation)
        if match:
            marks.append(match.group("marks"))
    return marks


def expected_subparts(expectations: list[str], qbank: dict[str, Any], structure: dict[str, Any]) -> list[str]:
    expected: set[str] = set()
    for value in qbank.get("subparts") or []:
        if isinstance(value, str) and len(value) == 1:
            expected.add(value.lower())
    for value in structure.get("subparts") or []:
        if isinstance(value, str) and len(value) == 1:
            expected.add(value.lower())
    if expected:
        return sorted(expected)
    if any("subpart" in expectation.lower() for expectation in expectations):
        return ["a", "b"]
    return []


def component_from_path(path: str) -> str:
    match = re.search(r"(?:paper|scheme)\s+(\d{2})\b", path, re.IGNORECASE)
    return match.group(1) if match else ""


def normalized_session_token(path: str) -> str:
    match = re.search(r"mathematics\s+(march|june|november)\s+(\d{4})", path, re.IGNORECASE)
    if not match:
        return ""
    return f"{match.group(1).lower()}-{match.group(2)}"


def contains_number_token(text: str, number: str) -> bool:
    return any(match.group("number") == number for match in QUESTION_ANCHOR_RE.finditer(text))


def warning(
    code: str,
    category: str,
    gate_candidate: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "category": category,
        "gate_candidate": gate_candidate,
        "message": message,
        "evidence": evidence or {},
    }


def dedupe_warnings(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in warnings:
        key = (item["code"], item["message"])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def tokens(value: str) -> list[str]:
    return WORD_RE.findall(value.lower())


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def threshold_note(records_with_useful_warnings: int) -> str:
    if records_with_useful_warnings >= 20:
        return f"At least 20 fixture records received useful crop-context warnings ({records_with_useful_warnings})."
    return (
        f"Only {records_with_useful_warnings} fixture records received useful crop-context warnings. "
        "The available fixture data does not carry enough crop geometry or selector candidate metadata to classify more safely."
    )
