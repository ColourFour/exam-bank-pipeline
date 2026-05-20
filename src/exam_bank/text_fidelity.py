from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from exam_bank.text_normalization import normalize_advisory_question_text


SCHEMA_NAME = "exam_bank.text_fidelity.fixture_report"
SCHEMA_VERSION = 1
FIXTURE_SCHEMA_NAME = "text_fidelity_bad_text_fixture_manifest"

FAILURE_TYPE_BY_CODE = {
    "missing_question_number": "question_anchor",
    "question_number_not_at_start": "question_anchor",
    "missing_mark_bracket": "mark_bracket",
    "missing_specific_mark_bracket": "mark_bracket",
    "missing_subpart_labels": "subpart_labels",
    "subpart_label_order_displaced": "subpart_labels",
    "suspiciously_short_selected_text": "selected_text_length",
    "selected_text_much_shorter_than_raw": "selected_text_length",
    "ocr_native_disagreement": "source_disagreement",
    "selected_differs_from_native": "source_disagreement",
    "selected_differs_from_ocr": "source_disagreement",
    "likely_math_symbol_loss": "math_symbol_loss",
    "likely_next_question_contamination": "contamination",
    "selected_source_rejected_by_structural_checks": "structural_rejection",
    "expected_structural_requirement_missing": "expectation_gap",
}

STRUCTURAL_REJECTION_FLAGS = {
    "hybrid_math_text_requires_review",
    "math_text_corruption_detected",
    "native_compacted_math_corruption",
    "ocr_noise_fragment_present",
    "sparse_or_merged_question_text",
    "weak_extracted_text",
}

MATH_LOSS_TAGS = {
    "calculus_expression",
    "complex_number_layout",
    "derivative_layout",
    "fraction_structure",
    "greek_symbol",
    "inequality_direction",
    "integral_bounds",
    "math_notation",
    "polynomial_reading_order",
    "radical_or_power_structure",
    "rational_expression",
    "trig_symbol_fidelity",
    "units_symbol",
    "vector_matrix_layout",
}

QUESTION_NUMBER_RE = re.compile(r"^\s*(?:[A-Za-z]\s+)?(?P<number>\d{1,2})\b")
MARK_BRACKET_RE = re.compile(r"\[(?P<marks>\d{1,2})\]")
SUBPART_RE = re.compile(r"\([a-z]\)")
WORD_RE = re.compile(r"[a-z0-9]+")


def load_fixture_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != FIXTURE_SCHEMA_NAME:
        raise ValueError(f"Unexpected fixture schema: {manifest.get('schema_name')!r}")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Fixture manifest must contain a records list.")
    if manifest.get("record_count") != len(records):
        raise ValueError("Fixture record_count does not match records length.")
    return manifest


def build_fixture_report(manifest: dict[str, Any], *, include_normalized: bool = False) -> dict[str, Any]:
    records = [score_fixture_record(record, include_normalized=include_normalized) for record in manifest["records"]]
    status_counts = Counter(record["status"] for record in records)
    issue_code_counts: Counter[str] = Counter()
    failure_type_counts: Counter[str] = Counter()
    expected_missing_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()

    for record in records:
        for issue in record["issues"]:
            code = issue["code"]
            issue_code_counts[code] += 1
            failure_type_counts[issue["failure_type"]] += 1
            severity_counts[issue["severity"]] += 1
            if code == "expected_structural_requirement_missing":
                expected_missing_counts[issue["requirement_code"]] += 1

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "normalized_advisory_candidates_included": include_normalized,
        "fixture_schema_name": manifest.get("schema_name"),
        "fixture_schema_version": manifest.get("schema_version"),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "failure_type_counts": dict(sorted(failure_type_counts.items())),
        "issue_code_counts": dict(sorted(issue_code_counts.items())),
        "expected_requirement_missing_counts": dict(sorted(expected_missing_counts.items())),
        "top_failure_types": [
            {"failure_type": failure_type, "count": count}
            for failure_type, count in failure_type_counts.most_common()
        ],
        "records": records,
    }
    if include_normalized:
        report["normalization_summary"] = build_normalization_summary(records)
    return report


def score_fixture_record(record: dict[str, Any], *, include_normalized: bool = False) -> dict[str, Any]:
    selected = text_value(record.get("currently_selected_text"))
    native = text_value(record.get("native_pdf_text_raw"))
    ocr = text_value(record.get("ocr_text_raw"))
    question_number = str(record.get("question_number") or "").strip()
    expectations = list(
        (record.get("expected_normalized_text_or_structural_expectations") or {}).get("expectations") or []
    )
    failure_tags = [str(tag) for tag in record.get("failure_tags") or []]
    existing_flags = [str(flag) for flag in record.get("text_fidelity_flags") or []]

    issues = score_text_issues(record, selected, native, ocr, question_number, expectations, failure_tags, existing_flags)
    status = "fail" if any(issue["severity"] == "fail" for issue in issues) else "warn" if issues else "pass"
    scored = {
        "record_id": record.get("record_id"),
        "paper_id": record.get("paper_id"),
        "paper_family": record.get("paper_family"),
        "question_number": question_number,
        "source_profile": record.get("source_profile"),
        "existing_text_fidelity_status": record.get("text_fidelity_status"),
        "existing_text_fidelity_flags": existing_flags,
        "fixture_failure_tags": failure_tags,
        "selected_text_length": len(selected),
        "native_text_length": len(native),
        "ocr_text_length": len(ocr),
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
        "measurable_improvement_targets": measurable_improvement_targets(issues),
    }
    if include_normalized:
        normalized = normalize_advisory_question_text(
            selected,
            native_pdf_text_raw=native,
            ocr_text_raw=ocr,
            metadata={
                "record_id": record.get("record_id"),
                "failure_tags": failure_tags,
                "expectations": expectations,
            },
        )
        comparable_raw_issues = score_text_issues(
            record,
            selected,
            native,
            ocr,
            question_number,
            expectations,
            failure_tags,
            existing_flags,
            include_disagreement=False,
            include_structural_rejection=False,
        )
        normalized_issues = score_text_issues(
            record,
            normalized.normalized_text,
            native,
            ocr,
            question_number,
            expectations,
            failure_tags,
            existing_flags,
            include_disagreement=False,
            include_structural_rejection=False,
        )
        raw_issue_codes = {issue_key(issue) for issue in comparable_raw_issues}
        normalized_issue_codes = {issue_key(issue) for issue in normalized_issues}
        resolved = sorted(raw_issue_codes - normalized_issue_codes)
        introduced = sorted(normalized_issue_codes - raw_issue_codes)
        scored.update(
            {
                "native_pdf_text_raw": native,
                "ocr_text_raw": ocr,
                "selected_text_raw": selected,
                "question_text_normalized": normalized.normalized_text,
                "normalization_flags": normalized.flags,
                "normalization_confidence": normalized.confidence,
                "normalization_warnings": normalized.warnings,
                "normalization_is_advisory": True,
                "normalization_issue_count": len(normalized_issues),
                "normalization_resolved_issue_keys": resolved,
                "normalization_introduced_issue_keys": introduced,
                "normalization_classification": classify_normalization_result(resolved, introduced, normalized.flags),
            }
        )
    return scored


def score_text_issues(
    record: dict[str, Any],
    selected: str,
    native: str,
    ocr: str,
    question_number: str,
    expectations: list[str],
    failure_tags: list[str],
    existing_flags: list[str],
    *,
    include_disagreement: bool = True,
    include_structural_rejection: bool = True,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    add_question_number_issues(issues, selected, question_number, expectations)
    add_mark_bracket_issues(issues, selected, expectations)
    add_subpart_issues(issues, selected, native, ocr, expectations)
    add_short_text_issues(issues, selected, native, ocr)
    if include_disagreement:
        add_disagreement_issues(issues, selected, native, ocr)
    add_math_symbol_loss_issues(issues, selected, native, ocr, failure_tags, expectations)
    add_contamination_issues(issues, selected, question_number)
    if include_structural_rejection:
        add_structural_rejection_issues(issues, record, existing_flags)
    add_expected_requirement_issues(issues, selected, question_number, expectations)
    return dedupe_issues(issues)


def add_question_number_issues(
    issues: list[dict[str, Any]],
    selected: str,
    question_number: str,
    expectations: list[str],
) -> None:
    if not question_number:
        return
    selected_norm = normalize_for_match(selected)
    expected_start = any("starts with question number" in expectation.lower() for expectation in expectations)
    contains = re.search(rf"(?:^|\D){re.escape(question_number)}(?:\D|$)", selected_norm) is not None
    if not contains:
        issues.append(issue("missing_question_number", "fail", f"question number {question_number} not found"))
        return
    first_match = QUESTION_NUMBER_RE.search(selected)
    if expected_start and (not first_match or first_match.group("number") != question_number):
        issues.append(
            issue("question_number_not_at_start", "fail", f"expected selected text to start with {question_number}")
        )


def add_mark_bracket_issues(issues: list[dict[str, Any]], selected: str, expectations: list[str]) -> None:
    mark_expectations = [expectation for expectation in expectations if "mark bracket" in expectation.lower()]
    if not mark_expectations:
        return
    brackets = set(MARK_BRACKET_RE.findall(selected))
    if not brackets:
        issues.append(issue("missing_mark_bracket", "fail", "no [n] mark bracket found in selected text"))
    for expectation in mark_expectations:
        expected = MARK_BRACKET_RE.search(expectation)
        if expected and expected.group("marks") not in brackets:
            issues.append(
                issue(
                    "missing_specific_mark_bracket",
                    "fail",
                    f"expected mark bracket [{expected.group('marks')}] not found",
                )
            )


def add_subpart_issues(
    issues: list[dict[str, Any]],
    selected: str,
    native: str,
    ocr: str,
    expectations: list[str],
) -> None:
    if not any("subpart" in expectation.lower() for expectation in expectations):
        return
    lower_selected = selected.lower()
    selected_parts = set(SUBPART_RE.findall(selected.lower()))
    raw_parts = set(SUBPART_RE.findall((native + " " + ocr).lower()))
    if raw_parts and not selected_parts.issuperset(raw_parts):
        issues.append(
            issue(
                "missing_subpart_labels",
                "fail",
                f"selected labels {sorted(selected_parts)} do not preserve raw labels {sorted(raw_parts)}",
            )
        )
    elif len(selected_parts) < 2:
        issues.append(issue("missing_subpart_labels", "warn", "expected subpart labels are not clearly present"))
    first_part = SUBPART_RE.search(lower_selected)
    a_index = lower_selected.find("(a)")
    b_index = lower_selected.find("(b)")
    if first_part and first_part.group(0) != "(a)":
        issues.append(issue("subpart_label_order_displaced", "fail", f"selected text starts subparts at {first_part.group(0)}"))
    elif a_index != -1 and b_index != -1 and b_index < a_index:
        issues.append(issue("subpart_label_order_displaced", "fail", "subpart (b) appears before subpart (a)"))


def add_short_text_issues(issues: list[dict[str, Any]], selected: str, native: str, ocr: str) -> None:
    selected_words = len(tokens(selected))
    if selected_words < 8:
        issues.append(issue("suspiciously_short_selected_text", "warn", f"selected text has {selected_words} words"))
    raw_lengths = [len(value) for value in (native, ocr) if value]
    if raw_lengths and len(selected) < max(raw_lengths) * 0.55:
        issues.append(
            issue(
                "selected_text_much_shorter_than_raw",
                "warn",
                f"selected length {len(selected)} is much shorter than raw max {max(raw_lengths)}",
            )
        )


def add_disagreement_issues(issues: list[dict[str, Any]], selected: str, native: str, ocr: str) -> None:
    if native and ocr:
        ratio = similarity(native, ocr)
        if ratio < 0.72:
            issues.append(issue("ocr_native_disagreement", "warn", f"native/OCR similarity {ratio:.2f}"))
    if selected and native and selected != native:
        ratio = similarity(selected, native)
        if ratio < 0.80:
            issues.append(issue("selected_differs_from_native", "warn", f"selected/native similarity {ratio:.2f}"))
    if selected and ocr and selected != ocr:
        ratio = similarity(selected, ocr)
        if ratio < 0.80:
            issues.append(issue("selected_differs_from_ocr", "warn", f"selected/OCR similarity {ratio:.2f}"))


def add_math_symbol_loss_issues(
    issues: list[dict[str, Any]],
    selected: str,
    native: str,
    ocr: str,
    failure_tags: list[str],
    expectations: list[str],
) -> None:
    expectation_text = " ".join(expectations).lower()
    math_expected = bool(set(failure_tags) & MATH_LOSS_TAGS) or any(
        marker in expectation_text
        for marker in (
            "theta",
            "integral",
            "radical",
            "square-root",
            "fraction",
            "denominator",
            "dy/dx",
            "vector",
            "inequality",
            "absolute value",
            "cos",
            "sin",
            "sigma",
        )
    )
    if not math_expected:
        return
    raw = native + " " + ocr
    selected_symbols = math_signal_count(selected)
    raw_symbols = math_signal_count(raw)
    suspicious_text = any(fragment in selected for fragment in ("?", "@", "¿")) or bool(re.search(r"\b\d+[a-z]{2,}\b", selected))
    if raw_symbols and selected_symbols < raw_symbols * 0.45:
        issues.append(
            issue(
                "likely_math_symbol_loss",
                "fail",
                f"selected math signal count {selected_symbols} is low versus raw {raw_symbols}",
            )
        )
    elif suspicious_text:
        issues.append(issue("likely_math_symbol_loss", "warn", "selected text contains suspicious math glyph substitutions"))


def add_contamination_issues(issues: list[dict[str, Any]], selected: str, question_number: str) -> None:
    if not question_number.isdigit():
        return
    next_number = str(int(question_number) + 1)
    trailing_next_question = re.search(rf"\[\d+\]\s+{re.escape(next_number)}\s+[A-Z][a-z]", selected)
    page_furniture = re.search(r"cambridge international|additional materials|answer all questions", selected, re.IGNORECASE)
    if trailing_next_question or page_furniture:
        issues.append(issue("likely_next_question_contamination", "fail", "selected text includes likely page furniture or next question text"))


def add_structural_rejection_issues(
    issues: list[dict[str, Any]],
    record: dict[str, Any],
    existing_flags: list[str],
) -> None:
    status = str(record.get("text_fidelity_status") or "")
    role = str(record.get("question_text_role") or "")
    trust = str(record.get("question_text_trust") or "")
    rejection_flags = sorted(set(existing_flags) & STRUCTURAL_REJECTION_FLAGS)
    if rejection_flags or status in {"degraded", "unusable"} or role == "untrusted_math_text" or trust in {"low", "unusable"}:
        details = rejection_flags or [f"text_fidelity_status={status}" if status else "low trust structural metadata"]
        issues.append(
            issue(
                "selected_source_rejected_by_structural_checks",
                "fail",
                "; ".join(details),
            )
        )


def add_expected_requirement_issues(
    issues: list[dict[str, Any]],
    selected: str,
    question_number: str,
    expectations: list[str],
) -> None:
    for expectation in expectations:
        result = check_expectation(selected, question_number, expectation)
        if result is not None and not result["passed"]:
            issues.append(
                issue(
                    "expected_structural_requirement_missing",
                    "fail",
                    result["message"],
                    requirement_code=result["requirement_code"],
                    expectation=expectation,
                )
            )


def check_expectation(selected: str, question_number: str, expectation: str) -> dict[str, Any] | None:
    lower = expectation.lower()
    compact = compact_text(selected)

    if "starts with question number" in lower:
        passed = QUESTION_NUMBER_RE.search(selected) is not None and QUESTION_NUMBER_RE.search(selected).group("number") == question_number
        return expectation_result("question_number_start", passed, f"selected text does not start with question number {question_number}")
    if "contains question number" in lower:
        passed = bool(question_number and re.search(rf"(?:^|\D){re.escape(question_number)}(?:\D|$)", normalize_for_match(selected)))
        return expectation_result("question_number_present", passed, f"selected text does not contain question number {question_number}")
    if "mark bracket" in lower:
        expected = MARK_BRACKET_RE.search(expectation)
        if expected:
            passed = expected.group("marks") in set(MARK_BRACKET_RE.findall(selected))
            return expectation_result("specific_mark_bracket", passed, f"missing expected mark bracket [{expected.group('marks')}]")
        return expectation_result("any_mark_bracket", bool(MARK_BRACKET_RE.search(selected)), "missing any mark bracket")
    if "subpart order" in lower:
        lower_selected = selected.lower()
        a_index = lower_selected.find("(a)")
        b_index = lower_selected.find("(b)")
        return expectation_result(
            "subpart_order",
            a_index != -1 and b_index != -1 and a_index < b_index,
            "subpart labels are missing or out of order",
        )
    if "subpart" in lower:
        return expectation_result("subpart_labels", len(set(SUBPART_RE.findall(selected.lower()))) >= 2, "missing clear subpart labels")
    if "theta" in lower:
        return expectation_result("theta_symbol", "θ" in selected or "theta" in lower_ascii(selected), "missing theta symbol/text")
    if "integral sign" in lower:
        return expectation_result("integral_sign", "∫" in selected or "|" in selected, "missing integral sign")
    if "dy/dx" in lower or "derivative notation" in lower:
        return expectation_result("derivative_notation", "dy/dx" in lower_ascii(selected), "missing dy/dx notation")
    if "absolute value bars" in lower:
        return expectation_result("absolute_value_bars", "|" in selected, "missing absolute value bars")
    if "'<'" in lower or "< direction" in lower:
        return expectation_result("less_than_direction", "<" in selected, "missing less-than sign")
    if "x > 0" in lower:
        return expectation_result("x_greater_than_zero", "x>0" in compact, "missing x > 0 inequality")
    if "factors (2x - 1) and (x - 3)" in lower:
        return expectation_result("partial_fraction_factors", "2x-1" in compact and "x-3" in compact, "missing expected denominator factors")
    if "denominator 1 - 2i" in lower:
        return expectation_result("complex_denominator", "1-2i" in compact, "missing signed denominator 1 - 2i")
    if "point (4, 5)" in lower:
        return expectation_result("point_4_5", "(4,5)" in compact, "missing point (4, 5)")
    if "sigma" in lower:
        return expectation_result("sigma_symbol", "σ" in selected or "sigma" in lower_ascii(selected), "missing sigma symbol/text")
    if "does not begin selected text at (b)" in lower:
        return expectation_result("subpart_start", not selected.strip().lower().startswith("(b)"), "selected text begins at subpart (b)")
    if "does not truncate" in lower:
        return expectation_result("not_truncated", len(tokens(selected)) >= 12, "selected text looks truncated")

    return None


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Text Fidelity Fixture Baseline",
        "",
        "This report scores advisory text only. Canonical question and mark-scheme images remain the source of truth.",
        "",
        "## Summary",
        "",
        f"- Records: {report['record_count']}",
        f"- Status counts: {format_counts(report['status_counts'])}",
        f"- Failure type counts: {format_counts(report['failure_type_counts'])}",
    ]
    if report.get("normalized_advisory_candidates_included"):
        summary = report.get("normalization_summary") or {}
        lines.extend(
            [
                "- Advisory normalized candidates: included; raw native/OCR/selected text is preserved per fixture",
                f"- Normalization classifications: {format_counts(summary.get('classification_counts') or {})}",
                f"- Fixtures with measurable normalized improvement: {summary.get('measurable_improvement_count', 0)}",
            ]
        )
    lines.extend(["", "## Top Failure Types", "", "| Failure type | Count |", "| --- | ---: |"])
    for row in report["top_failure_types"]:
        lines.append(f"| {row['failure_type']} | {row['count']} |")

    lines.extend(["", "## Per-Fixture Details", ""])
    if report.get("normalized_advisory_candidates_included"):
        lines.extend(
            [
                "| Record | Status | Issues | Measurable targets | Normalized classification | Flags | Confidence |",
                "| --- | --- | --- | --- | --- | --- | ---: |",
            ]
        )
    else:
        lines.extend(["| Record | Status | Issues | Measurable targets |", "| --- | --- | --- | --- |"])
    for record in report["records"]:
        issue_codes = ", ".join(issue["code"] for issue in record["issues"]) or "none"
        targets = ", ".join(record["measurable_improvement_targets"]) or "none"
        if report.get("normalized_advisory_candidates_included"):
            flags = ", ".join(record["normalization_flags"]) or "none"
            lines.append(
                f"| {record['record_id']} | {record['status']} | {issue_codes} | {targets} | "
                f"{record['normalization_classification']} | {flags} | {record['normalization_confidence']:.2f} |"
            )
        else:
            lines.append(f"| {record['record_id']} | {record['status']} | {issue_codes} | {targets} |")

    if report.get("normalized_advisory_candidates_included"):
        lines.extend(
            [
                "",
                "## Advisory Normalization Examples",
                "",
                "These normalized strings are candidates for review only; they are not canonical question text.",
                "",
                "| Record | Classification | Resolved issue keys | Warnings |",
                "| --- | --- | --- | --- |",
            ]
        )
        improved_records = [
            record
            for record in report["records"]
            if record.get("normalization_classification") in {"improved", "clearer_failure_classification"}
        ][:12]
        for record in improved_records:
            resolved = ", ".join(record.get("normalization_resolved_issue_keys") or []) or "none"
            warnings = ", ".join(record.get("normalization_warnings") or []) or "none"
            lines.append(f"| {record['record_id']} | {record['normalization_classification']} | {resolved} | {warnings} |")

    lines.extend(["", "## Expected Requirement Gaps", ""])
    if report["expected_requirement_missing_counts"]:
        lines.extend([f"- {key}: {value}" for key, value in report["expected_requirement_missing_counts"].items()])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def lower_ascii(value: str) -> str:
    return value.lower()


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower().replace("−", "-").replace("—", "-")


def tokens(value: str) -> list[str]:
    return WORD_RE.findall(value.lower())


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_for_match(left), normalize_for_match(right)).ratio()


def math_signal_count(value: str) -> int:
    return len(re.findall(r"[∫θσπ≤≥<>|^_*/=+\-−]|\b(?:sin|cos|tan|ln|sqrt|var|dx|dy)\b", value.lower()))


def issue(code: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "code": code,
        "failure_type": FAILURE_TYPE_BY_CODE[code],
        "severity": severity,
        "message": message,
        **extra,
    }


def expectation_result(requirement_code: str, passed: bool, message: str) -> dict[str, Any]:
    return {"requirement_code": requirement_code, "passed": passed, "message": message}


def dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in issues:
        key = (item["code"], item.get("requirement_code", ""), item["message"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def issue_key(item: dict[str, Any]) -> str:
    if item.get("code") == "expected_structural_requirement_missing":
        return f"{item['code']}:{item.get('requirement_code', '')}"
    return str(item["code"])


def measurable_improvement_targets(issues: list[dict[str, Any]]) -> list[str]:
    return sorted({issue["failure_type"] for issue in issues if issue["severity"] in {"fail", "warn"}})


def classify_normalization_result(resolved: list[str], introduced: list[str], flags: list[str]) -> str:
    if resolved and not introduced:
        return "improved"
    if resolved and introduced:
        return "mixed"
    if flags:
        return "clearer_failure_classification"
    return "unchanged"


def build_normalization_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    classification_counts = Counter(record.get("normalization_classification", "missing") for record in records)
    flag_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for record in records:
        flag_counts.update(record.get("normalization_flags") or [])
        warning_counts.update(record.get("normalization_warnings") or [])
    measurable = [
        record["record_id"]
        for record in records
        if record.get("normalization_classification") in {"improved", "clearer_failure_classification"}
    ]
    return {
        "classification_counts": dict(sorted(classification_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "measurable_improvement_count": len(measurable),
        "measurable_improvement_record_ids": measurable,
        "note": "Normalized candidates are advisory report fields only and do not overwrite raw or selected text.",
    }


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
