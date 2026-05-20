from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SCHEMA_NAME = "exam_bank.text_fidelity.review_queue"
SCHEMA_VERSION = 1

QUESTION_NUMBER_RE = re.compile(r"^\s*(?:[A-Za-z]\s+)?(?P<number>\d{1,2})\b")
MARK_BRACKET_RE = re.compile(r"\[(?P<marks>\d{1,2})\]")
SUBPART_RE = re.compile(r"\(([a-z])\)")
WORD_RE = re.compile(r"[a-z0-9]+")

REASON_WEIGHTS = {
    "known_fixture_membership": 200,
    "selected_ocr_with_structural_warnings": 70,
    "next_question_contamination": 70,
    "clean_visual_crop_but_degraded_text": 65,
    "missing_question_number": 60,
    "lost_subpart_labels": 55,
    "missing_marks": 50,
    "likely_math_symbol_loss": 45,
    "ocr_native_disagreement": 40,
    "suspiciously_short_text": 35,
}

STRUCTURAL_WARNING_FLAGS = {
    "question_scope_contaminated",
    "question_subparts_incomplete",
    "weak_question_anchor",
    "weak_question_text",
    "ocr_merged_sparse_lower_region",
    "hybrid_math_text_requires_review",
    "math_text_corruption_detected",
    "native_compacted_math_corruption",
    "ocr_noise_fragment_present",
    "sparse_or_merged_question_text",
    "weak_extracted_text",
}

MATH_WARNING_FLAGS = {
    "broken_superscript_or_power",
    "flattened_display_math",
    "heavy_math_density",
    "likely_needs_visual_review",
    "math_corruption_suspected",
    "suspicious_symbol_run",
    "contains_equation_layout",
    "contains_flattened_math_structure",
}

MATH_TERMS = (
    "sin",
    "cos",
    "tan",
    "theta",
    "integral",
    "vector",
    "fraction",
    "denominator",
    "sigma",
    "dy/dx",
)


def load_question_bank(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Question bank must be a list or contain a questions list.")
    return records


def load_fixture_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    return {str(record.get("record_id") or record.get("question_id")) for record in records if record.get("record_id") or record.get("question_id")}


def build_review_queue(records: list[dict[str, Any]], fixture_ids: set[str] | None = None) -> dict[str, Any]:
    fixture_ids = fixture_ids or set()
    entries = [score_record(record, fixture_ids) for record in records]
    entries.sort(key=entry_sort_key)
    for index, entry in enumerate(entries, start=1):
        entry["rank"] = index

    reason_counts: Counter[str] = Counter()
    for entry in entries:
        reason_counts.update(entry["reason_codes"])

    fixture_entries = [entry for entry in entries if entry["record_id"] in fixture_ids]
    fixtures_outside_top_50 = [
        {
            "record_id": entry["record_id"],
            "rank": entry["rank"],
            "priority_score": entry["priority_score"],
            "reason_codes": entry["reason_codes"],
            "explanation": "Fixture was boosted but ranked below the top 50 because other records had more concrete concurrent failure signals.",
        }
        for entry in fixture_entries
        if entry["rank"] > 50
    ]

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "record_count": len(entries),
        "queued_count": sum(1 for entry in entries if entry["priority_score"] > 0),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "top_reason_codes": [
            {"reason_code": reason, "count": count}
            for reason, count in reason_counts.most_common()
        ],
        "fixture_summary": {
            "known_fixture_count": len(fixture_ids),
            "known_fixtures_found": len(fixture_entries),
            "known_fixtures_missing_from_bank": sorted(fixture_ids - {entry["record_id"] for entry in fixture_entries}),
            "known_fixtures_in_top_50": sum(1 for entry in fixture_entries if entry["rank"] <= 50),
            "known_fixtures_in_top_100": sum(1 for entry in fixture_entries if entry["rank"] <= 100),
            "fixtures_outside_top_50": fixtures_outside_top_50,
        },
        "top_50": entries[:50],
        "entries": entries,
    }


def score_record(record: dict[str, Any], fixture_ids: set[str] | None = None) -> dict[str, Any]:
    fixture_ids = fixture_ids or set()
    record_id = str(record.get("question_id") or record.get("record_id") or "")
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    selected = text_value(record.get("question_text"))
    ocr = text_value(record.get("ocr_text"))
    question_number = str(record.get("question_number") or "").strip()
    reasons: list[dict[str, Any]] = []

    add_known_fixture_reason(reasons, record_id, fixture_ids)
    add_ocr_native_disagreement_reason(reasons, selected, ocr, notes)
    add_missing_question_number_reason(reasons, selected, question_number)
    add_missing_marks_reason(reasons, selected, record, notes)
    add_lost_subparts_reason(reasons, selected, record, notes)
    add_short_text_reason(reasons, selected, notes)
    add_math_symbol_loss_reason(reasons, selected, ocr, record, notes)
    add_selected_ocr_warning_reason(reasons, notes)
    add_clean_crop_degraded_reason(reasons, notes)
    add_next_question_contamination_reason(reasons, selected, question_number, notes)

    reasons = dedupe_reasons(reasons)
    priority_score = sum(reason["weight"] for reason in reasons)
    reason_codes = [reason["code"] for reason in reasons]

    return {
        "rank": None,
        "record_id": record_id,
        "paper_id": record.get("paper"),
        "paper_family": record.get("paper_family"),
        "question_number": question_number,
        "priority_score": priority_score,
        "reason_codes": reason_codes,
        "reasons": reasons,
        "text_source_profile": notes.get("text_source_profile"),
        "text_candidate_source": notes.get("text_candidate_source"),
        "ocr_selected": bool(notes.get("ocr_selected")),
        "text_fidelity_status": notes.get("text_fidelity_status"),
        "question_text_trust": record.get("question_text_trust") or notes.get("question_text_trust"),
        "question_crop_confidence": notes.get("question_crop_confidence"),
        "selected_text_score": notes.get("selected_text_score"),
        "native_text_score": notes.get("native_text_score"),
        "ocr_text_score": notes.get("ocr_text_score"),
        "question_image_path": record.get("question_image_path"),
        "mark_scheme_image_path": record.get("mark_scheme_image_path"),
        "selected_text_excerpt": excerpt(selected),
    }


def add_known_fixture_reason(reasons: list[dict[str, Any]], record_id: str, fixture_ids: set[str]) -> None:
    if record_id in fixture_ids:
        reasons.append(reason("known_fixture_membership", "Record is in the frozen bad-text fixture manifest."))


def add_ocr_native_disagreement_reason(reasons: list[dict[str, Any]], selected: str, ocr: str, notes: dict[str, Any]) -> None:
    if not selected or not ocr:
        return
    score_gap = abs(int_value(notes.get("native_text_score")) - int_value(notes.get("ocr_text_score")))
    similarity = text_similarity(selected, ocr)
    if similarity < 0.72 or score_gap >= 30:
        reasons.append(
            reason(
                "ocr_native_disagreement",
                f"OCR/native candidate diagnostics disagree; selected/OCR similarity={similarity:.2f}, score_gap={score_gap}.",
                evidence={"similarity": round(similarity, 3), "score_gap": score_gap},
            )
        )


def add_missing_question_number_reason(reasons: list[dict[str, Any]], selected: str, question_number: str) -> None:
    if not question_number:
        reasons.append(reason("missing_question_number", "Record has no expected question number metadata."))
        return
    if not contains_number_token(selected, question_number):
        reasons.append(reason("missing_question_number", f"Selected text does not contain expected question number {question_number}."))
        return
    first = QUESTION_NUMBER_RE.search(selected)
    if not first or first.group("number") != question_number:
        reasons.append(reason("missing_question_number", f"Expected question number {question_number} is not the leading anchor."))


def add_missing_marks_reason(reasons: list[dict[str, Any]], selected: str, record: dict[str, Any], notes: dict[str, Any]) -> None:
    if not expects_mark_bracket(record, notes):
        return
    present = {int(value) for value in MARK_BRACKET_RE.findall(selected)}
    if not present:
        reasons.append(reason("missing_marks", "Selected text has no [n] mark bracket despite mark metadata."))


def add_lost_subparts_reason(reasons: list[dict[str, Any]], selected: str, record: dict[str, Any], notes: dict[str, Any]) -> None:
    expected = expected_subparts(record, notes)
    if not expected:
        return
    present = set(SUBPART_RE.findall(selected.lower()))
    missing = sorted(expected - present)
    lower = selected.lower()
    starts_at_b = lower.strip().startswith("(b)")
    b_before_a = "(b)" in lower and "(a)" in lower and lower.index("(b)") < lower.index("(a)")
    if missing or starts_at_b or b_before_a:
        detail = f"Expected subparts {sorted(expected)}, present {sorted(present)}."
        if starts_at_b:
            detail += " Selected text begins at (b)."
        if b_before_a:
            detail += " Subpart (b) appears before (a)."
        reasons.append(reason("lost_subpart_labels", detail))


def add_short_text_reason(reasons: list[dict[str, Any]], selected: str, notes: dict[str, Any]) -> None:
    word_count = len(tokens(selected))
    selected_score = int_value(notes.get("selected_text_score"))
    structure = structure_metadata(notes)
    detected_length = structure.get("text_length")
    if word_count < 8:
        reasons.append(reason("suspiciously_short_text", f"Selected text has only {word_count} words."))
    elif selected_score <= 40:
        reasons.append(reason("suspiciously_short_text", f"Selected text score is {selected_score}."))
    elif isinstance(detected_length, int) and detected_length < 60:
        reasons.append(reason("suspiciously_short_text", f"Structure detector text_length is {detected_length}."))


def add_math_symbol_loss_reason(
    reasons: list[dict[str, Any]],
    selected: str,
    ocr: str,
    record: dict[str, Any],
    notes: dict[str, Any],
) -> None:
    flags = set(str(flag) for flag in notes.get("extraction_quality_flags") or [])
    flags.update(str(flag) for flag in record.get("visual_reason_flags") or [])
    rejected = " ".join(str(value) for value in notes.get("ocr_rejected_reasons") or [])
    selected_math = math_signal_count(selected)
    ocr_math = math_signal_count(ocr)
    suspicious_glyphs = bool(re.search(r"[?@¿]|\b\d+[a-z]{2,}\b", selected))
    math_context = bool(flags & MATH_WARNING_FLAGS) or any(term in (selected + " " + ocr + " " + rejected).lower() for term in MATH_TERMS)
    if not math_context:
        return
    if suspicious_glyphs or (ocr_math and selected_math < ocr_math * 0.55) or "lost" in rejected or "corruption" in rejected:
        reasons.append(
            reason(
                "likely_math_symbol_loss",
                "Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings.",
                evidence={
                    "selected_math_signal_count": selected_math,
                    "ocr_math_signal_count": ocr_math,
                    "flags": sorted(flags & MATH_WARNING_FLAGS),
                    "ocr_rejected_reasons": notes.get("ocr_rejected_reasons") or [],
                },
            )
        )


def add_selected_ocr_warning_reason(reasons: list[dict[str, Any]], notes: dict[str, Any]) -> None:
    if not notes.get("ocr_selected"):
        return
    flags = selector_warning_flags(notes)
    if flags:
        reasons.append(
            reason(
                "selected_ocr_with_structural_warnings",
                "OCR was selected while selector or structure metadata contains warnings.",
                evidence={"warning_flags": sorted(flags)},
            )
        )


def add_clean_crop_degraded_reason(reasons: list[dict[str, Any]], notes: dict[str, Any]) -> None:
    high_crop = notes.get("question_crop_confidence") == "high"
    degraded = notes.get("text_fidelity_status") == "degraded" or int_value(notes.get("selected_text_score")) < 50
    if high_crop and degraded:
        reasons.append(
            reason(
                "clean_visual_crop_but_degraded_text",
                "Question crop confidence is high while selected advisory text is degraded or low-scoring.",
            )
        )


def add_next_question_contamination_reason(
    reasons: list[dict[str, Any]],
    selected: str,
    question_number: str,
    notes: dict[str, Any],
) -> None:
    structure = structure_metadata(notes)
    indicators = structure.get("contamination_indicators") if isinstance(structure.get("contamination_indicators"), dict) else {}
    foreign = indicators.get("foreign_question_anchors") or []
    next_number = str(int(question_number) + 1) if question_number.isdigit() else ""
    text_pattern = bool(next_number and re.search(rf"\[\d+\]\s+{re.escape(next_number)}\s+[A-Z][A-Za-z]", selected))
    if structure.get("contamination_detected") or foreign or text_pattern:
        reasons.append(
            reason(
                "next_question_contamination",
                "Selected text or structure metadata indicates possible next-question contamination.",
                evidence={"foreign_question_anchors": foreign[:5]},
            )
        )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Text Fidelity Review Queue",
        "",
        "This queue ranks advisory text records for manual fidelity review. Canonical question and mark-scheme images remain the source of truth.",
        "",
        "## Summary",
        "",
        f"- Records inspected: {report['record_count']}",
        f"- Records with non-zero queue score: {report['queued_count']}",
        f"- Known bad fixtures found: {report['fixture_summary']['known_fixtures_found']} / {report['fixture_summary']['known_fixture_count']}",
        f"- Known bad fixtures in top 50: {report['fixture_summary']['known_fixtures_in_top_50']}",
        f"- Known bad fixtures in top 100: {report['fixture_summary']['known_fixtures_in_top_100']}",
        f"- Top reason codes: {format_top_reason_codes(report['top_reason_codes'])}",
        "",
        "## Reason Weights",
        "",
        "| Reason code | Weight |",
        "| --- | ---: |",
    ]
    for code, weight in sorted(REASON_WEIGHTS.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{code}` | {weight} |")

    lines.extend(["", "## Top 50 Review Items", "", "| Rank | Record | Score | Reasons | Explanation |", "| ---: | --- | ---: | --- | --- |"])
    for entry in report["top_50"]:
        explanation = "; ".join(reason["detail"] for reason in entry["reasons"][:3])
        lines.append(
            f"| {entry['rank']} | `{entry['record_id']}` | {entry['priority_score']} | {', '.join(f'`{code}`' for code in entry['reason_codes'])} | {escape_table(explanation)} |"
        )

    fixture_summary = report["fixture_summary"]
    lines.extend(["", "## Fixture Rank Summary", ""])
    if fixture_summary["known_fixtures_missing_from_bank"]:
        lines.append(f"- Missing fixture records: {', '.join(fixture_summary['known_fixtures_missing_from_bank'])}")
    if fixture_summary["fixtures_outside_top_50"]:
        lines.extend(["- Some known bad fixtures are outside the top 50; each retained the fixture boost, but other records accumulated more concurrent concrete signals.", ""])
        lines.extend(["| Fixture | Rank | Score | Reasons | Explanation |", "| --- | ---: | ---: | --- | --- |"])
        for fixture in fixture_summary["fixtures_outside_top_50"]:
            lines.append(
                f"| `{fixture['record_id']}` | {fixture['rank']} | {fixture['priority_score']} | {', '.join(f'`{code}`' for code in fixture['reason_codes'])} | {fixture['explanation']} |"
            )
    else:
        lines.append("- All known bad fixtures found in the bank ranked in the top 50.")

    lines.extend(["", "## Advisory Boundary", "", "The queue is a review aid only. It does not change OCR/native selection, production exports, canonical images, or `question_bank.json`."])
    return "\n".join(lines) + "\n"


def reason(code: str, detail: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "weight": REASON_WEIGHTS[code], "detail": detail, "evidence": evidence or {}}


def dedupe_reasons(reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in reasons:
        if item["code"] in seen:
            continue
        seen.add(item["code"])
        deduped.append(item)
    return deduped


def entry_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (-entry["priority_score"], entry["record_id"])


def expects_mark_bracket(record: dict[str, Any], notes: dict[str, Any]) -> bool:
    total = int_value(record.get("question_solution_marks"))
    if total > 0 or record.get("subparts_solution_marks"):
        return True
    structure = structure_metadata(notes)
    return bool(structure.get("mark_values_detected") or structure.get("question_total_detected"))


def expected_subparts(record: dict[str, Any], notes: dict[str, Any]) -> set[str]:
    expected = {str(part).lower() for part in record.get("subparts") or [] if str(part).strip()}
    expected.update(str(part).lower() for part in (record.get("subparts_solution_marks") or {}).keys())
    structure = structure_metadata(notes)
    expected.update(str(part).lower() for part in structure.get("subparts") or [])
    return expected


def structure_metadata(notes: dict[str, Any]) -> dict[str, Any]:
    structure = notes.get("question_structure_detected")
    return structure if isinstance(structure, dict) else {}


def selector_warning_flags(notes: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    for key in ("review_flags", "validation_flags", "extraction_quality_flags", "text_fidelity_flags"):
        flags.update(str(flag) for flag in notes.get(key) or [])
    structure = structure_metadata(notes)
    if structure.get("weak_anchor"):
        flags.add("weak_question_anchor")
    if structure.get("missing_internal_subparts") or structure.get("subpart_sequence_gap"):
        flags.add("question_subparts_incomplete")
    if structure.get("contamination_detected"):
        flags.add("question_scope_contaminated")
    return flags & STRUCTURAL_WARNING_FLAGS


def text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def tokens(value: str) -> list[str]:
    return WORD_RE.findall(value.lower())


def contains_number_token(selected: str, question_number: str) -> bool:
    return re.search(rf"(?:^|\D){re.escape(question_number)}(?:\D|$)", selected) is not None


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, compact_text(left), compact_text(right)).ratio()


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def math_signal_count(value: str) -> int:
    lower = value.lower()
    count = sum(lower.count(term) for term in MATH_TERMS)
    count += len(re.findall(r"[∫θπσ√≤≥<>^*/=|]", value))
    count += len(re.findall(r"_\{|\^\{", value))
    return count


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def excerpt(value: str, limit: int = 220) -> str:
    clean = re.sub(r"\s+", " ", value).strip()
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def format_top_reason_codes(rows: list[dict[str, Any]], limit: int = 6) -> str:
    if not rows:
        return "none"
    return ", ".join(f"{row['reason_code']}={row['count']}" for row in rows[:limit])
