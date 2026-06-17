from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CANONICAL_YEAR_START = 2008
CANONICAL_YEAR_END = 2020
DEFAULT_INPUT_PATH = Path("exam_bank_input.jsonl")
DEFAULT_SAMPLE_OUTPUT_PATH = Path("output/samples/canonical_one_per_type_2008_2020.jsonl")
DEFAULT_REPORT_OUTPUT_PATH = Path("output/samples/canonical_sample_report.json")

PAPER_TYPE_LABELS = {
    "pure_math_1": "P1",
    "pure_math_3": "P3",
    "mechanics_1": "M1",
    "statistics_1": "S1",
}
TARGET_PAPER_TYPES = tuple(PAPER_TYPE_LABELS)
REQUIRED_FIELDS = (
    "board",
    "component",
    "id",
    "mark_scheme_url",
    "paper",
    "paper_name",
    "qualification",
    "question_paper_url",
    "session",
    "session_code",
    "source",
    "source_page",
    "subject",
    "syllabus",
    "year",
)
TEXT_SIGNAL_FIELDS = (
    "question_text",
    "combined_question_text",
    "questions",
    "question_count",
    "mark_scheme_text",
    "markscheme_text",
    "markscheme_mapping_status",
)
SESSION_TIEBREAK_RANK = {"m": 0, "s": 1, "w": 2}
VARIANT_TIEBREAK_RANK = {2: 0, "2": 0, None: 1, 1: 2, "1": 2, 3: 3, "3": 3}


@dataclass(frozen=True)
class CandidateScore:
    score: int
    score_components: dict[str, int]
    quality_notes: list[str]
    broken_fields: list[str]
    tiebreak_key: tuple[Any, ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_canonical_sample(
    rows: list[dict[str, Any]],
    *,
    year_start: int = CANONICAL_YEAR_START,
    year_end: int = CANONICAL_YEAR_END,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_keys = {_row_identity_key(row) for row in rows}
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    out_of_scope_count = 0
    duplicate_ids = _duplicate_ids(rows)

    for row in rows:
        year = row.get("year")
        paper = row.get("paper")
        if isinstance(year, int) and year_start <= year <= year_end and paper in TARGET_PAPER_TYPES:
            grouped[(year, str(paper))].append(row)
        else:
            out_of_scope_count += 1

    selected: list[dict[str, Any]] = []
    years: dict[str, Any] = {}
    missing_types_total = 0
    skipped_total = 0

    for year in range(year_start, year_end + 1):
        year_entry: dict[str, Any] = {
            "selected": {},
            "skipped": {},
            "missing_types": [],
            "available_types": [],
        }
        for paper_type in TARGET_PAPER_TYPES:
            candidates = grouped.get((year, paper_type), [])
            paper_label = PAPER_TYPE_LABELS[paper_type]
            if not candidates:
                year_entry["missing_types"].append(paper_label)
                missing_types_total += 1
                continue
            year_entry["available_types"].append(paper_label)

            ranked = sorted(
                ((score_candidate(candidate, duplicate_ids=duplicate_ids), candidate) for candidate in candidates),
                key=lambda item: (-item[0].score, item[0].tiebreak_key),
            )
            best_score, best = ranked[0]
            if _row_identity_key(best) not in source_keys:
                raise ValueError(f"Selected row does not map back to source dataset: {best.get('id')}")
            selected.append(best)
            year_entry["selected"][paper_label] = _report_candidate(best, best_score)

            skipped: list[dict[str, Any]] = []
            for score, candidate in ranked[1:]:
                skipped.append(
                    {
                        **_report_candidate(candidate, score),
                        "skip_reason": _skip_reason(score, best_score),
                    }
                )
            year_entry["skipped"][paper_label] = skipped
            skipped_total += len(skipped)

        years[str(year)] = year_entry

    validate_sample(rows, selected, year_start=year_start, year_end=year_end)

    per_year_coverage = {
        year: {
            "selected_count": len(entry["selected"]),
            "selected_types": sorted(entry["selected"]),
            "missing_types": entry["missing_types"],
        }
        for year, entry in years.items()
    }
    report = {
        "schema_name": "exam_bank.canonical_one_per_type_sample_report",
        "schema_version": 1,
        "source": {
            "row_count": len(rows),
            "out_of_scope_row_count": out_of_scope_count,
            "year_start": year_start,
            "year_end": year_end,
            "target_paper_types": PAPER_TYPE_LABELS,
            "selection_limitation": (
                "Source rows are paper-level manifest records. Direct question text completeness, "
                "visual dependency, and mark-scheme text coverage fields are not present, so scoring "
                "uses available provenance, paired question-paper/mark-scheme URLs, and parse-field integrity."
            ),
        },
        "summary": {
            "total_selected": len(selected),
            "skipped_candidate_count": skipped_total,
            "missing_types_count": missing_types_total,
            "years_covered": len(years),
            "per_year_coverage": per_year_coverage,
        },
        "years": years,
    }
    return selected, report


def validate_sample(
    source_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    *,
    year_start: int = CANONICAL_YEAR_START,
    year_end: int = CANONICAL_YEAR_END,
) -> None:
    source_keys = {_row_identity_key(row) for row in source_rows}
    available = {
        (int(row["year"]), str(row["paper"]))
        for row in source_rows
        if isinstance(row.get("year"), int)
        and year_start <= int(row["year"]) <= year_end
        and row.get("paper") in TARGET_PAPER_TYPES
    }
    seen: set[tuple[int, str]] = set()
    for row in selected_rows:
        key = _row_identity_key(row)
        if key not in source_keys:
            raise ValueError(f"Selected row not present in source dataset: {row.get('id')}")
        year = row.get("year")
        paper = row.get("paper")
        if not isinstance(year, int) or not (year_start <= year <= year_end):
            raise ValueError(f"Selected row has out-of-range year: {row.get('id')}")
        if paper not in TARGET_PAPER_TYPES:
            raise ValueError(f"Selected row has unsupported paper type: {row.get('id')}")
        group_key = (year, str(paper))
        if group_key in seen:
            raise ValueError(f"Duplicate selected year/type: {year} {paper}")
        seen.add(group_key)
    missing_available = available - seen
    if missing_available:
        missing = ", ".join(f"{year}:{paper}" for year, paper in sorted(missing_available))
        raise ValueError(f"Missing selections for available year/types: {missing}")


def score_candidate(row: dict[str, Any], *, duplicate_ids: set[str] | None = None) -> CandidateScore:
    duplicate_ids = duplicate_ids or set()
    broken_fields = _broken_fields(row)
    components = {
        "parse_field_integrity": 100 if not broken_fields else -100 * len(broken_fields),
        "question_paper_url_present": 40 if _has_url(row.get("question_paper_url")) else -80,
        "mark_scheme_url_present": 60 if _has_url(row.get("mark_scheme_url")) else -120,
        "paired_component_match": 30 if _paired_component_match(row) else 0,
        "source_page_present": 10 if _has_url(row.get("source_page")) else 0,
        "stable_identifier": 10 if isinstance(row.get("id"), str) and row["id"].strip() else -50,
        "unique_identifier": 10 if row.get("id") not in duplicate_ids else -50,
        "text_signal_fields_present": _text_signal_score(row),
    }
    notes = []
    if not any(field in row for field in TEXT_SIGNAL_FIELDS):
        notes.append("paper_manifest_has_no_question_text_or_mark_scheme_text_fields")
    if broken_fields:
        notes.append("broken_required_fields_detected")
    if not _has_url(row.get("mark_scheme_url")):
        notes.append("missing_mark_scheme_url")
    if not _has_url(row.get("question_paper_url")):
        notes.append("missing_question_paper_url")
    return CandidateScore(
        score=sum(components.values()),
        score_components=components,
        quality_notes=notes,
        broken_fields=broken_fields,
        tiebreak_key=_tiebreak_key(row),
    )


def build_and_write(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    sample_output_path: Path = DEFAULT_SAMPLE_OUTPUT_PATH,
    report_output_path: Path = DEFAULT_REPORT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    selected, report = build_canonical_sample(rows)
    write_jsonl(sample_output_path, selected)
    report = {
        **report,
        "outputs": {
            "sample_jsonl": str(sample_output_path),
            "report_json": str(report_output_path),
        },
    }
    write_json(report_output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical one-paper-per-type sample for CAIE 9709.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE_OUTPUT_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    report = build_and_write(
        input_path=args.input,
        sample_output_path=args.sample_output,
        report_output_path=args.report_output,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _report_candidate(row: dict[str, Any], score: CandidateScore) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "year": row.get("year"),
        "type": PAPER_TYPE_LABELS.get(str(row.get("paper")), row.get("paper")),
        "paper": row.get("paper"),
        "component": row.get("component"),
        "session": row.get("session"),
        "session_code": row.get("session_code"),
        "variant": row.get("variant"),
        "question_paper_url": row.get("question_paper_url"),
        "mark_scheme_url": row.get("mark_scheme_url"),
        "score": score.score,
        "score_components": score.score_components,
        "quality_notes": score.quality_notes,
        "broken_fields": score.broken_fields,
    }


def _skip_reason(score: CandidateScore, best_score: CandidateScore) -> str:
    if score.score < best_score.score:
        return "lower_quality_score"
    return "quality_tie_lost_deterministic_session_variant_tiebreak"


def _broken_fields(row: dict[str, Any]) -> list[str]:
    broken: list[str] = []
    for field in REQUIRED_FIELDS:
        value = row.get(field)
        if value is None or value == "":
            broken.append(field)
    if not isinstance(row.get("year"), int):
        broken.append("year_type")
    if row.get("paper") not in PAPER_TYPE_LABELS:
        broken.append("paper_type")
    return sorted(set(broken))


def _has_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _paired_component_match(row: dict[str, Any]) -> bool:
    raw_component = str(row.get("component") or "")
    component_tokens = {raw_component, raw_component.zfill(2)}
    qp = str(row.get("question_paper_url") or "").lower()
    ms = str(row.get("mark_scheme_url") or "").lower()
    return bool(raw_component) and any(token in qp and token in ms for token in component_tokens)


def _text_signal_score(row: dict[str, Any]) -> int:
    score = 0
    question_text = row.get("question_text") or row.get("combined_question_text")
    if isinstance(question_text, str) and len(question_text.strip()) >= 200:
        score += 80
    mark_text = row.get("mark_scheme_text") or row.get("markscheme_text")
    if isinstance(mark_text, str) and len(mark_text.strip()) >= 100:
        score += 80
    if row.get("markscheme_mapping_status") == "pass":
        score += 40
    return score


def _tiebreak_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        SESSION_TIEBREAK_RANK.get(row.get("session_code"), 99),
        VARIANT_TIEBREAK_RANK.get(row.get("variant"), 50),
        str(row.get("component") or ""),
        str(row.get("id") or ""),
    )


def _duplicate_ids(rows: list[dict[str, Any]]) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, str):
            counts[row_id] += 1
    return {row_id for row_id, count in counts.items() if count > 1}


def _row_identity_key(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))
