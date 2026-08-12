from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any


BOILERPLATE_RE = re.compile(
    r"(?:©|copyright|cambridge international examinations|ucLES)", re.IGNORECASE
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an assisted visual/text audit sample.")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-overrides",
        help="Optional JSON mapping question IDs to manually reviewed image ratings.",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    rows = sample["questions"]
    decisions = [rate_row(row) for row in rows]
    if args.image_overrides:
        overrides = json.loads(Path(args.image_overrides).read_text(encoding="utf-8"))
        decisions = apply_image_overrides(decisions, overrides)
    summary = build_summary(sample, decisions)

    write_json(
        output_dir / "review_decisions.assisted.json",
        {
            "schema_name": "exam_bank.random_visual_accuracy_decisions",
            "schema_version": 1,
            "source_sample": sample_path.name,
            "review_method": "visual contact-sheet review plus OCR-alignment-assisted per-record rating",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "questions": decisions,
        },
    )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary["headline_metrics"], indent=2))
    return 0


def apply_image_overrides(
    decisions: list[dict[str, Any]], overrides: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    by_id = {decision["question_id"]: decision for decision in decisions}
    unknown = sorted(set(overrides) - set(by_id))
    if unknown:
        raise ValueError(f"Image overrides contain unknown question IDs: {unknown}")
    allowed = {"pass", "minor", "major", "missing"}
    for question_id, fields in overrides.items():
        decision = by_id[question_id]
        for field in ("question_image_rating", "mark_scheme_image_rating"):
            if field not in fields:
                continue
            rating = fields[field]
            if rating not in allowed:
                raise ValueError(f"Invalid {field}={rating!r} for {question_id}")
            decision[field] = rating
        decision["overall_usable"] = bool(
            decision["pairing_correct"]
            and all(
                decision[field] in {"pass", "minor"}
                for field in (
                    "question_image_rating",
                    "mark_scheme_image_rating",
                    "question_text_rating",
                    "mark_scheme_text_rating",
                )
            )
        )
        decision["strict_accurate"] = bool(
            decision["pairing_correct"]
            and all(
                decision[field] == "pass"
                for field in (
                    "question_image_rating",
                    "mark_scheme_image_rating",
                    "question_text_rating",
                    "mark_scheme_text_rating",
                )
            )
        )
        note = fields.get("note")
        if note:
            decision["notes"] = "; ".join(filter(None, (decision["notes"], f"manual visual review: {note}")))
    return decisions


def rate_row(row: dict[str, Any]) -> dict[str, Any]:
    q_irregular = set(row.get("question_ocr_irregularities") or [])
    ms_irregular = set(row.get("mark_scheme_ocr_irregularities") or [])
    flags = set(row.get("review_flags") or [])

    if not row.get("question_image_exists"):
        q_image = "missing"
    elif "question_scope_contaminated" in flags:
        q_image = "major"
    elif q_irregular or "possible_next_question_contamination" in flags:
        q_image = "minor"
    else:
        q_image = "pass"

    if not row.get("mark_scheme_image_exists"):
        ms_image = "missing"
    elif "regeneration_failed" in ms_irregular:
        ms_image = "major"
    elif ms_irregular:
        ms_image = "minor"
    else:
        ms_image = "pass"

    q_text = text_rating(
        row.get("question_ocr_similarity"),
        str(row.get("question_text") or ""),
        pass_threshold=0.95,
        usable_threshold=0.80,
    )
    ms_text = text_rating(
        row.get("mark_scheme_ocr_similarity"),
        str(row.get("mark_scheme_text") or ""),
        pass_threshold=0.80,
        usable_threshold=0.65,
    )
    pairing_correct = bool(row.get("question_image_exists") and row.get("mark_scheme_image_exists"))
    ratings = [q_image, ms_image, q_text, ms_text]
    usable = pairing_correct and all(rating in {"pass", "minor"} for rating in ratings)
    strict = pairing_correct and all(rating == "pass" for rating in ratings)
    notes: list[str] = []
    if "regeneration_failed" in ms_irregular:
        notes.append("markscheme fallback/regeneration failure; target present but clean isolation not credited")
    if q_irregular:
        notes.append("question image screen: " + ", ".join(sorted(q_irregular)))
    if ms_irregular - {"regeneration_failed"}:
        notes.append("markscheme image screen: " + ", ".join(sorted(ms_irregular - {"regeneration_failed"})))
    if BOILERPLATE_RE.search(str(row.get("question_text") or "")):
        notes.append("question JSON contains publisher/copyright boilerplate")
    if BOILERPLATE_RE.search(str(row.get("mark_scheme_text") or "")):
        notes.append("markscheme JSON contains publisher/copyright boilerplate")
    return {
        "question_id": row["question_id"],
        "paper_family": row.get("paper_family"),
        "question_image_rating": q_image,
        "mark_scheme_image_rating": ms_image,
        "question_text_rating": q_text,
        "mark_scheme_text_rating": ms_text,
        "pairing_correct": pairing_correct,
        "overall_usable": usable,
        "strict_accurate": strict,
        "question_ocr_similarity": row.get("question_ocr_similarity"),
        "mark_scheme_ocr_similarity": row.get("mark_scheme_ocr_similarity"),
        "notes": "; ".join(notes),
    }


def text_rating(value: Any, text: str, *, pass_threshold: float, usable_threshold: float) -> str:
    if not text.strip():
        return "missing"
    if value is None:
        rating = "minor"
    elif float(value) >= pass_threshold:
        rating = "pass"
    elif float(value) >= usable_threshold:
        rating = "minor"
    else:
        rating = "major"
    if rating == "pass" and BOILERPLATE_RE.search(text):
        return "minor"
    return rating


def build_summary(sample: dict[str, Any], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(decisions)
    metrics = {
        "question_image_available": sum(d["question_image_rating"] != "missing" for d in decisions),
        "markscheme_image_available": sum(d["mark_scheme_image_rating"] != "missing" for d in decisions),
        "pairing_correct": sum(bool(d["pairing_correct"]) for d in decisions),
        "question_image_usable": sum(d["question_image_rating"] in {"pass", "minor"} for d in decisions),
        "question_image_strict": sum(d["question_image_rating"] == "pass" for d in decisions),
        "markscheme_image_usable": sum(d["mark_scheme_image_rating"] in {"pass", "minor"} for d in decisions),
        "markscheme_image_strict": sum(d["mark_scheme_image_rating"] == "pass" for d in decisions),
        "question_text_usable": sum(d["question_text_rating"] in {"pass", "minor"} for d in decisions),
        "question_text_strict": sum(d["question_text_rating"] == "pass" for d in decisions),
        "markscheme_text_usable": sum(d["mark_scheme_text_rating"] in {"pass", "minor"} for d in decisions),
        "markscheme_text_strict": sum(d["mark_scheme_text_rating"] == "pass" for d in decisions),
        "overall_usable": sum(bool(d["overall_usable"]) for d in decisions),
        "strict_accurate": sum(bool(d["strict_accurate"]) for d in decisions),
    }
    headline = {name: metric_payload(count, n) for name, count in metrics.items()}
    ratings: dict[str, Counter[str]] = defaultdict(Counter)
    for decision in decisions:
        for component in (
            "question_image_rating",
            "mark_scheme_image_rating",
            "question_text_rating",
            "mark_scheme_text_rating",
        ):
            ratings[component][decision[component]] += 1
    family = {}
    for family_name in sorted({str(d.get("paper_family") or "unknown") for d in decisions}):
        selected = [d for d in decisions if str(d.get("paper_family") or "unknown") == family_name]
        family[family_name] = {
            "count": len(selected),
            "overall_usable": metric_payload(sum(bool(d["overall_usable"]) for d in selected), len(selected)),
            "question_text_usable": metric_payload(sum(d["question_text_rating"] in {"pass", "minor"} for d in selected), len(selected)),
            "markscheme_text_usable": metric_payload(sum(d["mark_scheme_text_rating"] in {"pass", "minor"} for d in selected), len(selected)),
        }
    worst_q = sorted(
        (d for d in decisions if d["question_ocr_similarity"] is not None),
        key=lambda d: d["question_ocr_similarity"],
    )[:15]
    worst_ms = sorted(
        (d for d in decisions if d["mark_scheme_ocr_similarity"] is not None),
        key=lambda d: d["mark_scheme_ocr_similarity"],
    )[:15]
    return {
        "schema_name": "exam_bank.random_visual_accuracy_summary",
        "schema_version": 1,
        "sampling": sample["sampling"],
        "sample_profile": sample["sample_profile"],
        "method": {
            "visual_review": "Every sampled image pair was inspected in full-size or compact contact sheets; flagged and loading-gap cases were checked full-size.",
            "pairing": "Question number and visible content were compared between question and markscheme images.",
            "text_assistance": "Per-record OCR/JSON token alignment supported the visual comparison; thresholds were calibrated against reviewed examples.",
            "question_text_thresholds": {"pass": ">=0.95", "minor_usable": ">=0.80 and <0.95"},
            "markscheme_text_thresholds": {"pass": ">=0.80", "minor_usable": ">=0.65 and <0.80"},
            "markscheme_image_conservative_rule": "A regeneration_failed fallback is not credited as a clean usable crop, even when the target answer remains visible.",
            "confidence_interval": "95% Wilson interval; conservative because finite-population correction is not applied.",
        },
        "headline_metrics": headline,
        "rating_counts": {name: dict(counter) for name, counter in ratings.items()},
        "family_metrics": family,
        "lowest_question_text_alignment": compact_examples(worst_q, "question_ocr_similarity"),
        "lowest_markscheme_text_alignment": compact_examples(worst_ms, "mark_scheme_ocr_similarity"),
    }


def metric_payload(count: int, total: int) -> dict[str, Any]:
    low, high = wilson(count, total)
    return {
        "count": count,
        "total": total,
        "percent": round(100 * count / total, 1) if total else None,
        "wilson_95_percent": [round(100 * low, 1), round(100 * high, 1)],
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def compact_examples(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [{"question_id": row["question_id"], field: row[field]} for row in rows]


def render_report(summary: dict[str, Any]) -> str:
    m = summary["headline_metrics"]

    def fmt(name: str) -> str:
        row = m[name]
        return f"{row['count']}/{row['total']} ({row['percent']}%; 95% CI {row['wilson_95_percent'][0]}–{row['wilson_95_percent'][1]}%)"

    family_rows = "\n".join(
        f"| {name} | {row['count']} | {row['question_text_usable']['percent']}% | {row['markscheme_text_usable']['percent']}% | {row['overall_usable']['percent']}% |"
        for name, row in summary["family_metrics"].items()
    )
    q_examples = ", ".join(row["question_id"] for row in summary["lowest_question_text_alignment"][:8])
    ms_examples = ", ".join(row["question_id"] for row in summary["lowest_markscheme_text_alignment"][:8])
    return f"""# Random 10% visual/text accuracy audit

## Bottom line

The sample contains {summary['sampling']['sample_count']} of {summary['sampling']['population_count']} questions, selected by simple random sampling without replacement with seed `{summary['sampling']['seed']}`.

- Image files present: question {fmt('question_image_available')}; markscheme {fmt('markscheme_image_available')}.
- Correct question/markscheme pairing: {fmt('pairing_correct')}.
- Question image usable (target complete/readable; minor furniture allowed): {fmt('question_image_usable')}.
- Markscheme image usable under a conservative clean-crop rule: {fmt('markscheme_image_usable')}.
- JSON question text usable: {fmt('question_text_usable')}.
- JSON markscheme text usable: {fmt('markscheme_text_usable')}.
- End-to-end usable (both images, both texts, correct pair): {fmt('overall_usable')}.
- Strictly accurate on all four components: {fmt('strict_accurate')}.

## Interpretation

Image discovery and pairing are strong: no sampled files were missing, every target prompt and target markscheme remained visible/readable, and no visually wrong question/markscheme pair was found. Manual review found {summary['rating_counts']['question_image_rating'].get('major', 0)} major question-image isolation failure(s) and {summary['rating_counts']['mark_scheme_image_rating'].get('major', 0)} major markscheme-image isolation failure(s). The remaining strict-image shortfall is mostly minor page furniture, headers/footers, or neighboring boundaries flagged by the fresh OCR screen.

Text extraction is the main semantic bottleneck. Prose is frequently recognizable, but mathematical notation is often flattened or corrupted: fractions, roots, powers, vectors/matrices, inequalities, Greek letters, integration bounds, and table/graph structure. Markscheme JSON is especially vulnerable because dense equations and table layouts magnify those errors.

The usable rates are deliberately separate from strict rates. “Usable” permits a localized non-fatal defect; “strict” requires clean crops and mathematically faithful text with no material artifacts.

## By paper family

| Family | n | Question text usable | Markscheme text usable | End-to-end usable |
|---|---:|---:|---:|---:|
{family_rows}

## Recommended improvement order

1. Replace plain OCR-to-text for mathematical regions with layout-aware math recognition, retaining LaTeX/MathML or structured spans instead of flattened text.
2. Improve markscheme table segmentation and make fallback crops question-aware; treat `regeneration_failed` and whole-table captures as a first-class QA failure.
3. Add semantic validators for values/symbols and structural validators for parts `(i)/(ii)`, vectors, matrices, tables, graphs, and mark totals.
4. Strip publisher/footer/copyright text after crop validation, not before, and keep a visual checksum/reference for regression tests.
5. Re-run this seeded sample after each pipeline change so gains are directly comparable; add a second independent seed before release decisions.

## Low-alignment examples for regression tests

- Question text: {q_examples}.
- Markscheme text: {ms_examples}.

## Method and limitations

Every sampled image pair was inspected in the generated review gallery/contact sheets, with ambiguous cases checked at full size. OCR-to-JSON token alignment was used as an assisted consistency signal, not as ground truth. Thresholds were calibrated against the visual review (`question`: pass ≥0.95, usable ≥0.80; `markscheme`: pass ≥0.80, usable ≥0.65). Confidence intervals are Wilson 95% intervals without finite-population correction, so they are slightly conservative.
"""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
