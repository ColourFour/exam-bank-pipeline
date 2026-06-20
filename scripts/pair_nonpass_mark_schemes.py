from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Pair non-pass mark-scheme PNG records with their question records.")
    parser.add_argument("--question-bank", default="output/json/question_bank.json")
    parser.add_argument("--mark-scheme-report", default="reports/regenerate_mark_scheme_pngs_full_20260620.json")
    parser.add_argument("--ocr-audit-jsonl", default="reports/png_ocr_audit_20260620.jsonl")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--stamp", default="20260620")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    question_bank = json.loads(Path(args.question_bank).read_text(encoding="utf-8"))
    rows = [row for row in question_bank.get("questions", []) if isinstance(row, dict)]
    rows_by_id = {str(row.get("question_id") or ""): row for row in rows}
    report = json.loads(Path(args.mark_scheme_report).read_text(encoding="utf-8"))
    ocr_by_path = _load_ocr_audit(Path(args.ocr_audit_jsonl))

    outputs = [item for item in report.get("outputs", []) if isinstance(item, dict)]
    no_fresh_image = [item for item in outputs if not item.get("image_path")]
    nonpass = [item for item in outputs if item.get("status") != "pass"]

    no_fresh_pairs = [_pair_record(item, rows_by_id, output_root, ocr_by_path) for item in no_fresh_image]
    nonpass_pairs = [_pair_record(item, rows_by_id, output_root, ocr_by_path) for item in nonpass]

    failed_csv = report_root / f"mark_scheme_failed_question_pairs_{args.stamp}.csv"
    nonpass_csv = report_root / f"mark_scheme_nonpass_question_pairs_{args.stamp}.csv"
    summary_json = report_root / f"mark_scheme_nonpass_question_pairs_{args.stamp}.json"
    html_report = report_root / f"mark_scheme_failed_question_pairs_{args.stamp}.html"

    _write_csv(failed_csv, no_fresh_pairs)
    _write_csv(nonpass_csv, nonpass_pairs)
    _write_html(html_report, no_fresh_pairs)

    summary = {
        "question_bank": {
            "path": str(Path(args.question_bank)),
            "record_count_field": question_bank.get("record_count"),
            "question_records": len(rows),
        },
        "mark_scheme_regeneration_report": {
            "path": str(Path(args.mark_scheme_report)),
            "selected_count": report.get("selected_count"),
            "rendered_count": report.get("rendered_count"),
            "failed_count_no_fresh_image": report.get("failed_count"),
            "skipped_count": report.get("skipped_count"),
            "missing_requested_ids": len(report.get("missing_requested_ids", [])),
            "nonpass_status_count": len(nonpass),
            "nonpass_with_fresh_image_count": sum(bool(item.get("image_path")) for item in nonpass),
        },
        "paired_outputs": {
            "no_fresh_image_pair_count": len(no_fresh_pairs),
            "nonpass_pair_count": len(nonpass_pairs),
            "unmatched_no_fresh_image_count": sum(not item["json_record_found"] for item in no_fresh_pairs),
            "unmatched_nonpass_count": sum(not item["json_record_found"] for item in nonpass_pairs),
            "no_fresh_image_failure_reason_counts": dict(sorted(Counter(item["failure_reason"] for item in no_fresh_pairs).items())),
            "nonpass_failure_reason_counts": dict(sorted(Counter(item["failure_reason"] for item in nonpass_pairs).items())),
            "no_fresh_image_by_paper_family": dict(sorted(Counter(item["paper_family"] for item in no_fresh_pairs).items())),
            "nonpass_by_paper_family": dict(sorted(Counter(item["paper_family"] for item in nonpass_pairs).items())),
            "existing_stale_mark_scheme_png_count": sum(item["existing_mark_scheme_png_exists"] for item in no_fresh_pairs),
            "missing_existing_mark_scheme_png_count": sum(not item["existing_mark_scheme_png_exists"] for item in no_fresh_pairs),
            "question_png_exists_count": sum(item["question_png_exists"] for item in no_fresh_pairs),
        },
        "artifacts": {
            "no_fresh_image_pairs_csv": str(failed_csv),
            "all_nonpass_pairs_csv": str(nonpass_csv),
            "summary_json": str(summary_json),
            "html_review": str(html_report),
        },
        "pairs_no_fresh_image": no_fresh_pairs,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "pairs_no_fresh_image"}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _pair_record(
    output: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    output_root: Path,
    ocr_by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    question_id = str(output.get("question_id") or "")
    row = rows_by_id.get(question_id, {})
    notes = row.get("notes") if isinstance(row.get("notes"), dict) else {}
    question_png = str(row.get("canonical_question_artifact") or row.get("question_image_path") or "")
    mark_scheme_png = str(row.get("canonical_mark_scheme_artifact") or row.get("mark_scheme_image_path") or "")
    question_ocr = ocr_by_path.get(question_png, {})
    mark_scheme_ocr = ocr_by_path.get(mark_scheme_png, {})
    return {
        "question_id": question_id,
        "json_record_found": bool(row),
        "paper": str(row.get("paper") or ""),
        "paper_family": str(row.get("paper_family") or ""),
        "canonical_year_folder": str(row.get("canonical_year_folder") or ""),
        "canonical_session": str(row.get("canonical_session") or ""),
        "component": str(notes.get("source_paper_code") or ""),
        "question_number": str(output.get("question_number") or row.get("question_number") or ""),
        "failure_reason": str(output.get("failure_reason") or ""),
        "regeneration_status": str(output.get("status") or ""),
        "fresh_mark_scheme_image_path": str(output.get("image_path") or ""),
        "expected_question_png": question_png,
        "expected_mark_scheme_png": mark_scheme_png,
        "question_png_exists": bool(question_png and (output_root / question_png).exists()),
        "existing_mark_scheme_png_exists": bool(mark_scheme_png and (output_root / mark_scheme_png).exists()),
        "source_question_pdf": str(notes.get("source_pdf") or row.get("source_pdf") or ""),
        "source_mark_scheme_pdf": str(output.get("source_pdf") or notes.get("mark_scheme_source_pdf") or ""),
        "question_text_preview": _preview(str(row.get("question_text") or row.get("body_text_raw") or "")),
        "question_ocr_word_count": question_ocr.get("ocr_word_count", ""),
        "question_ocr_irregularities": ";".join(question_ocr.get("irregularities", [])) if isinstance(question_ocr.get("irregularities"), list) else "",
        "question_ocr_preview": _preview(str(question_ocr.get("ocr_preview") or "")),
        "mark_scheme_ocr_word_count": mark_scheme_ocr.get("ocr_word_count", ""),
        "mark_scheme_ocr_irregularities": ";".join(mark_scheme_ocr.get("irregularities", []))
        if isinstance(mark_scheme_ocr.get("irregularities"), list)
        else "",
        "mark_scheme_ocr_preview": _preview(str(mark_scheme_ocr.get("ocr_preview") or "")),
    }


def _load_ocr_audit(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            records[str(item.get("path") or "")] = item
    return records


def _write_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "json_record_found",
        "paper",
        "paper_family",
        "canonical_year_folder",
        "canonical_session",
        "component",
        "question_number",
        "failure_reason",
        "regeneration_status",
        "fresh_mark_scheme_image_path",
        "expected_question_png",
        "expected_mark_scheme_png",
        "question_png_exists",
        "existing_mark_scheme_png_exists",
        "source_question_pdf",
        "source_mark_scheme_pdf",
        "question_text_preview",
        "question_ocr_word_count",
        "question_ocr_irregularities",
        "question_ocr_preview",
        "mark_scheme_ocr_word_count",
        "mark_scheme_ocr_irregularities",
        "mark_scheme_ocr_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)


def _write_html(path: Path, pairs: list[dict[str, Any]]) -> None:
    rows = []
    for pair in pairs:
        question_src = "../output/" + pair["expected_question_png"] if pair["expected_question_png"] else ""
        mark_scheme_src = "../output/" + pair["expected_mark_scheme_png"] if pair["expected_mark_scheme_png"] else ""
        question_img = f'<img loading="lazy" src="{html.escape(question_src)}" alt="question">' if pair["question_png_exists"] else ""
        mark_scheme_img = (
            f'<img loading="lazy" src="{html.escape(mark_scheme_src)}" alt="mark scheme">'
            if pair["existing_mark_scheme_png_exists"]
            else "<em>No existing PNG on disk</em>"
        )
        rows.append(
            "<section>"
            f"<h2>{html.escape(pair['question_id'])} | q{html.escape(pair['question_number'])} | {html.escape(pair['failure_reason'])}</h2>"
            f"<p><code>{html.escape(pair['expected_question_png'])}</code><br><code>{html.escape(pair['expected_mark_scheme_png'])}</code></p>"
            '<div class="pair">'
            f'<figure><figcaption>Question</figcaption>{question_img}</figure>'
            f'<figure><figcaption>Existing mark scheme PNG</figcaption>{mark_scheme_img}</figure>'
            "</div>"
            f"<p><strong>Question OCR:</strong> {html.escape(pair['question_ocr_preview'])}</p>"
            f"<p><strong>Mark scheme OCR:</strong> {html.escape(pair['mark_scheme_ocr_preview'])}</p>"
            "</section>"
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<meta charset='utf-8'>",
                "<title>Non-pass mark scheme question pairs</title>",
                "<style>",
                "body{font-family:system-ui,sans-serif;margin:24px;color:#111}",
                "section{border-top:1px solid #ccc;padding:18px 0}",
                ".pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}",
                "figure{margin:0}",
                "figcaption{font-weight:700;margin-bottom:6px}",
                "img{max-width:100%;border:1px solid #ddd;background:white}",
                "code{font-size:12px}",
                "</style>",
                f"<h1>Non-pass mark schemes paired with questions ({len(pairs)})</h1>",
                *rows,
            ]
        ),
        encoding="utf-8",
    )


def _preview(value: str, limit: int = 500) -> str:
    cleaned = " ".join(value.split())
    return cleaned[:limit]


if __name__ == "__main__":
    raise SystemExit(main())
