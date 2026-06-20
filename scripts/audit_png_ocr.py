from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image


HEADER_RE = re.compile(
    r"\b(?:Page\s+\d+|Mark Scheme|Question Paper|Syllabus|Paper|GCE A/?AS LEVEL|May/June|October/November|Cambridge|UCLES|9709|BLANK PAGE)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z]{2,}")


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR-audit generated question and mark-scheme PNG artifacts.")
    parser.add_argument("--question-bank", default="output/json/question_bank.json")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--question-regeneration-report", default="reports/regenerate_question_pngs_full_20260620.json")
    parser.add_argument("--mark-scheme-regeneration-report", default="reports/regenerate_mark_scheme_pngs_full_20260620.json")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument(
        "--prune-unreferenced",
        action="store_true",
        help="Delete canonical-looking question/mark-scheme PNGs under output-root that are not referenced by question_bank.json.",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run --prune-unreferenced cleanup and skip OCR auditing.",
    )
    parser.add_argument(
        "--prune-manifest",
        default="",
        help="Optional path for the unreferenced PNG deletion manifest. Defaults under report-root.",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ocr-timeout", type=int, default=45)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    question_bank = json.loads(Path(args.question_bank).read_text(encoding="utf-8"))
    rows = [row for row in question_bank.get("questions", []) if isinstance(row, dict)]
    expected, paper_order = _expected_artifacts(rows)
    question_regen = _load_report(Path(args.question_regeneration_report))
    mark_scheme_regen = _load_report(Path(args.mark_scheme_regeneration_report))
    regen_by_path = _regen_by_path(question_regen, "canonical_question_artifact") | _regen_by_path(
        mark_scheme_regen,
        "canonical_mark_scheme_artifact",
    )

    actual_paths = sorted(
        [
            path
            for path in output_root.rglob("*.png")
            if path.name.endswith("_question.png") or path.name.endswith("_markscheme.png")
        ],
        key=lambda path: str(path.relative_to(output_root)),
    )
    cleanup_report: dict[str, Any] | None = None
    if args.prune_unreferenced:
        cleanup_report = _prune_unreferenced_pngs(
            actual_paths,
            output_root,
            expected,
            question_bank_path=Path(args.question_bank),
            manifest_path=Path(args.prune_manifest) if args.prune_manifest else report_root / "pruned_unreferenced_pngs_20260620.json",
        )
        actual_paths = [path for path in actual_paths if path.exists()]
    if args.cleanup_only:
        if cleanup_report is None:
            raise SystemExit("--cleanup-only requires --prune-unreferenced")
        print(json.dumps(cleanup_report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if args.limit:
        actual_paths = actual_paths[: args.limit]

    per_png_path = report_root / "png_ocr_audit_20260620.jsonl"
    irregularities_csv = report_root / "png_ocr_irregularities_20260620.csv"
    summary_path = report_root / "png_ocr_audit_summary_20260620.json"

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _audit_png,
                path,
                output_root,
                expected,
                paper_order,
                regen_by_path,
                args.ocr_timeout,
            ): path
            for path in actual_paths
        }
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda item: item["path"])
    irregular_records = [record for record in records if record["irregularities"]]

    expected_question = {path for path, meta in expected.items() if meta["artifact_type"] == "question"}
    expected_mark_scheme = {path for path, meta in expected.items() if meta["artifact_type"] == "mark_scheme"}
    actual_relative = {record["path"] for record in records}
    actual_question = {record["path"] for record in records if record["artifact_type"] == "question"}
    actual_mark_scheme = {record["path"] for record in records if record["artifact_type"] == "mark_scheme"}

    summary = {
        "question_bank_json": {
            "path": str(Path(args.question_bank)),
            "record_count_field": question_bank.get("record_count"),
            "question_records": len(rows),
            "expected_question_pngs": len(expected_question),
            "expected_mark_scheme_pngs": len(expected_mark_scheme),
            "records_with_question_image": sum(1 for row in rows if row.get("canonical_question_artifact") or row.get("question_image_path")),
            "records_with_mark_scheme_image": sum(
                1 for row in rows if row.get("canonical_mark_scheme_artifact") or row.get("mark_scheme_image_path")
            ),
        },
        "actual_pngs": {
            "audited_total": len(records),
            "question_pngs": len(actual_question),
            "mark_scheme_pngs": len(actual_mark_scheme),
            "missing_expected_question_pngs": sorted(expected_question - actual_relative),
            "missing_expected_mark_scheme_pngs": sorted(expected_mark_scheme - actual_relative),
            "extra_question_pngs": sorted(actual_question - expected_question),
            "extra_mark_scheme_pngs": sorted(actual_mark_scheme - expected_mark_scheme),
        },
        "regeneration": {
            "question": _report_totals(question_regen),
            "mark_scheme": _report_totals(mark_scheme_regen),
        },
        "ocr": {
            "ocr_engine": "tesseract",
            "ocr_attempted": len(records),
            "ocr_failed": sum("ocr_failed" in record["irregularities"] for record in records),
            "ocr_empty_or_near_empty": sum("ocr_empty_or_near_empty" in record["irregularities"] for record in records),
        },
        "irregularities": {
            "pngs_with_irregularities": len(irregular_records),
            "expected_pngs_with_irregularities": sum(record["expected_by_json"] for record in irregular_records),
            "extra_pngs_with_irregularities": sum(not record["expected_by_json"] for record in irregular_records),
            "question_pngs_with_irregularities": sum(record["artifact_type"] == "question" for record in irregular_records),
            "mark_scheme_pngs_with_irregularities": sum(record["artifact_type"] == "mark_scheme" for record in irregular_records),
            "by_flag": dict(sorted(Counter(flag for record in irregular_records for flag in record["irregularities"]).items())),
        },
        "outputs": {
            "per_png_jsonl": str(per_png_path),
            "irregularities_csv": str(irregularities_csv),
            "summary_json": str(summary_path),
        },
    }
    if cleanup_report is not None:
        summary["cleanup"] = cleanup_report

    with per_png_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with irregularities_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "artifact_type",
                "path",
                "question_png",
                "mark_scheme_png",
                "irregularities",
                "regen_status",
                "regen_failure_reason",
                "ocr_word_count",
                "image_width",
                "image_height",
                "ocr_preview",
            ],
        )
        writer.writeheader()
        for record in irregular_records:
            writer.writerow(
                {
                    "question_id": record.get("question_id", ""),
                    "artifact_type": record["artifact_type"],
                    "path": record["path"],
                    "question_png": record.get("question_png", ""),
                    "mark_scheme_png": record.get("mark_scheme_png", ""),
                    "irregularities": ";".join(record["irregularities"]),
                    "regen_status": record.get("regen_status", ""),
                    "regen_failure_reason": record.get("regen_failure_reason", ""),
                    "ocr_word_count": record["ocr_word_count"],
                    "image_width": record["image_width"],
                    "image_height": record["image_height"],
                    "ocr_preview": record["ocr_preview"],
                }
            )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _expected_artifacts(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    expected: dict[str, dict[str, Any]] = {}
    paper_order: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        question_id = str(row.get("question_id") or "")
        question_number = str(row.get("question_number") or "")
        paper = str(row.get("paper") or "")
        question_png = str(row.get("canonical_question_artifact") or row.get("question_image_path") or "")
        mark_scheme_png = str(row.get("canonical_mark_scheme_artifact") or row.get("mark_scheme_image_path") or "")
        if paper and question_number:
            paper_order[paper].append(question_number)
        common = {
            "question_id": question_id,
            "question_number": question_number,
            "paper": paper,
            "question_png": question_png,
            "mark_scheme_png": mark_scheme_png,
        }
        if question_png:
            expected[question_png] = {**common, "artifact_type": "question"}
        if mark_scheme_png:
            expected[mark_scheme_png] = {**common, "artifact_type": "mark_scheme"}
    for paper, values in paper_order.items():
        paper_order[paper] = sorted(set(values), key=_question_sort_key)
    return expected, paper_order


def _audit_png(
    path: Path,
    output_root: Path,
    expected: dict[str, dict[str, Any]],
    paper_order: dict[str, list[str]],
    regen_by_path: dict[str, dict[str, Any]],
    timeout: int,
) -> dict[str, Any]:
    relative = str(path.relative_to(output_root))
    artifact_type = "mark_scheme" if path.name.endswith("_markscheme.png") else "question"
    meta = expected.get(relative, {})
    regen = regen_by_path.get(relative, {})
    irregularities: list[str] = []

    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:  # noqa: BLE001 - report invalid image artifact.
        width = 0
        height = 0
        irregularities.append("invalid_png")
        ocr_text = ""
        ocr_error = str(exc)
    else:
        ocr_text, ocr_error = _run_ocr(path, timeout)

    normalized_text = " ".join(ocr_text.split())
    word_count = len(WORD_RE.findall(normalized_text))

    if not meta:
        irregularities.append("not_referenced_by_question_bank_json")
    if regen and (regen.get("status") != "pass" or not regen.get("image_path")):
        irregularities.append("regeneration_failed")
    if "saved_crop_diagnostics_fallback" in regen.get("review_flags", []):
        irregularities.append("saved_crop_diagnostics_fallback")
    if ocr_error:
        irregularities.append("ocr_failed")
    if word_count < 2:
        irregularities.append("ocr_empty_or_near_empty")
    if width < 300 or height < 35:
        irregularities.append("suspiciously_small_png")
    if height > 3200:
        irregularities.append("suspiciously_tall_png")
    if HEADER_RE.search(normalized_text):
        irregularities.append("possible_page_header_footer_text")

    question_number = str(meta.get("question_number") or "")
    if meta and question_number and not _contains_number_line_or_token(normalized_text, question_number):
        irregularities.append("target_question_number_not_seen_by_ocr")

    neighbor_flags = _neighbor_question_flags(normalized_text, meta, paper_order)
    irregularities.extend(neighbor_flags)

    return {
        "path": relative,
        "artifact_type": artifact_type,
        "expected_by_json": bool(meta),
        "question_id": meta.get("question_id", ""),
        "paper": meta.get("paper", ""),
        "question_number": question_number,
        "question_png": meta.get("question_png", ""),
        "mark_scheme_png": meta.get("mark_scheme_png", ""),
        "image_width": width,
        "image_height": height,
        "ocr_word_count": word_count,
        "ocr_error": ocr_error,
        "ocr_preview": normalized_text[:500],
        "regen_status": regen.get("status", ""),
        "regen_failure_reason": regen.get("failure_reason", ""),
        "irregularities": sorted(set(irregularities)),
    }


def _run_ocr(path: Path, timeout: int) -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["tesseract", str(path), "stdout", "--psm", "6", "-l", "eng"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - report OCR failure per artifact.
        return "", str(exc)
    if result.returncode != 0:
        return result.stdout or "", result.stderr.strip() or f"tesseract exited {result.returncode}"
    return result.stdout or "", ""


def _prune_unreferenced_pngs(
    actual_paths: list[Path],
    output_root: Path,
    expected: dict[str, dict[str, Any]],
    *,
    question_bank_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    expected_paths = set(expected)
    deleted: list[dict[str, Any]] = []
    kept_count = 0
    for path in actual_paths:
        relative = str(path.relative_to(output_root))
        if relative in expected_paths:
            kept_count += 1
            continue
        stat = path.stat()
        deleted.append(
            {
                "path": relative,
                "artifact_type": "mark_scheme" if path.name.endswith("_markscheme.png") else "question",
                "size_bytes": stat.st_size,
            }
        )
        path.unlink()

    report = {
        "schema_name": "exam_bank.png_unreferenced_cleanup",
        "schema_version": 1,
        "question_bank": str(question_bank_path),
        "output_root": str(output_root),
        "candidate_png_count": len(actual_paths),
        "referenced_png_count": kept_count,
        "deleted_unreferenced_png_count": len(deleted),
        "deleted_unreferenced_pngs": deleted,
        "manifest_path": str(manifest_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def _neighbor_question_flags(text: str, meta: dict[str, Any], paper_order: dict[str, list[str]]) -> list[str]:
    if not meta:
        return []
    question_number = str(meta.get("question_number") or "")
    paper = str(meta.get("paper") or "")
    if not question_number or not paper:
        return []
    order = paper_order.get(paper, [])
    if question_number not in order:
        return []
    index = order.index(question_number)
    flags: list[str] = []
    if index > 0 and _contains_primary_number_line(text, order[index - 1]):
        flags.append("possible_previous_question_number_seen_by_ocr")
    if index + 1 < len(order) and _contains_primary_number_line(text, order[index + 1]):
        flags.append("possible_next_question_number_seen_by_ocr")
    return flags


def _contains_number_line_or_token(text: str, number: str) -> bool:
    return _contains_primary_number_line(text, number) or bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(number)}(?![A-Za-z0-9])", text))


def _contains_primary_number_line(text: str, number: str) -> bool:
    pattern = re.compile(rf"(?:^|\n|\r)\s*(?:Q\.?\s*)?{re.escape(number)}(?:\s|[).:]|$)")
    return bool(pattern.search(text))


def _regen_by_path(report: dict[str, Any], path_field: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in report.get("outputs", []):
        path = str(item.get(path_field) or "")
        if path:
            output[path] = item
    return output


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _report_totals(report: dict[str, Any]) -> dict[str, Any]:
    outputs = report.get("outputs", [])
    return {
        "selected_count": report.get("selected_count", 0),
        "rendered_count": report.get("rendered_count", 0),
        "failed_count": report.get("failed_count", 0),
        "skipped_count": report.get("skipped_count", 0),
        "missing_requested_ids": len(report.get("missing_requested_ids", [])),
        "status_counts": dict(sorted(Counter(str(item.get("status") or "") for item in outputs).items())),
        "failure_reason_counts": dict(
            sorted(Counter(str(item.get("failure_reason") or "") for item in outputs if item.get("failure_reason")).items())
        ),
    }


def _question_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (999, value)


if __name__ == "__main__":
    raise SystemExit(main())
