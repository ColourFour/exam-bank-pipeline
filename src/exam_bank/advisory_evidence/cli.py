from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from exam_bank.advisory_evidence.extraction import extract_all_advisory_text
from exam_bank.advisory_evidence.inventory import write_all_inventories
from exam_bank.advisory_evidence.linking import build_all_links
from exam_bank.advisory_evidence.parsing import parse_all_examiner_reports, parse_all_grade_thresholds
from exam_bank.advisory_evidence.reports import build_review_reports
from exam_bank.advisory_evidence.sidecar import build_final_sidecar
from exam_bank.advisory_evidence.signals import build_examiner_difficulty, build_grade_threshold_context, build_topic_evidence
from exam_bank.advisory_evidence.validation import validate_advisory_evidence


def run_inventory(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic advisory PDF inventories.")
    parser.add_argument("--examiner-reports-dir", type=Path, default=Path("input/examiner_reports"))
    parser.add_argument("--grade-thresholds-dir", type=Path, default=Path("input/grade_thresholds"))
    parser.add_argument("--output-root", type=Path, default=Path("output/advisory_evidence"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = write_all_inventories(
        output_root=args.output_root,
        examiner_reports_dir=args.examiner_reports_dir,
        grade_thresholds_dir=args.grade_thresholds_dir,
        dry_run=args.dry_run,
    )
    _print(
        {
            "dry_run": summary["dry_run"],
            "outputs": summary["outputs"],
            "document_counts": {key: value["document_count"] for key, value in summary["inventories"].items()},
            "warning_counts": {key: len(value.get("warnings", [])) for key, value in summary["inventories"].items()},
        }
    )
    return 0


def run_extract(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract native text from inventoried advisory PDFs.")
    parser.add_argument("--inventory-dir", type=Path, default=Path("output/advisory_evidence/inventory"))
    parser.add_argument("--output-root", type=Path, default=Path("output/advisory_evidence"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _print(extract_all_advisory_text(inventory_dir=args.inventory_dir, output_root=args.output_root, dry_run=args.dry_run))
    return 0


def run_parse_examiner(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse extracted examiner reports into structured sections.")
    parser.add_argument("--extracted-dir", type=Path, default=Path("output/advisory_evidence/extracted_text/examiner_reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/advisory_evidence/parsed/examiner_reports"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _print(parse_all_examiner_reports(extracted_dir=args.extracted_dir, output_dir=args.output_dir, dry_run=args.dry_run))
    return 0


def run_parse_thresholds(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse extracted grade-threshold PDFs into structured tables.")
    parser.add_argument("--extracted-dir", type=Path, default=Path("output/advisory_evidence/extracted_text/grade_thresholds"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/advisory_evidence/parsed/grade_thresholds"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _print(parse_all_grade_thresholds(extracted_dir=args.extracted_dir, output_dir=args.output_dir, dry_run=args.dry_run))
    return 0


def run_link(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Link parsed advisory evidence to question-bank records.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--parsed-root", type=Path, default=Path("output/advisory_evidence/parsed"))
    parser.add_argument("--output-root", type=Path, default=Path("output/advisory_evidence"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = build_all_links(question_bank_path=args.question_bank, parsed_root=args.parsed_root, output_root=args.output_root, dry_run=args.dry_run)
    _print(
        {
            "dry_run": payload["dry_run"],
            "outputs": payload["outputs"],
            "examiner_report_summary": payload["examiner_report"].get("summary", {}),
            "grade_threshold_summary": payload["grade_thresholds"].get("summary", {}),
        }
    )
    return 0


def run_topic_evidence(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic advisory topic evidence from examiner-report comments.")
    parser.add_argument("--parsed-dir", type=Path, default=Path("output/advisory_evidence/parsed/examiner_reports"))
    parser.add_argument("--links", type=Path, default=Path("output/advisory_evidence/linking/examiner_report_question_links.json"))
    parser.add_argument("--output", type=Path, default=Path("output/advisory_evidence/predictions/advisory_topic_evidence.v1.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = build_topic_evidence(parsed_dir=args.parsed_dir, links_path=args.links, output_path=args.output, dry_run=args.dry_run)
    _print({"dry_run": args.dry_run, "output": str(args.output), "record_count": len(payload.get("records", [])), "warnings": payload.get("warnings", [])})
    return 0


def run_examiner_difficulty(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build examiner-report item difficulty hints.")
    parser.add_argument("--parsed-dir", type=Path, default=Path("output/advisory_evidence/parsed/examiner_reports"))
    parser.add_argument("--links", type=Path, default=Path("output/advisory_evidence/linking/examiner_report_question_links.json"))
    parser.add_argument("--output", type=Path, default=Path("output/advisory_evidence/predictions/advisory_examiner_report_difficulty.v1.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = build_examiner_difficulty(parsed_dir=args.parsed_dir, links_path=args.links, output_path=args.output, dry_run=args.dry_run)
    _print({"dry_run": args.dry_run, "output": str(args.output), "record_count": len(payload.get("records", [])), "warnings": payload.get("warnings", [])})
    return 0


def run_threshold_context(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build grade-threshold component/session context.")
    parser.add_argument("--parsed-dir", type=Path, default=Path("output/advisory_evidence/parsed/grade_thresholds"))
    parser.add_argument("--output", type=Path, default=Path("output/advisory_evidence/predictions/advisory_grade_threshold_context.v1.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = build_grade_threshold_context(parsed_dir=args.parsed_dir, output_path=args.output, dry_run=args.dry_run)
    _print({"dry_run": args.dry_run, "output": str(args.output), "context_count": len(payload.get("contexts", [])), "warnings": payload.get("warnings", [])})
    return 0


def run_validate(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate advisory evidence stage outputs.")
    parser.add_argument("--advisory-root", type=Path, default=Path("output/advisory_evidence"))
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = validate_advisory_evidence(advisory_root=args.advisory_root, question_bank_path=args.question_bank, output_path=args.output)
    _print(report)
    return 0 if report["ok"] else 1


def run_reports(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Markdown advisory review reports.")
    parser.add_argument("--advisory-root", type=Path, default=Path("output/advisory_evidence"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/advisory_evidence/reports"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _print(build_review_reports(advisory_root=args.advisory_root, output_dir=args.output_dir, dry_run=args.dry_run))
    return 0


def run_sidecar(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the final advisory evidence sidecar.")
    parser.add_argument("--advisory-root", type=Path, default=Path("output/advisory_evidence"))
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--output", type=Path, default=Path("output/advisory_evidence/question_bank.advisory_evidence.v1.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = build_final_sidecar(advisory_root=args.advisory_root, question_bank_path=args.question_bank, output_path=args.output, dry_run=args.dry_run)
    _print({"dry_run": args.dry_run, "output": str(args.output), "records_count": payload.get("records_count", 0), "warnings": payload.get("warnings", [])})
    return 0


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
