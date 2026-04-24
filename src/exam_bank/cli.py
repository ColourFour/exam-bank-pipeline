from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import write_audit
from .config import AppConfig, load_config
from .pipeline import PipelineResult, process_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CAIE question papers and mark schemes into paper-first folders plus JSON metadata.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser(
        "process",
        help="Process a folder of question paper PDFs and mark scheme PDFs.",
    )
    process.add_argument(
        "--input",
        default="input",
        help="Input folder. The scan is recursive and supports either mixed PDFs or question_papers/mark_schemes subfolders.",
    )
    process.add_argument(
        "--output",
        default="output",
        help="Output root folder for paper-first exports and JSON metadata.",
    )
    process.add_argument(
        "--config",
        default="config.yaml",
        help="Optional config.yaml path for operational overrides.",
    )
    process.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Run optional Tesseract OCR on rendered question crop PNGs and sparse PDF regions.",
    )
    process.add_argument("--ocr-language", default="", help="Optional Tesseract language override, e.g. eng.")
    process.add_argument("--tesseract-cmd", default="", help="Optional path to the tesseract binary.")
    process.set_defaults(func=cmd_process)

    audit = subparsers.add_parser(
        "audit",
        help="Audit visual-first question text trust and curation readiness in an exported question bank JSON.",
    )
    audit.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    audit.add_argument("--output", default="", help="Optional path to write the audit JSON report.")
    audit.set_defaults(func=cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def cmd_process(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.enable_ocr:
        config.ocr.enabled = True
    if args.ocr_language:
        config.ocr.language = args.ocr_language
    if args.tesseract_cmd:
        config.ocr.tesseract_cmd = args.tesseract_cmd
    _configure_runtime_paths(config, Path(args.input), Path(args.output))
    result = process_inputs(args.input, config)
    _print_result(result)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    report = write_audit(args.input, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _configure_runtime_paths(config: AppConfig, input_root: Path, output_root: Path) -> None:
    config.output.apply_root(output_root)

    if input_root.is_dir():
        question_dir = input_root / "question_papers"
        mark_scheme_dir = input_root / "mark_schemes"
        mappings_dir = input_root / "mappings"
        if question_dir.exists():
            config.input.question_papers_dir = question_dir
        else:
            config.input.question_papers_dir = input_root
        if mark_scheme_dir.exists():
            config.input.mark_schemes_dir = mark_scheme_dir
        else:
            config.input.mark_schemes_dir = input_root
        if mappings_dir.exists():
            config.input.mappings_dir = mappings_dir


def _print_result(result: PipelineResult) -> None:
    papers = sorted({record.paper_name for record in result.records})
    print(f"Processed questions: {len(result.records)}")
    print(f"Processed papers: {len(papers)}")
    print(f"Output root: {result.output_root}")
    print(f"JSON: {result.json_path}")


if __name__ == "__main__":
    raise SystemExit(main())
