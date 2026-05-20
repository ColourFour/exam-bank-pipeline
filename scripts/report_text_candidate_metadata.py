from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.text_candidate_metadata import build_metadata_coverage_report, load_question_bank, render_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION_BANK = REPO_ROOT / "output" / "json" / "question_bank.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "output"
DEFAULT_JSON_OUT = REPO_ROOT / "output" / "reports" / "text_candidate_metadata_coverage.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / "output" / "reports" / "text_candidate_metadata_coverage.md"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report advisory text candidate metadata coverage without mutating canonical records."
    )
    parser.add_argument("--question-bank", default=str(DEFAULT_QUESTION_BANK), help="Question-bank JSON path.")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT), help="Root for rendered question images.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="JSON report output path.")
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT), help="Markdown report output path.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-record inventory details from the JSON report.",
    )
    args = parser.parse_args()

    records = load_question_bank(Path(args.question_bank))
    report = build_metadata_coverage_report(
        records,
        artifact_root=Path(args.artifact_root),
        include_records=not args.summary_only,
    )
    markdown = render_markdown(report)

    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_out.write_text(markdown, encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {markdown_out}")
    print(f"Records inspected: {report['record_count']}")
    for field, counts in report["field_counts"].items():
        print(
            f"{field}: present={counts['present']} missing={counts['missing']} "
            f"populated={report['populated_counts'][field]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
