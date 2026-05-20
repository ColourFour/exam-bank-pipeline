from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.text_review_queue import build_review_queue, load_fixture_ids, load_question_bank, render_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION_BANK = REPO_ROOT / "output" / "json" / "question_bank.json"
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "text_fidelity" / "bad_text_records.json"
DEFAULT_JSON_OUT = REPO_ROOT / "output" / "reports" / "text_fidelity_review_queue.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / "output" / "reports" / "text_fidelity_review_queue.md"
DEFAULT_DOC_OUT = REPO_ROOT / "docs" / "text_extraction" / "TEXT_FIDELITY_REVIEW_QUEUE.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an advisory text-fidelity review queue from concrete failure signals.")
    parser.add_argument("--question-bank", default=str(DEFAULT_QUESTION_BANK), help="Question-bank JSON path.")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="Known bad text fixture manifest path.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="JSON queue report output path.")
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT), help="Markdown queue report output path.")
    parser.add_argument("--doc-out", default=str(DEFAULT_DOC_OUT), help="Documentation copy output path.")
    args = parser.parse_args()

    records = load_question_bank(Path(args.question_bank))
    fixture_ids = load_fixture_ids(Path(args.fixture))
    report = build_review_queue(records, fixture_ids)
    markdown = render_markdown(report)

    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    doc_out = Path(args.doc_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    doc_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_out.write_text(markdown, encoding="utf-8")
    doc_out.write_text(markdown, encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {markdown_out}")
    print(f"Wrote {doc_out}")
    print(f"Records inspected: {report['record_count']}")
    print(f"Queued records: {report['queued_count']}")
    print(f"Top reason codes: {report['top_reason_codes'][:6]}")
    print(f"Known bad fixtures in top 50: {report['fixture_summary']['known_fixtures_in_top_50']}")
    print(f"Known bad fixtures in top 100: {report['fixture_summary']['known_fixtures_in_top_100']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
