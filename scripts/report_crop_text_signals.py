from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.crop_text_signals import build_crop_text_signal_audit, load_question_bank_index, render_markdown
from exam_bank.text_fidelity import load_fixture_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "text_fidelity" / "bad_text_records.json"
DEFAULT_QUESTION_BANK_PATH = REPO_ROOT / "output" / "json" / "question_bank.json"
DEFAULT_JSON_OUT = REPO_ROOT / "output" / "reports" / "crop_text_signal_audit.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / "output" / "reports" / "crop_text_signal_audit.md"
DEFAULT_DOC_OUT = REPO_ROOT / "docs" / "text_extraction" / "CROP_CONTEXT_SIGNAL_AUDIT.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit crop-context advisory text signals without changing selection behavior.")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE_PATH), help="Bad-text fixture manifest path.")
    parser.add_argument("--question-bank", default=str(DEFAULT_QUESTION_BANK_PATH), help="Question-bank JSON context path.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="JSON report output path.")
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT), help="Markdown report output path.")
    parser.add_argument("--doc-out", default=str(DEFAULT_DOC_OUT), help="Documentation copy output path.")
    args = parser.parse_args()

    manifest = load_fixture_manifest(Path(args.fixture))
    question_bank_index = load_question_bank_index(Path(args.question_bank))
    report = build_crop_text_signal_audit(manifest, question_bank_index)

    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    doc_out = Path(args.doc_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    doc_out.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_out.write_text(markdown, encoding="utf-8")
    doc_out.write_text(markdown, encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {markdown_out}")
    print(f"Wrote {doc_out}")
    print(f"Records with useful warnings: {report['records_with_useful_warnings']}")
    print(f"Records caught by practical-now gates: {report['records_caught_by_practical_now_gates']}")
    print(f"Warning counts: {report['warning_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
