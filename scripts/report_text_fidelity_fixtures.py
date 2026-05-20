from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.text_fidelity import build_fixture_report, load_fixture_manifest, render_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "text_fidelity" / "bad_text_records.json"
DEFAULT_JSON_OUT = REPO_ROOT / "output" / "reports" / "text_fidelity_fixture_baseline.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / "output" / "reports" / "text_fidelity_fixture_baseline.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score frozen bad-text fixtures without changing extraction behavior.")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE_PATH), help="Bad-text fixture manifest path.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="JSON report output path.")
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT), help="Markdown report output path.")
    args = parser.parse_args()

    manifest = load_fixture_manifest(Path(args.fixture))
    report = build_fixture_report(manifest)
    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {markdown_out}")
    print(f"Status counts: {report['status_counts']}")
    print(f"Failure type counts: {report['failure_type_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

