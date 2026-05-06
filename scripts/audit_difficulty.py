from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.audit import write_difficulty_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit deterministic difficulty metadata in an exported question bank JSON.",
    )
    parser.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    parser.add_argument("--json-output", default="", help="Optional path to write the difficulty audit JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.json_output) if args.json_output else None
    report = write_difficulty_audit(args.input, output_path=output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
