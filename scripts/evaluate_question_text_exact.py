from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from exam_bank.question_text_exact import (
    DEFAULT_MINIMUM_ACCURACY,
    QuestionTextExactError,
    evaluate_question_text_exact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate normalized whole-question exact text accuracy.")
    parser.add_argument("--gold", required=True, help="Gold registry JSON containing a records list.")
    parser.add_argument("--candidate", required=True, help="Candidate question_bank JSON containing a questions list.")
    parser.add_argument("--cohort", default="", help="Optional fixed cohort sample JSON containing a questions list.")
    parser.add_argument("--minimum-accuracy", type=float, default=DEFAULT_MINIMUM_ACCURACY)
    parser.add_argument("--output", default="", help="Optional path for the JSON evaluation report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        gold = load_json(Path(args.gold))
        candidate = load_json(Path(args.candidate))
        cohort = load_json(Path(args.cohort)) if args.cohort else None
        report = evaluate_question_text_exact(
            gold,
            candidate,
            cohort_sample=cohort,
            minimum_accuracy=args.minimum_accuracy,
        )
    except (OSError, json.JSONDecodeError, QuestionTextExactError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QuestionTextExactError(f"{path} must contain a JSON object.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
