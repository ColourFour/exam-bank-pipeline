from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.auto_grade.constants import (
    DEFAULT_REVIEWED_RUBRICS_PATH,
    DEFAULT_REVIEWED_RUBRICS_VALIDATION_REPORT_PATH,
)
from exam_bank.auto_grade.reviewed_rubrics import validate_reviewed_rubrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 2A reviewed_rubrics.v1.json.")
    parser.add_argument("--reviewed-rubrics", type=Path, default=Path(DEFAULT_REVIEWED_RUBRICS_PATH))
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--output", type=Path, default=Path("output/auto_grade/reviewed_rubrics.validation.v1.json"))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_REVIEWED_RUBRICS_VALIDATION_REPORT_PATH))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=args.reviewed_rubrics,
        question_bank_path=args.question_bank,
        allow_missing=args.allow_missing,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
