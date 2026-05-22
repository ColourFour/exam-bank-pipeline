from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.auto_grade.constants import DEFAULT_REVIEWED_RUBRICS_PATH
from exam_bank.auto_grade.registry import DEFAULT_REGISTRY_REPORT_PATH, promote_reviewed_rubrics_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote already human-approved reviewed rubrics into the live reviewed-rubric registry."
    )
    parser.add_argument(
        "--source-reviewed-rubrics",
        type=Path,
        required=True,
        help="Reviewed-rubrics draft/workspace file to read. Only already approved records are selected.",
    )
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_REVIEWED_RUBRICS_PATH))
    parser.add_argument("--mode", choices=["replace-approved", "merge-approved"], default="replace-approved")
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_REGISTRY_REPORT_PATH))
    args = parser.parse_args(argv)

    result = promote_reviewed_rubrics_registry(
        source_reviewed_rubrics_path=args.source_reviewed_rubrics,
        question_bank_path=args.question_bank,
        output_path=args.output,
        mode=args.mode,
        report_path=args.report,
    )
    printable = {key: value for key, value in result.items() if key != "registry"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
