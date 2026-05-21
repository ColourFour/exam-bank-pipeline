from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.auto_grade.review_batch import (
    DEFAULT_COMPLETION_REPORT_PATH,
    DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH,
    check_rubric_review_completion,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Phase 2B reviewed-rubric draft completion without approving it.")
    parser.add_argument("--reviewed-rubrics", type=Path, default=Path(DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_COMPLETION_REPORT_PATH))
    args = parser.parse_args(argv)

    report = check_rubric_review_completion(reviewed_rubrics_path=args.reviewed_rubrics, report_path=args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
