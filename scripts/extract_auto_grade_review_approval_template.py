from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.auto_grade.review_batch import DEFAULT_REVIEW_BATCH_PATH
from exam_bank.auto_grade.reviewer_packet import build_approval_template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract an unapproved reviewed-rubric JSON template for one review-batch candidate."
    )
    parser.add_argument("--review-batch", type=Path, default=Path(DEFAULT_REVIEW_BATCH_PATH))
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--rubric-id")
    selector.add_argument("--question-id")
    selector.add_argument("--first", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    template = build_approval_template(
        review_batch_path=args.review_batch,
        rubric_id=args.rubric_id,
        question_id=args.question_id,
        first=args.first,
        output_path=args.output,
    )
    print(json.dumps({"output": str(args.output), "rubric_id": template["rubric"]["rubric_id"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
