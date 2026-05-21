from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.auto_grade.review_batch import DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH, DEFAULT_REVIEW_BATCH_PATH
from exam_bank.auto_grade.reviewer_packet import DEFAULT_REVIEWER_PACKET_DIR, build_reviewer_packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 2C reviewer-facing markdown packet from a rubric review batch.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--review-batch", type=Path, default=Path(DEFAULT_REVIEW_BATCH_PATH))
    parser.add_argument("--reviewed-rubrics", type=Path, default=Path(DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_REVIEWER_PACKET_DIR))
    parser.add_argument("--no-html", action="store_true", help="Skip the optional dependency-free HTML index.")
    args = parser.parse_args(argv)

    report = build_reviewer_packet(
        review_batch_path=args.review_batch,
        question_bank_path=args.question_bank,
        reviewed_rubrics_path=args.reviewed_rubrics,
        output_dir=args.output_dir,
        include_html=not args.no_html,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
