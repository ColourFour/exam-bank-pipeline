from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill.batch_conclusions import (
    DEFAULT_BATCH_ID,
    DEFAULT_OUTPUT_JSON_PATH,
    DEFAULT_OUTPUT_REPORT_PATH,
    build_manual_review_batch_conclusions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build reviewed-batch conclusion reports from P3 review batch artifacts.")
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--batch-dir", type=Path, default=Path("data/review/p3_exact_skill_batches"))
    parser.add_argument("--queue", type=Path, default=Path("reports/p3_exact_skill_review_queue.v1.json"))
    parser.add_argument("--reviewed-registry", type=Path, default=Path("data/review/p3_exact_skill_reviewed_decisions.v1.json"))
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON_PATH))
    parser.add_argument("--output-report", type=Path, default=Path(DEFAULT_OUTPUT_REPORT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = build_manual_review_batch_conclusions(
        batch_id=args.batch_id,
        batch_dir=args.batch_dir,
        queue_path=args.queue,
        reviewed_registry_path=args.reviewed_registry,
        output_json_path=args.output_json,
        output_report_path=args.output_report,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "dry_run": args.dry_run,
                "batch_id": payload["batch_id"],
                "reviewed_item_counts": payload["reviewed_item_counts"],
                "outcome_counts": payload["outcome_counts"],
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
