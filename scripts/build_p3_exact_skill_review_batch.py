from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill import (
    DEFAULT_REVIEW_BATCH_DIR,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
    DEFAULT_REVIEWED_DECISIONS_PATH,
)
from exam_bank.p3_exact_skill.review_batch import build_p3_exact_skill_review_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a small P3 exact-skill human review batch packet.")
    parser.add_argument("--queue", type=Path, default=Path(DEFAULT_REVIEW_QUEUE_JSON_PATH))
    parser.add_argument("--reviewed", type=Path, default=Path(DEFAULT_REVIEWED_DECISIONS_PATH))
    parser.add_argument("--batch-id", default="batch_0001")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_REVIEW_BATCH_DIR))
    parser.add_argument("--status", default="clean_candidate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = build_p3_exact_skill_review_batch(
        queue_path=args.queue,
        reviewed_path=args.reviewed,
        batch_id=args.batch_id,
        limit=args.limit,
        out_dir=args.out_dir,
        status=args.status,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "dry_run": result["dry_run"],
                "batch_id": result["batch_id"],
                "selected_count": result["selected_count"],
                "paths": result["paths"],
                "skipped_count_by_reason": result["skipped_count_by_reason"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
