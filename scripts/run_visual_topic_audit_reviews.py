#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.visual_topic_audit import run_visual_topic_audit_reviews


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AI-assisted visual topic audit reviews for a batch.")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Decision JSONL output path. Defaults to visual_topic_audit_decisions.jsonl beside the batch.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out_path = args.out or args.batch.parent / "visual_topic_audit_decisions.jsonl"
    report = run_visual_topic_audit_reviews(
        batch_path=args.batch,
        out_path=out_path,
        max_records=args.max_records,
        model=args.model,
        provider=args.provider,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
