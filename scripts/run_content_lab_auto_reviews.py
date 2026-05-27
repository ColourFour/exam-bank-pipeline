from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.content_lab_auto_review import run_auto_reviews


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run image-first automated reviews for a Content Lab P3 batch.")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--provider", default="openai")
    args = parser.parse_args(argv)
    try:
        manifest = run_auto_reviews(
            batch_path=args.batch,
            out_path=args.out,
            max_records=args.max_records,
            dry_run=args.dry_run,
            model=args.model,
            provider=args.provider,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
