from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill import DEFAULT_REVIEW_QUEUE_JSON_PATH
from exam_bank.p3_exact_skill.part_decomposition import (
    DEFAULT_PART_DECOMPOSITION_JSON_PATH,
    DEFAULT_PART_DECOMPOSITION_REPORT_PATH,
    build_p3_exact_skill_part_decomposition,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the P3 exact-skill part-level decomposition review report.")
    parser.add_argument("--queue", type=Path, default=Path(DEFAULT_REVIEW_QUEUE_JSON_PATH))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_PART_DECOMPOSITION_JSON_PATH))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_PART_DECOMPOSITION_REPORT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = build_p3_exact_skill_part_decomposition(
        queue_path=args.queue,
        output_path=args.output,
        report_path=args.report,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "dry_run": args.dry_run,
                "summary": report["summary"],
                "output": str(args.output),
                "report": str(args.report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
