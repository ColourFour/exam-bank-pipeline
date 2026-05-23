from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill import DEFAULT_REVIEW_BATCH_DIR
from exam_bank.p3_exact_skill.visual_review import build_p3_exact_skill_visual_review_packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a visual HTML packet for a P3 exact-skill review batch.")
    parser.add_argument("--batch-dir", type=Path, default=Path(DEFAULT_REVIEW_BATCH_DIR))
    parser.add_argument("--batch-id", default="batch_0001")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = build_p3_exact_skill_visual_review_packet(
        batch_dir=args.batch_dir,
        batch_id=args.batch_id,
        repo_root=args.repo_root,
        output_path=args.output,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "dry_run": result["dry_run"],
                "batch_id": result["batch_id"],
                "selected_count": result["selected_count"],
                "output_path": result["output_path"],
                "inputs": result["inputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
