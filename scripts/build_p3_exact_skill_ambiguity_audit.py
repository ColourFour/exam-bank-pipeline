from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill import (
    DEFAULT_AMBIGUITY_AUDIT_JSON_PATH,
    DEFAULT_AMBIGUITY_AUDIT_REPORT_PATH,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
)
from exam_bank.p3_exact_skill.ambiguity_audit import build_p3_exact_skill_ambiguity_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a P3 exact-skill ambiguity-reduction audit.")
    parser.add_argument("--queue", type=Path, default=Path(DEFAULT_REVIEW_QUEUE_JSON_PATH))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_AMBIGUITY_AUDIT_JSON_PATH))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_AMBIGUITY_AUDIT_REPORT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    audit = build_p3_exact_skill_ambiguity_audit(
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
                "summary": audit["summary"],
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
