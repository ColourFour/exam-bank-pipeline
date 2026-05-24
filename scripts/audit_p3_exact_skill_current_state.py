from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill import (
    DEFAULT_AMBIGUITY_AUDIT_JSON_PATH,
    DEFAULT_CURRENT_STATE_AUDIT_JSON_PATH,
    DEFAULT_CURRENT_STATE_AUDIT_REPORT_PATH,
    DEFAULT_REVIEWED_DECISIONS_PATH,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
)
from exam_bank.p3_exact_skill.current_state_audit import build_p3_exact_skill_current_state_audit
from exam_bank.p3_exact_skill.part_decomposition import DEFAULT_PART_DECOMPOSITION_JSON_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the current P3 exact-skill evidence workflow state.")
    parser.add_argument("--queue", type=Path, default=Path(DEFAULT_REVIEW_QUEUE_JSON_PATH))
    parser.add_argument("--ambiguity-audit", type=Path, default=Path(DEFAULT_AMBIGUITY_AUDIT_JSON_PATH))
    parser.add_argument("--part-decomposition", type=Path, default=Path(DEFAULT_PART_DECOMPOSITION_JSON_PATH))
    parser.add_argument("--reviewed", type=Path, default=Path(DEFAULT_REVIEWED_DECISIONS_PATH))
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("output/asterion/exports/latest/p3_exact_skill_evidence_v1.json"),
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_CURRENT_STATE_AUDIT_JSON_PATH))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_CURRENT_STATE_AUDIT_REPORT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    audit = build_p3_exact_skill_current_state_audit(
        queue_path=args.queue,
        ambiguity_audit_path=args.ambiguity_audit,
        part_decomposition_path=args.part_decomposition,
        reviewed_path=args.reviewed,
        sidecar_path=args.sidecar,
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
                "readiness_verdicts": audit["readiness_verdicts"],
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
