from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.difficulty_index.sidecar import build_difficulty_index_sidecar, sidecar_summary


def run_build(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build advisory difficulty_index_v1 sidecar and reports.")
    parser.add_argument("--input", type=Path, default=Path("output/json/question_bank.json"), help="Path to question_bank.json.")
    parser.add_argument("--output", type=Path, default=Path("output/json/question_bank.difficulty_index.v1.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--mark-events", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--topic-routing", type=Path, default=Path("output/json/question_bank.topic_routing.v1.json"))
    parser.add_argument("--advisory-evidence", type=Path, default=Path("output/advisory_evidence/question_bank.advisory_evidence.v1.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    sidecar = build_difficulty_index_sidecar(
        question_bank_path=args.input,
        output_path=args.output,
        reports_dir=args.reports_dir,
        artifact_root=args.artifact_root,
        mark_events_path=args.mark_events,
        topic_routing_path=args.topic_routing,
        advisory_evidence_path=args.advisory_evidence,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "sidecar": str(args.output),
                "reports_dir": str(args.reports_dir),
                "summary": sidecar_summary(sidecar),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

