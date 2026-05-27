from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.content_lab_auto_review import import_auto_review_decisions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import validated automated Content Lab P3 review decisions.")
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out-review-file", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument(
        "--skill-map",
        type=Path,
        default=Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
    )
    parser.add_argument("--mark-events", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    report = import_auto_review_decisions(
        decisions_path=args.decisions,
        batch_path=args.batch,
        out_review_file=args.out_review_file,
        dry_run=args.dry_run,
        confidence_threshold=args.confidence_threshold,
        skill_map_path=args.skill_map,
        mark_events_path=args.mark_events,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
