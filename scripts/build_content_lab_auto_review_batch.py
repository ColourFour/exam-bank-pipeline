from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.content_lab_auto_review import build_auto_review_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Content Lab P3 automated review batch.")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--question-bank", type=Path, required=True)
    parser.add_argument("--mark-events", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--target-pass-count", type=int, default=70)
    parser.add_argument("--buffer-count", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--skill-map",
        type=Path,
        default=Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
    )
    parser.add_argument(
        "--question-skill-mappings",
        type=Path,
        default=Path("exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"),
    )
    args = parser.parse_args(argv)
    payload = build_auto_review_batch(
        audit_dir=args.audit_dir,
        candidates_path=args.candidates,
        question_bank_path=args.question_bank,
        mark_events_path=args.mark_events,
        artifact_root=args.artifact_root,
        target_pass_count=args.target_pass_count,
        buffer_count=args.buffer_count,
        out_dir=args.out_dir,
        skill_map_path=args.skill_map,
        question_skill_mappings_path=args.question_skill_mappings,
    )
    print(json.dumps(payload["manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
