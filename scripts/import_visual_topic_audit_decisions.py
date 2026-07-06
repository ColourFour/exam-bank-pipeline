#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.visual_topic_audit import (
    DEFAULT_BASE_OVERLAP_REVIEW,
    DEFAULT_MERGED_OVERLAP_REVIEW,
    import_visual_topic_audit_decisions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import visual topic audit decisions into a merged overlap sidecar.")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--base-overlap-review", type=Path, default=DEFAULT_BASE_OVERLAP_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_MERGED_OVERLAP_REVIEW)
    parser.add_argument("--taxonomy", type=Path, default=Path("exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = import_visual_topic_audit_decisions(
        batch_path=args.batch,
        decisions_path=args.decisions,
        base_overlap_review_path=args.base_overlap_review,
        out_overlap_review_path=args.out,
        taxonomy_path=args.taxonomy,
        artifact_root=args.artifact_root,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
