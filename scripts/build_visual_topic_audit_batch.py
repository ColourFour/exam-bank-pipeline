#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.atomic_json import write_atomic_json
from exam_bank.visual_topic_audit import (
    DEFAULT_BASE_OVERLAP_REVIEW,
    DEFAULT_PACKET_AUDIT_REPORT,
    DEFAULT_TOPIC_PACKET_SUMMARY,
    DEFAULT_VISUAL_AUDIT_OUT_DIR,
    build_visual_topic_audit_batch,
    render_visual_topic_audit_batch_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a visual topic audit batch.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--packet-audit", type=Path, default=DEFAULT_PACKET_AUDIT_REPORT)
    parser.add_argument("--packet-summary", type=Path, default=DEFAULT_TOPIC_PACKET_SUMMARY)
    parser.add_argument("--taxonomy", type=Path, default=Path("exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--existing-overlap-review", type=Path, default=DEFAULT_BASE_OVERLAP_REVIEW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_VISUAL_AUDIT_OUT_DIR)
    parser.add_argument("--queue", choices=["missing", "ge3", "both"], default="both")
    parser.add_argument("--paper-family", choices=["p1", "p3", "p4", "p5"], default=None)
    parser.add_argument("--limit-papers", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_visual_topic_audit_batch(
        question_bank_path=args.question_bank,
        packet_audit_path=args.packet_audit,
        packet_summary_path=args.packet_summary,
        taxonomy_path=args.taxonomy,
        artifact_root=args.artifact_root,
        existing_overlap_review_path=args.existing_overlap_review,
        queue=args.queue,
        paper_family=args.paper_family,
        limit_papers=args.limit_papers,
        limit_questions=args.limit_questions,
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_atomic_json(report, args.out_dir / "visual_topic_audit_batch.json", sort_keys=True)
        (args.out_dir / "visual_topic_audit_batch.md").write_text(
            render_visual_topic_audit_batch_markdown(report),
            encoding="utf-8",
        )
    summary = {
        "batch_id": report["batch_id"],
        "out_dir": "" if args.dry_run else str(args.out_dir),
        **report["selection"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
