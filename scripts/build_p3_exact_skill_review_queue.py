from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill import (
    DEFAULT_P3_SKILL_MAPPINGS_PATH,
    DEFAULT_P3_TOPIC_ASSIGNMENTS_PATH,
    DEFAULT_REVIEWED_DECISIONS_PATH,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
    DEFAULT_REVIEW_QUEUE_REPORT_PATH,
)
from exam_bank.p3_exact_skill.review_queue import build_p3_exact_skill_review_queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the P3 exact-skill human review queue report.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--topic-routing", type=Path, default=Path("output/json/question_bank.topic_routing.v1.json"))
    parser.add_argument(
        "--asterion-question-bank",
        type=Path,
        default=Path("output/asterion/exports/latest/asterion_question_bank_v1.json"),
    )
    parser.add_argument(
        "--content-lab-candidates",
        type=Path,
        default=Path("output/asterion/exports/latest/asterion_content_lab_candidates_v1.json"),
    )
    parser.add_argument("--mark-events", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--p3-skill-mappings", type=Path, default=Path(DEFAULT_P3_SKILL_MAPPINGS_PATH))
    parser.add_argument("--p3-topic-assignments", type=Path, default=Path(DEFAULT_P3_TOPIC_ASSIGNMENTS_PATH))
    parser.add_argument("--reviewed-decisions", type=Path, default=Path(DEFAULT_REVIEWED_DECISIONS_PATH))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_REVIEW_QUEUE_JSON_PATH))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_REVIEW_QUEUE_REPORT_PATH))
    args = parser.parse_args(argv)

    queue = build_p3_exact_skill_review_queue(
        question_bank_path=args.question_bank,
        topic_routing_path=args.topic_routing,
        asterion_question_bank_path=args.asterion_question_bank,
        content_lab_candidates_path=args.content_lab_candidates,
        mark_events_path=args.mark_events,
        p3_skill_mappings_path=args.p3_skill_mappings,
        p3_topic_assignments_path=args.p3_topic_assignments,
        reviewed_decisions_path=args.reviewed_decisions,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps({"ok": True, "summary": queue["summary"], "report": str(args.report)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
