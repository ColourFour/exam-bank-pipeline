from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill import DEFAULT_REVIEWED_MARK_EVENTS_PATH
from exam_bank.p3_exact_skill.reviewed_mark_events import validate_reviewed_mark_events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate P3 reviewed mark-event decision records.")
    parser.add_argument("--reviewed-mark-events", type=Path, default=Path(DEFAULT_REVIEWED_MARK_EVENTS_PATH))
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--mark-events", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = validate_reviewed_mark_events(
        reviewed_mark_events_path=args.reviewed_mark_events,
        question_bank_path=args.question_bank,
        mark_events_path=args.mark_events,
        base_dir=args.base_dir,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
