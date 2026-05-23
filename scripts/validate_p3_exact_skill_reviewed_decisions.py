from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.p3_exact_skill import DEFAULT_P3_SKILL_MAP_PATH, DEFAULT_REVIEWED_DECISIONS_PATH
from exam_bank.p3_exact_skill.reviewed_decisions import validate_reviewed_decisions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate P3 exact-skill reviewed-decision input records.")
    parser.add_argument("--reviewed-decisions", type=Path, default=Path(DEFAULT_REVIEWED_DECISIONS_PATH))
    parser.add_argument("--p3-skill-map", type=Path, default=Path(DEFAULT_P3_SKILL_MAP_PATH))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = validate_reviewed_decisions(
        reviewed_decisions_path=args.reviewed_decisions,
        p3_skill_map_path=args.p3_skill_map,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
