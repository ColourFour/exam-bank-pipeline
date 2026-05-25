from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.asset_manifest import validate_asset_references


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate canonical asset references across question-bank and Asterion exports.")
    parser.add_argument("--question-bank", default="output/json/question_bank.json")
    parser.add_argument("--asset-manifest", default="output/json/asset_manifest.v1.json")
    parser.add_argument("--asterion", default="output/asterion/exports/latest/asterion_question_bank_v1.json")
    parser.add_argument("--content-lab", default="output/asterion/exports/latest/asterion_content_lab_candidates_v1.json")
    parser.add_argument("--topic-routing", default="output/json/question_bank.topic_routing.v1.json")
    parser.add_argument("--artifact-root", default="output")
    parser.add_argument("--output", default="", help="Optional JSON validation report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_asset_references(
        question_bank_path=args.question_bank,
        asset_manifest_path=args.asset_manifest,
        asterion_path=args.asterion,
        content_lab_path=args.content_lab,
        topic_routing_path=args.topic_routing,
        artifact_root=args.artifact_root,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
