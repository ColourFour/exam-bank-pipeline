from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.storage_audit import (
    DEFAULT_REFERENCE_JSON_FILES,
    DEFAULT_SCAN_ROOTS,
    run_storage_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit output/report storage duplication using SHA-256 and write optimization reports.",
    )
    parser.add_argument("--scan-root", action="append", default=[], help="Directory to scan. May be repeated.")
    parser.add_argument("--reference-json", action="append", default=[], help="JSON export to inspect for file references. May be repeated.")
    parser.add_argument("--question-bank", default="output/json/question_bank.json", help="Question bank used to build asset manifest.")
    parser.add_argument("--asset-manifest", default="output/json/asset_manifest.v1.json", help="Asset manifest output path.")
    parser.add_argument("--json", default="reports/output_storage_duplicate_audit.v1.json", help="Audit JSON output path.")
    parser.add_argument("--markdown", default="reports/output_storage_duplicate_audit.md", help="Audit Markdown output path.")
    parser.add_argument("--plan", default="reports/output_storage_optimization_plan.md", help="Optimization plan Markdown output path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="Do not move files. This is the default.")
    mode.add_argument("--apply", action="store_true", help="Move conservative duplicate cleanup candidates to quarantine.")
    mode.add_argument("--apply-delete", "--delete", action="store_true", help="Hard-delete allowlisted exact non-canonical duplicates after writing a deletion manifest.")
    parser.add_argument(
        "--delete-manifest",
        default="reports/output_storage_delete_manifest.v1.json",
        help="Deletion manifest path for --apply-delete.",
    )
    parser.add_argument(
        "--quarantine-dir",
        default="",
        help="Quarantine directory for --apply. Defaults to output/_quarantine_storage_cleanup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scan_roots = [Path(path) for path in args.scan_root] if args.scan_root else list(DEFAULT_SCAN_ROOTS)
    reference_json = [Path(path) for path in args.reference_json] if args.reference_json else list(DEFAULT_REFERENCE_JSON_FILES)
    audit = run_storage_audit(
        scan_roots=scan_roots,
        reference_json_files=reference_json,
        question_bank_path=args.question_bank,
        asset_manifest_path=args.asset_manifest,
        json_path=args.json,
        markdown_path=args.markdown,
        plan_path=args.plan,
        dry_run=not args.apply and not args.apply_delete,
        apply=bool(args.apply),
        apply_delete=bool(args.apply_delete),
        delete_manifest_path=args.delete_manifest,
        quarantine_dir=Path(args.quarantine_dir) if args.quarantine_dir else None,
    )
    delete_result = dict(audit["delete_result"])
    if len(delete_result.get("deleted_files") or []) > 10:
        delete_result["deleted_files_sample"] = delete_result["deleted_files"][:10]
        del delete_result["deleted_files"]
    print(
        json.dumps(
            {
                "ok": True,
                "summary": audit["summary"],
                "cleanup_result": audit["cleanup_result"],
                "delete_result": delete_result,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
