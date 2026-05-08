from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.audit import write_audit
from exam_bank.triage import compare_iteration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a current question bank against a frozen triage baseline and fail on quality regressions.",
    )
    parser.add_argument("--iteration", default="output/triage/iteration_004", help="Frozen triage iteration folder.")
    parser.add_argument("--current", default="output/json/question_bank.json", help="Current question bank JSON.")
    parser.add_argument("--max-hard-failure-delta", type=int, default=0, help="Allowed hard-failure increase.")
    parser.add_argument("--max-target-issue", type=int, default=-1, help="Optional maximum current target issue count.")
    parser.add_argument(
        "--require-target-improvement",
        action="store_true",
        help="Fail unless the target issue count improves relative to the baseline.",
    )
    parser.add_argument("--audit-output", default="", help="Optional path to write the audit report JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_iteration(args.iteration, current_path=args.current)
    audit_output = Path(args.audit_output) if args.audit_output else None
    audit = write_audit(args.current, audit_output)

    failures = _quality_failures(
        report,
        max_hard_failure_delta=args.max_hard_failure_delta,
        max_target_issue=args.max_target_issue,
        require_target_improvement=args.require_target_improvement,
    )
    summary = {
        "current": args.current,
        "iteration": args.iteration,
        "target": report["target"],
        "baseline_hard_failure_count": report["baseline_hard_failure_count"],
        "current_hard_failure_count": report["current_hard_failure_count"],
        "hard_failure_delta": report["hard_failure_delta"],
        "baseline_target_issue_count": report["baseline_target_issue_count"],
        "current_target_issue_count": report["current_target_issue_count"],
        "target_issue_delta": report["target_issue_delta"],
        "visual_curation_status_counts": audit.get("visual_curation_status_counts", {}),
        "text_only_status_counts": audit.get("text_only_status_counts", {}),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failures else 0


def _quality_failures(
    report: dict[str, object],
    *,
    max_hard_failure_delta: int,
    max_target_issue: int,
    require_target_improvement: bool,
) -> list[str]:
    failures: list[str] = []
    hard_failure_delta = int(report.get("hard_failure_delta") or 0)
    target_issue_delta = int(report.get("target_issue_delta") or 0)
    current_target_issue_count = int(report.get("current_target_issue_count") or 0)

    if hard_failure_delta > max_hard_failure_delta:
        failures.append(f"hard_failure_delta {hard_failure_delta} exceeds allowed {max_hard_failure_delta}")
    if max_target_issue >= 0 and current_target_issue_count > max_target_issue:
        failures.append(f"target issue count {current_target_issue_count} exceeds allowed {max_target_issue}")
    if require_target_improvement and target_issue_delta >= 0:
        failures.append(f"target issue delta {target_issue_delta} is not an improvement")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
