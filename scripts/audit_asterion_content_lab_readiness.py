from __future__ import annotations

import argparse
from pathlib import Path

from exam_bank.asterion_content_lab_audit import DEFAULT_SAMPLE_SEED, run_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Asterion Content Lab P3 candidate readiness gates.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("output/asterion/exports/latest/asterion_content_lab_candidates_v1.json"),
        help="Asterion Content Lab candidate export.",
    )
    parser.add_argument(
        "--question-bank",
        type=Path,
        default=Path("output/json/question_bank.json"),
        help="Source exam-bank question bank for validation status and fallback topic context.",
    )
    parser.add_argument(
        "--topic-routing",
        type=Path,
        default=Path("output/json/question_bank.topic_routing.v1.json"),
        help="Topic routing sidecar used only for deterministic stratification fallback.",
    )
    parser.add_argument(
        "--asterion-bank",
        type=Path,
        default=Path("output/asterion/exports/latest/asterion_question_bank_v1.json"),
        help="Asterion question-bank export used for artifact and quality-gate evidence.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("output"), help="Root for relative canonical artifacts.")
    parser.add_argument("--sample-size", type=int, default=100, help="Deterministic stratified P3 sample size.")
    parser.add_argument("--sample-seed", default=DEFAULT_SAMPLE_SEED, help="Stable seed for deterministic sample ordering.")
    parser.add_argument("--target-pass-rate", type=float, default=0.70, help="Target pass rate for reporting target_met.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/audits/asterion_content_lab_loop/latest"),
        help="Directory for generated audit reports.",
    )
    parser.add_argument(
        "--skill-map",
        type=Path,
        default=Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
        help="P3 skill map used for region stratification.",
    )
    parser.add_argument(
        "--question-skill-mappings",
        type=Path,
        default=Path("exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"),
        help="Question-skill mappings used only for region stratification fallback.",
    )
    parser.add_argument(
        "--reviewed-source-skills",
        type=Path,
        default=Path("data/review/p3_exact_skill_reviewed_decisions.v1.json"),
        help="Reviewed exact-skill/source-skill decisions used for evidence coverage reporting.",
    )
    parser.add_argument(
        "--reviewed-mark-events",
        type=Path,
        default=Path("data/review/p3_exact_skill_reviewed_mark_events.v1.json"),
        help="Reviewed mark-event decisions used for evidence coverage reporting.",
    )
    parser.add_argument(
        "--reviewed-decisions",
        type=Path,
        default=None,
        help="Optional imported automated reviewed-decision file for Content Lab P3 evidence.",
    )
    parser.add_argument(
        "--full-pool-report",
        action="store_true",
        help="Write full-pool P3 readiness candidate, region, blocker, and summary reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_audit(
        candidates_path=args.candidates,
        question_bank_path=args.question_bank,
        topic_routing_path=args.topic_routing,
        asterion_bank_path=args.asterion_bank,
        artifact_root=args.artifact_root,
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        skill_map_path=args.skill_map,
        question_skill_mappings_path=args.question_skill_mappings,
        reviewed_source_skills_path=args.reviewed_source_skills,
        reviewed_mark_events_path=args.reviewed_mark_events,
        reviewed_decisions_path=args.reviewed_decisions,
        target_pass_rate=args.target_pass_rate,
        full_pool_report=args.full_pool_report,
    )
    print(
        "Asterion Content Lab P3 readiness: "
        f"{summary['sample_passed']}/{summary['sample_size']} "
        f"({summary['sample_pass_rate']:.2%}) target_met={summary['target_met']}"
    )
    print(f"Wrote audit reports to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
