from __future__ import annotations

import argparse
from pathlib import Path

from exam_bank.asterion_student_runtime_safe import run_runtime_safe_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and project Asterion P3 student-runtime-safe Content Lab candidates.")
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
        help="Topic routing sidecar used for region fallback.",
    )
    parser.add_argument(
        "--asterion-bank",
        type=Path,
        default=Path("output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json"),
        help="Asterion all-course catalog export used for runtime roles and canonical artifact evidence.",
    )
    parser.add_argument(
        "--mark-events",
        type=Path,
        default=Path("output/json/question_bank.mark_events.v1.json"),
        help="Canonical mark-event sidecar.",
    )
    parser.add_argument(
        "--reviewed-decisions",
        type=Path,
        default=Path("data/review/content_lab_p3_auto_reviewed_decisions_merged_0002.v1.json"),
        help="Imported Content Lab reviewed-decision evidence.",
    )
    parser.add_argument(
        "--reviewed-source-skills",
        type=Path,
        default=Path("data/review/p3_exact_skill_reviewed_decisions.v1.json"),
        help="Reviewed exact/source skill decisions.",
    )
    parser.add_argument(
        "--reviewed-mark-events",
        type=Path,
        default=Path("data/review/p3_exact_skill_reviewed_mark_events.v1.json"),
        help="Reviewed mark-event decisions.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("output"), help="Root for relative canonical artifacts.")
    parser.add_argument("--target-pass-rate", type=float, default=0.50, help="Runtime-safe target pass rate.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/audits/asterion_student_runtime_safe_loop/iteration_001"),
        help="Directory for generated audit reports.",
    )
    parser.add_argument(
        "--skill-map",
        type=Path,
        default=Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
        help="P3 skill map used for region classification.",
    )
    parser.add_argument(
        "--question-skill-mappings",
        type=Path,
        default=Path("exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"),
        help="Question-skill mappings used for region fallback.",
    )
    parser.add_argument(
        "--deterministic-sample",
        type=Path,
        default=Path("output/audits/asterion_content_lab_loop/iteration_005/sample_results.csv"),
        help="Loop 005 deterministic sample for overlap reporting.",
    )
    parser.add_argument(
        "--regeneration-backlog",
        type=Path,
        default=Path("output/review/content_lab_p3_auto_loop_005/candidate_regeneration_backlog.json"),
        help="Loop 005 regeneration/remapping backlog.",
    )
    parser.add_argument(
        "--promotion-decisions",
        type=Path,
        default=None,
        help="Existing runtime-safe promotion decisions to validate and project.",
    )
    parser.add_argument(
        "--write-promotion-decisions",
        type=Path,
        default=None,
        help="Write reviewed student_runtime_safe promotion decisions for candidates satisfying the contract.",
    )
    parser.add_argument(
        "--write-export-decisions",
        type=Path,
        default=None,
        help="Write export-sidecar copy of valid runtime-safe decisions.",
    )
    parser.add_argument(
        "--write-export-candidates",
        type=Path,
        default=None,
        help="Write export-sidecar projected student-runtime-safe candidates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_runtime_safe_audit(
        candidates_path=args.candidates,
        question_bank_path=args.question_bank,
        topic_routing_path=args.topic_routing,
        asterion_bank_path=args.asterion_bank,
        mark_events_path=args.mark_events,
        reviewed_decisions_path=args.reviewed_decisions,
        reviewed_source_skills_path=args.reviewed_source_skills,
        reviewed_mark_events_path=args.reviewed_mark_events,
        artifact_root=args.artifact_root,
        out_dir=args.out_dir,
        target_pass_rate=args.target_pass_rate,
        skill_map_path=args.skill_map,
        question_skill_mappings_path=args.question_skill_mappings,
        deterministic_sample_path=args.deterministic_sample,
        regeneration_backlog_path=args.regeneration_backlog,
        promotion_decisions_path=args.promotion_decisions,
        write_promotion_decisions_path=args.write_promotion_decisions,
        write_export_decisions_path=args.write_export_decisions,
        write_export_candidates_path=args.write_export_candidates,
    )
    print(
        "Asterion P3 student-runtime-safe: "
        f"current={summary['current_student_runtime_safe_true_count']}/{summary['total_p3_candidate_count']} "
        f"({summary['current_student_runtime_safe_percentage']:.2%}) "
        f"final={summary['final_student_runtime_safe_true_count']}/{summary['total_p3_candidate_count']} "
        f"({summary['final_student_runtime_safe_percentage']:.2%}) "
        f"target_met={summary['target_met']}"
    )
    print(f"Wrote runtime-safe audit reports to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
