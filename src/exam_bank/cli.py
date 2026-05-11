from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_status import RunStatusTracker, completed_batch_ids, default_status_root_for_output, resolve_run_id
from .audit import write_audit
from .asterion_export import (
    ASTERION_EXPORT_FILENAME,
    CONTENT_LAB_EXPORT_FILENAME,
    export_asterion_content_lab_candidates,
    export_asterion_question_bank,
)
from .auto_triage import (
    build_auto_triage_runbook,
    compare_auto_triage_iteration,
    create_auto_triage_plan,
    write_status_report,
)
from .config import AppConfig, load_config
from . import deepseek_enrich
from .pipeline import PipelineResult, process_inputs
from .triage import ISSUE_SET_ALL_NON_READY, ISSUE_SET_HARD_FAILURES, compare_iteration, create_triage_iteration, serve_iteration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CAIE question papers and mark schemes into paper-first folders plus JSON metadata.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser(
        "process",
        help="Process a folder of question paper PDFs and mark scheme PDFs.",
    )
    process.add_argument(
        "--input",
        default="input",
        help="Input folder. The scan is recursive and supports either mixed PDFs or question_papers/mark_schemes subfolders.",
    )
    process.add_argument(
        "--output",
        default="output",
        help="Output root folder for paper-first exports and JSON metadata.",
    )
    process.add_argument(
        "--config",
        default="config.yaml",
        help="Optional config.yaml path for operational overrides.",
    )
    process.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Run optional Tesseract OCR on rendered question crop PNGs and sparse PDF regions.",
    )
    process.add_argument("--ocr-language", default="", help="Optional Tesseract language override, e.g. eng.")
    process.add_argument("--tesseract-cmd", default="", help="Optional path to the tesseract binary.")
    _add_run_tracking_arguments(process)
    process.set_defaults(func=cmd_process)

    audit = subparsers.add_parser(
        "audit",
        help="Audit visual-first question text trust and curation readiness in an exported question bank JSON.",
    )
    audit.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    audit.add_argument("--output", default="", help="Optional path to write the audit JSON report.")
    audit.set_defaults(func=cmd_audit)

    asterion = subparsers.add_parser(
        "asterion-export",
        help="Write the conservative Asterion-safe question-bank projection.",
    )
    asterion.add_argument("--input", default="output/json/question_bank.json", help="Path to schema v2 question_bank.json.")
    asterion.add_argument(
        "--output",
        default="",
        help=f"Output path. Defaults to {ASTERION_EXPORT_FILENAME} beside the input JSON.",
    )
    asterion.add_argument(
        "--artifact-root",
        default="",
        help="Root used to resolve question and mark-scheme artifact paths. Defaults to the parent of the input json/ folder.",
    )
    asterion.add_argument(
        "--skill-map",
        default="",
        help="Optional skill-map JSON sidecar used to attach mapped skill IDs to Asterion subpart mark events.",
    )
    asterion.set_defaults(func=cmd_asterion_export)

    content_lab = subparsers.add_parser(
        "asterion-content-lab-candidates",
        help="Write Asterion Content Lab candidate metadata without generating student-facing content.",
    )
    content_lab.add_argument(
        "--input",
        default="output/json/question_bank.json",
        help="Path to schema v2 question_bank.json or asterion_question_bank_v1.json.",
    )
    content_lab.add_argument(
        "--output",
        default="",
        help=f"Output path. Defaults to {CONTENT_LAB_EXPORT_FILENAME} beside the input JSON.",
    )
    content_lab.add_argument(
        "--artifact-root",
        default="",
        help="Root used to resolve question and mark-scheme artifact paths when input is question_bank.json.",
    )
    content_lab.add_argument(
        "--skill-map",
        default="",
        help="Optional skill-map JSON sidecar used to attach mapped skill IDs to Content Lab mark-event candidates.",
    )
    content_lab.set_defaults(func=cmd_asterion_content_lab_candidates)

    enrich_ai = subparsers.add_parser(
        "enrich-ai",
        help="Run DeepSeek AI-assisted enrichment against canonical topic/subtopic/skill IDs.",
    )
    deepseek_enrich.add_ai_assisted_cli_arguments(enrich_ai)
    enrich_ai.set_defaults(func=cmd_enrich_ai)

    triage_sample = subparsers.add_parser(
        "triage-sample",
        help="Create a deterministic hard-failure triage sample and HTML review gallery.",
    )
    triage_sample.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    triage_sample.add_argument(
        "--output-root",
        default="",
        help="Triage output root. Defaults to a triage/ folder beside the exported JSON output tree.",
    )
    triage_sample.add_argument("--iteration", default="", help="Optional iteration folder name, e.g. iteration_001.")
    triage_sample.add_argument(
        "--issue-set",
        choices=[ISSUE_SET_HARD_FAILURES, ISSUE_SET_ALL_NON_READY],
        default=ISSUE_SET_HARD_FAILURES,
        help="Record set to sample from.",
    )
    triage_sample.add_argument("--sample-size", type=int, default=30, help="Maximum number of questions to sample.")
    triage_sample.add_argument("--target", default="auto", help="Primary issue to sample, or auto for the largest issue.")
    triage_sample.add_argument("--seed", type=int, default=1, help="Deterministic sample seed.")
    triage_sample.set_defaults(func=cmd_triage_sample)

    triage_serve = subparsers.add_parser(
        "triage-serve",
        help="Serve a triage iteration gallery and capture review labels to review.jsonl.",
    )
    triage_serve.add_argument("--iteration", required=True, help="Path to an output/triage/iteration_### folder.")
    triage_serve.add_argument("--host", default="127.0.0.1", help="Host for the local review server.")
    triage_serve.add_argument("--port", type=int, default=8765, help="Port for the local review server.")
    triage_serve.set_defaults(func=cmd_triage_serve)

    triage_compare = subparsers.add_parser(
        "triage-compare",
        help="Compare a triage baseline with a current full-pipeline export.",
    )
    triage_compare.add_argument("--iteration", required=True, help="Path to an output/triage/iteration_### folder.")
    triage_compare.add_argument("--current", default="output/json/question_bank.json", help="Current question_bank.json.")
    triage_compare.add_argument("--output", default="", help="Optional path to write the comparison report JSON.")
    triage_compare.set_defaults(func=cmd_triage_compare)

    auto_status = subparsers.add_parser(
        "auto-triage-status",
        help="Report corpus health metrics used by the automated triage loop.",
    )
    auto_status.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    auto_status.add_argument("--output", default="", help="Optional path to write the status JSON report.")
    auto_status.set_defaults(func=cmd_auto_triage_status)

    auto_plan = subparsers.add_parser(
        "auto-triage-plan",
        help="Create the next evidence-gated auto-triage agent handoff iteration.",
    )
    auto_plan.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    auto_plan.add_argument(
        "--handoff-root",
        default="agent_handoffs/auto_triage",
        help="Root folder for auto-triage handoff iterations.",
    )
    auto_plan.add_argument(
        "--target-max-hard-failures",
        type=int,
        required=True,
        help="Stop planning when hard failures are at or below this count.",
    )
    auto_plan.add_argument("--sample-size", type=int, default=30, help="Triage sample size for the selected target.")
    auto_plan.add_argument("--seed", type=int, default=1, help="Deterministic triage sample seed.")
    auto_plan.add_argument(
        "--triage-root",
        default="",
        help="Output triage root. Defaults to the triage folder beside the input output root.",
    )
    auto_plan.add_argument(
        "--candidate-output",
        default="output_ocr_candidate",
        help="Output folder to use in generated OCR rerun commands.",
    )
    auto_plan.set_defaults(func=cmd_auto_triage_plan)

    auto_compare = subparsers.add_parser(
        "auto-triage-compare",
        help="Compare a completed auto-triage iteration and write an acceptance decision.",
    )
    auto_compare.add_argument("--iteration", required=True, help="Path to agent_handoffs/auto_triage/iteration_###.")
    auto_compare.add_argument("--baseline-triage", required=True, help="Frozen output/triage/iteration_### baseline.")
    auto_compare.add_argument("--current", default="output/json/question_bank.json", help="Current question_bank.json.")
    auto_compare.add_argument("--output", default="", help="Optional path to write the triage comparison JSON.")
    auto_compare.add_argument(
        "--test-status",
        choices=["pass", "fail", "unknown"],
        default="unknown",
        help="Full pytest result. Accepted decisions require pass.",
    )
    auto_compare.add_argument(
        "--max-worsened-records",
        type=int,
        default=0,
        help="Maximum allowed worsened_records sample count.",
    )
    auto_compare.add_argument(
        "--target-material-decrease",
        type=int,
        default=1,
        help="Minimum selected-target issue decrease that counts as material.",
    )
    auto_compare.add_argument(
        "--max-hard-failure-increase",
        type=int,
        default=0,
        help="Allowed hard-failure increase before rejection.",
    )
    auto_compare.add_argument(
        "--max-status-regression",
        type=int,
        default=10,
        help="Allowed increase in each broad bad-status bucket before rejection.",
    )
    auto_compare.set_defaults(func=cmd_auto_triage_compare)

    auto_runbook = subparsers.add_parser(
        "auto-triage-runbook",
        help="Print the next commands for an auto-triage iteration.",
    )
    auto_runbook.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    auto_runbook.add_argument(
        "--handoff-root",
        default="agent_handoffs/auto_triage",
        help="Root folder for auto-triage handoff iterations.",
    )
    auto_runbook.add_argument("--iteration", default="", help="Auto-triage handoff iteration path or name.")
    auto_runbook.add_argument("--baseline-triage", default="", help="Frozen output/triage iteration path.")
    auto_runbook.add_argument(
        "--candidate-output",
        default="output_ocr_candidate",
        help="Output folder for generated OCR rerun commands.",
    )
    auto_runbook.add_argument("--sample-size", type=int, default=30, help="Fallback sample size.")
    auto_runbook.add_argument("--seed", type=int, default=1, help="Fallback triage sample seed.")
    auto_runbook.set_defaults(func=cmd_auto_triage_runbook)
    return parser


def _add_run_tracking_arguments(parser: argparse.ArgumentParser, *, include_resume: bool = True) -> None:
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true", default=True, help="Show terminal progress updates.")
    progress.add_argument("--no-progress", dest="progress", action="store_false", help="Disable terminal progress updates.")
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=None,
        help="Directory that stores run_status/<run_id> files. Defaults under the output root.",
    )
    parser.add_argument("--run-id", default="", help="Optional stable run ID for status files and resume.")
    if include_resume:
        parser.add_argument("--resume", action="store_true", help="Resume a previous run and skip completed batches.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="With --resume, rerun completed batches instead of skipping them.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def cmd_process(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.enable_ocr:
        config.ocr.enabled = True
    if args.ocr_language:
        config.ocr.language = args.ocr_language
    if args.tesseract_cmd:
        config.ocr.tesseract_cmd = args.tesseract_cmd
    _configure_runtime_paths(config, Path(args.input), Path(args.output))
    status_root = Path(args.status_dir) if getattr(args, "status_dir", None) else default_status_root_for_output(args.output)
    run_id = resolve_run_id(
        status_root=status_root,
        run_type="standard",
        requested_run_id=getattr(args, "run_id", "") or None,
        resume=bool(getattr(args, "resume", False)),
    )
    tracker = RunStatusTracker(
        run_id=run_id,
        run_type="standard",
        status_root=status_root,
        command=_command_text(args, "process"),
        input_paths=[args.input],
        output_paths=[Path(args.output) / "json" / config.naming.json_name],
        config_paths=[args.config],
        progress=bool(getattr(args, "progress", True)),
    )
    tracker.start(phase="scanning_inputs")
    resume_batches = completed_batch_ids(tracker.status_dir) if getattr(args, "resume", False) else set()
    try:
        result = process_inputs(
            args.input,
            config,
            progress=tracker,
            resume_completed_batch_ids=resume_batches,
            force_rerun=bool(getattr(args, "force_rerun", False)),
        )
        tracker.finish("completed")
    except KeyboardInterrupt:
        tracker.finish("interrupted", error_summary="KeyboardInterrupt")
        print(tracker.final_summary())
        return 130
    except Exception as exc:
        tracker.finish("failed", error_summary=f"{exc.__class__.__name__}: {exc}")
        print(tracker.final_summary())
        raise
    _print_result(result)
    print(tracker.final_summary())
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    report = write_audit(args.input, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_asterion_export(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    skill_map_path = Path(args.skill_map) if args.skill_map else None
    path = export_asterion_question_bank(args.input, output, artifact_root=artifact_root, skill_map_path=skill_map_path)
    print(json.dumps({"output": str(path)}, indent=2, ensure_ascii=False))
    return 0


def cmd_asterion_content_lab_candidates(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    skill_map_path = Path(args.skill_map) if args.skill_map else None
    path = export_asterion_content_lab_candidates(
        args.input,
        output,
        artifact_root=artifact_root,
        skill_map_path=skill_map_path,
    )
    print(json.dumps({"output": str(path)}, indent=2, ensure_ascii=False))
    return 0


def cmd_enrich_ai(args: argparse.Namespace) -> int:
    deepseek_enrich.finalize_ai_assisted_args(args)
    return deepseek_enrich.run_ai_assisted_from_args(args)


def cmd_triage_sample(args: argparse.Namespace) -> int:
    triage_root = Path(args.output_root) if args.output_root else None
    iteration = Path(args.iteration) if args.iteration else None
    summary = create_triage_iteration(
        args.input,
        triage_root=triage_root,
        iteration=iteration,
        issue_set=args.issue_set,
        sample_size=args.sample_size,
        target=args.target,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_triage_serve(args: argparse.Namespace) -> int:
    serve_iteration(args.iteration, host=args.host, port=args.port)
    return 0


def cmd_triage_compare(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    report = compare_iteration(args.iteration, current_path=args.current, output_path=output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_auto_triage_status(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    report = write_status_report(args.input, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_auto_triage_plan(args: argparse.Namespace) -> int:
    triage_root = Path(args.triage_root) if args.triage_root else None
    report = create_auto_triage_plan(
        args.input,
        handoff_root=args.handoff_root,
        target_max_hard_failures=args.target_max_hard_failures,
        sample_size=args.sample_size,
        seed=args.seed,
        triage_root=triage_root,
        candidate_output=args.candidate_output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_auto_triage_compare(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    report = compare_auto_triage_iteration(
        iteration=args.iteration,
        baseline_triage=args.baseline_triage,
        current_path=args.current,
        output_path=output,
        test_status=args.test_status,
        max_worsened_records=args.max_worsened_records,
        target_material_decrease=args.target_material_decrease,
        max_hard_failure_increase=args.max_hard_failure_increase,
        max_status_regression=args.max_status_regression,
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 1 if report["decision"]["decision"] == "rejected" else 0


def cmd_auto_triage_runbook(args: argparse.Namespace) -> int:
    report = build_auto_triage_runbook(
        input_path=args.input,
        handoff_root=args.handoff_root,
        iteration=Path(args.iteration) if args.iteration else None,
        baseline_triage=Path(args.baseline_triage) if args.baseline_triage else None,
        candidate_output=args.candidate_output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(_runbook_text(report))
    return 0


def _configure_runtime_paths(config: AppConfig, input_root: Path, output_root: Path) -> None:
    config.output.apply_root(output_root)

    if input_root.is_dir():
        question_dir = input_root / "question_papers"
        mark_scheme_dir = input_root / "mark_schemes"
        mappings_dir = input_root / "mappings"
        if question_dir.exists():
            config.input.question_papers_dir = question_dir
        else:
            config.input.question_papers_dir = input_root
        if mark_scheme_dir.exists():
            config.input.mark_schemes_dir = mark_scheme_dir
        else:
            config.input.mark_schemes_dir = input_root
        if mappings_dir.exists():
            config.input.mappings_dir = mappings_dir


def _print_result(result: PipelineResult) -> None:
    papers = sorted({record.paper_name for record in result.records})
    print(f"Processed questions: {result.question_count if result.question_count is not None else len(result.records)}")
    print(f"Processed papers: {result.paper_count if result.paper_count is not None else len(papers)}")
    print(f"Output root: {result.output_root}")
    print(f"JSON: {result.json_path}")


def _command_text(args: argparse.Namespace, fallback: str) -> str:
    command = getattr(args, "command", fallback) or fallback
    parts = [str(command)]
    for key, value in sorted(vars(args).items()):
        if key in {"func", "command"} or value is None or value == "" or value is False:
            continue
        option = f"--{key.replace('_', '-')}"
        if value is True:
            parts.append(option)
        else:
            parts.extend([option, str(value)])
    return " ".join(parts)


def _runbook_text(report: dict[str, object]) -> str:
    commands = report.get("commands") if isinstance(report.get("commands"), dict) else {}
    selected_target = report.get("selected_target") if isinstance(report.get("selected_target"), dict) else {}
    lines = [
        f"Iteration: {report.get('iteration')}",
        f"Baseline triage: {report.get('baseline_triage')}",
        f"Selected target: {selected_target.get('issue', 'unknown')}",
        "",
        "Next commands:",
    ]
    for label in [
        "triage_sample",
        "triage_serve",
        "full_ocr_rerun",
        "ocr_verification",
        "full_tests",
        "triage_comparison",
    ]:
        command = commands.get(label)
        if command:
            lines.extend(["", f"{label}:", str(command)])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
