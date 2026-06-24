from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_status import RunStatusTracker, completed_batch_ids, default_status_root_for_output, resolve_run_id
from .audit import write_audit, write_current_output_integrity_audit
from .asterion_export import (
    ASTERION_CATALOG_FILENAME,
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
from . import deepseek_enrich, topic_confidence_rescoring, topic_packets, topic_routing
from .export_summary_diff import ExportSummaryDiffError, compare_export_summaries, render_export_summary_diff
from .mark_scheme_regeneration import regenerate_mark_scheme_pngs_from_question_bank
from .output_management import (
    build_cleanup_plan,
    build_output_inventory,
    render_cleanup_plan_markdown,
    render_inventory_markdown,
    write_cleanup_plan_outputs,
    write_inventory_outputs,
)
from .output_structure_normalization import normalize_output_structure, validate_normalized_output
from .pipeline import PipelineResult, process_inputs
from .question_regeneration import regenerate_question_pngs_from_question_bank
from .triage import (
    ISSUE_SET_ALL_NON_READY,
    ISSUE_SET_HARD_FAILURES,
    compare_iteration,
    create_suspicious_crop_review_pack,
    create_triage_iteration,
    serve_iteration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CAIE question papers and mark schemes into paper-first folders plus JSON metadata.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingress = subparsers.add_parser(
        "ingress",
        help="Run external source ingestion commands.",
    )
    ingress_subparsers = ingress.add_subparsers(dest="ingress_command", required=True)
    pastpapers_co = ingress_subparsers.add_parser(
        "pastpapers-co",
        help="Scrape CAIE 9709 Mathematics papers from PastPapers.co into exam-bank input records.",
    )
    pastpapers_co.add_argument("--input", type=Path, help="Existing exam-bank input JSON/JSONL to merge into.")
    pastpapers_co.add_argument(
        "--output",
        type=Path,
        default=Path("exam_bank_input.jsonl"),
        help="Output JSON/JSONL path. Defaults to exam_bank_input.jsonl.",
    )
    pastpapers_co.add_argument("--min-year", type=int, default=2008)
    pastpapers_co.add_argument("--max-year", type=int, default=2025)
    pastpapers_co.set_defaults(func=cmd_ingress_pastpapers_co)

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

    quiz_packet = subparsers.add_parser(
        "quiz-packet",
        help="Run the local quiz packet workflow from assignment.pdf and scans/ in one command.",
    )
    quiz_packet.add_argument("--quiz-dir", type=Path, required=True, help="Quiz folder containing assignment.pdf and scans/.")
    quiz_packet.add_argument("--course-id", required=True, help="Course ID for generated assignment metadata.")
    quiz_packet.add_argument("--assignment-pdf", type=Path, default=None, help="Optional assignment PDF path. Defaults to quiz-dir/assignment.pdf.")
    quiz_packet.add_argument("--assignment-id", default="", help="Optional assignment ID. Defaults to the quiz folder name.")
    quiz_packet.add_argument("--class-id", default="local_class", help="Class ID for generated assignment and roster rows.")
    quiz_packet.add_argument("--timezone", default="", help="Timezone for generated assignment metadata.")
    quiz_packet.add_argument("--output-root", type=Path, default=Path("output/submissions"), help="Submission output root.")
    quiz_packet.add_argument("--reports-root", type=Path, default=Path("reports/submissions"), help="Submission reports root.")
    quiz_packet.add_argument("--open-report", action="store_true", help="Open the teacher report after it is written when possible.")
    quiz_packet.add_argument("--no-grade", action="store_true", help="Skip quiz-packet draft grading artifacts.")
    quiz_packet.add_argument("--force", action="store_true", help="Regenerate assignment.json even if it already exists.")
    quiz_packet.set_defaults(func=cmd_quiz_packet)

    grade_quiz_bma = subparsers.add_parser(
        "grade-quiz-bma",
        help="Build evidence-based B/M/A visual-first grading matrix artifacts for a local quiz packet.",
    )
    grade_quiz_bma.add_argument("--assignment-id", required=True, help="Assignment ID under output/submissions.")
    grade_quiz_bma.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    grade_quiz_bma.add_argument("--reports-root", type=Path, default=Path("reports/submissions"))
    grade_quiz_bma.add_argument("--mode", default="visual-first", choices=["visual-first"])
    grade_quiz_bma.add_argument("--open-report", action="store_true")
    grade_quiz_bma.add_argument("--allow-suspicious-full-marks", action="store_true")
    grade_quiz_bma.set_defaults(func=cmd_grade_quiz_bma)

    regenerate_mark_schemes = subparsers.add_parser(
        "regenerate-mark-scheme-pngs",
        help="Regenerate mark-scheme PNG crops from an existing question_bank.json by question ID or sample filter.",
    )
    regenerate_mark_schemes.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    regenerate_mark_schemes.add_argument("--output", default="output", help="Output root for regenerated PNGs.")
    regenerate_mark_schemes.add_argument("--config", default="config.yaml", help="Optional config.yaml path.")
    regenerate_mark_schemes.add_argument(
        "--question-ids",
        default="",
        help="Comma-separated question IDs or canonical mark-scheme artifact stems to regenerate.",
    )
    regenerate_mark_schemes.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Question ID or canonical mark-scheme artifact stem to regenerate. May be repeated.",
    )
    regenerate_mark_schemes.add_argument("--all", action="store_true", help="Regenerate every mark-scheme PNG in the question bank.")
    regenerate_mark_schemes.add_argument("--year-min", type=int, default=None, help="Optional minimum canonical year filter.")
    regenerate_mark_schemes.add_argument("--year-max", type=int, default=None, help="Optional maximum canonical year filter.")
    regenerate_mark_schemes.add_argument("--limit", type=int, default=0, help="Optional maximum number of selected records.")
    regenerate_mark_schemes.set_defaults(func=cmd_regenerate_mark_scheme_pngs)

    regenerate_questions = subparsers.add_parser(
        "regenerate-question-pngs",
        help="Regenerate question PNG crops from an existing question_bank.json by question ID or sample filter.",
    )
    regenerate_questions.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    regenerate_questions.add_argument("--output", default="output", help="Output root for regenerated PNGs.")
    regenerate_questions.add_argument("--config", default="config.yaml", help="Optional config.yaml path.")
    regenerate_questions.add_argument(
        "--question-ids",
        default="",
        help="Comma-separated question IDs or canonical question artifact stems to regenerate.",
    )
    regenerate_questions.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Question ID or canonical question artifact stem to regenerate. May be repeated.",
    )
    regenerate_questions.add_argument("--all", action="store_true", help="Regenerate every question PNG in the question bank.")
    regenerate_questions.add_argument("--year-min", type=int, default=None, help="Optional minimum canonical year filter.")
    regenerate_questions.add_argument("--year-max", type=int, default=None, help="Optional maximum canonical year filter.")
    regenerate_questions.add_argument("--limit", type=int, default=0, help="Optional maximum number of selected records.")
    regenerate_questions.set_defaults(func=cmd_regenerate_question_pngs)

    audit = subparsers.add_parser(
        "audit",
        help="Audit visual-first question text trust and curation readiness in an exported question bank JSON.",
    )
    audit.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    audit.add_argument("--output", default="", help="Optional path to write the audit JSON report.")
    audit.set_defaults(func=cmd_audit)

    output_integrity = subparsers.add_parser(
        "output-integrity-audit",
        help="Fail-fast integrity audit for the current generated question-bank output and image artifacts.",
    )
    output_integrity.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    output_integrity.add_argument(
        "--artifact-root",
        default="",
        help="Root used to resolve relative image artifact paths. Defaults to the input run manifest or output root.",
    )
    output_integrity.add_argument("--output", default="", help="Optional path to write the audit JSON report.")
    output_integrity.add_argument("--example-limit", type=int, default=10, help="Maximum examples per failure group.")
    output_integrity.set_defaults(func=cmd_output_integrity_audit)

    asterion = subparsers.add_parser(
        "asterion-export",
        help="Write the Asterion static-site catalog and reviewed/safe student question-bank projection.",
    )
    asterion.add_argument("--input", default="output/json/question_bank.json", help="Path to schema v2 question_bank.json.")
    asterion.add_argument(
        "--output",
        default="",
        help=(
            f"Student-runtime output path. Defaults to output-root/asterion/exports/latest/{ASTERION_EXPORT_FILENAME}. "
            f"The all-course catalog is written beside it as {ASTERION_CATALOG_FILENAME}."
        ),
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
    asterion.add_argument(
        "--topic-routing",
        default="",
        help="Optional topic-routing sidecar used to attach canonical topic IDs and student-runtime topic gates.",
    )
    asterion.add_argument(
        "--allow-unusable-ai-sidecar",
        action="store_true",
        help="Allow using a failed or mixed AI-assisted sidecar as an explicitly documented fallback.",
    )
    asterion.set_defaults(func=cmd_asterion_export)

    content_lab = subparsers.add_parser(
        "asterion-content-lab-candidates",
        help="Write Asterion Content Lab candidate metadata without generating student-facing content.",
    )
    content_lab.add_argument(
        "--input",
        default="output/json/question_bank.json",
        help="Path to schema v2 question_bank.json, asterion_exam_bank_catalog_v1.json, or a legacy Asterion question-bank projection.",
    )
    content_lab.add_argument(
        "--output",
        default="",
        help=f"Output path. Defaults to output-root/asterion/exports/latest/{CONTENT_LAB_EXPORT_FILENAME}.",
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
    content_lab.add_argument(
        "--reviewed-source-skills",
        default="data/review/p3_exact_skill_reviewed_decisions.v1.json",
        help="Reviewed exact-skill decision artifact used for Content Lab source-skill and mapping gating.",
    )
    content_lab.add_argument(
        "--reviewed-mark-events",
        default="data/review/p3_exact_skill_reviewed_mark_events.v1.json",
        help="Reviewed mark-event decision artifact used for Content Lab generation gating.",
    )
    content_lab.add_argument(
        "--mark-events",
        default="",
        help="Optional canonical mark-events sidecar used to map Content Lab subpart events to reviewed event IDs.",
    )
    content_lab.add_argument(
        "--topic-routing",
        default="",
        help="Optional topic-routing sidecar used when building the intermediate Asterion catalog.",
    )
    content_lab.add_argument(
        "--allow-unusable-ai-sidecar",
        action="store_true",
        help="Allow using a failed or mixed AI-assisted sidecar as an explicitly documented fallback.",
    )
    content_lab.set_defaults(func=cmd_asterion_content_lab_candidates)

    enrich_ai = subparsers.add_parser(
        "enrich-ai",
        help="Run DeepSeek AI-assisted enrichment against canonical topic/subtopic/skill IDs.",
    )
    deepseek_enrich.add_ai_assisted_cli_arguments(enrich_ai)
    enrich_ai.set_defaults(func=cmd_enrich_ai)

    topic_route_ai = subparsers.add_parser(
        "topic-route-ai",
        help="Run strict DeepSeek topic routing against canonical parent topic IDs only.",
    )
    topic_routing.add_topic_routing_cli_arguments(topic_route_ai)
    topic_route_ai.set_defaults(func=cmd_topic_route_ai)

    topic_confidence_rescore = subparsers.add_parser(
        "topic-confidence-rescore",
        help="Write deterministic topic-confidence rescoring reports with typed extraction flags.",
    )
    topic_confidence_rescoring.add_topic_confidence_rescoring_cli_arguments(topic_confidence_rescore)
    topic_confidence_rescore.set_defaults(func=cmd_topic_confidence_rescore)

    topic_packet = subparsers.add_parser(
        "topic-packets",
        help="Generate image-first CAIE 9709 syllabus topic packets from canonical crops.",
    )
    topic_packets.add_topic_packet_cli_arguments(topic_packet)
    topic_packet.set_defaults(func=cmd_topic_packets)

    ai_sidecar_audit = subparsers.add_parser(
        "ai-sidecar-audit",
        help="Audit an AI-assisted enrichment sidecar for freshness, failures, and Asterion usability.",
    )
    ai_sidecar_audit.add_argument("--input", required=True, help="Path to question_bank.ai_assisted sidecar JSON.")
    ai_sidecar_audit.set_defaults(func=cmd_ai_sidecar_audit)

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

    suspicious_crop_pack = subparsers.add_parser(
        "suspicious-crop-review-pack",
        help="Create a deterministic review gallery from output-integrity suspicious rendered-crop candidates.",
    )
    suspicious_crop_pack.add_argument("--input", default="output/json/question_bank.json", help="Path to question_bank.json.")
    suspicious_crop_pack.add_argument(
        "--artifact-root",
        default="",
        help="Root used to resolve relative image artifact paths. Defaults to the input output root.",
    )
    suspicious_crop_pack.add_argument(
        "--output-root",
        default="",
        help="Review-pack output root. Defaults to a triage/ folder beside the exported JSON output tree.",
    )
    suspicious_crop_pack.add_argument("--iteration", default="", help="Optional iteration folder name or path.")
    suspicious_crop_pack.add_argument("--sample-size", type=int, default=30, help="Maximum number of ranked candidates.")
    suspicious_crop_pack.set_defaults(func=cmd_suspicious_crop_review_pack)

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

    inventory = subparsers.add_parser(
        "output-inventory",
        help="Inventory generated output roots without modifying files.",
    )
    inventory.add_argument("--root", action="append", default=[], help="Generated output root to inspect. May be repeated.")
    inventory.add_argument("--write", default="", help="Optional Markdown report path.")
    inventory.add_argument("--json", default="", help="Optional JSON report path.")
    inventory.add_argument("--include-size", action="store_true", help="Include recursive byte sizes for reported paths.")
    inventory.add_argument("--max-depth", type=int, default=6, help="Maximum scan depth below each root.")
    inventory.set_defaults(func=cmd_output_inventory)

    cleanup = subparsers.add_parser(
        "output-cleanup-plan",
        help="Create a dry-run cleanup plan for generated output roots.",
    )
    cleanup.add_argument("--root", action="append", default=[], help="Generated output root to inspect. May be repeated.")
    cleanup.add_argument("--write", default="", help="Optional Markdown cleanup plan path.")
    cleanup.add_argument("--json", default="", help="Optional JSON cleanup plan path.")
    cleanup.add_argument("--include-size", action="store_true", help="Include recursive byte sizes for planned paths.")
    cleanup.add_argument("--max-depth", type=int, default=6, help="Maximum scan depth below each root.")
    cleanup.set_defaults(func=cmd_output_cleanup_plan)

    normalize = subparsers.add_parser(
        "output-normalize-structure",
        help="Normalize generated output image folders and filenames to canonical subject names.",
    )
    normalize.add_argument("--root", default="output", help="Generated output root to normalize.")
    normalize.add_argument("--dry-run", action="store_true", help="Only report planned changes.")
    normalize.add_argument("--validate-only", action="store_true", help="Only validate the current output structure.")
    normalize.set_defaults(func=cmd_output_normalize_structure)

    summary_diff = subparsers.add_parser(
        "export-summary-diff",
        help="Print a concise before/after summary diff for comparable generated exports.",
    )
    summary_diff.add_argument("before", help="Earlier export or sidecar JSON path.")
    summary_diff.add_argument("after", help="Later comparable export or sidecar JSON path.")
    summary_diff.set_defaults(func=cmd_export_summary_diff)

    email_check = subparsers.add_parser(
        "email-check",
        help="Check a configured email provider connection without sending student messages.",
    )
    email_check.add_argument("--provider", default="", help="Email provider name, e.g. outlook-cn.")
    email_check.add_argument("--from", dest="from_address", default="", help="Requested sender/account address for providers that support it.")
    email_check.add_argument("--allow-partial", action="store_true", help="Exit successfully if only SMTP or IMAP succeeds.")
    email_check.set_defaults(func=cmd_email_check)

    email_send_test = subparsers.add_parser(
        "email-send-test",
        help="Send exactly one controlled smoke-test email.",
    )
    email_send_test.add_argument("--provider", default="", help="Email provider name, e.g. outlook-cn.")
    email_send_test.add_argument("--from", dest="from_address", default="", help="Requested sender/account address.")
    email_send_test.add_argument("--to", required=True, help="Single smoke-test recipient.")
    email_send_test.add_argument("--subject", required=True, help='Subject. Must contain "Smoke Test" by default.')
    email_send_test.add_argument("--body", required=True, help="Plain-text smoke-test body.")
    email_send_test.add_argument("--allow-non-smoke-subject", action="store_true")
    email_send_test.add_argument("--dry-run", action="store_true")
    email_send_test.add_argument("--attachment", dest="attachments", type=Path, action="append", default=[])
    email_send_test.add_argument("--cc", default="")
    email_send_test.add_argument("--bcc", default="")
    email_send_test.add_argument("--assignment-id", default="")
    email_send_test.add_argument("--roster-file", type=Path, default=None)
    email_send_test.add_argument("--score-file", type=Path, default=None)
    email_send_test.add_argument("--allow-default-mail-account", action="store_true")
    email_send_test.set_defaults(func=cmd_email_send_test)

    email_receive_test = subparsers.add_parser(
        "email-receive-test",
        help="Search provider mailbox for a controlled smoke-test email.",
    )
    email_receive_test.add_argument("--provider", default="", help="Email provider name, e.g. outlook-cn.")
    email_receive_test.add_argument("--query", required=True, help="Safe subject/token query.")
    email_receive_test.add_argument("--limit", type=int, default=5)
    email_receive_test.add_argument("--assignment-id", default="")
    email_receive_test.add_argument("--roster-file", type=Path, default=None)
    email_receive_test.add_argument("--score-file", type=Path, default=None)
    email_receive_test.set_defaults(func=cmd_email_receive_test)

    email_smoke_test = subparsers.add_parser(
        "email-smoke-test",
        help="Run the full controlled email smoke test and write local reports.",
    )
    email_smoke_test.add_argument("--provider", default="", help="Email provider name, e.g. outlook-cn.")
    email_smoke_test.add_argument("--from", dest="from_address", default="", help="Requested sender/account address.")
    email_smoke_test.add_argument("--to", required=True, help="Single smoke-test recipient.")
    email_smoke_test.add_argument("--open-report", action="store_true")
    email_smoke_test.add_argument("--attachment", dest="attachments", type=Path, action="append", default=[])
    email_smoke_test.add_argument("--assignment-id", default="")
    email_smoke_test.add_argument("--roster-file", type=Path, default=None)
    email_smoke_test.add_argument("--score-file", type=Path, default=None)
    email_smoke_test.add_argument("--allow-default-mail-account", action="store_true")
    email_smoke_test.set_defaults(func=cmd_email_smoke_test)

    classroom = subparsers.add_parser(
        "classroom",
        help="Run the local browser classroom dashboard.",
    )
    classroom.add_argument("--host", default="127.0.0.1")
    classroom.add_argument("--port", type=int, default=8765)
    classroom.add_argument("--open", action="store_true")
    classroom.add_argument("--classes-root", type=Path, default=Path("data/classes"))
    classroom.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    classroom.add_argument("--reports-root", type=Path, default=Path("reports/submissions"))
    classroom.set_defaults(func=cmd_classroom_dashboard)

    class_init = subparsers.add_parser(
        "class-init",
        help="Create a local class workspace with roster.csv and assignments/.",
    )
    class_init.add_argument("--class-id", required=True)
    class_init.add_argument("--roster", type=Path, default=None, help="Optional existing roster CSV to normalize into the class workspace.")
    class_init.add_argument("--classes-root", type=Path, default=Path("data/classes"))
    class_init.set_defaults(func=cmd_class_init)

    class_add_assignment = subparsers.add_parser(
        "class-add-assignment",
        help="Add an assignment PDF to a class workspace and build the message/reminder schedule.",
    )
    class_add_assignment.add_argument("--class-id", required=True)
    class_add_assignment.add_argument("--pdf", type=Path, required=True)
    class_add_assignment.add_argument("--assignment-id", required=True)
    class_add_assignment.add_argument("--title", default="")
    class_add_assignment.add_argument("--due-at", default="", help="ISO datetime. If omitted, prompts interactively.")
    class_add_assignment.add_argument("--send-at", default="", help="ISO datetime. If omitted, prompts interactively.")
    class_add_assignment.add_argument("--classes-root", type=Path, default=Path("data/classes"))
    class_add_assignment.set_defaults(func=cmd_class_add_assignment)

    class_dispatch_due = subparsers.add_parser(
        "class-dispatch-due",
        help="Dispatch scheduled assignment/reminder messages due at or before --now.",
    )
    class_dispatch_due.add_argument("--class-id", required=True)
    class_dispatch_due.add_argument("--assignment-id", required=True)
    class_dispatch_due.add_argument("--now", default="")
    class_dispatch_due.add_argument("--from", dest="from_address", default="")
    class_dispatch_due.add_argument("--send-live", action="store_true", help="Actually send due messages through Mail.app. Omit for dry-run.")
    class_dispatch_due.add_argument("--classes-root", type=Path, default=Path("data/classes"))
    class_dispatch_due.set_defaults(func=cmd_class_dispatch_due)

    class_ingest_submissions = subparsers.add_parser(
        "class-ingest-submissions",
        help="Ingest PDFs from a class assignment inbox, update roster status, and optionally send acknowledgements.",
    )
    class_ingest_submissions.add_argument("--class-id", required=True)
    class_ingest_submissions.add_argument("--assignment-id", required=True)
    class_ingest_submissions.add_argument("--classes-root", type=Path, default=Path("data/classes"))
    class_ingest_submissions.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    class_ingest_submissions.add_argument("--reports-root", type=Path, default=Path("reports/submissions"))
    class_ingest_submissions.add_argument("--send-acknowledgements", action="store_true")
    class_ingest_submissions.add_argument("--from", dest="from_address", default="")
    class_ingest_submissions.add_argument("--from-mailapp", action="store_true", help="Import matching PDF attachments from Mail.app before ingesting.")
    class_ingest_submissions.add_argument("--mail-query", default="", help="Mail.app subject query. Defaults to assignment id.")
    class_ingest_submissions.add_argument("--mail-limit", type=int, default=50)
    class_ingest_submissions.set_defaults(func=cmd_class_ingest_submissions)
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


def cmd_quiz_packet(args: argparse.Namespace) -> int:
    from exam_bank.submissions.quiz_packet import run_quiz_packet

    result = run_quiz_packet(
        quiz_dir=args.quiz_dir,
        course_id=args.course_id,
        assignment_pdf=args.assignment_pdf,
        assignment_id=args.assignment_id or None,
        class_id=args.class_id,
        timezone_name=args.timezone or None,
        output_root=args.output_root,
        reports_root=args.reports_root,
        open_report=bool(args.open_report),
        no_grade=bool(args.no_grade),
        force=bool(args.force),
    )
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
        if key not in {"warnings"}
    }
    printable["warnings"] = result["warnings"]
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


def cmd_grade_quiz_bma(args: argparse.Namespace) -> int:
    from exam_bank.submissions.bma_grading import build_bma_grading_matrix

    result = build_bma_grading_matrix(
        assignment_id=args.assignment_id,
        submission_output_root=args.submission_output_root,
        reports_root=args.reports_root,
        mode=args.mode,
        allow_suspicious_full_marks=bool(args.allow_suspicious_full_marks),
        open_report=bool(args.open_report),
    )
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
        if key not in {"summary"}
    }
    printable["summary"] = result["summary"]
    print(json.dumps(printable, indent=2, sort_keys=True))
    return int(result["exit_code"])


def cmd_email_check(args: argparse.Namespace) -> int:
    from exam_bank.emailing.providers import build_email_provider

    try:
        provider = build_email_provider(args.provider or None)
        _configure_email_provider_from_args(provider, args)
        status = provider.check_connection()
    except ValueError as exc:
        print(f"Email check failed: {exc}")
        return 2
    print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    if status.smtp_ok and status.imap_ok and status.error_code is None:
        return 0
    if args.allow_partial and (status.smtp_ok or status.imap_ok):
        return 0
    return 1


def cmd_email_send_test(args: argparse.Namespace) -> int:
    from exam_bank.emailing.providers import build_email_provider
    from exam_bank.emailing.smoke import DEFAULT_REPORTS_ROOT, EmailSmokeSafetyError, validate_smoke_send_request, write_audit_event

    try:
        validate_smoke_send_request(
            to=args.to,
            subject=args.subject,
            attachments=list(args.attachments or []),
            allow_non_smoke_subject=bool(args.allow_non_smoke_subject),
            assignment_id=args.assignment_id or None,
            roster_file=args.roster_file,
            score_file=args.score_file,
            cc=args.cc or None,
            bcc=args.bcc or None,
        )
        provider = build_email_provider(args.provider or None)
        _configure_email_provider_from_args(provider, args)
        result = provider.send_message(
            to=args.to,
            subject=args.subject,
            body_text=args.body,
            from_address=args.from_address or None,
            attachments=None,
            dry_run=bool(args.dry_run),
        )
    except EmailSmokeSafetyError as exc:
        print(f"Email send blocked: {exc}")
        return 2
    except ValueError as exc:
        print(f"Email send failed: {exc}")
        return 2
    write_audit_event(
        event_type="email_send_test",
        provider=result.provider,
        subject=result.subject,
        recipient=result.to,
        dry_run=result.dry_run,
        status="sent" if result.sent else "dry_run" if result.dry_run and result.error_code is None else "failed",
        error_code=result.error_code,
        provider_message_id=result.provider_message_id,
        reports_root=DEFAULT_REPORTS_ROOT,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.error_code is None and (result.sent or result.dry_run) else 1


def cmd_email_receive_test(args: argparse.Namespace) -> int:
    from exam_bank.emailing.providers import build_email_provider
    from exam_bank.emailing.smoke import DEFAULT_REPORTS_ROOT, EmailSmokeSafetyError, safe_summary_dict, validate_smoke_send_request, write_audit_event

    try:
        validate_smoke_send_request(
            to="smoke@example.invalid",
            subject="Smoke Test receive safety check",
            assignment_id=args.assignment_id or None,
            roster_file=args.roster_file,
            score_file=args.score_file,
        )
        provider = build_email_provider(args.provider or None)
        messages = provider.search_messages(query=args.query, limit=args.limit)
    except EmailSmokeSafetyError as exc:
        print(f"Email receive blocked: {exc}")
        return 2
    except RuntimeError as exc:
        code = str(exc) or "imap_login_failed"
        write_audit_event(
            event_type="email_receive_test",
            provider=args.provider or "",
            subject=args.query,
            status="failed",
            error_code=code,
            reports_root=DEFAULT_REPORTS_ROOT,
        )
        print(json.dumps({"provider": args.provider or "", "ok": False, "error_code": code}, indent=2, sort_keys=True))
        return 1
    except ValueError as exc:
        print(f"Email receive failed: {exc}")
        return 2
    write_audit_event(
        event_type="email_receive_test",
        provider=provider.provider_name,
        subject=args.query,
        status="found" if messages else "not_found",
        reports_root=DEFAULT_REPORTS_ROOT,
    )
    print(json.dumps({"provider": provider.provider_name, "matches": [safe_summary_dict(message) for message in messages]}, indent=2, sort_keys=True))
    return 0


def cmd_email_smoke_test(args: argparse.Namespace) -> int:
    from exam_bank.emailing.providers import build_email_provider
    from exam_bank.emailing.smoke import EmailSmokeSafetyError, run_email_smoke_test, validate_smoke_send_request

    try:
        validate_smoke_send_request(
            to=args.to,
            subject="Exam Bank Email Smoke Test",
            attachments=list(args.attachments or []),
            assignment_id=args.assignment_id or None,
            roster_file=args.roster_file,
            score_file=args.score_file,
        )
        provider = build_email_provider(args.provider or None)
        _configure_email_provider_from_args(provider, args)
        report = run_email_smoke_test(provider=provider, to=args.to, from_address=args.from_address or None, open_report=bool(args.open_report))
    except EmailSmokeSafetyError as exc:
        print(f"Email smoke test blocked: {exc}")
        return 2
    except ValueError as exc:
        print(f"Email smoke test failed: {exc}")
        return 2
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print(f"Smoke report: {report.report_path}")
    return 0 if report.smtp_ok and report.imap_ok and report.send_ok and report.receive_ok else 1


def cmd_classroom_dashboard(args: argparse.Namespace) -> int:
    from exam_bank.classroom_dashboard import ClassroomDashboardConfig, serve_dashboard

    config = ClassroomDashboardConfig(
        classes_root=args.classes_root,
        output_root=args.submission_output_root,
        reports_root=args.reports_root,
    )
    serve_dashboard(host=args.host, port=args.port, open_browser=bool(args.open), config=config)
    return 0


def cmd_class_init(args: argparse.Namespace) -> int:
    from exam_bank.classroom import init_class_workspace

    result = init_class_workspace(class_id=args.class_id, roster_source=args.roster, classes_root=args.classes_root)
    print(json.dumps(_printable_paths(result), indent=2, sort_keys=True))
    return 0


def cmd_class_add_assignment(args: argparse.Namespace) -> int:
    from exam_bank.classroom import add_assignment, parse_cli_datetime

    due_at = parse_cli_datetime(args.due_at or input("Due date/time (ISO, e.g. 2026-06-30T17:00:00+08:00): ").strip())
    send_at = parse_cli_datetime(args.send_at or input("Send date/time (ISO, e.g. 2026-06-24T09:00:00+08:00): ").strip())
    title = args.title or args.pdf.stem.replace("_", " ").strip()
    result = add_assignment(
        class_id=args.class_id,
        pdf_path=args.pdf,
        assignment_id=args.assignment_id,
        title=title,
        due_at=due_at,
        send_at=send_at,
        classes_root=args.classes_root,
    )
    print(json.dumps(_printable_paths(result), indent=2, sort_keys=True))
    return 0


def cmd_class_dispatch_due(args: argparse.Namespace) -> int:
    from exam_bank.classroom import dispatch_due_messages, parse_cli_datetime

    now = parse_cli_datetime(args.now) if args.now else None
    result = dispatch_due_messages(
        class_id=args.class_id,
        assignment_id=args.assignment_id,
        now=now,
        send_live=bool(args.send_live),
        from_address=args.from_address or None,
        classes_root=args.classes_root,
    )
    print(json.dumps(_printable_paths(result), indent=2, sort_keys=True))
    return 0 if int(result["failed"]) == 0 else 1


def cmd_class_ingest_submissions(args: argparse.Namespace) -> int:
    from exam_bank.classroom import ingest_class_assignment

    result = ingest_class_assignment(
        class_id=args.class_id,
        assignment_id=args.assignment_id,
        classes_root=args.classes_root,
        output_root=args.submission_output_root,
        reports_root=args.reports_root,
        send_acknowledgements=bool(args.send_acknowledgements),
        from_address=args.from_address or None,
        import_from_mailapp=bool(args.from_mailapp),
        mail_query=args.mail_query or None,
        mail_limit=args.mail_limit,
    )
    print(json.dumps(_printable_paths(result), indent=2, sort_keys=True))
    return 0


def _printable_paths(payload: dict[str, object]) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _configure_email_provider_from_args(provider: object, args: argparse.Namespace) -> None:
    from_address = getattr(args, "from_address", "") or None
    if from_address is not None and hasattr(provider, "requested_from_address"):
        setattr(provider, "requested_from_address", from_address)
    if hasattr(provider, "allow_default_mail_account"):
        setattr(provider, "allow_default_mail_account", bool(getattr(args, "allow_default_mail_account", False)))


def cmd_regenerate_mark_scheme_pngs(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _configure_runtime_paths(config, Path("."), Path(args.output))
    question_ids = list(getattr(args, "question_id", []) or [])
    if getattr(args, "question_ids", ""):
        question_ids.append(args.question_ids)
    report = regenerate_mark_scheme_pngs_from_question_bank(
        question_bank_path=args.input,
        config=config,
        question_ids=question_ids,
        all_records=bool(getattr(args, "all", False)),
        year_min=getattr(args, "year_min", None),
        year_max=getattr(args, "year_max", None),
        limit=int(getattr(args, "limit", 0) or 0) or None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["failed_count"] == 0 and report["skipped_count"] == 0 and not report["missing_requested_ids"] else 1


def cmd_regenerate_question_pngs(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _configure_runtime_paths(config, Path("."), Path(args.output))
    question_ids = list(getattr(args, "question_id", []) or [])
    if getattr(args, "question_ids", ""):
        question_ids.append(args.question_ids)
    report = regenerate_question_pngs_from_question_bank(
        question_bank_path=args.input,
        config=config,
        question_ids=question_ids,
        all_records=bool(getattr(args, "all", False)),
        year_min=getattr(args, "year_min", None),
        year_max=getattr(args, "year_max", None),
        limit=int(getattr(args, "limit", 0) or 0) or None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["failed_count"] == 0 and report["skipped_count"] == 0 and not report["missing_requested_ids"] else 1


def cmd_audit(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    report = write_audit(args.input, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_output_integrity_audit(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    report = write_current_output_integrity_audit(
        args.input,
        output,
        artifact_root=artifact_root,
        example_limit=args.example_limit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def cmd_asterion_export(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    skill_map_path = Path(args.skill_map) if args.skill_map else None
    topic_routing_path = Path(args.topic_routing) if args.topic_routing else None
    path = export_asterion_question_bank(
        args.input,
        output,
        artifact_root=artifact_root,
        skill_map_path=skill_map_path,
        topic_routing_path=topic_routing_path,
        allow_unusable_ai_sidecar=bool(getattr(args, "allow_unusable_ai_sidecar", False)),
    )
    print(
        json.dumps(
            {"output": str(path), "catalog_output": str(path.parent / ASTERION_CATALOG_FILENAME)},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_asterion_content_lab_candidates(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    skill_map_path = Path(args.skill_map) if args.skill_map else None
    reviewed_source_skills_path = Path(args.reviewed_source_skills) if args.reviewed_source_skills else None
    reviewed_mark_events_path = Path(args.reviewed_mark_events) if args.reviewed_mark_events else None
    mark_events_path = Path(args.mark_events) if args.mark_events else None
    topic_routing_path = Path(args.topic_routing) if args.topic_routing else None
    path = export_asterion_content_lab_candidates(
        args.input,
        output,
        artifact_root=artifact_root,
        skill_map_path=skill_map_path,
        reviewed_source_skills_path=reviewed_source_skills_path,
        reviewed_mark_events_path=reviewed_mark_events_path,
        mark_events_path=mark_events_path,
        topic_routing_path=topic_routing_path,
        allow_unusable_ai_sidecar=bool(getattr(args, "allow_unusable_ai_sidecar", False)),
    )
    print(json.dumps({"output": str(path)}, indent=2, ensure_ascii=False))
    return 0


def cmd_enrich_ai(args: argparse.Namespace) -> int:
    deepseek_enrich.finalize_ai_assisted_args(args)
    return deepseek_enrich.run_ai_assisted_from_args(args)


def cmd_topic_route_ai(args: argparse.Namespace) -> int:
    topic_routing.finalize_topic_routing_args(args)
    return topic_routing.run_topic_routing_from_args(args)


def cmd_topic_confidence_rescore(args: argparse.Namespace) -> int:
    return topic_confidence_rescoring.run_topic_confidence_rescoring_from_args(args)


def cmd_topic_packets(args: argparse.Namespace) -> int:
    report = topic_packets.run_topic_packets_from_args(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_ai_sidecar_audit(args: argparse.Namespace) -> int:
    report = deepseek_enrich.audit_ai_assisted_sidecar(args.input)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_ingress_pastpapers_co(args: argparse.Namespace) -> int:
    from .ingress import pastpapers_co

    argv = [
        "--url",
        pastpapers_co.BASE_URL,
        "--output",
        str(args.output),
        "--min-year",
        str(args.min_year),
        "--max-year",
        str(args.max_year),
    ]
    if args.input:
        argv.extend(["--input", str(args.input)])
    return pastpapers_co.main(argv)


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


def cmd_suspicious_crop_review_pack(args: argparse.Namespace) -> int:
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    review_root = Path(args.output_root) if args.output_root else None
    iteration = Path(args.iteration) if args.iteration else None
    summary = create_suspicious_crop_review_pack(
        args.input,
        artifact_root=artifact_root,
        review_root=review_root,
        iteration=iteration,
        sample_size=args.sample_size,
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


def cmd_output_inventory(args: argparse.Namespace) -> int:
    roots = args.root or ["output"]
    report = build_output_inventory(roots, include_size=args.include_size, max_depth=args.max_depth)
    write_inventory_outputs(
        report,
        json_path=Path(args.json) if args.json else None,
        markdown_path=Path(args.write) if args.write else None,
    )
    if args.write:
        print(f"Wrote output inventory to {args.write}")
    else:
        print(render_inventory_markdown(report), end="")
    return 0


def cmd_output_cleanup_plan(args: argparse.Namespace) -> int:
    roots = args.root or ["output"]
    plan = build_cleanup_plan(roots, include_size=args.include_size, max_depth=args.max_depth)
    write_cleanup_plan_outputs(
        plan,
        json_path=Path(args.json) if args.json else None,
        markdown_path=Path(args.write) if args.write else None,
    )
    if args.write:
        print(f"Wrote output cleanup plan to {args.write}")
    else:
        print(render_cleanup_plan_markdown(plan), end="")
    return 0


def cmd_output_normalize_structure(args: argparse.Namespace) -> int:
    if args.validate_only:
        report = validate_normalized_output(args.root)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1
    report = normalize_output_structure(args.root, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_export_summary_diff(args: argparse.Namespace) -> int:
    try:
        report = compare_export_summaries(args.before, args.after)
    except ExportSummaryDiffError as exc:
        print(f"Invalid export summary comparison: {exc}")
        return 2
    print(render_export_summary_diff(report), end="")
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
