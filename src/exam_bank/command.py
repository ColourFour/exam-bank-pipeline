from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandSpec:
    description: str
    legacy_argv: tuple[str, ...] = ()
    module: str = ""
    function: str = ""


COMMANDS: dict[str, dict[str, CommandSpec]] = {
    "extract": {
        "run": CommandSpec("Run the canonical extraction pipeline.", ("process",)),
        "build-sample": CommandSpec(
            "Build the canonical regression sample.", module="exam_bank.canonical_sample", function="main"
        ),
        "audit-sample": CommandSpec(
            "Audit the canonical regression sample.", module="exam_bank.canonical_sample_audit", function="main"
        ),
        "audit": CommandSpec("Audit extraction trust and readiness.", ("audit",)),
        "integrity": CommandSpec("Run the canonical output integrity gate.", ("output-integrity-audit",)),
        "regenerate-questions": CommandSpec("Regenerate canonical question images.", ("regenerate-question-pngs",)),
        "regenerate-mark-schemes": CommandSpec(
            "Regenerate canonical mark-scheme images.", ("regenerate-mark-scheme-pngs",)
        ),
        "repair-mark-schemes": CommandSpec(
            "Repair only missing or cross-question mark-scheme records.",
            module="exam_bank.mark_scheme_repair",
            function="run_repair",
        ),
        "recover-partial": CommandSpec(
            "Recover partial question blocks.",
            module="exam_bank.partial_question_block_recovery",
            function="main",
        ),
    },
    "data": {
        "hydrate": CommandSpec("Restore checksum-verified source documents.", module="exam_bank.commands.data", function="run_hydrate"),
        "verify": CommandSpec("Verify source documents against the corpus manifest.", module="exam_bank.commands.data", function="run_verify"),
        "manifest": CommandSpec("Build a source corpus manifest.", module="exam_bank.commands.data", function="run_manifest"),
        "inventory": CommandSpec("Inventory generated output.", ("output-inventory",)),
        "cleanup-plan": CommandSpec("Build a non-destructive output cleanup plan.", ("output-cleanup-plan",)),
        "normalize": CommandSpec("Normalize generated output layout.", ("output-normalize-structure",)),
        "diff": CommandSpec("Compare export summaries.", ("export-summary-diff",)),
    },
    "topic": {
        "route": CommandSpec("Run strict AI topic routing.", ("topic-route-ai",)),
        "refresh-routing": CommandSpec(
            "Refresh deterministic topic-routing artifacts.",
            module="exam_bank.topic_routing_refresh",
            function="main",
        ),
        "rescore": CommandSpec("Rescore topic confidence deterministically.", ("topic-confidence-rescore",)),
        "review-batch": CommandSpec("Build a topic review batch.", ("topic-review-batch",)),
        "review-run": CommandSpec("Run topic reviews.", ("topic-review-run",)),
        "review-import": CommandSpec("Import topic review decisions.", ("topic-review-import",)),
        "review-merge": CommandSpec("Merge topic review decisions.", ("topic-review-merge",)),
        "visual-audit": CommandSpec("Run image-backed topic auditing.", ("visual-topic-audit",)),
        "difficulty": CommandSpec("Review topic-packet difficulty.", ("topic-difficulty-review",)),
        "difficulty-index": CommandSpec(
            "Build the deterministic difficulty-index sidecar.",
            module="exam_bank.difficulty_index.cli",
            function="run_build",
        ),
        "packet-visual-audit": CommandSpec("Audit rendered topic packet pages.", ("topic-packet-visual-audit",)),
        "packets": CommandSpec("Generate image-first topic packets.", ("topic-packets",)),
    },
    "asterion": {
        "export": CommandSpec("Build Asterion catalog and runtime exports.", ("asterion-export",)),
        "content-lab": CommandSpec("Build Asterion Content Lab candidates.", ("asterion-content-lab-candidates",)),
        "package": CommandSpec(
            "Package a checksum-verified Asterion release.",
            module="exam_bank.asterion_release_bundle",
            function="main",
        ),
        "verify": CommandSpec(
            "Verify an Asterion release package.",
            module="exam_bank.asterion_release_bundle",
            function="verify_main",
        ),
    },
    "ai": {
        "enrich": CommandSpec("Run advisory AI enrichment.", ("enrich-ai",)),
        "audit": CommandSpec("Audit an AI sidecar.", ("ai-sidecar-audit",)),
    },
    "triage": {
        "sample": CommandSpec("Build a deterministic triage sample.", ("triage-sample",)),
        "crop-pack": CommandSpec("Build a suspicious-crop review pack.", ("suspicious-crop-review-pack",)),
        "serve": CommandSpec("Serve a triage iteration.", ("triage-serve",)),
        "compare": CommandSpec("Compare a triage iteration.", ("triage-compare",)),
        "status": CommandSpec("Report auto-triage status.", ("auto-triage-status",)),
        "plan": CommandSpec("Build an auto-triage plan.", ("auto-triage-plan",)),
        "iteration-compare": CommandSpec("Compare auto-triage iterations.", ("auto-triage-compare",)),
        "runbook": CommandSpec("Build an auto-triage runbook.", ("auto-triage-runbook",)),
    },
    "review": {
        "promote": CommandSpec(
            "Promote a provenance-stamped canonical review artifact.",
            module="exam_bank.review_promotion",
            function="run_promote",
        ),
    },
    "classroom": {
        "serve": CommandSpec("Serve the local classroom dashboard.", ("classroom",)),
        "init": CommandSpec("Create a local class workspace.", ("class-init",)),
        "add-assignment": CommandSpec("Add an assignment to a class.", ("class-add-assignment",)),
        "dispatch-due": CommandSpec("Dispatch due classroom messages.", ("class-dispatch-due",)),
        "ingest-submissions": CommandSpec("Ingest assignment submissions.", ("class-ingest-submissions",)),
        "quiz": CommandSpec("Run the local quiz-packet workflow.", ("quiz-packet",)),
        "grade-bma": CommandSpec("Build B/M/A grading artifacts.", ("grade-quiz-bma",)),
        "ingest-assignment": CommandSpec(
            "Ingest assignment submissions into the local workspace.",
            module="exam_bank.submissions.cli",
            function="main",
        ),
        "review-submissions": CommandSpec(
            "Build a human review queue for submissions.",
            module="exam_bank.submissions.review_cli",
            function="main",
        ),
        "draft-grades": CommandSpec(
            "Build draft grades from reviewed evidence.",
            module="exam_bank.submissions.draft_grading_cli",
            function="main",
        ),
    },
    "email": {
        "check": CommandSpec("Check the configured email provider.", ("email-check",)),
        "send-test": CommandSpec("Run a controlled send test.", ("email-send-test",)),
        "receive-test": CommandSpec("Run a controlled receive test.", ("email-receive-test",)),
        "smoke-test": CommandSpec("Run the email smoke workflow.", ("email-smoke-test",)),
        "ingest-submissions": CommandSpec(
            "Ingest submissions from a local email export.",
            module="exam_bank.submissions.email_intake_cli",
            function="main",
        ),
        "import-live": CommandSpec(
            "Import live mailbox submissions into the restricted local root.",
            module="exam_bank.submissions.live_email_import_cli",
            function="main",
        ),
        "build-outgoing": CommandSpec(
            "Build the outgoing classroom email queue.",
            module="exam_bank.submissions.outgoing_email_cli",
            function="build_queue_main",
        ),
        "dry-run-outgoing": CommandSpec(
            "Preview the outgoing email queue without sending.",
            module="exam_bank.submissions.outgoing_email_cli",
            function="dry_run_main",
        ),
        "fake-send-outgoing": CommandSpec(
            "Exercise outgoing delivery using the fake provider.",
            module="exam_bank.submissions.outgoing_email_cli",
            function="fake_send_main",
        ),
    },
    "marks": {
        "build": CommandSpec("Build the deterministic mark-event sidecar.", module="exam_bank.mark_events.cli", function="run_build"),
        "validate": CommandSpec("Validate the mark-event sidecar.", module="exam_bank.mark_events.cli", function="run_validate"),
    },
    "autograde": {
        "build": CommandSpec("Build fail-closed eligible items.", module="exam_bank.auto_grade.cli", function="run_build"),
        "validate": CommandSpec("Validate fail-closed eligible items.", module="exam_bank.auto_grade.cli", function="run_validate"),
    },
    "advisory": {
        "inventory": CommandSpec("Inventory advisory PDFs.", module="exam_bank.advisory_evidence.cli", function="run_inventory"),
        "extract": CommandSpec("Extract advisory PDF text.", module="exam_bank.advisory_evidence.cli", function="run_extract"),
        "parse-examiner": CommandSpec("Parse examiner reports.", module="exam_bank.advisory_evidence.cli", function="run_parse_examiner"),
        "parse-thresholds": CommandSpec("Parse grade thresholds.", module="exam_bank.advisory_evidence.cli", function="run_parse_thresholds"),
        "link": CommandSpec("Link advisory evidence to questions.", module="exam_bank.advisory_evidence.cli", function="run_link"),
        "topic-evidence": CommandSpec("Build advisory topic evidence.", module="exam_bank.advisory_evidence.cli", function="run_topic_evidence"),
        "examiner-difficulty": CommandSpec("Build examiner difficulty hints.", module="exam_bank.advisory_evidence.cli", function="run_examiner_difficulty"),
        "threshold-context": CommandSpec("Build grade-threshold context.", module="exam_bank.advisory_evidence.cli", function="run_threshold_context"),
        "validate": CommandSpec("Validate advisory evidence.", module="exam_bank.advisory_evidence.cli", function="run_validate"),
        "reports": CommandSpec("Build advisory review reports.", module="exam_bank.advisory_evidence.cli", function="run_reports"),
        "sidecar": CommandSpec("Build the final advisory sidecar.", module="exam_bank.advisory_evidence.cli", function="run_sidecar"),
    },
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_top_help()
        return 0
    domain = argv.pop(0)
    actions = COMMANDS.get(domain)
    if actions is None:
        _fail(f"Unknown command domain: {domain}")
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_domain_help(domain, actions)
        return 0
    action = argv.pop(0)
    spec = actions.get(action)
    if spec is None:
        _fail(f"Unknown {domain} command: {action}")
    return _run(spec, argv)


def _run(spec: CommandSpec, argv: list[str]) -> int:
    if spec.legacy_argv:
        legacy = importlib.import_module("exam_bank.cli")
        return int(legacy.main([*spec.legacy_argv, *argv]))
    module = importlib.import_module(spec.module)
    function: Callable[[list[str] | None], int] = getattr(module, spec.function)
    return int(function(argv))


def _print_top_help() -> None:
    print("usage: exam-bank <domain> <command> [options]\n")
    print("CAIE 9709 exam-bank extraction and classroom platform.\n")
    print("domains:")
    for domain in COMMANDS:
        print(f"  {domain}")
    print("\nRun 'exam-bank <domain> --help' for commands in a domain.")


def _print_domain_help(domain: str, actions: dict[str, CommandSpec]) -> None:
    print(f"usage: exam-bank {domain} <command> [options]\n")
    print("commands:")
    for name, spec in actions.items():
        print(f"  {name:24} {spec.description}")


def render_command_reference() -> str:
    lines = [
        "# Command reference",
        "",
        "This file is generated from the lazy public command registry in `exam_bank.command`.",
        "",
    ]
    for domain, actions in COMMANDS.items():
        lines.extend([f"## `exam-bank {domain}`", ""])
        for action, spec in actions.items():
            lines.append(f"- `exam-bank {domain} {action}` — {spec.description}")
        lines.append("")
    return "\n".join(lines)


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
