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
        "normalize-corpus-sessions": CommandSpec(
            "Normalize source session filenames from publisher evidence.",
            module="exam_bank.corpus_session_identity",
            function="main",
        ),
        "migrate-session-identity": CommandSpec(
            "Migrate legacy March question identities using source evidence.",
            module="exam_bank.canonical_session_migration",
            function="main",
        ),
        "rebind-text-gold": CommandSpec(
            "Rebind verified text gold to current canonical image hashes.",
            module="exam_bank.question_text_gold",
            function="main",
        ),
        "validate-review-assets": CommandSpec(
            "Validate review decisions against current canonical image hashes.",
            module="exam_bank.review_asset_binding",
            function="main",
        ),
        "quarantine-invalid": CommandSpec(
            "Recoverably quarantine structurally invalid source PDFs.",
            module="exam_bank.commands.data",
            function="run_quarantine_invalid",
        ),
        "inventory": CommandSpec("Inventory generated output.", ("output-inventory",)),
        "cleanup-plan": CommandSpec("Build a non-destructive output cleanup plan.", ("output-cleanup-plan",)),
        "audit-storage": CommandSpec(
            "Audit exact output duplicates and optionally quarantine safe candidates.",
            module="exam_bank.storage_audit",
            function="main",
        ),
        "build-asset-manifest": CommandSpec(
            "Build the canonical image asset manifest.",
            module="exam_bank.asset_manifest",
            function="run_build",
        ),
        "validate-assets": CommandSpec(
            "Validate asset references across canonical and downstream artifacts.",
            module="exam_bank.asset_manifest",
            function="run_validate",
        ),
        "export-questions": CommandSpec(
            "Export canonical questions through the versioned interchange contract.",
            module="exam_bank.question_interchange",
            function="run_export",
        ),
        "validate-questions": CommandSpec(
            "Validate a versioned Question interchange export.",
            module="exam_bank.question_interchange",
            function="run_validate",
        ),
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
        "release-manifest": CommandSpec(
            "Bind the question bank and durable topic routing into a release manifest.",
            module="exam_bank.topic_routing_artifact",
            function="manifest_main",
        ),
        "verify-release": CommandSpec(
            "Verify the hash-bound topic-routing release.",
            module="exam_bank.topic_routing_artifact",
            function="verify_main",
        ),
        "restore-release": CommandSpec(
            "Restore the local topic-routing cache from the verified release.",
            module="exam_bank.topic_routing_artifact",
            function="restore_main",
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
    "release": {
        "build": CommandSpec(
            "Build the hash-bound multi-artifact release manifest.",
            module="exam_bank.release_manifest",
            function="run_build",
        ),
        "verify": CommandSpec(
            "Verify the release manifest and every bound artifact.",
            module="exam_bank.release_manifest",
            function="run_verify",
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
    "marks": {
        "build": CommandSpec("Build the deterministic mark-event sidecar.", module="exam_bank.mark_events.cli", function="run_build"),
        "validate": CommandSpec("Validate the mark-event sidecar.", module="exam_bank.mark_events.cli", function="run_validate"),
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
    print("CAIE 9709 exam-bank extraction and normalization pipeline.\n")
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
