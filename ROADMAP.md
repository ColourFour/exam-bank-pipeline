# Roadmap

This roadmap reflects the current project state as of the local export generated on `2026-05-18`. Historical audit and handoff docs remain useful evidence, but this file is the active planning view.

The project remains image-first. Rendered question PNGs and rendered mark-scheme PNGs are the source of truth. Native PDF text, OCR text, normalized text, AI output, topic routing, advisory evidence, mark-event evidence, difficulty scores, readiness tiers, and Asterion projections are support metadata unless a documented consumer role gate explicitly permits use.

## Current Baseline

Current canonical export:

- Path: `output/json/question_bank.json`
- Schema: `exam_bank.question_bank` version 2
- Run: `20260518T235946Z-4e93c881aa77`
- Generated at: `2026-05-18T23:59:46.928802+00:00`
- Records: `1301`
- Paper families: `p1: 401`, `p3: 396`, `p4: 258`, `p5: 246`
- OCR: ran for all records with Tesseract `5.5.2`; selected over native text for `33` records
- Mapping status: `1291 pass`, `10 fail`
- Validation status: `917 pass`, `369 review`, `15 fail`
- Scope quality: `923 clean`, `378 review`
- Text fidelity: `1259 clean`, `42 degraded`
- Visual curation: `213 ready`, `1073 review`, `15 fail`
- Text-only status: `201 ready`, `1049 review`, `51 fail`
- Question crop confidence: `337 high`, `964 low`
- Mark-scheme crop confidence: `746 high`, `555 medium`
- Missing question image paths: `0`
- Missing mark-scheme image paths: `0`

The dated audit baseline remains [Project Audit and Optimization Review](docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md). Treat it as the May 14 cleanup/audit baseline, not the live count source. Refresh current evidence with:

```bash
.venv/bin/python -m exam_bank.cli audit \
  --input output/json/question_bank.json \
  --output output/json/audit.current.json

.venv/bin/python -m exam_bank.cli output-integrity-audit \
  --input output/json/question_bank.json \
  --artifact-root output \
  --output output/json/audit.current.integrity.json

.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --artifact-root output \
  --out-dir output/audits/current
```

## Completed Foundations

The following work is implemented and should be treated as current infrastructure, not future scope:

- Image-first extraction to paper-family image trees and `question_bank.json`.
- OCR-enabled production-style export with run manifest, QA rollups, artifact root, and output-layout metadata.
- Standard CLI commands for extraction, audit, output integrity, Asterion export, Content Lab candidates, topic packets, AI sidecars, triage, auto-triage, output inventory, cleanup planning, and export summary diffs.
- Generator safety for taxonomy scripts: `--help` is safe and `--dry-run` reports planned writes.
- Atomic JSON write helpers and tests.
- Output inventory and dry-run cleanup planning.
- Current release validation checklist.
- Conservative Asterion export contract and role-gated Asterion projection.
- Strict topic-routing sidecar contract and current sidecar at `output/json/question_bank.topic_routing.v1.json`.
- Deterministic mark-event sidecar and validation report.
- Deterministic advisory evidence sidecar from examiner reports and grade thresholds.
- Advisory difficulty index sidecar and reports.
- Image-first topic packet generation under `output/topic_packets/`.
- Auto-triage planning, runbook, comparison, and decision files.
- Text-extraction failure audit, bad-text fixture manifest, crop/context signal audit, OCR profile experiments, normalized text candidate contract, and text-fidelity review queue.

## Current Constraints

These constraints should drive near-term work:

- Canonical images remain authoritative; no text pipeline may silently replace image evidence.
- Current topic routing is not safe for strict filters until its metadata says `safe_for_strict_filters=true`.
- Difficulty index v1 is advisory and not approved for student-facing sequencing.
- Mark-event evidence is advisory and not approved for automated student marking.
- Advisory examiner-report and grade-threshold evidence may support review and difficulty context, but it must not overwrite canonical records.
- AI enrichment sidecars remain review/debug evidence unless separately audited and role-gated.
- Low question-crop confidence is still common and should be handled as review evidence, not hidden by text improvements.
- Current generated outputs are large and ignored by git; cleanup must remain manifest-driven and dry-run first.

## Active Priorities

### 1. Release Validation Pass

Goal: make the current export reproducibly releasable.

Work:

- Run the full validation checklist in [docs/RELEASE_VALIDATION_CHECKLIST.md](docs/RELEASE_VALIDATION_CHECKLIST.md).
- Record current test count, audit summaries, integrity audit, sidecar validation, topic sidecar safety metadata, and Asterion export status.
- Use `export-summary-diff` before promoting any regenerated JSON.
- Keep current outputs fixed unless a regeneration is intentional and audited.

Acceptance:

- Full tests pass.
- Integrity audit passes.
- Sidecar validations pass or only emit documented warnings.
- Release notes identify which downstream roles are allowed, blocked, or review-only.

### 2. Hard-Failure Reduction

Goal: reduce the remaining mapping, validation, and visual-curation failures without loosening trust gates.

Work:

- Use `audit_question_bank_readiness.py` and `auto-triage-status` to identify dominant failure clusters.
- Use `auto-triage-plan` for bounded implementation handoffs.
- Prioritize mapping failures, validation failures, visual curation failures, and records where status fields contradict each other.
- Preserve OCR-enabled comparisons when claiming production improvement.

Acceptance:

- Hard-failure counts decrease in an OCR-enabled comparison.
- No broad trust-gate loosening.
- Worsened records are explained or fixed.
- Focused tests and full tests pass.

### 3. Text-Fidelity Review Workflow

Goal: turn text-fidelity evidence into a practical review loop while keeping text advisory.

Work:

- Use the text-fidelity review queue to prioritize known bad and high-risk records.
- Keep normalized text and OCR-profile outputs as candidate/report layers.
- Add reviewed-state handling where it improves repeatability.
- Use crop/context warnings to prevent over-trusting selected text.

Acceptance:

- Known bad fixtures remain captured near the top of the queue.
- Review outcomes are persisted without changing canonical text truth.
- New candidate text fields include provenance and warnings.
- Asterion and student-facing projections do not consume advisory candidates without a contract update.

### 4. Topic Routing Recovery

Goal: make strict topic filters usable by reducing current topic-routing sidecar failures.

Work:

- Re-run or repair strict topic routing against canonical taxonomy IDs only.
- Keep provider outputs sidecar-only.
- Validate that failed records, review-required records, and `safe_for_strict_filters` metadata are accurate.

Acceptance:

- Schema-validation failures are reduced or explained.
- `safe_for_strict_filters=true` is only set when the sidecar actually qualifies.
- Downstream consumers can fail closed from sidecar metadata.

### 5. Output Hygiene

Goal: keep generated outputs understandable and safe to clean.

Work:

- Use `output-inventory` and `output-cleanup-plan` before touching generated roots.
- Preserve current canonical JSON, image trees, Asterion exports, topic packets, sidecars, and frozen triage baselines.
- Move or banner historical docs rather than mixing old counts with current instructions.

Acceptance:

- Cleanup plans remain dry-run and reviewed before action.
- Archive decisions are documented.
- No current export or image tree is deleted during doc or hygiene work.

## Deferred Work

These items are still valuable but should wait until the active priorities above are stable:

- Tiered Asterion slice files such as gold, multimodal, and master exports.
- Automated subpart mark promotion for nested marks.
- Production-quality canonical text candidate schema with raw native/OCR/vision candidates, normalized candidate text, provenance, and confidence.
- Layout-aware math text recovery promoted beyond report-only experiments.
- OCR profile routing promoted beyond report-only experiments.
- AI batch checkpoint/resume improvements beyond current status support.
- OCR/rendering cache-key refactors.
- Audit-script consolidation.
- Mark-scheme subpart parsing suitable for automated marking workflows.
- Topic/difficulty model refresh after crop, mapping, marks, and text reliability improve.

## Historical References

Use these documents for context, not as live count sources:

- [Project Audit and Optimization Review](docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md)
- [Phase 1 Cleanup Prerequisite Closeout](docs/PHASE_1_CLEANUP_PREREQ_CLOSEOUT.md)
- [Text Extraction Improvement Decision Packet](docs/text_extraction/TEXT_EXTRACTION_IMPROVEMENT_DECISION_PACKET.md)
- [Auto-Triage Workflow](docs/AUTO_TRIAGE.md)
- [Command Atlas](docs/COMMAND_ATLAS.md)
- [Release Validation Checklist](docs/RELEASE_VALIDATION_CHECKLIST.md)
