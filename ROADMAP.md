# Roadmap

This roadmap reflects the current unified historical + modern pipeline architecture. Historical audit and handoff docs remain useful evidence, but this file is the active planning view.

The project remains image-first. Rendered question PNGs and rendered mark-scheme PNGs are the source of truth. Native PDF text, OCR text, normalized text, AI output, topic routing, advisory evidence, mark-event evidence, difficulty scores, readiness tiers, and Asterion projections are support metadata unless a documented consumer role gate explicitly permits use.

## Current Baseline

Current canonical export contract:

- Path: `output/json/question_bank.json`
- Schema: `exam_bank.question_bank` version 2
- Dataset coverage target: CAIE 9709 `2008-2025`
- Legacy source era: `2008-2020`
- Modern source era: `2021-2025`
- Input tree: `input/pastpapers/9709/<year>/`
- Canonical asset folders: `output/pm1/`, `output/pm3/`, `output/stats/`, `output/mechanics/`
- Run evidence: `output/run_status/<run_id>/run_manifest.json`

Run-specific record counts, QA rollups, OCR counts, mapping status, validation status, and missing-asset counts must come from the current export `run_manifest`, not from this roadmap. The dated audit baseline remains [Project Audit and Optimization Review](docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md). Treat it as the May 14 cleanup/audit baseline, not the live count source. Refresh current evidence with:

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

- Modern pipeline-only phase for `2021-2025` extraction. This is complete and superseded by the unified historical + modern pipeline.
- Image-first extraction to canonical `pm1`, `pm3`, `stats`, and `mechanics` image trees plus `question_bank.json`.
- Unified historical + modern source scanning through the recursive `process` command.
- Legacy ingestion support for `2008-2020` via the PastPapers.co ingress path.
- `PaperIdentity`-based canonical paper IDs, question IDs, mark-scheme pairing, and asset paths.
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

## Phase: Unified Historical Extraction

Status: implemented.

Completed:

- Legacy ingestion integrated for the `2008-2020` source era.
- Modern extraction retained for the `2021-2025` source era.
- Full pipeline execution enabled through one recursive extraction command over `input/pastpapers/9709`.
- Question crop and mark-scheme crop asset extraction unified under the canonical `pm1`, `pm3`, `stats`, and `mechanics` folder schema.
- `PaperIdentity` dependency wired into paper identity, mark-scheme matching, export normalization, and asset-path generation.

Next:

- Identity stabilization for edge cases in the `PaperIdentity` system, especially ambiguous legacy sessions, component aliases, and cross-source filename variants.
- Extraction reliability improvements for low-confidence crop, mapping, scope, and validation cases across the wider 2008-2025 corpus.
- Image extraction hardening for historical layouts, missing companion documents, mark-scheme crop boundaries, and canonical asset-manifest validation.

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

### 1. Identity Stabilization

Goal: harden `PaperIdentity` as the shared dependency for historical and modern extraction.

Work:

- Audit ambiguous legacy session labels, compact session codes, and filename variants.
- Verify component aliases and subject-family mapping for `pm1`, `pm3`, `stats`, and `mechanics`.
- Keep mark-scheme pairing, asset paths, question IDs, and export normalization on the same identity contract.
- Add focused fixtures for edge cases discovered in the 2008-2025 input tree.

Acceptance:

- Identity failures are explicit and reviewable.
- Question assets and mark-scheme assets derive from the same canonical identity.
- Legacy and modern files with equivalent metadata produce equivalent IDs.
- Focused identity and output-contract tests pass.

### 2. Extraction Reliability Improvements

Goal: reduce mapping, validation, scope, and visual-curation failures without loosening trust gates.

Work:

- Use `audit_question_bank_readiness.py` and `auto-triage-status` to identify dominant failure clusters.
- Use `auto-triage-plan` for bounded implementation handoffs.
- Prioritize records where crop confidence, mapping status, validation status, or scope quality contradict each other.
- Preserve OCR-enabled comparisons when claiming production improvement.

Acceptance:

- Hard-failure counts decrease in an OCR-enabled comparison.
- No broad trust-gate loosening.
- Worsened records are explained or fixed.
- Focused tests and full tests pass.

### 3. Image Extraction Hardening

Goal: make canonical question and mark-scheme crops more reliable across historical and modern layouts.

Work:

- Harden question crop boundaries for historical layouts and 2024-2025 format variants.
- Harden mark-scheme crop boundaries and missing-companion handling.
- Validate canonical asset manifests after unified runs.
- Keep image evidence authoritative over native/OCR/AI text.

Acceptance:

- Missing or mismatched image references remain zero in accepted exports.
- Low-confidence crop clusters are reduced or clearly queued for review.
- Asset-manifest validation passes.
- Normalization reports no active legacy output paths.

### 4. Release Validation Pass

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

### 5. Text-Fidelity Review Workflow

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

### 6. Topic Routing Recovery

Goal: make strict topic filters usable by reducing current topic-routing sidecar failures.

Work:

- Re-run or repair strict topic routing against canonical taxonomy IDs only.
- Keep provider outputs sidecar-only.
- Validate that failed records, review-required records, and `safe_for_strict_filters` metadata are accurate.

Acceptance:

- Schema-validation failures are reduced or explained.
- `safe_for_strict_filters=true` is only set when the sidecar actually qualifies.
- Downstream consumers can fail closed from sidecar metadata.

### 7. Output Hygiene

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
