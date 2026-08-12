# Command reference

This file is generated from the lazy public command registry in `exam_bank.command`.

## `exam-bank extract`

- `exam-bank extract run` — Run the canonical extraction pipeline.
- `exam-bank extract build-sample` — Build the canonical regression sample.
- `exam-bank extract audit-sample` — Audit the canonical regression sample.
- `exam-bank extract audit` — Audit extraction trust and readiness.
- `exam-bank extract integrity` — Run the canonical output integrity gate.
- `exam-bank extract regenerate-questions` — Regenerate canonical question images.
- `exam-bank extract regenerate-mark-schemes` — Regenerate canonical mark-scheme images.
- `exam-bank extract repair-mark-schemes` — Repair only missing or cross-question mark-scheme records.
- `exam-bank extract recover-partial` — Recover partial question blocks.

## `exam-bank data`

- `exam-bank data hydrate` — Restore checksum-verified source documents.
- `exam-bank data verify` — Verify source documents against the corpus manifest.
- `exam-bank data manifest` — Build a source corpus manifest.
- `exam-bank data normalize-corpus-sessions` — Normalize source session filenames from publisher evidence.
- `exam-bank data migrate-session-identity` — Migrate legacy March question identities using source evidence.
- `exam-bank data rebind-text-gold` — Rebind verified text gold to current canonical image hashes.
- `exam-bank data validate-review-assets` — Validate review decisions against current canonical image hashes.
- `exam-bank data quarantine-invalid` — Recoverably quarantine structurally invalid source PDFs.
- `exam-bank data inventory` — Inventory generated output.
- `exam-bank data cleanup-plan` — Build a non-destructive output cleanup plan.
- `exam-bank data audit-storage` — Audit exact output duplicates and optionally quarantine safe candidates.
- `exam-bank data build-asset-manifest` — Build the canonical image asset manifest.
- `exam-bank data validate-assets` — Validate asset references across canonical and downstream artifacts.
- `exam-bank data export-questions` — Export canonical questions through the versioned interchange contract.
- `exam-bank data validate-questions` — Validate a versioned Question interchange export.
- `exam-bank data normalize` — Normalize generated output layout.
- `exam-bank data diff` — Compare export summaries.

## `exam-bank topic`

- `exam-bank topic route` — Run strict AI topic routing.
- `exam-bank topic refresh-routing` — Refresh deterministic topic-routing artifacts.
- `exam-bank topic release-manifest` — Bind the question bank and durable topic routing into a release manifest.
- `exam-bank topic verify-release` — Verify the hash-bound topic-routing release.
- `exam-bank topic restore-release` — Restore the local topic-routing cache from the verified release.
- `exam-bank topic rescore` — Rescore topic confidence deterministically.
- `exam-bank topic review-batch` — Build a topic review batch.
- `exam-bank topic review-run` — Run topic reviews.
- `exam-bank topic review-import` — Import topic review decisions.
- `exam-bank topic review-merge` — Merge topic review decisions.
- `exam-bank topic visual-audit` — Run image-backed topic auditing.
- `exam-bank topic difficulty` — Review topic-packet difficulty.
- `exam-bank topic difficulty-index` — Build the deterministic difficulty-index sidecar.
- `exam-bank topic packet-visual-audit` — Audit rendered topic packet pages.
- `exam-bank topic packets` — Generate image-first topic packets.

## `exam-bank asterion`

- `exam-bank asterion export` — Build Asterion catalog and runtime exports.
- `exam-bank asterion content-lab` — Build Asterion Content Lab candidates.
- `exam-bank asterion package` — Package a checksum-verified Asterion release.
- `exam-bank asterion verify` — Verify an Asterion release package.

## `exam-bank release`

- `exam-bank release build` — Build the hash-bound multi-artifact release manifest.
- `exam-bank release verify` — Verify the release manifest and every bound artifact.

## `exam-bank ai`

- `exam-bank ai enrich` — Run advisory AI enrichment.
- `exam-bank ai audit` — Audit an AI sidecar.

## `exam-bank triage`

- `exam-bank triage sample` — Build a deterministic triage sample.
- `exam-bank triage crop-pack` — Build a suspicious-crop review pack.
- `exam-bank triage serve` — Serve a triage iteration.
- `exam-bank triage compare` — Compare a triage iteration.
- `exam-bank triage status` — Report auto-triage status.
- `exam-bank triage plan` — Build an auto-triage plan.
- `exam-bank triage iteration-compare` — Compare auto-triage iterations.
- `exam-bank triage runbook` — Build an auto-triage runbook.

## `exam-bank review`

- `exam-bank review promote` — Promote a provenance-stamped canonical review artifact.

## `exam-bank marks`

- `exam-bank marks build` — Build the deterministic mark-event sidecar.
- `exam-bank marks validate` — Validate the mark-event sidecar.

## `exam-bank advisory`

- `exam-bank advisory inventory` — Inventory advisory PDFs.
- `exam-bank advisory extract` — Extract advisory PDF text.
- `exam-bank advisory parse-examiner` — Parse examiner reports.
- `exam-bank advisory parse-thresholds` — Parse grade thresholds.
- `exam-bank advisory link` — Link advisory evidence to questions.
- `exam-bank advisory topic-evidence` — Build advisory topic evidence.
- `exam-bank advisory examiner-difficulty` — Build examiner difficulty hints.
- `exam-bank advisory threshold-context` — Build grade-threshold context.
- `exam-bank advisory validate` — Validate advisory evidence.
- `exam-bank advisory reports` — Build advisory review reports.
- `exam-bank advisory sidecar` — Build the final advisory sidecar.
