# Topic Routing Improvement Closeout - 2026-06-11

## Executive Summary

PRs 3 through 9 moved topic routing from a stale, failure-prone production sidecar toward a hash-fresh, per-record-validatable routing pipeline that can safely use existing text evidence for visual-required records. The PR 9 provider-backed 80-record sample completed with 80 successful records, 0 failures, 1 review-required record, 79 strict-filter candidates, 0 provider failures, and 0 evidence-used repairs.

The improvement chain is ready for a later full sidecar refresh to `/tmp`, followed by audit and export validation. It is not yet a recommendation to overwrite `output/json/question_bank.topic_routing.v1.json`; production replacement should happen only after a full `/tmp` refresh passes the route audit and downstream export gates.

## Baseline Problem

- Stale sidecar rows: production rows lacked `evidence_packet_hash`, so resume freshness could not distinguish current packet inputs from stale routes.
- Batch-amplified failures: one malformed or invalid model row could fail an entire provider batch.
- Unsupported `evidence_used`: model outputs sometimes cited evidence fields that were not supplied in the packet.
- Visual-required packet evidence withheld: visual-required records had OCR/search-hint evidence in the question bank, but packets often supplied only mark-scheme text or no useful question-side text.

## Changes Across PR 3-9

- PR 3: added `evidence_packet_hash` and hash-aware resume freshness.
- PR 4: added per-record batch salvage so valid siblings survive a bad model row.
- PR 5: added deterministic `evidence_used` repair without loosening topic validation.
- PR 6: added deterministic sample selection and delta reporting.
- PR 6.1: fixed missing-output delta semantics so absent refresh outputs do not imply improvement.
- PR 7: added review-required sample triage.
- PR 8: added visual-required evidence audit.
- PR 9: changed packet construction so visual-required records without trusted question text can include existing OCR/search-hint fallback evidence, with source metadata distinguishing fallback from trusted text.

## Sample Results

| Metric | PR 6 sample | PR 9 sample | Change |
| --- | ---: | ---: | ---: |
| Attempted records | 80 | 80 | 0 |
| Successful records | 80 | 80 | 0 |
| Failed records | 0 | 0 | 0 |
| Review-required records | 10 | 1 | -9 |
| Strict-filter candidates | 70 | 79 | +9 |
| Provider failures | 0 | 0 | 0 |
| Evidence-used repairs | 11 | 0 | -11 |

The PR 9 delta tool run against the current production sidecar for the same 80 IDs reports `46 -> 0` failed records, `63 -> 1` review-required records, `17 -> 79` strict-filter candidates, and `80 -> 0` missing evidence packet hashes. That production comparison is larger than the PR 6 provider sample comparison because the production sidecar predates the PR 3-9 repair chain.

## Remaining PR 9 Review-Required Case

- Question ID: `11summer21_q04`
- Inferred bucket: `taxonomy_or_topic_fit_unclear`
- Review reason: `visual_required_without_sufficient_text_evidence`
- Available evidence fields: `question_text`, `ocr_text`, `mark_scheme_text`
- Course/component: `p1`, Pure Mathematics 1
- Text status/crop confidence: `review`, low question crop confidence
- Current route: no primary topic, confidence `low`, review-required `true`

The triage output indicates this is no longer a broad missing-packet-evidence problem. The remaining case is a low-confidence visual/text case that still needs human or taxonomy review before it can be treated as strict-filter truth.

## Visual Evidence Audit

| Metric | Before PR 9 | After PR 9 |
| --- | ---: | ---: |
| Evidence exists but withheld | 1000 | 38 |
| Packet only mark-scheme text | 191 | 0 |
| OCR/search-hint safe candidate | 960 | resolved into supplied fallback evidence |
| OCR fallback supplied | n/a | 1000 |
| Search-hint fallback supplied | n/a | 962 |
| Remaining safe-text-withheld count | n/a | 38 |

The PR 9 closeout visual audit reports 1000 visual-required records, 1000 with OCR text available, 0 with trusted question text, and 1000 with mark-scheme text available. Remaining candidate categories are:

- `crop confidence is low and likely needs recrop/re-extraction`: 714
- `no immediate visual evidence fix indicated`: 238
- `packet can include existing safe text currently withheld`: 38
- `genuinely visual/math-diagram dependent and should remain review-required`: 8
- `taxonomy/topic ambiguity, not evidence quality`: 2

## Full Refresh Readiness Checklist

- Write the full refreshed sidecar only to `/tmp` first. Do not overwrite `output/json/question_bank.topic_routing.v1.json` until the full `/tmp` run passes audit.
- Use hash-aware resume and keep `--no-progress` status output under a dedicated `/tmp` status directory.
- Expected runtime: the PR 9 80-record sample completed in 253.9 seconds, about 3.17 seconds per record. A 1301-record refresh scales to about 69 minutes at that rate; reserve 75-90 minutes for provider variance.
- Required route validation after full refresh:
  - run topic-routing sidecar audit against the `/tmp` full sidecar
  - verify failed count is 0 or explain each failure
  - verify missing `evidence_packet_hash` count is 0
  - inspect review-required bucket counts
  - inspect evidence-used repair counts and dropped fields
  - verify strict-filter candidate count and course metadata completeness
- Required Asterion export validation after full refresh:
  - regenerate export artifacts only if route audit passes
  - run the Asterion catalog/runtime validation already used by the project
  - confirm review/error/stale routes stay blocked from strict routing
  - confirm runtime promotion is not performed until export gates pass
- No cleanup, quarantine, taxonomy edits, prompt edits, OCR/extraction edits, reviewed-decision edits, or auto-grade eligibility changes are part of the full-refresh step.

## Recommended Next Step

Run a full topic-routing refresh to `/tmp` later, then audit the generated sidecar. If the route audit passes, regenerate Asterion export artifacts in a separate step and validate those outputs before any runtime promotion. The remaining 38 safe-text-withheld records can be handled in a follow-up PR after the full-refresh audit confirms whether they materially affect review-required outcomes.

## Generated Evidence

- PR 9 delta JSON: `/tmp/topic_routing_sample_refresh_delta_pr9.json`
- PR 9 delta markdown: `/tmp/topic_routing_sample_refresh_delta_pr9.md`
- PR 9 triage JSON: `/tmp/topic_routing_sample_refresh_triage_pr9.json`
- PR 9 triage markdown: `/tmp/topic_routing_sample_refresh_triage_pr9.md`
- PR 9 visual closeout JSON: `/tmp/topic_routing_visual_evidence_audit_pr9_closeout.json`
- PR 9 visual closeout markdown: `/tmp/topic_routing_visual_evidence_audit_pr9_closeout.md`
