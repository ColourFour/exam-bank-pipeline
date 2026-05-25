# Asterion Export Contract

This contract covers the current Asterion-facing exports:

- `output/asterion/exports/latest/asterion_question_bank_v1.json`
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`

These files are role-gated projections from the image-first question bank. Asterion must not treat either export as a globally student-facing-safe corpus. A record is usable only for the specific role whose gate allows that use.

Strict topic filters, when backed by `output/json/question_bank.topic_routing.v1.json`, are governed by [Topic Routing Sidecar Contract](TOPIC_ROUTING_SIDECAR_CONTRACT.md). Asterion must require `metadata.run_summary.safe_for_strict_filters=true` before using that sidecar for strict topic filtering.

## Export Purposes

`asterion_question_bank_v1.json` is the main Asterion projection. It carries stable identifiers, provenance, canonical image artifact references, asset IDs, image integrity metadata, quality gate summaries, subpart records, advisory text snippets, machine mark-event candidates, and `usage_roles`. Its purpose is to give Asterion one conservative handoff file while preserving blocked and review states.

`asterion_content_lab_candidates_v1.json` is a metadata-only candidate projection derived from the Asterion question-bank projection. It is for Content Lab review, planning, and future generation workflows. It does not contain generated student-facing content. Its `policy`, `role_statuses`, `generation_gate`, and `review_status` fields are part of the permission contract.

## Canonical And Advisory Fields

Canonical fields for Asterion consumption:

- `schema_name`, `schema_version`, `source_schema`, and `record_count` define the file contract and source export relationship.
- `question_id`, `paper`, `paper_family`, and `question_number` are stable identity and routing fields.
- `canonical_question_artifact`, `canonical_mark_scheme_artifact`, `canonical_question_asset_id`, `canonical_mark_scheme_asset_id`, `artifact_integrity`, and `source_pdf` are the canonical provenance and artifact fields.
- `usage_roles` is the canonical role permission surface for `asterion_question_bank_v1.json`.
- `quality_gate` booleans are canonical gate inputs and summaries. `quality_gate.reason_codes` are diagnostics, not a replacement for `usage_roles`.
- In the Content Lab export, `policy`, `role_statuses`, `generation_gate`, and top-level `review_status` are canonical permission fields.

Advisory fields:

- `subparts[].question_text.text`, `subparts[].mark_scheme_text.text`, text trust levels, and `detected_mark_values` are advisory extraction metadata.
- `subparts[].marks` is useful when present, but subpart mark promotion is incomplete. Do not assume subpart marks are complete just because the full-question `total_marks` exists.
- `mark_events` are machine candidates unless their own `review_status` or approval metadata explicitly says they have been reviewed or approved.
- `candidate_selection`, `possible_content_lab_roles`, `source_skill_ids`, `source_mark_event_count`, and `generated_warmup_pattern_source` are planning metadata. They do not override `role_statuses` or `generation_gate`.

The canonical rendered images remain the source of truth for student-visible question and mark-scheme content. Native text and OCR text are advisory unless a specific role gate permits their use. OCR/native text must not replace the canonical question or mark-scheme images for math-heavy display. Consumers may resolve `*_asset_id` values through `output/json/asset_manifest.v1.json`; path fields remain for compatibility and should still point at canonical relative paths.

## Role Gates

Consumers must honor role-specific `allow`, `block`, `block_until_reviewed`, `include`, and `exclude` decisions exactly. A status for one role does not imply permission for another role. Unknown or missing role statuses should be treated as denied.

`usage_roles.canonical_practice` controls student-facing canonical practice. It has `allow` or `block`. Only `allow` records may enter canonical practice. `block` records remain out of student-facing practice even if they have useful advisory text or metadata.

`usage_roles.field_guide_source` controls field-guide source use. `allow` may be consumed for that role. `block_until_reviewed` may be shown only in review or teacher-controlled workflows that preserve the review state. `block` must not be used.

`usage_roles.quick_check_source` controls quick-check source use. This role is stricter about text use: a record can be blocked until reviewed when text-only display is not allowed even if the image artifacts are otherwise useful. Use quick-check text only when this role is `allow` and the relevant text field also reports `text_only_display_allowed=true`.

`usage_roles.warmup_generator_source` controls question-bank-level eligibility for warmup generation source material. `allow` means the question-bank gate allows that role. Content Lab generation still also requires candidate-level `generation_gate.status=allow`, reviewed or approved mappings, reviewed or approved mark events, non-quarantined mark events, and source skill IDs.

`usage_roles.guardian_candidate` controls Guardian candidate use. It currently follows the same conservative allow/block boundary as canonical practice. `allow` records can be considered Guardian candidates; `block` records must not be promoted by advisory text, medium trust search metadata, or Content Lab candidate presence.

`usage_roles.p3_readiness_metric` is a metrics gate only. `include` means the record belongs in P3 readiness reporting. `exclude` means it should not count in that metric. It is not a student-facing permission.

Content Lab candidate `role_statuses` are separate from question-bank `usage_roles`. A candidate may be useful as a review object even when generation remains blocked. `generation_gate.blocked=true` or `generation_gate.status=blocked_until_reviewed` means no student-facing generated warmup content should be emitted from that candidate.

## Current Export State

Measured release evidence as of audit date `2026-05-14`, source run `20260513T070200Z-56d469c1dd52`:

That dated `asterion_question_bank_v1.json` has `1301` records:

- Canonical practice: `252 allow`, `1049 block`.
- Field guide source: `252 allow`, `1021 block_until_reviewed`, `28 block`.
- Quick-check source: `51 allow`, `1222 block_until_reviewed`, `28 block`.
- Warmup generator source: `228 allow`, `1045 block_until_reviewed`, `28 block`.
- Guardian candidate: `252 allow`, `1049 block`.
- P3 readiness metric: `396 include`, `905 exclude`.

That dated `asterion_content_lab_candidates_v1.json` has `2416` candidates:

- Candidate `review_status`: `502 machine_candidate`, `1914 blocked_until_reviewed`.
- Candidate `generation_gate.status`: all current candidates are `blocked_until_reviewed`.
- The dominant generation blocker is missing source skill IDs, followed by unreviewed mark events and unreviewed mappings or subparts.

These counts are dated release evidence, not eligibility rules. Regenerated exports may change counts, but the role-gate semantics above must remain conservative unless the contract is intentionally revised.

## Known Limitations

Student-facing readiness is limited. The current export is useful for review and controlled downstream workflows, but only role-allowed subsets are eligible for student-facing practice, quick checks, Guardian candidate flows, or generation inputs.

Subpart marks are incomplete. In the dated export evidence above, the Asterion projection has `968` records with labeled subparts, and `48` records have missing subpart marks. Full-question mark totals and rendered mark-scheme images are more reliable than subpart-level automated marking.

The source set is missing the mark scheme for `9709_2025_November_33`. This accounts for `11` records with missing mark-scheme image paths, including `33autumn25_q01` through `33autumn25_q11`. These records must remain blocked or review-only until the source companion mark scheme is added and the export is regenerated and validated.

Mark events and generated warmup pattern metadata are not reviewed content. They are machine candidates unless reviewed or approved status is present and the candidate generation gate allows use.

## Asterion Consumer Checklist

1. Verify `schema_name`, `schema_version`, `source_schema`, and `record_count` before loading.
2. Resolve image paths against the artifact root and require expected integrity metadata for any student-visible image display.
3. Gate every use by the exact role field for that workflow. Default deny unknown, missing, `block`, and inappropriate `block_until_reviewed` statuses.
4. Never use a role's `allow` status to imply another role's permission.
5. Treat canonical images as source of truth. Treat native/OCR text, mark-scheme text, detected mark values, and mark events as advisory unless the relevant role gate permits use.
6. Preserve blocked and review states in user interfaces, logs, and downstream queues.
7. For Content Lab, require both candidate `role_statuses` and `generation_gate.status=allow` before any generated content is considered. Do not emit student-facing generated content from `blocked_until_reviewed` candidates.
8. Keep `p3_readiness_metric` separate from product eligibility; it is a reporting inclusion flag, not a practice gate.
9. For strict topic filters, require the topic routing sidecar contract and `safe_for_strict_filters=true`; default to review-only behavior when the sidecar is missing, unsafe, or failed.
10. Recheck known limitations before release: limited student-facing eligibility, incomplete subpart marks, and the missing `9709_2025_November_33` mark scheme.
