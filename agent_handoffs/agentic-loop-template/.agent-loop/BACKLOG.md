# Improvement Backlog

Agent 1 may select from this backlog, but it may also propose a better item when current repo evidence supports it. Prefer the first category that has actionable evidence.

## 1. Clear Output Images

- [ ] Reduce missing question-image or mark-scheme-image detection failures.
- [ ] Improve crop boundaries for historical layouts and 2024-2025 variants.
- [ ] Repair mark-scheme crop segmentation where the official solution block is missing, truncated, or paired to the wrong question.
- [ ] Fix canonical asset-path normalization so accepted records point to `pm1`, `pm3`, `stats`, or `mechanics` flat paths.
- [ ] Add or tighten tests around rendered crop dimensions, page-span selection, artifact existence, and asset-manifest references.
- [ ] Use visual triage samples before assuming metadata describes an image failure correctly.

## 2. Correct Topic Content

- [ ] Repair topic parsing or routing where records use invalid taxonomy IDs, stale labels, or review-required sidecar entries.
- [ ] Keep topic labels sidecar-only until `safe_for_strict_filters=true` is justified by validation.
- [ ] Improve topic packet filtering so invalid topics, missing images, mapping failures, and validation failures are excluded unless explicitly review-only.
- [ ] Add focused tests for taxonomy lookup, reviewed-decision precedence, sidecar checksum/restore behavior, and topic packet manifests.

## 3. Correct Per-Question JSON Data

- [ ] Fix `question_id`, `paper`, `paper_family`, `question_number`, and `PaperIdentity` mismatches.
- [ ] Fix wrong or missing `question_image_path`, `mark_scheme_image_path`, canonical artifact IDs, and alternate path arrays.
- [ ] Repair mark totals, subpart structures, `question_solution_marks`, and `subparts_solution_marks` when the official mark scheme supports the correction.
- [ ] Tighten `notes.validation_status`, `notes.mapping_status`, visual flags, text trust fields, and provenance fields so records fail closed when evidence is weak.
- [ ] Add focused tests that load representative JSON records and assert contract-level field consistency.

## 4. Validation And Triage Support

- [ ] Improve auto-triage handoffs when they directly reduce image, topic, or JSON correctness failures.
- [ ] Improve audit output only when it makes the next extraction-quality decision more reliable.
- [ ] Add regression fixtures from reviewed triage examples instead of broad synthetic fixtures.

## Do Not Select

- Large rewrites.
- Framework swaps.
- New dependencies.
- Cosmetic churn.
- Broad generated-output cleanup unless it is blocking image or JSON correctness.
- Trust-gate loosening, flag suppression, or status downgrades without image-backed evidence.
- Replacing canonical image evidence with OCR/native/AI text.
- Student-runtime or Asterion promotion changes unless the plan is explicitly about role-gated export behavior.
