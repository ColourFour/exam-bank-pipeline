# P3 Exact-Skill Current-State Audit

This is a review-diagnostic audit. It is not reviewed evidence and must not be used as runtime authority.

## Verdict

- `READY_FOR_ASTERION_CONTENT_LAB_REVIEW_DIAGNOSTICS`
- `NOT_READY_FOR_ASTERION_RUNTIME_MASTERY`
- `NOT_READY_FOR_GUARDIAN`
- `NOT_READY_FOR_CANDIDATE_GENERATION`
- `NOT_READY_FOR_SOURCE_BACKED_WORKED_EXAMPLES`

## Queue Counts

- Total queue items: 749
- `conflict_candidate`: 126
- `cross_topic_candidate`: 568
- `fallback_only`: 34
- `split_needed_candidate`: 21

## Cross-Topic Counts

- `conflict_needs_review`: 126
- `cross_topic_reviewable`: 602
- `cross_topic_split_needed`: 21

## Decomposition Counts

- `already_part_scoped`: 513
- `conflict_needs_review`: 126
- `insufficient_part_signal`: 81
- `needs_manual_split`: 21
- `not_decomposable`: 8

## Reviewed Registry Counts

- `blocked`: 1
- `review_needed`: 1
- `thin`: 1

## Safety Counts

- Already-reviewed queue scopes: 3
- Advisory-only mark-event count: 749
- Missing question asset count: 0
- Missing mark-scheme asset count: 0
- No candidate P3 skill count: 0
- Forbidden runtime sidecar exists: `false`

## Comparison To Previous Known State

- `total_queue_items`: 749 -> 749 (delta +0)
- `cross_topic_candidate`: 568 -> 568 (delta +0)
- `split_needed_candidate`: 21 -> 21 (delta +0)
- `conflict_candidate`: 126 -> 126 (delta +0)
- `fallback_only`: 34 -> 34 (delta +0)
- `ambiguous_candidate`: 0 -> 0 (delta +0)
- `clean_candidate`: 0 -> 0 (delta +0)

## Asterion Recommendation

- Can connect now: `true`
- Allowed capacity: Content Lab/admin/reviewer diagnostics only

Safe to consume now:

- review status summaries
- candidate status
- cross-topic status
- split-needed/conflict/fallback flags
- proposed part-level decomposition
- recommended review action
- canonical asset refs for reviewer inspection
- blocker diagnostics

Must not consume as authority:

- suggested_source_skill_ids
- candidate skills as mastery evidence
- advisory mark events as source-backed example evidence
- OCR/native/advisory text labels
- browser review responses before registry validation
- cross-topic candidates as clean evidence
- decomposition candidates as reviewed part boundaries
