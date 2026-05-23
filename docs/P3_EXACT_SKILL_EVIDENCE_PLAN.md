# P3 Exact Skill Evidence Sidecar Plan

Status: Phase 1 and Phase 2 foundations are implemented. This document does not implement the exporter, create the final sidecar, change extraction behavior, or modify generated bank/export artifacts.

Planned future sidecar:

```text
output/asterion/exports/latest/p3_exact_skill_evidence_v1.json
```

`docs/asterion/` does not currently exist. This plan is placed in `docs/` beside `ASTERION_EXPORT_CONTRACT.md`, `TOPIC_ROUTING_SIDECAR_CONTRACT.md`, `MARK_EVENTS_CONTRACT.md`, and `TRUST_MODEL.md` because those root-level docs are the current contract location for Asterion-facing and sidecar-facing behavior.

## Phase 1 Reviewed-Decision Foundation

Phase 1 adds the reviewed-decision input contract that a future P3 exact-skill evidence sidecar can consume. The manual registry lives at:

```text
data/review/p3_exact_skill_reviewed_decisions.v1.json
```

This path is intentionally under `data/review/` because it is curated review input, not generated output. It is not the Asterion sidecar and does not create or update:

```text
output/asterion/exports/latest/p3_exact_skill_evidence_v1.json
```

The validator can be run with:

```text
.venv/bin/python scripts/validate_p3_exact_skill_reviewed_decisions.py
```

The reusable validation implementation is in `src/exam_bank/p3_exact_skill/reviewed_decisions.py`. It checks the registry schema, required record fields, allowed route statuses, allowed use-case keys, evidence ID uniqueness, duplicate question scopes, clean-record prerequisites, conservative runtime use-case permissions, blocker requirements for non-clean records, P3-only mastery evidence, missing asset refs for clean records, and advisory-only clean promotion.

The validator proves only that reviewed-decision records obey the local contract and fail closed. It does not prove that the mathematical review is correct, does not inspect image contents, does not promote advisory text or mark events, does not validate full coverage by P3 skill, and does not authorize Asterion runtime behavior. The seed fixture is deliberately conservative: it records plausible review inputs and blockers, but it does not assert any clean P3 exact-skill evidence.

The future exporter must consume this registry as an input source, preserve non-clean decisions and blockers, and independently write the final sidecar only after the export contract and sidecar validation are implemented.

## Phase 2 Review Queue Foundation

Phase 2 adds a reviewer-facing queue that aggregates current P3 candidate evidence without promoting any record to reviewed evidence. The generated report artifacts are:

```text
reports/p3_exact_skill_review_queue.v1.json
reports/p3_exact_skill_review_queue.md
```

These are report artifacts, not Asterion runtime exports. They do not create or update:

```text
output/asterion/exports/latest/p3_exact_skill_evidence_v1.json
```

The queue can be rebuilt with:

```text
.venv/bin/python scripts/build_p3_exact_skill_review_queue.py
```

The builder reads the canonical question bank, topic-routing sidecar, Asterion question-bank projection, Asterion Content Lab candidates, mark-event sidecar, P3 skill mappings, P3 topic assignments, and the Phase 1 reviewed-decision registry. It uses those inputs only as review context. Candidate skill mappings, topic routes, OCR/native text, Content Lab labels, and advisory mark events remain non-authoritative.

Queue statuses deliberately use candidate names such as `clean_candidate`, `ambiguous_candidate`, `blocked_candidate`, and `fallback_only`. A `clean_candidate` is only a promising human-review target; it is not clean reviewed evidence. The reviewed-decision registry remains the only source that can assert actual `clean` route status.

The Markdown report summarizes status counts, top blockers, priority review items, blocked/fallback groups, existing reviewed-decision reconciliation, and reviewer checklist guidance. The JSON report preserves the full queue item list for downstream review tooling.

## Purpose

Asterion Content Lab needs a reviewed, image-backed evidence surface for exact P3 skill examples. The sidecar should answer a narrow question: which canonical question or part has been safely reviewed as evidence for one exact P3 skill, with stable question and mark-scheme image refs, explicit route status, provenance, blockers, and allowed runtime use cases?

The sidecar must fail closed. Current OCR, native text, advisory evidence, topic routing, mark-event parsing, and generated skill mappings can help prepare review packets, but they are not curriculum authority and must not directly create clean records.

## Audit Snapshot

Current inspected source run:

- `output/json/question_bank.json`: schema `exam_bank.question_bank` v2, `1301` records, `396` P3 records.
- Question-bank run id: `20260518T235946Z-4e93c881aa77`.
- Question-bank generated at: `2026-05-18T23:59:46.928802+00:00`.
- Question-bank run manifest git commit: `0a8dbf360408f3e6990c090e24be0147de8e0575`.
- Current workspace `HEAD`: `0ac73426535d1cf6f38a48c069d797a8bb2ce79f`.
- `output/asterion/exports/latest/asterion_question_bank_v1.json`: `1301` Asterion records, `396` P3 records, all P3 records have `quality_gate.canonical_assets_ok=true`.
- P3 Asterion quality: `395` of `396` P3 records have both canonical assets and consistent marks according to current Asterion gates.
- P3 Asterion roles: `57` P3 records have `guardian_candidate=allow`; `54` have `warmup_generator_source=allow`, but Content Lab candidates still block generation until review.
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`: `2416` candidates, `702` P3 candidates, all current P3 candidate `generation_gate.status` values are `blocked_until_reviewed`.
- `output/json/question_bank.topic_routing.v1.json`: `1301` records, `safe_for_strict_filters=false`, `153` failed records, `221` review-required records. P3 has `327` high-confidence topic routes and `337` non-review-required topic routes, but this is parent-topic advisory evidence only.
- `output/json/question_bank.mark_events.v1.json`: `1301` records. P3 has `378` advisory-safe mark-event records, `18` not advisory-safe, and all `396` remain `safe_for_marking_use=false`. Validation currently reports `ok=true`, `0` errors, `0` warnings.
- `exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json`: `26` P3 skill IDs, but the map says `review_status=needs_review`.
- `exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json`: `749` P3 mapping candidates, all `review_status=needs_review`; `608` are subpart-level and `141` are whole-question-only.
- `output/auto_grade/reviewed_rubrics.v1.json`: `3` approved reviewed rubrics, all P1. This is useful as a review/validation pattern but does not provide P3 evidence yet.

## A. Current Artifacts That Can Support The Sidecar

`output/json/question_bank.json` is the canonical extraction input. It already provides stable question IDs, paper metadata, canonical question and mark-scheme image paths, source PDF metadata, page refs, question numbers, top-level marks, subpart labels, detected part structure, text trust, crop confidence, mapping status, validation status, visual flags, and review flags.

`output/asterion/exports/latest/asterion_question_bank_v1.json` already projects the question bank into a conservative Asterion shape. It adds artifact integrity hashes, source PDF hashes, quality-gate summaries, subpart records, advisory mark-event candidates, and role gates. Its image integrity logic is the closest existing behavior to what the future sidecar needs for stable asset refs.

`output/asterion/exports/latest/asterion_content_lab_candidates_v1.json` already expresses Content Lab role statuses, generation gates, source artifacts, source mark-event counts, source skill IDs when supplied, and block reasons. It is useful for blocker vocabulary and candidate prioritization, not for clean evidence promotion.

`output/json/question_bank.mark_events.v1.json` provides deterministic advisory mark-event IDs, part paths, parsed mark codes, total checks, mark-scheme image refs, extraction status, advisory safety, and review flags. It is a good source of `mark_event_refs`, but not scoring or curriculum authority.

`output/json/question_bank.topic_routing.v1.json` provides parent-topic route records keyed by `question_id`, with confidence, review-required status, evidence-used metadata, and provider provenance. It can support review queue prioritization and route diagnostics. It cannot provide exact skill authority.

`exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json` is the best current P3 skill ID namespace and contains names, descriptions, prerequisite IDs, Asterion region IDs, syllabus sections, recognizer signals, and Content Lab priority. Its own `needs_review` status means v1 evidence must require a reviewer decision before treating any skill link as clean.

`exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json` and `question_topic_assignments_9709_p3_v1.json` are useful review inputs. They contain subpart IDs, candidate primary/secondary/prerequisite skills, candidate topic/subtopic assignment, confidence, evidence snippets, and review statuses. They are not clean evidence because every skill mapping is still `needs_review`.

`output/auto_grade/*` and `reports/auto_grade/*` provide a reviewed-rubric workflow pattern: review queues, reviewer packets, approved registry, validation, reviewer identity, timestamps, total verification, and accepted evidence. The current approved registry has no P3 records, so it is process support only.

## B. Fields Already Available

The current repo can already supply these fields without changing extraction:

| Planned field | Current source | Notes |
| --- | --- | --- |
| `question_id` | `question_bank.questions[]`, Asterion export | Stable key. |
| `paper`, `paper_family`, `question_number` | Question bank and Asterion export | P3 filter is `paper_family == "p3"`. |
| `year`, `session`, `variant` | P3 taxonomy mappings and mark-event sidecar | Not consistently top-level in question bank; derive carefully from paper or mapping records. |
| `part_id`, `subpart_id`, `subpart_label`, `part_path` | Question bank subparts, Asterion subparts, mark-event `part_path` | Existing subpart marks are incomplete in some records. |
| Canonical question image path | Question bank `canonical_question_artifact`, `question_image_path`, Asterion `canonical_question_artifact` | Asterion export adds hash/existence metadata. |
| Canonical mark-scheme image path | Question bank `mark_scheme_image_path`, Asterion `canonical_mark_scheme_artifact`, mark-event `source_mark_scheme_image_path` | Must require the image to exist for clean records. |
| Source PDF refs and hashes | Question-bank notes, Asterion `source_pdf` | Asterion export hashes source PDFs. |
| Top-level marks and detected totals | Question bank, Asterion, mark-event sidecar | Part marks are weaker than full-question totals. |
| Crop/text/visual blockers | Question-bank notes and Asterion `quality_gate.reason_codes` | Useful for clean blockers and review queue priority. |
| Parent topic route status | Topic-routing sidecar | Advisory only while sidecar safety is false. |
| Candidate P3 skill IDs | P3 skill map and P3 question skill mappings | Candidate only until reviewed. |
| Mark-event IDs | Mark-events sidecar | Advisory unless separately reviewed. |
| Existing runtime role vocabulary | Asterion `usage_roles`, Content Lab `role_statuses`, `generation_gate` | Reuse semantics, but do not infer new permissions. |

## C. Missing Or Partial Fields

The critical missing field is a human-reviewed P3 exact-skill decision. The current P3 skill map and P3 question-skill mappings are machine-generated or mixed-evidence candidates with `needs_review` status. There are no approved P3 reviewed rubrics in the current auto-grade registry.

Other missing or partial fields:

- Stable `evidence_id` values for one exact skill evidence unit.
- Reviewer identity, reviewed timestamp, review status, review notes, and image verification status for P3 exact-skill evidence.
- Explicit `route_status` using the requested clean/thin/ambiguous/blocked/deferred/review-needed/fallback-only vocabulary.
- A reviewed `reviewed_region` object tying the exact skill to Asterion region metadata.
- A first-class `allowed_use_cases` contract for mastery, Guardian, export, source-backed examples, and candidate generation.
- A sidecar-specific validator that refuses clean promotion from advisory-only text, AI routing, or unreviewed mappings.
- Coverage/diversity summary by exact skill.
- Review queue/report that preserves ambiguous, blocked, deferred, and fallback-only cases instead of dropping them.

## D. Authoritative Source Decisions

Canonical question asset refs should come from the canonical question bank path fields and be verified with the same integrity behavior used by the Asterion export. A clean record should store path, sha256, existence, source PDF metadata, and page refs where available.

Canonical mark-scheme asset refs should come from the canonical question bank mark-scheme image fields and be verified against the artifact root. The mark-event sidecar can cross-check `source_mark_scheme_image_path`, but it should not override the canonical question-bank pairing.

P3 skill IDs should come from the P3 canonical skill map as an allowed ID dictionary, but the exact question-to-skill assertion must come from a new manually reviewed decision fixture or registry. The current generated P3 question-skill mappings are review inputs only.

Topic/region route status should be owned by the new reviewed exact-skill decision. Topic routing sidecar output may be stored as advisory provenance or blocker context, but it cannot produce `clean` because the sidecar is currently not safe for strict filters and it routes only parent topics.

Part/subpart data should start with question-bank `subparts`, Asterion subpart IDs, and mark-event `part_path`. A whole-question evidence unit can be clean only when the reviewer explicitly verifies that the whole question is exact-skill evidence, or when the item has no mixed subparts. Mixed whole-question evidence must remain ambiguous/review-needed until part-level evidence is reviewed.

Mark-event evidence should use `output/json/question_bank.mark_events.v1.json` for event IDs, mark-code candidates, part paths, and advisory safety. Clean evidence can reference advisory event IDs only as supporting evidence. If a record allows candidate generation, the relevant mark events should be reviewed or approved, not merely advisory-safe.

Crop/text/visual blockers should come from question-bank trust fields, Asterion `quality_gate.reason_codes`, mark-event review flags, topic-routing review flags, and Content Lab `generation_gate.block_reasons`.

## E. Proposed V1 Sidecar Contract

Top-level shape:

```json
{
  "schema_name": "asterion.p3_exact_skill_evidence",
  "schema_version": 1,
  "generated_at": "ISO-8601 timestamp",
  "source_bank_version": {
    "question_bank_path": "output/json/question_bank.json",
    "question_bank_sha256": "...",
    "question_bank_run_id": "...",
    "question_bank_git_commit": "...",
    "asterion_question_bank_path": "output/asterion/exports/latest/asterion_question_bank_v1.json",
    "reviewed_decisions_path": "...",
    "taxonomy_paths": []
  },
  "record_count": 0,
  "records": [],
  "skill_summary": [],
  "blocker_summary": {}
}
```

Record shape:

```json
{
  "evidence_id": "p3_exact_skill:v1:32spring21_q01:whole:9709_p3_3_2_log_exponential_equations",
  "question_id": "32spring21_q01",
  "paper": "32spring21",
  "paper_family": "p3",
  "year": 2021,
  "session": "March",
  "variant": "32",
  "question_number": "1",
  "part_path": [],
  "subpart_id": "32spring21_q01_whole",
  "subpart_label": "whole",
  "reviewed_source_skill_ids": ["9709_p3_3_2_log_exponential_equations"],
  "reviewed_region": {
    "component": "p3",
    "section": "3.2 Logarithmic and exponential functions",
    "skill_id": "9709_p3_3_2_log_exponential_equations",
    "asterion_region_id": "asterion:9709:p3:..."
  },
  "route_status": "clean",
  "route_reasons": [],
  "source_question_asset_refs": [],
  "source_mark_scheme_asset_refs": [],
  "mark_event_refs": [],
  "evidence_basis": {
    "basis_type": "canonical_images_reviewed",
    "advisory_inputs_used": [],
    "review_notes": ""
  },
  "blockers": [],
  "allowed_use_cases": {
    "mastery": {"status": "block", "reasons": []},
    "guardian": {"status": "block", "reasons": []},
    "export": {"status": "allow", "reasons": []},
    "source_backed_examples": {"status": "allow", "reasons": []},
    "candidate_generation": {"status": "block", "reasons": []}
  },
  "review": {
    "review_status": "approved",
    "reviewed_by": "reviewer identity",
    "reviewed_at": "ISO-8601 timestamp",
    "source_question_image_verified": true,
    "source_mark_scheme_image_verified": true,
    "exact_skill_verified": true
  },
  "provenance": {
    "decision_id": "...",
    "decision_source_path": "...",
    "source_question_bank_sha256": "...",
    "source_mark_events_sha256": "...",
    "source_taxonomy_sha256": "..."
  }
}
```

For `clean`, `reviewed_source_skill_ids` should contain exactly one P3 skill ID. Multi-skill records should be `ambiguous`, `thin`, or `review_needed` unless the reviewer splits evidence to event/subpart level or explicitly marks an inseparable co-assessed case with restricted use cases.

`allowed_use_cases.*.status` should be one of `allow`, `review_only`, or `block`. Unknown or missing use cases default to `block`.

Skill summary shape:

```json
{
  "skill_id": "9709_p3_3_2_log_exponential_equations",
  "skill_name": "Logarithmic and exponential equations",
  "clean_evidence_count": 0,
  "thin_evidence_count": 0,
  "ambiguous_evidence_count": 0,
  "blocked_evidence_count": 0,
  "deferred_evidence_count": 0,
  "fallback_only_evidence_count": 0,
  "clean_evidence_ids": [],
  "clean_source_question_ids": [],
  "source_diversity": {
    "paper_count": 0,
    "years": [],
    "sessions": [],
    "variants": []
  },
  "readiness": "no_clean_evidence"
}
```

Suggested readiness values: `no_clean_evidence`, `seeded_single_source`, `thin_reviewed`, `source_backed_examples_ready`, `candidate_generation_ready`.

## F. Deliberately Excluded From V1

V1 should not include generated worked examples, generated hints, model-generated student-facing explanations, scoring rubrics, student submissions, or auto-grading decisions.

V1 should not promote OCR/native/raw/advisory text into curriculum authority. Text snippets may appear only as review context or provenance, never as the proof that a record is clean.

V1 should not consume the current P3 question-skill mappings as clean evidence. They are useful review seeds but all current mappings are `needs_review`.

V1 should not collapse mixed whole-question evidence into a single exact skill when part-level evidence is required.

V1 should not treat P1 prerequisite skills as P3 mastery evidence. P1 IDs may appear only in a separate prerequisite/context field, not in `reviewed_source_skill_ids`.

V1 should not hide ambiguous, deferred, blocked, or fallback-only records. Reports must preserve them with reason codes.

V1 should not let Asterion consume the sidecar for runtime behavior until validator, review queue, and coverage reports are stable and the Asterion contract is explicitly updated.

## G. Validation Rules Required Before `clean`

A record may be `clean` only if all of these pass:

- `paper_family == "p3"`.
- `question_id` exists in `output/json/question_bank.json`.
- `review.review_status` is approved/reviewed according to a sidecar-specific approved enum.
- `reviewed_by` is non-empty and `reviewed_at` is a valid ISO timestamp.
- `source_question_image_verified=true` and `source_mark_scheme_image_verified=true`.
- Canonical question image path exists, has sha256, and resolves under the artifact root.
- Canonical mark-scheme image path exists, has sha256, and resolves under the artifact root.
- Source question paper and mark-scheme metadata are present and consistent with the question-bank pairing.
- `reviewed_source_skill_ids` contains exactly one ID from the P3 skill map.
- No P1/P2/M1/S1 skill ID appears in `reviewed_source_skill_ids`.
- If prerequisite IDs are present, they are stored separately and do not contribute to P3 mastery counts.
- The record's part path or subpart ID resolves to the canonical record. Whole-question evidence is allowed only when the reviewer explicitly verifies that the whole question is exact-skill evidence.
- Mapping, validation, scope, total, and mark-scheme pairing blockers are absent or explicitly resolved by reviewer image verification.
- Mark-event refs, when present, resolve to the mark-event sidecar and do not reference unknown event IDs.
- Advisory mark events alone do not authorize candidate generation. `candidate_generation=allow` requires reviewed/approved mark events or a future reviewed generation contract.
- `route_status=clean` cannot have unresolved blockers.
- `allowed_use_cases` cannot exceed route status. For example, `ambiguous`, `blocked`, `deferred`, `review_needed`, and `fallback_only` cannot allow `mastery` or `candidate_generation`.
- Topic-routing sidecar output is not required for clean if the exact skill decision is human reviewed, but any topic-routing conflict must be retained in provenance/blockers.

## H. Review Queue And Report Design

Before promotion, generate a review queue and a human-readable report, probably under future paths like:

```text
output/asterion/reports/p3_exact_skill_evidence_review_queue_v1.json
reports/asterion/p3_exact_skill_evidence_review_queue_v1.md
```

The queue should include every P3 candidate, including blocked and ambiguous records. It should never emit only easy cases.

Queue fields should include:

- `question_id`, `paper`, `question_number`, `subpart_id`, `part_path`, `marks`.
- Candidate skill IDs from P3 question-skill mappings, topic assignments, Asterion subpart skill IDs, and reviewer seed fixtures.
- Canonical question and mark-scheme image refs with existence/hash state.
- Question-bank validation, mapping, scope, text fidelity, visual, and crop status.
- Mark-event extraction status, advisory safety, event count, dependency/follow-through/unknown-code risks.
- Topic-routing confidence, review-required status, and parent topic.
- Content Lab generation blockers and role statuses.
- Suggested route status and blocker reasons.
- Reviewer fields to complete: exact skill, part scope, source image verification, mark-scheme verification, allowed use cases, notes.

Priority should favor records with canonical assets, consistent totals, advisory-safe mark events, subpart-level candidate mappings, high-confidence topic route agreement, and few dependency risks. Priority must not remove blocked records from the queue.

## I. Coverage And Diversity Report By Skill

Generate a skill coverage/diversity report after validation, probably as both JSON and Markdown. It should group by P3 skill ID and include:

- Counts by route status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.
- Clean evidence IDs and clean source question IDs.
- Number of distinct papers, years, sessions, variants, and question numbers.
- Whether evidence is concentrated in one paper/session/year.
- Whether all clean evidence is whole-question-only.
- Whether mark-event refs are reviewed, advisory-safe, or missing.
- Top blockers by skill.
- Readiness classification.

Suggested readiness policy:

- `no_clean_evidence`: zero clean records.
- `seeded_single_source`: one clean record.
- `thin_reviewed`: two or more clean records but weak diversity or missing reviewed mark-event support.
- `source_backed_examples_ready`: enough clean image-backed records for source-backed example selection, with no unresolved asset blockers.
- `candidate_generation_ready`: clean exact-skill records plus reviewed mark-event/generation basis and explicit candidate-generation permission.

## J. Safest Implementation Sequence

Phase 0: Freeze and audit current evidence surfaces. Record source paths, hashes, current output contracts, and no-mutation expectations.

Phase 1: Define a reviewed-decision fixture schema before writing the exporter. This fixture should be the only source allowed to assert exact P3 skill review.

Phase 2: Add validator design and tests for the reviewed-decision fixture and future sidecar. The validator must reject advisory-only clean promotion.

Phase 3: Add review queue/report design and implementation. The queue should aggregate question bank, Asterion projection, mark events, topic routing, and P3 candidate mappings without promoting any record.

Phase 4: Add a small manually reviewed seed fixture. Keep incomplete entries as `review_needed`, `blocked`, or `deferred`; do not omit them.

Phase 5: Add the sidecar exporter consuming reviewed decisions only. It should write `p3_exact_skill_evidence_v1.json` and preserve blocked/ambiguous/deferred records.

Phase 6: Add skill coverage/diversity report generation and validation.

Phase 7: Update the Asterion export contract and only then let Asterion consume the sidecar for any runtime behavior.

## K. Acceptance Criteria For A First Useful Milestone

A useful first implementation milestone should produce no generated student content and no Asterion runtime behavior changes. It should deliver:

- A reviewed-decision schema and validator.
- A P3 review queue/report covering all current P3 candidates.
- A seed reviewed-decision fixture with at least a small set of clean P3 exact-skill records and explicit blocked/ambiguous examples.
- A generated sidecar that validates with zero errors.
- Clean records with canonical question and mark-scheme image refs, sha256s, reviewer metadata, exact P3 skill IDs, route statuses, blockers, and allowed use cases.
- Skill summary showing which P3 skills have clean evidence and which remain sparse.
- All current advisory-only or machine-candidate records either excluded from clean or retained with non-clean route status.
- `candidate_generation` still blocked unless reviewed mark-event/generation requirements are deliberately satisfied.

A reasonable first seed target is `10` clean evidence records across at least `5` P3 skills, plus representative blocked and ambiguous records. The milestone is still useful with fewer clean records if the review queue and validator are strong, because the main bottleneck is safe review throughput rather than bulk export.

## L. Biggest Risks And Avoidance

Risk: treating advisory text as authority. Avoid this by requiring reviewer image verification and by storing advisory text only as provenance.

Risk: promoting current P3 generated mappings. Avoid this by requiring a reviewed-decision fixture and rejecting `review_status=needs_review` as clean input.

Risk: collapsing mixed whole-question evidence into one skill. Avoid this by making clean whole-question evidence require explicit reviewer verification and by prioritizing part-level review.

Risk: treating P1 prerequisites as P3 mastery. Avoid this by validating that `reviewed_source_skill_ids` are P3-only and keeping prerequisites separate.

Risk: over-permissioning runtime use cases. Avoid this by defaulting all unknown/missing use cases to `block` and making `candidate_generation=allow` stricter than `source_backed_examples=allow`.

Risk: relying on topic routing while `safe_for_strict_filters=false`. Avoid this by using topic routing only for review context until it is safe and separately validated.

Risk: mark-event overreach. Avoid this by preserving `safe_for_marking_use=false`, using advisory mark-event IDs as refs only, and requiring reviewed events for generation permissions.

Risk: losing blocked cases. Avoid this by making blocker-preserving reports part of acceptance criteria and by validating status counts.

Risk: stale asset references. Avoid this by recording source question-bank hashes, image sha256s, source PDFs, taxonomy hashes, and reviewed-decision source hashes in provenance.
