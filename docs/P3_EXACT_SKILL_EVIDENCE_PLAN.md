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

## Phase 3 Review Batch Packet Workflow

Phase 3 adds a small-batch review workflow so a human reviewer can work through the highest-value P3 exact-skill candidates without scanning the full queue. The batch builder consumes the Phase 2 queue and the Phase 1 reviewed-decision registry, excludes already-reviewed scopes, and writes reviewer handoff artifacts under:

```text
data/review/p3_exact_skill_batches/
```

This location is intentional: the packet and manifest are generated handoff artifacts, and the decision template is human-editable review input. They are not Asterion runtime exports and they do not create or update:

```text
output/asterion/exports/latest/p3_exact_skill_evidence_v1.json
```

Generate the default first batch with:

```text
.venv/bin/python scripts/build_p3_exact_skill_review_batch.py --batch-id batch_0001 --limit 25
```

The command writes:

```text
data/review/p3_exact_skill_batches/batch_0001_review_packet.md
data/review/p3_exact_skill_batches/batch_0001_decision_template.v1.json
data/review/p3_exact_skill_batches/batch_0001_manifest.v1.json
```

The review packet Markdown is the reviewer-facing checklist. For each selected item it includes the queue ID, question and part/subpart scope, paper/session/variant, candidate P3 skill IDs, prerequisite/support skill context, candidate region/topic, topic-routing context, Content Lab blocker context, canonical question and mark-scheme asset refs, advisory-only mark-event refs, proposed blockers, recommended review action, and an exact checklist for inspecting images and deciding scope, skill, blockers, route status, allowed use cases, and evidence basis.

The decision template is deliberately not the reviewed-decision registry schema. It uses `exam_bank.p3_exact_skill.review_batch_template` so it cannot be confused with final reviewed evidence. Each generated record defaults to `route_status: review_needed`, `blockers: ["pending_human_review"]`, empty `reviewed_source_skill_ids`, a separate `suggested_source_skill_ids` draft field, empty reviewer fields, empty `evidence_basis`, and all allowed use cases set to `false`. A reviewer must manually inspect the source images, decide the exact skill and safe scope, write project-wording evidence basis, and then copy or merge approved records into `data/review/p3_exact_skill_reviewed_decisions.v1.json`.

The manifest records the queue and registry paths used, filters, selected and skipped counts, selected queue/question IDs, and an estimated sparse-skill coverage delta. It also warns that the batch is not reviewed evidence and is not the final Asterion sidecar.

Selection is conservative. The default batch selects from `clean_candidate` queue items only, requires candidate P3 skill IDs and both canonical question and mark-scheme asset refs, excludes already-reviewed scopes, prefers lower-ambiguity single-skill candidates, boosts skills with zero clean reviewed evidence, prefers Content Lab candidates and image/mark-event-backed review context, and applies light diversity penalties for repeated question, paper, session, and skill. Mark-event refs are preserved only as advisory context; they are never promotion authority.

No candidate is clean until a human reviewer has inspected the canonical question and mark-scheme images and manually moved an approved record into the reviewed-decision registry. The future exporter must continue to treat `clean_candidate` and review-batch templates as non-authoritative inputs.

## Phase 3B Visual Review Packet Workflow

Phase 3B adds a visual HTML packet for an existing Phase 3 batch. The purpose is to make manual review practical: the reviewer can open one local file and inspect the question image, mark-scheme image, candidate skill context, blockers, advisory mark-event refs, and checklist together for each selected item.

Generate the visual packet with:

```text
.venv/bin/python scripts/build_p3_exact_skill_visual_review_packet.py --batch-id batch_0001
```

The default output is:

```text
data/review/p3_exact_skill_batches/batch_0001_visual_review.html
```

The visual builder reads the existing batch manifest, Markdown packet, and decision template. It also reads the queue JSON referenced by the manifest when available, so it can recover candidate region/topic, topic-routing context, Content Lab blocker context, proposed blockers, and recommended review action. The HTML is static and has no external network dependencies.

The HTML includes draft response controls for each item. Responses autosave in browser `localStorage` under a batch-specific key and can be exported as JSON. To save responses directly into the repo while reviewing, serve the page with:

```text
.venv/bin/python scripts/serve_p3_exact_skill_visual_review.py --batch-id batch_0001
```

Open the printed local URL, fill in the response fields, and click `Submit to local save server`. The server writes:

```text
data/review/p3_exact_skill_batches/batch_0001_review_responses.v1.json
```

This response file is a draft human-review note artifact only. It is not the reviewed-decision registry and is not consumed as clean evidence.

Asset refs are resolved against the repo root. If a ref such as `p3/...` is not present directly under the repo root, the builder also checks `output/p3/...`, which matches the current canonical crop layout. Existing assets are linked with relative paths from the HTML file so the packet can be opened locally without copying images. Missing assets are shown as visible warnings beside the review item.

This visual packet is still not reviewed evidence. It does not edit `data/review/p3_exact_skill_reviewed_decisions.v1.json`, does not promote any candidate, does not change the decision template, and does not create:

```text
output/asterion/exports/latest/p3_exact_skill_evidence_v1.json
```

Use the visual packet to inspect images and make the human decision. Approved decisions must still be manually written or merged into the reviewed-decision registry, with project-wording `evidence_basis`, explicit `route_status`, reviewer metadata, blockers, and allowed use cases. The reviewed-decision validator remains the gate before any future sidecar exporter can consume those records.

### Known Skill-Boundary Risk: DE vs Parametric/Implicit Differentiation

Reviewers must not classify a record as parametric/implicit differentiation merely because `dy/dx` appears. Parametric/implicit differentiation should be used only when the task requires differentiating a parametric relation or an implicit relation where `y` cannot simply be treated as an explicit function of `x`, especially where `dy/dx` must be isolated from multiple differentiable terms.

Parametric-equation evidence needs an actual parameterized setup: separate `x` and `y` relations in a parameter such as `t` or `theta`, or method evidence such as `dx/dt` and `dy/dt`. A source-topic hint, the word `parametric`, or loose `x =` / `y =` OCR text is not enough. The queue flags weak cases with `weak_parametric_equation_evidence_missing_parameter` and recommended action `verify_parametric_equation_parameter`.

If the first meaningful mark-scheme method step is separation of variables, integration of separated terms, solving a differential equation, or applying an initial/boundary condition to a differential-equation solution, the item should route to differential equations / separation of variables instead. Mark-scheme method order is useful review context for distinguishing these cases, but OCR/native/advisory text still is not curriculum authority. Uncertain cases should remain `ambiguous` or `review_needed`, not `clean_candidate`.

The review queue now flags this boundary with blockers such as `possible_differential_equation_not_parametric_or_implicit` or `weak_parametric_implicit_evidence_dydx_only` and recommended action `verify_de_vs_implicit_differentiation`.

### Cross-Topic Review Rules

Cross-topic context does not automatically mean a candidate is wrong. Many P3 questions use one method inside another, such as differentiating first and then solving a trigonometric equation, or using integration inside a differential-equation solution. The review queue therefore records cross-topic fields including `cross_topic_status`, `primary_candidate_skill_ids`, `supporting_candidate_skill_ids`, `topic_routing_topic_ids`, `topic_routing_alignment`, `cross_topic_notes`, and `recommended_scope`.

`cross_topic_reviewable` means the item can remain in a human review packet, but the reviewer must decide which exact skill is the target and whether the supporting context makes the current scope unsafe. Supporting skills are not mastery evidence unless they are reviewed directly as the actual target skill. Whole-question evidence should not be marked clean if only one part supports the exact skill; split by part or subpart where possible.

If a question requires method A and then method B, the reviewer must decide whether reviewed evidence is for A, B, both as separate records, or too mixed for clean use. A topic-routing mismatch should produce a review cue, not automatic rejection, unless method context clearly indicates misclassification. `cross_topic_split_needed` means the current whole-question scope is too broad for a clean exact-skill decision. `conflict_needs_review` is reserved for genuine unsafe boundary patterns.

The differential-equation vs parametric/implicit guard remains a special known-risk boundary: `dy/dx` alone is insufficient for parametric/implicit differentiation, and separation of variables should route toward differential-equation review.

### Ambiguity Reduction Is Triage, Not Trust Promotion

The review queue now uses sharper candidate statuses so broad `ambiguous_candidate` records are not all handled the same way. This does not make any record reviewed evidence and does not increase allowed use cases. Only `data/review/p3_exact_skill_reviewed_decisions.v1.json`, after manual review and validation, can assert clean evidence.

The queue statuses are review categories:

- `clean_candidate`: single-skill clean-looking review candidate, still not reviewed evidence.
- `cross_topic_candidate`: plausible primary/supporting P3 context; reviewer must identify the target skill and avoid treating support context as mastery evidence.
- `split_needed_candidate`: current scope is likely too broad; part/subpart review is needed before any clean decision.
- `conflict_candidate`: known-risk or method-critical mismatch, including DE vs parametric/implicit conflicts; treat as ambiguous or blocked unless canonical images clearly resolve it.
- `weak_candidate`: some skill evidence exists, but the context is weaker or too thin for priority review.
- `ambiguous_candidate`: reserved for true unresolved uncertainty after sharper triage.
- `fallback_only` and `blocked_candidate`: low-quality fallback or blocked review contexts remain visible in reports.

Lowering the ambiguous count means the review workflow has better triage, not more trusted evidence. The ambiguity audit artifacts, `reports/p3_exact_skill_ambiguity_audit.v1.json` and `reports/p3_exact_skill_ambiguity_audit.md`, show where broad ambiguity moved and how each group should be handled.

### Part-Level Decomposition Review

Many P3 questions are cross-topic at whole-question level because different parts test different skills. A subpart is often closer to one exact skill than the full question, so the workflow now generates conservative part-level decomposition candidates as review assistance.

The decomposition pass uses existing queue scopes, question-bank subpart labels, and advisory mark-event `part_path` labels. It does not create part crops, does not infer curriculum authority from OCR/native text, and does not promote any candidate. When part-level crops are unavailable, the visual packet links the whole-question and whole mark-scheme images and warns the reviewer to confirm the part boundary manually.

The report artifacts are:

```text
reports/p3_exact_skill_part_decomposition.v1.json
reports/p3_exact_skill_part_decomposition.md
```

Decomposition statuses include `part_level_candidate`, `subpart_level_candidate`, `already_part_scoped`, `needs_manual_split`, `insufficient_part_signal`, `conflict_needs_review`, and `not_decomposable`. A `part_level_candidate` means there is enough existing part-label signal to make a human review packet more focused; it is still not reviewed evidence.

Use `scripts/build_p3_exact_skill_review_batch.py --batch-purpose part_decomposition_review` to build a batch that prioritizes cross-topic or split-needed records with proposed part-level candidates. The reviewer must still decide whether the reviewed registry record should be whole-question, part-level, subpart-level, ambiguous, blocked, or deferred. Supporting skills remain context only unless reviewed directly as the target skill.

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
