# Manual Review Batch 0002 Conclusions

Batch 0002 was evaluated as a controlled evidence-building pass. It promoted a small clean subset for source-backed examples only; mark-event approval and generation readiness remain blocked.

## Source Files Inspected

- `review_responses`: `data/review/p3_exact_skill_batches/batch_0002_review_responses.v1.json`
- `manifest`: `data/review/p3_exact_skill_batches/batch_0002_manifest.v1.json`
- `decision_template`: `data/review/p3_exact_skill_batches/batch_0002_decision_template.v1.json`
- `review_packet`: `data/review/p3_exact_skill_batches/batch_0002_review_packet.md`
- `batch_0002_plan`: `reports/p3_exact_skill_batch_0002_plan.md`
- `batch_0002_plan_json`: `reports/p3_exact_skill_batch_0002_plan.v1.json`
- `reviewed_registry`: `data/review/p3_exact_skill_reviewed_decisions.v1.json`
- `batch_0001_conclusions`: `reports/manual_review_batch_0001_conclusions.md`
- `batch_0001_conclusions_json`: `reports/manual_review_batch_0001_conclusions.v1.json`
- `batch_0001_seed_report_json`: `reports/p3_exact_skill_registry_seed_0001.v1.json`
- `reviewed_decisions_validation`: `reports/p3_exact_skill_reviewed_decisions_validation.v1.json`
- `content_lab_candidates`: `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- `asset_manifest`: `output/json/asset_manifest.v1.json`

## Outcome Counts

- total_reviewed_responses: `37`
- clean: `30`
- ambiguous: `4`
- blocked: `1`
- rejected: `1`
- needs_further_review: `6`
- thin: `2`
- part_level: `29`
- whole_question: `8`
- reliable_pattern_confirmation_items: `12`
- known_failure_mode_probes: `12`
- deferred_batch_0001_clean_items: `6`
- seed_mark_event_alignment_probes: `7`

## Category Outcomes

- `deferred_batch_0001_clean|clean`: 6
- `known_failure_mode_probe|ambiguous`: 4
- `known_failure_mode_probe|blocked`: 1
- `known_failure_mode_probe|clean`: 7
- `reliable_pattern_confirmation|clean`: 10
- `reliable_pattern_confirmation|thin`: 2
- `seed_mark_event_alignment_probe|clean`: 7

## Compared With Batch 0001

### Reliable Patterns Confirmed
- Complex polar/modulus, fixed-point iteration, trig identities/equations, vector lines, and parametric/implicit differentiation remained reliable when part boundaries were explicit.
- Whole-question-safe algebra/log/polynomial tasks can be clean when the whole question assesses one target skill.
- Part-level scoping continues to prevent supporting-method confusion; adjacent thin or supporting parts were not promoted.

### Failure Modes Confirmed
- Trig identities used inside integration-area work remain unsafe as exact-skill targets.
- Log/exponential algebra inside differential-equation work remains a supporting method, not clean source-skill evidence.
- Wrong-topic parametric/implicit differentiation probes can still be completely blocked.
- Broad standard-integration labels remain unsafe where substitution/improper-limit structure is the actual target.

### Previously Suspicious Patterns Now Safer
- A supporting-topic route can be safe only after part-level review proves the selected subpart itself is the target skill, as with 31summer21_q04_a.
- Whole-question scope is safer for single-skill algebra/log/polynomial questions, but remains review-gated.

### Previously Trusted Patterns Less Safe
- No Batch 0001 trusted pattern became broadly unsafe, but thin adjacent subparts showed that fixed-point/vector labels are not enough without assessed-work depth.

- Whole-question vs part-level: Part-level ambiguity remains a blocker for mixed questions. Batch 0002 found 8 whole-question-safe cases but still required subpart scope on 29 of 37 responses.
- Mark-event alignment: The seven seed mark-event probes found aligned-looking refs, but every response kept them advisory-only. No mark-event approval was explicit enough to record as reviewed or approved.

## Promotions

- `p3_exact_skill_review:batch_0002_seed:31summer24_q01:31summer24_q01_whole` from `batch_0002:8:p3_exact_skill_review_queue:v1:31summer24_q01:31summer24_q01_whole`: Adds a whole-question-safe binomial rational expansion record; canonical evidence is a single expansion/product task with no part-boundary ambiguity.
- `p3_exact_skill_review:batch_0002_seed:33autumn23_q01:33autumn23_q01_whole` from `batch_0002:9:p3_exact_skill_review_queue:v1:33autumn23_q01:33autumn23_q01_whole`: Adds log/exponential equation coverage where the whole task is the target skill rather than a supporting method inside another topic.
- `p3_exact_skill_review:batch_0002_seed:33autumn23_q03:33autumn23_q03_whole` from `batch_0002:10:p3_exact_skill_review_queue:v1:33autumn23_q03:33autumn23_q03_whole`: Adds polynomial division/factor/remainder coverage from a whole-question-safe algebra task.
- `p3_exact_skill_review:batch_0002_seed:31summer21_q04:31summer21_q04_a` from `batch_0002:15:p3_exact_skill_review_queue:v1:31summer21_q04:31summer21_q04_a`: Promotes a resolved suspicious pattern: the whole question later uses integration, but reviewed part (a) itself is a direct trigonometric identity proof.
- `p3_exact_skill_review:batch_0002_seed:32spring24_q03:32spring24_q03_a` from `batch_0002:26:p3_exact_skill_review_queue:v1:32spring24_q03:32spring24_q03_a`: Adds a new part-specific complex polar/modulus example from outside the Batch 0001 seed records.
- `p3_exact_skill_review:batch_0002_seed:31summer24_q06:31summer24_q06_d` from `batch_0002:30:p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_d`: Promotes the strong fixed-point iteration subpart and leaves adjacent thin part (c) unpromoted.
- `p3_exact_skill_review:batch_0002_seed:31summer24_q09:31summer24_q09_b` from `batch_0002:34:p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_b`: Adds a vector-line intersection record while leaving the adjacent one-mark scalar-product part unpromoted as thin.
- `p3_exact_skill_review:batch_0002_seed:33summer25_q05:33summer25_q05_a` from `batch_0002:35:p3_exact_skill_review_queue:v1:33summer25_q05:33summer25_q05_a`: Adds a recent implicit differentiation proof with derivative-rule work treated as support only.

## Mark-Event Alignment Findings

- Explicit approvals: `0`
- Finding: Mark-event refs sometimes aligned to the reviewed part, but Batch 0002 responses did not explicitly approve them. All promoted refs remain advisory.

## Generation Readiness

- New generation-ready candidates from this pass: `0`
- Finding: No Content Lab candidate became generation-ready; reviewed source-skill evidence was kept separate from mark-event approval and generation gates.

## Fields That Remain Review-Gated

- mark_event_refs and mark-event approval
- candidate_generation readiness
- mastery/guardian/export use cases
- topic rerouting for ambiguous or blocked probes
- whole-question scope for mixed-topic questions

## Fields Now Safer For Triage Indexing

- route_status clean/ambiguous/blocked/thin counts
- selection category as review workload signal
- part-level vs whole-question scope decisions
- supporting_context_only vs possible_target_skill decisions
- Content Lab generation gate block reasons as diagnostics only

## Improvements Implemented

- Promoted 8 clean Batch 0002 records into the reviewed-decision registry with verified canonical asset refs and Batch 0002 provenance.
- Created Batch 0002 conclusion and seed reports separating source-skill evidence, mark-event status, and generation readiness.
- Kept all promoted mark-event refs advisory-only and all candidate_generation flags false.

## Improvements Deferred

- No Content Lab exporter bridge.
- No broad topic-routing repair.
- No difficulty-band changes.
- No automatic promotion rules.
- No advisory text treated as canonical evidence.

## Risks And Concerns

- Batch 0002 response notes are still draft review notes; this pass promotes only a manually selected subset after checking canonical asset refs and schema gates.
- Promoted part-level records still use whole-question and whole mark-scheme images as canonical refs; exact reviewed fields/regions are stored in subpart_id, part_id, and evidence notes.
- All mark events remain advisory, so registry warnings for clean records with unreviewed mark events are expected.
- Content Lab candidates still have empty source_skill_ids in the current export, so generation remains blocked independently of reviewed source-skill evidence.

## Recommended Batch 0003 Composition

- A smaller 20-25 item batch focused on unresolved ambiguous/blocked probes after retagging target skills.
- Include explicit mark-event approval probes only if the reviewer can approve individual event IDs, not just source-skill alignment.
- Add paired adjacent subparts where one is thin and one is substantive to harden depth gating.
- Include more whole-question-safe single-skill items from underrepresented skills such as modulus equations and narrower integration subskills.

## Skipped Clean Records

- `33summer23_q11_b`: `already_promoted_in_batch_0001_seed_registry`
- `31summer24_q04_b`: `already_promoted_in_batch_0001_seed_registry`
- `32summer23_q06_c`: `already_promoted_in_batch_0001_seed_registry`
- `32autumn23_q06_c`: `already_promoted_in_batch_0001_seed_registry`
- `33summer23_q06_b`: `already_promoted_in_batch_0001_seed_registry`
- `33summer23_q09_b`: `already_promoted_in_batch_0001_seed_registry`
- `32spring23_q05_b`: `already_promoted_in_batch_0001_seed_registry`
- `33summer23_q04_whole`: `not_in_small_batch_0002_seed_subset`
- `31autumn23_q08_d`: `not_in_small_batch_0002_seed_subset`
- `31summer23_q04_b`: `not_in_small_batch_0002_seed_subset`
- `32autumn21_q06_a`: `not_in_small_batch_0002_seed_subset`
- `31autumn23_q06_a`: `not_in_small_batch_0002_seed_subset`
- `31autumn23_q06_b`: `not_in_small_batch_0002_seed_subset`
- `32autumn23_q03_whole`: `not_in_small_batch_0002_seed_subset`
- `32spring23_q03_whole`: `not_in_small_batch_0002_seed_subset`
- `31autumn21_q01_whole`: `not_in_small_batch_0002_seed_subset`
- `32spring24_q03_b`: `not_in_small_batch_0002_seed_subset`
- `32spring24_q05_b`: `clean_but_narrower_argand_loci_skill_uncertainty_not_resolved`
- `33autumn23_q06_a`: `not_in_small_batch_0002_seed_subset`
- `33summer23_q06_a`: `not_in_small_batch_0002_seed_subset`
- `32spring23_q10_b`: `not_in_small_batch_0002_seed_subset`
- `32spring24_q07_c`: `not_in_small_batch_0002_seed_subset`

## Ambiguous Blocked Or Thin Records

- `32autumn23_q09_b`: `ambiguous`; Draft review adjustment: part (b) asks for exact area under y = sin x cos 2x and the mark scheme is integration with trig identities as method support. The proposed trig-identities skill is not the safest target skill for the selected part; this should be retagged toward integration/area evidence before clean use. Draft Batch 0002 review only; canonical images remain the authority for user double-check and this is not reviewed-registry evidence.
- `33autumn22_q10_a`: `ambiguous`; Draft review adjustment: selected part (a) only identifies constants in a differential equation model. The proposed log/exponential algebra skill is not assessed in this subpart; this should stay out of clean source-skill evidence. Draft Batch 0002 review only; canonical images remain the authority for user double-check and this is not reviewed-registry evidence.
- `33autumn22_q10_b`: `ambiguous`; Draft review adjustment: part (b) solves a separable differential equation and uses logarithms during integration/constant evaluation. Log/exponential algebra is supporting method context, while the target skill is differential equations. Draft Batch 0002 review only; canonical images remain the authority for user double-check and this is not reviewed-registry evidence.
- `31summer23_q02_a`: `blocked`; Draft review blocked: selected item is a modulus graph/linear inequality question, not parametric or implicit differentiation. The proposed P3 exact skill is unsupported by the canonical text and mark scheme; do not promote without rerouting. Draft Batch 0002 review only; canonical images remain the authority for user double-check and this is not reviewed-registry evidence.
- `31autumn21_q04_whole`: `ambiguous`; Draft review adjustment: whole question explicitly requires substitution u = sqrt(x) and changing/evaluating limits for an improper integral. The proposed standard-integration skill is too broad; safer exact target is integration by substitution and by parts/substitution before clean use. Draft Batch 0002 review only; canonical images remain the authority for user double-check and this is not reviewed-registry evidence.
- `31summer24_q06_c`: `thin`; Draft review: part (c) only rearranges the equation to show the fixed-point formula converges to the root; the actual iterative calculation is in part (d). This is exact but thin fixed-point evidence and should not be promoted as a strong clean example alone. Draft Batch 0002 review only; canonical images remain the authority for user double-check and this is not reviewed-registry evidence.
- `31summer24_q09_a`: `thin`; Draft review: part (a) is a one-mark scalar-product check to show the parameter value making two vector lines perpendicular. It is aligned to vector-line evidence but thin as standalone reviewed evidence. Draft Batch 0002 review only; canonical images remain the authority for user double-check and this is not reviewed-registry evidence.