# P3 Exact-Skill Review Packet: batch_0003

This packet is for human review only. It does not assert clean evidence, does not update the reviewed-decision registry, and does not create the Asterion sidecar.

## Batch Metadata

- Generated at: `2026-05-26T02:26:00Z`
- Source queue: `reports/p3_exact_skill_review_queue.v1.json`
- Reviewed registry checked for exclusions: `data/review/p3_exact_skill_reviewed_decisions.v1.json`
- Selection status: `multiple`
- Included statuses: `cross_topic_candidate, conflict_candidate, split_needed_candidate, fallback_only`
- Excluded statuses: `none`
- Batch purpose: `batch_0003_adversarial_mark_event_review`
- Selection limit: `20`
- Selected items: `14`

## Reviewer Checklist

- Inspect the canonical question image.
- Inspect the canonical mark-scheme image.
- Confirm the exact P3 skill.
- Confirm whether whole-question or part-level scope is safe.
- Confirm whether P1 prerequisite/support-only material is involved.
- Confirm allowed use cases.
- Write evidence_basis in project wording.
- Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

## Batch 0003 Review Instructions

- Review exact-skill decisions and mark-event decisions separately for every record.
- Use approved, retagged, deferred, blocked, or thin for the exact-skill outcome; do not infer mark-event approval from that outcome.
- Keep mark-event refs advisory-only unless an explicit reviewed-mark-event schema and validator path already exist.
- Treat known controls as controls only: they test that clean source-skill evidence stays distinguishable from unsafe probes.
- Do not change Content Lab generation readiness from advisory mark-event refs.

> Mark-event refs are advisory-only review context. They are not authority for clean evidence, marking use, or candidate generation.

## Review Items

### 1. `32autumn23_q09` / `32autumn23_q09_b`

- Selection category: `prior_ambiguous_retag_probe`
- Selection reason: Retest Batch 0001/0002 trig-identity ambiguity where area/integration is the assessed target and trig identities are method support.
- Known risk flags: `supporting_method_confusion`, `integration_trig_area_target`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32autumn23_q09:32autumn23_q09_b`
- Question ID: `32autumn23_q09`
- Part/subpart: `b` / `32autumn23_q09_b`
- Paper/session/variant: `32autumn23` / `November` / `2`
- Candidate P3 skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `topic_routing_candidate_mismatch`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "parametric_equations", "subtopic_id": "9709_p3_subtopic_identities_compound_double_angle_equations", "subtopic_name": "Identities, compound angles and trigonometric equations", "topic_assignment_id": "9709_p3_topic_trigonometry", "topic_assignment_name": "Trigonometry", "topic_routing_primary_topic_id": "9709_p3_topic_integration"}`
- Topic-routing context: `{"confidence": "medium", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_integration", "review_reasons": ["Question involves both differentiation and integration; primary topic set to integration based on final objective."], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_integration", "9709_p3_topic_differentiation"]`
- Topic-routing alignment: `supporting_topic`
- Recommended scope: `subpart_level`
- Cross-topic notes: Candidate skill/topic and topic-routing context differ but may describe different stages of one solution.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: parametric_equations.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_3_identities_compound_double_angle_equations"], "candidate_topic_ids": ["9709_p3_topic_integration", "9709_p3_topic_differentiation"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32autumn23_q09:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32autumn23_q09_me0011", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0012", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0013", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0014", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0015", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0016", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0017", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0018", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0019", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0020", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0021", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0022", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32autumn23_q09_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0003", "mark_code": "DM1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0005", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0006", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0007", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0008", "mark_code": "DM1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0009", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q09_me0010", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "32autumn23_q09", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32autumn23/mark_scheme/q09.png", "sha256": "7c70eb1c41efdf43528634ac48130a048852b8a20decca8f40028f2f392aad62"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32autumn23/questions/q09.png", "sha256": "6e17028be9d57ce6d5668a932e446117d2deaf2b2ffa1ccf1ceadd228cc0f2aa"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32autumn23_q09:32autumn23_q09_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32autumn23_q09_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 12}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32autumn23/questions/q09.png", "sha256": "6e17028be9d57ce6d5668a932e446117d2deaf2b2ffa1ccf1ceadd228cc0f2aa"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32autumn23/mark_scheme/q09.png", "sha256": "7c70eb1c41efdf43528634ac48130a048852b8a20decca8f40028f2f392aad62"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0003", "mark_code": "DM1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0005", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0006", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0007", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0008", "mark_code": "DM1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0009", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0010", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0011", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0012", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0013", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0014", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0015", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0016", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0017", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0018", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0019", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0020", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0021", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q09_me0022", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 2. `31autumn21_q07` / `31autumn21_q07_c`

- Selection category: `prior_ambiguous_retag_probe`
- Selection reason: Retest Batch 0001 DE/log ambiguity where the selected part is limiting behaviour from a differential-equation solution.
- Known risk flags: `supporting_method_confusion`, `de_log_context`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31autumn21_q07:31autumn21_q07_c`
- Question ID: `31autumn21_q07`
- Part/subpart: `c` / `31autumn21_q07_c`
- Paper/session/variant: `31autumn21` / `November` / `1`
- Candidate P3 skill IDs: `9709_p3_3_2_log_exponential_equations`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `topic_routing_candidate_mismatch`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_2_log_exponential_equations`, `9709_p1_functions_domain_range_one_one`
- Primary candidate skill IDs: `9709_p3_3_2_log_exponential_equations`
- Supporting candidate skill IDs: `9709_p1_functions_domain_range_one_one`
- Candidate region/topic: `{"mapping_source_topic": "logarithms_and_exponentials", "subtopic_id": "9709_p3_subtopic_log_exponential_equations", "subtopic_name": "Logarithmic and exponential equations", "topic_assignment_id": "9709_p3_topic_logarithmic_and_exponential_functions", "topic_assignment_name": "Logarithmic and exponential functions", "topic_routing_primary_topic_id": "9709_p3_topic_differential_equations"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differential_equations", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differential_equations"]`
- Topic-routing alignment: `supporting_topic`
- Recommended scope: `subpart_level`
- Cross-topic notes: Candidate skill/topic and topic-routing context differ but may describe different stages of one solution.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: logarithms_and_exponentials.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_2_log_exponential_equations"], "candidate_topic_ids": ["9709_p3_topic_differential_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31autumn21_q07:c", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "31autumn21_q07_me0009", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31autumn21_q07_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31autumn21_q07_me0002", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31autumn21_q07_me0003", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31autumn21_q07_me0004", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31autumn21_q07_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31autumn21_q07_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31autumn21_q07_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31autumn21_q07_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["c"], "proposed_part_id": "c", "proposed_subpart_id": null, "question_id": "31autumn21_q07", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31autumn21/mark_scheme/q07.png", "sha256": "10dd73f6cca934197876cb68a0257abcdfb13a479134dbb5a831aeb0be0ce240"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31autumn21/questions/q07.png", "sha256": "94854043f612f37e702b9fe3f6f56ee45787056d7b02b0df951586037cb323ba"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31autumn21_q07:31autumn21_q07_c", "supporting_skill_ids": ["9709_p1_functions_domain_range_one_one"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31autumn21_q07_c", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 1}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31autumn21/questions/q07.png", "sha256": "94854043f612f37e702b9fe3f6f56ee45787056d7b02b0df951586037cb323ba"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31autumn21/mark_scheme/q07.png", "sha256": "10dd73f6cca934197876cb68a0257abcdfb13a479134dbb5a831aeb0be0ce240"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0002", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0003", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0004", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q07_me0009", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 3. `32summer23_q07` / `32summer23_q07_b`

- Selection category: `prior_ambiguous_retag_probe`
- Selection reason: Retest Batch 0001 derivative-rules ambiguity where implicit differentiation and tangent conditions are the safer target.
- Known risk flags: `wrong_skill_routing`, `implicit_vs_derivative_rules`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32summer23_q07:32summer23_q07_b`
- Question ID: `32summer23_q07`
- Part/subpart: `b` / `32summer23_q07_b`
- Paper/session/variant: `32summer23` / `June` / `2`
- Candidate P3 skill IDs: `9709_p3_3_4_derivative_rules`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_4_derivative_rules`, `9709_p1_differentiation_power_chain`
- Primary candidate skill IDs: `9709_p3_3_4_derivative_rules`
- Supporting candidate skill IDs: `9709_p1_differentiation_power_chain`
- Candidate region/topic: `{"mapping_source_topic": "differentiation", "subtopic_id": "9709_p3_subtopic_derivative_rules", "subtopic_name": "Derivative rules for P3 functions", "topic_assignment_id": "9709_p3_topic_differentiation", "topic_assignment_name": "Differentiation", "topic_routing_primary_topic_id": "9709_p3_topic_differentiation"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differentiation", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differentiation"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: differentiation.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_4_derivative_rules"], "candidate_topic_ids": ["9709_p3_topic_differentiation"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32summer23_q07:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32summer23_q07_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer23_q07_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer23_q07_me0007", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer23_q07_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer23_q07_me0009", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer23_q07_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32summer23_q07_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer23_q07_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer23_q07_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer23_q07_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "32summer23_q07", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32summer23/mark_scheme/q07.png", "sha256": "bfce284f2bee9b9e626b239d5b7cf61a1b916cc0ec9aac9dc02383a13659f8b3"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32summer23/questions/q07.png", "sha256": "d844c7af809670b8e6beca8011b133b98d41bd83817dbea9f1f6775ba1c55693"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32summer23_q07:32summer23_q07_b", "supporting_skill_ids": ["9709_p1_differentiation_power_chain"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32summer23_q07_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 6}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32summer23/questions/q07.png", "sha256": "d844c7af809670b8e6beca8011b133b98d41bd83817dbea9f1f6775ba1c55693"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32summer23/mark_scheme/q07.png", "sha256": "bfce284f2bee9b9e626b239d5b7cf61a1b916cc0ec9aac9dc02383a13659f8b3"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32summer23_q07_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0007", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0009", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q07_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 4. `33autumn22_q10` / `33autumn22_q10_a`

- Selection category: `prior_ambiguous_retag_probe`
- Selection reason: Retest Batch 0002 log/exponential ambiguity where the subpart identifies constants in a differential-equation model.
- Known risk flags: `supporting_method_confusion`, `de_log_context`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_a`
- Question ID: `33autumn22_q10`
- Part/subpart: `a` / `33autumn22_q10_a`
- Paper/session/variant: `33autumn22` / `November` / `3`
- Candidate P3 skill IDs: `9709_p3_3_2_log_exponential_equations`
- Suggested candidate status: `fallback_only`
- Suggested review priority: `6_fallback_only`
- Suggested ambiguity reason: `fallback_or_low_quality_context`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_2_log_exponential_equations`, `9709_p1_functions_domain_range_one_one`
- Primary candidate skill IDs: `9709_p3_3_2_log_exponential_equations`
- Supporting candidate skill IDs: `9709_p1_functions_domain_range_one_one`
- Candidate region/topic: `{"mapping_source_topic": "logarithms_and_exponentials", "subtopic_id": "9709_p3_subtopic_log_exponential_equations", "subtopic_name": "Logarithmic and exponential equations", "topic_assignment_id": "9709_p3_topic_logarithmic_and_exponential_functions", "topic_assignment_name": "Logarithmic and exponential functions", "topic_routing_primary_topic_id": "9709_p3_topic_differential_equations"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differential_equations", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differential_equations"]`
- Topic-routing alignment: `supporting_topic`
- Recommended scope: `subpart_level`
- Cross-topic notes: Candidate skill/topic and topic-routing context differ but may describe different stages of one solution.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: logarithms_and_exponentials.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_2_log_exponential_equations"], "candidate_topic_ids": ["9709_p3_topic_differential_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33autumn22_q10:a", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33autumn22_q10_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33autumn22_q10_me0002", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0003", "mark_code": "A1FT", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0004", "mark_code": "A1FT", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0008", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0009", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}], "part_path": ["a"], "proposed_part_id": "a", "proposed_subpart_id": null, "question_id": "33autumn22_q10", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33autumn22/mark_scheme/q10.png", "sha256": "fe3fdb597c0d9064437178c8dd3d3f0c1a7a32bb03ecd251433466781c0df179"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33autumn22/questions/q10.png", "sha256": "7aed56b1cf4ce1612c84f6b741d46b97ba3608a494de787138af91b25b5eb440"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_a", "supporting_skill_ids": ["9709_p1_functions_domain_range_one_one"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33autumn22_q10_a", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 1}`
- Proposed blockers: `mark_events_advisory_only`, `question_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `defer_visual_dependency`

Question asset refs:
- `{"exists": true, "path": "p3/33autumn22/questions/q10.png", "sha256": "7aed56b1cf4ce1612c84f6b741d46b97ba3608a494de787138af91b25b5eb440"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33autumn22/mark_scheme/q10.png", "sha256": "fe3fdb597c0d9064437178c8dd3d3f0c1a7a32bb03ecd251433466781c0df179"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0002", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0003", "mark_code": "A1FT", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0004", "mark_code": "A1FT", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0008", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0009", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 5. `33autumn22_q10` / `33autumn22_q10_b`

- Selection category: `prior_ambiguous_retag_probe`
- Selection reason: Retest Batch 0002 log/exponential ambiguity where logarithms occur during separable differential-equation solving.
- Known risk flags: `supporting_method_confusion`, `de_log_context`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_b`
- Question ID: `33autumn22_q10`
- Part/subpart: `b` / `33autumn22_q10_b`
- Paper/session/variant: `33autumn22` / `November` / `3`
- Candidate P3 skill IDs: `9709_p3_3_2_log_exponential_equations`
- Suggested candidate status: `fallback_only`
- Suggested review priority: `6_fallback_only`
- Suggested ambiguity reason: `fallback_or_low_quality_context`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_2_log_exponential_equations`, `9709_p1_functions_domain_range_one_one`
- Primary candidate skill IDs: `9709_p3_3_2_log_exponential_equations`
- Supporting candidate skill IDs: `9709_p1_functions_domain_range_one_one`
- Candidate region/topic: `{"mapping_source_topic": "logarithms_and_exponentials", "subtopic_id": "9709_p3_subtopic_log_exponential_equations", "subtopic_name": "Logarithmic and exponential equations", "topic_assignment_id": "9709_p3_topic_logarithmic_and_exponential_functions", "topic_assignment_name": "Logarithmic and exponential functions", "topic_routing_primary_topic_id": "9709_p3_topic_differential_equations"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differential_equations", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differential_equations"]`
- Topic-routing alignment: `supporting_topic`
- Recommended scope: `subpart_level`
- Cross-topic notes: Candidate skill/topic and topic-routing context differ but may describe different stages of one solution.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: logarithms_and_exponentials.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_2_log_exponential_equations"], "candidate_topic_ids": ["9709_p3_topic_differential_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33autumn22_q10:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33autumn22_q10_me0002", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0003", "mark_code": "A1FT", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0004", "mark_code": "A1FT", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33autumn22_q10_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0008", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "33autumn22_q10_me0009", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "33autumn22_q10", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33autumn22/mark_scheme/q10.png", "sha256": "fe3fdb597c0d9064437178c8dd3d3f0c1a7a32bb03ecd251433466781c0df179"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33autumn22/questions/q10.png", "sha256": "7aed56b1cf4ce1612c84f6b741d46b97ba3608a494de787138af91b25b5eb440"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_b", "supporting_skill_ids": ["9709_p1_functions_domain_range_one_one"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33autumn22_q10_b", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 6}`
- Proposed blockers: `mark_events_advisory_only`, `question_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `defer_visual_dependency`

Question asset refs:
- `{"exists": true, "path": "p3/33autumn22/questions/q10.png", "sha256": "7aed56b1cf4ce1612c84f6b741d46b97ba3608a494de787138af91b25b5eb440"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33autumn22/mark_scheme/q10.png", "sha256": "fe3fdb597c0d9064437178c8dd3d3f0c1a7a32bb03ecd251433466781c0df179"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0002", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0003", "mark_code": "A1FT", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0004", "mark_code": "A1FT", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0008", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn22_q10_me0009", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 6. `31autumn21_q04` / `31autumn21_q04_whole`

- Selection category: `prior_ambiguous_retag_probe`
- Selection reason: Retest broad standard-integration routing where substitution, changed limits, and improper-limit structure need narrower treatment.
- Known risk flags: `broad_integration_label`, `retag_to_narrower_integration`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `whole_question_review_scope`, `mark_events_advisory_only`
- Review scope level: `whole_question`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31autumn21_q04:31autumn21_q04_whole`
- Question ID: `31autumn21_q04`
- Part/subpart: `whole` / `31autumn21_q04_whole`
- Paper/session/variant: `31autumn21` / `November` / `1`
- Candidate P3 skill IDs: `9709_p3_3_5_standard_integration`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `insufficient_part_signal`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_5_standard_integration`, `9709_p1_integration_reverse_differentiation`
- Primary candidate skill IDs: `9709_p3_3_5_standard_integration`
- Supporting candidate skill IDs: `9709_p1_integration_reverse_differentiation`
- Candidate region/topic: `{"mapping_source_topic": "integration", "subtopic_id": "9709_p3_subtopic_standard_integration", "subtopic_name": "Standard integration", "topic_assignment_id": "9709_p3_topic_integration", "topic_assignment_name": "Integration", "topic_routing_primary_topic_id": "9709_p3_topic_integration"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_integration", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_integration"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `reviewer_decide`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: integration.
Part-level decomposition candidates:
- None
- Content Lab blocker context: `{"candidate_id": "content_lab_31autumn21_q04_whole", "generation_gate_block_reasons": ["question_quality_gate_blocks_content_lab_generation", "mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 6}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `question_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `defer_ambiguous_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31autumn21/questions/q04.png", "sha256": "ad5427ac36cf4c78ff3cb2f8939815d36b9e38c43ea9f18cae27a6afdff967c1"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31autumn21/mark_scheme/q04.png", "sha256": "b907f807c2fc1c8435b226e171fc2256e4daf6cb327f81cd24d8c6e9e883b461"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31autumn21_q04_me0001", "mark_code": "B1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q04_me0002", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q04_me0003", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q04_me0004", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q04_me0005", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q04_me0006", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 7. `33autumn23_q07` / `33autumn23_q07_b`

- Selection category: `prior_blocked_confirmation`
- Selection reason: Confirm Batch 0001 blocked polynomial/remainder route on an implicit-differentiation stationary-tangent subpart.
- Known risk flags: `wrong_skill_routing`, `polynomial_support_only`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:33autumn23_q07:33autumn23_q07_b`
- Question ID: `33autumn23_q07`
- Part/subpart: `b` / `33autumn23_q07_b`
- Paper/session/variant: `33autumn23` / `November` / `3`
- Candidate P3 skill IDs: `9709_p3_3_1_polynomial_division_factor_remainder`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `topic_routing_candidate_mismatch`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_1_polynomial_division_factor_remainder`, `9709_p1_quadratics_discriminant_intersections`
- Primary candidate skill IDs: `9709_p3_3_1_polynomial_division_factor_remainder`
- Supporting candidate skill IDs: `9709_p1_quadratics_discriminant_intersections`
- Candidate region/topic: `{"mapping_source_topic": "polynomials", "subtopic_id": "9709_p3_subtopic_polynomial_division_factor_remainder", "subtopic_name": "Polynomial division, factors and remainders", "topic_assignment_id": "9709_p3_topic_algebra", "topic_assignment_name": "Algebra", "topic_routing_primary_topic_id": "9709_p3_topic_differentiation"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differentiation", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differentiation"]`
- Topic-routing alignment: `supporting_topic`
- Recommended scope: `subpart_level`
- Cross-topic notes: Candidate skill/topic and topic-routing context differ but may describe different stages of one solution.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: polynomials.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_1_polynomial_division_factor_remainder"], "candidate_topic_ids": ["9709_p3_topic_differentiation"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33autumn23_q07:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33autumn23_q07_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q07_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q07_me0007", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q07_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q07_me0009", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33autumn23_q07_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q07_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q07_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q07_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "33autumn23_q07", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33autumn23/mark_scheme/q07.png", "sha256": "e3f52fb16c21599508e8229f93b0e42d82cb21f725802ed1fe36129522fa0a99"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33autumn23/questions/q07.png", "sha256": "8aa237bc12120cdfed4bd7e9fde899b0abe39270eaa6bd413388ecf21785c864"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33autumn23_q07:33autumn23_q07_b", "supporting_skill_ids": ["9709_p1_quadratics_discriminant_intersections"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33autumn23_q07_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 5}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33autumn23/questions/q07.png", "sha256": "8aa237bc12120cdfed4bd7e9fde899b0abe39270eaa6bd413388ecf21785c864"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33autumn23/mark_scheme/q07.png", "sha256": "e3f52fb16c21599508e8229f93b0e42d82cb21f725802ed1fe36129522fa0a99"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0007", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q07_me0009", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 8. `31summer23_q02` / `31summer23_q02_a`

- Selection category: `prior_blocked_confirmation`
- Selection reason: Confirm Batch 0002 blocked parametric/implicit route on a modulus graph or linear-inequality subpart.
- Known risk flags: `wrong_skill_routing`, `parametric_implicit_absent`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31summer23_q02:31summer23_q02_a`
- Question ID: `31summer23_q02`
- Part/subpart: `a` / `31summer23_q02_a`
- Paper/session/variant: `31summer23` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_4_parametric_implicit_differentiation`
- Suggested candidate status: `conflict_candidate`
- Suggested review priority: `5_conflict_candidate`
- Suggested ambiguity reason: `weak_candidate_skill_context`
- Decomposition status: `conflict_needs_review`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_4_derivative_rules`
- Primary candidate skill IDs: `9709_p3_3_4_parametric_implicit_differentiation`
- Supporting candidate skill IDs: `9709_p3_3_4_derivative_rules`
- Candidate region/topic: `{"mapping_source_topic": "parametric_equations", "subtopic_id": "9709_p3_subtopic_parametric_implicit_differentiation", "subtopic_name": "Parametric and implicit differentiation", "topic_assignment_id": "9709_p3_topic_differentiation", "topic_assignment_name": "Differentiation", "topic_routing_primary_topic_id": ""}`
- Topic-routing context: `{"confidence": "low", "evidence_used": ["mark_scheme_text", "ocr_text"], "primary_topic_id": "", "review_reasons": ["schema_validation_error"], "review_required": true, "routing_source": "deepseek_topic_routing_error"}`
- Cross-topic status: `conflict_needs_review`
- Topic-routing topic IDs: `[]`
- Topic-routing alignment: `conflicting`
- Recommended scope: `reviewer_decide`
- Cross-topic notes: Only one side of candidate topic or topic-routing context is available.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: parametric_equations.; Method-critical mismatch flagged; do not treat this as ordinary supporting-topic context.
Part-level decomposition candidates:
- None
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer23_q02_a", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 1}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `question_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`, `weak_parametric_equation_evidence_missing_parameter`
- Reconciliation flags: none
- Recommended review action: `verify_parametric_equation_parameter`

Question asset refs:
- `{"exists": true, "path": "p3/31summer23/questions/q02.png", "sha256": "42aec17208c0f262fb00333135d667c13497000d96131fcc434cb2120a92055a"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer23/mark_scheme/q02.png", "sha256": "89931a8e0eb14c43b871a7a4129f3e29fa417579635f5be3f57056605f480804"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer23_q02_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0002", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0003", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0008", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0009", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q02_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 9. `31summer24_q06` / `31summer24_q06_c`

- Selection category: `thin_adjacent_part_probe`
- Selection reason: Retest thin fixed-point adjacent part against promoted part (d); exact but not enough evidence as a standalone source example.
- Known risk flags: `thin_adjacent_part`, `adjacent_part_contamination`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_c`
- Question ID: `31summer24_q06`
- Part/subpart: `c` / `31summer24_q06_c`
- Paper/session/variant: `31summer24` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`
- Primary candidate skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Supporting candidate skill IDs: `9709_p3_3_6_root_location`
- Candidate region/topic: `{"mapping_source_topic": "numerical_methods", "subtopic_id": "9709_p3_subtopic_fixed_point_iteration", "subtopic_name": "Fixed-point iteration", "topic_assignment_id": "9709_p3_topic_numerical_solution_of_equations", "topic_assignment_name": "Numerical solution of equations", "topic_routing_primary_topic_id": "9709_p3_topic_numerical_solution_of_equations"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "9709_p3_topic_numerical_solution_of_equations", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_numerical_solution_of_equations"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: numerical_methods.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"], "candidate_topic_ids": ["9709_p3_topic_numerical_solution_of_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31summer24_q06:c", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "31summer24_q06_me0005", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31summer24_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0006", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0007", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0008", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0009", "mark_code": "B1", "part_path": ["e"], "review_status": "advisory"}], "part_path": ["c"], "proposed_part_id": "c", "proposed_subpart_id": null, "question_id": "31summer24_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31summer24/mark_scheme/q06.png", "sha256": "bdfdda551e3b32bb16b4a617d5d58f3b72f15f983cb2788339561e35f8184d41"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31summer24/questions/q06.png", "sha256": "82fb5f1e01c9e0b9a5dc97c19b2ec40dcbbd26864be605a002ad2731dd28e00a"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_c", "supporting_skill_ids": ["9709_p3_3_6_root_location"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer24_q06_c", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 1}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer24/questions/q06.png", "sha256": "82fb5f1e01c9e0b9a5dc97c19b2ec40dcbbd26864be605a002ad2731dd28e00a"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer24/mark_scheme/q06.png", "sha256": "bdfdda551e3b32bb16b4a617d5d58f3b72f15f983cb2788339561e35f8184d41"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer24_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0005", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0006", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0007", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0008", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0009", "mark_code": "B1", "part_path": ["e"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 10. `31summer24_q09` / `31summer24_q09_a`

- Selection category: `thin_adjacent_part_probe`
- Selection reason: Retest thin vector adjacent part against promoted part (b); one-mark scalar-product evidence remains too thin.
- Known risk flags: `thin_adjacent_part`, `adjacent_part_contamination`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_a`
- Question ID: `31summer24_q09`
- Part/subpart: `a` / `31summer24_q09_a`
- Paper/session/variant: `31summer24` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_7_vector_lines`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_7_vector_lines`, `9709_p1_coordinate_line_geometry`
- Primary candidate skill IDs: `9709_p3_3_7_vector_lines`
- Supporting candidate skill IDs: `9709_p1_coordinate_line_geometry`
- Candidate region/topic: `{"mapping_source_topic": "vectors", "subtopic_id": "9709_p3_subtopic_vector_lines", "subtopic_name": "Vector equations of lines and intersections", "topic_assignment_id": "9709_p3_topic_vectors", "topic_assignment_name": "Vectors", "topic_routing_primary_topic_id": "9709_p3_topic_vectors"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_vectors", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_vectors"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: vectors.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_7_vector_lines"], "candidate_topic_ids": ["9709_p3_topic_vectors"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31summer24_q09:a", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "31summer24_q09_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31summer24_q09_me0002", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0006", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0007", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0008", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0009", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}], "part_path": ["a"], "proposed_part_id": "a", "proposed_subpart_id": null, "question_id": "31summer24_q09", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31summer24/mark_scheme/q09.png", "sha256": "3143cc68c0c31368f421d1b1bbea90fa5a784062fcf9c8f1e1b7a67adaeba1f7"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31summer24/questions/q09.png", "sha256": "e8d1b32432ca7c5c8d76096a33d6c139842e98e04e8edb4855c24203cb23a711"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_a", "supporting_skill_ids": ["9709_p1_coordinate_line_geometry"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer24_q09_a", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 1}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer24/questions/q09.png", "sha256": "e8d1b32432ca7c5c8d76096a33d6c139842e98e04e8edb4855c24203cb23a711"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer24/mark_scheme/q09.png", "sha256": "3143cc68c0c31368f421d1b1bbea90fa5a784062fcf9c8f1e1b7a67adaeba1f7"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer24_q09_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0002", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0006", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0007", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0008", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0009", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 11. `32spring24_q05` / `32spring24_q05_b`

- Selection category: `deferred_exact_skill_boundary_probe`
- Selection reason: Retest clean-looking complex evidence skipped in Batch 0002 because the narrower Argand/loci boundary was not resolved.
- Known risk flags: `narrower_skill_boundary`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32spring24_q05:32spring24_q05_b`
- Question ID: `32spring24_q05`
- Part/subpart: `b` / `32spring24_q05_b`
- Paper/session/variant: `32spring24` / `March` / `2`
- Candidate P3 skill IDs: `9709_p3_3_9_complex_arithmetic_polar_form`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_9_complex_arithmetic_polar_form`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_9_complex_arithmetic_polar_form`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "complex_numbers", "subtopic_id": "9709_p3_subtopic_complex_arithmetic_polar_form", "subtopic_name": "Complex arithmetic, modulus, argument and polar form", "topic_assignment_id": "9709_p3_topic_complex_numbers", "topic_assignment_name": "Complex numbers", "topic_routing_primary_topic_id": "9709_p3_topic_complex_numbers"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_complex_numbers", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_complex_numbers"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: complex_numbers.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32spring24_q05:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32spring24_q05_me0006", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring24_q05_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring24_q05_me0008", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32spring24_q05_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q05_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q05_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q05_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q05_me0005", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "32spring24_q05", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32spring24/mark_scheme/q05.png", "sha256": "56bd1e4d8f6186b2965e5c97d40a16ef9fd1fa133cbeef125243c1c901885d97"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32spring24/questions/q05.png", "sha256": "3be3316fd569b3d49be602e8dce04f2ce25c7511deefc332882ae00f84a9d359"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32spring24_q05:32spring24_q05_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring24_q05_b", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32spring24/questions/q05.png", "sha256": "3be3316fd569b3d49be602e8dce04f2ce25c7511deefc332882ae00f84a9d359"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32spring24/mark_scheme/q05.png", "sha256": "56bd1e4d8f6186b2965e5c97d40a16ef9fd1fa133cbeef125243c1c901885d97"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32spring24_q05_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q05_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q05_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q05_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q05_me0005", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q05_me0006", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q05_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q05_me0008", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 12. `33summer23_q11` / `33summer23_q11_b`

- Selection category: `clean_control_mark_event_probe`
- Selection reason: Already-promoted clean control: verify source-skill machinery still marks it safe while mark-event refs remain advisory-only.
- Known risk flags: `already_promoted_clean_control`, `mark_events_advisory_only`, `mark_event_approval_probe`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0001_seed:33summer23_q11:33summer23_q11_b`
- Queue ID: `p3_exact_skill_review_queue:v1:33summer23_q11:33summer23_q11_b`
- Question ID: `33summer23_q11`
- Part/subpart: `b` / `33summer23_q11_b`
- Paper/session/variant: `33summer23` / `June` / `3`
- Candidate P3 skill IDs: `9709_p3_3_9_complex_arithmetic_polar_form`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_9_complex_arithmetic_polar_form`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_9_complex_arithmetic_polar_form`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "complex_numbers", "subtopic_id": "9709_p3_subtopic_complex_arithmetic_polar_form", "subtopic_name": "Complex arithmetic, modulus, argument and polar form", "topic_assignment_id": "9709_p3_topic_complex_numbers", "topic_assignment_name": "Complex numbers", "topic_routing_primary_topic_id": "9709_p3_topic_complex_numbers"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text"], "primary_topic_id": "9709_p3_topic_complex_numbers", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_complex_numbers"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: complex_numbers.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33summer23_q11:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33summer23_q11_me0028", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0029", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0030", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0031", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33summer23_q11_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0002", "mark_code": "A3", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0004", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0005", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0006", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0007", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0008", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0009", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0010", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0011", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0012", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0013", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0014", "mark_code": "A2", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0015", "mark_code": "A3", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0016", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0017", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0018", "mark_code": "A2", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0019", "mark_code": "A3", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0020", "mark_code": "A2", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0021", "mark_code": "A3", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0022", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0023", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0024", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0025", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0026", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q11_me0027", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "33summer23_q11", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33summer23/mark_scheme/q11.png", "sha256": "3d9aef0a40d6a74a47948e1255762f5fbbcadb6cc649b4e95b06c74979451daf"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33summer23/questions/q11.png", "sha256": "a8efab1f28aabe6e55a5487a87b60b67ede6e3998c7dfbbc44e33a76e11f0bf3"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33summer23_q11:33summer23_q11_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33summer23_q11_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 4}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33summer23/questions/q11.png", "sha256": "a8efab1f28aabe6e55a5487a87b60b67ede6e3998c7dfbbc44e33a76e11f0bf3"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33summer23/mark_scheme/q11.png", "sha256": "3d9aef0a40d6a74a47948e1255762f5fbbcadb6cc649b4e95b06c74979451daf"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33summer23_q11_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0002", "mark_code": "A3", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0004", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0005", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0006", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0007", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0008", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0009", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0010", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0011", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0012", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0013", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0014", "mark_code": "A2", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0015", "mark_code": "A3", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0016", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0017", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0018", "mark_code": "A2", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0019", "mark_code": "A3", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0020", "mark_code": "A2", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0021", "mark_code": "A3", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0022", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0023", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0024", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0025", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0026", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0027", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0028", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0029", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0030", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q11_me0031", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 13. `31summer24_q06` / `31summer24_q06_d`

- Selection category: `clean_control_mark_event_probe`
- Selection reason: Already-promoted clean control: compare against thin adjacent part (c) and probe whether event-level approval exists.
- Known risk flags: `already_promoted_clean_control`, `mark_events_advisory_only`, `mark_event_approval_probe`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0002_seed:31summer24_q06:31summer24_q06_d`
- Queue ID: `p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_d`
- Question ID: `31summer24_q06`
- Part/subpart: `d` / `31summer24_q06_d`
- Paper/session/variant: `31summer24` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`
- Primary candidate skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Supporting candidate skill IDs: `9709_p3_3_6_root_location`
- Candidate region/topic: `{"mapping_source_topic": "numerical_methods", "subtopic_id": "9709_p3_subtopic_fixed_point_iteration", "subtopic_name": "Fixed-point iteration", "topic_assignment_id": "9709_p3_topic_numerical_solution_of_equations", "topic_assignment_name": "Numerical solution of equations", "topic_routing_primary_topic_id": "9709_p3_topic_numerical_solution_of_equations"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "9709_p3_topic_numerical_solution_of_equations", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_numerical_solution_of_equations"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: numerical_methods.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"], "candidate_topic_ids": ["9709_p3_topic_numerical_solution_of_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31summer24_q06:d", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "31summer24_q06_me0006", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0007", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0008", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31summer24_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0005", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "31summer24_q06_me0009", "mark_code": "B1", "part_path": ["e"], "review_status": "advisory"}], "part_path": ["d"], "proposed_part_id": "d", "proposed_subpart_id": null, "question_id": "31summer24_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31summer24/mark_scheme/q06.png", "sha256": "bdfdda551e3b32bb16b4a617d5d58f3b72f15f983cb2788339561e35f8184d41"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31summer24/questions/q06.png", "sha256": "82fb5f1e01c9e0b9a5dc97c19b2ec40dcbbd26864be605a002ad2731dd28e00a"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_d", "supporting_skill_ids": ["9709_p3_3_6_root_location"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer24_q06_d", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer24/questions/q06.png", "sha256": "82fb5f1e01c9e0b9a5dc97c19b2ec40dcbbd26864be605a002ad2731dd28e00a"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer24/mark_scheme/q06.png", "sha256": "bdfdda551e3b32bb16b4a617d5d58f3b72f15f983cb2788339561e35f8184d41"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer24_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0005", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0006", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0007", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0008", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q06_me0009", "mark_code": "B1", "part_path": ["e"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.

### 14. `31summer24_q09` / `31summer24_q09_b`

- Selection category: `clean_control_mark_event_probe`
- Selection reason: Already-promoted clean control: compare against thin adjacent part (a) and probe whether event-level approval exists.
- Known risk flags: `already_promoted_clean_control`, `mark_events_advisory_only`, `mark_event_approval_probe`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0002_seed:31summer24_q09:31summer24_q09_b`
- Queue ID: `p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_b`
- Question ID: `31summer24_q09`
- Part/subpart: `b` / `31summer24_q09_b`
- Paper/session/variant: `31summer24` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_7_vector_lines`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_7_vector_lines`, `9709_p1_coordinate_line_geometry`
- Primary candidate skill IDs: `9709_p3_3_7_vector_lines`
- Supporting candidate skill IDs: `9709_p1_coordinate_line_geometry`
- Candidate region/topic: `{"mapping_source_topic": "vectors", "subtopic_id": "9709_p3_subtopic_vector_lines", "subtopic_name": "Vector equations of lines and intersections", "topic_assignment_id": "9709_p3_topic_vectors", "topic_assignment_name": "Vectors", "topic_routing_primary_topic_id": "9709_p3_topic_vectors"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_vectors", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_vectors"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: vectors.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_7_vector_lines"], "candidate_topic_ids": ["9709_p3_topic_vectors"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31summer24_q09:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "31summer24_q09_me0002", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31summer24_q09_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0006", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0007", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0008", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31summer24_q09_me0009", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "31summer24_q09", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31summer24/mark_scheme/q09.png", "sha256": "3143cc68c0c31368f421d1b1bbea90fa5a784062fcf9c8f1e1b7a67adaeba1f7"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31summer24/questions/q09.png", "sha256": "e8d1b32432ca7c5c8d76096a33d6c139842e98e04e8edb4855c24203cb23a711"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_b", "supporting_skill_ids": ["9709_p1_coordinate_line_geometry"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer24_q09_b", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 4}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer24/questions/q09.png", "sha256": "e8d1b32432ca7c5c8d76096a33d6c139842e98e04e8edb4855c24203cb23a711"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer24/mark_scheme/q09.png", "sha256": "3143cc68c0c31368f421d1b1bbea90fa5a784062fcf9c8f1e1b7a67adaeba1f7"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer24_q09_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0002", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0006", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0007", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0008", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q09_me0009", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`

Reviewer checklist:
- [ ] Inspect the canonical question image.
- [ ] Inspect the canonical mark-scheme image.
- [ ] Confirm the exact P3 skill.
- [ ] Confirm whether whole-question or part-level scope is safe.
- [ ] Confirm whether P1 prerequisite/support-only material is involved.
- [ ] Confirm allowed use cases.
- [ ] Write evidence_basis in project wording.
- [ ] Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

Cross-topic reviewer checklist:
- [ ] Identify the main skill being assessed.
- [ ] Identify any supporting skills used in the method.
- [ ] Decide whether the current whole-question/part scope is safe.
- [ ] Split by part/subpart if the item tests multiple independent skills.
- [ ] Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.
- [ ] Do not use supporting skill context as mastery evidence unless reviewed directly.
