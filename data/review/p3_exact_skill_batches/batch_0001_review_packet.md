# P3 Exact-Skill Review Packet: batch_0001

This packet is for human review only. It does not assert clean evidence, does not update the reviewed-decision registry, and does not create the Asterion sidecar.

## Batch Metadata

- Generated at: `2026-05-24T23:30:17Z`
- Source queue: `reports/p3_exact_skill_review_queue.v1.json`
- Reviewed registry checked for exclusions: `data/review/p3_exact_skill_reviewed_decisions.v1.json`
- Selection status: `multiple`
- Included statuses: `clean_candidate, cross_topic_candidate`
- Excluded statuses: `conflict_candidate, fallback_only, ambiguous_candidate, blocked_candidate`
- Batch purpose: `exact_skill_review`
- Selection limit: `25`
- Selected items: `25`

## Reviewer Checklist

- Inspect the canonical question image.
- Inspect the canonical mark-scheme image.
- Confirm the exact P3 skill.
- Confirm whether whole-question or part-level scope is safe.
- Confirm whether P1 prerequisite/support-only material is involved.
- Confirm allowed use cases.
- Write evidence_basis in project wording.
- Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

> Mark-event refs are advisory-only review context. They are not authority for clean evidence, marking use, or candidate generation.

## Review Items

### 1. `33summer23_q11` / `33summer23_q11_b`

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

### 2. `33autumn21_q10` / `33autumn21_q10_c`

- Queue ID: `p3_exact_skill_review_queue:v1:33autumn21_q10:33autumn21_q10_c`
- Question ID: `33autumn21_q10`
- Part/subpart: `c` / `33autumn21_q10_c`
- Paper/session/variant: `33autumn21` / `November` / `3`
- Candidate P3 skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`
- Primary candidate skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Supporting candidate skill IDs: `9709_p3_3_6_root_location`
- Candidate region/topic: `{"mapping_source_topic": "numerical_methods", "subtopic_id": "9709_p3_subtopic_fixed_point_iteration", "subtopic_name": "Fixed-point iteration", "topic_assignment_id": "9709_p3_topic_numerical_solution_of_equations", "topic_assignment_name": "Numerical solution of equations", "topic_routing_primary_topic_id": "9709_p3_topic_differential_equations"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differential_equations", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differential_equations", "9709_p3_topic_numerical_solution_of_equations"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: numerical_methods.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"], "candidate_topic_ids": ["9709_p3_topic_differential_equations", "9709_p3_topic_numerical_solution_of_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33autumn21_q10:c", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33autumn21_q10_me0008", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0010", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33autumn21_q10_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0006", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn21_q10_me0011", "mark_code": "B1", "part_path": ["d"], "review_status": "advisory"}], "part_path": ["c"], "proposed_part_id": "c", "proposed_subpart_id": null, "question_id": "33autumn21_q10", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33autumn21/mark_scheme/q10.png", "sha256": "6ab9d16b944ba3a25a865758d81fc3aa9599102fa393f6294fa33abad791a148"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33autumn21/questions/q10.png", "sha256": "8fa117575334f51c1312580d312fb0551f7df77cc3d3b082382e8adc627fd3df"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33autumn21_q10:33autumn21_q10_c", "supporting_skill_ids": ["9709_p3_3_6_root_location"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33autumn21_q10_c", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33autumn21/questions/q10.png", "sha256": "8fa117575334f51c1312580d312fb0551f7df77cc3d3b082382e8adc627fd3df"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33autumn21/mark_scheme/q10.png", "sha256": "6ab9d16b944ba3a25a865758d81fc3aa9599102fa393f6294fa33abad791a148"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0006", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0008", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0010", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn21_q10_me0011", "mark_code": "B1", "part_path": ["d"], "review_status": "advisory"}`

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

### 3. `32autumn23_q09` / `32autumn23_q09_b`

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

### 4. `32spring23_q05` / `32spring23_q05_b`

- Queue ID: `p3_exact_skill_review_queue:v1:32spring23_q05:32spring23_q05_b`
- Question ID: `32spring23_q05`
- Part/subpart: `b` / `32spring23_q05_b`
- Paper/session/variant: `32spring23` / `March` / `2`
- Candidate P3 skill IDs: `9709_p3_3_4_parametric_implicit_differentiation`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_4_derivative_rules`
- Primary candidate skill IDs: `9709_p3_3_4_parametric_implicit_differentiation`
- Supporting candidate skill IDs: `9709_p3_3_4_derivative_rules`
- Candidate region/topic: `{"mapping_source_topic": "parametric_equations", "subtopic_id": "9709_p3_subtopic_parametric_implicit_differentiation", "subtopic_name": "Parametric and implicit differentiation", "topic_assignment_id": "9709_p3_topic_differentiation", "topic_assignment_name": "Differentiation", "topic_routing_primary_topic_id": "9709_p3_topic_differentiation"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differentiation", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differentiation"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: parametric_equations.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_4_parametric_implicit_differentiation"], "candidate_topic_ids": ["9709_p3_topic_differentiation"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32spring23_q05:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32spring23_q05_me0004", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring23_q05_me0005", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring23_q05_me0006", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring23_q05_me0007", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32spring23_q05_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q05_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q05_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "32spring23_q05", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32spring23/mark_scheme/q05.png", "sha256": "fe64329e305bd1560aa8acb873aae885e45ef6220c001e31327920bc57858df0"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32spring23/questions/q05.png", "sha256": "f01c011e876a81bafd3465e64a2a03120f3eb9d18aceb31e1fa6277dee9dfa05"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32spring23_q05:32spring23_q05_b", "supporting_skill_ids": ["9709_p3_3_4_derivative_rules"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring23_q05_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 4}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32spring23/questions/q05.png", "sha256": "f01c011e876a81bafd3465e64a2a03120f3eb9d18aceb31e1fa6277dee9dfa05"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32spring23/mark_scheme/q05.png", "sha256": "fe64329e305bd1560aa8acb873aae885e45ef6220c001e31327920bc57858df0"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32spring23_q05_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q05_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q05_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q05_me0004", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q05_me0005", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q05_me0006", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q05_me0007", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`

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

### 5. `33summer23_q09` / `33summer23_q09_b`

- Queue ID: `p3_exact_skill_review_queue:v1:33summer23_q09:33summer23_q09_b`
- Question ID: `33summer23_q09`
- Part/subpart: `b` / `33summer23_q09_b`
- Paper/session/variant: `33summer23` / `June` / `3`
- Candidate P3 skill IDs: `9709_p3_3_7_vector_lines`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_7_vector_lines`, `9709_p1_coordinate_line_geometry`
- Primary candidate skill IDs: `9709_p3_3_7_vector_lines`
- Supporting candidate skill IDs: `9709_p1_coordinate_line_geometry`
- Candidate region/topic: `{"mapping_source_topic": "vectors", "subtopic_id": "9709_p3_subtopic_vector_lines", "subtopic_name": "Vector equations of lines and intersections", "topic_assignment_id": "9709_p3_topic_vectors", "topic_assignment_name": "Vectors", "topic_routing_primary_topic_id": "9709_p3_topic_vectors"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text"], "primary_topic_id": "9709_p3_topic_vectors", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_vectors"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: vectors.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_7_vector_lines"], "candidate_topic_ids": ["9709_p3_topic_vectors"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33summer23_q09:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33summer23_q09_me0006", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0008", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0009", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0011", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0012", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0013", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0014", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0015", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0016", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0017", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0018", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0019", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0020", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0021", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0022", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33summer23_q09_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q09_me0005", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "33summer23_q09", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33summer23/mark_scheme/q09.png", "sha256": "989a429c7dcc1da0b6f346fc1f601127f7bbbd6a65ae6b9e931278a7946232fb"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33summer23/questions/q09.png", "sha256": "d7fdd8693551e08abc2c847c7f75c1a27cc52a3a79e2bec220b504752af4ff81"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33summer23_q09:33summer23_q09_b", "supporting_skill_ids": ["9709_p1_coordinate_line_geometry"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33summer23_q09_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 17}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33summer23/questions/q09.png", "sha256": "d7fdd8693551e08abc2c847c7f75c1a27cc52a3a79e2bec220b504752af4ff81"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33summer23/mark_scheme/q09.png", "sha256": "989a429c7dcc1da0b6f346fc1f601127f7bbbd6a65ae6b9e931278a7946232fb"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33summer23_q09_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0005", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0006", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0008", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0009", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0011", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0012", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0013", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0014", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0015", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0016", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0017", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0018", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0019", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0020", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0021", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q09_me0022", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 6. `32autumn21_q11` / `32autumn21_q11_c`

- Queue ID: `p3_exact_skill_review_queue:v1:32autumn21_q11:32autumn21_q11_c`
- Question ID: `32autumn21_q11`
- Part/subpart: `c` / `32autumn21_q11_c`
- Paper/session/variant: `32autumn21` / `November` / `2`
- Candidate P3 skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`
- Primary candidate skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Supporting candidate skill IDs: `9709_p3_3_6_root_location`
- Candidate region/topic: `{"mapping_source_topic": "numerical_methods", "subtopic_id": "9709_p3_subtopic_fixed_point_iteration", "subtopic_name": "Fixed-point iteration", "topic_assignment_id": "9709_p3_topic_numerical_solution_of_equations", "topic_assignment_name": "Numerical solution of equations", "topic_routing_primary_topic_id": "9709_p3_topic_differentiation"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differentiation", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differentiation", "9709_p3_topic_algebra", "9709_p3_topic_numerical_solution_of_equations"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: numerical_methods.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"], "candidate_topic_ids": ["9709_p3_topic_differentiation", "9709_p3_topic_algebra", "9709_p3_topic_numerical_solution_of_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32autumn21_q11:c", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32autumn21_q11_me0009", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0010", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0011", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32autumn21_q11_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0007", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn21_q11_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["c"], "proposed_part_id": "c", "proposed_subpart_id": null, "question_id": "32autumn21_q11", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32autumn21/mark_scheme/q11.png", "sha256": "da8841a9d8624fc178f2578920cabaf55a9cffcb55a50e45780b854cf2ddb40d"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32autumn21/questions/q11.png", "sha256": "0e1f9bf63d72f5532fb8e106649360ff911f60bb8adb6b77b5f3ccf2429ed8c8"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32autumn21_q11:32autumn21_q11_c", "supporting_skill_ids": ["9709_p3_3_6_root_location"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32autumn21_q11_c", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32autumn21/questions/q11.png", "sha256": "0e1f9bf63d72f5532fb8e106649360ff911f60bb8adb6b77b5f3ccf2429ed8c8"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32autumn21/mark_scheme/q11.png", "sha256": "da8841a9d8624fc178f2578920cabaf55a9cffcb55a50e45780b854cf2ddb40d"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0007", "mark_code": "DM1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0009", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0010", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q11_me0011", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`

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

### 7. `32summer23_q06` / `32summer23_q06_c`

- Queue ID: `p3_exact_skill_review_queue:v1:32summer23_q06:32summer23_q06_c`
- Question ID: `32summer23_q06`
- Part/subpart: `c` / `32summer23_q06_c`
- Paper/session/variant: `32summer23` / `June` / `2`
- Candidate P3 skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`
- Primary candidate skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Supporting candidate skill IDs: `9709_p3_3_6_root_location`
- Candidate region/topic: `{"mapping_source_topic": "numerical_methods", "subtopic_id": "9709_p3_subtopic_fixed_point_iteration", "subtopic_name": "Fixed-point iteration", "topic_assignment_id": "9709_p3_topic_numerical_solution_of_equations", "topic_assignment_name": "Numerical solution of equations", "topic_routing_primary_topic_id": "9709_p3_topic_numerical_solution_of_equations"}`
- Topic-routing context: `{"confidence": "medium", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "9709_p3_topic_numerical_solution_of_equations", "review_reasons": ["No OCR text provided, mark scheme only used; involves iterative method"], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_numerical_solution_of_equations"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: numerical_methods.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"], "candidate_topic_ids": ["9709_p3_topic_numerical_solution_of_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32summer23_q06:c", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32summer23_q06_me0007", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32summer23_q06_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32summer23_q06_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32summer23_q06_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer23_q06_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer23_q06_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer23_q06_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer23_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["c"], "proposed_part_id": "c", "proposed_subpart_id": null, "question_id": "32summer23_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32summer23/mark_scheme/q06.png", "sha256": "2b89e4606516c0bfeed84431dd0c75e13b227e63c9735e1a95106ffca95a5720"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32summer23/questions/q06.png", "sha256": "206bd704c709de7b3ef72d009f30d1c2d785dd42ff7e190c448be97431a98cf1"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32summer23_q06:32summer23_q06_c", "supporting_skill_ids": ["9709_p3_3_6_root_location"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32summer23_q06_c", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32summer23/questions/q06.png", "sha256": "206bd704c709de7b3ef72d009f30d1c2d785dd42ff7e190c448be97431a98cf1"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32summer23/mark_scheme/q06.png", "sha256": "2b89e4606516c0bfeed84431dd0c75e13b227e63c9735e1a95106ffca95a5720"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32summer23_q06_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q06_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q06_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q06_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q06_me0007", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q06_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer23_q06_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`

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

### 8. `33autumn23_q03` / `33autumn23_q03_whole`

- Queue ID: `p3_exact_skill_review_queue:v1:33autumn23_q03:33autumn23_q03_whole`
- Question ID: `33autumn23_q03`
- Part/subpart: `whole` / `33autumn23_q03_whole`
- Paper/session/variant: `33autumn23` / `November` / `3`
- Candidate P3 skill IDs: `9709_p3_3_1_polynomial_division_factor_remainder`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `insufficient_part_signal`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_1_polynomial_division_factor_remainder`, `9709_p1_quadratics_discriminant_intersections`
- Primary candidate skill IDs: `9709_p3_3_1_polynomial_division_factor_remainder`
- Supporting candidate skill IDs: `9709_p1_quadratics_discriminant_intersections`
- Candidate region/topic: `{"mapping_source_topic": "polynomials", "subtopic_id": "9709_p3_subtopic_polynomial_division_factor_remainder", "subtopic_name": "Polynomial division, factors and remainders", "topic_assignment_id": "9709_p3_topic_algebra", "topic_assignment_name": "Algebra", "topic_routing_primary_topic_id": "9709_p3_topic_algebra"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["question_text", "ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_algebra", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_algebra"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `reviewer_decide`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: polynomials.
Part-level decomposition candidates:
- None
- Content Lab blocker context: `{"candidate_id": "content_lab_33autumn23_q03_whole", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "allow"}, "source_mark_event_count": 11}`
- Proposed blockers: `mark_events_advisory_only`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33autumn23/questions/q03.png", "sha256": "3e610053bb74b9bdfe1739d34cc7dada0b22f840a1085b0da8c4c6e9c73f0dd7"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33autumn23/mark_scheme/q03.png", "sha256": "8db99bbf0140b32642fe6da8561285182cc8be9687eb4fc992f738c35bd8950c"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0001", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0002", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0003", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0004", "mark_code": "B2", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0005", "mark_code": "A2", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0006", "mark_code": "A2", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0007", "mark_code": "A2", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0008", "mark_code": "B2", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0009", "mark_code": "A4", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0010", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q03_me0011", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`

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

### 9. `32summer22_q10` / `32summer22_q10_d`

- Queue ID: `p3_exact_skill_review_queue:v1:32summer22_q10:32summer22_q10_d`
- Question ID: `32summer22_q10`
- Part/subpart: `d` / `32summer22_q10_d`
- Paper/session/variant: `32summer22` / `June` / `2`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32summer22_q10:d", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32summer22_q10_me0016", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0017", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32summer22_q10_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0002", "mark_code": "DM1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0004", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0005", "mark_code": "DM1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0006", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0007", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0008", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0009", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0011", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0012", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0013", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0014", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32summer22_q10_me0015", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}], "part_path": ["d"], "proposed_part_id": "d", "proposed_subpart_id": null, "question_id": "32summer22_q10", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32summer22/mark_scheme/q10.png", "sha256": "ed383fc7b7cad2d740efe151a1e1d3688fbe499ee7e8091a379a7b0457376b73"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32summer22/questions/q10.png", "sha256": "415dc6969e0cddd4a9dc6a9b24a14ed3766db8ec91dccc1b6a8415f001d39b9f"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32summer22_q10:32summer22_q10_d", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32summer22_q10_d", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 2}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32summer22/questions/q10.png", "sha256": "415dc6969e0cddd4a9dc6a9b24a14ed3766db8ec91dccc1b6a8415f001d39b9f"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32summer22/mark_scheme/q10.png", "sha256": "ed383fc7b7cad2d740efe151a1e1d3688fbe499ee7e8091a379a7b0457376b73"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32summer22_q10_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0002", "mark_code": "DM1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0004", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0005", "mark_code": "DM1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0006", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0007", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0008", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0009", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0011", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0012", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0013", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0014", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0015", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0016", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32summer22_q10_me0017", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`

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

### 10. `32autumn23_q06` / `32autumn23_q06_c`

- Queue ID: `p3_exact_skill_review_queue:v1:32autumn23_q06:32autumn23_q06_c`
- Question ID: `32autumn23_q06`
- Part/subpart: `c` / `32autumn23_q06_c`
- Paper/session/variant: `32autumn23` / `November` / `2`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"], "candidate_topic_ids": ["9709_p3_topic_numerical_solution_of_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32autumn23_q06:c", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32autumn23_q06_me0007", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32autumn23_q06_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32autumn23_q06_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32autumn23_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q06_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q06_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["c"], "proposed_part_id": "c", "proposed_subpart_id": null, "question_id": "32autumn23_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32autumn23/mark_scheme/q06.png", "sha256": "2ec28eece625619c2b90dea97529b36b581f3c974c33527385804c903a3f1fcc"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32autumn23/questions/q06.png", "sha256": "043cb0b4e96c54933f7d2345277756773f23b08db89d57b9b49116a57739bc25"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32autumn23_q06:32autumn23_q06_c", "supporting_skill_ids": ["9709_p3_3_6_root_location"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32autumn23_q06_c", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32autumn23/questions/q06.png", "sha256": "043cb0b4e96c54933f7d2345277756773f23b08db89d57b9b49116a57739bc25"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32autumn23/mark_scheme/q06.png", "sha256": "2ec28eece625619c2b90dea97529b36b581f3c974c33527385804c903a3f1fcc"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0007", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q06_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`

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

### 11. `33summer23_q04` / `33summer23_q04_whole`

- Queue ID: `p3_exact_skill_review_queue:v1:33summer23_q04:33summer23_q04_whole`
- Question ID: `33summer23_q04`
- Part/subpart: `whole` / `33summer23_q04_whole`
- Paper/session/variant: `33summer23` / `June` / `3`
- Candidate P3 skill IDs: `9709_p3_3_4_parametric_implicit_differentiation`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `insufficient_part_signal`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_4_derivative_rules`
- Primary candidate skill IDs: `9709_p3_3_4_parametric_implicit_differentiation`
- Supporting candidate skill IDs: `9709_p3_3_4_derivative_rules`
- Candidate region/topic: `{"mapping_source_topic": "parametric_equations", "subtopic_id": "9709_p3_subtopic_parametric_implicit_differentiation", "subtopic_name": "Parametric and implicit differentiation", "topic_assignment_id": "9709_p3_topic_differentiation", "topic_assignment_name": "Differentiation", "topic_routing_primary_topic_id": "9709_p3_topic_differentiation"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "9709_p3_topic_differentiation", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_differentiation"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `reviewer_decide`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: parametric_equations.
Part-level decomposition candidates:
- None
- Content Lab blocker context: `{"candidate_id": "content_lab_33summer23_q04_whole", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 8}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33summer23/questions/q04.png", "sha256": "d4f78dec267f1d42d92288f53c23ab50cc2f4c9503398e902ec4db35def1cc2a"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33summer23/mark_scheme/q04.png", "sha256": "c59ed425a9400a336f328c37b24d3cdb248781471db36bf3a24c64164d4697b6"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33summer23_q04_me0001", "mark_code": "B1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q04_me0002", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q04_me0003", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q04_me0004", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q04_me0005", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q04_me0006", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q04_me0007", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q04_me0008", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`

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

### 12. `31autumn21_q07` / `31autumn21_q07_c`

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

### 13. `32summer23_q07` / `32summer23_q07_b`

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

### 14. `31summer24_q01` / `31summer24_q01_whole`

- Queue ID: `p3_exact_skill_review_queue:v1:31summer24_q01:31summer24_q01_whole`
- Question ID: `31summer24_q01`
- Part/subpart: `whole` / `31summer24_q01_whole`
- Paper/session/variant: `31summer24` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_1_binomial_rational_expansion`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `insufficient_part_signal`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_1_binomial_rational_expansion`, `9709_p1_series_binomial_positive_integer`
- Primary candidate skill IDs: `9709_p3_3_1_binomial_rational_expansion`
- Supporting candidate skill IDs: `9709_p1_series_binomial_positive_integer`
- Candidate region/topic: `{"mapping_source_topic": "binomial_expansion", "subtopic_id": "9709_p3_subtopic_binomial_rational_expansion", "subtopic_name": "Binomial expansion for rational powers", "topic_assignment_id": "9709_p3_topic_algebra", "topic_assignment_name": "Algebra", "topic_routing_primary_topic_id": "9709_p3_topic_algebra"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["question_text", "ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_algebra", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_algebra"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `reviewer_decide`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: binomial_expansion.
Part-level decomposition candidates:
- None
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer24_q01_whole", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "allow"}, "source_mark_event_count": 4}`
- Proposed blockers: `mark_events_advisory_only`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer24/questions/q01.png", "sha256": "f35aa464001bc18fec1fad5ac821756363f9ea4916bd8b0d4b25877b851b8baa"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer24/mark_scheme/q01.png", "sha256": "6e8d66300a20d3a3831fc9b4b41370b867eecb3e8e000a5bd93786e9ab351240"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer24_q01_me0001", "mark_code": "B1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q01_me0002", "mark_code": "B1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q01_me0003", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q01_me0004", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`

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

### 15. `31autumn23_q08` / `31autumn23_q08_d`

- Queue ID: `p3_exact_skill_review_queue:v1:31autumn23_q08:31autumn23_q08_d`
- Question ID: `31autumn23_q08`
- Part/subpart: `d` / `31autumn23_q08_d`
- Paper/session/variant: `31autumn23` / `November` / `1`
- Candidate P3 skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`
- Primary candidate skill IDs: `9709_p3_3_6_fixed_point_iteration`
- Supporting candidate skill IDs: `9709_p3_3_6_root_location`
- Candidate region/topic: `{"mapping_source_topic": "numerical_methods", "subtopic_id": "9709_p3_subtopic_fixed_point_iteration", "subtopic_name": "Fixed-point iteration", "topic_assignment_id": "9709_p3_topic_numerical_solution_of_equations", "topic_assignment_name": "Numerical solution of equations", "topic_routing_primary_topic_id": ""}`
- Topic-routing context: `{"confidence": "low", "evidence_used": ["mark_scheme_text", "ocr_text"], "primary_topic_id": "", "review_reasons": ["schema_validation_error"], "review_required": true, "routing_source": "deepseek_topic_routing_error"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `[]`
- Topic-routing alignment: `unknown`
- Recommended scope: `subpart_level`
- Cross-topic notes: Only one side of candidate topic or topic-routing context is available.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: numerical_methods.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"], "candidate_topic_ids": [], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31autumn23_q08:d", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": false}, "matching_mark_event_refs": [{"event_id": "31autumn23_q08_me0007", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31autumn23_q08_me0008", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}, {"event_id": "31autumn23_q08_me0009", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31autumn23_q08_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31autumn23_q08_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31autumn23_q08_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31autumn23_q08_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31autumn23_q08_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31autumn23_q08_me0006", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}], "part_path": ["d"], "proposed_part_id": "d", "proposed_subpart_id": null, "question_id": "31autumn23_q08", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31autumn23/mark_scheme/q08.png", "sha256": "f867fd427fc822c1ab0e90ab70ac7e88b1bb7ffcbfeac53a5210e60ac13c6f20"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31autumn23/questions/q08.png", "sha256": "5203536a27e92b2a888f535ac17401acabb9fe986be08f2783284ba2e76fa296"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31autumn23_q08:31autumn23_q08_d", "supporting_skill_ids": ["9709_p3_3_6_root_location"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31autumn23_q08_d", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `defer_ambiguous_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31autumn23/questions/q08.png", "sha256": "5203536a27e92b2a888f535ac17401acabb9fe986be08f2783284ba2e76fa296"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31autumn23/mark_scheme/q08.png", "sha256": "f867fd427fc822c1ab0e90ab70ac7e88b1bb7ffcbfeac53a5210e60ac13c6f20"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0006", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0007", "mark_code": "M1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0008", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q08_me0009", "mark_code": "A1", "part_path": ["d"], "review_status": "advisory"}`

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

### 16. `33summer23_q06` / `33summer23_q06_b`

- Queue ID: `p3_exact_skill_review_queue:v1:33summer23_q06:33summer23_q06_b`
- Question ID: `33summer23_q06`
- Part/subpart: `b` / `33summer23_q06_b`
- Paper/session/variant: `33summer23` / `June` / `3`
- Candidate P3 skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "trigonometry", "subtopic_id": "9709_p3_subtopic_identities_compound_double_angle_equations", "subtopic_name": "Identities, compound angles and trigonometric equations", "topic_assignment_id": "9709_p3_topic_trigonometry", "topic_assignment_name": "Trigonometry", "topic_routing_primary_topic_id": "9709_p3_topic_trigonometry"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text"], "primary_topic_id": "9709_p3_topic_trigonometry", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_trigonometry"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: trigonometry.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_3_identities_compound_double_angle_equations"], "candidate_topic_ids": ["9709_p3_topic_trigonometry"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33summer23_q06:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33summer23_q06_me0006", "mark_code": "B1FT", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0009", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0010", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0011", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33summer23_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0002", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0004", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0005", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "33summer23_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33summer23/mark_scheme/q06.png", "sha256": "c75546cd79657e209d4b54179f4ff50f38457d35b564cadb2b4516bbfd21f0fa"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33summer23/questions/q06.png", "sha256": "f7b0303d29c351ea60598e68150b6c2a2862b4834a3e8a5829c587bbeae02287"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33summer23_q06:33summer23_q06_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33summer23_q06_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 6}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33summer23/questions/q06.png", "sha256": "f7b0303d29c351ea60598e68150b6c2a2862b4834a3e8a5829c587bbeae02287"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33summer23/mark_scheme/q06.png", "sha256": "c75546cd79657e209d4b54179f4ff50f38457d35b564cadb2b4516bbfd21f0fa"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33summer23_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0002", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0004", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0005", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0006", "mark_code": "B1FT", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0009", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0010", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer23_q06_me0011", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 17. `33autumn23_q07` / `33autumn23_q07_b`

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

### 18. `32spring23_q10` / `32spring23_q10_c`

- Queue ID: `p3_exact_skill_review_queue:v1:32spring23_q10:32spring23_q10_c`
- Question ID: `32spring23_q10`
- Part/subpart: `c` / `32spring23_q10_c`
- Paper/session/variant: `32spring23` / `March` / `2`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_7_vector_lines"], "candidate_topic_ids": ["9709_p3_topic_vectors"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32spring23_q10:c", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32spring23_q10_me0006", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0007", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32spring23_q10_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["c"], "proposed_part_id": "c", "proposed_subpart_id": null, "question_id": "32spring23_q10", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32spring23/mark_scheme/q10.png", "sha256": "35155bc3e80c0f15956cf35b03b56b738e74a03a39f4add742b2758adc7ca25e"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32spring23/questions/q10.png", "sha256": "e3f3bbae28ab21ddc5453d65ba87a49b64b1bac169ef5e118f06008e31132040"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32spring23_q10:32spring23_q10_c", "supporting_skill_ids": ["9709_p1_coordinate_line_geometry"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring23_q10_c", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 4}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32spring23/questions/q10.png", "sha256": "e3f3bbae28ab21ddc5453d65ba87a49b64b1bac169ef5e118f06008e31132040"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32spring23/mark_scheme/q10.png", "sha256": "35155bc3e80c0f15956cf35b03b56b738e74a03a39f4add742b2758adc7ca25e"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32spring23_q10_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q10_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q10_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q10_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q10_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q10_me0006", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q10_me0007", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q10_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q10_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`

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

### 19. `33autumn23_q06` / `33autumn23_q06_b`

- Queue ID: `p3_exact_skill_review_queue:v1:33autumn23_q06:33autumn23_q06_b`
- Question ID: `33autumn23_q06`
- Part/subpart: `b` / `33autumn23_q06_b`
- Paper/session/variant: `33autumn23` / `November` / `3`
- Candidate P3 skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "trigonometry", "subtopic_id": "9709_p3_subtopic_identities_compound_double_angle_equations", "subtopic_name": "Identities, compound angles and trigonometric equations", "topic_assignment_id": "9709_p3_topic_trigonometry", "topic_assignment_name": "Trigonometry", "topic_routing_primary_topic_id": "9709_p3_topic_trigonometry"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "9709_p3_topic_trigonometry", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_trigonometry"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: trigonometry.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_3_identities_compound_double_angle_equations"], "candidate_topic_ids": ["9709_p3_topic_trigonometry"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33autumn23_q06:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33autumn23_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33autumn23_q06_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "33autumn23_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33autumn23/mark_scheme/q06.png", "sha256": "e76fca79346f0e4a08a6e64d9af661a1b74cf178d41edf3f20f796f889e928c8"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33autumn23/questions/q06.png", "sha256": "2e16cf3fcb03cac37635d4e5cce9219699bb03f82a230c454976719defe69c23"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33autumn23_q06:33autumn23_q06_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33autumn23_q06_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 4}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33autumn23/questions/q06.png", "sha256": "2e16cf3fcb03cac37635d4e5cce9219699bb03f82a230c454976719defe69c23"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33autumn23/mark_scheme/q06.png", "sha256": "e76fca79346f0e4a08a6e64d9af661a1b74cf178d41edf3f20f796f889e928c8"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33autumn23_q06_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q06_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q06_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q06_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q06_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 20. `33autumn23_q01` / `33autumn23_q01_whole`

- Queue ID: `p3_exact_skill_review_queue:v1:33autumn23_q01:33autumn23_q01_whole`
- Question ID: `33autumn23_q01`
- Part/subpart: `whole` / `33autumn23_q01_whole`
- Paper/session/variant: `33autumn23` / `November` / `3`
- Candidate P3 skill IDs: `9709_p3_3_2_log_exponential_equations`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `insufficient_part_signal`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_2_log_exponential_equations`, `9709_p1_functions_domain_range_one_one`
- Primary candidate skill IDs: `9709_p3_3_2_log_exponential_equations`
- Supporting candidate skill IDs: `9709_p1_functions_domain_range_one_one`
- Candidate region/topic: `{"mapping_source_topic": "logarithms_and_exponentials", "subtopic_id": "9709_p3_subtopic_log_exponential_equations", "subtopic_name": "Logarithmic and exponential equations", "topic_assignment_id": "9709_p3_topic_logarithmic_and_exponential_functions", "topic_assignment_name": "Logarithmic and exponential functions", "topic_routing_primary_topic_id": "9709_p3_topic_logarithmic_and_exponential_functions"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_logarithmic_and_exponential_functions", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_logarithmic_and_exponential_functions"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `reviewer_decide`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: logarithms_and_exponentials.
Part-level decomposition candidates:
- None
- Content Lab blocker context: `{"candidate_id": "content_lab_33autumn23_q01_whole", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 8}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33autumn23/questions/q01.png", "sha256": "65fca41d1592fd29bcb866912a68cb6712cc69f428cc4814a4952b67cbba5619"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33autumn23/mark_scheme/q01.png", "sha256": "49c2957fd30c93274422dae5834e65c047053071881ae4c020691757f80ef644"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33autumn23_q01_me0001", "mark_code": "B1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q01_me0002", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q01_me0003", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q01_me0004", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q01_me0005", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q01_me0006", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q01_me0007", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33autumn23_q01_me0008", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`

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

### 21. `32autumn23_q04` / `32autumn23_q04_b`

- Queue ID: `p3_exact_skill_review_queue:v1:32autumn23_q04:32autumn23_q04_b`
- Question ID: `32autumn23_q04`
- Part/subpart: `b` / `32autumn23_q04_b`
- Paper/session/variant: `32autumn23` / `November` / `2`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32autumn23_q04:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32autumn23_q04_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q04_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q04_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn23_q04_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32autumn23_q04_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q04_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q04_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn23_q04_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "32autumn23_q04", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32autumn23/mark_scheme/q04.png", "sha256": "07f1e8ccdf4ede398f6b7d141be0f1087989ac748463e9c90e3fb938abe08cbc"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32autumn23/questions/q04.png", "sha256": "d5f82ef5fd8b4435c671b744a4c4c3c464eaedfacd25bf0271779fa8d6fa2939"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32autumn23_q04:32autumn23_q04_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32autumn23_q04_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 4}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32autumn23/questions/q04.png", "sha256": "d5f82ef5fd8b4435c671b744a4c4c3c464eaedfacd25bf0271779fa8d6fa2939"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32autumn23/mark_scheme/q04.png", "sha256": "07f1e8ccdf4ede398f6b7d141be0f1087989ac748463e9c90e3fb938abe08cbc"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32autumn23_q04_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q04_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q04_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q04_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q04_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q04_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q04_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q04_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 22. `32spring23_q02` / `32spring23_q02_b`

- Queue ID: `p3_exact_skill_review_queue:v1:32spring23_q02:32spring23_q02_b`
- Question ID: `32spring23_q02`
- Part/subpart: `b` / `32spring23_q02_b`
- Paper/session/variant: `32spring23` / `March` / `2`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32spring23_q02:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32spring23_q02_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring23_q02_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring23_q02_me0007", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32spring23_q02_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q02_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q02_me0003", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q02_me0004", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "32spring23_q02", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32spring23/mark_scheme/q02.png", "sha256": "0329b3e0a8f8268b80fd21eb373590d25812d2a601cf17ad036d1b68563643e5"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32spring23/questions/q02.png", "sha256": "7d7fb67b9ba329e4e7f28b547d2c5d478b3a984ae5dbb41c23e0d327f4d94957"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32spring23_q02:32spring23_q02_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring23_q02_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32spring23/questions/q02.png", "sha256": "7d7fb67b9ba329e4e7f28b547d2c5d478b3a984ae5dbb41c23e0d327f4d94957"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32spring23/mark_scheme/q02.png", "sha256": "0329b3e0a8f8268b80fd21eb373590d25812d2a601cf17ad036d1b68563643e5"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32spring23_q02_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q02_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q02_me0003", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q02_me0004", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q02_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q02_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q02_me0007", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`

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

### 23. `31summer23_q04` / `31summer23_q04_b`

- Queue ID: `p3_exact_skill_review_queue:v1:31summer23_q04:31summer23_q04_b`
- Question ID: `31summer23_q04`
- Part/subpart: `b` / `31summer23_q04_b`
- Paper/session/variant: `31summer23` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "trigonometry", "subtopic_id": "9709_p3_subtopic_identities_compound_double_angle_equations", "subtopic_name": "Identities, compound angles and trigonometric equations", "topic_assignment_id": "9709_p3_topic_trigonometry", "topic_assignment_name": "Trigonometry", "topic_routing_primary_topic_id": ""}`
- Topic-routing context: `{"confidence": "low", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "", "review_reasons": ["schema_validation_error"], "review_required": true, "routing_source": "deepseek_topic_routing_error"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `[]`
- Topic-routing alignment: `unknown`
- Recommended scope: `subpart_level`
- Cross-topic notes: Only one side of candidate topic or topic-routing context is available.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: trigonometry.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_3_identities_compound_double_angle_equations"], "candidate_topic_ids": [], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31summer23_q04:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": false}, "matching_mark_event_refs": [{"event_id": "31summer23_q04_me0003", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0007", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0008", "mark_code": "A3", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0009", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0011", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0012", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0013", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0014", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0015", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31summer23_q04_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer23_q04_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "31summer23_q04", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31summer23/mark_scheme/q04.png", "sha256": "5b92dc93a3c74bcca8c309f0a740921cbe681fa487058e4a37e6dcef3daf1ca2"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31summer23/questions/q04.png", "sha256": "7932906afdef2ee454edb27a47ad1b3c16451689dd7bc1726af5c85cc4d11668"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31summer23_q04:31summer23_q04_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer23_q04_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 13}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `defer_ambiguous_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer23/questions/q04.png", "sha256": "7932906afdef2ee454edb27a47ad1b3c16451689dd7bc1726af5c85cc4d11668"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer23/mark_scheme/q04.png", "sha256": "5b92dc93a3c74bcca8c309f0a740921cbe681fa487058e4a37e6dcef3daf1ca2"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer23_q04_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0003", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0007", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0008", "mark_code": "A3", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0009", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0010", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0011", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0012", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0013", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0014", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer23_q04_me0015", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 24. `31summer24_q07` / `31summer24_q07_b`

- Queue ID: `p3_exact_skill_review_queue:v1:31summer24_q07:31summer24_q07_b`
- Question ID: `31summer24_q07`
- Part/subpart: `b` / `31summer24_q07_b`
- Paper/session/variant: `31summer24` / `June` / `1`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31summer24_q07:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "31summer24_q07_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q07_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31summer24_q07_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q07_me0002", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q07_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q07_me0004", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "31summer24_q07", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31summer24/mark_scheme/q07.png", "sha256": "b5befbb3e3dc2de11155cee164849c3122a400e64a0766353503f471e8823262"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31summer24/questions/q07.png", "sha256": "ddbe73ff02bfd9c7c14c95a9e81f849525610a775c469dcf5bbe0f34a71aec9f"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31summer24_q07:31summer24_q07_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer24_q07_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 2}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer24/questions/q07.png", "sha256": "ddbe73ff02bfd9c7c14c95a9e81f849525610a775c469dcf5bbe0f34a71aec9f"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer24/mark_scheme/q07.png", "sha256": "b5befbb3e3dc2de11155cee164849c3122a400e64a0766353503f471e8823262"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer24_q07_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q07_me0002", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q07_me0003", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q07_me0004", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q07_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q07_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 25. `31summer24_q04` / `31summer24_q04_b`

- Queue ID: `p3_exact_skill_review_queue:v1:31summer24_q04:31summer24_q04_b`
- Question ID: `31summer24_q04`
- Part/subpart: `b` / `31summer24_q04_b`
- Paper/session/variant: `31summer24` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_9_complex_arithmetic_polar_form`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_9_complex_arithmetic_polar_form`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_9_complex_arithmetic_polar_form`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "complex_numbers", "subtopic_id": "9709_p3_subtopic_complex_arithmetic_polar_form", "subtopic_name": "Complex arithmetic, modulus, argument and polar form", "topic_assignment_id": "9709_p3_topic_complex_numbers", "topic_assignment_name": "Complex numbers", "topic_routing_primary_topic_id": "9709_p3_topic_complex_numbers"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "9709_p3_topic_complex_numbers", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_complex_numbers"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: complex_numbers.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31summer24_q04:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "31summer24_q04_me0003", "mark_code": "B1FT", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer24_q04_me0004", "mark_code": "B1FT", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31summer24_q04_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer24_q04_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "31summer24_q04", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31summer24/mark_scheme/q04.png", "sha256": "313c4569535f9715209434b86c896cccc13a6561b88da1fe88c38a2ec149fda2"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31summer24/questions/q04.png", "sha256": "45b9960f8e8233dc6d8044de904914d246113f08b97fea5346753b5df0cd98c7"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31summer24_q04:31summer24_q04_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer24_q04_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 2}`
- Proposed blockers: `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer24/questions/q04.png", "sha256": "45b9960f8e8233dc6d8044de904914d246113f08b97fea5346753b5df0cd98c7"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer24/mark_scheme/q04.png", "sha256": "313c4569535f9715209434b86c896cccc13a6561b88da1fe88c38a2ec149fda2"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer24_q04_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q04_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q04_me0003", "mark_code": "B1FT", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer24_q04_me0004", "mark_code": "B1FT", "part_path": ["b"], "review_status": "advisory"}`

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
