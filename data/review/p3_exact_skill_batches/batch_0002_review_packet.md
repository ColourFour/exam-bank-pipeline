# P3 Exact-Skill Review Packet: batch_0002

This packet is for human review only. It does not assert clean evidence, does not update the reviewed-decision registry, and does not create the Asterion sidecar.

## Batch Metadata

- Generated at: `2026-05-26T00:55:02Z`
- Source queue: `reports/p3_exact_skill_review_queue.v1.json`
- Reviewed registry checked for exclusions: `data/review/p3_exact_skill_reviewed_decisions.v1.json`
- Selection status: `multiple`
- Included statuses: `cross_topic_candidate, conflict_candidate, split_needed_candidate, fallback_only`
- Excluded statuses: `none`
- Batch purpose: `batch_0002_mixed_manual_review`
- Selection limit: `40`
- Selected items: `37`

## Reviewer Checklist

- Inspect the canonical question image.
- Inspect the canonical mark-scheme image.
- Confirm the exact P3 skill.
- Confirm whether whole-question or part-level scope is safe.
- Confirm whether P1 prerequisite/support-only material is involved.
- Confirm allowed use cases.
- Write evidence_basis in project wording.
- Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.

## Batch 0002 Review Instructions

- Distinguish target skill from supporting method. Supporting algebra, trigonometry, or differentiation is not clean source-skill evidence unless the image evidence shows it is the assessed target.
- Distinguish whole-question evidence from part-level evidence. Whole-question images are canonical, but a clean decision may still need a part-level scope.
- Distinguish source-skill review from mark-event review. Reviewed source-skill evidence does not approve mark events.
- Treat advisory text, OCR, topic routing, and mark-event refs as context. Canonical question and mark-scheme images are the source of truth.
- Use clean only when the exact skill, scope, and evidence basis are clear. Use ambiguous when target-vs-support or part scope remains uncertain.
- Reviewed source-skill evidence is not generation readiness. Leave candidate_generation false unless a later explicit promotion pass changes it.

> Mark-event refs are advisory-only review context. They are not authority for clean evidence, marking use, or candidate generation.

## Review Items

### 1. `33summer23_q11` / `33summer23_q11_b`

- Selection category: `seed_mark_event_alignment_probe`
- Selection reason: Seed registry source-skill evidence exists; review whether advisory mark-event refs can be paired safely without changing generation readiness.
- Known risk flags: `reviewed_source_skill_exists`, `mark_events_advisory_only`, `part_level_uses_whole_question_images`, `part_level_scope_uses_whole_images`
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

### 2. `31summer24_q04` / `31summer24_q04_b`

- Selection category: `seed_mark_event_alignment_probe`
- Selection reason: Seed registry source-skill evidence exists; review whether advisory mark-event refs can be paired safely without changing generation readiness.
- Known risk flags: `reviewed_source_skill_exists`, `mark_events_advisory_only`, `part_level_uses_whole_question_images`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0001_seed:31summer24_q04:31summer24_q04_b`
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

### 3. `32summer23_q06` / `32summer23_q06_c`

- Selection category: `seed_mark_event_alignment_probe`
- Selection reason: Seed registry source-skill evidence exists; review whether advisory mark-event refs can be paired safely without changing generation readiness.
- Known risk flags: `reviewed_source_skill_exists`, `mark_events_advisory_only`, `part_level_uses_whole_question_images`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0001_seed:32summer23_q06:32summer23_q06_c`
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

### 4. `32autumn23_q06` / `32autumn23_q06_c`

- Selection category: `seed_mark_event_alignment_probe`
- Selection reason: Seed registry source-skill evidence exists; review whether advisory mark-event refs can be paired safely without changing generation readiness.
- Known risk flags: `reviewed_source_skill_exists`, `mark_events_advisory_only`, `part_level_uses_whole_question_images`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0001_seed:32autumn23_q06:32autumn23_q06_c`
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

### 5. `33summer23_q06` / `33summer23_q06_b`

- Selection category: `seed_mark_event_alignment_probe`
- Selection reason: Seed registry source-skill evidence exists; review whether advisory mark-event refs can be paired safely without changing generation readiness.
- Known risk flags: `reviewed_source_skill_exists`, `mark_events_advisory_only`, `part_level_uses_whole_question_images`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0001_seed:33summer23_q06:33summer23_q06_b`
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

### 6. `33summer23_q09` / `33summer23_q09_b`

- Selection category: `seed_mark_event_alignment_probe`
- Selection reason: Seed registry source-skill evidence exists; review whether advisory mark-event refs can be paired safely without changing generation readiness.
- Known risk flags: `reviewed_source_skill_exists`, `mark_events_advisory_only`, `part_level_uses_whole_question_images`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0001_seed:33summer23_q09:33summer23_q09_b`
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

### 7. `32spring23_q05` / `32spring23_q05_b`

- Selection category: `seed_mark_event_alignment_probe`
- Selection reason: Seed registry source-skill evidence exists; review whether advisory mark-event refs can be paired safely without changing generation readiness.
- Known risk flags: `reviewed_source_skill_exists`, `mark_events_advisory_only`, `part_level_uses_whole_question_images`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `True`
- Related reviewed evidence IDs: `p3_exact_skill_review:batch_0001_seed:32spring23_q05:32spring23_q05_b`
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

### 8. `31summer24_q01` / `31summer24_q01_whole`

- Selection category: `deferred_batch_0001_clean`
- Selection reason: Batch 0001 clean draft was intentionally deferred during seed promotion: clean_whole_question_candidate_deferred_for_later_pass.
- Known risk flags: `batch_0001_clean_draft_not_promoted`, `mark_events_advisory_only`, `whole_question_scope_requires_part_boundary_check`, `whole_question_review_scope`
- Review scope level: `whole_question`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
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

### 9. `33autumn23_q01` / `33autumn23_q01_whole`

- Selection category: `deferred_batch_0001_clean`
- Selection reason: Batch 0001 clean draft was intentionally deferred during seed promotion: clean_whole_question_candidate_deferred_for_later_pass.
- Known risk flags: `batch_0001_clean_draft_not_promoted`, `mark_events_advisory_only`, `whole_question_scope_requires_part_boundary_check`, `whole_question_review_scope`
- Review scope level: `whole_question`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
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

### 10. `33autumn23_q03` / `33autumn23_q03_whole`

- Selection category: `deferred_batch_0001_clean`
- Selection reason: Batch 0001 clean draft was intentionally deferred during seed promotion: clean_whole_question_candidate_deferred_for_later_pass.
- Known risk flags: `batch_0001_clean_draft_not_promoted`, `mark_events_advisory_only`, `whole_question_scope_requires_part_boundary_check`, `whole_question_review_scope`
- Review scope level: `whole_question`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
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

### 11. `33summer23_q04` / `33summer23_q04_whole`

- Selection category: `deferred_batch_0001_clean`
- Selection reason: Batch 0001 clean draft was intentionally deferred during seed promotion: clean_whole_question_candidate_deferred_for_later_pass.
- Known risk flags: `batch_0001_clean_draft_not_promoted`, `mark_events_advisory_only`, `whole_question_scope_requires_part_boundary_check`, `whole_question_review_scope`
- Review scope level: `whole_question`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
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

### 12. `31autumn23_q08` / `31autumn23_q08_d`

- Selection category: `deferred_batch_0001_clean`
- Selection reason: Batch 0001 clean draft was intentionally deferred during seed promotion: clean_but_topic_alignment_unknown_deferred.
- Known risk flags: `batch_0001_clean_draft_not_promoted`, `mark_events_advisory_only`, `unknown_topic_alignment`, `mixed_or_ambiguous_topic`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
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

### 13. `31summer23_q04` / `31summer23_q04_b`

- Selection category: `deferred_batch_0001_clean`
- Selection reason: Batch 0001 clean draft was intentionally deferred during seed promotion: clean_but_topic_alignment_unknown_deferred.
- Known risk flags: `batch_0001_clean_draft_not_promoted`, `mark_events_advisory_only`, `unknown_topic_alignment`, `mixed_or_ambiguous_topic`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
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

### 14. `32autumn23_q09` / `32autumn23_q09_b`

- Selection category: `known_failure_mode_probe`
- Selection reason: Trigonometry appears inside integration or area work; verify it is not only supporting method context.
- Known risk flags: `trig_supporting_integration`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
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

### 15. `31summer21_q04` / `31summer21_q04_a`

- Selection category: `known_failure_mode_probe`
- Selection reason: Trigonometry appears inside integration or area work; verify it is not only supporting method context.
- Known risk flags: `trig_supporting_integration`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31summer21_q04:31summer21_q04_a`
- Question ID: `31summer21_q04`
- Part/subpart: `a` / `31summer21_q04_a`
- Paper/session/variant: `31summer21` / `June` / `1`
- Candidate P3 skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `topic_routing_candidate_mismatch`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "integration", "subtopic_id": "9709_p3_subtopic_identities_compound_double_angle_equations", "subtopic_name": "Identities, compound angles and trigonometric equations", "topic_assignment_id": "9709_p3_topic_trigonometry", "topic_assignment_name": "Trigonometry", "topic_routing_primary_topic_id": "9709_p3_topic_integration"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_integration", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_integration"]`
- Topic-routing alignment: `supporting_topic`
- Recommended scope: `subpart_level`
- Cross-topic notes: Candidate skill/topic and topic-routing context differ but may describe different stages of one solution.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: integration.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_3_identities_compound_double_angle_equations"], "candidate_topic_ids": ["9709_p3_topic_integration"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:31summer21_q04:a", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "31summer21_q04_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "31summer21_q04_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "31summer21_q04_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer21_q04_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer21_q04_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "31summer21_q04_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["a"], "proposed_part_id": "a", "proposed_subpart_id": null, "question_id": "31summer21_q04", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/31summer21/mark_scheme/q04.png", "sha256": "e2e0a5782c24b2e458268958059fe692b7d42527e01fc97f2e92ff131e2897d4"}], "source_question_asset_refs": [{"exists": true, "path": "p3/31summer21/questions/q04.png", "sha256": "07fca842fa9ae7277cd007ede5453d56afe5566f7c9bee48cd2d3c78026e7f9a"}], "source_queue_id": "p3_exact_skill_review_queue:v1:31summer21_q04:31summer21_q04_a", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_31summer21_q04_a", "generation_gate_block_reasons": ["question_quality_gate_blocks_content_lab_generation", "mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 2}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `question_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `defer_ambiguous_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31summer21/questions/q04.png", "sha256": "07fca842fa9ae7277cd007ede5453d56afe5566f7c9bee48cd2d3c78026e7f9a"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31summer21/mark_scheme/q04.png", "sha256": "e2e0a5782c24b2e458268958059fe692b7d42527e01fc97f2e92ff131e2897d4"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31summer21_q04_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer21_q04_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer21_q04_me0003", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer21_q04_me0004", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer21_q04_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31summer21_q04_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 16. `32autumn21_q06` / `32autumn21_q06_a`

- Selection category: `known_failure_mode_probe`
- Selection reason: Trigonometry appears inside integration or area work; verify it is not only supporting method context.
- Known risk flags: `trig_supporting_integration`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32autumn21_q06:32autumn21_q06_a`
- Question ID: `32autumn21_q06`
- Part/subpart: `a` / `32autumn21_q06_a`
- Paper/session/variant: `32autumn21` / `November` / `2`
- Candidate P3 skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `cross_topic_reviewable`
- Decomposition status: `already_part_scoped`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p1_trigonometry_equations_intervals`
- Primary candidate skill IDs: `9709_p3_3_3_identities_compound_double_angle_equations`
- Supporting candidate skill IDs: `9709_p1_trigonometry_equations_intervals`
- Candidate region/topic: `{"mapping_source_topic": "trigonometry", "subtopic_id": "9709_p3_subtopic_identities_compound_double_angle_equations", "subtopic_name": "Identities, compound angles and trigonometric equations", "topic_assignment_id": "9709_p3_topic_trigonometry", "topic_assignment_name": "Trigonometry", "topic_routing_primary_topic_id": "9709_p3_topic_integration"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_integration", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_integration", "9709_p3_topic_trigonometry"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: trigonometry.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_3_identities_compound_double_angle_equations"], "candidate_topic_ids": ["9709_p3_topic_integration", "9709_p3_topic_trigonometry"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32autumn21_q06:a", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32autumn21_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn21_q06_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32autumn21_q06_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32autumn21_q06_me0004", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn21_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn21_q06_me0006", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32autumn21_q06_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["a"], "proposed_part_id": "a", "proposed_subpart_id": null, "question_id": "32autumn21_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32autumn21/mark_scheme/q06.png", "sha256": "a87311d58080dbca6f6862059e1631439910835d58bf1b942b060e8fa0ccb658"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32autumn21/questions/q06.png", "sha256": "e29a3203f9573cae5b16ec96f661912ba42536b5e5169fc1160de20d332cfa21"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32autumn21_q06:32autumn21_q06_a", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32autumn21_q06_a", "generation_gate_block_reasons": ["question_quality_gate_blocks_content_lab_generation", "mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `question_crop_not_high_confidence`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `defer_ambiguous_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32autumn21/questions/q06.png", "sha256": "e29a3203f9573cae5b16ec96f661912ba42536b5e5169fc1160de20d332cfa21"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32autumn21/mark_scheme/q06.png", "sha256": "a87311d58080dbca6f6862059e1631439910835d58bf1b942b060e8fa0ccb658"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32autumn21_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q06_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q06_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q06_me0004", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q06_me0006", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn21_q06_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 17. `33autumn22_q10` / `33autumn22_q10_a`

- Selection category: `known_failure_mode_probe`
- Selection reason: Log/exponential algebra appears inside differential-equation evidence; verify the assessed target skill.
- Known risk flags: `log_exp_supporting_differential_equations`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
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

### 18. `33autumn22_q10` / `33autumn22_q10_b`

- Selection category: `known_failure_mode_probe`
- Selection reason: Log/exponential algebra appears inside differential-equation evidence; verify the assessed target skill.
- Known risk flags: `log_exp_supporting_differential_equations`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `part_level_scope_uses_whole_images`, `mark_events_advisory_only`
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

### 19. `31autumn23_q06` / `31autumn23_q06_a`

- Selection category: `known_failure_mode_probe`
- Selection reason: Derivative-rule context may be mistaken for parametric or implicit differentiation.
- Known risk flags: `derivative_rules_vs_implicit`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31autumn23_q06:31autumn23_q06_a`
- Question ID: `31autumn23_q06`
- Part/subpart: `a` / `31autumn23_q06_a`
- Paper/session/variant: `31autumn23` / `November` / `1`
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
- Content Lab blocker context: `{"candidate_id": "content_lab_31autumn23_q06_a", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`, `weak_parametric_equation_evidence_missing_parameter`
- Reconciliation flags: none
- Recommended review action: `verify_parametric_equation_parameter`

Question asset refs:
- `{"exists": true, "path": "p3/31autumn23/questions/q06.png", "sha256": "2d960d858de2ea4b6ef938656e30013fe4209a6faf718ab5e59b3f3cc5bf0027"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31autumn23/mark_scheme/q06.png", "sha256": "08fb37b732fd16c8131a17df3bdc1493ce79e0ec3ba27225e69cf1bb0dcac644"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 20. `31autumn23_q06` / `31autumn23_q06_b`

- Selection category: `known_failure_mode_probe`
- Selection reason: Derivative-rule context may be mistaken for parametric or implicit differentiation.
- Known risk flags: `derivative_rules_vs_implicit`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `mark_events_advisory_only`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31autumn23_q06:31autumn23_q06_b`
- Question ID: `31autumn23_q06`
- Part/subpart: `b` / `31autumn23_q06_b`
- Paper/session/variant: `31autumn23` / `November` / `1`
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
- Content Lab blocker context: `{"candidate_id": "content_lab_31autumn23_q06_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `text_or_ocr_not_authoritative`, `visual_dependency`, `weak_parametric_equation_evidence_missing_parameter`
- Reconciliation flags: none
- Recommended review action: `verify_parametric_equation_parameter`

Question asset refs:
- `{"exists": true, "path": "p3/31autumn23/questions/q06.png", "sha256": "2d960d858de2ea4b6ef938656e30013fe4209a6faf718ab5e59b3f3cc5bf0027"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31autumn23/mark_scheme/q06.png", "sha256": "08fb37b732fd16c8131a17df3bdc1493ce79e0ec3ba27225e69cf1bb0dcac644"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 21. `31summer23_q02` / `31summer23_q02_a`

- Selection category: `known_failure_mode_probe`
- Selection reason: Derivative-rule context may be mistaken for parametric or implicit differentiation.
- Known risk flags: `derivative_rules_vs_implicit`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `mark_events_advisory_only`
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

### 22. `32autumn23_q03` / `32autumn23_q03_whole`

- Selection category: `known_failure_mode_probe`
- Selection reason: Polynomial/remainder theorem evidence may be supporting work rather than the target skill.
- Known risk flags: `polynomial_remainder_vs_differentiation`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `whole_question_review_scope`, `mark_events_advisory_only`
- Review scope level: `whole_question`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32autumn23_q03:32autumn23_q03_whole`
- Question ID: `32autumn23_q03`
- Part/subpart: `whole` / `32autumn23_q03_whole`
- Paper/session/variant: `32autumn23` / `November` / `2`
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
- Content Lab blocker context: `{"candidate_id": "content_lab_32autumn23_q03_whole", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 5}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32autumn23/questions/q03.png", "sha256": "9825707e28246c49b900b6956e1dbf8c10088f5840eea34b68b0df7f8e9f6b88"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32autumn23/mark_scheme/q03.png", "sha256": "cd417883cecd17e3fa9b1f65b6ee25179acac8b801262a8c49fc80498334934f"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32autumn23_q03_me0001", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q03_me0002", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q03_me0003", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q03_me0004", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32autumn23_q03_me0005", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`

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

### 23. `32spring23_q03` / `32spring23_q03_whole`

- Selection category: `known_failure_mode_probe`
- Selection reason: Polynomial/remainder theorem evidence may be supporting work rather than the target skill.
- Known risk flags: `polynomial_remainder_vs_differentiation`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `whole_question_review_scope`, `mark_events_advisory_only`
- Review scope level: `whole_question`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32spring23_q03:32spring23_q03_whole`
- Question ID: `32spring23_q03`
- Part/subpart: `whole` / `32spring23_q03_whole`
- Paper/session/variant: `32spring23` / `March` / `2`
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
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring23_q03_whole", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 18}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32spring23/questions/q03.png", "sha256": "56cc156ad1205eff46a0668279af3c540048fa2770e3335f585d220da4e4df1d"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32spring23/mark_scheme/q03.png", "sha256": "ef5f22370a858d61561126f50ea76661b3d9afbfc454407fd17674b066115849"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32spring23_q03_me0001", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0002", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0003", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0004", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0005", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0006", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0007", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0008", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0009", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0010", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0011", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0012", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0013", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0014", "mark_code": "B2", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0015", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0016", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0017", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring23_q03_me0018", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`

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

### 24. `31autumn21_q01` / `31autumn21_q01_whole`

- Selection category: `known_failure_mode_probe`
- Selection reason: Whole-question candidate likely contains only one part matching the proposed skill.
- Known risk flags: `whole_question_single_part_match`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `whole_question_review_scope`, `mark_events_advisory_only`
- Review scope level: `whole_question`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:31autumn21_q01:31autumn21_q01_whole`
- Question ID: `31autumn21_q01`
- Part/subpart: `whole` / `31autumn21_q01_whole`
- Paper/session/variant: `31autumn21` / `November` / `1`
- Candidate P3 skill IDs: `9709_p3_3_1_modulus_equations_inequalities`
- Suggested candidate status: `cross_topic_candidate`
- Suggested review priority: `2_cross_topic_candidate`
- Suggested ambiguity reason: `topic_routing_candidate_mismatch`
- Decomposition status: `insufficient_part_signal`
- Candidate source skill IDs, including prerequisite/support context: `9709_p3_3_1_modulus_equations_inequalities`, `9709_p1_quadratics_discriminant_intersections`, `9709_p1_functions_domain_range_one_one`
- Primary candidate skill IDs: `9709_p3_3_1_modulus_equations_inequalities`
- Supporting candidate skill IDs: `9709_p1_quadratics_discriminant_intersections`, `9709_p1_functions_domain_range_one_one`
- Candidate region/topic: `{"mapping_source_topic": "modulus", "subtopic_id": "9709_p3_subtopic_modulus_equations_inequalities", "subtopic_name": "Modulus equations and inequalities", "topic_assignment_id": "9709_p3_topic_algebra", "topic_assignment_name": "Algebra", "topic_routing_primary_topic_id": "9709_p3_topic_logarithmic_and_exponential_functions"}`
- Topic-routing context: `{"confidence": "high", "evidence_used": ["question_text", "ocr_text", "mark_scheme_text"], "primary_topic_id": "9709_p3_topic_logarithmic_and_exponential_functions", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_logarithmic_and_exponential_functions"]`
- Topic-routing alignment: `supporting_topic`
- Recommended scope: `reviewer_decide`
- Cross-topic notes: Candidate skill/topic and topic-routing context differ but may describe different stages of one solution.; Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: modulus.
Part-level decomposition candidates:
- None
- Content Lab blocker context: `{"candidate_id": "content_lab_31autumn21_q01_whole", "generation_gate_block_reasons": ["question_quality_gate_blocks_content_lab_generation", "mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 8}`
- Proposed blockers: `mixed_or_ambiguous_topic`, `mark_events_advisory_only`, `question_crop_not_high_confidence`
- Reconciliation flags: none
- Recommended review action: `defer_ambiguous_skill`

Question asset refs:
- `{"exists": true, "path": "p3/31autumn21/questions/q01.png", "sha256": "d2fe27d2f528f404a8a41dea157e004006286c129a7b114f13039cad25033bd3"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/31autumn21/mark_scheme/q01.png", "sha256": "797aad9d00db4376ab538f3d4913ae536f6307658fa7de4fe23bfdced7e5993e"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "31autumn21_q01_me0001", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q01_me0002", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q01_me0003", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q01_me0004", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q01_me0005", "mark_code": "B1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q01_me0006", "mark_code": "B1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q01_me0007", "mark_code": "M1", "part_path": [], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "31autumn21_q01_me0008", "mark_code": "A1", "part_path": [], "review_status": "advisory"}`

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

### 25. `31autumn21_q04` / `31autumn21_q04_whole`

- Selection category: `known_failure_mode_probe`
- Selection reason: Whole-question candidate likely contains only one part matching the proposed skill.
- Known risk flags: `whole_question_single_part_match`, `known_batch_0001_failure_mode`, `do_not_default_to_clean`, `mixed_or_ambiguous_topic`, `whole_question_review_scope`, `mark_events_advisory_only`
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

### 26. `32spring24_q03` / `32spring24_q03_a`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Complex polar/modulus evidence was reliable in Batch 0001; confirm the pattern on additional items.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32spring24_q03:32spring24_q03_a`
- Question ID: `32spring24_q03`
- Part/subpart: `a` / `32spring24_q03_a`
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
- Topic-routing context: `{"confidence": "high", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "9709_p3_topic_complex_numbers", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_complex_numbers"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: complex_numbers.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32spring24_q03:a", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32spring24_q03_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0005", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0006", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0007", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0008", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0009", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0010", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0011", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0012", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32spring24_q03_me0013", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0014", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0015", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["a"], "proposed_part_id": "a", "proposed_subpart_id": null, "question_id": "32spring24_q03", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32spring24/mark_scheme/q03.png", "sha256": "6e59eddc570aaed3e3f10d0fc0a1d0194f3663d0d8223178d618990f820c1f71"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32spring24/questions/q03.png", "sha256": "de18052cd64852e065ea7a9597274c69ba4a9be6390de6c0057a8c9194ee21fe"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32spring24_q03:32spring24_q03_a", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring24_q03_a", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 12}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32spring24/questions/q03.png", "sha256": "de18052cd64852e065ea7a9597274c69ba4a9be6390de6c0057a8c9194ee21fe"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32spring24/mark_scheme/q03.png", "sha256": "6e59eddc570aaed3e3f10d0fc0a1d0194f3663d0d8223178d618990f820c1f71"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32spring24_q03_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0005", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0006", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0007", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0008", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0009", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0010", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0011", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0012", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0013", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0014", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0015", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 27. `32spring24_q03` / `32spring24_q03_b`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Complex polar/modulus evidence was reliable in Batch 0001; confirm the pattern on additional items.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32spring24_q03:32spring24_q03_b`
- Question ID: `32spring24_q03`
- Part/subpart: `b` / `32spring24_q03_b`
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
- Topic-routing context: `{"confidence": "high", "evidence_used": ["mark_scheme_text"], "primary_topic_id": "9709_p3_topic_complex_numbers", "review_reasons": [], "review_required": false, "routing_source": "deepseek_topic_routing"}`
- Cross-topic status: `cross_topic_reviewable`
- Topic-routing topic IDs: `["9709_p3_topic_complex_numbers"]`
- Topic-routing alignment: `aligned`
- Recommended scope: `subpart_level`
- Cross-topic notes: Supporting candidate skills are review context only, not mastery evidence.; Source topic hint: complex_numbers.
Part-level decomposition candidates:
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_9_complex_arithmetic_polar_form"], "candidate_topic_ids": ["9709_p3_topic_complex_numbers"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32spring24_q03:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32spring24_q03_me0013", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0014", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0015", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32spring24_q03_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0005", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0006", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0007", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0008", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0009", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0010", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0011", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q03_me0012", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "32spring24_q03", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32spring24/mark_scheme/q03.png", "sha256": "6e59eddc570aaed3e3f10d0fc0a1d0194f3663d0d8223178d618990f820c1f71"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32spring24/questions/q03.png", "sha256": "de18052cd64852e065ea7a9597274c69ba4a9be6390de6c0057a8c9194ee21fe"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32spring24_q03:32spring24_q03_b", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring24_q03_b", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 3}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32spring24/questions/q03.png", "sha256": "de18052cd64852e065ea7a9597274c69ba4a9be6390de6c0057a8c9194ee21fe"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32spring24/mark_scheme/q03.png", "sha256": "6e59eddc570aaed3e3f10d0fc0a1d0194f3663d0d8223178d618990f820c1f71"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32spring24_q03_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0002", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0004", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0005", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0006", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0007", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0008", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0009", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0010", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0011", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0012", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0013", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0014", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q03_me0015", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`

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

### 28. `32spring24_q05` / `32spring24_q05_b`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Complex polar/modulus evidence was reliable in Batch 0001; confirm the pattern on additional items.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
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

### 29. `31summer24_q06` / `31summer24_q06_c`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Fixed-point iteration evidence was reliable in Batch 0001; confirm the pattern on additional items.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
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

### 30. `31summer24_q06` / `31summer24_q06_d`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Fixed-point iteration evidence was reliable in Batch 0001; confirm the pattern on additional items.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
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

### 31. `33autumn23_q06` / `33autumn23_q06_a`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Trig identities/equations were reliable when they were the target skill; confirm clean target-vs-support separation.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:33autumn23_q06:33autumn23_q06_a`
- Question ID: `33autumn23_q06`
- Part/subpart: `a` / `33autumn23_q06_a`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_3_identities_compound_double_angle_equations"], "candidate_topic_ids": ["9709_p3_topic_trigonometry"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33autumn23_q06:a", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33autumn23_q06_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33autumn23_q06_me0005", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0006", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0007", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33autumn23_q06_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["a"], "proposed_part_id": "a", "proposed_subpart_id": null, "question_id": "33autumn23_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33autumn23/mark_scheme/q06.png", "sha256": "e76fca79346f0e4a08a6e64d9af661a1b74cf178d41edf3f20f796f889e928c8"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33autumn23/questions/q06.png", "sha256": "2e16cf3fcb03cac37635d4e5cce9219699bb03f82a230c454976719defe69c23"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33autumn23_q06:33autumn23_q06_a", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33autumn23_q06_a", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 4}`
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

### 32. `33summer23_q06` / `33summer23_q06_a`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Trig identities/equations were reliable when they were the target skill; confirm clean target-vs-support separation.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:33summer23_q06:33summer23_q06_a`
- Question ID: `33summer23_q06`
- Part/subpart: `a` / `33summer23_q06_a`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_3_identities_compound_double_angle_equations"], "candidate_topic_ids": ["9709_p3_topic_trigonometry"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33summer23_q06:a", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33summer23_q06_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0002", "mark_code": "B1FT", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0004", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0005", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33summer23_q06_me0006", "mark_code": "B1FT", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0007", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0008", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0009", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0010", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer23_q06_me0011", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["a"], "proposed_part_id": "a", "proposed_subpart_id": null, "question_id": "33summer23_q06", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33summer23/mark_scheme/q06.png", "sha256": "c75546cd79657e209d4b54179f4ff50f38457d35b564cadb2b4516bbfd21f0fa"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33summer23/questions/q06.png", "sha256": "f7b0303d29c351ea60598e68150b6c2a2862b4834a3e8a5829c587bbeae02287"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33summer23_q06:33summer23_q06_a", "supporting_skill_ids": ["9709_p1_trigonometry_equations_intervals"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33summer23_q06_a", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 5}`
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

### 33. `31summer24_q09` / `31summer24_q09_a`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Vector-line evidence was reliable in Batch 0001; confirm the pattern on additional items.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
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

### 34. `31summer24_q09` / `31summer24_q09_b`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Vector-line evidence was reliable in Batch 0001; confirm the pattern on additional items.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
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

### 35. `33summer25_q05` / `33summer25_q05_a`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Parametric/implicit differentiation evidence was reliable in Batch 0001; confirm the pattern on additional items.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:33summer25_q05:33summer25_q05_a`
- Question ID: `33summer25_q05`
- Part/subpart: `a` / `33summer25_q05_a`
- Paper/session/variant: `33summer25` / `June` / `3`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_4_parametric_implicit_differentiation"], "candidate_topic_ids": ["9709_p3_topic_differentiation"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:33summer25_q05:a", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "33summer25_q05_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer25_q05_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer25_q05_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer25_q05_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer25_q05_me0005", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer25_q05_me0006", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer25_q05_me0007", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "33summer25_q05_me0008", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "33summer25_q05_me0009", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "33summer25_q05_me0010", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["a"], "proposed_part_id": "a", "proposed_subpart_id": null, "question_id": "33summer25_q05", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/33summer25/mark_scheme/q05.png", "sha256": "a006e672763a3e135f50f5ba6b83a0b4dbfd92053387a44f92bdc182d132ab90"}], "source_question_asset_refs": [{"exists": true, "path": "p3/33summer25/questions/q05.png", "sha256": "9d0c55b1160d3c182d5ea450423924fd4d61e7c6ad10a5e18d47f42fd49a6b41"}], "source_queue_id": "p3_exact_skill_review_queue:v1:33summer25_q05:33summer25_q05_a", "supporting_skill_ids": ["9709_p3_3_4_derivative_rules"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_33summer25_q05_a", "generation_gate_block_reasons": ["question_quality_gate_blocks_content_lab_generation", "mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 8}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/33summer25/questions/q05.png", "sha256": "9d0c55b1160d3c182d5ea450423924fd4d61e7c6ad10a5e18d47f42fd49a6b41"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/33summer25/mark_scheme/q05.png", "sha256": "a006e672763a3e135f50f5ba6b83a0b4dbfd92053387a44f92bdc182d132ab90"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "33summer25_q05_me0001", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0002", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0003", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0004", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0005", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0006", "mark_code": "B1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0007", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0008", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0009", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "33summer25_q05_me0010", "mark_code": "B1", "part_path": ["b"], "review_status": "advisory"}`

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

### 36. `32spring23_q10` / `32spring23_q10_b`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Top-up confidence-confirmation item from a Batch 0001 reliable skill family.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32spring23_q10:32spring23_q10_b`
- Question ID: `32spring23_q10`
- Part/subpart: `b` / `32spring23_q10_b`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_7_vector_lines"], "candidate_topic_ids": ["9709_p3_topic_vectors"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32spring23_q10:b", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32spring23_q10_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32spring23_q10_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0006", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0007", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring23_q10_me0009", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}], "part_path": ["b"], "proposed_part_id": "b", "proposed_subpart_id": null, "question_id": "32spring23_q10", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32spring23/mark_scheme/q10.png", "sha256": "35155bc3e80c0f15956cf35b03b56b738e74a03a39f4add742b2758adc7ca25e"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32spring23/questions/q10.png", "sha256": "e3f3bbae28ab21ddc5453d65ba87a49b64b1bac169ef5e118f06008e31132040"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32spring23_q10:32spring23_q10_b", "supporting_skill_ids": ["9709_p1_coordinate_line_geometry"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring23_q10_b", "generation_gate_block_reasons": ["mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "machine_candidate", "role_statuses": {"field_guide_source": "allow", "generated_warmup_pattern_source": "block", "guardian_candidate": "allow", "mixed_review_source": "block", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 2}`
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

### 37. `32spring24_q07` / `32spring24_q07_c`

- Selection category: `reliable_pattern_confirmation`
- Selection reason: Top-up confidence-confirmation item from a Batch 0001 reliable skill family.
- Known risk flags: `pattern_confirmed_clean_in_batch_0001`, `mark_events_advisory_only`, `part_level_scope_uses_whole_images`
- Review scope level: `part_level`
- Related reviewed registry evidence exists: `False`
- Related reviewed evidence IDs: none
- Queue ID: `p3_exact_skill_review_queue:v1:32spring24_q07:32spring24_q07_c`
- Question ID: `32spring24_q07`
- Part/subpart: `c` / `32spring24_q07_c`
- Paper/session/variant: `32spring24` / `March` / `2`
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
- `{"blockers": ["uses_whole_question_images_for_part_review"], "candidate_source_skill_ids": ["9709_p3_3_6_fixed_point_iteration"], "candidate_topic_ids": ["9709_p3_topic_numerical_solution_of_equations"], "confidence": "medium", "decomposition_id": "p3_part_decomp:v1:32spring24_q07:c", "decomposition_status": "already_part_scoped", "evidence_signals": {"mark_event_part_match": true, "mark_scheme_method_signal": true, "part_label_signal": true, "skill_mapping_signal": true, "topic_assignment_signal": true}, "matching_mark_event_refs": [{"event_id": "32spring24_q07_me0006", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring24_q07_me0007", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring24_q07_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}, {"event_id": "32spring24_q07_me0009", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}], "other_part_mark_event_refs": [{"event_id": "32spring24_q07_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q07_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q07_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}, {"event_id": "32spring24_q07_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}, {"event_id": "32spring24_q07_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}], "part_path": ["c"], "proposed_part_id": "c", "proposed_subpart_id": null, "question_id": "32spring24_q07", "recommended_review_action": "review_part_scope_and_skill", "source_mark_scheme_asset_refs": [{"exists": true, "path": "p3/32spring24/mark_scheme/q07.png", "sha256": "575e363d3295242acfc28caa9a58f459b3b8916eb2b0cf42c9b55dff5303c815"}], "source_question_asset_refs": [{"exists": true, "path": "p3/32spring24/questions/q07.png", "sha256": "923c6134605101873665ee25687c759d444f795b65aa40138117a93e8c3c376f"}], "source_queue_id": "p3_exact_skill_review_queue:v1:32spring24_q07:32spring24_q07_c", "supporting_skill_ids": ["9709_p3_3_6_root_location"], "warning": "This is a decomposition candidate, not reviewed evidence."}`
- Content Lab blocker context: `{"candidate_id": "content_lab_32spring24_q07_c", "generation_gate_block_reasons": ["mapping_or_subpart_not_reviewed_or_approved", "mark_events_not_reviewed_or_approved", "missing_source_skill_ids"], "generation_gate_blocked": true, "generation_gate_status": "blocked_until_reviewed", "review_status": "blocked_until_reviewed", "role_statuses": {"field_guide_source": "block", "generated_warmup_pattern_source": "block", "guardian_candidate": "block", "mixed_review_source": "blocked_until_reviewed", "prerequisite_repair_source": "block", "quick_check_source": "block"}, "source_mark_event_count": 4}`
- Proposed blockers: `mark_events_advisory_only`, `mark_scheme_crop_not_high_confidence`, `text_or_ocr_not_authoritative`, `visual_dependency`
- Reconciliation flags: none
- Recommended review action: `review_assets_and_skill`

Question asset refs:
- `{"exists": true, "path": "p3/32spring24/questions/q07.png", "sha256": "923c6134605101873665ee25687c759d444f795b65aa40138117a93e8c3c376f"}`

Mark-scheme asset refs:
- `{"exists": true, "path": "p3/32spring24/mark_scheme/q07.png", "sha256": "575e363d3295242acfc28caa9a58f459b3b8916eb2b0cf42c9b55dff5303c815"}`

Advisory-only mark-event refs:
- `{"advisory_only": true, "event_id": "32spring24_q07_me0001", "mark_code": "M1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q07_me0002", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q07_me0003", "mark_code": "A1", "part_path": ["a"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q07_me0004", "mark_code": "M1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q07_me0005", "mark_code": "A1", "part_path": ["b"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q07_me0006", "mark_code": "M1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q07_me0007", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q07_me0008", "mark_code": "A1", "part_path": ["c"], "review_status": "advisory"}`
- `{"advisory_only": true, "event_id": "32spring24_q07_me0009", "mark_code": "B1", "part_path": ["c"], "review_status": "advisory"}`

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
