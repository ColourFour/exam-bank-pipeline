# Post Storage Cleanup Delta Review

## Summary

Compared previous Content Lab export `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/json/asterion_content_lab_candidates_v1.json` against current `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`.

- Content Lab candidates: `2416` -> `2432` (`+16`).
- Candidate ID delta: `19` added, `3` removed, `2413` unchanged IDs.
- Assessment: expected export-regeneration delta, not a storage-cleanup effect.
- All newly appearing candidates are review-only/blocked; none are generation-ready.
- Storage cleanup overlap with current candidate source artifacts: `0` paths.
- Topic routing remains unsafe for strict filters because `153` existing schema-validation failure records remain.
- Remaining storage duplicates: `10` groups, `785.98 KiB` waste, `0 B` reclaimable under hard-delete policy.

## Content Lab Delta

The net `+16` count is not just ordering or count metadata. The current export has 19 candidate IDs that were not in the archived previous export, and 3 archived candidate IDs are no longer present.

The 19 added candidates all come from `33autumn25`. In the archived previous question bank these records had `mapping_status=fail` and no `mark_scheme_image_path`; in the current question bank they have canonical mark-scheme paths and `mapping_status=pass`, so the Content Lab builder can emit review-only mixed-review candidates. The generated candidates are still blocked by review gates, unreviewed mark events, and missing source skill IDs.

The 3 removed candidates now have downgraded question crop/readiness status and low mark-event confidence, so they no longer pass the unreviewed machine-candidate threshold. They were not made unsafe by storage cleanup.

Common candidate field deltas are broader than the 16-count change: all common records now include asset IDs in `source_artifacts`, and many role/gate fields changed because the current export was regenerated from the current question-bank/Asterion source rather than the archived OCR-candidate source.

## Newly Appearing Candidates

| candidate_id | question_id | topic / confidence | marks | source refs | gate / review | safety |
| --- | --- | --- | ---: | --- | --- | --- |
| `content_lab_33autumn25_q01_whole` | `33autumn25_q01` | `parametric_equations` / `medium`; skills `0` | 4 | `p3/33autumn25/questions/q01.png` / `p3/33autumn25/mark_scheme/q01.png`; `question_image:33autumn25:33autumn25_q01` / `mark_scheme_image:33autumn25:33autumn25_q01` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q02_whole` | `33autumn25_q02` | `polynomials` / `low`; skills `0` | 3 | `p3/33autumn25/questions/q02.png` / `p3/33autumn25/mark_scheme/q02.png`; `question_image:33autumn25:33autumn25_q02` / `mark_scheme_image:33autumn25:33autumn25_q02` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q03_whole` | `33autumn25_q03` | `logarithms_and_exponentials` / `low`; skills `0` | 4 | `p3/33autumn25/questions/q03.png` / `p3/33autumn25/mark_scheme/q03.png`; `question_image:33autumn25:33autumn25_q03` / `mark_scheme_image:33autumn25:33autumn25_q03` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q04_whole` | `33autumn25_q04` | `complex_numbers` / `low`; skills `0` | 5 | `p3/33autumn25/questions/q04.png` / `p3/33autumn25/mark_scheme/q04.png`; `question_image:33autumn25:33autumn25_q04` / `mark_scheme_image:33autumn25:33autumn25_q04` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q05_a` | `33autumn25_q05` | `trigonometry` / `low`; skills `0` | 4 | `p3/33autumn25/questions/q05.png` / `p3/33autumn25/mark_scheme/q05.png`; `question_image:33autumn25:33autumn25_q05` / `mark_scheme_image:33autumn25:33autumn25_q05` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q05_b` | `33autumn25_q05` | `trigonometry` / `low`; skills `0` | 4 | `p3/33autumn25/questions/q05.png` / `p3/33autumn25/mark_scheme/q05.png`; `question_image:33autumn25:33autumn25_q05` / `mark_scheme_image:33autumn25:33autumn25_q05` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q06_whole` | `33autumn25_q06` | `integration` / `low`; skills `0` | 6 | `p3/33autumn25/questions/q06.png` / `p3/33autumn25/mark_scheme/q06.png`; `question_image:33autumn25:33autumn25_q06` / `mark_scheme_image:33autumn25:33autumn25_q06` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q07_whole` | `33autumn25_q07` | `algebra` / `low`; skills `0` | 6 | `p3/33autumn25/questions/q07.png` / `p3/33autumn25/mark_scheme/q07.png`; `question_image:33autumn25:33autumn25_q07` / `mark_scheme_image:33autumn25:33autumn25_q07` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q08_a` | `33autumn25_q08` | `numerical_methods` / `low`; skills `0` | 3 | `p3/33autumn25/questions/q08.png` / `p3/33autumn25/mark_scheme/q08.png`; `question_image:33autumn25:33autumn25_q08` / `mark_scheme_image:33autumn25:33autumn25_q08` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q08_b` | `33autumn25_q08` | `numerical_methods` / `low`; skills `0` | 2 | `p3/33autumn25/questions/q08.png` / `p3/33autumn25/mark_scheme/q08.png`; `question_image:33autumn25:33autumn25_q08` / `mark_scheme_image:33autumn25:33autumn25_q08` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q08_c` | `33autumn25_q08` | `numerical_methods` / `low`; skills `0` | 2 | `p3/33autumn25/questions/q08.png` / `p3/33autumn25/mark_scheme/q08.png`; `question_image:33autumn25:33autumn25_q08` / `mark_scheme_image:33autumn25:33autumn25_q08` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q08_d` | `33autumn25_q08` | `numerical_methods` / `low`; skills `0` | 3 | `p3/33autumn25/questions/q08.png` / `p3/33autumn25/mark_scheme/q08.png`; `question_image:33autumn25:33autumn25_q08` / `mark_scheme_image:33autumn25:33autumn25_q08` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q09_a` | `33autumn25_q09` | `vectors` / `low`; skills `0` | 3 | `p3/33autumn25/questions/q09.png` / `p3/33autumn25/mark_scheme/q09.png`; `question_image:33autumn25:33autumn25_q09` / `mark_scheme_image:33autumn25:33autumn25_q09` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q09_b` | `33autumn25_q09` | `vectors` / `low`; skills `0` | 3 | `p3/33autumn25/questions/q09.png` / `p3/33autumn25/mark_scheme/q09.png`; `question_image:33autumn25:33autumn25_q09` / `mark_scheme_image:33autumn25:33autumn25_q09` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q09_c` | `33autumn25_q09` | `vectors` / `low`; skills `0` | 3 | `p3/33autumn25/questions/q09.png` / `p3/33autumn25/mark_scheme/q09.png`; `question_image:33autumn25:33autumn25_q09` / `mark_scheme_image:33autumn25:33autumn25_q09` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q10_a` | `33autumn25_q10` | `partial_fractions` / `low`; skills `0` | 2 | `p3/33autumn25/questions/q10.png` / `p3/33autumn25/mark_scheme/q10.png`; `question_image:33autumn25:33autumn25_q10` / `mark_scheme_image:33autumn25:33autumn25_q10` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q10_b` | `33autumn25_q10` | `partial_fractions` / `low`; skills `0` | 6 | `p3/33autumn25/questions/q10.png` / `p3/33autumn25/mark_scheme/q10.png`; `question_image:33autumn25:33autumn25_q10` / `mark_scheme_image:33autumn25:33autumn25_q10` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q11_a` | `33autumn25_q11` | `parametric_equations` / `low`; skills `0` | 6 | `p3/33autumn25/questions/q11.png` / `p3/33autumn25/mark_scheme/q11.png`; `question_image:33autumn25:33autumn25_q11` / `mark_scheme_image:33autumn25:33autumn25_q11` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |
| `content_lab_33autumn25_q11_b` | `33autumn25_q11` | `parametric_equations` / `low`; skills `0` | 6 | `p3/33autumn25/questions/q11.png` / `p3/33autumn25/mark_scheme/q11.png`; `question_image:33autumn25:33autumn25_q11` / `mark_scheme_image:33autumn25:33autumn25_q11` | `blocked_until_reviewed`; `blocked_until_reviewed`; reasons `question_quality_gate_blocks_content_lab_generation, mapping_or_subpart_not_reviewed_or_approved, mark_events_not_reviewed_or_approved, missing_source_skill_ids` | `review_only_blocked_until_reviewed` |

## No Longer Present

| candidate_id | question_id | previous status | current reason |
| --- | --- | --- | --- |
| `content_lab_12autumn22_q11_c` | `12autumn22_q11` | `blocked_until_reviewed` | current crop/status downgraded; quality reasons `question_crop_not_high_confidence, text_only_blocked_status_review, text_only_blocked_trust_medium, text_only_blocked_visual_required`; min mark-event confidence `0.81` |
| `content_lab_41summer23_q06_a` | `41summer23_q06` | `machine_candidate` | current crop/status downgraded; quality reasons `content_lab_blocked_topic_confidence_low, content_lab_blocked_topic_uncertain, question_crop_not_high_confidence, text_only_blocked_status_review, text_only_blocked_trust_medium, text_only_blocked_visual_required`; min mark-event confidence `0.81` |
| `content_lab_42summer22_q05_whole` | `42summer22_q05` | `machine_candidate` | current crop/status downgraded; quality reasons `question_crop_not_high_confidence, text_only_blocked_status_review, text_only_blocked_trust_medium, text_only_blocked_visual_required`; min mark-event confidence `0.81` |

## Storage Cleanup Effect

The delete manifest records `13198` hard-deleted files (`1.96 GiB`). None of those paths overlap the current Content Lab source artifacts, and asset-reference validation reports `ok=true` with `2432` candidates. The storage cleanup therefore did not cause a functional candidate export change.

## Topic Routing Failures

Topic routing still reports `153` failed records and `safe_for_strict_filters=false`. All failure records have `review_reasons=[schema_validation_error]` and category `unsupported_evidence_used`.

The failure pattern is batch-level: each failed paper has one blamed record whose model response listed an `evidence_used` value that was not supplied in that routing packet, and the whole batch/paper was written as error records.

| paper_family | failures |
| --- | ---: |
| `p5` | 61 |
| `p3` | 52 |
| `p1` | 33 |
| `p4` | 7 |

| blamed record | affected records | message |
| --- | ---: | --- |
| `12autumn21_q06` | 12 | `records[12autumn21_q06]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `13autumn23_q11` | 11 | `records[13autumn23_q11]: evidence_used contains evidence that was not supplied: ['question_text'].` |
| `31autumn23_q07` | 11 | `records[31autumn23_q07]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `33autumn24_q05` | 11 | `records[33autumn24_q05]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `12summer24_q02` | 10 | `records[12summer24_q02]: evidence_used contains evidence that was not supplied: ['question_text'].` |
| `31summer23_q04` | 10 | `records[31summer23_q04]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `32autumn22_q07` | 10 | `records[32autumn22_q07]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `32spring21_q05` | 10 | `records[32spring21_q05]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `42spring22_q03` | 7 | `records[42spring22_q03]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `51summer21_q04` | 7 | `records[51summer21_q04]: evidence_used contains evidence that was not supplied: ['question_text'].` |
| `51summer23_q04` | 7 | `records[51summer23_q04]: evidence_used contains evidence that was not supplied: ['question_text'].` |
| `51summer24_q05` | 7 | `records[51summer24_q05]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `52spring23_q01` | 7 | `records[52spring23_q01]: evidence_used contains evidence that was not supplied: ['question_text'].` |
| `52summer24_q03` | 7 | `records[52summer24_q03]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `53autumn21_q06` | 7 | `records[53autumn21_q06]: evidence_used contains evidence that was not supplied: ['question_text'].` |
| `53summer23_q06` | 7 | `records[53summer23_q06]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |
| `51summer22_q03` | 6 | `records[51summer22_q03]: evidence_used contains evidence that was not supplied: ['question_text'].` |
| `53autumn23_q03` | 6 | `records[53autumn23_q03]: evidence_used contains evidence that was not supplied: ['ocr_text'].` |

## Remaining Duplicate Groups

The final storage audit has `10` duplicate groups, `785.98 KiB` duplicate waste, and `0 B` reclaimable under the hard-delete policy. These groups are marked `do not touch` and were left untouched. They are mostly archive-vs-targeted comparison image pairs plus before/after output inventory files.

## Validation

- `.venv/bin/python scripts/validate_asset_references.py --output reports/asset_reference_validation.v1.json`: passed, `ok=true`.
- `.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --input output/json/question_bank.json --artifact-root output`: passed, wrote current Content Lab export.
- `audit_topic_routing_sidecar('output/json/question_bank.topic_routing.v1.json')`: passed as an audit call; summary still reports 153 failed records and `safe_for_strict_filters=false`.
- `.venv/bin/python -m pytest -q tests/test_asterion_export.py tests/test_topic_routing.py tests/test_asset_manifest_storage_audit.py`: passed, `30 passed`.

## Recommendation

Treat the Content Lab count delta as expected for the current regenerated export, but keep all newly added `33autumn25` candidates in review-only workflows until source skill mappings and mark-event review are available. Address topic routing separately by fixing or rerunning the affected batches whose model responses used unavailable evidence labels.
