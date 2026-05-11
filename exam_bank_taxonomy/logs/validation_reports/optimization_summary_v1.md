# Optimization Summary v1

Generated: 2026-05-11T05:18:25.418204+00:00

## Cleaned Up

Created `exam_bank_taxonomy/` with separated canonical files, review queues, archives, changelogs, and validation reports. Root-level taxonomy artifacts were copied into canonical folders, then archived as superseded or deprecated. The skill-map and topic-filter generator scripts were updated so future runs write to the organized structure instead of recreating root-level JSON.

## Canonical Files

Canonical files by type: {'changelog': 1, 'coverage_report': 5, 'index': 5, 'question_skill_mappings': 4, 'question_topic_assignments': 4, 'skill_map': 4, 'strict_topic_filtering_report': 5, 'topic_filter_map': 4}

These should power product work only with the caveat that current mappings are candidate data, not reviewed labels. Strict-filter product reads should use the canonical strict filtering reports and question-topic assignments, while respecting `strict_filter_eligible`.

## Archived

Archived files: 30

Archive reasons: {'Root-level artifact superseded by organized exam_bank_taxonomy structure.': 30}

All archived files are represented in `canonical/indexes/archive_index_v1.json`.

## Preserved Data

Reviewed status, evidence snippets, confidence values, mapping_source values, source metadata, and original IDs were preserved. No reviewed records were overwritten by machine candidates.

## Machine Candidates Separated

Low-confidence, whole-question-only, prerequisite-only, context-only, missing-evidence, questionable-reviewed, duplicate-conflict, invalid-reference, and legacy-topic cleanup queues were created under `review_queue/`.

Non-empty review queues: [('low_confidence_skill_mappings', 1150), ('low_confidence_topic_assignments', 2768), ('whole_question_only_mappings', 1025), ('prerequisite_only_assignments', 2054), ('legacy_topic_cleanup', 842)]

## Should Not Power Strict Filtering

Do not use review queue files, legacy question-bank topic labels, low-confidence mappings, prerequisite-only assignments, context-only assignments, or whole-question-only mappings for default strict filters.

## Validation

Validation status: pass

Errors: 0
Warnings: 0

## Remaining Risks

The main unresolved risk is lack of human-reviewed mapping data. High-confidence machine candidates may be structurally eligible for strict filtering, but they should remain clearly labeled until subject expert review promotes them.
