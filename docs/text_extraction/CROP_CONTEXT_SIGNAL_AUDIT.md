# Crop Context Signal Audit

This is a non-mutating audit of advisory text selection for frozen bad-text fixtures.
Canonical question and mark-scheme images remain the source of truth.

## Summary

- Records: 36
- Question-bank context joined: 36
- Records with useful warnings: 35
- Records caught by practical-now gates: 35
- Warning categories: available_now=94, requires_new_metadata=73, too_risky_or_ambiguous=54

## Signals Available Now

- expected question number appears in selected text: Safe as a review gate when the expected number is known.
- mark bracket appears in selected text: Safe as a review gate for CAIE-style prompts with known mark totals.
- expected subpart labels appear: Safe when subparts are present in fixture expectations or detected structure.
- selected text likely includes next question: Safe as a high-severity warning when a following question anchor appears after marks.
- suspiciously short selected text: Useful warning, but should combine with marks/subparts/selector warnings before blocking.
- page furniture dominates selected text: Safe when boilerplate or answer-space phrases dominate rather than merely appear.
- mark-scheme/question mismatch risk: Safe when source metadata, mapping status, or missing mark-scheme image contradicts the question.
- current selector warnings: Available in question-bank notes and useful as non-mutating warning evidence.

## Signals Requiring New Metadata

- crop pixel dimensions and normalized crop area: Needed to judge whether a text crop is implausibly small or too page-like across papers.
- text line bboxes linked to selected text spans: Needed to distinguish prompt text from headers, footers, diagrams, and answer spaces reliably.
- raw candidate windows with rejected reasons: Needed to replay selector choices and test alternative advisory text gates without mutation.

## Signals Too Risky Or Ambiguous

- math expression correctness from text alone: Requires visual/manual or math-aware comparison; crop metadata can only flag risk.
- short text alone: Some valid prompts are very short, especially single-part questions.
- answer-space language alone: Can appear legitimately in instructions; only dominant or combined evidence is safe.
- visual crop correctness from selected text only: Text anomalies do not prove the canonical image crop is wrong.

## Warning Counts

| Warning | Count |
| --- | ---: |
| low_crop_confidence | 23 |
| math_expression_semantics | 13 |
| missing_crop_pixel_dimensions | 36 |
| missing_expected_question_number | 5 |
| missing_mark_bracket | 4 |
| missing_raw_candidate_windows | 36 |
| missing_text_line_bboxes | 1 |
| selector_structural_warning_present | 33 |
| selector_warning_present | 27 |
| short_single_part_question | 5 |
| suspiciously_short_selected_text | 2 |
| visual_crop_scope_from_text_only | 36 |

## Per-Fixture Warnings

| Record | Practical now | Requires metadata | Risky/ambiguous |
| --- | --- | --- | --- |
| 12summer21_q03 | low_crop_confidence, selector_structural_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |
| 32autumn21_q04 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | short_single_part_question, visual_crop_scope_from_text_only |
| 33autumn21_q04 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |
| 33autumn21_q05 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, short_single_part_question, visual_crop_scope_from_text_only |
| 31summer22_q09 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |
| 32spring23_q04 | none | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 12summer23_q01 | selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 13summer23_q01 | selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows, missing_text_line_bboxes | visual_crop_scope_from_text_only |
| 33summer24_q03 | low_crop_confidence, missing_mark_bracket, selector_structural_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 35summer25_q04 | low_crop_confidence, missing_mark_bracket, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |
| 33autumn25_q07 | low_crop_confidence, missing_mark_bracket, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 13autumn25_q09 | low_crop_confidence, missing_mark_bracket, selector_structural_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 43autumn21_q06 | missing_expected_question_number, selector_structural_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 12spring22_q08 | low_crop_confidence, missing_expected_question_number, selector_structural_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 52spring22_q06 | missing_expected_question_number, selector_structural_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 32summer23_q09 | low_crop_confidence, missing_expected_question_number, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |
| 41summer23_q06 | low_crop_confidence, missing_expected_question_number, selector_structural_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 52spring21_q05 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 51summer21_q05 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 52summer21_q02 | low_crop_confidence, selector_structural_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 52summer22_q02 | selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 53summer22_q07 | selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 11autumn23_q04 | selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |
| 12autumn23_q06 | selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |
| 51autumn23_q05 | selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 42autumn21_q01 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 52autumn21_q01 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 31summer21_q01 | low_crop_confidence, selector_structural_warning_present, selector_warning_present, suspiciously_short_selected_text | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, short_single_part_question, visual_crop_scope_from_text_only |
| 32summer21_q01 | low_crop_confidence, selector_structural_warning_present, selector_warning_present, suspiciously_short_selected_text | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, short_single_part_question, visual_crop_scope_from_text_only |
| 33autumn21_q02 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |
| 32spring23_q03 | selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 11summer23_q01 | selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, short_single_part_question, visual_crop_scope_from_text_only |
| 12spring21_q02 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 33summer21_q07 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 52summer21_q04 | low_crop_confidence, selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | visual_crop_scope_from_text_only |
| 31summer23_q09 | selector_structural_warning_present, selector_warning_present | missing_crop_pixel_dimensions, missing_raw_candidate_windows | math_expression_semantics, visual_crop_scope_from_text_only |

## Fixture Warning Threshold

At least 20 fixture records received useful crop-context warnings (35).
