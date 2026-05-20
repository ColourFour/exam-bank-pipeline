# Crop Context Gate Implementation

Date: 2026-05-20

This implementation adds crop/context diagnostics as advisory review metadata only. It does not change selected text, raw native text, raw OCR text, OCR/native source selection, canonical question or mark-scheme image assets, question-bank JSON semantics, or Asterion exports.

## Implemented Now

The deterministic warning source is `compute_crop_context_warnings` in `src/exam_bank/crop_text_signals.py`. `compute_crop_context_warning_codes` returns sorted practical-now codes for stable report and queue output.

Implemented practical-now warning codes:

| Warning code | Trigger summary |
| --- | --- |
| `selector_warning_present` | Existing selector, validation, extraction-quality, or text-fidelity flags are present. |
| `selector_structural_warning_present` | Existing review flags or selector decision reasons indicate weak anchors, merged text, lost labels, or related structural risk. |
| `low_crop_confidence` | `question_crop_confidence` exists and is not `high`. |
| `missing_expected_question_number` | An expected question number is known but absent from selected advisory text. |
| `missing_mark_bracket` | Mark metadata or expectations require Cambridge-style marks and selected text has no `[n]` bracket. |
| `suspiciously_short_selected_text` | Selected text has very few tokens, is much shorter than fixture context, or existing structure metadata reports very short text. |
| `clean_visual_crop_but_degraded_text` | Crop confidence is high while selected advisory text is degraded or low-scoring. |
| `selected_ocr_with_structural_warnings` | OCR is selected and selector/structure metadata also contains structural warnings. |
| `possible_next_question_contamination` | Strong text or structure evidence indicates the selected text may continue into the next question. |
| `missing_subpart_labels` | Expected subpart labels are known from fixture, bank, or structure metadata and are absent from selected text. |

The crop signal audit may still emit additional non-approved diagnostic codes for report context, such as missing metadata or ambiguous/risky text-only checks. Those are not treated as practical-now review gates unless their `gate_candidate` is `practical_now`.

## Review Queue Integration

`src/exam_bank/text_review_queue.py` now calls the crop/context warning function for each question-bank record. Practical-now warnings are exposed on each queue entry as:

- `crop_context_warning_codes`
- `crop_context_warnings`

The same practical-now codes are also added to `reason_codes` with advisory weights, so they can affect rank and reason-code worklists. Existing text-fidelity reasons remain in place for continuity.

## Deferred

The following remain deferred because they need image-reviewed labels or richer crop metadata:

- Automatic OCR/native source switching.
- Any overwrite of `question_text`, `ocr_text`, native text, normalized text, or candidate text.
- Any canonical image crop failure classification from text-only evidence.
- Asterion export changes.
- Layout-aware math recovery as selected text.
- OCR profile routing as production source selection.
- Pixel-dimension, normalized-area, line-bbox, and raw-candidate-window gates beyond report diagnostics.

## Regression Boundary

Tests cover warning generation, queue scoring, deterministic output, and non-mutation of selected text and canonical image fields. The implementation only reads existing record metadata and writes separate report/review metadata.
