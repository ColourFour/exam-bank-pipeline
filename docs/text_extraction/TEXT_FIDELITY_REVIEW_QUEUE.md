# Text Fidelity Review Queue

This queue ranks advisory text records for manual fidelity review. Canonical question and mark-scheme images remain the source of truth.

## Summary

- Records inspected: 1301
- Records with non-zero queue score: 841
- Known bad fixtures found: 36 / 36
- Known bad fixtures in top 50: 36
- Known bad fixtures in top 100: 36
- Top reason codes: likely_math_symbol_loss=650, ocr_native_disagreement=301, suspiciously_short_text=77, known_fixture_membership=36, clean_visual_crop_but_degraded_text=35, selected_ocr_with_structural_warnings=33

## Reason Weights

| Reason code | Weight |
| --- | ---: |
| `known_fixture_membership` | 200 |
| `next_question_contamination` | 70 |
| `selected_ocr_with_structural_warnings` | 70 |
| `clean_visual_crop_but_degraded_text` | 65 |
| `missing_question_number` | 60 |
| `lost_subpart_labels` | 55 |
| `missing_marks` | 50 |
| `likely_math_symbol_loss` | 45 |
| `ocr_native_disagreement` | 40 |
| `suspiciously_short_text` | 35 |

## Top 50 Review Items

| Rank | Record | Score | Reasons | Explanation |
| ---: | --- | ---: | --- | --- |
| 1 | `52spring22_q06` | 430 | `known_fixture_membership`, `missing_question_number`, `lost_subpart_labels`, `likely_math_symbol_loss`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; Expected question number 6 is not the leading anchor.; Expected subparts ['a', 'b', 'c', 'd', 'e'], present ['a', 'b', 'c', 'd', 'e']. Selected text begins at (b). Subpart (b) appears before (a). |
| 2 | `41summer23_q06` | 375 | `known_fixture_membership`, `missing_question_number`, `likely_math_symbol_loss`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; Expected question number 6 is not the leading anchor.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 3 | `43autumn21_q06` | 375 | `known_fixture_membership`, `missing_question_number`, `likely_math_symbol_loss`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; Expected question number 6 is not the leading anchor.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 4 | `12spring22_q08` | 370 | `known_fixture_membership`, `ocr_native_disagreement`, `missing_question_number`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=1.00, score_gap=33.; Expected question number 8 is not the leading anchor. |
| 5 | `32summer23_q09` | 370 | `known_fixture_membership`, `ocr_native_disagreement`, `missing_question_number`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=1.00, score_gap=46.; Expected question number 9 is not the leading anchor. |
| 6 | `11summer23_q01` | 345 | `known_fixture_membership`, `suspiciously_short_text`, `likely_math_symbol_loss`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; Selected text score is -7.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 7 | `13summer23_q01` | 340 | `known_fixture_membership`, `ocr_native_disagreement`, `suspiciously_short_text`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=0.47, score_gap=8.; Selected text score is 37. |
| 8 | `53summer22_q07` | 335 | `known_fixture_membership`, `selected_ocr_with_structural_warnings`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; OCR was selected while selector or structure metadata contains warnings.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 9 | `33autumn21_q04` | 320 | `known_fixture_membership`, `ocr_native_disagreement`, `suspiciously_short_text`, `likely_math_symbol_loss` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=0.62, score_gap=10.; Selected text score is -7. |
| 10 | `11autumn23_q04` | 310 | `known_fixture_membership`, `likely_math_symbol_loss`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 11 | `12autumn23_q06` | 310 | `known_fixture_membership`, `likely_math_symbol_loss`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 12 | `12summer21_q03` | 310 | `known_fixture_membership`, `ocr_native_disagreement`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=1.00, score_gap=33.; OCR was selected while selector or structure metadata contains warnings. |
| 13 | `12summer23_q01` | 310 | `known_fixture_membership`, `likely_math_symbol_loss`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 14 | `51summer21_q05` | 310 | `known_fixture_membership`, `ocr_native_disagreement`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=1.00, score_gap=54.; OCR was selected while selector or structure metadata contains warnings. |
| 15 | `52spring21_q05` | 310 | `known_fixture_membership`, `ocr_native_disagreement`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=1.00, score_gap=37.; OCR was selected while selector or structure metadata contains warnings. |
| 16 | `32spring23_q03` | 305 | `known_fixture_membership`, `ocr_native_disagreement`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=0.69, score_gap=1.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 17 | `35summer25_q04` | 295 | `known_fixture_membership`, `missing_marks`, `likely_math_symbol_loss` | Record is in the frozen bad-text fixture manifest.; Selected text has no [n] mark bracket despite mark metadata.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 18 | `31summer22_q09` | 285 | `known_fixture_membership`, `ocr_native_disagreement`, `likely_math_symbol_loss` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=0.66, score_gap=20.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 19 | `31summer23_q09` | 285 | `known_fixture_membership`, `ocr_native_disagreement`, `likely_math_symbol_loss` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=0.07, score_gap=20.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 20 | `33autumn25_q07` | 285 | `known_fixture_membership`, `missing_marks`, `suspiciously_short_text` | Record is in the frozen bad-text fixture manifest.; Selected text has no [n] mark bracket despite mark metadata.; Selected text score is 38. |
| 21 | `42autumn21_q01` | 285 | `known_fixture_membership`, `ocr_native_disagreement`, `likely_math_symbol_loss` | Record is in the frozen bad-text fixture manifest.; OCR/native candidate diagnostics disagree; selected/OCR similarity=0.59, score_gap=13.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 22 | `32autumn21_q04` | 280 | `known_fixture_membership`, `suspiciously_short_text`, `likely_math_symbol_loss` | Record is in the frozen bad-text fixture manifest.; Selected text score is -12.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 23 | `33autumn21_q05` | 280 | `known_fixture_membership`, `suspiciously_short_text`, `likely_math_symbol_loss` | Record is in the frozen bad-text fixture manifest.; Selected text score is -5.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 24 | `52summer21_q02` | 270 | `known_fixture_membership`, `selected_ocr_with_structural_warnings` | Record is in the frozen bad-text fixture manifest.; OCR was selected while selector or structure metadata contains warnings. |
| 25 | `32spring23_q04` | 265 | `known_fixture_membership`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 26 | `51autumn23_q05` | 265 | `known_fixture_membership`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 27 | `52summer22_q02` | 265 | `known_fixture_membership`, `clean_visual_crop_but_degraded_text` | Record is in the frozen bad-text fixture manifest.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 28 | `13autumn25_q09` | 250 | `known_fixture_membership`, `missing_marks` | Record is in the frozen bad-text fixture manifest.; Selected text has no [n] mark bracket despite mark metadata. |
| 29 | `33summer24_q03` | 250 | `known_fixture_membership`, `missing_marks` | Record is in the frozen bad-text fixture manifest.; Selected text has no [n] mark bracket despite mark metadata. |
| 30 | `33summer21_q07` | 245 | `known_fixture_membership`, `likely_math_symbol_loss` | Record is in the frozen bad-text fixture manifest.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 31 | `12spring21_q02` | 235 | `known_fixture_membership`, `suspiciously_short_text` | Record is in the frozen bad-text fixture manifest.; Selected text score is -7. |
| 32 | `31summer21_q01` | 235 | `known_fixture_membership`, `suspiciously_short_text` | Record is in the frozen bad-text fixture manifest.; Selected text score is -7. |
| 33 | `32summer21_q01` | 235 | `known_fixture_membership`, `suspiciously_short_text` | Record is in the frozen bad-text fixture manifest.; Selected text score is -7. |
| 34 | `33autumn21_q02` | 235 | `known_fixture_membership`, `suspiciously_short_text` | Record is in the frozen bad-text fixture manifest.; Selected text score is 12. |
| 35 | `52autumn21_q01` | 200 | `known_fixture_membership` | Record is in the frozen bad-text fixture manifest. |
| 36 | `52summer21_q04` | 200 | `known_fixture_membership` | Record is in the frozen bad-text fixture manifest. |
| 37 | `31autumn23_q02` | 185 | `ocr_native_disagreement`, `suspiciously_short_text`, `likely_math_symbol_loss`, `clean_visual_crop_but_degraded_text` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.93, score_gap=54.; Selected text score is -12.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 38 | `33autumn23_q02` | 175 | `ocr_native_disagreement`, `selected_ocr_with_structural_warnings`, `clean_visual_crop_but_degraded_text` | OCR/native candidate diagnostics disagree; selected/OCR similarity=1.00, score_gap=53.; OCR was selected while selector or structure metadata contains warnings.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 39 | `12spring22_q10` | 155 | `ocr_native_disagreement`, `likely_math_symbol_loss`, `selected_ocr_with_structural_warnings` | OCR/native candidate diagnostics disagree; selected/OCR similarity=1.00, score_gap=36.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings.; OCR was selected while selector or structure metadata contains warnings. |
| 40 | `12autumn23_q02` | 145 | `suspiciously_short_text`, `likely_math_symbol_loss`, `clean_visual_crop_but_degraded_text` | Selected text score is 39.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 41 | `32spring23_q01` | 145 | `suspiciously_short_text`, `likely_math_symbol_loss`, `clean_visual_crop_but_degraded_text` | Selected text score is -3.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 42 | `12spring24_q01` | 140 | `ocr_native_disagreement`, `suspiciously_short_text`, `clean_visual_crop_but_degraded_text` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.69, score_gap=0.; Selected text score is -7.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 43 | `32autumn23_q05` | 140 | `ocr_native_disagreement`, `suspiciously_short_text`, `clean_visual_crop_but_degraded_text` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.62, score_gap=0.; Selected text score is -7.; Question crop confidence is high while selected advisory text is degraded or low-scoring. |
| 44 | `43summer22_q07` | 140 | `ocr_native_disagreement`, `lost_subpart_labels`, `likely_math_symbol_loss` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.64, score_gap=17.; Expected subparts ['a', 'b'], present ['a', 'b']. Subpart (b) appears before (a).; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 45 | `12spring21_q03` | 120 | `ocr_native_disagreement`, `suspiciously_short_text`, `likely_math_symbol_loss` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.58, score_gap=46.; Selected text score is 39.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 46 | `13summer25_q03` | 120 | `ocr_native_disagreement`, `suspiciously_short_text`, `likely_math_symbol_loss` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.69, score_gap=5.; Selected text score is 39.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 47 | `13summer25_q05` | 120 | `ocr_native_disagreement`, `suspiciously_short_text`, `likely_math_symbol_loss` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.83, score_gap=46.; Selected text score is -7.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 48 | `31autumn25_q07` | 120 | `ocr_native_disagreement`, `suspiciously_short_text`, `likely_math_symbol_loss` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.80, score_gap=37.; Selected text score is 39.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 49 | `32spring23_q11` | 120 | `ocr_native_disagreement`, `suspiciously_short_text`, `likely_math_symbol_loss` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.69, score_gap=2.; Selected text score is 14.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |
| 50 | `32spring25_q11` | 120 | `ocr_native_disagreement`, `suspiciously_short_text`, `likely_math_symbol_loss` | OCR/native candidate diagnostics disagree; selected/OCR similarity=0.67, score_gap=2.; Selected text score is -7.; Math-heavy text has symbol-loss indicators, suspicious glyphs, or OCR rejection symbol warnings. |

## Fixture Rank Summary

- All known bad fixtures found in the bank ranked in the top 50.

## Advisory Boundary

The queue is a review aid only. It does not change OCR/native selection, production exports, canonical images, or `question_bank.json`.
