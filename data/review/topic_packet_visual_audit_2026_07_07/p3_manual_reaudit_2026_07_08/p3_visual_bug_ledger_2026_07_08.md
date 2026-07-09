# P3 Topic Packet Visual Bug Ledger

Date: 2026-07-08

Scope: P3 topic packets only. No Asterion, app export, or student-runtime export regeneration.

## Current Status

The P3 question PNGs were regenerated, the nine P3 topic packets were rebuilt from their difficulty sidecars, and fresh post-guard question contact sheets were reviewed:

- Question contact-sheet index: `data/review/topic_packet_visual_audit_2026_07_07/p3_manual_reaudit_2026_07_08/post_guard_question_page_index.json`
- Question contact sheets reviewed: `post_guard_question_pages_001.jpg` through `post_guard_question_pages_077.jpg`
- Mark-scheme suspect sheets reviewed earlier in this pass: `suspect_mark_scheme_pages_001.jpg` through `suspect_mark_scheme_pages_010.jpg`

Do not import complete visual-audit decisions for P3 yet. Unresolved source PNG and mark-scheme visual bugs remain.

## Packet Manifest Check

All rebuilt P3 packet manifests currently report no missing answers, no oversized-block warnings, and sidecar application counts matching included problem counts:

- `p3/algebra`: `problem_count=173`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=173`
- `p3/complex_numbers`: `problem_count=114`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=114`
- `p3/differential_equations`: `problem_count=87`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=87`
- `p3/differentiation`: `problem_count=156`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=156`
- `p3/integration`: `problem_count=119`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=119`
- `p3/logarithmic_and_exponential_functions`: `problem_count=111`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=111`
- `p3/numerical_solution_of_equations`: `problem_count=98`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=98`
- `p3/trigonometry`: `problem_count=99`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=99`
- `p3/vectors`: `problem_count=104`, `missing_answer_count=0`, `topic_difficulty_review_applied_count=104`

## Fixed During This Pass

- `p3/algebra` page 3, problem 19, `33winter11_q01`: source-crop/page-furniture issue is now clean in `output/audits/topic_packet_visual_audit_2026_07_07/p3_algebra/page_0003.png`.
- `p3/algebra` page 18, problem 114, `33winter24_q09`: dense answer-rule/barcode band is removed and remains clean in `output/audits/topic_packet_visual_audit_2026_07_07/p3_algebra/page_0018.png`.
- Vector/matrix rows that were being incorrectly removed by the dense-band cleaner are preserved again. Representative checked file: `output/pm3/pm3_2015_w15_32_qp_q07_question.png`.

## Unresolved Question Bugs

Severity key:

- `S1`: clipped/missing question content, duplicated diagram/text, wrong-question spillover, or unusable crop.
- `S2`: answer blank/rule, source footer, barcode/header/watermark/furniture that should not be in a question crop.
- `S3`: small source fragment that is readable but still not clean.

### Algebra

- `S3` page 6 problem 47, `33winter25_q01`, 2025 November P33 Question 1: source/header remnant above prompt.
- `S3` page 10 problem 74, `31summer16_q01`, 2016 June P31 Question 1: tiny stray remnant after the question.
- `S1` page 15 problem 100, `12summer19_q05`, 2019 June P12 Question 5: clipped/fragmented first line before vector definitions.
- `S2` page 19 problem 116, `32summer25_q10`, 2025 June P32 Question 10: dotted answer-rule line remains in the question.
- `S2` page 20 problem 122, `31summer12_q03`, 2012 June P31 Question 3: clipped/source contamination at the top of the prompt.

### Complex Numbers

- `S2` page 2 problem 10, `31summer24_q04`, 2024 June P31 Question 4: dotted answer blank between parts.
- `S2` page 7 problem 29, `32winter25_q03`, 2025 November P32 Question 3: top dense/source furniture line.
- `S2` page 8 problem 30, `31winter25_q05`, 2025 November P31 Question 5: top dense/source furniture line.
- `S3` page 9 problem 32, `32summer17_q08`, 2017 June P32 Question 8: top source fragments.
- `S2` page 9 problem 35, `33summer24_q06`, 2024 June P33 Question 6: black square marker below text.
- `S2` page 10 problem 39, `35summer25_q05`, 2025 June P35 Question 5: top dense/source furniture line.
- `S3` page 11 problem 41, `31summer17_q07`, 2017 June P31 Question 7: top source fragments.
- `S3` page 14 problem 51, `31winter16_q09`, 2016 November P31 Question 9: top source fragments.
- `S3` page 14 problem 52, `32winter16_q09`, 2016 November P32 Question 9: top source fragments.
- `S3` page 15 problem 54, `31summer12_q04`, 2012 June P31 Question 4: top source fragments.
- `S1` page 17 problem 61, `31summer10_q07`, 2010 June P31 Question 7: following-question spillover at bottom.
- `S3` page 18 problem 64, `33winter19_q06`, 2019 November P33 Question 6: top source fragments.
- `S3` page 18 problem 65, `33summer17_q11`, 2017 June P33 Question 11: top source fragments.
- `S2` page 19 problem 67, `32winter09_q07`, 2009 November P32 Question 7: top/source fragments and right-edge artifact.
- `S3` page 20 problem 71, `32winter17_q07`, 2017 November P32 Question 7: top source fragments.
- `S1` page 24 problem 82, `31winter11_q10`, 2011 November P31 Question 10: large diagonal watermark fragment inside question page.
- `S3` page 26 problem 89, `32summer16_q10`, 2016 June P32 Question 10: top source fragment.
- `S3` page 27 problem 92, `33summer19_q08`, 2019 June P33 Question 8: top source fragment.
- `S2` page 28 problem 96, `32summer24_q09`, 2024 June P32 Question 9: dotted answer/source rule after part (c).
- `S2` page 29 problem 98, `31winter24_q08`, 2024 November P31 Question 8: dotted answer line between parts.
- `S2` page 33 problem 110, `03winter08_q10`, 2008 November P03 Question 10: small black source mark above question.
- `S3` page 35 problem 114, `31summer19_q10`, 2019 June P31 Question 10: top source fragments.

### Differential Equations

- `S3` page 7 problem 34, `31winter25_q08`, 2025 November P31 Question 8: small source row above prompt, low confidence.
- `S1` page 15 problem 55, `32summer11_q06`, 2011 June P32 Question 6: duplicated/foreign first-sentence material at question start.

### Differentiation

- `S3` page 2 problem 9, `33winter15_q02`, 2015 November P33 Question 2: sparse fragments above prompt.
- `S3` page 2 problem 10, `31winter24_q03`, 2024 November P31 Question 3: possible clipped/top-contaminated first line.
- `S2` page 3 problem 20, `32summer25_q04`, 2025 June P32 Question 4: dense barcode/header remnant above prompt.
- `S3` page 5 problem 29, `31winter09_q04`, 2009 November P31 Question 4: leftover source row before next problem.
- `S1` page 12 problem 59, `31summer16_q05`, 2016 June P31 Question 5: following-question spillover at bottom.
- `S3` page 16 problem 77, `31summer23_q07`, 2023 June P31 Question 7: possible slightly clipped bottom line, low confidence.
- `S2` page 21 problem 97, `32summer10_q06`, 2010 June P32 Question 6: source fragments around first line.
- `S1` page 34 problem 140, `33winter19_q08`, 2019 November P33 Question 8: duplicated diagram/question material in same crop.
- `S2` page 36 problem 145, `32winter20_q05`, 2020 November P32 Question 5: dotted answer-rule line after part (a).
- `S1` page 38 problem 147, `33winter11_q08`, 2011 November P33 Question 8: duplicate/foreign diagram content remains.

### Integration

- `S3` page 2 problem 11, `31summer14_q02`, 2014 June P31 Question 2: top sparse fragments above integral.
- `S3` page 11 problem 50, `32winter09_q06`, 2009 November P32 Question 6: lower-confidence top fragments.
- `S2` page 15 problem 61, `32winter25_q10`, 2025 November P32 Question 10: source-artifact row across top of tank question crop.
- `S2` page 24 problem 85, `35winter25_q07`, 2025 November P35 Question 7: dotted blank-like row in prompt.
- `S2` page 27 problem 94, `31summer24_q10`, 2024 June P31 Question 10: long dotted rule between parts.
- `S2` page 34 problem 108, `32summer12_q09`, 2012 June P32 Question 9: source pagination footer retained in question crop.
- `S3` page 36 problem 113, `31winter09_q09`, 2009 November P31 Question 9: stray source mark above diagram.

### Logarithmic And Exponential Functions

- `S1` page 5 problem 36, `33summer11_q01`, 2011 June P33 Question 1: right-clipped/truncated sentence.
- `S1` page 11 problem 62, `31winter10_q02`, 2010 November P31 Question 2: corrupted question start.
- `S1` page 11 problem 63, `32winter10_q02`, 2010 November P32 Question 2: corrupted question start.
- `S2` page 12 problem 68, `33summer25_q02`, 2025 June P33 Question 2: retained barcode/header strip above question.
- `S3` page 16 problem 89, `32summer11_q02`, 2011 June P32 Question 2: corrupted question-start marker.

### Numerical Solution Of Equations

- `S1` page 3 problem 8, `32summer16_q03`, 2016 June P32 Question 3: bottom-truncated, final instruction incomplete.
- `S2` page 5 problem 11, `31summer20_q06`, 2020 June P31 Question 6: dotted answer rule between parts.
- `S2` page 7 problem 17, `32summer12_q02`, 2012 June P32 Question 2: right-edge black source marker near diagram.
- `S1` page 16 problem 36, `31summer10_q06`, 2010 June P31 Question 6: duplicated/partial repeated geometry diagram plus right-edge marker.
- `S1` page 20 problem 46, `31winter11_q05`, 2011 November P31 Question 5: bottom-truncated before final instruction is complete.
- `S2` page 29 problem 65, `33winter16_q09`, 2016 November P33 Question 9: source pagination footer retained.
- `S3` page 32 problem 70, `32winter22_q09`, 2022 November P32 Question 9: stray source mark above diagram.
- `S2` page 33 problem 72, `32winter21_q11`, 2021 November P32 Question 11: dotted blank line between parts.
- `S1` page 38 problem 82, `31winter12_q08`, 2012 November P31 Question 8: duplicated diagram/text and right-edge source marks.
- `S1` page 39 problem 83, `32winter12_q08`, 2012 November P32 Question 8: duplicated diagram/text and right-edge source marks.
- `S2` page 43 problem 91, `32summer21_q10`, 2021 June P32 Question 10: visible dotted blank/header artifact.
- `S2` page 44 problem 92, `33summer25_q11`, 2025 June P33 Question 11: top source header/artifact row.

### Trigonometry

- `S3` page 3 problem 13, `33winter12_q02`, 2012 November P33 Question 2: possible clipped word in prompt, low confidence.
- `S2` page 5 problem 25, `32winter09_q04`, 2009 November P32 Question 4: stray previous/source line above question.
- `S1` page 10 problem 49, `31winter21_q05`, 2021 November P31 Question 5: corrupted first line.
- `S2` page 11 problem 58, `33summer18_q07`, 2018 June P33 Question 7: source fragments above question text.
- `S2` page 15 problem 74, `35winter25_q08`, 2025 November P35 Question 8: blank/rule fragment between parts.
- `S2` page 21 problem 98, `31summer11_q09`, 2011 June P31 Question 9: source pagination footer retained.

### Vectors

- `S1` page 7 problem 19, `33winter21_q08`, 2021 November P33 Question 8: duplicated prompt text beneath diagram.
- `S2` page 9 problem 25, `33summer16_q08`, 2016 June P33 Question 8: stray fragment before actual question.
- `S2` page 19 problem 55, `32winter25_q11`, 2025 November P32 Question 11: top source furniture plus dotted blank line.
- `S2` page 28 problem 82, `33summer23_q09`, 2023 June P33 Question 9: top source fragments.
- `S2` page 33 problem 94, `32winter18_q10`, 2018 November P32 Question 10: embedded source pagination note.
- `S2` page 35 problem 100, `31winter13_q09`, 2013 November P31 Question 9: embedded source pagination note.
- `S2` page 36 problem 101, `32winter13_q09`, 2013 November P32 Question 9: embedded source pagination note.

## Unresolved Mark-Scheme Bugs

Blocking mark-scheme failures:

- `p3/algebra` problem 17, `33summer20_q01`, answer page 39: generic note/header only; no usable worked mark scheme.
- `p3/integration` problem 1, `33summer20_q02`, answer page 42: generic note only; no usable worked mark scheme.
- `p3/logarithmic_and_exponential_functions` problem 97, `33summer20_q03`, answer page 55: generic note only; no usable worked mark scheme.
- `p3/trigonometry` problem 9, `33summer20_q05`, answer page 25: generic note fragment only; no usable worked mark scheme.
- `p3/numerical_solution_of_equations` problem 2, `32winter09_q02`, answer page 50: severe crop contamination with paper header, watermark, PapaCambridge content, and neighboring question content.

Dirty but readable mark-scheme crop bugs:

- `p3/algebra`: problems 83 `33winter19_q02`, 100 `12summer19_q05`, 105 `33winter24_q05`, 146 `12winter24_q09`: readable answer content with trailing table-header fragments.
- `p3/differential_equations`: problem 61 `33summer17_q08`: trailing table header.
- `p3/differentiation`: problems 87 `33winter21_q07`, 100 `33summer19_q04`, 150 `32winter23_q09`: table-header or split-boundary tails.
- `p3/logarithmic_and_exponential_functions`: problem 68 `33summer25_q02`: table-header tail.
- `p3/numerical_solution_of_equations`: problems 44 `33summer17_q06`, 51 `32winter23_q06`, 62 `31summer18_q08`, 68 `31winter18_q03`, 69 `33winter18_q03`: table-header tails.
- `p3/trigonometry`: problem 57 `33winter19_q04`: table-header tail before next answer title.
- `p3/vectors`: problem 5 `31summer17_q06`: table-header tail.
- `p3/vectors`: problem 35 `32summer24_q08`: repeated table headers after answer content.

Duplicate or wrong mark-scheme mapping/crop:

- `p3/vectors` page 57 problem 23 `33winter17_q10` and page 58 problem 24 `31winter17_q10`: same answer crop appears repeated for two different packet problems.

Suspect but mostly readable mark-scheme crops:

- `p3/complex_numbers`: problems 41 `31summer17_q07`, 87 `31winter18_q08`, 88 `33winter18_q08`.
- `p3/differentiation`: problems 53 `33summer17_q05`, 105 `32winter17_q04`, 132 `33summer17_q07`.
- `p3/integration`: problem 72 `33summer20_q07`.
- `p3/numerical_solution_of_equations`: problems 31 `31summer17_q05`, 71 `33summer20_q06`.
- `p3/trigonometry`: problem 37 `31summer19_q04`.
- `p3/vectors`: problems 39 `33summer18_q10`, 49 `32summer17_q06`.

## Verification

Completed:

- Rebuilt all nine P3 packets from their difficulty sidecars.
- Rebuilt all nine P3 visual-audit batches.
- Regenerated 1061 unique P3 source question PNGs.
- Reviewed fresh question contact sheets `post_guard_question_pages_001.jpg` through `post_guard_question_pages_077.jpg`.
- Checked rebuilt manifests for missing answers, sidecar counts, and warnings.
- `.venv/bin/python -m pytest tests/test_image_limits.py tests/test_image_rendering.py tests/test_question_png_segmentation.py tests/test_topic_packet_visual_audit.py tests/test_topic_packets.py tests/test_visual_topic_audit.py`: `199 passed, 5 warnings`.
- `git diff --check`: clean.
