# Iteration 007 Residual Failure Burndown Report

Verdict: ACCEPTED WITH CAVEATS

The revised-policy hard-failure and visual-fail targets were met through parser/crop/mapping repairs. The text-only and untrusted math-text targets were not met, so this is not a full clean pass.

## Audit Sanity Checks

- Trusted candidate path: `output/json/question_bank.json`.
- Reason: this pass continued the no-OCR default export. The earlier `output_ocr_candidate/json/question_bank.json` was the separate OCR candidate from the prior review. The current regenerated export has `ocr_ran=false` for all 1301 records.
- Count discrepancy: the prior revised-policy count of 44 was for the OCR candidate. The 39 count was for the no-OCR default candidate at `output/json/question_bank.json`.
- The 39 count was under the revised triage policy, not the old strict mark-total policy.
- Final `output/json/audit.current.json` was regenerated from the final `output/json/question_bank.json`.
- Direct recount matched `audit.current.json` for `record_count`, `question_text_role_counts`, `question_text_trust_counts`, `visual_required_counts`, `visual_curation_status_counts`, `text_only_status_counts`, and `visual_reason_flag_counts`.
- Record count: 1301.
- Scope: `non_9709_sources=0`, `beyond_sources=0`.
- Pairing safety: `paired_paper_code_mismatches=0`.
- Missing exact mark scheme safety: all 11 `33autumn25` records remain unmatched with blank `mark_scheme_source_pdf` and blank `mark_scheme_image_path`; no fallback/fuzzy wrong-paper pairing was used.
- Image path sanity: `question_empty=0`, `question_missing=0`, `mark_scheme_missing=0`; the only `mark_scheme_empty=11` records are the expected `33autumn25_q01` through `33autumn25_q11`.
- `public/data/question_bank.json` unchanged.

## Final Metrics

Before Iteration 007, from the audited no-OCR candidate:

- Revised hard failures: 39
- Old strict hard failures: 114
- `visual_curation_status`: `fail=28`, `ready=668`, `review=605`
- `text_only_status`: `fail=45`, `ready=224`, `review=1032`
- `question_text_role`: `readable_text=330`, `search_hint=952`, `untrusted_math_text=19`
- `question_text_trust`: `high=330`, `medium=950`, `low=17`, `unusable=4`

After Iteration 007, from the final regenerated candidate:

- Revised hard failures: 20
- Old strict hard failures: 23
- `visual_curation_status`: `fail=9`, `ready=681`, `review=611`
- `text_only_status`: `fail=26`, `ready=228`, `review=1047`
- `question_text_role`: `readable_text=332`, `search_hint=954`, `untrusted_math_text=15`
- `question_text_trust`: `high=332`, `medium=952`, `low=17`
- `paper_total_status`: `matched=1301`
- `ocr_ran`: `false=1301`

Target status:

- Hard failures below 25: met, 39 -> 20.
- Visual fail below 10: met, 28 -> 9.
- Text-only fail below 20: missed, 45 -> 26.
- Untrusted math text below 5: missed, 19 -> 15.

## Repairs Made

- Mark-scheme pairing repair: kept the prior canonical exact-match behavior intact. Missing exact mark schemes return unmatched/fail instead of fuzzy-pairing to nearby papers.
- Mark-total extraction repair: fixed special-case/alternative mark-scheme branches so SC rows do not add extra marks to the main method, while legitimate total rows immediately before SC/alternative guidance remain usable.
- Mark-total extraction repair: fixed multi-bracket question totals so real part marks like `[2] [3] [5]` sum to 10, while a standalone final total line like `[5]` is still treated as an overall total.
- Mark-total extraction repair: fixed nested alpha/roman subpart validation so `(a)(i)`, `(a)(ii)`, `(b)(i)` style questions compare against top-level mark-scheme subparts without false gaps or duplicate-label contamination.
- Crop/image repair: retained low-confidence first-body question starts when the anchor is a plausible left-margin question prompt; this restored `33summer25_q04` as its own record instead of merging it into Q3.
- Crop/image repair: retained the prior inline quantity false-start guard so `51summer25_q05` is not split at “6 musicians...”.
- Text extraction repair: recovered embedded first alpha labels such as `origin(a)` where PDF text extraction collapsed the whitespace before `(a)`.
- Test expectation update: updated the N24 P12 sample expectation because Q1 is now a real 5/5 mark-total match, so the previous `question_mark_total_review_only` expectation is stale.

Improved record classes observed:

- Crop/image repairs: `33summer25_q04`, `51summer25_q05`, `51summer25_q06`.
- Mark-total extraction repairs: `32spring22_q05`, `33summer23_q09`, `33summer23_q11`, `42autumn22_q01`, `42summer23_q04`, `52autumn23_q06`, `12autumn24_q01`, `12autumn24_q08`, plus prior repaired rows preserved for `51summer22_q04`, `52autumn23_q06`, `12autumn21_q12`, and related paper-total recoveries.
- Nested subpart/crop-scope repairs: `43autumn21_q04`, `42summer21_q05`, `52spring21_q07`, `53summer21_q07`, `12autumn24_q05`, `15autumn25_q11`.
- Text extraction repair: `33summer21_q09`.
- Validation-policy reclassification: none used as the primary mechanism for final gains.
- OCR/native text selection repair: none; final export remains no-OCR.
- Counting/reporting correction: clarified `output` vs `output_ocr_candidate` and revised vs strict policy counts.

## Remaining Failure Clusters

Revised hard failures remaining: 20.

- `partial_question_block`: 11 records, all `33autumn25_q01` through `33autumn25_q11`, caused by missing exact `9709_2025_November_33` mark scheme. These are safely unmatched/review/fail and should not be product-ready.
- `weak_question_anchor`: 9 records with matching question/mark totals but weak anchors:
  - `13autumn24_q04`
  - `11summer25_q01`
  - `13summer25_q05`
  - `32summer25_q04`
  - `33summer25_q02`
  - `35summer25_q02`
  - `12autumn25_q02`
  - `15autumn25_q05`
  - `32autumn25_q04`

Other residuals:

- `text_only_status=fail`: 26 records remain. These are still unsafe for text-only student use.
- `question_text_role=untrusted_math_text`: 15 records remain. These should continue to use canonical question images, not extracted text, as product source of truth.
- `question_text_trust=low`: 17 records remain.
- `mark_scheme_crop_confidence=medium`: 464 records remain, mostly review-grade rather than hard failures.

## Verification

- Full export: completed, `Processed questions: 1301`, `Processed papers: 148`, output `output/json/question_bank.json`.
- Audit regenerated: `output/json/audit.current.json`.
- Direct recomputation: matched audit-owned counts exactly.
- Full test suite: `325 passed, 3 skipped`.
- `git diff --check`: passed.
- `public/data/question_bank.json`: unchanged.

## Plain-English Summary For Blake

Trust the final number 20 for revised-policy hard failures in `output/json/question_bank.json`. The old strict recount is 23 on the same file.

What improved: wrong-paper mark-scheme fallback stayed fixed; all paper totals now match; hard failures dropped from 39 to 20; visual fail dropped from 28 to 9; several real mark-total, nested-subpart, and crop-boundary defects were repaired.

What did not improve enough: text-only fail is still 26 and untrusted math text is still 15. Those records remain unsafe for student/product use as extracted text.

What remains unsafe: the 11 `33autumn25` records have no exact mark scheme and are intentionally unmatched; the 9 weak-anchor records need further crop/anchor review; the remaining text-fail/untrusted records should use canonical images, not extracted text, for student-facing output.
