# Roadmap

This roadmap is evidence-gated, not feature-gated. Move forward when the extraction evidence supports it.

## v0.x: Extraction Correctness And Triage Loop

Goal: keep improving the base PDF-to-crop pipeline with measurable, repeatable triage and auto-triage handoffs.

Evidence required:

- Full test suite passes.
- Every improvement pass has a frozen baseline and a named comparison file.
- OCR-enabled runs are compared against OCR-enabled baselines.
- `worsened_records` is empty or every worsening is intentionally stricter and documented.
- New regression tests cover fixed triage examples.
- Auto-triage decisions are accepted only with passing tests, matching OCR mode, target improvement, and no broad trust-gate loosening.

Current evidence:

- Latest test run before docs: `293 passed, 3 skipped`.
- Latest OCR-to-OCR comparison: `output/triage/iteration_003/comparison.math-repair-ocr.json`, hard failures `385 -> 259`, target `126 -> 3`, worsened `0`.
- Latest current-output comparison: `output/triage/iteration_004/comparison.layout-review-current.json`, hard failures `259 -> 148`, target `114 -> 4`, worsened `0`, but not canonical for OCR because current output is no-OCR.
- Current OCR candidate: `output_ocr_candidate/json/question_bank.json`, OCR-enabled for all `1301` records, hard failures `133`, dominant target `paper_total_mismatch: 86`.
- Latest accepted auto-triage comparison: `output_ocr_candidate/triage/iteration_002/comparison.auto-iteration-003.json`, hard failures `153 -> 133`, `paper_total_mismatch` `107 -> 86`, worsened `0`.

Next required evidence:

- Source-pairing mismatch guard proves `33autumn25` cannot point at `12autumn21`.
- Fresh OCR-enabled output is compared to an OCR-enabled baseline after the guard.
- Continue auto-triage planning until the configured hard-failure threshold is met or the dominant target is no longer actionable without human source review.

## v0.y: Trusted-Subset Export

Goal: produce downstream-safe slices of `question_bank.json` without changing the canonical full bank.

Evidence required:

- Tier filters are implemented and tested.
- Export reports included/excluded counts by reason.
- Source-pairing mismatch is an exclusion reason.
- Missing images are exclusion reasons.
- Student practice export does not include failed mapping, failed validation, failed scope, or review-only records.

Suggested outputs:

```text
output/json/question_bank.trusted_image.json
output/json/question_bank.trusted_metadata.json
output/json/question_bank.trusted_practice.json
output/json/question_bank.trusted_rejections.json
```

Do not move to student-app integration until this exists.

## v0.z: Student-App Integration Contract

Goal: define exactly how a downstream student app consumes the extraction output.

Evidence required:

- Trusted-subset export exists.
- App contract documents required fields and fallback behavior.
- App renders image crops as canonical content.
- App shows text snippets only when trust gates allow it.
- App does not use DeepSeek/topic sidecar fields as canonical truth.
- App can explain why an item is excluded.

Suggested contract checks:

- Every student item has existing question and mark-scheme image files.
- Every student item has source question and mark-scheme metadata agreement.
- Every topic-routed item has `topic_trust_status=normal`.
- Every adaptive item has marks and difficulty metadata.

## v1.0: Stable Reviewed Corpus For Initial Paper Family

Goal: ship a reviewed, stable subset for one paper family before broad expansion.

Evidence required:

- Choose one initial family, likely `p1` or `p3`, based on trusted-subset size and review cost.
- All included records are Tier 3 or manually reviewed Tier 2.
- No known source-pairing mismatches.
- Mark-scheme crops visually verified for sampled and high-risk records.
- Topic labels have either high local confidence or reviewed sidecar agreement.
- A regression suite covers representative failures from that family.

Release criteria:

- Full tests pass.
- Triage comparison against the family baseline has no unexplained worsened records.
- Trusted export includes a rejection report.
- Documentation states exact included/excluded counts.

## Later

Only after the extraction trust loop stabilizes:

- Broaden reviewed coverage across all paper families.
- Improve reviewer UX around notes and fixture generation.
- Calibrate difficulty scores against real student performance data if available.
- Add app-specific analytics and adaptive behavior.
- Consider additional syllabuses only after CAIE 9709 is stable.
