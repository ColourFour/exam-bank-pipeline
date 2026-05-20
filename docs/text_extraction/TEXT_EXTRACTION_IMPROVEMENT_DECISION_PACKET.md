# Text Extraction Improvement Decision Packet

Date: 2026-05-20

Scope: final decision packet for the five proposed advisory text extraction improvements. This document uses the current fixture reports, text-fidelity audits, crop-context audit, math-normalization spike, OCR-profile experiment, and review-queue report. It does not change extraction behavior, OCR/native selection, canonical images, question-bank JSON, or Asterion exports.

## Source-Of-Truth Boundary

This project remains image-first. Canonical question images and mark-scheme images are the source of truth. Native PDF text, OCR text, normalized text, profile OCR text, review queue scores, topic labels, difficulty labels, and any future advisory text candidates are secondary metadata for search, triage, and review workflows only.

No recommendation below permits advisory text to overwrite canonical image evidence. Any production implementation must keep raw selected text, raw OCR/native candidates, advisory candidate text, provenance, confidence, and warnings separated.

## Evidence Base

| Evidence file | Current status |
| --- | --- |
| `docs/text_extraction/TEXT_EXTRACTION_FAILURE_AUDIT.md` | 36 concrete bad-text examples across P1/P3/P4/P5; 1301 records inspected; 42 degraded text records; 33 OCR-selected records; 77 records with selected text score <= 40; 7 missing leading question numbers; 5 missing mark brackets. |
| `tests/fixtures/text_fidelity/bad_text_records.json` | Frozen 36-record fixture manifest used by current reports and experiments. |
| `output/reports/text_fidelity_fixture_baseline_normalized.json` | Refreshed on 2026-05-20; 36 fixtures; status counts fail=24, pass=8, warn=4; failure types include expectation_gap=17, source_disagreement=12, structural_rejection=11, math_symbol_loss=8, mark_bracket=6, question_anchor=5, subpart_labels=1. |
| `output/reports/crop_text_signal_audit.json` and `docs/text_extraction/CROP_CONTEXT_SIGNAL_AUDIT.md` | 36 fixture records audited; 35 records with useful warnings; 35 caught by practical-now gates; key available warnings include low crop confidence=23, selector structural warning=33, selector warning=27, missing expected question number=5, missing mark bracket=4. |
| `output/reports/ocr_profile_experiment.json` and `docs/text_extraction/OCR_PROFILE_EXPERIMENT.md` | 36 fixture records tested against OCR profiles; `formula_heavy` best average score at 91.25 with 13 improvements and 10 regressions; every non-baseline profile had material regressions. |
| `output/reports/text_fidelity_review_queue.json` and `docs/text_extraction/TEXT_FIDELITY_REVIEW_QUEUE.md` | 1301 records inspected; 841 queued; all 36 known bad fixtures ranked in top 50 and top 100. Top reason codes: likely_math_symbol_loss=650, ocr_native_disagreement=301, suspiciously_short_text=77, clean_visual_crop_but_degraded_text=35, selected_ocr_with_structural_warnings=33. |
| `docs/text_extraction/MATH_NORMALIZATION_SPIKE.md` | Advisory-only normalized candidates improve 4 fixtures at issue level, make 26 clearer for failure classification, and leave 6 unchanged. Normalization remains warning-bearing and non-canonical. |

## Decision Summary

| # | Proposed improvement | Decision | Next action | Why |
| ---: | --- | --- | --- | --- |
| 1 | Add layout-aware math text recovery for common CAIE patterns. | Conditional go for narrow advisory candidates only. | Keep experimental. | Fixture evidence proves math layout failures, but reliable recovery needs image-reviewed ground truth and per-rule precision checks. |
| 2 | Use image crop context to gate OCR/native text selection. | Go for warning/review gates; no automatic source switching yet. | Implement now. | 35/36 known bad fixtures are caught by practical-now crop/context warnings, with low source-of-truth risk if used as review gates. |
| 3 | Introduce a math-normalized text candidate layer. | Go only as separated advisory candidate layer. | Keep experimental. | Normalization produces measurable report value on 30/36 fixtures, but only 4 issue-level improvements and several inferred repairs require visual review. |
| 4 | Add targeted OCR profiles by paper/layout family. | No-go for production selection; go for report-only profile candidates. | Keep experimental. | Profiles improve some anchor/mark cases but regress symbol-heavy calculus/trig/integral cases. |
| 5 | Build a text-fidelity review queue from concrete failure signals. | Go. | Implement now. | The queue ranks all 36 known bad fixtures in the top 50 out of 1301 records, using concrete non-canonical signals. |

No proposal is approved to change canonical source-of-truth policy. No proposal is approved to change Asterion exports until a later export-contract decision explicitly adds advisory-only fields with provenance and warnings.

## Improvement 1: Layout-Aware Math Text Recovery

**Recommendation:** keep experimental.

**Evidence found:** The failure audit shows repeated CAIE math layout failures: powers, fractions, radicals, integrals, trig arguments, derivatives, vectors, complex numbers, inequalities, and diagram/math reading order. The normalized fixture baseline still has 24 failures and 4 warnings across 36 fixtures. Failure type counts include math_symbol_loss=8, source_disagreement=12, structural_rejection=11, expectation_gap=17, and mark_bracket=6.

**Number/type of fixture failures addressed:** Potentially addresses many P1/P3 math fixtures, especially examples involving derivative notation, powers, radicals, trig/log notation, inequalities, vectors, fractions, and integrals. The current normalized-candidate spike only resolves 4 issue-level failures. It improves classification for 26 more records but does not prove text correctness.

**Measurability:** Measurable through fixture expectation resolution, per-rule precision, warning counts, candidate-vs-selected diffs, and image-reviewed ground truth. Current fixture-only measurement is insufficient for go-live recovery because some repairs are inferred from local patterns.

**Implementation difficulty:** High. Correct recovery requires layout-aware parsing, candidate provenance, confidence scoring, visual review links, and per-pattern tests. Text-only regex repairs are not enough for display math semantics.

**Risk level:** High if promoted to selected text; medium if kept as advisory candidates.

**False-positive risks:** A flattened expression can be "repaired" into a plausible but wrong formula, such as an incorrect root span, trig argument, exponent, fraction boundary, or vector. This is especially dangerous because the text may look cleaner while becoming less faithful.

**False-negative risks:** Conservative rules will miss many valid layout failures, especially integrals, matrices, diagrams, table-aligned expressions, and split-line algebra.

**Canonical policy impact:** Must not affect canonical policy. Recovered math may only be advisory and must point users back to the question image.

**Asterion export impact:** No current export impact. Do not include recovered math in Asterion exports unless a future export contract adds explicit advisory candidate fields with provenance and warning flags.

**Rollback strategy:** Keep behind a report-only flag or separate candidate field. Roll back by disabling the candidate-generation flag and removing candidate fields from downstream reports; leave raw selected/native/OCR text untouched.

**Decision:** Do not implement as production recovery now. Continue as experimental layout-aware candidate generation after image-reviewed ground truth is added.

## Improvement 2: Crop Context Gates For OCR/Native Selection

**Recommendation:** implement now as advisory/review gates, not automatic OCR/native replacement.

**Evidence found:** The crop-context audit found useful warnings for 35/36 fixtures and practical-now gates caught 35/36. Existing available signals include low_crop_confidence=23, selector_structural_warning_present=33, selector_warning_present=27, missing_expected_question_number=5, missing_mark_bracket=4, and suspiciously_short_selected_text=2. The failure audit also shows 21/33 OCR-selected records with low question crop confidence and 5 degraded records despite high crop confidence.

**Number/type of fixture failures addressed:** Addresses structural and selection-risk failures across 35 known bad fixtures: missing question anchors, missing marks, OCR-selected records with structural warnings, low crop confidence, clean visual crop plus degraded text, and selector warning metadata.

**Measurability:** Strong. Gate hit rate, known-fixture recall, false-positive rate on clean records, queue rank movement, selected-source stability, and downstream JSON/export hash checks are measurable.

**Implementation difficulty:** Medium. Most signals already exist in question-bank metadata or fixture reports. Stronger gates need new metadata for crop pixel dimensions, raw candidate windows, and line bounding boxes.

**Risk level:** Low if gates only add review flags or prevent over-trusting text; medium if they automatically switch selected source.

**False-positive risks:** Short valid questions, legitimate answer-space instructions, or low-confidence crops with acceptable text may be over-queued. This is acceptable for review queues but not as a hard rejection without combined evidence.

**False-negative risks:** Some mathematically wrong selected text can have good crop context and no obvious anchor/mark failure, such as complex-number denominator corruption.

**Canonical policy impact:** None if implemented as warnings and review gates. The canonical image remains the authority.

**Asterion export impact:** None for internal reports. If exposed later, only export review metadata as advisory, never as canonical text replacement.

**Rollback strategy:** Feature-flag the gates and keep them additive. Roll back by disabling gate scoring or removing the new warning codes from reports. Since source selection is not mutated, rollback does not require regenerating canonical image assets.

**Decision:** Implement now as additive text-fidelity warnings, queue signals, and selection-risk diagnostics. Defer automatic source switching until raw candidate windows and image-reviewed labels exist.

## Improvement 3: Math-Normalized Text Candidate Layer

**Recommendation:** keep experimental, with a clear path to advisory-only implementation.

**Evidence found:** The math-normalization spike generated separate advisory candidates without changing production extraction. It reported classification counts: improved=4, clearer_failure_classification=26, unchanged=6. It found measurable improvement or clearer classification on 30/36 fixtures. Flag counts included spacing_artifacts_normalized=25, subpart_line_breaks_inserted=22, unicode_math_glyphs_normalized=9, subpart_labels_spaced=6, fraction_notation_normalized=4, inequality_notation_normalized=4, power_notation_normalized=3, trig_log_notation_normalized=3, derivative_notation_normalized=2, root_notation_normalized=2, and vector_matrix_notation_normalized=2.

**Number/type of fixture failures addressed:** Directly improves 4 issue-level failures: derivative notation, inequality notation, lost square glyphs, and vector/matrix classification. It helps classify 26 additional failures but does not repair them enough to claim reliable text.

**Measurability:** Strong for fixture reports and per-rule diffs. Insufficient for production display until each rule has precision/recall against image-reviewed ground truth.

**Implementation difficulty:** Medium. The report-layer prototype exists. Production-quality candidate layering needs schema decisions, candidate provenance, confidence/warning fields, and tests that raw fields are preserved.

**Risk level:** Medium. Low for report-only candidate fields; high if normalized text replaces `question_text`.

**False-positive risks:** Inferred repairs can create attractive but wrong text: `?` to `^{2}`, `cos 21` to `cos(2θ)`, root span inference, vector/matrix glyph substitutions, and `x20` to `x > 0`.

**False-negative risks:** Conservative normalization will not handle many actual layout errors, including integral bounds, complex fractions, table/diagram reading order, and missing mark brackets.

**Canonical policy impact:** None if stored as `question_text_normalized` or similar advisory candidate with `normalization_is_advisory=true`. It must never overwrite raw selected text.

**Asterion export impact:** No current impact. Keep out of Asterion exports for now. A future export change should require an advisory evidence contract update.

**Rollback strategy:** Keep candidates generated only by opt-in report flag or isolated field. Roll back by disabling normalized candidate generation and retaining raw selected/native/OCR text.

**Decision:** Keep experimental. Next step is an advisory candidate schema proposal and per-rule image-reviewed precision tests, not production replacement.

## Improvement 4: Targeted OCR Profiles By Paper/Layout Family

**Recommendation:** keep experimental; reject production default promotion.

**Evidence found:** The OCR profile experiment covered 36 fixtures with no runtime blockers. `formula_heavy` had the best average fixture score at 91.25 with 13 improved, 10 regressed, and 13 unchanged records. `dense_algebra` improved 11 and regressed 10. `grayscale_threshold` improved 12 and regressed 11. `padding_variant` improved 11 and regressed 11. `diagram_safe` improved 10 and regressed 11. `table_preserving` improved 5 and regressed 11. Non-uniform family/layout results suggest routing may matter, but the regressions are too material for automatic selection.

**Number/type of fixture failures addressed:** Helps some anchor-sensitive, mark-bracket, and structural failures, including cases where OCR recovers question anchors or marks. It regresses symbol-heavy calculus, trig, integrals, theta, complex algebra, and some diagram/table prompts.

**Measurability:** Strong in fixture experiments: average score, pass/warn/fail, improved/regressed counts, runtime, family/layout slices, and per-record diffs. Needs image-reviewed OCR snippets before any production routing.

**Implementation difficulty:** Medium to high. Running profiles is feasible, but safe routing requires layout classification, profile output persistence, candidate provenance, regressions review, and strict gating.

**Risk level:** High if used to select production text; medium as report-only OCR candidates.

**False-positive risks:** OCR profiles can recover anchors or brackets while damaging mathematical symbols, creating selected text that passes structural checks but changes meaning.

**False-negative risks:** Profile routing may miss records outside known fixture families, and the best average profile still fails or regresses meaningful cases.

**Canonical policy impact:** None if profile outputs remain candidate text for review. Must not affect canonical image status.

**Asterion export impact:** No current impact. Do not export profile-selected text as final text.

**Rollback strategy:** Keep profile execution isolated from the extraction pipeline. Roll back by disabling profile experiments or candidate persistence. Since selected text is not changed, no canonical regeneration is needed.

**Decision:** Reject production promotion now. Keep report-only profile experiments and add a routing experiment by layout family.

## Improvement 5: Text-Fidelity Review Queue

**Recommendation:** implement now.

**Evidence found:** The refreshed review queue inspected 1301 records and queued 841 with non-zero score. It found 36/36 known fixtures and ranked all 36 in the top 50 and top 100. Top reason codes were likely_math_symbol_loss=650, ocr_native_disagreement=301, suspiciously_short_text=77, known_fixture_membership=36, clean_visual_crop_but_degraded_text=35, selected_ocr_with_structural_warnings=33, missing_question_number=5, missing_marks=5, and lost_subpart_labels=3.

**Number/type of fixture failures addressed:** Addresses all 36 known bad fixtures as review-prioritization targets. It does not fix text directly; it makes failure review tractable and measurable.

**Measurability:** Very strong. Known-fixture recall, top-N capture, reason-code distribution, reviewer outcomes, queue churn, and regression stability are all measurable.

**Implementation difficulty:** Low to medium. The builder and reports already exist. Next work is hardening CI/report integration and adding reviewed-state handling.

**Risk level:** Low. The queue is additive and advisory.

**False-positive risks:** 841/1301 queued records is broad. Many `likely_math_symbol_loss` records may be acceptable for search metadata, so reviewer workload must be controlled by rank and reason filters.

**False-negative risks:** Unknown failure modes may not score highly, especially if they lack OCR/native disagreement or obvious structural warnings.

**Canonical policy impact:** None. The queue should point reviewers to canonical question/mark-scheme images for truth.

**Asterion export impact:** None unless a future export adds advisory review status. Current recommendation is no export change.

**Rollback strategy:** Disable queue generation or remove it from CI/report publication. It does not mutate source selection, canonical crops, or export records.

**Decision:** Implement now as the primary operational lane for text-fidelity improvement.

## Suggested Implementation Order

1. Implement the review queue as the operational control surface. Preserve the current reason-code model, top-N fixture recall metric, and links to canonical question/mark-scheme images.
2. Add crop-context and structural gates as additive warnings feeding the queue. Start with practical-now signals: selector warnings, low crop confidence, missing question anchor, missing mark bracket, selected OCR with structural warnings, clean crop plus degraded text, and suspiciously short text only in combination.
3. Add missing metadata needed for safer future gates: crop pixel dimensions, normalized crop area, raw candidate windows with rejected reasons, and text-line bounding boxes linked to candidate spans.
4. Keep math-normalized candidates report-only. Add a candidate schema with provenance, confidence, flags, warnings, and raw-field preservation tests.
5. Continue OCR profile experiments as report-only candidates. Add layout-family routing experiments and require per-record snippet review before any source-selection proposal.
6. After image-reviewed ground truth exists, revisit layout-aware math recovery as a narrow advisory candidate layer with per-rule precision thresholds.

## Recommended Next Goals

1. **Goal 9: Review Queue Hardening**
   - Add stable queue output checks, top-N known-fixture recall assertions, reviewer-facing image links, and reviewed/unreviewed state handling.

2. **Goal 10: Crop Context Gate Implementation**
   - Add additive warning fields for practical-now crop/context gates without changing selected text. Include regression tests that `question_bank.json`, canonical image paths, and Asterion exports are unchanged unless explicitly requested.

3. **Goal 11: Candidate Metadata Capture**
   - Persist crop dimensions, normalized crop area, raw candidate windows, rejection reasons, and text-line bounding boxes for replay and measurement.

4. **Goal 12: Advisory Normalized Candidate Contract**
   - Define a separate normalized candidate schema with provenance, confidence, warning flags, and explicit non-canonical semantics.

5. **Goal 13: OCR Profile Routing Experiment**
   - Run layout-family profile routing against fixtures and reviewed samples. Report improvements and regressions separately; do not change production source selection.

## Current Validation Status

Commands run before this packet was written:

```bash
.venv/bin/python scripts/report_text_fidelity_fixtures.py --include-normalized
.venv/bin/python scripts/build_text_fidelity_review_queue.py
```

Current report results:

| Command | Result |
| --- | --- |
| `report_text_fidelity_fixtures.py --include-normalized` | Passed; wrote `output/reports/text_fidelity_fixture_baseline_normalized.json` and `.md`; status counts fail=24, pass=8, warn=4; normalization classifications clearer_failure_classification=26, improved=4, unchanged=6. |
| `build_text_fidelity_review_queue.py` | Passed; wrote JSON/Markdown queue reports and `docs/text_extraction/TEXT_FIDELITY_REVIEW_QUEUE.md`; inspected 1301 records; queued 841; all 36 known bad fixtures in top 50/top 100. |

Targeted validation run after this packet was written:

```bash
.venv/bin/python -m pytest tests/test_text_fidelity_fixtures.py tests/test_text_review_queue.py tests/test_text_normalization.py
```

Result: passed, 17 tests.

Full-suite validation:

```bash
.venv/bin/python -m pytest
```

Result: passed, 532 tests passed, 3 skipped, 5 warnings in 141.64s.

## Final Go/No-Go

| Improvement | Final recommendation |
| --- | --- |
| Layout-aware math text recovery | No production go. Keep experimental as advisory candidate generation after image-reviewed ground truth and per-rule precision checks. |
| Crop context to gate OCR/native text selection | Go now for additive review gates and diagnostics. No automatic source switching. |
| Math-normalized text candidate layer | Go only as an experimental advisory candidate layer. Do not export or overwrite selected text. |
| Targeted OCR profiles | No production go. Keep as report-only experiments and routing candidates. |
| Text-fidelity review queue | Go now. Use it as the primary implementation and measurement surface. |

The safe path is therefore not "make text reliable." The safe path is to make text failures visible, measurable, and reviewable while preserving canonical image authority.
