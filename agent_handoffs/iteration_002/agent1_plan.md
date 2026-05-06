# Agent 1 Plan — Exam Bank Pipeline iteration_002

## 1. Iteration 1 carry-forward

Iteration 1 was accepted as OCR candidate-selection reporting infrastructure only. The agents agreed that the implementation was scoped, tested, and honest about missing data, but it did not validate actual OCR selection quality because the available export was stale or candidate-unaware.

Carry-forward facts from iteration 1:

- `output/json/question_bank.json` has `1301` records.
- All six OCR candidate metadata fields are missing on all records:
  - `text_candidate_source`
  - `text_candidate_decision`
  - `ocr_selected`
  - `native_text_score`
  - `ocr_text_score`
  - `selected_text_score`
- The OCR audit command exists:
  - `.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json`
- Baseline comparison remains blocked by missing baseline export.
- Real OCR quality review remains blocked until a fresh OCR-enabled full-bank export exists.
- Agent 5's recommended next target was a fresh OCR-enabled export and audit.

Decision for this iteration:

- Defer Agent 5's OCR target to iteration 3.
- Do not tune OCR thresholds in iteration 2.
- Do not regenerate the full bank for OCR in iteration 2 unless it is incidental to validating difficulty export fields.
- Keep iteration 1's OCR stop conditions alive for iteration 3.

## 2. Iteration 2 target

Quantify and clarify question difficulty while preserving the current coarse difficulty label contract.

Current difficulty is effectively only a label-scale decision (`easy`, `average`, `difficult` internally; user-facing language may refer to `easy`, `medium`, `hard`). The next step is to produce auditable numeric difficulty metadata that explains why a question sits in a bucket and gives downstream tools more resolution than three labels.

Iteration 2 should add a local, deterministic difficulty assessment layer based on existing extraction/classification signals:

- numeric difficulty score
- score scale definition
- coarse label derived from score
- confidence
- evidence features
- review flags for low-quality or contradictory difficulty evidence
- export/audit coverage

## 3. Current repo findings relevant to difficulty

Code-level findings:

- Local difficulty is inferred in `src/exam_bank/classification.py` by `_infer_difficulty`.
- `_infer_difficulty` currently computes a private score, then discards it and returns only:
  - `difficulty`
  - `difficulty_confidence`
  - `difficulty_evidence`
  - `difficulty_uncertain`
  - `numeric_confidence`
  - `review_flags`
- `src/exam_bank/runtime_profile.json` defines difficulty labels as:
  - `easy`
  - `average`
  - `difficult`
- DeepSeek enrichment normalizes user/model labels:
  - `medium` -> `average`
  - `hard` -> `difficult`
- `src/exam_bank/exporters.py` currently does not export local difficulty fields in top-level JSON or `notes`.
- The existing `output/json/question_bank.json` has no `notes.difficulty`, `notes.difficulty_confidence`, or `notes.difficulty_evidence`; all `1301` records are missing difficulty in the export.

Bank-level findings from the current stale export:

- Records by paper family:
  - `p1`: 401
  - `p3`: 396
  - `p4`: 258
  - `p5`: 246
- Average detected solution marks by paper family:
  - `p1`: about `7.06`
  - `p3`: about `6.97`
  - `p4`: about `7.20`
  - `p5`: about `7.52`
- Because difficulty is not exported, current bank-level difficulty distribution cannot be audited.

## 4. Non-goals

- No OCR threshold or scoring changes.
- No DeepSeek provider behavior changes.
- No broad topic-classification rewrite.
- No schema-breaking removal or rename of existing coarse difficulty labels.
- No generated full-bank exports committed to git.
- No change that makes text the student-facing source of truth.
- No adaptive trainer work beyond supplying clearer metadata.
- No attempt to calibrate difficulty against real student performance data unless such data is already present and explicitly approved.

## 5. Proposed difficulty model

Add a deterministic local model that returns both a coarse label and numeric details.

Recommended score scale:

- `difficulty_score`: integer `0` to `100`
- `difficulty_band`: one of `easy`, `average`, `difficult`
- `difficulty_label`: preserve existing canonical label values
- `difficulty_confidence`: existing `high`, `medium`, `low`
- `difficulty_features`: structured evidence used to compute score
- `difficulty_review_flags`: difficulty-specific audit flags

Initial banding:

- `0-34`: `easy`
- `35-69`: `average`
- `70-100`: `difficult`

The exact thresholds should be implemented as named constants or config-adjacent runtime constants, not scattered literals.

Initial feature groups:

- Marks:
  - low marks lower the score
  - high marks raise the score
  - missing marks lowers confidence and adds a review flag
- Structure:
  - number of subparts
  - linked part count
  - continuation wording such as `hence`, `deduce`, `using your answer`
- Cognitive demand:
  - direct routine command
  - proof/show/deduce wording
  - interpretation/justify/comment wording
  - modelling/contextual wording
- Mathematical density:
  - symbol/operator density
  - number of formula-like lines
  - presence of multi-step algebra/calculus/statistics notation
- Topic prior:
  - paper-family routine topics lower the score
  - paper-family later-paper or difficult topics raise the score
- Mixed-topic complexity:
  - secondary topics and close topic-score competition raise score only when the text is good enough
- Trust modifiers:
  - degraded or unusable text lowers confidence, not necessarily the raw score
  - weak topic confidence lowers difficulty confidence
  - missing marks, mapping failures, or validation failures add review flags

The goal is not perfect psychometrics in iteration 2. The goal is traceable, inspectable scoring that improves over opaque labels.

## 6. Data contract proposal

Preserve existing labels and add detail fields. Recommended export placement:

Top-level fields:

- `difficulty`
- `difficulty_score`
- `difficulty_band`

`notes` fields:

- `difficulty_confidence`
- `difficulty_evidence`
- `difficulty_uncertain`
- `difficulty_score`
- `difficulty_score_scale`: `0-100`
- `difficulty_features`
- `difficulty_review_flags`
- `difficulty_model_version`

Rationale:

- Top-level `difficulty` keeps downstream access simple.
- `difficulty_score` and `difficulty_band` make sorting/filtering possible without parsing notes.
- `notes` keeps explanatory and model-version details traceable.
- Duplicating score in `notes` is acceptable if existing export style keeps operational metadata there; otherwise put the score only top-level and document it.

Schema compatibility:

- Increment `QUESTION_BANK_SCHEMA_VERSION` only if the project treats additive fields as schema-version changes. If current tests expect version `2`, update tests intentionally.
- Keep `easy`, `average`, `difficult` canonical internally. If user-facing `medium`/`hard` labels are needed later, add a display mapping rather than changing canonical labels in iteration 2.

## 7. Implementation plan for Agent 2

Agent 2 should implement the scoring layer and export contract narrowly.

Suggested steps:

1. Introduce a structured difficulty assessment result.
   - Extend `DifficultyDecision` or add a nested dataclass for numeric details.
   - Keep existing fields intact for compatibility.
2. Refactor `_infer_difficulty`.
   - Keep current heuristics as the starting point.
   - Convert the private floating score into a normalized `0-100` score.
   - Return structured feature contributions.
   - Preserve current coarse-label behavior unless tests show the previous thresholds were accidental.
3. Thread details through classification state and record models.
   - `ClassificationResult`
   - `QuestionClassificationState`
   - `QuestionRecord`
   - `to_*_state` helpers if needed
4. Export difficulty fields.
   - Add top-level `difficulty`, `difficulty_score`, `difficulty_band`.
   - Add explanatory fields under `notes`.
5. Add or update tests.
   - Unit tests for easy/average/difficult score boundaries.
   - Missing marks lowers confidence and flags difficulty evidence.
   - Direct routine low-mark questions remain easy.
   - High-mark linked/mixed questions score higher.
   - Degraded text lowers confidence or marks uncertain.
   - Export contract includes difficulty score/details.
6. Add a lightweight difficulty audit if small.
   - Prefer `src/exam_bank/audit.py` if existing audit helpers fit.
   - Otherwise add `scripts/audit_difficulty.py`.
   - Report distribution by paper family, topic, marks decile/bucket, label, score range, and confidence.

## 8. Test plan for Agent 3

Agent 3 should verify behavior and guard against overclaiming.

Required tests:

- `classify_question` returns numeric difficulty details for fixture questions.
- Score is bounded in `0..100`.
- Coarse label is derived consistently from the score.
- Missing marks produces a review flag and no fake high confidence.
- Low-quality text does not receive high difficulty confidence.
- Export includes difficulty fields for every record.
- Existing DeepSeek normalization tests still pass.
- Existing output contract tests are updated only for additive fields.

Recommended regression fixtures:

- 2-3 mark direct differentiation/integration question -> low score/easy.
- 7-9 mark linked `hence` or multi-part question -> middle to high score.
- 10+ mark mixed-topic or proof-style question -> high score/difficult.
- Missing marks with weak text -> low confidence and review flag.

## 9. Integration audit for Agent 4

Agent 4 should check the implementation boundary and sample outputs.

Audit items:

- No OCR scoring/threshold changes.
- No DeepSeek behavior changes except tests remaining compatible.
- No topic taxonomy churn.
- Difficulty fields are present in exported JSON.
- Difficulty score is deterministic.
- Score explanations are concise and machine-readable.
- Difficulty label distribution is plausible across paper families.
- High scores are not caused only by noisy OCR/native text.
- Low confidence is surfaced for weak text, missing marks, or failed validation.
- Generated exports remain untracked.

Suggested command set:

```bash
.venv/bin/python -m pytest tests/test_classification.py tests/test_output_contract.py tests/test_deepseek_enrich.py -q
.venv/bin/python -m pytest -q
```

If a difficulty audit script exists:

```bash
.venv/bin/python scripts/audit_difficulty.py --input output/json/question_bank.json --json-output /tmp/difficulty_audit_iteration_002.json
```

## 10. Final review for Agent 5

Agent 5 should accept iteration 2 only if it gives the project a usable quantified difficulty contract.

Acceptance criteria:

- Full tests pass.
- Difficulty is exported for all newly built records.
- Numeric difficulty score is bounded and documented.
- Coarse label remains compatible with the current canonical label set.
- Evidence/features explain the score without relying on opaque prose only.
- Missing or weak evidence lowers confidence and adds review flags.
- No OCR or DeepSeek scope creep.
- No generated bank artifacts are accidentally tracked.

Reject or require fixes if:

- Difficulty remains only a coarse label.
- Score exists but cannot be explained by structured evidence.
- Score is not deterministic.
- Existing labels are renamed in a schema-breaking way.
- Weak or missing text receives high-confidence difficulty.
- Tests only check field presence and not scoring behavior.

## 11. Optional measurement after implementation

If time allows after implementation and tests:

1. Generate a small or full local export without OCR scope changes.
2. Run a difficulty audit.
3. Manually inspect representative records:
   - low-score examples
   - boundary easy/average examples
   - boundary average/difficult examples
   - high-score examples
   - low-confidence examples

Representative table columns:

- `question_id`
- `paper_family`
- `topic`
- `question_solution_marks`
- `difficulty`
- `difficulty_score`
- `difficulty_confidence`
- `difficulty_features`
- `difficulty_review_flags`
- manual judgment: too low / plausible / too high

## 12. Iteration 3 target preserved

Iteration 3 should return to the Agent 5 iteration 1 target:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_iteration_003 --enable-ocr
.venv/bin/python scripts/audit_ocr_candidates.py --input output_iteration_003/json/question_bank.json --json-output /tmp/ocr_candidate_audit_iteration_003.json
```

Iteration 3 should review OCR-selected or high-margin rejected records before any threshold tuning.

Iteration 3 stop conditions carried forward:

- Full suite fails.
- Fresh export still lacks candidate metadata.
- OCR-selected records lose question numbers, subparts, or mark brackets.
- OCR-selected records include page furniture, headers, barcode fragments, diagram-label contamination, or next-question text.
- OCR selection occurs on hard scope, mapping, or validation failures.
- `text_only_status`, `visual_curation_status`, or `question_text_trust` improves without clear text-quality evidence.
- Generated exports, PDFs, or secrets become tracked.
- Any OCR scoring change is attempted before reviewing fresh candidate-aware audit evidence.
