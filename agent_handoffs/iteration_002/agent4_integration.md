# Agent 4 Integration Review — iteration_002

## Verdict

PASS for the quantified difficulty iteration.

The implementation is scoped to deterministic local difficulty scoring, difficulty metadata threading, JSON export contract updates, audit reporting, and tests. I found no blocking integration issues. The only substantive caveat is that the checked-in/stale `output/json/question_bank.json` has not been regenerated, so bank-level difficulty distribution cannot yet be audited from the full export.

## Commands and results

```bash
sed -n '1,240p' agent_handoffs/iteration_002/agent1_plan.md
sed -n '1,260p' agent_handoffs/iteration_002/Prompt/agent4_prompt.md
sed -n '1,260p' agent_handoffs/iteration_002/agent2_impl_notes.md
sed -n '1,260p' agent_handoffs/iteration_002/agent3_tests.md
```

Result: handoffs reviewed. Agent 1 scoped iteration 2 to difficulty metadata and explicitly deferred OCR work; Agents 2 and 3 reported scoped implementation plus passing tests.

```bash
git status --short
git diff --stat
git diff -- src/exam_bank/classification.py src/exam_bank/classification_models.py src/exam_bank/models.py src/exam_bank/pipeline.py src/exam_bank/exporters.py src/exam_bank/audit.py scripts/audit_difficulty.py tests/test_classification.py tests/test_output_contract.py tests/test_audit.py tests/test_deepseek_enrich.py
```

Result: changed files are difficulty classification/model/export/audit/test files plus handoff reports and `scripts/audit_difficulty.py`. No `tests/test_deepseek_enrich.py` changes were present in the diff.

```bash
.venv/bin/python -m pytest
```

Result:

```text
269 passed, 3 skipped in 137.96s (0:02:17)
```

```bash
.venv/bin/python - <<'PY'
from exam_bank.classification import classify_question
from exam_bank.config import AppConfig

cases = [
    ("routine", "Differentiate y = 3x^2 - 4x + 1. [2]", 2, "9709_s21_qp_12.pdf"),
    ("linked", "Express the rational function in partial fractions. Hence integrate the result with respect to x. [8]", 8, "9709_s21_qp_32.pdf"),
    ("weak", "???", None, "9709_s21_qp_12.pdf"),
]
for name, text, marks, source in cases:
    result = classify_question(text, marks=marks, config=AppConfig(), source_name=source)
    print(name, result.difficulty, result.difficulty_score, result.difficulty_band, result.difficulty_confidence, result.difficulty_review_flags)
PY
```

Result:

```text
routine easy 0 easy high []
linked average 60 average high []
weak easy 26 easy low ['marks_missing_for_difficulty', 'difficulty_uncertain']
```

```bash
.venv/bin/python scripts/audit_difficulty.py --input output/json/question_bank.json | sed -n '1,120p'
```

Result: command exits successfully. Current stale export has `record_count: 1301` and all difficulty metadata missing (`difficulty_label_counts: {"missing": 1301}`, `difficulty_score_summary.count: 0`).

```bash
git ls-files output input '*.pdf' '*.png' '*.csv' '.env' '.env.*' '*secret*' '*key*'
git ls-files -o --exclude-standard | rg '(^output/|\.env|secret|key|\.pdf$|\.png$|\.csv$|\.json$)' || true
```

Result: no untracked generated exports, PDFs, PNGs, CSVs, JSON files, env files, or secret/key-looking files are visible to git. Tracked `output` content is only `output/json/.gitkeep`; the many tracked PDFs are the existing `input/` source corpus, not generated exports. A local `.env` exists in the workspace but is not tracked.

## Scope review

- OCR: PASS. No OCR scoring, threshold, candidate-selection, or OCR export behavior changes are in the diff. Existing OCR references are in pre-existing OCR audit/tests/runtime code and iteration handoffs.
- DeepSeek: PASS. No DeepSeek provider behavior changes are in the diff. Existing DeepSeek normalization tests are included in the full-suite pass, and Agent 3 already ran the focused DeepSeek tests.
- Schema compatibility: PASS. `QUESTION_BANK_SCHEMA_VERSION` remains `2`; fields are additive and canonical labels remain `easy`, `average`, `difficult`.

## Data-contract findings

- Top-level export fields are present in `src/exam_bank/exporters.py`: `difficulty`, `difficulty_score`, and `difficulty_band`.
- `notes` export fields are present: `difficulty`, `difficulty_confidence`, `difficulty_evidence`, `difficulty_uncertain`, `difficulty_score`, `difficulty_score_scale`, `difficulty_features`, `difficulty_review_flags`, and `difficulty_model_version`.
- Score scale is explicit as `0-100`; band thresholds are named constants in `src/exam_bank/classification.py`: `0-34 easy`, `35-69 average`, `70-100 difficult`.
- `difficulty == difficulty_band` is tested for local classification and derived from score thresholds.
- Evidence is structured enough to audit: `difficulty_features` includes `marks`, `structure`, `cognitive_demand`, `mathematical_density`, `mixed_topic_complexity`, `topic_prior`, and `trust`; `difficulty_review_flags` are exported separately from general `review_flags`.
- Weak evidence cannot produce high-confidence difficulty in tested paths: missing marks plus degraded text produces low confidence, `difficulty_uncertain`, and `marks_missing_for_difficulty`; low topic confidence and low text quality also reduce confidence in the implementation.
- Representative fixture outputs are plausible for this iteration: routine low-mark differentiation is easy/high confidence, linked mixed work is average/high confidence, and weak unreadable text is easy but low confidence and flagged.

## Blockers

None for the iteration 2 code/data contract.

Audit blocker: the existing full-bank export is stale and missing difficulty metadata for all `1301` records, so real bank-level score distribution and per-paper/topic distribution cannot be validated until a fresh export is generated.

## Iteration 3 deferrals

- Generate a fresh full-bank or representative export with the additive difficulty fields and run `scripts/audit_difficulty.py` on it.
- Carry forward iteration 1 OCR work: fresh OCR-enabled candidate-aware export, OCR candidate audit, and manual review before any OCR threshold tuning.
- Do not tune OCR thresholds until fresh candidate metadata shows OCR-selected and high-margin rejected examples are safe.
- Calibrate difficulty against real student performance data only if that data becomes available and is explicitly approved.
- Consider bank-level review of score distribution by paper family, topic, and marks bucket after the fresh export exists.
