# Difficulty Index Contract

This contract covers the deterministic advisory difficulty sidecar generated at `output/json/question_bank.difficulty_index.v1.json` and the Markdown reports under `reports/`.

The sidecar is advisory metadata only. Canonical question images and mark-scheme images remain the source of truth. The difficulty index must not replace official source material, strict topic routing, Asterion role gates, or human review.

## Outputs

Primary sidecar:

- `output/json/question_bank.difficulty_index.v1.json`

Review reports:

- `reports/difficulty_index_summary.md`
- `reports/difficulty_index_by_paper.md`
- `reports/difficulty_index_review_queue.md`

## Build Command

```bash
.venv/bin/python scripts/generate_difficulty_index.py \
  --input output/json/question_bank.json \
  --output output/json/question_bank.difficulty_index.v1.json \
  --reports-dir reports \
  --artifact-root output \
  --mark-events output/json/question_bank.mark_events.v1.json \
  --topic-routing output/json/question_bank.topic_routing.v1.json \
  --advisory-evidence output/advisory_evidence/question_bank.advisory_evidence.v1.json
```

Use `--dry-run` to print the summary without writing the sidecar or reports.

## Interpretation

`difficulty_index_0_100` is an internal advisory ordering score. It is not a psychometric measurement, a candidate success-rate estimate, or a statement that a question with score `80` is twice as difficult as a question with score `40`.

`paper_relative_difficulty_band` is the downstream-friendly field. It assigns band `1` through `5` within each paper after sorting by the advisory index:

- `1`: easiest quintile within paper
- `2`: lower-middle quintile within paper
- `3`: middle quintile within paper
- `4`: upper-middle quintile within paper
- `5`: hardest quintile within paper

The current v1 sidecar can support teacher/reviewer filtering when `safe_for_teacher_filtering=true`. It does not enable student-facing sequencing; generated records currently keep `safe_for_student_sequencing=false`.

## Evidence Inputs

The builder reads:

- canonical extraction metadata from `output/json/question_bank.json`
- mark totals and mark-event safety from `output/json/question_bank.mark_events.v1.json`
- strict-topic safety from `output/json/question_bank.topic_routing.v1.json`
- examiner-report and grade-threshold context from `output/advisory_evidence/question_bank.advisory_evidence.v1.json`
- canonical image availability under `output/`

Missing or unsafe evidence lowers confidence, creates warnings, or moves records into the review queue. Grade-threshold context can provide component/session context only; it must not directly prove individual-question difficulty.

## Review Rules

Records need review when confidence is `low` or `unsafe`, when unsafe reasons are present, or when important evidence is missing or review-required.

Unsafe reasons include missing canonical images, missing mark-scheme images, failed mapping or validation, question-total/mark-scheme-total disagreement, serious mark-event flags, and unsafe advisory evidence.

The review queue is intentionally conservative. Review entries do not mean the canonical image artifact is unusable; they mean the advisory placement is not strong enough to rely on without human calibration.

## Forbidden Uses

Do not use the difficulty index to:

- claim psychometric validity or candidate success rates
- sequence student-facing practice directly in v1
- override Asterion role gates
- override strict topic-routing safety metadata
- repair or replace canonical question-bank fields
- replace rendered question or mark-scheme images
- infer item difficulty from grade thresholds alone
