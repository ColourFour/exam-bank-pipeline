# Mark-Event Evidence Contract

This contract covers deterministic mark-event evidence generated from the existing image-first question bank.

The sidecar is advisory evidence only. Canonical question crops and mark-scheme crops remain the source of truth. Mark events must not replace official mark-scheme images, infer student scores, or enable auto-marking.

## Outputs

Primary sidecar:

- `output/json/question_bank.mark_events.v1.json`

Review outputs:

- `output/reports/mark_events_audit.md`
- `output/reports/mark_events_review_queue.json`
- optional validation report: `output/json/question_bank.mark_events.validation.v1.json`

## Build Commands

```bash
.venv/bin/python scripts/build_mark_events.py \
  --question-bank output/json/question_bank.json \
  --artifact-root output \
  --out output/json/question_bank.mark_events.v1.json \
  --report output/reports/mark_events_audit.md \
  --review-queue output/reports/mark_events_review_queue.json
```

```bash
.venv/bin/python scripts/validate_mark_events.py \
  --question-bank output/json/question_bank.json \
  --mark-events output/json/question_bank.mark_events.v1.json \
  --artifact-root output \
  --output output/json/question_bank.mark_events.validation.v1.json
```

## Schema

The sidecar schema is `exam_bank.question_bank.mark_events` version `1`.

Top-level fields:

- `schema_name`
- `schema_version`
- `generated_at`
- `source_question_bank_path`
- `source_question_bank_sha256`
- `record_count`
- `records`

Each record links to one question-bank `question_id` and includes source metadata, mark-scheme image provenance, extraction status, advisory safety flags, detected and expected totals, part summaries, mark events, and review evidence.

Every generated record must set:

- `safe_for_marking_use: false`

`safe_for_advisory_use` may be true only when the source image exists, deterministic mark events are present, totals match, and no serious review flags are present.

## Evidence Rules

The parser recognizes common CAIE mark codes and annotations, including `M`, `A`, `B`, `E`, `DM`, follow-through markers, `AG`, `cao`, `oe`, `www`, `isw`, dependent, and independent notes.

Unknown or ambiguous semantics must remain review evidence. The parser should prefer `unknown`, `partial`, or `review` over invented dependencies, alternatives, or student-facing interpretations.

Lines without deterministic mark codes are not converted into mark events. Reviewable prose is retained in unparsed evidence so reviewers can improve future parsing rules.

## Validation Rules

Validation checks:

- sidecar schema and record count
- every record links to a known question-bank `question_id`
- source mark-scheme image paths exist unless the source question-bank record already has a known missing companion issue
- mark values are positive integers
- part paths are normalized lists
- detected totals do not exceed expected totals
- total mismatch flags are consistent
- unknown mark codes are flagged for review
- `safe_for_marking_use` is false for every generated record
- `safe_for_advisory_use` is not set when serious review flags are present

Validation warnings are allowed for review-only records. Validation errors indicate malformed sidecar structure or unsafe claims.

## Forbidden Uses

Do not use mark events to:

- auto-grade student work
- infer student scores
- replace official mark-scheme images
- promote AI or OCR text to canonical mark-scheme content
- bypass existing image-first, review, topic, difficulty, or generation gates

