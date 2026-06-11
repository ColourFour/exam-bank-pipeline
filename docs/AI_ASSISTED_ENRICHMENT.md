# AI-Assisted Enrichment

The extraction pipeline remains image-first. `question_bank.json` and the canonical taxonomy files under `exam_bank_taxonomy/canonical/` are the source of truth. DeepSeek enrichment is sidecar metadata only.

## Sidecar Versions

`question_bank.deepseek.json` is the older DeepSeek sidecar. It stores broad topic, subtopic, difficulty, confidence, rationale, review flags, and reconciliation fields. It is still readable and can be used as evidence for a v2 run.

`question_bank.topic_routing.v1.json` is the current Asterion-facing DeepSeek topic sidecar. It is deliberately narrow and records only canonical parent-topic routing against `exam_bank_taxonomy/canonical/`:

- `primary_topic_id`
- `topic_distribution`
- `confidence`
- `review_required`
- `review_reasons`
- `evidence_used`

It does not ask for or store difficulty, subtopics, skills, rationales, Content Lab metadata, Asterion readiness, or student-facing explanations. It also does not claim image evidence; `evidence_used` may list only supplied text evidence such as `question_text`, `ocr_text`, and `mark_scheme_text`. Records with `review_required=true` are review-only and must not enter strict Asterion topic filters. The full sidecar contract is in [Topic Routing Sidecar Contract](TOPIC_ROUTING_SIDECAR_CONTRACT.md).

`question_bank.ai_assisted.v2.json` is the broader canonical-ID AI-assisted sidecar. It preserves useful v1 fields when present, then adds `ai_assisted_items` for whole-question and optional subpart mappings. Each item must use existing canonical IDs for:

- `primary_topic_id`
- `secondary_topic_ids`
- `subtopic_ids`
- `skill_ids`
- `prerequisite_skill_ids`

DeepSeek may suggest taxonomy gaps only in `suggested_new_subtopic` and `suggested_new_skill`. Those fields are review-only and are never treated as canonical IDs. The broad v2 sidecar is currently review/debug evidence rather than the preferred Asterion strict-filter input.

## Product-Safe Fields

Product strict filters should consume only audited canonical IDs from an approved sidecar.

Asterion topic routing should consume audited, non-review-required records from `question_bank.topic_routing.v1.json`. The topic sidecar is parent-topic only and should be used for topic distribution/filtering after audit. Consumers must require `metadata.run_summary.safe_for_strict_filters=true` before using it for strict filters; if this field is missing or false, the sidecar is advisory/review-only.

Asterion region or skill routing from the broad v2 sidecar should consume only `strict_filter_candidates`, and only the canonical IDs inside those candidates, after the sidecar has passed an explicit audit. These candidates are produced after deterministic validation has confirmed that all IDs exist in the active canonical taxonomy and that strict filtering did not fail closed. Region routing should use `strict_filter_candidates[].asterion_region_ids`, derived from canonical skill-map `asterion_region_id` values. If a record has no strict candidates, route it to review or broad fallback behavior, not a strict student-facing filter.

Content Lab may use `ai_assisted_items[].worked_example_seed` and `ai_assisted_items[].warmup_seed` as lesson-planning seeds. They are not generated lessons and should still be paired with visual evidence and mark-scheme review.

The static Asterion learning runtime must not use broad AI labels or Content Lab candidates to promote records. Course-aware learning-runtime loading is limited to records explicitly exported with `learning_runtime_safe=true`, `student_runtime_safe=true`, and `review_status=reviewed` in the Asterion course contract. P3 legacy runtime is preserved. For P1, M1, and S1, the Asterion export may use the narrow topic-routing sidecar only as record-level route evidence for catalog/image practice: the individual route must have no error, `review_required=false`, high or medium confidence, and a non-empty, duplicate-free distribution totaling `100` that contains the primary topic. That route evidence can satisfy `advisory_topic_filter_ok`, but it does not allow text-only display, generated content, quick checks, field guides, Guardian use, Content Lab generation, or learning runtime.

## Review-Only Fields

The following are useful for review and mixed practice planning but are not strict-filter truth by themselves:

- `ai_assisted_items`
- `method_families`
- `exam_techniques`
- `common_mistakes`
- `suggested_new_subtopic`
- `suggested_new_skill`
- `ai_difficulty_factors`
- any record with `ai_final_review_required: true`

Human-reviewed records are preserved by the merge step and are not overwritten by machine candidates.

## Strict Filtering

`strict_filter_candidate` fails closed when:

- confidence is below the strict threshold
- `review_required` is true
- required evidence is missing
- the mapping is broad or prerequisite-only
- any canonical topic, subtopic, or skill ID is unknown

Unknown canonical IDs are validation errors. Suggested new taxonomy entries may appear only in suggestion fields.

## Difficulty Calibration

DeepSeek provides `ai_difficulty_estimate`, `ai_difficulty_score`, and `ai_difficulty_factors`. Final product filtering should use deterministic fields computed after merge:

- `deterministic_difficulty_percentile`
- `deterministic_difficulty_band`
- `difficulty_rank_within_paper_family`
- `difficulty_rank_basis`

Calibration is computed within each `paper_family`, not globally. The default bands are:

- `foundation`: 0 <= percentile < 25
- `standard`: 25 <= percentile < 60
- `challenging`: 60 <= percentile < 85
- `advanced`: 85 <= percentile <= 100

The percentile scale assigns higher values to harder questions, and rank `1` is the most difficult question within that paper family.

Current deterministic difficulty work is represented by `output/json/question_bank.difficulty_index.v1.json`, built by `scripts/generate_difficulty_index.py`. That sidecar may use mark events, topic routing, and advisory examiner-report/grade-threshold evidence, but it remains advisory and does not enable student-facing sequencing in v1. See [Difficulty Index Contract](DIFFICULTY_INDEX_CONTRACT.md).

## Running Sidecars

Use the canonical taxonomy directory, not archived or root-level taxonomy artifacts:

For the current Asterion-facing topic sidecar, run:

```bash
set -a; source .env; set +a

.venv/bin/python -m exam_bank.cli topic-route-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --output output/json/question_bank.topic_routing.v1.json \
  --model deepseek-v4-flash \
  --status-dir output/run_status
```

Resume with:

```bash
.venv/bin/python -m exam_bank.cli topic-route-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --output output/json/question_bank.topic_routing.v1.json \
  --model deepseek-v4-flash \
  --status-dir output/run_status \
  --resume
```

The broader v2 enrichment command remains available for review/debug enrichment:

```bash
export DEEPSEEK_API_KEY=...
.venv/bin/python -m exam_bank.cli enrich-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --existing-sidecar output/json/question_bank.deepseek.json \
  --output output/json/question_bank.ai_assisted.v2.json \
  --component p3 \
  --limit 20 \
  --resume \
  --status-dir output/run_status \
  --model deepseek-v4-flash \
  --include-subparts \
  --recompute-difficulty
```

Processing is batched by paper by default. Topic-routing and broader enrichment runs write live status files under `output/run_status/<run_id>/` by default:

- `run_status.json`
- `batch_status.jsonl`
- `run_manifest.json`

The terminal progress line includes the current phase, paper, batch count, record count, elapsed time, ETA when enough data exists, failed/skipped counts, and output path. Topic routing also reports review-required and provider/API failure counts. Broad v2 runs additionally write batch cache files under `<output-stem>.batches/`.

Use `--dry-run` to inspect selected records without creating a client or calling the network. Use `--only-errors`, `--paper`, `--question-id`, or `--component` for targeted resume/retry runs.
