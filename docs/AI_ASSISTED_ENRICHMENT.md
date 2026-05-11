# AI-Assisted Enrichment

The extraction pipeline remains image-first. `question_bank.json` and the canonical taxonomy files under `exam_bank_taxonomy/canonical/` are the source of truth. DeepSeek enrichment is sidecar metadata only.

## Sidecar Versions

`question_bank.deepseek.json` is the older DeepSeek sidecar. It stores broad topic, subtopic, difficulty, confidence, rationale, review flags, and reconciliation fields. It is still readable and can be used as evidence for a v2 run.

`question_bank.ai_assisted.v2.json` is the canonical-ID AI-assisted sidecar. It preserves useful v1 fields when present, then adds `ai_assisted_items` for whole-question and optional subpart mappings. Each item must use existing canonical IDs for:

- `primary_topic_id`
- `secondary_topic_ids`
- `subtopic_ids`
- `skill_ids`
- `prerequisite_skill_ids`

DeepSeek may suggest taxonomy gaps only in `suggested_new_subtopic` and `suggested_new_skill`. Those fields are review-only and are never treated as canonical IDs.

## Product-Safe Fields

Product strict filters should consume only `strict_filter_candidates`, and only the canonical IDs inside those candidates. These candidates are produced after deterministic validation has confirmed that all IDs exist in the active canonical taxonomy and that strict filtering did not fail closed.

Asterion region routing should use `strict_filter_candidates[].asterion_region_ids`, derived from canonical skill-map `asterion_region_id` values. If a record has no strict candidates, route it to review or broad fallback behavior, not a strict student-facing filter.

Content Lab may use `ai_assisted_items[].worked_example_seed` and `ai_assisted_items[].warmup_seed` as lesson-planning seeds. They are not generated lessons and should still be paired with visual evidence and mark-scheme review.

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

## Running V2

Use the canonical taxonomy directory, not archived or root-level taxonomy artifacts:

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

Processing is batched by paper by default. Each batch writes a cache file under `<output-stem>.batches/`, and every run writes live status files under `output/run_status/<run_id>/` by default:

- `run_status.json`
- `batch_status.jsonl`
- `run_manifest.json`

The terminal progress line includes the current phase, paper, batch count, record count, elapsed time, ETA when enough data exists, failed/skipped counts, and output path. The final sidecar metadata includes a run manifest with batch IDs, input hashes, model, prompt version, and cache paths.

Use `--dry-run` to inspect selected records without creating a client or calling the network. Use `--only-errors`, `--paper`, `--question-id`, or `--component` for targeted resume/retry runs.
