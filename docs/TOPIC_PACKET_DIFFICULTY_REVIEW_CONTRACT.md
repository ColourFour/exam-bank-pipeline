# Topic Packet Difficulty Review Contract

This contract covers the image-first topic-packet difficulty review loop. It ranks questions within one generated topic packet at a time.

The output is advisory metadata. Canonical question images and mark-scheme images remain the source of truth. This loop must not mutate `question_bank.json`, topic-routing sidecars, topic packet manifests, or packet PDFs.

## Outputs

Primary sidecar:

- `data/review/topic_difficulty/<packet_id>/topic_packet_difficulty_review.v1.json`

Default reports:

- `reports/topic_difficulty/<packet_id>/summary.md`
- `reports/topic_difficulty/<packet_id>/ranking.md`
- `reports/topic_difficulty/<packet_id>/review_queue.md`

## Commands

Build an image-backed batch from a generated topic packet manifest:

```bash
.venv/bin/python -m exam_bank.cli topic-difficulty-review build \
  --manifest output/topic_packets/p3/integration/manifest.json \
  --artifact-root output \
  --out-dir data/review/topic_difficulty
```

Run append/resumable AI visual reviews:

```bash
.venv/bin/python -m exam_bank.cli topic-difficulty-review run \
  --batch data/review/topic_difficulty/<packet_id>/topic_packet_difficulty_batch.json \
  --out data/review/topic_difficulty/<packet_id>/topic_packet_difficulty_decisions.jsonl \
  --model gpt-5-mini
```

Import validated decisions and write ranked sidecar plus reports:

```bash
.venv/bin/python -m exam_bank.cli topic-difficulty-review import \
  --batch data/review/topic_difficulty/<packet_id>/topic_packet_difficulty_batch.json \
  --decisions data/review/topic_difficulty/<packet_id>/topic_packet_difficulty_decisions.jsonl \
  --artifact-root output
```

## Interpretation

`packet_rank` is relative to the topic packet only:

- `1`: hardest question in the packet
- `n`: easiest question in the packet

`difficulty_percentile_0_100` is also packet-relative. Higher means harder. It is computed after ranking as:

```text
100 * (n - rank) / (n - 1)
```

For a one-question packet, the percentile is `100`.

`visual_difficulty_score_0_100` is an AI reviewer score used to order the packet. It is not a psychometric measurement, a candidate success-rate estimate, or a global syllabus difficulty score.

## Evidence Inputs

The batch builder reads a topic packet `manifest.json`, especially `included_records[]`, and resolves:

- canonical question images
- canonical mark-scheme images
- packet section, problem number, paper, marks, topic IDs, and review flags

The AI runner sends the canonical question image and canonical mark-scheme image for each question. Metadata supports identity and ordering, but image evidence is authoritative.

## Validation Rules

Default import is fail-closed. A final complete sidecar requires:

- exactly one accepted decision for every packet question
- no unknown question IDs
- no duplicate decisions
- no pending decisions
- valid `visual_difficulty_score_0_100` values from `0` to `100`
- valid confidence values: `high`, `medium`, or `low`
- evidence refs for both canonical image types
- existing image paths

`--allow-incomplete` may write a draft sidecar and reports for inspection. Draft sidecars keep `complete=false`, `draft=true`, `safe_for_teacher_filtering=false`, and `safe_for_student_sequencing=false`.

## Ranking Policy

Accepted decisions are sorted hardest first by:

1. higher `visual_difficulty_score_0_100`
2. higher confidence
3. higher marks
4. later original packet problem number
5. `question_id`

Ranks are then assigned uniquely from `1..n`.

## Forbidden Uses

Do not use topic packet difficulty review to:

- claim psychometric validity or candidate success rates
- compare difficulty across unrelated packets
- override canonical extraction, topic routing, or Asterion role gates
- repair or replace rendered question or mark-scheme images
- mutate topic packet manifests or PDFs
- enable student-facing sequencing in v1
