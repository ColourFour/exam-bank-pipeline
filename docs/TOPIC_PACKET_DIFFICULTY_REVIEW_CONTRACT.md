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

Reconcile every current packet after routing or review inputs change:

```bash
.venv/bin/python -m exam_bank.cli topic-difficulty-review reconcile \
  --packets-root output/topic_packets \
  --difficulty-root data/review/topic_difficulty \
  --difficulty-index output/json/question_bank.difficulty_index.v1.json \
  --artifact-root output \
  --model gpt-5-mini
```

Reconciliation preserves reviewed evidence for questions that remain in the same packet. Questions moved from another
packet preserve their previous packet percentile provisionally; questions with no packet history use the deterministic
difficulty-index percentile. Both groups are automatically submitted for focused visual review. Provider or credential
failures leave explicit provisional records and do not abort reconciliation.

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

Packet PDFs consume ranks in reverse order, so larger/easier ranks appear first and rank 1 appears last. A packet with
provisional or missing records remains deterministically ordered but exposes `difficulty_ranking_complete=false`.

## Reconciliation and freshness

Difficulty never determines topic routing or packet membership. Current reviewed routing produces the packet membership;
the difficulty workflow only annotates and orders those records. Each packet manifest and reconciled v2 sidecar stores a
projection fingerprint covering membership, primary/secondary placement, packet section, routing input hashes, taxonomy,
and generator schema version. A mismatch blocks ranked regeneration and instructs the operator to reconcile.

Reconciled records use these statuses:

- `reviewed`: reusable evidence from the same packet or a completed focused review
- `provisional_topic_changed`: prior packet percentile carried into a new topic pending review
- `provisional_new_member`: deterministic difficulty-index percentile pending review
- `missing`: no reusable or deterministic evidence; ordered at the pending tail

Removed questions are excluded from active ranks and retained under `removed_records` for provenance and possible reuse.

## Forbidden Uses

Do not use topic packet difficulty review to:

- claim psychometric validity or candidate success rates
- compare difficulty across unrelated packets
- override canonical extraction, topic routing, or Asterion role gates
- repair or replace rendered question or mark-scheme images
- use difficulty evidence to change topic routing or packet membership
- claim that provisional ordering is a completed reviewed ranking
