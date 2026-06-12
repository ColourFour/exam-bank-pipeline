# Topic Routing Full Refresh PR11 - 2026-06-11

## Command Summary

Preflight confirmed the required inputs exist:

- `output/json/question_bank.json`
- `output/json/question_bank.topic_routing.v1.json`
- `reports/topic_routing_improvement_closeout_2026_06_11.md`

`.env` was loaded before provider calls, and `DEEPSEEK_API_KEY` was visible to `.venv` Python.

Commands run:

```bash
set -a
source .env
set +a
rm -rf /tmp/topic_routing_full_refresh_pr11_status
.venv/bin/python -m exam_bank.cli topic-route-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --output /tmp/question_bank.topic_routing.full_refresh.pr11.json \
  --failure-log /tmp/question_bank.topic_routing.full_refresh.pr11.failures.jsonl \
  --status-dir /tmp/topic_routing_full_refresh_pr11_status \
  --run-id pr11-full-refresh \
  --no-progress
```

The first provider pass completed in 1:05:58 with 1301 attempted, 1271 successful, 30 failed provider rows, 71 review-required, and 30 provider failures. The failed rows were isolated to four provider-error batches.

The same PR 11 `/tmp` output was then resumed:

```bash
.venv/bin/python -m exam_bank.cli topic-route-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --output /tmp/question_bank.topic_routing.full_refresh.pr11.json \
  --failure-log /tmp/question_bank.topic_routing.full_refresh.pr11.failures.jsonl \
  --status-dir /tmp/topic_routing_full_refresh_pr11_status \
  --run-id pr11-full-refresh \
  --resume \
  --no-progress
```

The resume preserved 1271 current rows, retried 30 stale/error rows, and completed in 1:59 with 0 failed rows and 0 provider failures.

Audit and comparison commands:

```bash
.venv/bin/python -m exam_bank.topic_routing_audit \
  --question-bank output/json/question_bank.json \
  --topic-routing /tmp/question_bank.topic_routing.full_refresh.pr11.json \
  --json-out /tmp/question_bank.topic_routing.full_refresh.pr11.audit.json \
  --markdown-out /tmp/question_bank.topic_routing.full_refresh.pr11.audit.md

.venv/bin/python /tmp/topic_routing_full_refresh_pr11_compare.py
```

## Full Refresh Headline Counts

Final refreshed sidecar: `/tmp/question_bank.topic_routing.full_refresh.pr11.json`

| Metric | Count |
| --- | ---: |
| Question-bank records | 1301 |
| Refreshed sidecar rows | 1301 |
| Successful route records | 1301 |
| Failed route records | 0 |
| Provider failures | 0 |
| Review-required records | 42 |
| Strict-filter candidates | 1259 |
| Missing `evidence_packet_hash` | 0 |
| Malformed `evidence_packet_hash` | 0 |
| Missing course metadata | 0 |
| Duplicate question IDs | 0 |
| Question-bank records missing from sidecar | 0 |
| Sidecar records not in question bank | 0 |

Route status distribution:

- `strict_filter_candidate`: 1259
- `review_required`: 42

Confidence distribution:

- `high`: 1254
- `medium`: 44
- `low`: 3

The refreshed `/tmp` sidecar represents all 1301 question-bank records exactly once.

## Comparison To Old Production Sidecar

The production sidecar was not overwritten. Comparison is against `output/json/question_bank.topic_routing.v1.json`.

| Metric | Old production sidecar | PR11 `/tmp` refresh |
| --- | ---: | ---: |
| Total rows | 1301 | 1301 |
| Failed rows | 46 | 0 |
| Non-error review-required rows | 70 | 42 |
| Non-strict rows (`safe_for_strict_filters=false`) | 116 | 42 |
| Strict-filter candidates | 1185 | 1259 |
| Missing `evidence_packet_hash` | 1301 | 0 |
| Provider failures | 0 | 0 |
| Missing question-bank IDs | 0 | 0 |
| Extra sidecar IDs | 0 | 0 |

The requested old review-required baseline of 116 corresponds to the old sidecar's non-strict rows: 46 failed rows plus 70 non-error review-required rows.

## Audit Status

Audit status: pass for sidecar readiness.

The deterministic audit reported:

- `topic_routing_safe_for_strict_filters`: `true`
- failed routes: 0
- review-required routes: 42
- strict-filter candidates: 1259
- unsupported `evidence_used` failures: 0
- unique failure messages: 0
- missing `evidence_packet_hash`: 0
- malformed rows: 0 by audit failure checks

One strict-filter candidate had repaired evidence metadata (`evidence_used_repaired_count=1`, dropped field `question_text`). Topic validation and strict-filter eligibility still passed.

## Remaining Review-Required Buckets And Reasons

Normalized review-required buckets:

- `visual_required_without_sufficient_text_evidence`: 22
- `unknown`: 19
- `ambiguous_multi_topic_fit`: 1

Top review-required reasons:

- `Visual-dependent question; image not provided for verification.`: 10
- `Visual-dependent question, no image provided; routing based on text.`: 4
- `Visual-dependent question; no image provided.`: 4
- `<missing>`: 9
- Single-record reasons include OCR garbling, diagram/text discrepancy, mixed-topic questions, and multi-part questions spanning differentiation/integration, probability/permutations, momentum/energy, and coordinate geometry/circular measure.

Review-required overlap:

- Visual-required overlap: 41
- Weak text or crop-readiness overlap: 42

## Provider Failures And Recovery

The first pass had 30 provider-failure rows in four batches:

- `m1_41autumn22_0002_4961e6ad`: 6 rows
- `p1_12spring22_0054_adbd61ed`: 11 rows
- `s1_51summer25_0121_0909b617`: 6 rows
- `s1_52autumn22_0123_296ecfed`: 7 rows

The same PR 11 run was resumed against the `/tmp` sidecar. Resume retried those stale/error rows and completed with 0 failed rows and 0 provider failures.

## Generated Evidence

- Full refreshed sidecar: `/tmp/question_bank.topic_routing.full_refresh.pr11.json`
- Failure log: `/tmp/question_bank.topic_routing.full_refresh.pr11.failures.jsonl`
- Status directory: `/tmp/topic_routing_full_refresh_pr11_status/pr11-full-refresh`
- Audit JSON: `/tmp/question_bank.topic_routing.full_refresh.pr11.audit.json`
- Audit markdown: `/tmp/question_bank.topic_routing.full_refresh.pr11.audit.md`
- Comparison JSON: `/tmp/question_bank.topic_routing.full_refresh.pr11.compare.json`
- Comparison markdown: `/tmp/question_bank.topic_routing.full_refresh.pr11.compare.md`
- One-off comparison script: `/tmp/topic_routing_full_refresh_pr11_compare.py`

## Readiness Recommendation

The refreshed `/tmp` sidecar appears ready for a later production replacement PR: it has exact 1301-record coverage, 0 failed records, 0 provider failures, 0 missing hashes, and the audit marks the sidecar safe for strict filters.

Do not replace `output/json/question_bank.topic_routing.v1.json` in this PR. Replace the production sidecar only in a later reviewed PR if this report is accepted.

After production sidecar replacement is reviewed, regenerate Asterion exports in a separate step and run the export validation gates. Do not change student runtime promotion or auto-grade eligibility in this PR.
