# Topic Routing Production Replacement PR 12

- Date: `2026-06-11`
- Source sidecar: `/tmp/question_bank.topic_routing.full_refresh.pr11.json`
- Destination production sidecar: `output/json/question_bank.topic_routing.v1.json`
- Replacement method: copied the validated PR 11 full-refresh sidecar to the production sidecar path.
- Source/destination SHA-256 after replacement: `e73559581b9cd5970d38496b1f6b334050a17789cc25f082eb5ccb94b1142e4e`

## Audit Outputs

- JSON audit: `/tmp/question_bank.topic_routing.production_replacement.pr12.audit.json`
- Markdown audit: `/tmp/question_bank.topic_routing.production_replacement.pr12.audit.md`

## Final Production Counts

| Metric | Count |
| --- | ---: |
| Total records | 1301 |
| Failed routes | 0 |
| Review-required routes | 42 |
| Strict-filter candidates | 1259 |
| Provider failures | 0 |
| Missing `evidence_packet_hash` | 0 |
| `safe_for_strict_filters` | true |

All 1301 question-bank records are represented exactly once in the production sidecar. The replacement audit found 0 missing question IDs, 0 duplicate question IDs, and 0 missing evidence packet hashes.

## Comparison Against Old Production

| Metric | Old production | PR 12 production |
| --- | ---: | ---: |
| Failed routes | 46 | 0 |
| Review-required/non-strict rows | 116 | 42 |
| Strict-filter candidates | 1185 | 1259 |
| Missing `evidence_packet_hash` | 1301 | 0 |
| `safe_for_strict_filters` | false | true |

## Remaining Review-Required Buckets

| Bucket | Count |
| --- | ---: |
| `visual_required_without_sufficient_text_evidence` | 22 |
| `unknown` | 19 |
| `ambiguous_multi_topic_fit` | 1 |

## Scope Notes

This PR replaces the production topic-routing sidecar only. The production sidecar replacement does not itself promote any student runtime material.

Asterion exports were not regenerated in this PR. This PR does not change Asterion export behavior, runtime behavior, reviewed decisions, routing behavior, prompt text, prompt version, taxonomy, or auto-grade eligibility.

## Recommended Next PR

Regenerate Asterion exports from the new production sidecar, validate export gates, and do not change runtime promotion or auto-grade eligibility unless export validation explicitly supports it.
