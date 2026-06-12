# Asterion Export Release Provenance PR 15

- Generated at: `2026-06-12T00:11:30Z`
- Durable sidecar: `data/topic_routing/question_bank.topic_routing.v1.json`
- Local export-consumed sidecar: `output/json/question_bank.topic_routing.v1.json`
- Validator report: `/tmp/asterion_export_release_provenance_pr15_validation.json`

## Commands Run

```bash
.venv/bin/python -m exam_bank.topic_routing_artifact restore
.venv/bin/python -m exam_bank.cli asterion-export --input output/json/question_bank.json --artifact-root output --topic-routing output/json/question_bank.topic_routing.v1.json
.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --input output/json/question_bank.json --artifact-root output --mark-events output/json/question_bank.mark_events.v1.json --topic-routing output/json/question_bank.topic_routing.v1.json
.venv/bin/python scripts/validate_asterion_all_course_export.py --output /tmp/asterion_export_release_provenance_pr15_validation.json
shasum -a 256 data/topic_routing/question_bank.topic_routing.v1.json output/json/question_bank.topic_routing.v1.json output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json output/asterion/exports/latest/asterion_question_bank_v1.json output/asterion/exports/latest/asterion_content_lab_candidates_v1.json /tmp/asterion_export_release_provenance_pr15_validation.json
```

## Sidecar Restore And Verify

The durable sidecar workflow restored and verified the local export-consumed sidecar successfully.

| Check | Result |
| --- | --- |
| Durable sidecar exists | pass |
| Checksum file exists | pass |
| Local sidecar exists | pass |
| Local sidecar matches durable SHA | pass |
| Question-bank IDs represented exactly once | pass |
| `safe_for_strict_filters` by audit computation | true |

`output/json/question_bank.topic_routing.v1.json` remains ignored/local and is not treated as durable release state.

## SHA-256 Evidence

| Artifact | SHA-256 |
| --- | --- |
| Durable topic sidecar | `e73559581b9cd5970d38496b1f6b334050a17789cc25f082eb5ccb94b1142e4e` |
| Local topic sidecar | `e73559581b9cd5970d38496b1f6b334050a17789cc25f082eb5ccb94b1142e4e` |
| Asterion catalog | `9ae6882ea8a35da7ccd09f514fb59ab3b1e0fe551e8539e3efe230bd23b71236` |
| Asterion student runtime | `e20bcb7649044194dcd3eb1988eaebfef35d39a8e25bd5f88584af25d9ebdabb` |
| Content Lab candidates | `085f17a49e318a622c7b1cb5f3fc864146d2905f40a7e8225d77260f9f3a4a56` |
| Validator JSON report | `9e3ffad9755bd487b08a902e0ff3aeae2c6575e56c77741432c222b479ddce38` |

## Sidecar Counts

| Metric | Count |
| --- | ---: |
| Records | 1301 |
| Failed routes | 0 |
| Review-required routes | 42 |
| Strict-filter candidates | 1259 |
| Missing `evidence_packet_hash` | 0 |
| Missing IDs | 0 |
| Extra IDs | 0 |
| Duplicate IDs | 0 |

## Export Counts

| Metric | Count |
| --- | ---: |
| Catalog records | 1301 |
| Student runtime records | 279 |
| P3 runtime records | 279 |
| Non-P3 runtime records | 0 |
| Content Lab candidates | 2432 |
| Advisory topic-filter OK | 887 |
| Reviewed topic-filter safe | 0 |
| Learning-runtime safe | 279 |
| Student-runtime safe | 279 |

Runtime course counts:

| Course | Runtime records |
| --- | ---: |
| P3 | 279 |
| P1 | 0 |
| M1 | 0 |
| S1 | 0 |

No P1, M1, or S1 material became student-facing.

## Validator Result

`scripts/validate_asterion_all_course_export.py` passed with `ok: true`.

Validator warnings:

- Catalog has P1/M1/S1 records without `topic_id`; see `student_runtime_missing_topic_counts`.
- P1/M1/S1 learning-runtime targets are report-only; non-P3 image/advisory records are not reviewed learning runtime.

The validation output includes `topic_routing_artifact_provenance` with matching durable/local SHA-256 values.

## Scope Confirmation

No provider calls were run. Topic routing was not rerouted. Topic-routing behavior, prompt text/version, taxonomy, reviewed decisions, Asterion app/runtime behavior, manual student-runtime promotion, and auto-grade eligibility were not changed.

Auto-grade eligibility did not change; `output/auto_grade/eligible_items.v1.json` was not regenerated.

## Recommended Next PR

Publish or hand off this release provenance bundle with the regenerated export artifacts. The next code PR should address only release packaging mechanics, such as where validated ignored export artifacts are stored outside Git or how deployment consumes this evidence bundle.
