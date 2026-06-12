# Asterion Export Regeneration PR 13

- Date: `2026-06-11`
- Topic sidecar input: `output/json/question_bank.topic_routing.v1.json`
- Question-bank input: `output/json/question_bank.json`
- Validation report: `/tmp/asterion_export_regeneration_pr13_validation.json`
- Comparison report: `/tmp/asterion_export_regeneration_pr13_compare.json`

## Sidecar Persistence Status

The refreshed topic-routing sidecar is not durable in Git in this checkout.

- `git check-ignore -v output/json/question_bank.topic_routing.v1.json || true` reported `.gitignore:18:output/json/*`.
- `git ls-files -- output/json/question_bank.topic_routing.v1.json` printed nothing.
- `git status --short --ignored output/json/question_bank.topic_routing.v1.json` reported `!! output/json/question_bank.topic_routing.v1.json`.

This means the sidecar is an ignored, untracked local/generated input here. The export commands below consumed the local file directly via `--topic-routing`; this report does not claim the sidecar replacement is committed or otherwise durable in Git.

## Refreshed Sidecar Preflight

The local topic-routing sidecar was confirmed before export:

| Metric | Count |
| --- | ---: |
| Question-bank records | 1301 |
| Sidecar records | 1301 |
| Unique sidecar IDs | 1301 |
| Missing IDs | 0 |
| Extra IDs | 0 |
| Duplicate IDs | 0 |
| Failed routes | 0 |
| Review-required routes | 42 |
| Strict-filter candidates | 1259 |
| Missing `evidence_packet_hash` | 0 |
| `safe_for_strict_filters` by audit computation | true |

## Commands Run

```bash
git check-ignore -v output/json/question_bank.topic_routing.v1.json || true
git ls-files -- output/json/question_bank.topic_routing.v1.json
git status --short --ignored output/json/question_bank.topic_routing.v1.json
.venv/bin/python -m exam_bank.cli asterion-export --input output/json/question_bank.json --artifact-root output --topic-routing output/json/question_bank.topic_routing.v1.json
.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --input output/json/question_bank.json --artifact-root output --mark-events output/json/question_bank.mark_events.v1.json --topic-routing output/json/question_bank.topic_routing.v1.json
PYTHONPATH=src:. .venv/bin/python scripts/validate_asterion_all_course_export.py --output /tmp/asterion_export_regeneration_pr13_validation.json
.venv/bin/python -m pytest -q tests/test_topic_routing.py tests/test_topic_routing_sample_refresh.py tests/test_topic_routing_audit.py tests/test_asterion_export.py tests/test_asterion_course_contract.py
```

All export commands exited `0`. The validator exited `0` with `ok: true`.

Validator warnings:

- Catalog has P1/M1/S1 records without `topic_id`; see `student_runtime_missing_topic_counts`.
- P1/M1/S1 learning-runtime targets are report-only; non-P3 image/advisory records are not reviewed learning runtime.

## Regenerated Files

- `output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json`
- `output/asterion/exports/latest/asterion_question_bank_v1.json`
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`

These files are also under ignored `output/` storage in this checkout.

## Before/After Counts

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Catalog records | 1301 | 1301 | 0 |
| Student runtime records | 281 | 279 | -2 |
| Content Lab candidates | 2432 | 2432 | 0 |
| Catalog advisory topic-filter OK | 822 | 887 | +65 |
| Catalog reviewed topic-filter safe | 0 | 0 | 0 |
| Catalog learning-runtime safe | 281 | 279 | -2 |
| Catalog student-runtime safe | 281 | 279 | -2 |
| P3 catalog records | 396 | 396 | 0 |
| P3 runtime records | 281 | 279 | -2 |
| Non-P3 runtime records | 0 | 0 | 0 |

After regeneration, runtime course counts are:

| Course | Runtime records |
| --- | ---: |
| P3 | 279 |
| P1 | 0 |
| M1 | 0 |
| S1 | 0 |

The P3 runtime count changed from the previous conservative baseline of `281` to `279`. The runtime set had `12` deterministic gate additions and `14` removals, for a net `-2` change. No P1, M1, or S1 material became student-facing.

## Topic-Route Gate Impact

| Metric | Count |
| --- | ---: |
| Sidecar strict-filter records | 1259 |
| Catalog records with `topic_route.filter_ok=true` | 1259 |
| Catalog records with `advisory_topic_filter_ok=true` | 887 |
| Net advisory topic-filter OK change vs before | +65 |
| Review-required sidecar records | 42 |
| Catalog records blocked by review-required topic route | 42 |
| Failed sidecar records | 0 |
| Catalog records blocked by failed topic route | 0 |

No review-required route is used as a strict student runtime filter, and no failed route is used.

## Runtime-Safe Gate Impact

The validator reported `1022` catalog records not in reviewed student runtime after regeneration.

Top gate/blocker counts from the regenerated catalog include:

| Reason | Count |
| --- | ---: |
| `student_runtime_safe_false` | 1022 |
| `learning_runtime_safe_false` | 1022 |
| `review_status_needs_review` | 1019 |
| `question_crop_not_high_confidence` | 758 |
| `text_only_blocked_visual_required` | 739 |
| `content_lab_blocked_topic_confidence_low` | 688 |
| `content_lab_blocked_topic_uncertain` | 688 |
| `mark_scheme_crop_not_high_confidence` | 463 |
| `validation_status_review` | 369 |
| `student_runtime_topic_route_review_required` | 37 |
| `student_runtime_topic_route_not_filter_ok` | 37 |
| `student_runtime_missing_topic_route` | 37 |

Non-P3 learning-runtime targets remain report-only and unmet by design:

| Course | Catalog records | Runtime-safe records |
| --- | ---: | ---: |
| P1 | 401 | 0 |
| M1 | 258 | 0 |
| S1 | 246 | 0 |

## Auto-Grade Eligibility Impact

No auto-grade command was run and `output/auto_grade/eligible_items.v1.json` was not regenerated as part of this PR. Auto-grade eligibility was not expanded.

## Scope Confirmation

No provider calls were run. Topic routing was not rerouted. Routing behavior, prompt text/version, taxonomy, reviewed decisions, Asterion app/runtime code, Asterion runtime behavior, and auto-grade eligibility were not changed.

The only student runtime changes came from the existing deterministic Asterion export gates consuming the refreshed local sidecar. There was no manual runtime promotion.

## Recommended Next PR

Decide how to make the refreshed topic-routing sidecar durable for releases: either track a reviewed release artifact, add a documented sync step from a release-artifact location, or document that export regeneration requires a validated local/generated sidecar. After that, run a release packaging PR that records the durable sidecar source and regenerated export hashes together.
