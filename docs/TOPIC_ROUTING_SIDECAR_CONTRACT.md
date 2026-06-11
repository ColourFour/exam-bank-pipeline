# Topic Routing Sidecar Contract

This contract covers:

- `output/json/question_bank.topic_routing.v1.json`
- strict topic routing behavior
- allowed topic structures for topic filtering consumers

The topic routing sidecar is separate from `output/json/question_bank.json`. It must not mutate canonical question-bank records and must not be merged into canonical records unless a future contract explicitly defines that promotion.

## Purpose

`question_bank.topic_routing.v1.json` is a narrow sidecar for routing CAIE 9709 questions to canonical parent topic IDs. Its intended use is downstream topic distribution and filtering after deterministic validation and audit.

The sidecar stores:

- top-level provenance and schema metadata
- one record per `question_id`
- a parent-topic `primary_topic_id`
- a parent-topic `topic_distribution`
- advisory `course_id` and `component_name` metadata for P1, P3, M1, and S1 consumers
- deterministic `evidence_packet_hash` freshness metadata for resume safety
- confidence, review, evidence, provider, and routing-source metadata

It does not store student-facing explanations, difficulty, subtopics, skills, Content Lab metadata, Asterion readiness, or generated learning content.

## Advisory Until Validated

The sidecar is advisory unless validation passes. Consumers must treat the sidecar as review evidence unless both conditions are true:

1. The sidecar has the expected schema and record contract.
2. The sidecar-level strict-filter safety gate is true.

The current safety gate is:

```text
metadata.run_summary.safe_for_strict_filters
```

If this field is missing, false, or unreadable, consumers must default to not using the sidecar for strict topic filtering.

## Current Known State

The current `output/json/question_bank.topic_routing.v1.json` is not safe for strict filters.

Current audited sidecar state as of `2026-05-14`, generated from source run `20260513T070200Z-56d469c1dd52`:

- `schema_name=exam_bank.topic_routing_sidecar`
- `schema_version=1`
- `record_count=1301`
- `successful_records=1148`
- `failed_records=153`
- `review_required_records=221`
- `strict_filter_records=1080`
- `failures_by_reason.schema_validation_error=153`
- `safe_for_strict_filters=false`

Even though the current sidecar contains individual records that look like strict-filter candidates, downstream strict filters must not use them while `safe_for_strict_filters=false`.

## Top-Level Contract

The sidecar top-level object contains:

- `schema_name`: must be `exam_bank.topic_routing_sidecar`.
- `schema_version`: currently `1`.
- `generated_at`: sidecar generation timestamp.
- `record_count`: number of records written.
- `taxonomy_path`: canonical taxonomy root used during routing, normally `exam_bank_taxonomy/canonical`.
- `taxonomy_version`: taxonomy version when available.
- `model`: provider model used for AI-routed records.
- `prompt_version`: currently `topic_routing_v1`.
- `course_contract`: static-site course IDs and component names; routing labels remain advisory.
- `records`: object keyed by `question_id`.
- `metadata`: run inputs, status manifest, selected count, and `run_summary`.

Consumers should verify `schema_name`, `schema_version`, `record_count`, `taxonomy_path`, and `metadata.run_summary` before reading record-level mappings.

## Record States

### Successful Records

A successful record has no `error` object. It may still require review.

Successful records include validated DeepSeek outputs and deterministic review-gate records. A successful record is eligible for strict filtering only if all strict candidate rules pass and the sidecar-level `safe_for_strict_filters` gate is true.

Strict candidate rules:

- no `error` object
- `review_required=false`
- `confidence` is `high` or `medium`
- `primary_topic_id` is a string
- `topic_distribution` is non-empty

### Failed Records

A failed record has an `error` object and must be treated as review-only.

Failed records are written fail-closed. They use:

- `primary_topic_id=null`
- `topic_distribution=[]`
- `confidence=low`
- `review_required=true`
- `routing_source=deepseek_topic_routing_error`
- `error.type` and `error.message`

Any failed record makes the generated sidecar unsafe for strict filters because `safe_for_strict_filters` is computed as false when `failed_records > 0`.

### Review-Required Records

A review-required record has `review_required=true` and must not enter strict filters.

Review-required records may come from:

- deterministic pre-provider gates, such as weak or missing text evidence
- visual-required questions without enough text evidence
- accepted AI output that marks the route as low confidence or review-needed
- failed records with provider, schema, or taxonomy validation errors

Review-required records may still be useful for teacher review, QA, or mixed practice planning, but they are not strict-filter truth.

### Evidence Packet Freshness

New topic-routing records include `evidence_packet_hash`, a SHA-256 hash of the deterministic packet content supplied to the router for that question. The hash is derived from the effective routing packet, including question identity, paper/component context, visual/text readiness fields, supplied evidence fields, and allowed topic context. It excludes provider timestamps, response IDs, generated output metadata, and local artifact paths that are not sent to the router.

When `topic-route-ai --resume` is used, a previous record is preserved only when all freshness checks pass:

- `llm_model` matches the requested model, except deterministic review-gate records may have `llm_model=null`
- `llm_prompt_version` matches the requested prompt version
- `evidence_packet_hash` exists on the previous record
- `evidence_packet_hash` matches the freshly rebuilt packet hash for the current question record

Legacy rows without `evidence_packet_hash` are treated as stale and rerouted on resume.

### Evidence Metadata Repair

The router treats the packet's supplied evidence fields as the source of truth for `evidence_used`. Model-returned `evidence_used` values are checked against the evidence fields actually sent in the packet.

If a returned route is otherwise valid but includes unsupported evidence labels, the router repairs only the evidence metadata:

- unsupported labels are dropped
- supported labels are preserved
- if all returned labels are unsupported but the packet has supplied evidence, the router uses a deterministic fallback of the supplied evidence fields
- if no supplied evidence fallback exists, the record is written fail-closed as review-required/error

Repaired records include:

- `evidence_used_repaired=true`
- `evidence_used_original`
- `evidence_used_dropped`

Evidence repair does not loosen validation for topic IDs, topic distributions, duplicate topics, confidence, primary-topic rules, malformed records, or any other routing contract field. Those failures remain per-record validation errors.

## Allowed Topic Structures

Strict topic routing uses only canonical parent topic IDs from `exam_bank_taxonomy/canonical/topic_filter_maps/`.

The active paper-family mapping is:

- `p1` uses `topic_filter_map_9709_p1_v1.json`
- `p3` uses `topic_filter_map_9709_p3_v1.json`
- `p4` uses Mechanics component topics from `topic_filter_map_9709_m1_v1.json`
- `p5` uses Statistics component topics from `topic_filter_map_9709_s1_v1.json`

The Asterion static-site course contract exposes these as course IDs `p1`, `p3`, `m1`, and `s1`. The mapping is paper-family-to-course: `p1 -> p1`, `p3 -> p3`, `p4 -> m1`, and `p5 -> s1`. Topic-routing records may include `course_id` and `component_name` for downstream catalog grouping, but those fields do not make the routing authoritative. Topic-routing labels remain advisory unless the sidecar-level strict-filter gate is true and the record itself is non-review-required.

Allowed parent topics have IDs shaped like:

```text
9709_<component>_topic_<slug>
```

Examples:

- `9709_p1_topic_quadratics`
- `9709_p3_topic_complex_numbers`
- `9709_m1_topic_forces_and_equilibrium`
- `9709_s1_topic_probability`

`topic_distribution` must be an array of objects with exactly:

- `topic_id`: one allowed parent topic ID for that record's paper family
- `fit_percent`: positive integer

Distribution rules:

- percentages must total exactly `100` when the distribution is non-empty
- duplicate `topic_id` values are invalid
- `primary_topic_id` must appear in `topic_distribution`
- `primary_topic_id=null` is allowed only when `review_required=true`
- `topic_distribution=[]` is allowed only when no defensible topic can be chosen

Strict topic routing does not accept subtopic IDs, skill IDs, legacy local topic labels, free-text topic names, or newly invented topic IDs.

## Topic Source Boundaries

### Local Deterministic Topic Classification

Local deterministic topic classification is produced during the extraction/export pipeline. It may populate broad fields such as `topic`, confidence fields, and `notes.topic_trust_status` in `question_bank.json`.

This local output is useful for search, review prioritization, and legacy routing, but it is not the same as strict parent-topic routing. A local topic label can be present while `notes.topic_trust_status` is `degraded_text` or `review_required`.

### DeepSeek/AI Enrichment

DeepSeek enrichment sidecars are broader AI metadata artifacts. They may include topic, subtopic, skill, difficulty, rationale, reconciliation, and review fields depending on sidecar version.

These sidecars remain sidecar metadata. Suggested new taxonomy entries are review-only. AI enrichment must not change the canonical curriculum structure or write new allowed IDs into canonical question records.

### Strict Topic Routing

Strict topic routing is the narrow `topic-route-ai` sidecar path. It sends each question only the allowed parent topics for that paper family and validates the returned parent-topic IDs before acceptance.

Strict routing is the only sidecar path documented here for parent-topic distributions. It still remains advisory unless sidecar-level safety metadata allows strict filtering.

### Advisory Evidence Sources

Deterministic examiner-report and grade-threshold evidence now exists under `output/advisory_evidence/`. It remains separate from strict topic routing. Until a separate audited release explicitly changes this contract, advisory evidence must not change strict topic routing, canonical question-bank records, or Asterion role gates. Grade-threshold context must not directly prove individual-question difficulty.

## AI Topic Structure Rule

AI must not invent the main curriculum or topic structure.

The canonical topic structure is owned by `exam_bank_taxonomy/canonical/`. Topic routing may choose among supplied parent topic IDs only. If AI returns an unknown topic ID, a topic ID from another paper family, unsupported evidence, malformed JSON, duplicate keys, extra fields, or invalid percentages, the response is rejected and written as a failed review-required record.

New topic, subtopic, or skill ideas belong in explicit review-only suggestion fields in broader enrichment workflows. They are not valid topic-routing IDs and must not be treated as canonical curriculum until the taxonomy is intentionally updated and validated.

## Strict Filtering Consumer Checklist

Before using the sidecar for strict topic filtering:

1. Verify `schema_name=exam_bank.topic_routing_sidecar` and `schema_version=1`.
2. Verify `record_count` matches the number of `records`.
3. Verify `taxonomy_path` points to the expected canonical taxonomy root.
4. Require `metadata.run_summary.safe_for_strict_filters=true`; default deny if missing or false.
5. Require `metadata.run_summary.failed_records=0`.
6. Use only records with no `error`, `review_required=false`, `confidence in {"high", "medium"}`, string `primary_topic_id`, and non-empty `topic_distribution`.
7. Reject any `topic_id` not allowed for that record's `paper_family`.
8. Reject distributions that do not total exactly `100`.
9. Preserve failed and review-required records for QA/review queues, not strict filters.
10. Do not merge sidecar topic output into `question_bank.json` canonical records.
11. Do not use raw sidecar records by themselves to promote records into static student learning runtime. Learning-runtime promotion comes from the course-aware Asterion contract. The Asterion export may use an individual topic route as one input to the advisory topic-filter gate only when that route is record-level filterable: no error, `review_required=false`, `confidence in {"high", "medium"}`, string `primary_topic_id`, and a non-empty, duplicate-free distribution totaling `100` that contains the primary topic. This can produce `advisory_topic_filter_ok`; it must not produce `reviewed_topic_filter_safe` or `learning_runtime_safe` without reviewed topic-alignment evidence.
