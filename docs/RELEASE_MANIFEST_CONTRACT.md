# Question-Bank Release Manifest Contract

`manifests/releases/question_bank_release_manifest.v1.json` is the cryptographic
bundle contract for one canonical question bank and the JSON artifacts released
with it. The topic-routing refresh creates the minimum valid release containing
the bank and durable topic sidecar. A downstream handoff should rebuild the
manifest with every consumed sidecar and projection included.

The manifest records each artifact's relative path, SHA-256, byte size, schema,
record collection, record count, and question-ID set. Every non-bank artifact is
bound to the canonical question-bank SHA-256. If an artifact declares
`source_question_bank_sha256`, verification also requires that declaration to
match the bank in the release.

When the `corpus_manifest` role is present, its exact file SHA-256 must equal
`question_bank.run_manifest.corpus_manifest_sha256`. This rejects a manifest
from another corpus, including the complete source contract when extraction
used a partial quarantined corpus.

## Question-ID Coverage

Choose coverage per role according to the artifact contract:

- `exact`: one unique record for every canonical question. Use for topic
  routing, mark events, the difficulty index, and the all-course Asterion
  catalog.
- `exact_set`: every canonical question must be referenced, while multiple
  records may reference the same question. Use for the asset manifest.
- `subset`: every referenced question must exist in the canonical bank, but
  records may be omitted or multiple records may reference the same question.
  Use for the Asterion student runtime, Content Lab candidates, and promotion
  decisions.
- `none`: the artifact is hash-bound but has no question-membership claim. Use
  for validation reports and metadata-only release packages.

Supported record collections are `questions`, `records`, `items`, `candidates`,
`decisions`, `rubrics`, `rows`, `assets`, and `artifacts`. JSON metadata without one of
these collections is still hash-bound with a zero record count.

## Dependency Bindings

`bound_artifact_sha256` records artifact-to-artifact dependencies inside the
same release. For example, a difficulty index can be bound to the exact topic
routing and mark-event artifacts used for the handoff, and an Asterion runtime
can be bound to its catalog. Conventional embedded hashes are checked when
present:

- top-level `source_<role>_sha256` fields;
- `<role>_sha256` fields inside `source_sidecars`;
- the durable topic sidecar and three export hashes inside an Asterion release
  package.

A declared dependency hash that disagrees with the same role in the release is
blocking.

## Build And Verify A Full Bundle

Run from the repository root. Include only artifacts that exist for the release;
repeat `--artifact`, `--coverage`, and `--depends-on` as needed.

```bash
.venv/bin/python -m exam_bank.command release build \
  --question-bank output/json/question_bank.json \
  --artifact corpus_manifest=manifests/corpora/caie_9709.active_partial.v1.json \
  --artifact topic_routing=data/topic_routing/question_bank.topic_routing.v1.json \
  --coverage topic_routing=exact \
  --artifact asset_manifest=output/json/asset_manifest.v1.json \
  --coverage asset_manifest=exact_set \
  --artifact mark_events=output/json/question_bank.mark_events.v1.json \
  --coverage mark_events=exact \
  --artifact difficulty_index=output/json/question_bank.difficulty_index.v1.json \
  --coverage difficulty_index=exact \
  --depends-on difficulty_index=topic_routing,mark_events \
  --output manifests/releases/question_bank_release_manifest.v1.json
```

Add the Asterion roles with `exact` catalog coverage and `subset` coverage for
runtime, Content Lab, and promotion decisions. A metadata-only Asterion package
uses the default `none` coverage and can depend on the topic, catalog, runtime,
and Content Lab roles.

```bash
.venv/bin/python -m exam_bank.command release verify \
  --manifest manifests/releases/question_bank_release_manifest.v1.json \
  --require-role question_bank \
  --require-role topic_routing \
  --require-role mark_events
```

Structural verification reports `provenance_ok` separately from `policy_ok`.
Roles ending in `_validation` contribute to the policy result: a missing or
non-true `ok` value makes `policy_ok=false` while hashes can remain valid. Use
the strict gate before approval or handoff:

```bash
.venv/bin/python -m exam_bank.command release verify \
  --manifest manifests/releases/question_bank_release_manifest.v1.json \
  --require-validation-ok
```

Release roles use lowercase letters, digits, and underscores. Duplicate roles,
duplicate dependency declarations, one file assigned to multiple roles,
ambiguous record collections, non-standard JSON, and manifest paths escaping
the declared artifact root are rejected.

## Existing Producer Provenance

- Mark events declare the source question-bank hash.
- The difficulty index declares the question-bank and topic-routing hashes, but
  does not currently embed the mark-event or advisory-evidence hashes it used.
- Promotion decisions carry semantic evidence references and contract metadata,
  but not hashes for every source registry.
- The Asterion package hashes the catalog, student runtime, Content Lab export,
  validation report, and durable topic sidecar, but does not directly declare
  the canonical question-bank hash.

The generic release bundle closes the handoff-integrity gaps when all consumed
inputs and outputs are included with dependency bindings. It does not prove how
an artifact was computed when its producer omits an input hash; that remaining
producer-level lineage gap must stay visible in release notes.
