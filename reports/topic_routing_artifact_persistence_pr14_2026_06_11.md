# Topic Routing Artifact Persistence PR 14

- Date: `2026-06-11`
- Durable sidecar path: `data/topic_routing/question_bank.topic_routing.v1.json`
- Durable checksum path: `data/topic_routing/question_bank.topic_routing.v1.sha256`
- Local export-consumed sidecar path: `output/json/question_bank.topic_routing.v1.json`

## Sidecar Persistence Status

`output/json/question_bank.topic_routing.v1.json` remains ignored and untracked in this checkout.

- `git check-ignore -v output/json/question_bank.topic_routing.v1.json || true` reports `.gitignore:18:output/json/*`.
- `git ls-files -- output/json/question_bank.topic_routing.v1.json` prints nothing.
- `git status --short --ignored output/json/question_bank.topic_routing.v1.json` reports `!! output/json/question_bank.topic_routing.v1.json`.

The refreshed sidecar is now durable through the tracked `data/topic_routing/` artifact path instead of relying on ignored `output/json/` state.

## SHA-256 Provenance

| Artifact | SHA-256 |
| --- | --- |
| Durable sidecar | `e73559581b9cd5970d38496b1f6b334050a17789cc25f082eb5ccb94b1142e4e` |
| Durable `.sha256` expected value | `e73559581b9cd5970d38496b1f6b334050a17789cc25f082eb5ccb94b1142e4e` |
| Local export-consumed sidecar | `e73559581b9cd5970d38496b1f6b334050a17789cc25f082eb5ccb94b1142e4e` |

The durable sidecar, checksum file, and local `output/json/` working copy match.

## Validation Counts

| Metric | Count |
| --- | ---: |
| Question-bank records | 1301 |
| Sidecar records | 1301 |
| Unique sidecar IDs | 1301 |
| Failed routes | 0 |
| Review-required routes | 42 |
| Strict-filter candidates | 1259 |
| Missing `evidence_packet_hash` | 0 |
| Missing question-bank IDs | 0 |
| Extra sidecar IDs | 0 |
| Duplicate sidecar IDs | 0 |
| `safe_for_strict_filters` by audit computation | true |

## Implementation

Created `src/exam_bank/topic_routing_artifact.py` with:

- `verify` action for checksum, count, evidence hash, and question-ID coverage checks.
- `restore` action for copying the durable sidecar to `output/json/question_bank.topic_routing.v1.json` and verifying it.
- A production-path provenance predicate used by export code and validation.

Updated Asterion export code so `asterion-export`, catalog export, and Content Lab candidate export verify provenance when consuming `output/json/question_bank.topic_routing.v1.json`.

Updated `scripts/validate_asterion_all_course_export.py` so validation embeds `topic_routing_artifact_provenance` and fails when the local production sidecar does not match the durable artifact.

## Docs Updated

- `README.md`
- `docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md`
- `docs/OUTPUT_STORAGE_CONTRACT.md`
- `docs/COMMAND_ATLAS.md`

The docs now state that PR 13 export validation was valid locally but depended on an ignored sidecar before this PR, and that future Asterion export regeneration must restore or verify the sidecar first.

## Tests Added

- `tests/test_topic_routing_artifact.py`

Covered:

- durable sidecar checksum calculation and verification
- mismatch detection when local `output/json` sidecar differs from the durable artifact
- missing local sidecar failure
- all-question-ID coverage checks
- restore behavior
- production sidecar path provenance enforcement detection

## Commands Run

```bash
.venv/bin/python -m exam_bank.topic_routing_artifact verify
PYTHONPATH=src:. .venv/bin/python scripts/validate_asterion_all_course_export.py --output /tmp/asterion_export_validation_pr14_provenance.json
.venv/bin/python -m pytest -q tests/test_topic_routing.py tests/test_topic_routing_sample_refresh.py tests/test_topic_routing_audit.py tests/test_topic_routing_artifact.py tests/test_asterion_export.py
.venv/bin/python -m pytest -q tests -k "topic_routing and sidecar"
git check-ignore -v output/json/question_bank.topic_routing.v1.json || true
git ls-files -- output/json/question_bank.topic_routing.v1.json
git status --short --ignored output/json/question_bank.topic_routing.v1.json
git diff --name-only -- output/json/question_bank.json
git diff --name-only -- output/json/question_bank.topic_routing.v1.json
git diff --check
```

Results:

- Topic-routing artifact verify: pass.
- All-course export validator with provenance: pass, with existing report-only P1/M1/S1 learning-runtime warnings.
- Targeted tests: pass.
- `tests -k "topic_routing and sidecar"`: pass.
- `git diff --check`: pass.

## Future Export Workflow

Before Asterion export regeneration, run:

```bash
.venv/bin/python -m exam_bank.topic_routing_artifact restore
.venv/bin/python -m exam_bank.topic_routing_artifact verify
.venv/bin/python -m exam_bank.cli asterion-export \
  --input output/json/question_bank.json \
  --artifact-root output \
  --topic-routing output/json/question_bank.topic_routing.v1.json
PYTHONPATH=src:. .venv/bin/python scripts/validate_asterion_all_course_export.py
```

The export command and validator now fail if the local production sidecar does not match the durable artifact checksum.

## Scope Confirmation

No provider calls were run. Topic routing was not rerouted. Asterion exports were not regenerated for this PR. Topic-routing behavior, prompt text/version, taxonomy, reviewed decisions, Asterion runtime behavior, student-runtime promotion, and auto-grade eligibility were not changed.

## Recommended Next PR

Run a release packaging/provenance PR that records the durable topic sidecar SHA, the regenerated Asterion export SHAs, and the validator report together as release evidence.
