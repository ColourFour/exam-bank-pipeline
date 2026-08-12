# Command migration

The public command surface is `exam-bank <domain> <command>`. The complete list
is generated from the parser registry in
[`COMMAND_REFERENCE.md`](COMMAND_REFERENCE.md).

| Removed flat command | Public replacement |
| --- | --- |
| `process` | `exam-bank extract run` |
| `audit` | `exam-bank extract audit` |
| `output-integrity-audit` | `exam-bank extract integrity` |
| `regenerate-question-pngs` | `exam-bank extract regenerate-questions` |
| `regenerate-mark-scheme-pngs` | `exam-bank extract regenerate-mark-schemes` |
| `topic-route-ai` | `exam-bank topic route` |
| `topic-packets` | `exam-bank topic packets` |
| `topic-difficulty-review` | `exam-bank topic difficulty` |
| `asterion-export` | `exam-bank asterion export` |
| `asterion-content-lab-candidates` | `exam-bank asterion content-lab` |
| `enrich-ai` | `exam-bank ai enrich` |

The former one-function script wrappers were also removed. Their replacements
are namespaced commands:

| Removed wrapper family | Public replacement |
| --- | --- |
| `build_*advisory*`, `parse_*`, `validate_advisory_evidence.py` | `exam-bank advisory ...` |
| `build_mark_events.py`, `validate_mark_events.py` | `exam-bank marks build/validate` |
| `build/run/import/merge_topic_review*.py` | `exam-bank topic review-*` |
| `build/audit_canonical_sample.py` | `exam-bank extract build-sample/audit-sample` |
| `generate_difficulty_index.py` | `exam-bank topic difficulty-index` |
| `package/verify_asterion_export_release.py` | `exam-bank asterion package/verify` |
| direct `exam_bank.release_manifest` invocation | `exam-bank release build/verify` |
| `audit_output_storage.py` | `exam-bank data audit-storage` |
| `validate_asset_references.py` | `exam-bank data validate-assets` |

Homework/classroom/email commands moved to the sibling `../homework-ingest`
repository and use the `homework-ingest classroom ...` and
`homework-ingest email ...` namespaces. Grading and rubric-readiness commands
moved to `../autograder`; use `autograder grade ...` or
`autograder autograde ...`. They are intentionally absent from the `exam-bank`
command registry.

Source acquisition is no longer a generated `exam_bank_input.jsonl` contract.
Use `manifests/corpora/caie_9709.active_partial.v1.json` with routine
`exam-bank data verify` and `exam-bank data hydrate` calls. The complete
`manifests/corpora/caie_9709.v1.json` ledger remains the restoration and audit
contract for quarantined documents.
