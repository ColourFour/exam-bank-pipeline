# Release Validation Checklist

Use this checklist before publishing a clean current question-bank export or Asterion handoff. It is intentionally document-first: it records the release sequence, expected evidence, and blocking versus warning criteria without adding hard gates to existing workflows.

Run commands from the repository root after installing the dev environment:

```bash
source .venv/bin/activate
```

## Release Inputs

- Canonical question bank: `output/json/question_bank.json`
- Canonical image artifacts: `output/pm1/*.png`, `output/pm3/*.png`, `output/stats/*.png`, and `output/mechanics/*.png`
- Durable topic routing sidecar: `data/topic_routing/question_bank.topic_routing.v1.json`
- Question-bank release manifest: `manifests/releases/question_bank_release_manifest.v1.json`
- Optional restored topic-routing cache: `output/json/question_bank.topic_routing.v1.json`
- Mark-event sidecar: `output/json/question_bank.mark_events.v1.json`
- Advisory evidence sidecar: `output/advisory_evidence/question_bank.advisory_evidence.v1.json`
- Difficulty index sidecar: `output/json/question_bank.difficulty_index.v1.json`
- Asterion all-course catalog: `output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json`
- Asterion student runtime question bank: `output/asterion/exports/latest/asterion_question_bank_v1.json`
- Content Lab candidates: `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- Validation report directory: `output/audits/current/`
- Inventory reports: `output/output_inventory.md`, `output/output_inventory.json`
- Cleanup-plan reports: `output/output_cleanup_plan.md`, `output/output_cleanup_plan.json`

## Gate Types

Blocking means the export should not be treated as release-quality until the issue is fixed, explicitly waived, or scoped out of the release.

Warning means the export can continue through controlled review or downstream handoff, but the condition must be recorded in release notes and must not be silently promoted to student-facing eligibility.

Do not reuse exceptions from an older release. Every missing artifact, count deviation, or waived gate must be recorded against the exact hash-bound release manifest being validated.

The canonical identity contract is component- and era-aware: Paper 4 is
`mechanics`/M1; current Paper 5 is `stats`/S1; current Paper 6 is `stats`/S2;
pre-2020 Paper 6 is `stats`/S1; and pre-2020 Paper 5 (M2) is unsupported. A
release containing reversed family labels, collapsed March/June IDs, or an
era-incompatible course is blocking.
Follow [Canonical Identity And Release Migration](CANONICAL_IDENTITY_RELEASE_MIGRATION.md) when moving an existing bank to this contract.

## Checklist

### 1. Run Tests

```bash
.venv/bin/python -m pytest -q
```

Expected evidence:

- Terminal result from the full pytest suite.

Blocking:

- Any failing test.
- Any test environment failure that prevents the suite from running.

Warning:

- Skipped tests are acceptable only when they match the existing expected skip profile and are not newly introduced by the release.

### 2. Run Question-Bank Audit

```bash
exam-bank extract audit \
  --input output/json/question_bank.json \
  --output output/json/audit.current.json
```

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --artifact-root output \
  --out-dir output/audits/current
```

Expected evidence:

- `output/json/audit.current.json`
- Readiness CSV, Markdown, and JSON reports under `output/audits/current/`

Blocking:

- Command exits nonzero.
- `record_count` does not match the canonical question-bank payload.
- New failed mapping, failed validation, unusable text-only status, missing identity, or missing canonical question image conditions that are not intentionally scoped and documented.

Warning:

- Review-tier records, visual-required records, degraded text, and low-confidence metadata when the affected records remain role-gated or review-only.

### 3. Run Image Integrity Check

```bash
exam-bank extract integrity \
  --input output/json/question_bank.json \
  --artifact-root output \
  --output output/json/audit.current.integrity.json
```

Expected evidence:

- `output/json/audit.current.integrity.json`

Blocking:

- `ok` is not true.
- Duplicate `question_id` values.
- Duplicate `(paper, question_number)` pairs.
- Missing, absolute, or unresolved question image paths.
- Missing, absolute, or unresolved mark-scheme image paths.
- Declared `record_count` mismatch.

Warning:

- Integrity passing does not prove crop correctness; visual crop spot-checks still belong in release review when the source set or detection logic changed.

### 4. Run OCR Candidate Audit

```bash
.venv/bin/python scripts/audit_ocr_candidates.py \
  --input output/json/question_bank.json \
  --json-output output/audits/current/ocr_candidate_audit.json
```

If validating a new OCR candidate against the current canonical export:

```bash
.venv/bin/python scripts/audit_ocr_candidates.py \
  --input output/candidates/ocr/latest/json/question_bank.json \
  --baseline output/json/question_bank.json \
  --json-output output/audits/current/ocr_candidate_comparison.json
```

Expected evidence:

- `output/audits/current/ocr_candidate_audit.json`
- Optional `output/audits/current/ocr_candidate_comparison.json`

Blocking:

- Command exits nonzero.
- OCR metadata is missing from an OCR-enabled release export.
- OCR selection movement introduces new failed validation, failed mapping, unusable text, or identity changes that are not explicitly scoped and reviewed.

Warning:

- OCR selected over native text for a small subset of records when selection reasons and trust metadata are present.
- Candidate comparison changes that are reviewable and do not alter canonical image artifacts or role gates.

### 5. Generate Or Validate Asterion Exports

Before regenerating Asterion projections, validate advisory sidecars that the export or downstream release notes may reference:

```bash
exam-bank data rebind-text-gold --write
exam-bank data validate-review-assets
```

```bash
exam-bank marks validate \
  --question-bank output/json/question_bank.json \
  --mark-events output/json/question_bank.mark_events.v1.json \
  --artifact-root output \
  --output output/json/question_bank.mark_events.validation.v1.json
```

```bash
exam-bank advisory validate \
  --advisory-root output/advisory_evidence \
  --question-bank output/json/question_bank.json \
  --output output/advisory_evidence/validation.v1.json
```

```bash
exam-bank topic difficulty-index --dry-run
```

Expected evidence:

- `manifests/validations/question_text_gold_asset_binding.v1.json`
- `manifests/validations/review_asset_binding_validation.v1.json`
- `output/json/question_bank.mark_events.validation.v1.json`
- `output/advisory_evidence/validation.v1.json`
- Difficulty-index dry-run summary

Blocking:

- A verified text-gold record does not bind to a unique current canonical
  question image by SHA-256.
- A release-affecting source-skill, mark-event, or Content Lab approval refers
  to reviewed image bytes that differ from the current canonical assets. The
  consumers fail closed by demoting this evidence, but it remains a release
  policy blocker until re-reviewed.
- Mark-event validation exits nonzero.
- Advisory-evidence validation exits nonzero or reports validation errors.
- Difficulty-index dry run exits nonzero.
- Any sidecar claims student-facing marking, strict topic filtering, or student sequencing without a separate approved release gate.

Warning:

- Advisory-evidence duplicate-source warnings are acceptable only when visible in validation/review reports and retained as review evidence.
- Difficulty-index low-confidence, unsafe, and review-queue records are expected; they must stay out of student-facing sequencing in v1.

When producing the release export, regenerate both projections from the canonical question bank:

```bash
exam-bank asterion export \
  --input output/json/question_bank.json \
  --artifact-root output
```

```bash
exam-bank asterion content-lab \
  --input output/json/question_bank.json \
  --artifact-root output
```

Expected evidence:

- `output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json`
- `output/asterion/exports/latest/asterion_question_bank_v1.json`
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- Passing Asterion tests from the test step.
- Passing course-aware Asterion loader tests from `tests/test_asterion_course_contract.py`.

Blocking:

- Command exits nonzero.
- Missing expected export files.
- Catalog `record_count` does not match the source question bank.
- Student runtime export contains unsafe, candidate, blocked, or needs-review records.
- Any runtime record with missing required canonical image integrity is marked student-facing `allow`.
- Any record missing a required canonical artifact is promoted to a role that requires it.
- Content Lab candidate `generation_gate.status=allow` appears without reviewed/approved prerequisites required by `docs/ASTERION_EXPORT_CONTRACT.md`.
- P1, M1, S1, or S2 student runtime uses records from another component or bypasses the reviewed/safe runtime gate.

Warning:

- Limited student-facing eligibility is expected; role-blocked and `blocked_until_reviewed` records may remain in the all-course catalog as long as downstream roles preserve those statuses.
- Incomplete subpart marks remain warning-level only when full-question mark totals and rendered mark-scheme images remain available and role gates are conservative.
- Any course or filter may legitimately show `No reviewed exam-bank records available yet.` when its reviewed/safe runtime subset is empty.

### 6. Bind And Verify Topic-Routing Release Inputs

Refresh writes the durable sidecar, its compatibility checksum, and the release manifest. Verify the manifest before any strict consumer runs:

```bash
exam-bank topic refresh-routing --write
exam-bank topic verify-release
```

Restore the optional local cache only for a tool that still requires the sibling path:

```bash
exam-bank topic restore-release
```

Expected evidence:

- `data/topic_routing/question_bank.topic_routing.v1.json`
- `data/topic_routing/question_bank.topic_routing.v1.sha256`
- `manifests/releases/question_bank_release_manifest.v1.json`
- A verification report with `ok=true`, exact question-ID coverage, and matching hashes.

For a downstream handoff, rebuild this as the full multi-role bundle after mark
events, difficulty, and Asterion outputs are regenerated. Use `exact`
coverage for one-record-per-question sidecars and the Asterion catalog, `subset`
for runtime/candidate/promotion artifacts, and dependency bindings for consumed
inputs. See [Question-Bank Release Manifest Contract](RELEASE_MANIFEST_CONTRACT.md)
for the command and role mapping.

Blocking:

- Missing release manifest or any SHA-256, size, schema, record-count, binding, or exact question-ID-set mismatch.
- For strict topic filtering only: `safe_for_strict_filters` not true or failed routes are nonzero.
- Unknown topic IDs, malformed distributions, duplicate topic IDs, or distributions that do not total `100` in strict-filter candidates.

Warning:

- P6/S2 records remain review-only until an approved P6 packet taxonomy exists; they must never be relabeled as P5/S1 to increase coverage.
- Review-required routes may remain in the complete sidecar, but consumers must exclude them from strict filters.
- Local `topic`, legacy `difficulty`, deterministic advisory evidence, and the difficulty index are advisory unless a separate release review approves their consumer role. Grade-threshold context must not be used as direct individual-question difficulty proof.

### 7. Run Output Inventory And Cleanup Plan

```bash
exam-bank data inventory \
  --root output \
  --write output/output_inventory.md \
  --json output/output_inventory.json
```

```bash
exam-bank data cleanup-plan \
  --root output \
  --write output/output_cleanup_plan.md \
  --json output/output_cleanup_plan.json
```

For cleanup work that depends on archived generated evidence, include the archive root:

```bash
exam-bank data inventory \
  --root output \
  --root output/archive/generated_cleanup_20260513T233456Z \
  --include-size \
  --max-depth 4 \
  --write output/output_inventory.md \
  --json output/output_inventory.json
```

```bash
exam-bank data cleanup-plan \
  --root output \
  --root output/archive/generated_cleanup_20260513T233456Z \
  --include-size \
  --max-depth 4 \
  --write output/output_cleanup_plan.md \
  --json output/output_cleanup_plan.json
```

Expected evidence:

- `output/output_inventory.md`
- `output/output_inventory.json`
- `output/output_cleanup_plan.md`
- `output/output_cleanup_plan.json`

Blocking:

- Command exits nonzero.
- Cleanup plan classifies current canonical bank, current image artifacts, current Asterion exports, current topic sidecar, or required archive evidence as disposable.
- Inventory shows missing current export files expected by this checklist.

Warning:

- Unknown/manual-review classifications in archive or historical evidence are acceptable if no deletion or move is performed during release validation.
- Cleanup plan is dry-run only. Actual deletion, movement, compression, or regeneration requires a separate reviewed cleanup task.

## Release Decision Record

Record the final release decision with:

- Date and operator.
- Git commit or worktree state.
- Whether the release regenerated exports or validated existing exports.
- Commands run and pass/fail status.
- Paths to audit, integrity, OCR, Asterion, topic-sidecar, inventory, and cleanup-plan evidence.
- Blocking issues found and how each was fixed, waived, or scoped out.
- Warnings carried forward against this exact release, including review-only P6 topic routing where applicable.
- Downstream role restrictions, especially Asterion role gates and strict topic-filter status.
