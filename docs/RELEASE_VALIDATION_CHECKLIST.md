# Release Validation Checklist

Use this checklist before publishing a clean current question-bank export or Asterion handoff. It is intentionally document-first: it records the release sequence, expected evidence, and blocking versus warning criteria without adding hard gates to existing workflows.

Run commands from the repository root after installing the dev environment:

```bash
source .venv/bin/activate
```

## Release Inputs

- Canonical question bank: `output/json/question_bank.json`
- Canonical image artifacts: `output/p*/<paper>/questions/*.png` and `output/p*/<paper>/mark_scheme/*.png`
- Topic routing sidecar: `output/json/question_bank.topic_routing.v1.json`
- Mark-event sidecar: `output/json/question_bank.mark_events.v1.json`
- Advisory evidence sidecar: `output/advisory_evidence/question_bank.advisory_evidence.v1.json`
- Difficulty index sidecar: `output/json/question_bank.difficulty_index.v1.json`
- Asterion export: `output/asterion/exports/latest/asterion_question_bank_v1.json`
- Content Lab candidates: `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- Validation report directory: `output/audits/current/`
- Inventory reports: `output/output_inventory.md`, `output/output_inventory.json`
- Cleanup-plan reports: `output/output_cleanup_plan.md`, `output/output_cleanup_plan.json`

## Gate Types

Blocking means the export should not be treated as release-quality until the issue is fixed, explicitly waived, or scoped out of the release.

Warning means the export can continue through controlled review or downstream handoff, but the condition must be recorded in release notes and must not be silently promoted to student-facing eligibility.

The current known exception is the missing source mark scheme for `9709_2025_November_33`. This accounts for `33autumn25_q01` through `33autumn25_q11` missing mark-scheme image paths. It is a warning only while those records remain blocked or review-only in Asterion-facing roles. It becomes blocking if any of those records are promoted to student-facing practice, quick checks, generation input, or any role that requires mark-scheme image availability.

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
.venv/bin/python -m exam_bank.cli audit \
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
.venv/bin/python -m exam_bank.cli output-integrity-audit \
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
- Missing, absolute, or unresolved mark-scheme image paths outside the documented `9709_2025_November_33` exception.
- Declared `record_count` mismatch.

Warning:

- The allowed `9709_2025_November_33` missing mark-scheme companion, provided affected records remain blocked or review-only.
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
.venv/bin/python scripts/validate_mark_events.py \
  --question-bank output/json/question_bank.json \
  --mark-events output/json/question_bank.mark_events.v1.json \
  --artifact-root output \
  --output output/json/question_bank.mark_events.validation.v1.json
```

```bash
.venv/bin/python scripts/validate_advisory_evidence.py \
  --advisory-root output/advisory_evidence \
  --question-bank output/json/question_bank.json \
  --output output/advisory_evidence/validation.v1.json
```

```bash
.venv/bin/python scripts/generate_difficulty_index.py --dry-run
```

Expected evidence:

- `output/json/question_bank.mark_events.validation.v1.json`
- `output/advisory_evidence/validation.v1.json`
- Difficulty-index dry-run summary

Blocking:

- Mark-event validation exits nonzero.
- Advisory-evidence validation exits nonzero or reports validation errors.
- Difficulty-index dry run exits nonzero.
- Any sidecar claims student-facing marking, strict topic filtering, or student sequencing without a separate approved release gate.

Warning:

- Advisory-evidence duplicate-source warnings are acceptable only when visible in validation/review reports and retained as review evidence.
- Difficulty-index low-confidence, unsafe, and review-queue records are expected; they must stay out of student-facing sequencing in v1.

When producing the release export, regenerate both projections from the canonical question bank:

```bash
.venv/bin/python -m exam_bank.cli asterion-export \
  --input output/json/question_bank.json \
  --artifact-root output
```

```bash
.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates \
  --input output/json/question_bank.json \
  --artifact-root output
```

Expected evidence:

- `output/asterion/exports/latest/asterion_question_bank_v1.json`
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- Passing Asterion tests from the test step.

Blocking:

- Command exits nonzero.
- Missing expected export files.
- Export `record_count` does not match the source question bank.
- Any record with missing required canonical image integrity is marked student-facing `allow`.
- Any `9709_2025_November_33` record is promoted to a role that requires mark-scheme availability before the missing mark scheme is resolved.
- Content Lab candidate `generation_gate.status=allow` appears without reviewed/approved prerequisites required by `docs/ASTERION_EXPORT_CONTRACT.md`.

Warning:

- Limited student-facing eligibility is expected; role-blocked and `blocked_until_reviewed` records may remain in the export as long as downstream roles preserve those statuses.
- Incomplete subpart marks remain warning-level only when full-question mark totals and rendered mark-scheme images remain available and role gates are conservative.

### 6. Validate Topic Sidecar Safety Metadata

Inspect `output/json/question_bank.topic_routing.v1.json` before using strict topic filters.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("output/json/question_bank.topic_routing.v1.json")
payload = json.loads(path.read_text(encoding="utf-8"))
summary = payload.get("metadata", {}).get("run_summary", {})
print(json.dumps({
    "schema_name": payload.get("schema_name"),
    "schema_version": payload.get("schema_version"),
    "record_count": payload.get("record_count"),
    "records": len(payload.get("records", {})),
    "safe_for_strict_filters": summary.get("safe_for_strict_filters"),
    "failed_records": summary.get("failed_records"),
    "review_required_records": summary.get("review_required_records"),
    "strict_filter_records": summary.get("strict_filter_records"),
}, indent=2))
PY
```

Expected evidence:

- Terminal summary of sidecar schema and `metadata.run_summary`.
- Current sidecar path: `output/json/question_bank.topic_routing.v1.json`

Blocking:

- For strict topic filtering only: missing sidecar, schema mismatch, `record_count` mismatch, `safe_for_strict_filters` not true, or `failed_records` not zero.
- Unknown topic IDs, malformed distributions, duplicate topic IDs, or distributions that do not total `100` in strict-filter candidates.

Warning:

- The current known state has `safe_for_strict_filters=false` and failed records, so strict topic filters must remain disabled. The sidecar can still be used as review evidence if the failed and review-required states are preserved.
- Local `topic`, legacy `difficulty`, deterministic advisory evidence, and the difficulty index are advisory unless a separate release review approves their consumer role. Grade-threshold context must not be used as direct individual-question difficulty proof.

### 7. Run Output Inventory And Cleanup Plan

```bash
.venv/bin/python -m exam_bank.cli output-inventory \
  --root output \
  --write output/output_inventory.md \
  --json output/output_inventory.json
```

```bash
.venv/bin/python -m exam_bank.cli output-cleanup-plan \
  --root output \
  --write output/output_cleanup_plan.md \
  --json output/output_cleanup_plan.json
```

For cleanup work that depends on archived generated evidence, include the archive root:

```bash
.venv/bin/python -m exam_bank.cli output-inventory \
  --root output \
  --root output/archive/generated_cleanup_20260513T233456Z \
  --include-size \
  --max-depth 4 \
  --write output/output_inventory.md \
  --json output/output_inventory.json
```

```bash
.venv/bin/python -m exam_bank.cli output-cleanup-plan \
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
- Warnings carried forward, including the `9709_2025_November_33` missing mark-scheme exception if still unresolved.
- Downstream role restrictions, especially Asterion role gates and strict topic-filter status.
