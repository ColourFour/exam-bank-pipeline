# Phase 1 Cleanup Prerequisite Closeout

Date: 2026-05-14

## Summary Verdict

Phase 1 cleanup prerequisite claims are verified against the repository with notes. The generator inspection surfaces are safe under `--help`, available generator `--dry-run` modes completed without generated-output content changes, current-output integrity checks pass, and the Asterion/topic/archive/README documentation is conservative enough to support Phase 2 planning.

Two small closeout test corrections were made during this audit:

- `tests/test_repo_hygiene.py` now checks the supported README `process` command in the current multi-line `.venv/bin/python` format.
- `tests/test_audit.py` now explicitly tests `record_count_mismatch` and absolute nonblank mark-scheme image paths.

No generated output content changed during validation. Phase 2 cleanup/reorganization has not started.

## Final Recommendation

READY_FOR_PHASE_2_WITH_NOTES

## Phase 1 Goal Checklist

| Phase 1 Goal | Claimed Complete? | Repo Evidence | Test Evidence | Verdict | Notes/Risks |
| --- | --- | --- | --- | --- | --- |
| 1. Generator scripts safe to inspect | Yes | `scripts/generate_topic_filter_maps.py` and `scripts/generate_skill_maps.py` both use argparse; `--help` exits before reads/writes; `--dry-run` prints `would_write`. | `tests/test_generator_cli_safety.py`; real `--help` and `--dry-run` commands passed. | pass | `--dry-run` still reads current inputs to validate the plan; that is expected. |
| 2. No-mutation tests added | Yes | `tests/test_generator_cli_safety.py` snapshots tracked state and taxonomy fixture hashes for `--help`; dry-run tests monkeypatch write functions. | `6 passed`; real validation hashes also showed no generated-output content mutation. | pass_with_notes | Tests cover these two generators, not every script in `scripts/`. |
| 3. Current-output integrity audit/test added | Yes | `src/exam_bank/audit.py`, `src/exam_bank/cli.py`, README command atlas. | `tests/test_audit.py tests/test_runtime_paths.py`: `13 passed`; command audit returned `ok: true`. | pass | This verifies identity/path minimum invariants only; it does not verify visual crop correctness, OCR correctness, mark-scheme semantic alignment, or topic accuracy. |
| 4. Asterion export contract documented | Yes | `docs/ASTERION_EXPORT_CONTRACT.md`; README links. | Covered indirectly by full suite/export tests; no doc-specific schema snapshot test. | pass_with_notes | Contract is clear and conservative, but consumer enforcement remains outside this repo unless implemented downstream. |
| 5. Topic routing sidecar contract documented | Yes | `docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md`; linked from README, Asterion contract, and AI enrichment docs. | Producer validation is covered in the full suite; downstream fail-closed consumer behavior is not separately tested. | pass_with_notes | Current sidecar is explicitly unsafe for strict filters: `safe_for_strict_filters=false`. |
| 6. Archive manifest created | Yes | `docs/history/archive_manifests/generated_cleanup_20260513T233456Z.md`. | `output-inventory` and `output-cleanup-plan` commands passed and matched the manifest classification. | pass_with_notes | Good enough for Phase 2 planning; 21 archive-only `p3` PNGs and `status.current.json` remain keep/review items. |
| 7. README current-state references updated | Yes | README centers canonical images as source of truth and points measured counts to the audit baseline. | Full suite passed after updating stale README hygiene assertion. | pass | README is aligned with the audited current state; other docs still need a later consistency pass. |

## Files Changed In Phase 1

Phase 1 evidence indicates these source/docs/tests were changed or added:

- `scripts/generate_topic_filter_maps.py`
- `scripts/generate_skill_maps.py`
- `tests/test_generator_cli_safety.py`
- `src/exam_bank/audit.py`
- `src/exam_bank/cli.py`
- `tests/test_audit.py`
- `tests/test_runtime_paths.py`
- `README.md`
- `docs/ASTERION_EXPORT_CONTRACT.md`
- `docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md`
- `docs/AI_ASSISTED_ENRICHMENT.md`
- `docs/history/archive_manifests/generated_cleanup_20260513T233456Z.md`

Closeout audit additions/corrections:

- `docs/PHASE_1_CLEANUP_PREREQ_CLOSEOUT.md`
- `tests/test_audit.py`
- `tests/test_repo_hygiene.py`

## Commands And Tests Run

- `git status --short --untracked-files=all`: clean before validation; after validation only closeout test corrections were modified before this report.
- `git diff --stat`: clean before validation; after closeout corrections, only test files changed before this report.
- `git diff --name-only`: clean before validation; after closeout corrections, `tests/test_audit.py` and `tests/test_repo_hygiene.py`.
- `git diff --check`: passed before and after validation.
- `.venv/bin/python -m pytest -q`: first run failed on stale README hygiene assertion; after correction, `427 passed, 3 skipped`.
- `.venv/bin/python -m pytest tests/test_generator_cli_safety.py -q`: `6 passed`.
- `.venv/bin/python -m pytest tests/test_audit.py tests/test_runtime_paths.py -q`: `13 passed`.
- `.venv/bin/python scripts/generate_topic_filter_maps.py --help`: exit 0.
- `.venv/bin/python scripts/generate_skill_maps.py --help`: exit 0.
- `.venv/bin/python scripts/generate_topic_filter_maps.py --dry-run`: exit 0, validation `pass`, printed topic-map `would_write`.
- `.venv/bin/python scripts/generate_skill_maps.py --dry-run`: exit 0, validation `pass`, printed skill-map `would_write`.
- `.venv/bin/python -m exam_bank.cli output-integrity-audit --input output/json/question_bank.json --artifact-root output --output output/json/audit.current.integrity.json`: exit 0, `ok: true`.
- `.venv/bin/python -m exam_bank.cli output-inventory --root output/archive/generated_cleanup_20260513T233456Z --include-size --max-depth 4`: exit 0; found 1 question bank, 4 artifact trees, 0 frozen baselines.
- `.venv/bin/python -m exam_bank.cli output-cleanup-plan --root output/archive/generated_cleanup_20260513T233456Z --include-size --max-depth 4`: exit 0; dry run only; keep `output_ocr_candidate`, manual-review `output`.

## Validation Results

Generator safety:

- Both generator `--help` commands exited with usage text and no generated-output content changes.
- Both generator `--dry-run` commands completed successfully and printed planned output paths instead of writing.
- The no-mutation tests protect the previous script-safety issue for these two generator scripts.

Output integrity:

- Current bank `record_count=1301`, actual question count `1301`.
- `question_id` values are unique.
- Duplicate `(paper, question_number)` pairs are rejected by the audit and tested.
- Question image paths are required, relative, and must exist.
- Nonblank mark-scheme image paths must be relative and must exist.
- Missing mark-scheme paths are limited to the documented `9709_2025_November_33` exception.
- Current missing mark-scheme records are exactly `33autumn25_q01` through `33autumn25_q11`.
- Unexpected missing mark schemes fail.
- The audit does not claim visual crop correctness, OCR correctness, mark-scheme semantic alignment, or topic accuracy.

Asterion contract:

- Clearly states canonical rendered images are source of truth.
- Defines Asterion files as role-gated projections, not canonical datasets.
- Treats OCR/native text, detected marks, mark events, and incomplete subpart marks as advisory unless the relevant role gate permits use.
- Defines `allow`, `block`, `block_until_reviewed`, `include`, and `exclude` conservatively.
- Handles missing mark schemes and incomplete subpart marks conservatively.

Topic routing sidecar contract:

- Defines `question_bank.topic_routing.v1.json` as advisory by default.
- Requires strict filters to fail closed unless `metadata.run_summary.safe_for_strict_filters=true`.
- Clearly states the current sidecar is not safe for strict filters.
- Separates deterministic local topic classification, broader AI enrichment, and strict topic routing.
- States AI may choose only supplied canonical parent topic IDs and may not invent topic structure.

Archive manifest:

- Identifies archive root `output/archive/generated_cleanup_20260513T233456Z`.
- Describes inventoried JSON sidecars and OCR candidate image trees.
- Classifies major groups as historical evidence, disposable run evidence, duplicated generated assets, keep-until-reviewed, or unknown.
- Flags archive-only `p3` PNGs.
- States no archive files were deleted, moved, or regenerated.
- Is useful enough as the Phase 2 cleanup planning input, but not enough to approve deletion by itself.

README:

- Aligned with the current audited state by pointing measured counts to `docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md`.
- Centers the image-first source-of-truth model.
- Documents current audit, Asterion, Content Lab, topic routing, AI enrichment, and test commands.

## Generated-Output Mutation Check

Baseline git status before validation was clean.

Pre/post content hashes matched for:

- `output/json/question_bank.json`: `0e14453581f266416264f0f85bf143db2cde627b234e1e1bae7fa2e9ff3a1230`
- `output/json/question_bank.topic_routing.v1.json`: `d05cbd204b4ec05c2108578d9196faf2d12c036ec431d7e7a8083120335c26e1`
- `output/asterion/exports/latest/asterion_question_bank_v1.json`: `1389168852e54a4476fff653a1c5a714c8871525470a82ba3bbefc57f47e266f`
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`: `332eb3c58700ae1e03718743d1f0b7a3ea580f3b7bf825c228cf96969f1119c8`
- `exam_bank_taxonomy/canonical/` aggregate: `17b4ef4d0cdf38b1e818be16aae1c7d88466576c808b040829f059f3955419c4`
- `output/json/` top-level JSON/JSONL aggregate: `4d9fb505a1d190b0c51ff226b612589178a8614419ed8f226b55a521e6efe3c3`
- `output/asterion/exports/latest/` aggregate: `456637d6b7eaa737b891b90a0a6a6b3e7b0bc6c2956b199744e549a6f4ec8d6e`
- `output/archive/generated_cleanup_20260513T233456Z/` aggregate: `ff6624ec46b3384e73730a4e5635ac3cad40bfce90dc5c397da4d2e0aab25ced`
- Current canonical image tree `output/p1 output/p3 output/p4 output/p5` aggregate: `e973776d3f60e11d31bf741751400e0f525930f7ef254e2a66188d3abbdfd12f`

`output-integrity-audit` intentionally wrote `output/json/audit.current.integrity.json`; the aggregate `output/json/` content hash remained unchanged, so this was an intentional and safe validation write with no content mutation.

No accidental or unknown generated-output changes were found. No taxonomy files, current outputs, Asterion exports, topic sidecars, OCR candidate outputs, archived generated outputs, or canonical image paths changed.

## Remaining Risks

- `9709_2025_November_33` still lacks the source mark scheme; 11 records remain an explicit integrity-audit allow-list exception.
- Current topic routing sidecar is not safe for strict filters because `safe_for_strict_filters=false` and `failed_records=153`.
- Asterion and topic sidecar contracts are clear, but downstream consumer fail-closed behavior is only documented unless the consumer implements/tests it.
- Archive manifest supports planning, not deletion approval; 21 archive-only `p3` PNGs need review.
- The integrity audit verifies minimum path/identity invariants only, not semantic or visual correctness.

## Stale Docs Still Known

- `docs/GOALS.md` contains historical command results and future Phase 2/3 request text; keep it as planning/history, not the current operating contract.
- The broader audit identifies additional documentation consistency work for Phase 2/3. This closeout did not perform a full docs consistency pass across ROADMAP and every workflow doc.
- Historical docs should remain clearly separated or bannered during Phase 2 so old commands and counts are not mistaken for current instructions.

## Blockers Before Phase 2

No Phase 1 verification blocker remains for starting Phase 2 planning and conservative reorganization.

Deletion remains blocked for:

- Current generated outputs and canonical image trees unless a fresh regeneration is intentionally validated.
- Archive-only `p3` PNGs until source/current-output expectations are checked.
- Unknown/manual-review archive items, including `status.current.json`, until reviewed.
- Strict topic-filter consumer use of the current sidecar until `safe_for_strict_filters=true`.

## Recommended Phase 2 Starting Point

Start with documentation and archive planning, not deletion:

1. Use the archive manifest plus `output-inventory` and `output-cleanup-plan` as the Phase 2 planning baseline.
2. Review the 21 archive-only `p3` PNGs and decide whether they represent source gaps, old naming, or disposable candidates.
3. Add historical banners or move stale workflow docs before changing generated-output layout.
4. Keep current question bank, Asterion exports, topic sidecar, taxonomy files, and canonical images fixed until a deliberate regeneration/change plan is approved and audited.
