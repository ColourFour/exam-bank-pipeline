# Phase 2 Cleanup/Reorganization Audit

Audit date: 2026-05-14

Repository: `exam-bank-pipeline`

Commit audited: `2eeb414cb54db98b30ea208bcd9c69415c42308b`

## 1. Summary Verdict

READY_FOR_PHASE_3_WITH_NOTES

Phase 2 cleanup and reorganization is complete enough to build on. The work is conservative, documentation-led, and preserves the image-first source-of-truth policy. Current generated outputs, current exports, taxonomy files, topic sidecars, and canonical image trees were not modified during this audit.

The notes are:

- The 21 archive-only `p3` PNG exceptions remain intentionally unresolved and protected.
- `status.current.json` remains review-needed, not disposable.
- The required literal question-bank count snippet is stale for the current schema because the bank uses top-level `questions`, not top-level `records`; a schema-correct count confirmed 1,301 records.

## 2. Scope

This was a Phase 2 cleanup/reorganization audit only.

This audit did not implement Phase 3, start behavioral optimization, regenerate outputs, rewrite taxonomy, alter current exports, delete current images, change source behavior, or modify topic routing / AI enrichment behavior.

## 3. Goal-by-Goal Findings

### Goal 8 - Move or label historical docs

Claimed complete: yes.

Evidence found:

- `docs/history/PROJECT_REVIEW.md` exists.
- `docs/PROJECT_REVIEW.md` does not exist.
- `docs/history/PROJECT_REVIEW.md` begins with a clear historical banner directing readers to `docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md` for current measured state.
- `ROADMAP.md` now points current measured state to the audit baseline and labels older iteration counts as historical evidence.
- `docs/TRUST_MODEL.md` no longer carries stale tier counts as current truth and tells readers to use the audit baseline or rerun audit/readiness commands.
- `docs/AUTO_TRIAGE.md` removes fixed hard-failure/current count snapshots and treats `output_ocr_candidate/` as historical/compatibility evidence.
- `docs/TRIAGE_WORKFLOW.md` uses generic/current-path examples such as `output/candidates/ocr/latest/`.
- `docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md` has a post-audit note explaining the `PROJECT_REVIEW.md` move.
- Old `docs/PROJECT_REVIEW.md` path mentions remain only as historical audit-time references or in `docs/GOALS.md` progress notes.

Commands/tests run:

- `test -f docs/history/PROJECT_REVIEW.md`: pass.
- `test ! -f docs/PROJECT_REVIEW.md`: pass.
- `rg -n "\(docs/PROJECT_REVIEW.md\)|docs/PROJECT_REVIEW.md" README.md ROADMAP.md docs || true`: only historical/progress-note matches.
- `git diff --name-only -- src tests exam_bank_taxonomy output README.md ROADMAP.md docs | sort`: no changed files before this report was written.

Verdict: pass.

Notes/risks:

- The moved project review still contains old "current" language by design, but the banner is prominent enough to prevent using it as the current baseline.

### Goal 9 - Normalize command documentation

Claimed complete: yes.

Evidence found:

- `docs/COMMAND_ATLAS.md` exists and is operator-useful: it includes purpose, inputs, outputs, category/runtime, and command examples.
- The atlas separates read-only/inspection/audit commands from mutating or output-generating commands through the `Mutates outputs` summary column and per-command category text.
- Mutating or potentially mutating surfaces are explicit: `process`, Asterion projections, AI sidecars, taxonomy generators unless `--dry-run`.
- Inspection-only surfaces are explicit: `output-inventory`, `output-cleanup-plan`, `ai-sidecar-audit`, tests, and audit commands with optional report writes.
- README points to the atlas and no longer contains stale command tables.
- Active docs no longer contain stale invalid `--output-dir` examples; the only active-doc hit is a `docs/GOALS.md` note saying such stale flags were replaced.
- CLI help surfaces were present for the documented commands.

Commands/tests run:

- `.venv/bin/python -m exam_bank.cli --help`: pass.
- `.venv/bin/python -m exam_bank.cli process --help`: pass.
- `.venv/bin/python -m exam_bank.cli audit --help`: pass.
- `.venv/bin/python -m exam_bank.cli output-integrity-audit --help`: pass.
- `.venv/bin/python -m exam_bank.cli output-inventory --help`: pass.
- `.venv/bin/python -m exam_bank.cli output-cleanup-plan --help`: pass.
- `.venv/bin/python -m exam_bank.cli asterion-export --help`: pass.
- `.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --help`: pass.
- `.venv/bin/python -m exam_bank.cli topic-route-ai --help`: pass.
- `.venv/bin/python -m exam_bank.cli enrich-ai --help`: pass.
- `.venv/bin/python -m exam_bank.cli ai-sidecar-audit --help`: pass.
- `rg -n -- "--output-dir" README.md ROADMAP.md docs/*.md || true`: only a `docs/GOALS.md` historical/progress note.

Verdict: pass.

Notes/risks:

- No command behavior changed during this audit. The pre-report working tree was clean.

### Goal 10 - Classify archived generated artifacts, no deletion

Claimed complete: yes.

Evidence found:

- `docs/history/archive_manifests/generated_cleanup_20260513T233456Z.md` is actionable and conservative.
- It classifies formal evidence, keep-until-reviewed material, compress-or-keep material, regenerate-on-demand material, and delete-later material.
- The manifest says no files were deleted, moved, compressed, regenerated, or rewritten during the classification/recommendation pass.
- The 21 archive-only `p3` PNGs are explicitly `keep-until-reviewed`.
- `status.current.json` is classified as `unknown` / `Keep until reviewed`, not disposable.
- AI/API sidecars are treated as evidence snapshots, with repeated warnings that model/API results are not exactly reproducible.

Commands/tests run:

- `sed -n '1,260p' docs/history/archive_manifests/generated_cleanup_20260513T233456Z.md`: reviewed classification and retention text.
- `test -f output/archive/generated_cleanup_20260513T233456Z/output/json/status.current.json`: pass.
- Targeted Python check for the 21 archive-only `p3` PNGs: all 21 exist in the archive and zero exist in current output.

Verdict: pass.

Notes/risks:

- The manifest file was later updated by Goal 11 to record a first cleanup pass. The Goal 10 no-deletion claim remains clearly documented for the classification stage.

### Goal 11 - First safe generated-output cleanup

Claimed complete: yes.

Evidence found:

- `find output/archive/generated_cleanup_20260513T233456Z -type d -name "*.batches" -print`: no remaining batch cache directories.
- `find output/archive/generated_cleanup_20260513T233456Z -type f -name "*.failures.jsonl" -print`: no remaining failure JSONL files.
- The archive manifest retains summaries for the removed failure JSONL files and batch cache directories.
- The manifest lists exactly seven removed `*.batches/` directories and five removed `*.failures.jsonl` files.
- Current `output/json/question_bank.json` exists, declares `record_count: 1301`, and contains 1,301 top-level `questions`.
- `output-integrity-audit` passed with `ok: true`, 1,301 records, and only the known 11 missing mark-scheme companions for `9709_2025_November_33`.
- Current canonical image trees remain present: 2,591 current PNGs under `output/p1`, `output/p3`, `output/p4`, and `output/p5`.
- Archive image trees remain present: 2,612 archived PNGs under `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1..p5`.
- The 21 archive-only `p3` exceptions remain preserved and unresolved.
- `output-cleanup-plan --root output` still classifies current canonical output as `keep: canonical/current` and `output/archive` as `unknown/manual review`.
- Full tests passed.

Commands/tests run:

- `.venv/bin/python -m exam_bank.cli output-integrity-audit`: pass, `ok: true`, `record_count: 1301`.
- `.venv/bin/python -m exam_bank.cli output-inventory --root output --include-size --max-depth 4`: pass, found one current question bank, eight artifact trees, and two current Asterion exports.
- `.venv/bin/python -m exam_bank.cli output-cleanup-plan --root output --include-size --max-depth 4`: pass, dry run only.
- `find output/archive/generated_cleanup_20260513T233456Z -type d -name "*.batches" -print`: no output.
- `find output/archive/generated_cleanup_20260513T233456Z -type f -name "*.failures.jsonl" -print`: no output.
- `find output/archive/generated_cleanup_20260513T233456Z -type f | wc -l`: `2626`.
- `du -sk output/archive/generated_cleanup_20260513T233456Z output`: `473764` KiB for the archive, `937328` KiB for `output`.
- `.venv/bin/python -m pytest -q`: `427 passed, 3 skipped in 121.02s`.

Verdict: pass.

Notes/risks:

- The literal required count snippet failed because it expects a top-level `records` key or a list and therefore printed `records: 5` from the number of top-level keys. The current schema uses top-level `questions`; the schema-correct check printed `records: 1301` and `declared_record_count: 1301`.

### Goal 12 - Update roadmap based on audit phases

Claimed complete: yes.

Evidence found:

- `ROADMAP.md` reflects the agreed phase model:
  - Phase 1: cleanup prerequisites.
  - Phase 2: cleanup and reorganization.
  - Phase 3: low-risk optimization.
  - Phase 4: future deeper refactors, not current implementation scope.
- The roadmap explicitly says Phase 2 should not silently change Asterion eligibility, record counts, topic safety, text readiness, or current generated output meaning.
- Exam-report and grade-boundary leverage remains deferred future work only.
- No topic, difficulty, enrichment, source, test, taxonomy, output, or export files were changed before this report was written.

Commands/tests run:

- `sed -n '1,110p' ROADMAP.md`: reviewed phase model.
- `rg -n "exam-report|grade-boundary|grade boundary|exam report" README.md ROADMAP.md docs src tests || true`: active implementation hits are roadmap/future-only notes; no source/test implementation found.
- `git diff --name-only -- src tests exam_bank_taxonomy output README.md ROADMAP.md docs | sort`: no changed files before this report was written.

Verdict: pass.

Notes/risks:

- Phase 3 and Phase 4 did not begin during this audit.

## 4. Current Output Safety

Current outputs remained fixed during this audit.

Explicitly checked:

- `output/json/question_bank.json`: present, `record_count: 1301`, 1,301 `questions`.
- Current canonical image trees: present under `output/p1`, `output/p3`, `output/p4`, and `output/p5`.
- Current Asterion exports: present in `output/asterion/exports/latest/`.
- Current topic sidecar and AI sidecars were not regenerated or rewritten.
- Canonical taxonomy under `exam_bank_taxonomy/` was not modified.
- Source behavior under `src/` and tests under `tests/` were not modified.

The commands run during the audit were read-only help, inventory, cleanup-plan dry-run, integrity, archive inspection, and tests. No command with output generation or deletion intent was run.

## 5. Archive Safety

Archive protections remain in place.

Explicitly checked:

- The 21 archive-only `p3` PNG exceptions are still present under `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p3/...`.
- Those 21 PNGs are still absent from current output and remain unresolved.
- `output/archive/generated_cleanup_20260513T233456Z/output/json/status.current.json` exists and remains review-needed.
- Archived AI sidecar JSON snapshots remain present.
- Archived image trees remain present with 2,612 PNGs.
- `output/archive` remains classified as `unknown/manual review` by the cleanup plan.
- No remaining `*.batches/` directories or `*.failures.jsonl` files were found under the archive, matching the documented Goal 11 cleanup scope.

## 6. Validation

Exact commands run and results:

```bash
git status --short
```

Result before report: clean working tree.

```bash
git rev-parse HEAD
```

Result: `2eeb414cb54db98b30ea208bcd9c69415c42308b`.

```bash
git diff --stat
git diff --check
```

Result before report: no diff and no whitespace errors.

```bash
rg -n "PROJECT_REVIEW.md|output_ocr_candidate|current|ready|review|fail|degraded|--output-dir|audit_.*\.py|COMMAND_ATLAS|generated_cleanup_20260513T233456Z" README.md ROADMAP.md docs || true
rg -n "\(docs/PROJECT_REVIEW.md\)|docs/PROJECT_REVIEW.md" README.md ROADMAP.md docs || true
test -f docs/history/PROJECT_REVIEW.md
test ! -f docs/PROJECT_REVIEW.md
```

Result: broad matches are expected current/historical references; old project-review path references are historical/audit-time notes; path tests passed.

```bash
.venv/bin/python -m exam_bank.cli --help
.venv/bin/python -m exam_bank.cli process --help
.venv/bin/python -m exam_bank.cli audit --help
.venv/bin/python -m exam_bank.cli output-integrity-audit --help
.venv/bin/python -m exam_bank.cli output-inventory --help
.venv/bin/python -m exam_bank.cli output-cleanup-plan --help
.venv/bin/python -m exam_bank.cli asterion-export --help
.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --help
.venv/bin/python -m exam_bank.cli topic-route-ai --help
.venv/bin/python -m exam_bank.cli enrich-ai --help
.venv/bin/python -m exam_bank.cli ai-sidecar-audit --help
```

Result: all passed.

```bash
.venv/bin/python -m exam_bank.cli output-integrity-audit
```

Result: passed with `ok: true`, `record_count: 1301`, and only the known 11 missing mark-scheme companions for `9709_2025_November_33`.

```bash
.venv/bin/python -m exam_bank.cli output-inventory --root output --include-size --max-depth 4
.venv/bin/python -m exam_bank.cli output-cleanup-plan --root output --include-size --max-depth 4
```

Result: passed. Inventory found current question bank, current image trees, archived candidate image trees, and two current Asterion exports. Cleanup plan was dry-run only, kept current canonical outputs, and left `output/archive` as manual review.

```bash
find output/archive/generated_cleanup_20260513T233456Z -type d -name "*.batches" -print
find output/archive/generated_cleanup_20260513T233456Z -type f -name "*.failures.jsonl" -print
find output/archive/generated_cleanup_20260513T233456Z -type f | wc -l
du -sk output/archive/generated_cleanup_20260513T233456Z output
```

Result: no batch directories, no failure JSONL files, 2,626 archive files, `473764` KiB archive size, `937328` KiB output size.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("output/json/question_bank.json")
data = json.loads(path.read_text())
records = data["records"] if isinstance(data, dict) and "records" in data else data
print("records:", len(records))
assert len(records) == 1301
PY
```

Result: failed because the current schema has top-level `questions`, not top-level `records`; it printed `records: 5`.

Schema-correct replacement check:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("output/json/question_bank.json")
data = json.loads(path.read_text())
records = data.get("records") or data.get("questions") if isinstance(data, dict) else data
print("records:", len(records))
print("declared_record_count:", data.get("record_count") if isinstance(data, dict) else "n/a")
assert len(records) == 1301
assert data.get("record_count") == 1301
PY
```

Result: passed, `records: 1301`, `declared_record_count: 1301`.

```bash
.venv/bin/python -m pytest -q
```

Result: `427 passed, 3 skipped in 121.02s`.

```bash
git ls-files output | head
git ls-files output/archive | head
git status --short
```

Result before report: only `output/json/.gitkeep` is tracked under `output`; no tracked archive files; working tree clean.

Additional targeted checks:

```bash
find output/p1 output/p3 output/p4 output/p5 -type f -name "*.png" | wc -l
find output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1 output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p3 output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p4 output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p5 -type f -name "*.png" | wc -l
```

Result: `2591` current PNGs and `2612` archived candidate PNGs.

## 7. Changes Made During Audit

Created this audit report:

- `docs/history/PHASE_2_AUDIT_20260514.md`

No source, tests, taxonomy, current outputs, current exports, topic sidecars, AI enrichment behavior, or canonical images were changed.

## 8. Risks / Follow-ups

- The 21 archive-only `p3` PNGs still require review against source documents and current-output expectations.
- `status.current.json` still requires review before it can be marked disposable.
- `output/archive` remains protected/manual-review unless a later audited deletion plan exists.
- Phase 3 should remain low-risk and operational unless separately approved.
- Future exam-report and grade-boundary enrichment should remain deferred until a separate audited proposal exists.
- The question-bank count snippet used in this audit request should be updated for the current `questions` schema before reuse.

## 9. Final Recommendation

Phase 2 is complete enough to proceed to Phase 3.

Proceed with Phase 3 only as low-risk operational optimization. Keep standard outputs, current exports, taxonomy, topic sidecars, AI sidecars, and canonical image trees fixed unless a separate audited regeneration/change plan explicitly authorizes a change.
