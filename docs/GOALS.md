Goals.md -> complete phase 1 - 3 then move to leveraging exam report and grade boundary

Phase 1 goals: safe cleanup prerequisites


Goal 1: make generator scripts safe to inspect
Files Changed


generate_topic_filter_maps.py

generate_skill_maps.py

test_generator_cli_safety.py


What Changed


Added proper argparse CLIs so --help exits before input reads or writes.

Added --dry-run to both generator scripts; it builds/validates and prints would_write without writing files.

Updated stale defaults from output_ocr_candidate/... to:

output/json/question_bank.json

output/asterion/exports/latest/asterion_question_bank_v1.json

output/asterion/exports/latest/asterion_content_lab_candidates_v1.json


Preserved normal generation write targets and exported filenames.


Validation


Targeted tests: .venv/bin/python -m pytest tests/test_generator_cli_safety.py -> 4 passed

Real help checks:

scripts/generate_topic_filter_maps.py --help -> exited 0

scripts/generate_skill_maps.py --help -> exited 0




Real dry-runs:

generate_topic_filter_maps.py --dry-run -> validation pass, no writes

generate_skill_maps.py --dry-run -> validation pass, no writes




Full suite: .venv/bin/python -m pytest -> 423 passed, 3 skipped

git status was unchanged after help/dry-run/full tests; no taxonomy files were modified.


Risks / Concerns


The worktree was not clean before this task due unrelated untracked docs/input files, and those remain.

--dry-run still reads the required current inputs to validate the generation plan; only --help is guaranteed not to read large inputs.


Suggested Next Steps


Consider documenting the new generator CLI usage in the taxonomy maintenance docs.


Goal 2: add no-mutation tests for script help/dry-run behavior
Implemented the regression coverage in tests/test_generator_cli_safety.py (line 1).

Files Changed


tests/test_generator_cli_safety.py (line 1)

Note: docs/GOALS.md was already modified before my changes and I left it untouched.


What Changed


Added subprocess-level --help tests for:

scripts/generate_topic_filter_maps.py

scripts/generate_skill_maps.py




The tests verify:

--help exits successfully.

tracked file status is unchanged.

generated taxonomy fixture contents/file set are unchanged.




Existing deterministic --dry-run no-write tests remain in place for both generators.


Commands Run


python3 scripts/generate_topic_filter_maps.py --help

python3 scripts/generate_skill_maps.py --help

python3 -m pytest tests/test_generator_cli_safety.py failed because system Python lacks pytest

.venv/bin/python -m pytest tests/test_generator_cli_safety.py

.venv/bin/python -m pytest

git diff --check


Validation


Targeted tests: 6 passed

Full suite: 425 passed, 3 skipped

Whitespace check: passed


Risks / Concerns


The new mutation test uses git status, so it assumes tests run inside a Git checkout. That is intentional for proving tracked-file safety.


Suggested Next Steps


Apply the same inspection-safety pattern to any other generator scripts before broader cleanup.

Goal 3: add current-output integrity audit/test
Files Changed


src/exam_bank/audit.py: added integrity audit, counts, failure groups, and documented allow-list for 9709_2025_November_33.

src/exam_bank/cli.py: added output-integrity-audit.

tests/test_audit.py: added pass/fail integrity tests.

tests/test_runtime_paths.py: updated CLI surface assertion.

README.md: documented the new audit command and known exception.


Validation also generated ignored report output at output/json/audit.current.integrity.json.

Commands Run


.venv/bin/python -m py_compile src/exam_bank/audit.py

.venv/bin/python -m pytest tests/test_audit.py tests/test_runtime_paths.py -q -> 13 passed

.venv/bin/python -m exam_bank.cli output-integrity-audit --input output/json/question_bank.json --artifact-root output --output output/json/audit.current.integrity.json -> passed

.venv/bin/python -m pytest tests/test_output_contract.py tests/test_asterion_export.py tests/test_output_management.py tests/test_question_bank_readiness_audit.py -q -> 28 passed

.venv/bin/python -m pytest -q -> 427 passed, 3 skipped

git diff --check -> passed


Note: initial python -m py_compile ... failed because there is no python shim; validation used .venv/bin/python.

Current Integrity Counts


Records: 1301; declared count matches.

Unique question_id: 1301; duplicate IDs: 0.

Duplicate (paper, question_number) pairs: 0.

Question image path entries checked: 3903; missing paths: 0; absolute paths: 0; missing files: 0.

Records with nonblank mark-scheme image paths: 1290.

Mark-scheme path entries checked: 2580; absolute paths: 0; missing files: 0.

Missing mark-scheme image paths: 11, all allowed for 9709_2025_November_33 / 33autumn25_q01 through 33autumn25_q11.

Unexpected missing mark schemes: 0.


Risks / Concerns


The audit verifies path integrity and duplicate identity constraints, not visual crop correctness.

The known missing mark-scheme companion remains an explicit allow-list item until the source PDF is added.


Suggested Next Steps


Add the missing 9709_2025_November_33 source mark scheme when available, rerun the export and integrity audit, then remove the allow-list entry.

Use output-integrity-audit as a required pre-cleanup gate before deleting or reorganizing generated artifacts.

Goal 4: document Asterion export contract
Files Changed


Added docs/ASTERION_EXPORT_CONTRACT.md (line 1)

Linked it from README.md (line 417)


Validation


No docs-specific check found in pyproject.toml or CI config.

git diff --check passed.

.venv/bin/python -m pytest tests/test_asterion_export.py tests/test_output_contract.py -q -> 14 passed

.venv/bin/python -m pytest -q -> 427 passed, 3 skipped


Key Contract Decisions


Documented both Asterion exports as role-gated projections, not globally student-facing-safe datasets.

Defined canonical fields around schema/provenance, image artifacts, integrity, quality_gate, and role permissions.

Defined text, detected values, subpart marks, mark events, and candidate planning fields as advisory unless role gates permit use.

Explicitly required exact handling of allow, block, block_until_reviewed, include, and exclude.

Reaffirmed canonical question and mark-scheme images as the source of truth.


Risks / Concerns


Student-facing eligibility remains limited.

Subpart marks remain incomplete.

9709_2025_November_33 still lacks a source mark scheme, leaving 11 records blocked/review-only.


Suggested Next Steps


Add JSON Schema or snapshot contract tests for the two Asterion export files.

Add the missing 9709_2025_November_33 mark scheme when available, then regenerate and validate exports.

Document the topic-routing sidecar contract next, since strict filters depend on sidecar safety metadata.

Goal 5: document topic routing sidecar contract
Files Changed


Added docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md (line 1)

Updated README.md (line 131), AI_ASSISTED_ENRICHMENT.md (line 18), and ASTERION_EXPORT_CONTRACT.md (line 10)


No topic routing logic, sidecar data, allowed topic IDs, or canonical records were changed.

Validation Run


.venv/bin/python -m pytest tests/test_topic_routing.py -q -> 19 passed

.venv/bin/python -m pytest -q -> 427 passed, 3 skipped

git diff --check -> passed


Key Contract Decisions


question_bank.topic_routing.v1.json is advisory unless metadata.run_summary.safe_for_strict_filters=true.

Missing or false safe_for_strict_filters means default deny for strict topic filtering.

Current sidecar is documented as unsafe: 153 schema-validation failures, safe_for_strict_filters=false.

Strict filtering requires both the sidecar-level gate and record-level eligibility: no error, review_required=false, high/medium confidence, string primary_topic_id, and non-empty valid distribution.

AI routing may choose only canonical parent topic IDs from exam_bank_taxonomy/canonical/; it must not invent curriculum/topic structure.


Risks / Concerns


This is documentation-only. Downstream consumers still need code-level enforcement of the hard gate.

Current strict topic filters should remain blocked until the sidecar is fixed and regenerated with safe_for_strict_filters=true.


Suggested Next Steps


Add downstream/Asterion consumer tests that fail closed when safe_for_strict_filters is missing or false.

Investigate and fix the schema-validation failures, then regenerate and audit the sidecar when ready.

Goal 6: create archive manifest for generated cleanup archive
Created the archive manifest at generated_cleanup_20260513T233456Z.md.

Files Changed


Added docs/history/archive_manifests/generated_cleanup_20260513T233456Z.md


Archive Classifications


baseline: none found; inventory found 0 frozen baselines.

historical evidence: archived OCR candidate JSONs, Asterion exports, audit snapshot, DeepSeek/AI sidecars.

disposable run evidence: AI/topic-routing smoke outputs, batch caches, and failure JSONL files, but retained until reviewed because they explain prior failures.

duplicated generated asset: most archived OCR candidate PNGs.

unknown: output/json/status.current.json.

keep-until-reviewed: full output_ocr_candidate/ tree and 21 archive-only p3 PNGs.


Commands Run


.venv/bin/python -m exam_bank.cli output-inventory --root output/archive/generated_cleanup_20260513T233456Z --include-size --max-depth 4

.venv/bin/python -m exam_bank.cli output-cleanup-plan --root output/archive/generated_cleanup_20260513T233456Z --include-size --max-depth 4

Supporting find, du, jq, wc, comm, and shasum inventory/checksum checks.

git status --short --untracked-files=all


Validation


Archive still contains 2,662 files.

Built-in cleanup plan was dry-run only.

Cleanup plan classified output_ocr_candidate as keep: latest candidate.

Cleanup plan classified archived output as unknown/manual review.

Git status shows only the new manifest doc:

?? docs/history/archive_manifests/generated_cleanup_20260513T233456Z.md


Risks / Concerns


21 archived p3 PNGs are not present in current output/; these need review before any deletion.

AI/API sidecars and failure files are not exactly reproducible.

status.current.json has unclear unique value and should stay until reviewed.


Suggested Next Steps


Review the 21 archive-only p3 artifacts.

Decide which AI/API failure and batch artifacts can be summarized elsewhere before cleanup.

Use this manifest as input for the next cleanup recommendation pass.

Goal 7: update README current-state references without over-documenting counts
Updated README.md.

Files Changed


README.md only


Commands Run


.venv/bin/python -m exam_bank.cli --help

.venv/bin/python -m pytest tests/test_runtime_paths.py tests/test_generator_cli_safety.py -q

rg checks for stale README OCR/count references

rg --files discovery for docs/markdown check tooling


Validation Results


Focused CLI-surface tests passed: 10 passed in 0.65s

No dedicated docs/markdown check tool was found in the repo.

Full pytest was not run because this was README-only and examples are not directly README-tested.


Stale References Fixed


Replaced old native-only/no-OCR current-state wording with the audited OCR-enabled state.

Removed duplicated measured count blocks and pointed to docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md as the current baseline.

Added a compact command atlas for standard/OCR runs, audits, Asterion export, Content Lab export, topic routing, AI enrichment, and tests.

Re-centered canonical images as source of truth; text/OCR/topic/AI/Asterion sidecars are advisory unless role-gated.

Removed stale full-suite result/count snapshots from README.


Risks / Concerns


Other docs remain stale per the audit; I did not touch them per scope.

README now intentionally depends on the audit report for measured current-state counts, so that report needs to stay the baseline source.


Suggested Next Steps


Update the other stale docs called out by the audit.

Add a lightweight docs lint or README command-snippet test if command drift keeps recurring.

Phase 1 closeout audit is done.

Files changed/created:


Created docs/PHASE_1_CLEANUP_PREREQ_CLOSEOUT.md

Updated tests/test_audit.py to explicitly test record-count mismatch and absolute mark-scheme paths.

Updated tests/test_repo_hygiene.py to match README’s current multi-line .venv/bin/python command format.


Validation run:


All required git status/diff/diff-check commands.

Full suite: .venv/bin/python -m pytest -q -> 427 passed, 3 skipped.

Focused tests:

tests/test_generator_cli_safety.py -> 6 passed

tests/test_audit.py tests/test_runtime_paths.py -> 13 passed




Both generator --help commands passed.

Both generator --dry-run commands passed with validation pass.

Output integrity audit passed with ok: true.

Archive inventory and cleanup-plan commands passed; cleanup plan remained dry-run only.


Generated files:


No unexpected generated-output mutation occurred.

Hashes stayed unchanged for current question bank, topic sidecar, Asterion exports, canonical taxonomy, top-level output JSON/JSONL, archive, and canonical image tree.

output-integrity-audit intentionally wrote output/json/audit.current.integrity.json, but content hash stayed unchanged.


Final readiness verdict:


READY_FOR_PHASE_2_WITH_NOTES


Main risks/concerns:


9709_2025_November_33 still has 11 allowed missing mark-scheme paths.

Current topic sidecar is not strict-filter safe: safe_for_strict_filters=false, 153 failed records.

Asterion/topic contracts are conservative but mostly documentation-only for downstream consumers.

21 archive-only p3 PNGs must be reviewed before any deletion.

Integrity audit checks minimum identity/path invariants only, not visual/OCR/topic correctness.


Suggested next steps:


Start Phase 2 with documentation/archive planning, not deletion.

Review the 21 archive-only p3 files.

Add historical banners or move stale docs before layout cleanup.

Keep current outputs, exports, taxonomy, topic sidecar, and canonical images fixed until an intentional audited regeneration/change plan exists.


Phase 2 goals: cleanup and reorganization

Goal 8: move or label historical docs
Implemented the docs-only separation.

Docs changed/moved


Moved docs/PROJECT_REVIEW.md to docs/history/PROJECT_REVIEW.md (line 1) and added a historical banner.

Updated ROADMAP.md (line 5) to point current measured state to the audit baseline and mark old iteration counts as historical.

Updated docs/TRUST_MODEL.md (line 184) to replace stale tier counts with audit/refresh commands.

Updated docs/AUTO_TRIAGE.md (line 27) to remove stale current evidence counts and treat output_ocr_candidate/ as legacy evidence.

Updated docs/TRIAGE_WORKFLOW.md (line 17) to make examples generic/current-path based.

Added a post-audit note in docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md (line 9) explaining the project review move.


Marked historical


PROJECT_REVIEW.md is now explicitly historical. Its old measured “current” wording is preserved as evidence, but no longer lives among active docs.


Validation


No dedicated docs/link checker was found.

Ran git diff --check / git show --check: no whitespace errors.

Ran targeted rg checks for stale active-doc phrases and markdown links to the old PROJECT_REVIEW.md path: no active matches.

Did not run pytest because this was docs-only and no imports, code paths, or tests changed.


Risks / Concerns


The moved historical review still contains old “current” wording by design; the banner is the guardrail.

The audit report preserves original path references as audit-time findings, with a top note clarifying the move.


Suggested Next Steps


Add a lightweight markdown/link check if docs drift keeps recurring.

Continue the planned final docs consistency pass after command atlas and validation checklist work.


Goal 9: normalize command documentation
Implemented the command atlas as documentation-only work. No command behavior changed and no outputs were regenerated.

Files Changed


docs/COMMAND_ATLAS.md: new operator command map with purpose, input, output, runtime, and category.

README.md: replaced duplicate inline commands with a pointer to the atlas; removed active output_ocr_candidate examples.

docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md: marked stale findings as resolved/historical and pointed command tables to the atlas.


Commands Verified


.venv/bin/python -m exam_bank.cli --help

process, audit, output-integrity-audit

asterion-export, asterion-content-lab-candidates

topic-route-ai, enrich-ai, ai-sidecar-audit

output-inventory, output-cleanup-plan

scripts/audit_ocr_candidates.py --help

scripts/audit_difficulty.py --help

scripts/audit_question_bank_readiness.py --help

scripts/generate_skill_maps.py --help

scripts/generate_topic_filter_maps.py --help

.venv/bin/python -m pytest --help

git diff --check


Stale References Updated


Removed active README output_ocr_candidate inventory/cleanup examples.

Replaced stale audit-table commands using invalid flags like --output-dir.

Removed references to nonexistent audit scripts from the active command table.

Kept output_ocr_candidate only where it is explicitly historical/archive evidence.


Risks / Concerns


Full pytest was not run because only docs changed.

Several atlas commands are intentionally mutating if executed, so I verified help surfaces rather than running output-generating workflows.


Suggested Next Steps


Add a lightweight docs smoke test later if you want command snippets checked automatically against CLI help.


Goal 10: classify archived generated artifacts, no deletion

Converted the Phase 1 archive manifest into actionable cleanup recommendations without deleting, moving, compressing, regenerating, or rewriting archive contents.

Files Changed


Updated docs/history/archive_manifests/generated_cleanup_20260513T233456Z.md

Updated docs/GOALS.md


Classifications


Keep formal evidence: archived OCR candidate question bank, archived Asterion exports, audit snapshot, legacy DeepSeek sidecar, post-fix clean AI smoke sidecar, and topic-routing smoke sidecars.

Keep until reviewed: status.current.json and the 21 archive-only p3 PNGs under 33autumn25 and 34autumn25.

Compress or keep until reviewed: mixed/stale AI full sidecar, because it explains the prior run but is not a formal export baseline.

Regenerate on demand: Asterion projections and duplicated PNG image trees, but only from retained source/candidate evidence and only after checksum gates pass.

Delete later: duplicated p1, p4, p5 image trees; 781 duplicated p3 images; superseded AI smoke/full attempts once summarized; batch caches and failure JSONL files once parent sidecars/failure summaries are retained.


Retention Recommendation


Preserve the full archive until the 21 p3 exceptions are resolved.

Prefer a compressed formal-evidence bundle over keeping expanded duplicate image trees indefinitely.

Treat AI/API sidecars as evidence snapshots, not byte-for-byte reproducible build outputs.

Before any deletion pass, rerun inventory, cleanup plan, duplicate checksum checks, and archive hashes.


Commands Run


.venv/bin/python -m exam_bank.cli output-inventory --root output/archive/generated_cleanup_20260513T233456Z --include-size --max-depth 4

.venv/bin/python -m exam_bank.cli output-cleanup-plan --root output/archive/generated_cleanup_20260513T233456Z --include-size --max-depth 4

Supporting find, du, jq, wc, and checksum/hash checks.


Risks / Concerns


The 21 archive-only p3 PNGs may be current-output gaps, old naming artifacts, or disposable candidates; they remain non-deletion items.

status.current.json still needs review before it can be called disposable.

AI/API sidecars are not exactly reproducible, so deletion should preserve enough summary evidence to explain prior failures and safety gates.


Suggested Next Steps


Review the 21 archive-only p3 artifacts against source documents and current-output expectations.

Summarize failure JSONL and batch-cache evidence before deleting those disposable run artifacts.

Use the updated manifest as the input for Goal 11's first safe generated-output cleanup.

Goal 11: perform first safe generated-output cleanup
Implemented the first conservative cleanup pass.

Removed


7 archived *.batches/ directories under output/archive/generated_cleanup_20260513T233456Z/output/json/

5 archived *.failures.jsonl files under the same archive path

No current question bank, current image tree, current Asterion export, unknown archive file, tracked generated file, or archived image tree was removed.


Changed


docs/history/archive_manifests/generated_cleanup_20260513T233456Z.md: recorded failure summaries, cache summaries, removed paths, before/after archive size, hashes, and validation notes.

README.md: restored the supported process --input input --output output snippet required by repo hygiene tests.

docs/GOALS.md was already modified before I started and I left that existing change untouched.


Size


Archive: 481,988 KiB before -> 473,764 KiB after.

Archive file count: 2,662 -> 2,626.

Output root: 945,536 KiB before cleanup pass -> 937,328 KiB after validation artifacts were written.


Validation


output-inventory before and after: passed.

output-cleanup-plan after: passed; current canonical outputs remain keep: canonical/current, archive remains unknown/manual review.

output-integrity-audit: passed with ok: true, 1,301 records, only the known 11 missing mark-scheme companions for 9709_2025_November_33.

pytest -q: initially found a README hygiene drift; after the README fix, full suite passed: 427 passed, 3 skipped.


Commands Run


.venv/bin/python -m exam_bank.cli output-inventory ...

.venv/bin/python -m exam_bank.cli output-cleanup-plan ...

.venv/bin/python -m exam_bank.cli output-integrity-audit ...

.venv/bin/python -m pytest -q

Supporting du, find, jq, wc, git check-ignore, git ls-files, and archive hash commands.


Risks / Concerns


output/archive still has manual-review material and should remain protected.

The 21 archive-only p3 PNG exceptions remain unresolved and were preserved.

AI sidecar JSON snapshots remain evidence; only raw cache/failure files were removed after summaries were retained.


Next Steps


Review the 21 archive-only p3 PNGs against source/current-output expectations.

Decide whether status.current.json has unique diagnostic value.

Only after that, consider duplicate archived PNG cleanup with a fresh checksum comparison.

Goal 12: update roadmap based on audit phases
Updated ROADMAP.md (line 24) only.

Files Changed


ROADMAP.md


Roadmap Decisions


Added an audit-backed cleanup/optimization section.

Marked Phase 1 as cleanup prerequisites.

Marked Phase 2 as cleanup/reorganization.

Marked Phase 3 as low-risk optimization.

Marked Phase 4 as deferred future deeper refactors, not current work.


Deferred Items


Exam-report and grade-boundary leverage for topic/difficulty is noted as future work only.

No topic/difficulty algorithms, generated outputs, or enrichment logic were changed.


Risks/Concerns


Phase 4 work has larger behavioral and migration risk, so it now explicitly waits until Phases 1-3 and acceptance gates are complete.

Future enrichment must stay advisory until separately audited.


Validation


Docs-only change; no tests run.


Suggested Next Steps


Continue Phase 2/3 documentation and low-risk operational cleanup.

Keep any future exam-report or grade-boundary work behind a separate audited proposal.


Phase 3 goals: low-risk optimization

Goal 13: add atomic JSON writes for exports
Implemented atomic JSON writes for the requested export surfaces.

Files Changed

src/exam_bank/atomic_json.py
src/exam_bank/exporters.py
src/exam_bank/asterion_export.py
src/exam_bank/topic_routing.py
src/exam_bank/deepseek_enrich.py
tests/test_atomic_json.py
Note: docs/GOALS.md is also modified in the worktree, but I did not touch it.

Behavior Changes

Main question bank exports now write through a shared atomic JSON helper.
Asterion exports and content-lab exports now write atomically.
Topic routing sidecar writes now use the shared helper instead of a local temp-file pattern.
DeepSeek sidecar, AI-assisted sidecar, and AI batch cache writes now use the shared helper.
Temp files are created in the target directory as hidden .tmp files, flushed/fsynced, then replaced with os.replace.
Temp files do not end in .json, so normal JSON output discovery should not treat them as complete outputs.
JSON payloads, schemas, field names, paths, and formatting intent were preserved.
Commands Run

pytest ... failed: pytest not on PATH.
python -m pytest ... failed: python not on PATH.
python3 -m pytest ... failed: global Python lacks pytest.
.venv/bin/python -m pytest tests/test_atomic_json.py tests/test_output_contract.py tests/test_asterion_export.py tests/test_topic_routing.py tests/test_deepseek_enrich.py
.venv/bin/python -m pytest
Validation Results

Targeted tests: 94 passed
Full suite: 430 passed, 3 skipped
Risks / Concerns

Atomicity depends on the target filesystem supporting same-directory os.replace, which is the expected local behavior.
Interrupted processes can still leave .tmp files behind, but those are intentionally not valid output names.
Suggested Next Steps

Optionally extend the same helper to non-requested JSON report/debug writers later, such as audit or triage outputs, if you want all generated JSON artifacts covered uniformly.

Goal 14: improve run-status terminal output for standard runs
Implemented progress/status reporting improvements for standard non-AI runs.

Files Changed


src/exam_bank/run_status.py: added richer status metadata and terminal progress output.

src/exam_bank/pipeline.py: passes paper/session/component/question context into run-status updates.

tests/test_run_status.py: added coverage for new metadata and terminal progress text.


Validation


pytest failed: command not found.

python -m pytest failed: command not found.

python3 -m pytest tests/test_run_status.py failed: pytest not installed for system Python.

.venv/bin/python -m pytest tests/test_run_status.py: 4 passed.

.venv/bin/python -m pytest tests/test_sample_pipeline.py: 32 passed, 3 skipped.

.venv/bin/python -m pytest: 431 passed, 3 skipped.


User-Visible Progress Changes


Progress line now includes elapsed time, run ID, output path, current stage, current paper, session, component, papers completed/total, records completed/total when known, and current question index/id when available.

run_status.json now includes current_stage, current_session, current_component, current_record_id, current_record_index, total_current_records, completed_papers, total_papers, and percent_complete_basis.

Percentage basis is explicit: records when a reliable total exists, otherwise papers.


Risks / Concerns


No extraction, OCR selection, or output schema changes were made.

run_status.json schema is expanded, not replaced; existing status files/tests are preserved.

Per-question updates overwrite the status file but terminal rendering remains rate-limited unless forced.


Suggested Next Steps


Run one real process command with --progress on a small input folder to visually confirm the terminal line is comfortable during an actual operator run.

Goal 15: improve run-status output for AI-heavy runs

**Start here!**
Goal 16: add export summary diffing

Goal: add export summary diffing so operators can see readiness changes between runs.
Context:
Cleanup and optimization should not silently change Asterion eligibility, text readiness, topic safety, or record counts. Add a low-risk summary diff tool or CLI option.
Scope:
Support comparison between two generated exports or summaries, likely:
- question_bank.json to question_bank.json
- Asterion export to Asterion export
- topic routing sidecar to topic routing sidecar, if straightforward
Tasks:
1. Add a small CLI command or script for summary diffs.
2. Compare record count, schema version, generated timestamp/run ID, readiness counts, role-gate counts, missing image counts, topic safety metadata, and major reason-code counts.
3. Print a concise before/after diff.
4. Return nonzero only for clearly invalid comparisons, not normal count changes.
5. Add tests with small fixtures.
Do not:
- Change export generation.
- Change readiness logic.
- Require full generated outputs in tests.
Validation:
Run targeted tests.
Run the diff on current exports if there is a previous comparable export available.
Run full pytest if practical.
Final summary required:
List files changed, commands run, validation results, example diff output, risks/concerns, and suggested next steps.

Goal 17: split test suite into fast and integration groups

Goal: add test markers or organization so fast tests and integration/rendering tests can be run separately while preserving current full-test behavior.
Context:
The full suite is healthy but takes around 2 minutes. This is acceptable, but optimization work would benefit from a fast local loop.
Scope:
Review test structure and pytest configuration.
Tasks:
1. Identify slower integration/rendering/sample-pipeline tests.
2. Add pytest markers or documented command groups.
3. Preserve `python -m pytest` or `python -m pytest -q` as the full suite.
4. Add README/command atlas examples for fast and full validation.
5. Avoid weakening CI coverage.
Do not:
- Skip important tests by default in CI.
- Remove regression coverage.
- Rewrite large test fixtures unnecessarily.
Validation:
Run fast test command.
Run full pytest.
Confirm CI command remains full coverage.
Final summary required:
List files changed, commands run, fast/full test timings if available, risks/concerns, and suggested next steps.

Goal 18: update stale script default paths

Goal: update stale script default paths so operational scripts point to the current generated layout.
Context:
The audit found scripts that still expect old paths such as output_ocr_candidate/json/question_bank.json. These should either use current defaults or require explicit input paths.
Scope:
Review scripts under scripts/ for stale default paths.
Tasks:
1. Search for references to old generated paths.
2. For active scripts, update defaults to current paths such as output/json/question_bank.json where appropriate.
3. For scripts where a default is unsafe, require explicit --input.
4. Preserve historical docs by marking old paths as historical, not current.
5. Add or update tests for argument parsing if practical.
Do not:
- Change script core behavior beyond path/default safety.
- Regenerate outputs.
- Delete old archive files.
Validation:
Run --help for affected scripts.
Run targeted script tests.
Run full pytest if practical.
Final summary required:
List files changed, stale paths fixed, commands run, validation results, risks/concerns, and suggested next steps.

Goal 19: create release validation checklist command or doc

Goal: create a release validation checklist for producing a clean current export.
Context:
After cleanup and low-risk optimization, the project needs a repeatable way to validate a release-quality question bank and Asterion export.
Scope:
Create either a docs checklist or a lightweight CLI/script if the existing code makes that easy.
Tasks:
1. Define the release validation sequence:
   - run tests
   - run question-bank audit
   - run image integrity check
   - run OCR candidate audit
   - generate or validate Asterion exports
   - validate topic sidecar safety metadata
   - run output inventory/cleanup-plan
2. Include expected output paths.
3. Include what counts as blocking versus warning.
4. Include current known exception: missing mark scheme for 9709_2025_November_33 unless resolved.
5. Keep this as a checklist first if a CLI wrapper would be too much.
Do not:
- Change generated outputs.
- Add hard gates that break current workflows without discussion.
- Implement topic/difficulty exam-report logic.
Validation:
Run any documented commands that are safe and fast.
Run pytest if code changed.
Final summary required:
List files changed, checklist decisions, commands run, validation results, risks/concerns, and suggested next steps.

Goal 20: README and docs final consistency pass for Phases 1–3

Goal: perform a final documentation consistency pass after Phase 1 through Phase 3 work.
Context:
After script safety, output integrity, archive classification, command atlas, atomic writes, status reporting, and validation checklist updates, the docs should agree with the current operating model.
Scope:
Review README, ROADMAP, command atlas, audit doc, Asterion contract doc, topic sidecar contract doc, and cleanup docs.
Tasks:
1. Remove contradictions about OCR state.
2. Remove stale current-state counts or label them with run IDs and dates.
3. Ensure all commands use current paths.
4. Ensure Asterion role gates are described consistently.
5. Ensure strict topic routing safety is described consistently.
6. Ensure cleanup/optimization phases are clear.
7. Add a note that future topic/difficulty leverage may later use exam reports and grade boundaries, but that this is deferred until after deeper refactors.
Do not:
- Rewrite docs unnecessarily.
- Add new implementation scope.
- Change code unless fixing broken command references in tests/docs.
Validation:
Run docs checks if available.
Run pytest if any code/test files changed.
Final summary required:
List files changed, contradictions fixed, validation run, remaining docs risks, and suggested next steps.

I’d run them in this order: Goals 1–7 for Phase 1, then 8–12 for Phase 2, then 13–20 for Phase 3. The future exam reports and grade boundaries idea is a good one, but I’d keep it parked as a roadmap note until the project is cleaner, safer, and easier to operate.
