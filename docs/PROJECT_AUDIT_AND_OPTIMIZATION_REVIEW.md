# Project Audit and Optimization Review

Audit date: 2026-05-14

Repository: `exam-bank-pipeline`

Scope: full project audit before cleanup or optimization. This report is intentionally documentation-only. It does not recommend treating advisory text, OCR, AI enrichment, or topic routing as canonical unless deterministic gates explicitly allow that use.

Post-audit documentation cleanup note: `docs/PROJECT_REVIEW.md` has been moved to `docs/history/PROJECT_REVIEW.md` and marked historical. Path references below describe the file location at audit time unless they explicitly refer to the new history path.

Post-Phase 1 through Phase 3 consistency note: this audit remains the dated measured baseline for run `20260513T070200Z-56d469c1dd52`, not a live count source. Later docs now carry the current command atlas, Asterion contract, topic sidecar contract, archive manifest, atomic-write/status-reporting notes, and validation checklist. Where this report says a doc or command "needs" one of those items, read that as original audit context unless the same risk is repeated in a current contract or checklist.

## 1. Executive Summary

The project has a strong engineering foundation for an image-first CAIE 9709 exam-bank pipeline. The core source-of-truth policy is reflected in code, tests, and export shape: canonical question and mark-scheme images are preserved separately from advisory native text, OCR text, AI sidecars, topic routing, and Asterion eligibility metadata.

The current canonical question bank is `output/json/question_bank.json`, schema `exam_bank.question_bank` version 2. It contains 1301 records generated on `2026-05-13T07:02:00.448742+00:00` from run `20260513T070200Z-56d469c1dd52`. All 1301 question images exist. Nonblank mark-scheme image paths also exist, but 11 records have no mark-scheme image path because the source input is missing the companion mark scheme for `9709_2025_November_33`.

At the time of this audit, the project was operational and well tested. A full test run passed: 419 passed, 3 skipped in 110.63 seconds. Later Phase 1 closeout validation recorded a newer full-suite result. Treat this line as dated audit evidence, not the current test count. The tests cover extraction, OCR selection, trust gates, output contracts, Asterion exports, topic routing, run status, hygiene, and several regression cases.

The main risks before cleanup are operational and semantic rather than architectural:

- Some generated outputs and documentation disagreed about OCR state and readiness counts at audit time. Current README and roadmap docs now state that the canonical export is OCR-enabled and avoid duplicating unlabeled count snapshots.
- Output artifacts are large and include both current and archived generated trees.
- Some generator scripts are not safe when invoked with `--help`; one taxonomy generator executed and rewrote generated taxonomy files during inspection before its changes were reverted.
- Asterion-facing exports are structurally useful, but only a limited subset is student-facing safe today.
- Topic routing is strict about allowed topic IDs, but the current routing sidecar has 153 schema-validation failures and is marked `safe_for_strict_filters=false`.
- Mark-scheme display is mostly usable, but subpart mark breakdowns are not populated and 3 records have question/mark-scheme total mismatches.
- Documentation was stale in several places at audit time, especially around OCR activation, current counts, archived output names, and hard blocker totals. Current operator docs should prefer `README.md`, `ROADMAP.md`, `docs/COMMAND_ATLAS.md`, `docs/ASTERION_EXPORT_CONTRACT.md`, `docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md`, and `docs/RELEASE_VALIDATION_CHECKLIST.md`.

Overall grade: B+. The core architecture and regression protection are strong. The cleanup readiness grade is closer to B- because generated artifact hygiene, documentation freshness, and command safety need work before broad reorganization.

## 2. Current Project State

The repository is a Python package with CLI entry points under `src/exam_bank`, tests under `tests`, operational scripts under `scripts`, input PDFs under `input`, generated outputs under `output`, and taxonomy/supporting assets under `exam_bank_taxonomy`.

Current important facts as of audit date `2026-05-14`, source run `20260513T070200Z-56d469c1dd52`:

| Area | State |
| --- | --- |
| Git status before audit | Clean |
| Python package | `exam-bank-pipeline`, Python `>=3.10` |
| Main CLI module | `exam_bank.cli` |
| Main config | `config.yaml` |
| Current canonical export | `output/json/question_bank.json` |
| Current Asterion export | `output/asterion/exports/latest/asterion_question_bank_v1.json` |
| Current Content Lab export | `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json` |
| Current topic routing sidecar | `output/json/question_bank.topic_routing.v1.json` |
| Current question count | 1301 |
| OCR in current export | Ran for all 1301 records |
| OCR selected over native | 27 records |
| Question images missing | 0 |
| Mark-scheme image paths missing | 11 |
| Full test suite | 419 passed, 3 skipped |

Top-level size profile:

| Path | Approx size | Classification |
| --- | ---: | --- |
| `output/` | 924M | Generated, important current plus archived generated artifacts |
| `input/` | 99M | Active source PDFs |
| `exam_bank_taxonomy/` | 44M | Generated/supporting taxonomy assets, important but needs ownership clarity |
| `src/` | 2.6M | Active source code |
| `tests/` | 2.4M | Active regression protection |
| `scripts/` | 840K | Active and mixed operational scripts |
| `agent_handoffs/` | 432K | Historical evidence/handoffs, preserve until reviewed |
| `docs/` | 68K before this report | Active and historical documentation |

Current generated output state at audit time:

- `output/json/question_bank.json`: current canonical question bank.
- `output/json/question_bank.topic_routing.v1.json`: current topic routing sidecar.
- `output/asterion/exports/latest/*`: current Asterion-facing projections.
- `output/p1`, `output/p3`, `output/p4`, `output/p5`: current generated image trees.
- `output/archive/generated_cleanup_20260513T233456Z`: archived generated sidecars and previous OCR candidate artifacts.

The IDE tabs mention some files such as `output/json/audit.current.json` and older clean-smoke AI files. Those are not present in current `output/json`; matching artifacts are now under `output/archive/generated_cleanup_20260513T233456Z`.

## 3. What Is Working Well

The image-first source-of-truth model is consistently represented in the main data model and exports. `QuestionRecord` separates canonical image fields from native/OCR text candidates, trust fields, classification metadata, topic metadata, validation flags, and mark-scheme metadata.

The main pipeline has clear staged responsibilities:

- registry construction from `input/`
- PDF layout extraction
- question span detection
- mark-scheme extraction and image stitching
- question crop rendering
- native text extraction
- optional OCR candidate generation
- OCR/native selection
- local classification and difficulty inference
- conservative trust status derivation
- JSON export with manifest and QA summary

The project has useful safety layers:

- Native text and OCR text are advisory by design.
- OCR is selected only when it clears structural checks and score margins.
- Text-only readiness remains conservative.
- Asterion exports preserve separate usage roles instead of one broad "safe" flag.
- Topic routing validates against allowed paper-family topic IDs.
- Tests cover many previous failure classes, including mark-scheme pairing and output contract shape.

The current image artifact integrity is strong:

- All 1301 question image paths exist.
- All nonblank mark-scheme image paths exist.
- There are no absolute image paths in the current question bank.
- Question IDs and paper/question path patterns are structurally consistent.
- No duplicate `question_id` values were found.
- No duplicate `(paper, question_number)` pairs were found.

The run-status system already has useful primitives for long runs: stage tracking, progress bars, status files, run IDs, ETA-like metadata, and batch cache paths. That should be preserved and expanded rather than replaced.

## 4. Major Risks

### Documentation Drift

Several docs described a previous no-OCR canonical state at audit time. The current `question_bank.json` for run `20260513T070200Z-56d469c1dd52` has `ocr_ran=true` for all 1301 records and 27 OCR-selected records. README and ROADMAP have since been reconciled; historical workflow docs may still contain dated evidence and should not be used as current state unless they name the same run/date.

Risk: operators may run the wrong workflow, trust stale readiness numbers, or regenerate from old assumptions.

### Generated Artifact Sprawl

`output/` is 924M and contains current generated trees plus archived generated runs. The cleanup planner already identifies `output/archive` as manual-review/unknown. Current `.gitignore` excludes generated output except `.gitkeep`, which is good, but local workspace hygiene still needs an archive/retention policy.

Risk: cleanup may delete evidence too early, or future runs may accidentally compare against stale archived sidecars.

### Script Command Safety

During the original audit, a help-inspection loop invoked `scripts/generate_topic_filter_maps.py --help`. At that time the script did not expose argparse help and executed its generation behavior, rewriting 14 taxonomy files before the changes were reverted. `scripts/generate_skill_maps.py --help` also failed because it expected `output_ocr_candidate/json/question_bank.json`. Phase 1 safety work later added argparse help/dry-run behavior and moved the generator defaults to current `output/...` paths.

Risk: simple inspection commands can mutate generated assets or fail against old paths.

### Topic Routing Sidecar Not Fully Safe

The current routing sidecar has:

- 1301 records
- 1148 successful records
- 153 schema-validation failures
- 221 review-required records
- `safe_for_strict_filters=false`

Risk: downstream consumers might treat the sidecar as safe for strict filters without checking the metadata.

### Asterion Student-Facing Readiness Is Limited

Audit-measured Asterion export roles show only 252 records allowed for canonical practice and Guardian candidate use. Quick-check source allows 51 records. Most records are blocked or blocked until reviewed because of visual requirements, crop confidence, topic uncertainty, mark-scheme confidence, or text-only trust limits.

Risk: Asterion can consume the projection, but it must respect role gates and must not treat the full file as student-facing safe.

### Mark-Scheme Structure Is Display-Ready, Not Marking-Ready

All current records have `question_solution_marks`, but all 968 records with subparts have null subpart solution marks. Three records have mismatched question-vs-mark-scheme totals:

- `51summer23_q04`: question 5, mark scheme 9
- `51summer24_q02`: question 3, mark scheme 7
- `32autumn25_q10`: question 4, mark scheme 9

Risk: mark-scheme images are useful for display, but not yet safe for automated self-marking, method/accuracy parsing, worked examples, or Guardian mark checks.

### Atomicity and Resumability Gaps

The exporter writes JSON files directly. Long runs already have run-status support, but generated JSON writes should be atomic to avoid partial files if a run is interrupted.

Risk: long full-bank or AI-heavy runs can leave ambiguous outputs after interruption.

## 5. Repository Structure Review

### Active Current

| Path | Role |
| --- | --- |
| `src/exam_bank/` | Active package source |
| `tests/` | Active test suite and regression protection |
| `config.yaml` | Active default pipeline configuration |
| `pyproject.toml` | Active package/dependency definition |
| `.github/workflows/tests.yml` | Active CI test workflow |
| `input/question_papers/` | Active source question PDFs |
| `input/mark_schemes/` | Active source mark-scheme PDFs, with one missing companion |
| `output/json/question_bank.json` | Current generated canonical JSON |
| `output/json/question_bank.topic_routing.v1.json` | Current generated topic routing sidecar |
| `output/asterion/exports/latest/` | Current generated Asterion projections |
| `output/p1`, `output/p3`, `output/p4`, `output/p5` | Current generated canonical image trees |

### Generated but Important

| Path | Why important |
| --- | --- |
| `output/json/question_bank.json` | Primary current export for audit and downstream projections |
| `output/asterion/exports/latest/asterion_question_bank_v1.json` | Asterion role-gated projection |
| `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json` | Content Lab candidate projection |
| `output/json/question_bank.topic_routing.v1.json` | Strict topic sidecar, not fully safe today |
| `exam_bank_taxonomy/` | Taxonomy export/support files used by skill/topic workflows |
| `output/archive/generated_cleanup_20260513T233456Z/` | Historical evidence of AI/OCR candidate runs |

### Generated and Safely Reproducible, Subject to Runtime

| Path | Notes |
| --- | --- |
| `output/p1`, `output/p3`, `output/p4`, `output/p5` | Reproducible from input PDFs, but expensive and should be preserved until regeneration is verified |
| `output/asterion/exports/latest/*` | Regenerable from current question bank and sidecars |
| `output/json/question_bank.topic_routing.v1.json` | Regenerable, but AI/API dependent and should be checkpointed |
| OCR candidate outputs | Regenerable but time-consuming; archived copies are useful evidence |

### Historical Evidence / Archive

| Path | Notes |
| --- | --- |
| `output/archive/generated_cleanup_20260513T233456Z/` | Preserve until cleanup plan records what each sidecar represented |
| `agent_handoffs/` | Likely historical implementation context; do not delete before indexing contents |
| `docs/PROJECT_REVIEW.md` | Stale as current state, useful as historical review |
| older AI sidecars in archive | Useful to explain prompt version changes, failures, and previous Asterion safety decisions |

### Stale or Obsolete

| Path / area | Reason |
| --- | --- |
| README measured state section | Resolved: current README now says the canonical export is OCR-enabled |
| `ROADMAP.md` measured state | Uses older hard-blocker and OCR activation status |
| `docs/TRUST_MODEL.md` tier counts | Counts are stale, though policy is useful |
| `docs/AUTO_TRIAGE.md` current evidence references | Resolved in active examples; legacy `output_ocr_candidate` is now marked historical/compatibility evidence |
| `docs/TRIAGE_WORKFLOW.md` examples | Refers to triage folders not present in current output |
| `docs/PROJECT_REVIEW.md` | Historical, not current |
| `scripts/generate_skill_maps.py` default input | Resolved: default input is `output/json/question_bank.json` |

### Duplicated

| Area | Duplication |
| --- | --- |
| Current and archived image trees | Archive contains generated `output_ocr_candidate/p1..p5` image trees duplicating current image volume in part |
| AI sidecars | Several full/smoke/fix sidecars preserve overlapping enrichment outputs |
| Readiness counts in docs | README, ROADMAP, TRUST_MODEL, PROJECT_REVIEW each carry measured counts that drift independently |
| Export readiness logic | Some concepts appear in trust, Asterion export, content-lab generation, and docs; keep code canonical and docs referential |

### Unclear Status

| Path / area | Question |
| --- | --- |
| `exam_bank_taxonomy/` generated files | Which files are canonical source inputs versus generated exports? |
| `agent_handoffs/` | Which handoffs remain actionable? |
| archived clean-smoke/full AI sidecars | Which are formal baselines versus temporary run evidence? |
| `output/archive` retention | How long should archived generated output remain local? |

### Candidate for Gitignore

Most generated outputs are already ignored. Confirm these remain ignored:

- `.venv/`
- `.pytest_cache/`
- `.DS_Store`
- `.env*`
- `output/*` except `output/json/.gitkeep`
- `output_ocr_candidate/`
- output inventory and cleanup reports

Additional candidates to verify after cleanup:

- `tests/__pycache__/`
- local run-status scratch files if reintroduced under ignored paths
- temporary topic/AI batch caches if generated outside ignored output paths

### Candidate for Relocation / Archive

| Path / area | Recommendation |
| --- | --- |
| stale measured docs | Move old reviews to `docs/history/` or mark as historical at top |
| archived AI/OCR sidecars | Keep under timestamped archive with manifest explaining provenance |
| generated taxonomy exports | Separate `source`, `generated`, and `reports` if mixed |
| old command references | Replace with a command atlas in README, then archive outdated workflow docs |

### Candidate for Deletion

No broad deletion is recommended in this audit pass. Later deletion candidates should be limited to artifacts that are documented as reproducible or already preserved in archive:

- stale duplicate generated sidecars after a manifest is written
- duplicate image trees after a verified regeneration and checksum comparison
- local cache/temp files under ignored paths

## 6. Active vs Stale File/Folder Map

| Path | Classification | Cleanup action |
| --- | --- | --- |
| `src/exam_bank/cli.py` | Active/current | Preserve; document commands |
| `src/exam_bank/pipeline.py` | Active/current | Preserve; add tests before refactors |
| `src/exam_bank/models.py` | Active/current | Preserve; schema source of truth |
| `src/exam_bank/exporters.py` | Active/current | Preserve; add atomic write later |
| `src/exam_bank/ocr.py` | Active/current | Preserve; cautious tuning only |
| `src/exam_bank/trust.py` | Active/current | Preserve semantics; document names |
| `src/exam_bank/asterion_export.py` | Active/current | Preserve file contracts |
| `src/exam_bank/topic_routing.py` | Active/current | Preserve strict validation; improve failure surfacing |
| `src/exam_bank/deepseek_enrich.py` | Active/current | Preserve as sidecar path; improve resumability |
| `src/exam_bank/run_status.py` | Active/current | Preserve; expand progress reporting |
| `src/exam_bank/output_management.py` | Active/current | Preserve; use as cleanup front door |
| `scripts/audit_*.py` | Active/mixed | Keep; document current inputs |
| `scripts/generate_topic_filter_maps.py` | Active but unsafe CLI | Add argparse/help/dry-run before use |
| `scripts/generate_skill_maps.py` | Active but stale default path | Update default path or require explicit input |
| `tests/` | Active/current | Preserve; add output-level guards |
| `input/` | Active/current | Preserve; add missing MS tracking |
| `output/json/question_bank.json` | Generated but important | Keep until verified regenerated |
| `output/asterion/exports/latest/` | Generated but important | Keep; document schemas |
| `output/archive/` | Historical/generated | Preserve until manifest and retention policy |
| `README.md` | Active but stale measured state | Update after cleanup prerequisites |
| `ROADMAP.md` | Active planning, stale facts | Reconcile with current audit |
| `docs/TRUST_MODEL.md` | Policy useful, counts stale | Update counts and cross-link this report |
| `docs/PROJECT_REVIEW.md` | Historical | Mark historical or move to docs/history |

## 7. Pipeline and Data-Flow Review

Current intended flow:

1. Source PDFs live in `input/question_papers` and `input/mark_schemes`.
2. `document_registry.py` parses filenames and builds paper/mark-scheme pairs.
3. `pipeline.py` loads the registry and processes question papers by family/component/session/year.
4. `pdf_extract.py` and related extraction modules derive page layout, blocks, question starts, and spans.
5. Question images are rendered from PDF pages and written under `output/<paper_family>/<paper_id>/questions/`.
6. Mark-scheme images are extracted/stitched and written under `output/<paper_family>/<paper_id>/mark_schemes/`.
7. Native text is extracted from PDF layout and normalized as advisory text.
8. If OCR is enabled, `ocr.py` produces OCR candidates and candidate metadata.
9. Native/OCR selection compares scores and structural safety checks; OCR is selected only if it is clearly better and not structurally rejected.
10. Local classification assigns coarse topics, difficulty, confidence, review flags, and metadata.
11. `trust.py` derives visual/text/topic trust statuses.
12. `exporters.py` writes schema v2 question bank JSON plus manifest/QA summary.
13. Optional AI enrichment writes sidecars, not canonical data.
14. Optional topic routing writes strict topic sidecar, not canonical curriculum.
15. Asterion export projects canonical assets, advisory text, sidecars, and role gates into consumer-facing files.
16. Audit/report scripts summarize readiness, OCR selection, difficulty, output inventory, and cleanup plans.

Boundary clarity:

- Good: canonical image assets, native/OCR candidates, topic sidecars, and Asterion role gates are distinct.
- Good: input registry pairing is deterministic and tested.
- Good: AI sidecars are separate and marked safe/unsafe in metadata.
- Risk: docs sometimes speak about "current" generated files without a run ID or generated timestamp.
- Risk: generated taxonomy files need clearer ownership and regeneration contract.
- Risk: `question_bank.json` is both the current operator artifact and a source for downstream generation; partial writes should be prevented.

Resolved or historical script-path concerns:

- `scripts/generate_skill_maps.py` now defaults to `output/json/question_bank.json`.
- Generator scripts now expose help and `--dry-run`.
- Active audit/report commands are covered in [`COMMAND_ATLAS.md`](COMMAND_ATLAS.md).

## 8. Image Integrity Review

Measured against current `output/json/question_bank.json`:

| Check | Result |
| --- | ---: |
| Total records | 1301 |
| Duplicate `question_id` | 0 |
| Duplicate `(paper, question_number)` | 0 |
| Missing question image paths | 0 |
| Missing question image files | 0 |
| Missing mark-scheme image paths | 11 |
| Missing mark-scheme image files for nonblank paths | 0 |
| Absolute image paths | 0 |
| Canonical question path mismatches | 0 |
| Question ID mismatch | 0 |
| Paper-family path mismatch | 0 |
| Question path question-number mismatch | 0 |
| Mark-scheme path question-number mismatch | 0 |

The 11 missing mark-scheme paths correspond to the missing source companion for `9709_2025_November_33`, including records such as `33autumn25_q01` through `33autumn25_q11`.

Input registry state:

- 148 question-paper entries.
- 147 mark-scheme entries.
- 1 missing mark-scheme companion: `9709_2025_November_33`.
- No unclassified source PDFs found during registry inspection.

Question crop confidence:

- `high`: 406
- `low`: 895

Mark-scheme crop confidence:

- `high`: 746
- `medium`: 544
- `blank`: 11

Interpretation:

- Canonical image existence and relative path structure are strong.
- Crop confidence is still a major student-facing gate. Most records are not high-confidence question crops.
- Missing mark-scheme images are localized to one missing source file, not a broad pairing failure.
- Cleanup must preserve relative paths and image directory structure unless all downstream consumers are migrated and tested.

## 9. Text Extraction and Trust Review

Audit-measured text/OCR state for run `20260513T070200Z-56d469c1dd52`:

| Field | Count |
| --- | ---: |
| `ocr_ran=true` | 1301 |
| `ocr_selected=true` | 27 |
| `text_candidate_source=native` | 1274 |
| `text_candidate_source=ocr` | 27 |
| `text_candidate_decision=native_retained` | 1274 |
| `text_candidate_decision=ocr_selected` | 27 |
| `question_text_trust=high` | 271 |
| `question_text_trust=medium` | 921 |
| `question_text_trust=low` | 109 |
| `question_text_role=readable_text` | 271 |
| `question_text_role=search_hint` | 926 |
| `question_text_role=untrusted_math_text` | 104 |
| `visual_required=true` | 1030 |
| `text_fidelity_status=clean` | 1192 |
| `text_fidelity_status=degraded` | 109 |

Text-only status:

- `ready`: 177
- `review`: 1008
- `fail`: 116

Visual curation status:

- `ready`: 252
- `review`: 1035
- `fail`: 14

Top visual-required reason flags:

- `contains_equation_layout`: 726
- `contains_graph_or_diagram_prompt`: 524
- `contains_inequality_or_region_prompt`: 285
- `contains_trig_expression`: 223
- `contains_fraction_or_integral_layout`: 204
- `contains_flattened_math_structure`: 144
- `contains_log_exponential_expression`: 88
- `contains_table_or_data_prompt`: 83
- `contains_vector_notation`: 75
- `contains_math_text_corruption`: 59
- `text_order_unreliable`: 59

Top text-fidelity flags:

- `weak_extracted_text`: 92
- `math_text_corruption_detected`: 59
- `sparse_or_merged_question_text`: 45
- `native_compacted_math_corruption`: 29
- `symbol_loss_detected`: 17
- `ocr_noise_fragment_present`: 6
- `hybrid_math_text_requires_review`: 2
- `ocr_math_notation_degraded`: 2
- `pdf_control_garbage_detected`: 1

OCR rejection patterns:

- `ocr_not_clearly_better`: 399
- `ocr_validation_status_not_pass`: 384
- `ocr_missing_question_number`: 289
- `ocr_lost_greek_symbol`: 175
- `ocr_introduced_compacted_math_corruption`: 130
- `ocr_introduced_unit_corruption`: 101
- `ocr_lost_function_structure`: 83
- `ocr_lost_mark_brackets`: 83
- `page_furniture_or_header_text`: 82
- `ocr_lost_unit_structure`: 58
- `ocr_lost_math_structure`: 45

Independent OCR audit results:

- Candidate metadata exists for all 1301 records.
- No data-quality findings in OCR candidate metadata.
- OCR selected count: 27.
- Suspicious OCR-selected records: 2, both because selected text is degraded:
  - `51summer21_q05`
  - `53summer22_q07`
- No readiness-inflation risk records were reported by the OCR audit.

Observed text risk examples:

- `12spring21_q11`: diagram text and exponential notation are degraded.
- `32spring21_q09`: exponent/function expression corruption.
- `11summer21_q08`: diagram/prose merged into extracted text.
- `42spring21_q02`: trigonometric symbol corruption.
- `52spring21_q02`: run-together prose.
- `42autumn23_q07`: unit/symbol corruption.
- `11autumn22_q09`: replacement/control-character-like corruption.

Interpretation:

- The text system is appropriately conservative. It is not merely trying to maximize `ready` counts.
- `text_fidelity_status=clean` is not equivalent to "student-facing text safe"; many clean records still require visual inspection because the question contains mathematical layout or diagrams.
- `question_text_trust=medium` plus `search_hint` is a useful state for search and routing, but not for canonical student display.
- Current OCR selection appears controlled; the bigger risk is native text that looks readable but flattens math/layout.

Recommendations:

- Keep the conservative gates.
- Add a documented matrix showing allowed uses for each combination of `question_text_trust`, `question_text_role`, `text_fidelity_status`, `text_only_status`, and `visual_required`.
- Add focused regression fixtures for text that looks clean but is semantically unsafe.
- Keep OCR selected records reviewable as a small curated list.
- Do not rename statuses until Asterion contracts and tests are updated.

## 10. Trust Gates and Status Definitions

Important statuses currently in use:

- `visual_curation_status`: readiness of image/crop/visual record for canonical visual practice.
- `text_only_status`: readiness for text-only student-facing use.
- `text_fidelity_status`: quality of extracted text relative to expected mathematical/layout structure.
- `question_text_trust`: high/medium/low advisory text trust.
- `question_text_role`: readable text, search hint, or untrusted math text.
- `topic_trust_status`: whether topic metadata is normal or degraded/review required.
- Asterion role statuses: per-use allow/block/block-until-reviewed decisions.

Audit-measured status counts for run `20260513T070200Z-56d469c1dd52` show the distinction is meaningful:

- 252 visual-ready records versus 177 text-only-ready records.
- 271 high-trust readable-text records versus 1030 records still visual-required.
- 1016 records with `topic_trust_status=degraded_text`, mostly because topic classification is affected by weak or visual-required text.

Strengths:

- The statuses are not collapsed into a single misleading "ready" flag.
- Asterion export preserves per-role gates.
- Tests protect core trust behavior and Asterion status projection.

Risks:

- `clean` can be misunderstood as globally safe. It means no detected text-fidelity degradation, not safe for every use.
- `ready` appears in multiple contexts and needs a namespace in documentation.
- `review` has different causes across visual, text, topic, and export roles.
- The current docs contain stale counts, so readers may misunderstand status semantics.

Recommended documentation contract:

| Status family | Canonical meaning | Student-facing implication |
| --- | --- | --- |
| visual curation | visual/image record readiness | Can allow canonical image practice only when ready and assets OK |
| text only | extracted text display readiness | Can allow text-only display only when ready and high trust |
| text fidelity | detected corruption/degradation | Advisory; `clean` does not bypass visual-required |
| topic trust | topic metadata confidence/safety | Controls filters/routing, not canonical content |
| Asterion role | downstream use decision | Consumer-facing contract; must be honored exactly |

## 11. Topic, Taxonomy, and Routing Review

The project has two related but separate AI/topic paths:

- `deepseek_enrich.py`: AI-assisted enrichment sidecar with topic/subtopic/skill validation.
- `topic_routing.py`: strict parent-topic routing sidecar for allowed paper-family topics.

The strict runtime topic profile is in `src/exam_bank/runtime_profile.json`. Current question-bank topics validated against that profile produced zero invalid current `topic` values.

Audit-measured topic distribution by paper family for run `20260513T070200Z-56d469c1dd52` is plausible and paper-specific:

P1 top topics:

- functions: 74
- differentiation: 69
- trigonometry: 51
- series: 46
- coordinate geometry: 41
- binomial/circular measure: 38 each

P3 top topics:

- parametric equations: 91
- logarithmic/exponential functions: 56
- complex numbers: 46
- vectors: 37
- trigonometry: 35
- numerical methods: 33

P4 top topics:

- work, energy, power: 63
- momentum and impulse: 35
- friction: 33
- constant acceleration kinematics: 27
- connected particles/equilibrium coplanar forces: 25 each

P5 top topics:

- central tendency and dispersion: 78
- probability: 51
- permutations/combinations: 44
- distributions: 42
- data representation: 17

Audit-measured topic confidence state for run `20260513T070200Z-56d469c1dd52`:

- `topic_confidence=high`: 333
- `topic_confidence=medium`: 113
- `topic_confidence=low`: 855
- `topic_uncertain=true`: 855
- `topic_trust_status=degraded_text`: 1016
- `topic_trust_status=review_required`: 208
- `topic_trust_status=normal`: 77

Audit-measured routing sidecar state for run `20260513T070200Z-56d469c1dd52`:

- Schema: `exam_bank.topic_routing_sidecar` v1.
- Records: 1301.
- Successful: 1148.
- Failed: 153.
- Review required: 221.
- Failure reason: 153 schema-validation errors.
- `safe_for_strict_filters=false`.

Strengths:

- AI cannot invent main topic IDs in the strict routing path.
- Topic IDs are paper-family scoped.
- Provider JSON is validated before acceptance.
- Review gates exist for weak evidence and insufficient text.

Risks:

- The current routing sidecar is not fully safe for strict filters due to schema failures.
- Docs must clearly distinguish local deterministic topic classification from AI routing sidecars.
- Generated taxonomy files need safer regeneration commands and a canonical ownership map.
- DeepSeek sidecars in archive include mixed prompt versions and error records; only explicitly clean sidecars should be used for Asterion export.

Recommendations:

- Keep `runtime_profile.json` as the allowed runtime topic authority unless intentionally migrating.
- Add a sidecar-level hard gate in Asterion docs: `safe_for_strict_filters=true` required for strict topic filtering.
- Add topic routing failure examples to tests.
- Add a "topic source" field to docs and exports wherever topic data can come from deterministic local, DeepSeek enrichment, or strict routing.
- Make topic map generation scripts dry-run safe before regenerating taxonomy assets.

## 12. Asterion Export Readiness Review

Audit-measured Asterion files for run `20260513T070200Z-56d469c1dd52`:

| File | Current | Regenerable | Role |
| --- | --- | --- | --- |
| `output/asterion/exports/latest/asterion_question_bank_v1.json` | Yes | Yes | Main role-gated projection |
| `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json` | Yes | Yes | Content Lab candidate projection |
| `output/json/question_bank.topic_routing.v1.json` | Yes | Yes, AI/API dependent | Topic routing sidecar |
| archived AI sidecars | Historical | Yes, AI/API dependent | Evidence or optional enrichment |

Main Asterion export state for run `20260513T070200Z-56d469c1dd52`:

- Schema: `asterion.question_bank` v1.
- Records: 1301.
- `canonical_assets_ok=true`: 1290.
- `canonical_assets_ok=false`: 11.

Usage-role counts:

| Role | Allow/include | Block until reviewed | Block/exclude |
| --- | ---: | ---: | ---: |
| canonical practice | 252 | 0 | 1049 |
| field guide source | 252 | 1021 | 28 |
| quick check source | 51 | 1222 | 28 |
| warmup generator source | 228 | 1045 | 28 |
| guardian candidate | 252 | 0 | 1049 |
| p3 readiness metric | 396 include | 0 | 905 exclude |

Top Asterion reason codes:

- `text_only_blocked_visual_required`: 1030
- `text_only_blocked_status_review`: 1008
- `text_only_blocked_trust_medium`: 921
- `question_crop_not_high_confidence`: 895
- topic confidence/uncertain: 855 each
- `mark_scheme_crop_not_high_confidence`: 555
- `validation_status_review`: 370
- `text_only_status_fail`: 116
- `text_only_blocked_trust_low`: 109
- `untrusted_math_text`: 104
- `subpart_marks_missing`: 48
- `mapping_status_fail`: 21
- `validation_status_fail`: 14
- `canonical_assets_missing_or_unhashed`: 11
- `missing_mark_scheme_image_path`: 11
- `marks_inconsistent`: 3

Content Lab candidate export:

- Schema: `asterion.content_lab_candidates` v1.
- Candidates: 2416.
- `machine_candidate`: 502.
- `blocked_until_reviewed`: 1914.

Safe use interpretation:

- Canonical image practice: safe only for role-allowed records.
- Text-assisted search: possible for medium/high trust records, but must display advisory status.
- Topic routing: use strict sidecar only after sidecar-level safety metadata is checked.
- Teacher review: many blocked-until-reviewed records are appropriate.
- Quick-check/field-guide candidates: use only role-specific gates, not global presence.
- Guardian candidates: limited to canonical-practice-safe records today.
- Unsafe/degraded records: remain useful for review queues, not student-facing automation.

Recommendations:

- Document Asterion schema v1 fields in a contract doc.
- Treat role status fields as the only downstream permission surface.
- Add a release checklist for Asterion exports: canonical assets OK, sidecar safety, schema validation, reason-code summary, and sample record inspection.
- Keep raw OCR out of broad Asterion projections unless there is a specific review-only field and contract.

## 13. Mark-Scheme Readiness Review

Audit-measured mark-scheme state for run `20260513T070200Z-56d469c1dd52`:

| Check | Result |
| --- | ---: |
| Missing mark-scheme image paths | 11 |
| Nonblank mark-scheme paths missing files | 0 |
| Mark-scheme crop confidence high | 746 |
| Mark-scheme crop confidence medium | 544 |
| Mark-scheme crop confidence blank | 11 |
| Records with question solution marks | 1301 |
| Records with subparts | 968 |
| Records with any subpart solution mark | 0 |
| Question/MS total mismatches | 3 |
| Paper total status matched | 1301 |

Basic display readiness:

- Good for records with nonblank mark-scheme image paths.
- Role gates should block missing/blank mark-scheme assets.

Multi-part marking readiness:

- Not ready. Subpart-level mark breakdowns are absent.
- Multi-part records can show full mark-scheme images but should not claim precise subpart self-marking.

Student self-marking readiness:

- Limited to displaying mark-scheme images for approved records.
- Not ready for structured method/accuracy mark parsing.

Guardian/worked-example/Content Lab readiness:

- Mark-scheme images can support teacher review and future lesson generation.
- Current mark-scheme text reliability and subpart mark gaps block automated marking, worked-solution derivation, and Guardian verification at scale.

Recommendations:

- Keep mark-scheme images canonical.
- Add a mark-scheme contract distinguishing image display, full-question mark total, subpart marks, and parsed marking events.
- Add an output-level test that current generated bank has no missing mark-scheme assets except explicitly documented missing source companions.
- Add targeted fixtures for the 3 mark-total mismatch records before trying to fix extraction.

## 14. Schema and Validation Review

Existing schema-like contracts:

- `QuestionRecord` and related dataclasses in `models.py`.
- `exporters.py` schema version and manifest fields.
- Asterion export schema names and role fields.
- DeepSeek sidecar validation logic.
- Topic routing sidecar validation logic.
- Runtime profile topic IDs.
- Output contract tests.

Strengths:

- The question bank has an explicit schema name/version.
- Asterion exports have schema identifiers.
- Tests assert important output contract fields.
- AI sidecars validate topic/subtopic/skill IDs.
- Topic routing validates strict allowed topics.

Gaps:

- There are no standalone JSON Schema files for the main artifacts.
- Asterion export schemas are code-defined but not separately documented.
- Content Lab candidate schema needs a consumer-facing contract.
- Topic routing sidecar schema needs documented failure semantics.
- Audit reports and text-confidence reports do not have formal contracts.
- Validation commands exist, but not all are wired into one documented release checklist.

Artifacts that should get formal schema/contracts:

1. Canonical question bank JSON.
2. Asterion question bank export.
3. Content Lab candidates.
4. Topic routing sidecar.
5. DeepSeek/AI enrichment sidecar.
6. Skill maps and topic filter maps.
7. Audit report JSON if it remains a generated artifact.
8. OCR/text confidence report.
9. Output inventory and cleanup plan.

Recommendation:

Before broad cleanup, create lightweight schema contract docs or JSON Schemas for Asterion and topic sidecars. Do not refactor file names or field names until contracts and tests are updated.

## 15. Test Coverage Review

Full test run at audit time:

```text
419 passed, 3 skipped in 110.63s (0:01:50)
```

CI:

- `.github/workflows/tests.yml` runs Python 3.11.
- Installs package with dev extras.
- Runs `python -m pytest`.

Coverage areas observed:

| Area | Status |
| --- | --- |
| extraction behavior | Covered by sample pipeline and extraction tests |
| OCR selection | Covered |
| trust gates | Covered |
| text quality/status flags | Covered in multiple targeted tests |
| topic normalization/routing | Covered |
| schema/output contract | Covered in tests |
| Asterion exports | Covered |
| image path integrity | Partly covered |
| mark-scheme pairing | Covered in code-level pairing tests |
| CLI behavior | Partly covered |
| run status | Covered |
| output cleanup planner | Covered |
| repo hygiene | Covered |

High-value missing or weak tests:

- Output-level test for the current generated bank: all question images exist; all nonblank mark-scheme images exist; missing MS paths must match documented missing source companions.
- Output-level test for no duplicate current `(paper, question_number)` pairs.
- Asterion fixture asserting blocked records cannot become student-facing because of medium-trust search text.
- Topic routing fixture for provider schema failures and `safe_for_strict_filters=false`.
- Generator-script CLI safety tests: `--help` must not mutate files.
- Atomic write behavior for exports.
- Fixture for each known text corruption class: flattened fractions, exponent corruption, dropped question number, table/diagram pollution, unit corruption.
- Mark-total mismatch regression fixtures for the 3 known records.
- Documentation command snippets smoke tests for README command atlas, if practical.

Brittle or slow areas:

- Full suite takes nearly 2 minutes, which is acceptable but could be split into fast and integration groups.
- `tests/test_sample_pipeline.py` is large and integration-heavy.
- Image rendering tests are valuable but should be marked clearly if they dominate runtime.

Tests that should be added before cleanup:

1. Script help/dry-run safety.
2. Asterion export contract snapshots.
3. Current-output integrity audit on a small fixture.
4. Topic sidecar strict failure behavior.
5. Output inventory/cleanup classification for archive retention.

## 16. Scripts and Command Atlas

### Main Commands

The current operator-facing command map is [`COMMAND_ATLAS.md`](COMMAND_ATLAS.md). It supersedes the older inline command table in this audit snapshot and includes purpose, input, output, runtime category, and workflow category for each active command.

### Audit Scripts

The active audit commands are documented in [`COMMAND_ATLAS.md`](COMMAND_ATLAS.md). Commands not listed there should be treated as historical until their CLI surface is re-verified.

### Commands Needing Cleanup

| Command / script | Issue | Recommendation |
| --- | --- | --- |
| historical audit table in this file | Preserved stale command examples from the original audit snapshot | Replaced with pointer to [`COMMAND_ATLAS.md`](COMMAND_ATLAS.md) |
| archived output commands | Old `output_ocr_candidate` references in history/archive evidence | Keep as historical evidence only; use `output/candidates/ocr/latest/` for new OCR candidate commands |

### Progress/Status Recommendations

Long-running workflows should report:

- elapsed time
- current stage
- current paper/component/session
- records completed and total
- percentage
- per-stage ETA
- output path being written
- cache hits/misses
- OCR selected/rejected counts so far
- AI batch count and retry count
- resumability/checkpoint path
- final next command suggestion

## 17. Performance and Runtime Review

Known runtime expectations:

- Standard full run: around 20 minutes.
- AI-heavy run: up to 2 hours.
- Full test suite: 1:50 observed.

Performance opportunities:

### Low-Risk Quick Wins

- Add atomic JSON writes for main exports and sidecars.
- Add script `--help` and `--dry-run` safety to generators.
- Expand run-status terminal summaries without changing data flow.
- Add output inventory summaries to README.
- Document current command atlas.
- Add small focused integrity tests for current output contracts.

### Medium-Risk Refactors

- Consolidate overlapping audit scripts into one CLI group while preserving old command wrappers.
- Introduce explicit cache keys for OCR image crops and OCR text candidates.
- Split tests into fast/unit and integration/rendering groups.
- Create formal schema files and validate generated artifacts in release commands.
- Add resumable checkpoint manifests for topic routing and DeepSeek enrichment.

### High-Risk / Defer

- Changing image directory layout.
- Renaming exported status fields.
- Rewriting question span extraction.
- Replacing topic taxonomy structure.
- Treating parsed mark-scheme text as canonical.
- Aggressively deleting archives before retention metadata exists.

### Requires Tests First

- OCR selection threshold tuning.
- Crop confidence logic changes.
- Mark-scheme extraction and stitching changes.
- Status renames or semantic changes.
- Asterion role-gate changes.
- Topic routing normalization changes.

### Requires Schema First

- Asterion export field changes.
- Content Lab candidate field changes.
- AI sidecar shape changes.
- Topic routing sidecar changes.
- Skill map/topic filter map relocation.

### Requires Documentation First

- Cleanup of historical docs.
- Archive retention policy.
- Standard operator workflow changes.
- Asterion consumer contract changes.
- Student-facing eligibility definition changes.

## 18. Documentation Review

Reviewed documentation:

- `README.md`
- `ROADMAP.md`
- `docs/TRUST_MODEL.md`
- `docs/AI_ASSISTED_ENRICHMENT.md`
- `docs/AUTO_TRIAGE.md`
- `docs/TRIAGE_WORKFLOW.md`
- `docs/PROJECT_REVIEW.md`

Accurate/useful docs:

- Image-first policy in README and TRUST_MODEL is directionally correct.
- AI sidecar policy in `AI_ASSISTED_ENRICHMENT.md` is useful and conservative.
- Auto-triage and triage workflow docs preserve valuable process knowledge.
- Roadmap captures many still-valid future work areas.

Stale docs:

- README current audited state now says the canonical export is OCR-enabled.
- ROADMAP measured current state uses older hard-blocker and OCR status.
- TRUST_MODEL measured tier counts are stale.
- AUTO_TRIAGE active examples now use `output/candidates/ocr/latest/`; legacy `output_ocr_candidate` is marked historical/compatibility evidence.
- TRIAGE_WORKFLOW examples reference output folders that are not present in current output.
- PROJECT_REVIEW is a historical review, not current.

Missing docs:

- Asterion export schema contract.
- Content Lab candidate schema contract.
- Topic routing sidecar contract and safety metadata.
- Output archive retention policy.
- Generator script safety policy.
- Current command atlas.
- Release checklist for generating and validating a current export.

Contradictions to fix:

- OCR enabled versus not enabled.
- Current generated file names and archive paths.
- Hard blocker counts.
- Whether old AI sidecars are safe for Asterion.
- Which docs are current versus historical.

Documentation update plan:

1. Keep this audit as the current baseline.
2. Add "current as of run ID" labels to measured-state sections.
3. Move stale reviews to `docs/history/` or add clear historical banners.
4. Replace repeated readiness counts in multiple docs with references to generated audit commands.
5. Add Asterion and topic sidecar contract docs before changing export fields.
6. Update README with a concise command atlas after cleanup command names are settled.

## 19. Dependency and Environment Review

Project configuration:

- Python: `>=3.10`.
- CI uses Python 3.11.
- Runtime dependencies:
  - PyMuPDF
  - pdfplumber
  - Pillow
  - pytesseract
  - PyYAML
  - openai
- Dev dependency:
  - pytest

Environment assumptions:

- Tesseract must be installed for OCR workflows.
- DeepSeek/OpenAI-compatible enrichment requires provider/API environment configuration.
- `config.yaml` disables OCR and classification API by default.
- Paths are mostly repository-relative and avoid absolute image paths in exports.

Health:

- `.venv` is ignored.
- `.env*` is ignored.
- `.pytest_cache` is ignored.
- Generated output is ignored except `output/json/.gitkeep`.
- No dependency bloat was obvious from `pyproject.toml`.

Risks:

- README should explicitly state Tesseract installation expectations for OCR-enabled runs.
- DeepSeek/API environment variables and safe sidecar usage should be documented in one place.
- Generator scripts should not assume old local output paths.
- CI uses 3.11 while package supports 3.10+; if 3.10 support matters, add a 3.10 CI job.

## 20. Cleanup Recommendations

Do not start with deletion. Start by making the current state explicit and testable.

### Archive

- Preserve `output/archive/generated_cleanup_20260513T233456Z` until a manifest lists each archived sidecar and whether it is a baseline, historical evidence, or disposable.
- Move or label `docs/PROJECT_REVIEW.md` as historical.
- Consider `docs/history/` for old implementation reviews and stale handoffs after indexing them.

### Delete Later

Only after manifest and regeneration checks:

- duplicate generated image trees in archives
- old smoke/fix sidecars that are superseded and not used as regression evidence
- local cache/temp files under ignored paths
- obsolete inventory reports

### Move

- Historical docs to `docs/history/`.
- Generated taxonomy reports away from source-like taxonomy files if mixed.
- Archived run sidecars under a consistent `output/archive/<run-id>/manifest.json` layout.

### Rename

Avoid renaming exported fields in the cleanup pass. Rename only local docs/scripts if needed.

Potential later renames after contracts:

- Make status names more namespace-explicit in docs, not necessarily JSON.
- Clarify `clean` as `text_fidelity_clean` in docs.
- Clarify "ready" with status family prefixes in docs.

### Gitignore

Keep current output ignore policy. Verify after cleanup:

- no generated outputs accidentally tracked
- no cache files outside ignored paths
- no virtualenv or pytest cache files tracked

### Regenerate Instead of Store

Safe candidates after acceptance checks:

- Asterion projections
- Content Lab candidates
- output inventories
- audit summaries
- topic routing sidecars only if provider access and checkpointing are stable

Not safe to delete casually:

- current canonical question bank
- current image trees
- archived AI/OCR evidence until manifest exists
- taxonomy files until source/generated ownership is clarified

### Tests Before Cleanup

- Script `--help` no-mutation tests.
- Output integrity fixture test.
- Asterion role-gate snapshot.
- Topic sidecar failure metadata test.
- Archive inventory classification test.

### Commands After Cleanup

- `git status --short`
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python -m exam_bank.cli output-inventory --root output --include-size --max-depth 4`
- `.venv/bin/python -m exam_bank.cli output-cleanup-plan --root output --include-size --max-depth 4`
- `.venv/bin/python -m exam_bank.cli audit --input output/json/question_bank.json`
- `.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json`
- regenerate Asterion exports and compare summary counts

## 21. Optimization Recommendations

### Runtime and Status Tracking

- Expand run status to show current paper, record counts, elapsed time, ETA, cache hits, OCR selected/rejected counts, and output paths.
- Persist stage-level summaries in `output/run_status/<run_id>/`.
- Print a final "next commands" block after long runs.

### Caching and Checkpointing

- Cache OCR results by image path plus image hash.
- Cache rendered question crops by source PDF hash, page range, crop box, and renderer version.
- Checkpoint AI batches with provider response status, retry count, prompt version, and schema validation result.
- Support resume for topic routing and enrichment without reprocessing successful records.

### Pipeline Simplification

- Keep the main pipeline stages explicit; do not collapse safety gates.
- Consolidate duplicate audit logic behind CLI subcommands.
- Move output cleanup and inventory workflows into documented release commands.
- Keep AI enrichment sidecars separate from canonical export.

### Schema and Validation

- Add JSON Schema or contract docs for major artifacts.
- Add a release validation command that checks question bank, topic sidecar, Asterion export, Content Lab export, and image paths together.
- Add sidecar compatibility checks by `question_id`, generated timestamp, source run ID, and schema version.

### Export Generation

- Write outputs atomically.
- Include input artifact hashes or source run IDs in Asterion exports.
- Add export summary diffing so operators can see changed allow/block counts.
- Block strict Asterion topic filters if the sidecar says `safe_for_strict_filters=false`.

### AI Run Safety

- Make prompt version, model, temperature, schema version, and retry policy mandatory metadata.
- Keep failed provider responses separate from successful but review-required records.
- Add resumable batches and partial-run audit summaries.
- Never merge AI output into canonical topic structures without deterministic normalization.

## 22. Proposed Phased Action Plan

### Phase 1: Safe Cleanup Prerequisites

1. Add script help/dry-run safety tests.
2. Add generator argparse/help guards.
3. Add output integrity checks for question images, mark-scheme images, duplicates, and known missing source companions.
4. Add Asterion and topic sidecar contract docs.
5. Add an archive manifest for `output/archive/generated_cleanup_20260513T233456Z`.
6. Update README measured state to point to this audit and current run ID.

Acceptance gate:

- Full tests pass.
- Generator `--help` commands do not modify files.
- Current output integrity report is reproducible.

### Phase 2: Cleanup and Reorganization

1. Move historical docs to `docs/history/` or add historical banners.
2. Normalize command documentation in README.
3. Classify archived generated artifacts using the manifest.
4. Remove only documented disposable files under ignored generated paths.
5. Keep current canonical output and image trees until a fresh regeneration is validated.

Acceptance gate:

- `output-inventory` and `output-cleanup-plan` produce expected classifications.
- Asterion export counts are unchanged unless intentionally regenerated.

### Phase 3: Low-Risk Optimization

1. Add atomic JSON writes.
2. Improve run-status terminal output.
3. Add export summary diffs.
4. Split full tests into fast/integration markers while keeping current CI behavior.
5. Update stale script default paths.

Acceptance gate:

- Full tests pass.
- Standard commands keep the same outputs or explain intentional metadata-only changes.

### Phase 4: Deeper Refactors

1. Add OCR/image rendering cache keys.
2. Add AI batch checkpoint/resume.
3. Consolidate audit scripts behind CLI subcommands.
4. Improve mark-scheme subpart parsing with fixtures.
5. Revisit crop confidence only with regression samples.

Acceptance gate:

- Runtime improves measurably.
- Readiness counts do not inflate without evidence.
- Known corruption and mark-scheme fixtures remain protected.

### Phase 5: Documentation and Acceptance Gates

1. Publish final README command atlas.
2. Publish artifact schema/contract docs.
3. Add release checklist.
4. Add Asterion consumer guidance.
5. Add "do not break" constraints to cleanup PR template or roadmap.

Acceptance gate:

- A new operator can run standard, OCR-enabled, audit, topic, AI, and Asterion workflows from docs.
- Consumer-facing safety semantics are unambiguous.

Post-audit scope clarification: the accepted active path after Phase 1 through Phase 3 keeps Phase 4 as future deeper-refactor work and does not treat Phase 5 as separate current implementation scope. Future topic/difficulty leverage may eventually use exam reports and grade boundaries, but that remains deferred until after deeper refactors and a separate audited plan.

## 23. Do-Not-Break Constraints

The cleanup and optimization work must not break:

- image-first canonical source-of-truth policy
- question image paths
- mark-scheme image paths
- question/mark-scheme pairing
- paper metadata
- session/year/component metadata
- question numbers
- multi-part question representation
- full-question mark totals
- paper-total validation and recovery semantics
- text trust status semantics
- distinction between `visual_curation_status`, `text_only_status`, and `text_fidelity_status`
- conservative student-facing eligibility
- strict topic mapping into allowed CAIE 9709 structures
- prevention of invented main curriculum topics
- Asterion export file contracts
- Asterion role-gate semantics
- ability to distinguish canonical images from advisory text
- ability to distinguish deterministic metadata from AI sidecars
- ability to run with OCR disabled
- ability to run with OCR enabled
- current regression tests
- current output contract tests
- run reproducibility from `input/` and `config.yaml`
- relative paths in generated JSON
- archive evidence needed to explain previous runs
- reviewer workflows for blocked/degraded records

## 24. Open Questions

1. Should `9709_2025_November_33` remain in the source set without a mark scheme, or should the missing mark scheme be added before the next canonical regeneration?
2. Which archived AI sidecars are formal baselines and which are disposable run evidence?
3. Is `exam_bank_taxonomy/` intended to contain canonical source data, generated exports, or both?
4. Should Python 3.10 be tested in CI if the package declares `>=3.10`?
5. Which Asterion consumer owns the schema contract for `asterion_question_bank_v1.json`?
6. Should topic routing sidecar failures block export generation or only block strict topic filtering?
7. What retention policy should apply to local generated image trees and archive folders?
8. Which status names are externally consumed and therefore cannot be renamed?
9. Should README carry current measured counts, or should it always point to generated audit commands?
10. What is the minimum accepted record set for Guardian candidates and quick-check candidates?

## 25. Suggested Next Codex Prompt

Use this prompt for the follow-up cleanup and optimization pass:

```text
Use docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md as the baseline. Do a safe cleanup-prep pass only. Do not delete generated archives yet.

Implement Phase 1 from the audit:
1. Add CLI help/dry-run safety to generator scripts, especially scripts/generate_topic_filter_maps.py and scripts/generate_skill_maps.py.
2. Add tests proving generator --help does not mutate files.
3. Add an output integrity audit/test that checks current question image paths, nonblank mark-scheme image paths, duplicate IDs, duplicate paper/question pairs, and explicitly documents the missing 9709_2025_November_33 mark scheme.
4. Add or update docs for Asterion export and topic routing sidecar contracts.
5. Add a manifest for output/archive/generated_cleanup_20260513T233456Z classifying archived sidecars as baseline, historical evidence, or disposable.
6. Update README only where it is factually stale, replacing duplicated measured counts with commands or references to the audit.

Keep changes small and reversible. Run pytest and the relevant audit commands. Report changed files, validation results, and any cleanup that should still be deferred.
```

## Appendix A: Commands Run During Audit

Representative inspection and validation commands:

```bash
git status --short
find . -maxdepth 2 -type d
du -sh *
git ls-files
rg --files
sed -n '1,220p' README.md
sed -n '1,220p' pyproject.toml
sed -n '1,220p' config.yaml
sed -n '1,260p' src/exam_bank/cli.py
sed -n '1,260p' src/exam_bank/pipeline.py
sed -n '1,260p' src/exam_bank/models.py
sed -n '1,260p' src/exam_bank/exporters.py
sed -n '1,260p' src/exam_bank/ocr.py
sed -n '1,260p' src/exam_bank/trust.py
sed -n '1,260p' src/exam_bank/asterion_export.py
sed -n '1,260p' src/exam_bank/topic_routing.py
sed -n '1,220p' src/exam_bank/deepseek_enrich.py
.venv/bin/python -m exam_bank.cli audit --input output/json/question_bank.json
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json
.venv/bin/python scripts/audit_difficulty.py --input output/json/question_bank.json
.venv/bin/python -m exam_bank.cli output-inventory --root output --include-size --max-depth 4
.venv/bin/python -m exam_bank.cli output-cleanup-plan --root output --include-size --max-depth 4
.venv/bin/python -m pytest -q
```

Additional inline Python inspections were used to count records, image-path existence, mark-scheme status, registry pairing, topic sidecar failures, Asterion role gates, Content Lab candidate status, and archived AI sidecar metadata.

During script help inspection, `scripts/generate_topic_filter_maps.py --help` executed generation behavior and modified generated taxonomy files. Those accidental changes were reverted immediately with targeted `git restore` for the affected taxonomy files. This is recorded as a command-safety audit finding.
