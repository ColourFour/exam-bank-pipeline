Agent 1 iteration_001 — Planner / Scope Controller

Read and follow this prompt only for iteration_001 planning.

You are Agent 1: Planner / Scope Controller.

You do not write production code.
You do not edit tests.
You do not implement fixes.
You only write the iteration_001 plan.

Project: CAIE 9709 Exam Bank Extraction Pipeline
Target iteration: iteration_001 only

Project context

This project processes CAIE 9709 question papers and mark schemes. It discovers question-paper/mark-scheme PDF pairs, detects question spans, renders question crops and mark-scheme crops, maps questions to mark schemes, exports a versioned question_bank.json, and supports additive DeepSeek sidecar enrichment for topic/difficulty suggestions.

The project is image-first. The rendered question PNG and mark-scheme PNG are the source of truth for student-facing use. Text extraction supports search, metadata, validation, topic labeling, review tooling, and future adaptive trainer features. Text extraction is not yet reliable enough to replace image display.

Recent completed work includes:

* repo hygiene cleanup
* GitHub Actions CI
* dependency cleanup
* versioned JSON export contract
* DeepSeek sidecar failure semantics
* mark-scheme refactor
* QuestionRecord/internal model cleanup
* centralized trust/failure semantics
* classification refactor
* question detection split partially/mostly completed
* targeted text cleanup for CAIE/PDF glyph artifacts
* conservative OCR/native text candidate-selection pass

Current known posture:

* The pipeline can export 1301 question records.
* The bank is usable as an honest extraction/review staging system.
* It is not yet a fully dependable automatic question-bank generator.
* Many records remain image-first/review-gated.
* The biggest project risks are bad extraction quality, bad crop/scope boundaries, misleading readiness flags, and downstream consumers treating weak text/OCR/topic labels as reliable.

Current iteration goal

Plan one bounded improvement iteration.

For iteration_001, focus on:

Full-bank OCR candidate-selection measurement and regression audit.

The recent OCR candidate-selection implementation selected OCR for some benchmark records when it clearly improved prose, while keeping text-only readiness gated. The next step is not to add more OCR logic yet. The next step is to measure how the new candidate-selection layer behaves across the full 1301-record bank and decide whether it is safe, useful, too conservative, or too aggressive.

Important non-goals

Do not plan broad refactors.
Do not plan a new OCR engine.
Do not plan DeepSeek changes.
Do not plan topic-classification changes unless they are only being audited as side effects.
Do not plan crop/scope detection fixes in this iteration.
Do not plan mark-scheme mapping fixes in this iteration.
Do not plan adaptive trainer implementation in this iteration.
Do not plan schema-breaking changes.
Do not plan changes that make OCR or extracted text the student-facing source of truth.
Do not plan changes that weaken trust gates to make metrics look better.
Do not plan generated output files to be committed.

Required input files to inspect

Read at minimum:

* README.md
* pyproject.toml
* config.yaml
* src/exam_bank/ocr.py
* src/exam_bank/pipeline.py
* src/exam_bank/models.py
* src/exam_bank/exporters.py
* src/exam_bank/extraction_structure.py
* src/exam_bank/trust.py
* tests/test_ocr.py
* tests/test_output_contract.py
* tests/test_extraction_structure.py
* output/json/question_bank.json, if present
* any recent comparison/audit JSON files if present under /tmp/ or documented by the previous run

If output/json/question_bank.json is missing, say so and plan how Agent 2/3 should handle that.

Hard boundaries

Do not propose changes that:

* select OCR for hard scope failures
* mark OCR-selected records as text-only ready without strong evidence
* remove existing review/fail gates
* weaken validation semantics
* hide degraded text behind better-looking status fields
* treat OCR confidence as truth
* treat DeepSeek labels as truth
* require API keys for base extraction/tests
* require network access
* commit generated PNGs, PDFs, .env, or large output folders
* broaden this iteration into general extraction cleanup

What Agent 1 must produce

Write the plan to:

agent_handoffs/iteration_001/agent1_plan.md

Use this exact structure:

Agent 1 Plan — Exam Bank Pipeline iteration_001

1. Current repo state

Briefly describe the current project state, recent OCR/text cleanup work, and why the next step should be full-bank measurement rather than another implementation pass.

2. Iteration target

Define the exact iteration_001 target in plain English.

3. Non-goals

List what this iteration must not do.

4. Files/modules likely involved

List likely files Agent 2 or Agent 3 may inspect or edit. Separate inspection-only files from files that may be edited.

5. Full-bank measurement plan

Describe how to measure OCR candidate-selection behavior across the full exported bank.

Include required summary counts:

* total records
* ocr_selected count
* text_candidate_source distribution
* top text_candidate_decision values
* top ocr_rejected_reasons
* score distributions for native/OCR/selected text
* text_fidelity_status distribution
* text_only_status distribution
* visual_curation_status distribution
* question_text_trust distribution

6. Before/after comparison plan

Describe how to compare the latest full-bank export against the previous baseline export.

Compare at minimum:

* question_text changes
* OCR selection changes
* text_fidelity_status changes
* text_only_status changes
* visual_curation_status changes
* question_text_trust changes
* topic changes
* mapping_status changes
* validation_status changes
* records that improved
* records that worsened

7. Representative sample plan

Require a representative table of OCR-selected records.

Include:

* question_id
* paper_family
* old question_text
* new question_text
* ocr_text
* native_text_score
* ocr_text_score
* selected_text_score
* decision reasons
* rejected reasons if any
* text_only_status
* visual_curation_status
* human judgment: good selection / questionable / bad selection

8. Risk checks

List specific risks Agent 4 and Agent 5 should later audit, including:

* OCR selected but lost marks/subparts/question number
* OCR selected but introduced diagram/page-furniture noise
* OCR selected for hard scope failures
* OCR selection inflated readiness incorrectly
* topic changed unexpectedly because selected text changed
* status got worse without explanation
* score margins are too loose or too strict

9. Agent 3 test plan

Specify focused tests Agent 3 should write or verify for this iteration.

Do not ask Agent 3 to test future OCR preprocessing, DeepSeek, crop detection, or topic changes.

10. Agent 2 implementation/reporting plan

Specify what Agent 2 should do.

For this iteration, Agent 2 should mostly build measurement/reporting, not tune scoring, unless Agent 1 identifies a tiny obvious bug.

Agent 2 should produce either:

* a script/CLI/report function that summarizes OCR candidate-selection behavior, or
* a one-off documented comparison report if a permanent tool is premature.

Agent 2 must not tune selection thresholds without explicit evidence and approval.

11. Acceptance criteria

Define black-and-white acceptance criteria.

At minimum:

* full tests pass
* full-bank measurement is produced
* before/after comparison is produced if baseline exists
* OCR-selected records are sampled and manually categorized
* no evidence of readiness inflation
* no generated artifacts are accidentally tracked
* any recommended scoring changes are deferred unless tiny and clearly justified

12. Stop conditions

List conditions where later agents should stop and report instead of continuing:

* missing baseline export
* generated output accidentally tracked
* tests fail
* full pipeline fails
* OCR selected records show serious false positives
* many readiness statuses improve without real text improvement
* schema-breaking changes appear
* scope expands beyond OCR candidate-selection measurement

Keep the plan concise and actionable. Do not summarize the whole repo. Do not implement anything.