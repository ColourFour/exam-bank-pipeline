# Trust Model

The pipeline is image-first. Question crops and mark-scheme crops are the source of truth. Extracted text, OCR text, local topics, difficulty scores, and DeepSeek enrichment are metadata layered around the visual evidence.

## Core Rules

Images are source of truth:

- `question_image_path` and `mark_scheme_image_path` should be used for student-visible question and answer content.
- If an image is missing, wrong, or mismatched, the record is not student-ready even if the text looks good.

Text is metadata:

- `question_text` supports search, routing, review, labeling, and enrichment.
- It can be clean enough for some workflows, but it is not the canonical representation of math-heavy CAIE questions unless trust gates pass.

OCR is candidate evidence:

- OCR is a fallback/candidate source.
- OCR should not override native text unless it is clearly better and avoids hard rejection reasons.
- OCR can lose question numbers, marks, subparts, fractions, exponents, vectors, matrices, and diagram context.

DeepSeek/topic enrichment is sidecar metadata:

- `question_bank.deepseek.json` does not mutate `question_bank.json`.
- Local and DeepSeek topic/difficulty labels support review and routing but should not replace visual evidence.
- Final review flags from enrichment should be honored by downstream workflows.

Auto-triage decisions are evidence gates:

- `auto-triage-compare` can accept or reject an implementation pass, but it does not promote individual records into a student-ready subset.
- Accepted auto-triage evidence still depends on record-level validation, mapping, scope, text-fidelity, topic-trust, and visual-curation gates.
- A no-OCR output can help debug layout, but it cannot be used as production OCR proof.

Validation statuses are gates:

- `notes.validation_status`, `notes.validation_flags`, `notes.mapping_status`, `notes.mapping_failure_reason`, `notes.scope_quality_status`, `notes.text_fidelity_status`, `notes.topic_trust_status`, and `visual_curation_status` control eligibility.
- Do not suppress flags to make status counts look better.

Mixed-good records are normal:

- A record can be visually usable but text-untrusted.
- A record can have good text but bad crop or bad mark-scheme mapping.
- A record can have a local topic label but still require review because text fidelity or scope is degraded.

## Current Status Fields

Use these fields for downstream filtering:

- `visual_curation_status`: readiness of visual artifacts.
- `text_only_status`: readiness for text-only workflows.
- `question_text_role`: `readable_text`, `search_hint`, `untrusted_math_text`, or `missing`.
- `question_text_trust`: `high`, `medium`, `low`, or `unusable`.
- `visual_required`: whether a visual crop is required to understand the item.
- `notes.validation_status`: extraction validation outcome.
- `notes.validation_flags`: reason flags for validation status.
- `notes.mapping_status`: question-to-mark-scheme mapping pass/fail.
- `notes.mapping_failure_reason`: specific mapping failure.
- `notes.scope_quality_status`: whether selected question scope is clean, review, or fail.
- `notes.text_source_profile`: `native_pdf`, `hybrid`, or `ocr`.
- `notes.text_fidelity_status`: `clean`, `degraded`, or `unusable`.
- `notes.text_fidelity_flags`: text corruption or missing-structure reasons.
- `notes.topic_trust_status`: `normal`, `degraded_text`, or `review_required`.
- `notes.review_flags`: additional review/debug flags.

When a field exists both top-level and under `notes`, prefer the top-level value. In the current schema, several trust fields are under `notes`.

## Recommended Filters

### Teacher Review Mode

Teacher/reviewer mode should include almost everything with a question image, but sort risky records first.

Suggested include:

- `question_image_path` exists.

Suggested priority:

- `notes.validation_status != "pass"`
- `notes.mapping_status != "pass"`
- `visual_curation_status != "ready"`
- `notes.scope_quality_status != "clean"`
- `notes.text_fidelity_status != "clean"`
- `notes.topic_trust_status != "normal"`
- non-empty `notes.review_flags`
- source-pairing mismatch between `notes.source_pdf` and `notes.mark_scheme_source_pdf`

### Student Practice Mode

Student practice should use a trusted image-first subset.

Suggested include:

- question image file exists
- mark-scheme image file exists
- source question paper and mark-scheme metadata agree
- `notes.mapping_status == "pass"`
- `notes.validation_status == "pass"`
- `notes.scope_quality_status == "clean"`
- `visual_curation_status == "ready"`

Suggested exclude:

- `notes.mapping_status == "fail"`
- `notes.validation_status == "fail"`
- `notes.scope_quality_status == "fail"`
- missing question or mark-scheme image file
- source-pairing mismatch
- `notes.mapping_failure_reason` is non-empty

Text can remain secondary in this mode. Render images to students and use text only for labels or search snippets when its trust allows it.

### Topic Search Mode

Topic search can use more records than student practice, but search results should carry trust badges.

Suggested strong include:

- `notes.topic_trust_status == "normal"`
- `notes.text_fidelity_status == "clean"`
- `question_text_trust in {"high", "medium"}`

Suggested review-only include:

- `notes.topic_trust_status == "degraded_text"`
- `notes.text_fidelity_status == "degraded"`
- `question_text_role == "search_hint"`

Suggested exclude from automatic topic filtering:

- `notes.topic_trust_status == "review_required"`
- `notes.text_fidelity_status == "unusable"`
- `question_text_trust == "unusable"`

### Future Adaptive Trainer Mode

Adaptive sequencing has the strictest requirements because wrong topic, difficulty, marks, or answer pairing can train the wrong skill.

Suggested include:

- Student practice filter passes.
- `notes.text_fidelity_status == "clean"` if text is used in prompts/search.
- `notes.topic_trust_status == "normal"`.
- `difficulty_score` exists.
- marks exist.
- subparts are trusted or empty by design.
- no source-pairing mismatch.
- enrichment sidecar either agrees or is not used for the adaptive decision.

Suggested exclude:

- all failed/review mapping or scope statuses
- all source-pairing mismatches
- all DeepSeek `final_review_required=True` records if DeepSeek metadata is used for routing
- records where local and DeepSeek topics mismatch unless reviewed

## Trust Tiers

Tier 1: image-ready

- question crop exists
- mark-scheme crop exists
- source pairing is correct
- mapping passes
- crop/scope status is acceptable
- text may still be secondary

Tier 2: metadata-ready

- Tier 1 passes
- text fidelity passes
- topic trust is acceptable
- marks and subparts are trusted

Tier 3: fully trusted practice item

- Tier 2 passes
- validation passes
- text-only status passes if text is used
- topic and difficulty metadata are stable
- reviewed or strong automated confidence exists

## Current Measured Tier Counts

Using a strict local filter on 2026-05-08:

- Current no-OCR `output/json/question_bank.json`: Tier 1 `616`, Tier 2 `567`, Tier 3 `150`.
- Frozen OCR `output/triage/iteration_004/baseline_question_bank.json`: Tier 1 `208`, Tier 2 `178`, Tier 3 `41`.

These counts are not exported yet. They are evidence for why a trusted-subset export should be built rather than asking downstream apps to infer the tier logic.

## Anti-Patterns

Do not:

- Treat `question_text` as the student-facing source of truth for math-heavy questions.
- Treat OCR text as canonical.
- Promote DeepSeek labels into canonical extraction fields.
- Use no-OCR comparison results as production OCR scores.
- Hide validation flags to increase ready counts.
- Load all records into a student practice app without filtering.
- Assume `visual_curation_status=ready` means text is student-ready.
- Assume `text_fidelity_status=clean` means the record has correct mark-scheme pairing.
- Treat an accepted auto-triage decision as permission to bypass record-level trust filters.
