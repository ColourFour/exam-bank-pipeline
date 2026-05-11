# File Review Summary v1

Generated: 2026-05-11T05:18:24.394805+00:00

## What Was Found

- Relevant files inventoried: 39
- Canonical source candidates: 29
- JSON issue categories found: 3
- Review queue categories populated: 4

## What Is Usable

The component skill maps, question-skill mappings, topic filter maps, question-topic assignments, coverage reports, strict filtering reports, and root indexes are valid JSON and internally structured. The skill and topic IDs are component-scoped, and the topic maps are aligned to component-specific CAIE syllabus sections rather than a shared P3 template.

## What Is Broken Or Risky

No malformed taxonomy JSON was found, but most mapping records are machine candidates rather than reviewed records. Strict filters currently rely on high-confidence machine candidates. Root-level canonical-looking files, validation files, reports, and changelogs are mixed together, so the source of truth is ambiguous.

## What Should Be Kept

Keep the current skill maps, question-skill mappings, topic filter maps, question-topic assignments, component coverage reports, strict filtering reports, and index files as the best available v1 canonical candidates. Keep OCR/Asterion question banks and content-lab candidates as source references, not canonical taxonomy files.

## What Should Be Archived

After canonical copies are created under `exam_bank_taxonomy/canonical`, archive root-level taxonomy artifacts as superseded. Archive stale validation artifacts as deprecated logs. Do not delete any file silently.

## What Should Be Merged

No same-component duplicate taxonomy files with conflicting data were found. Merging is therefore limited to creating canonical indexes and review queues across the existing component files.

## What Should Be Regenerated

Regenerate validation reports and indexes after the new canonical folder structure is created. Generator scripts should eventually be updated to write into the canonical structure directly.

## What Should Not Be Trusted

Do not treat legacy `question.topic` labels, low-confidence records, whole-question-only mappings, prerequisite-only assignments, context-only assignments, or high-confidence machine candidates as human-reviewed evidence.

## Biggest Risks

- Machine candidates are eligible for strict filtering when confidence and evidence rules pass.
- Reviewed data is effectively absent in current mapping artifacts.
- Legacy broad topics remain useful as evidence but are not strict-filter safe.

## Recommended Next Steps

Run the optimization pass, validate the organized structure, then begin human review from the generated review queues before promoting any mapping to reviewed.
