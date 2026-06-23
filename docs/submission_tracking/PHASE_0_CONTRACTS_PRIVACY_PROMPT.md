# 2026-06-22 - Submission Tracking Phase 0 Prompt

Goal:
Add the contracts and privacy guardrails needed before building the local submission tracker.

Context:
The roadmap section `Automated Assignment Submission / Grading Assets` in `ROADMAP.md` defines the phased plan. The readiness review is `reports/AUTOMATED_GRADING_READINESS_REVIEW_2026_06_22.md`.

Phase 0 is documentation and guardrails only. It should make the private-data boundary explicit before any submission ingestion code exists.

Hard boundaries:

- Do not implement submission ingestion.
- Do not add email integration.
- Do not add Gmail, Outlook, SMTP, or IMAP code.
- Do not add OCR or grading.
- Do not add dependencies.
- Do not edit production extraction behavior.
- Do not touch Asterion exports or canonical question-bank generation.
- Do not create real student data.
- Do not create private submission folders except ignored placeholders if needed.

Required work:

1. Inspect the existing docs and ignore rules:
   - `ROADMAP.md`
   - `.gitignore`
   - `docs/AUTO_GRADING_CONTRACT.md`
   - `docs/OUTPUT_STORAGE_CONTRACT.md`
   - `docs/ASTERION_EXPORT_CONTRACT.md`
   - `reports/AUTOMATED_GRADING_READINESS_REVIEW_2026_06_22.md`, if present locally
2. Add `docs/SUBMISSION_TRACKING_CONTRACT.md`.
3. The contract must include:
   - purpose and scope
   - local-first v1 behavior
   - explicit non-goals
   - private storage roots
   - fixture rules
   - assignment, roster, submission, validation-result, completion-report, audit-log, and feedback-draft schema expectations
   - PDF validation expectations
   - duplicate and late-submission policy
   - draft-only email policy
   - audit-log policy
   - fail-closed behavior
   - v1 acceptance criteria
4. Confirm `.gitignore` protects private roots. Add missing protections only if needed. Suggested private roots:
   - `data/submissions/*`
   - `output/submissions/*`
   - `reports/submissions/*`
   - `data/rosters/*`
   - `output/feedback_drafts/*`
5. If placeholders are necessary, use `.gitkeep` only and ensure real private data remains ignored.
6. Do not add production Python modules in Phase 0.
7. If `README.md` or `ROADMAP.md` needs a small pointer to the contract, add it without bloating the docs.

Verification:

- Run `git diff --check`.
- Do not run the full test suite for docs/ignore-only changes unless the repo has an obvious docs check.
- If `.gitignore` changes, verify representative private paths are ignored with `git check-ignore`.

Expected final answer:

- files changed
- summary of the contract sections added
- ignored private roots confirmed or added
- verification commands run and results
- whether Phase 1 is ready to start
- suggested commit message

Suggested commit message:
`docs: add submission tracking contract`
