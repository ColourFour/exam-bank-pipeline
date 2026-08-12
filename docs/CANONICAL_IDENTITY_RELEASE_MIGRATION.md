# Canonical Identity And Release Migration

This migration corrects the CAIE 9709 component contract:

- Paper 4: canonical family `mechanics`, course `m1`
- Paper 5 (2020+): canonical family `stats`, course `s1`
- Paper 6 (2020+): canonical family `stats`, course `s2`
- Paper 6 (pre-2020): canonical family `stats`, course `s1`
- Paper 5 (pre-2020): legacy M2, unsupported and fail-closed

It also restores the session contract: `mYY` maps to `springYY`, `sYY` maps to
`summerYY`, and `wYY` maps to `winterYY`. A mirror inversion affected 188 local
payload names from 2021–2025 (84 question papers, 84 mark schemes, 10 examiner
reports, and 10 grade-threshold files). The content-evidence ledger is
`manifests/migrations/caie_9709_corpus_session_identity.v1.json`.

The retained legacy bank contained 162 March questions across 19 papers from
2016–2020 under `summerYY` IDs. Run `exam-bank data migrate-session-identity`
before renaming source files; its alias manifest records source hashes and is
audit provenance only. Runtime joins must use corrected canonical IDs. Once the
full March and June corpus is restored, a legacy question ID can no longer be
rewritten safely without source evidence: the old collapsed ID may have been
used by either session. Image-backed review artifacts must therefore rebind by
their reviewed SHA-256, not by the alias string alone.

The pre-migration bank has 693 Paper 4 records stored as `stats` and 255 Paper
5 records stored as `mechanics`. Those 948 records require new family metadata,
canonical image paths, and asset IDs. The 414 Paper 6 records remain under the
`stats` storage family, but downstream course/topic metadata changes from the
former S1 collapse to explicit S2. In total, 1,362 records have identity or
course-facing migration impact.

Do not rename the entire `stats` or `mechanics` directory. `stats` legitimately
contains both P5 and P6, so migration must select records by component and
regenerate or move each bound question/mark-scheme asset together with its
metadata reference.

## Required Regeneration Order

1. Regenerate the canonical question bank and image assets with the corrected
   component contract.
2. Run `exam-bank extract integrity` and require zero missing, cross-question,
   absolute, or unresolved image references.
3. Regenerate the asset manifest with
   `exam-bank data build-asset-manifest`. Run
   `exam-bank data rebind-text-gold --write` to rebind verified text only
   to byte-identical current question images. Run
   `exam-bank data validate-review-assets` to validate source-skill,
   mark-event, and Content Lab review evidence. A changed reviewed image is a
   policy blocker: downstream consumers demote that evidence in memory instead
   of carrying the approval forward.
4. Run `exam-bank topic refresh-routing --write`. This writes the durable topic
   sidecar, compatibility checksum, and hash-bound question-bank release
   manifest. P6 records remain complete but review-only until a P6 taxonomy is
   approved.
5. Run `exam-bank topic verify-release` before rebuilding difficulty or
   Asterion projections and before publishing the new Question export.
6. Regenerate mark events, difficulty indexes, Asterion catalog/runtime/Content
   Lab projections, their release package, and
   `output/interchange/questions.v1.json`. These artifacts contain question IDs,
   family/course labels, paths, or source hashes and must not be carried across
   the identity migration. The sibling `../autograder` must independently
   rebuild rubric-readiness artifacts from the new versioned Question export;
   exam-bank no longer owns autograde eligibility.

The old durable topic sidecar, its `.sha256` file, any restored local copy, and
existing Asterion release provenance are intentionally invalid after the bank
changes. The release manifest must be generated only after bank and sidecar
membership agree exactly.
