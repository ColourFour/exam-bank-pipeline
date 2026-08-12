# Interchange schemas

These Draft 2020-12 JSON Schemas are the file boundary between the three local
repositories. They are contracts, not a shared runtime package.

- `question.v1.schema.json` is authoritative in `exam-bank-pipeline`.
- `submission.v1.schema.json` is a compatibility copy whose owning producer is
  `homework-ingest`.
- `grade-result.v1.schema.json` is a compatibility copy whose owning producer is
  `autograder`.
- `question-export.v1.schema.json` defines the exam-bank Question export
  envelope.

Consumer repositories copy the applicable files byte-for-byte and record their
SHA-256. A breaking change requires a new versioned filename and `$id`; do not
silently edit a deployed v1 contract. Question IDs are opaque strings. A
consumer must preserve them exactly instead of recreating paper/session logic.
Every standalone record is also self-identifying: `schema_name` and
`schema_version` are required constants, even when the record is carried in an
envelope.

The installed `exam_bank` package contains a byte-identical copy of
`question.v1.schema.json` so export validation does not depend on a source-tree
checkout. The top-level file remains authoritative; tests prevent the packaged
copy from drifting.

Question and submission asset paths are POSIX-style relative paths. In a
Question export they resolve from the envelope's `asset_root`. A grading command
may receive an explicit local asset-root override, but it must not import the
exam-bank output-layout package.

`rubric` and `marks` are arrays because mark codes such as `M1` or `A1` can
repeat within a question. `mark_id`, not `mark_code`, is the per-event identity.
