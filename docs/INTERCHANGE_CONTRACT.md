# Repository interchange contract

The exam-bank, homework-ingest, and autograder repositories exchange JSON files
and referenced assets. They do not import one another's Python packages.

## Ownership and versioning

- Exam-bank owns `schemas/question.v1.schema.json` and
  `schemas/question-export.v1.schema.json`, and exports Question records in the
  versioned envelope.
- Homework-ingest owns Submission production; its repository keeps byte-for-byte
  copies of all four boundary schemas so intake and handoff validation remain
  local.
- Autograder consumes Question and Submission records and owns GradeResult
  production; its repository also keeps byte-for-byte copies of all four
  boundary schemas.

Every record carries a required self-identifier:

- Question: `exam_bank.interchange.question`, version `1`
- Submission: `homework_ingest.interchange.submission`, version `1`
- GradeResult: `autograder.interchange.grade_result`, version `1`

A breaking change creates a new versioned schema file and record version. Schema
copies are verified by SHA-256; no shared runtime package is required.

## Question export

Create the deterministic, question-ID-sorted handoff file with:

```console
exam-bank data export-questions \
  --input output/json/question_bank.json \
  --output output/interchange/questions.v1.json \
  --check-assets
```

Validate an existing handoff without rewriting it with:

```console
exam-bank data validate-questions \
  --input output/interchange/questions.v1.json \
  --check-assets
```

The envelope is `exam_bank.interchange.questions` version 1. It binds the source
question bank and authoritative Question schema by SHA-256, states an
`asset_root`, and carries a record count. Question and mark-scheme image paths
are POSIX relative paths resolved from that asset root. Consumers may relocate a
handoff by supplying their own local asset-root override.

Question IDs are opaque. Consumers preserve them exactly and do not recreate
paper, session, or component parsing. `question_number` is likewise transported
as a string so labels such as `3(a)` are not coerced.

## Rubric safety

Only explicit, reviewed source rubrics are promoted into `Question.rubric`.
Advisory mark-event extraction is not silently converted into grading authority.
An absent rubric is represented by an empty array and `rubric_status` of
`not_included`; downstream grading must fail closed or route the item for review.

Rubric events and GradeResult mark awards are arrays because printed mark codes
can repeat. `mark_id`, not `mark_code`, is the event identity.
