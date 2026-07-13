# Submission Draft Grading Salvage Audit - 2026-06-22

This targeted audit checked existing grading, rubric, mark-event, extraction, submission, report, and audit helpers before implementing Phase 3 draft automated grading. Phase 3 remains teacher-facing, draft-only, and fail-closed.

| File/module | What it does | Salvage decision | Why | Risk | Action taken |
|---|---|---:|---|---|---|
| `src/exam_bank/submissions/models.py` | Local submission, student, assignment, completion, and draft-message dataclasses | adapt_now | Existing dataclass style and JSON conversion fit Phase 3 records | Missing Phase 3 invariants would allow unsafe output flags | Added draft extraction/grading dataclasses with `grading_mode="draft_auto"`, `student_facing=false`, and `teacher_review_required=true` validation |
| `src/exam_bank/submissions/audit_log.py` | Append-only assignment JSONL audit helper | reuse_now | Compatible local private audit pattern | Audit entries can expose private text if callers pass it | Reused without passing extracted submission text |
| `src/exam_bank/submissions/review_queue.py` | Phase 2 teacher review queue and manual grading-prep records | reuse_now | Provides accepted-submission review records and manual placeholder IDs | Overwriting Phase 2 files would destroy teacher notes | Phase 3 reads these artifacts and writes only under `draft_grading/` |
| `src/exam_bank/submissions/validation.py` | PyMuPDF PDF open/encryption/page validation | adapt_now | Uses existing dependency and quiet MuPDF setup | Validation alone does not extract text | Added native text/page extraction in a separate Phase 3 module |
| `src/exam_bank/atomic_json.py` | Atomic JSON writer | reuse_now | Fits non-destructive generated JSON writes | None for Phase 3 outputs | Used for Phase 3 JSON artifacts |
| `src/exam_bank/auto_grade/schemas.py` | Reviewed-rubric model and loader | reuse_now | Safe as a reviewed-rubric gate | Rubrics alone do not prove student answer mapping | Used only to classify confidence and max-score availability; not used to assign scores |
| `src/exam_bank/auto_grade/reviewed_rubrics.py` | Rubric validation and review queue tooling | reference_only | Useful contract: approved rubrics require human review and canonical evidence | Its broader auto-grade flow targets question-bank readiness, not submissions | Referenced contract; not invoked by Phase 3 submission grading |
| `src/exam_bank/auto_grade/eligibility.py` | Question-bank auto-grade eligibility from canonical artifacts, reviewed rubrics, and advisory sidecars | reference_only | Confirms advisory mark events are not scoring authority | Eligibility statuses are not per-student grading results | Mirrored the fail-closed convention |
| `src/exam_bank/auto_grade/reviewer_packet.py` | Teacher/reviewer packet generation patterns | reference_only | Useful teacher-facing packet shape and warnings | Candidate packets include advisory evidence text unsuitable for student submissions | Implemented a new private submission packet without feedback text |
| `src/exam_bank/mark_events/*` | Generated advisory mark-event extraction and validation | do_not_use | Mark events are advisory only and not student-answer evidence | Using them for marks would violate Phase 3 | Phase 3 records `advisory_mark_events_not_scoring_contract` and assigns no score |
| `src/exam_bank/pdf_extract.py` and `src/exam_bank/ocr.py` | Canonical paper layout extraction and OCR routing | reference_only | Native extraction concepts are relevant | OCR/text-selection logic is tuned for exam papers, not private submissions | Phase 3 uses direct native text extraction only; OCR is not run |
| `src/exam_bank/submissions/feedback_drafts.py` | Draft acknowledgement/resend/reminder messages | do_not_use | Produces message drafts, not grading artifacts | Student-facing feedback boundary | Not used by Phase 3 |
| `scripts/build_auto_grade_*`, `scripts/validate_auto_grade_*` | Question-bank rubric/eligibility workflows | reference_only | Helpful safety precedent | Not submission-specific | Left unchanged |
| `tests/fixtures/submissions/*` | Synthetic roster, assignment, and tiny PDFs | reuse_now | Fake data only, reserved email domain | Existing PDFs have no native text | Reused for fail-closed tests; generated native-text PDFs inside tests |

## Outcome

Salvaged implementation pieces were limited to local audit logging, atomic JSON writing, reviewed-rubric parsing, Phase 2 artifact shapes, and PyMuPDF-native PDF inspection. No legacy code that sends messages, produces student-facing feedback, assigns final scores, uses OCR as scoring evidence, or treats advisory mark events as scoring authority was reused.
