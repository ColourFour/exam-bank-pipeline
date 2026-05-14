Advisory Evidence Phase — Examiner Reports + Grade Thresholds

We are working in:

/Users/sbrooker/repos/exam-bank-pipeline

The project is image-first. Canonical question images and mark-scheme images remain the source of truth. Examiner reports and grade thresholds are advisory evidence only.

This phase has two main goals:

1. Extract usable text/data from PDFs in:

input/examiner_reports

input/grade_thresholds

2. Use that extracted evidence to support topic and difficulty signals for exam items.

This must be staged carefully. Do not jump straight to AI. Do not mutate canonical outputs. Build deterministic extraction, parsing, linking, validation, and review first.

Global constraints for every goal:

Do not mutate:

output/json/question_bank.json

output/json/question_bank.topic_routing.v1.json

output/asterion/exports/latest/*

output/p1

output/p3

output/p4

output/p5

exam_bank_taxonomy/*

canonical question image folders

canonical mark-scheme image folders

AI/DeepSeek sidecars

Do not enable strict topic filtering from this evidence.

Do not treat examiner reports as canonical student-facing content.

Do not treat grade thresholds as direct individual-question difficulty.

Do not use AI unless the specific goal explicitly allows it.

Every implementation pass must end with this final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 1 — Build deterministic advisory input inventory

/goal

Build a deterministic inventory for advisory evidence input PDFs.

Scan:

input/examiner_reports

input/grade_thresholds

Create inventory files showing every PDF found, inferred metadata, page count, parse-readiness, and warnings. Do not extract meaning yet. This goal is only about proving the repo can see the files and classify them.

Target outputs:

output/advisory_evidence/inventory/examiner_report_inventory.json

output/advisory_evidence/inventory/grade_threshold_inventory.json

Recommended schema fields:

{
  "schema": "exam_bank.advisory_evidence.inventory.v1",
  "generated_at": "...",
  "source_dir": "input/examiner_reports",
  "document_count": 0,
  "documents": [
    {
      "source_path": "input/examiner_reports/9709 Mathematics June 2025 Examiner Report.pdf",
      "filename": "9709 Mathematics June 2025 Examiner Report.pdf",
      "syllabus": "9709",
      "year": 2025,
      "session": "june",
      "document_type": "examiner_report",
      "page_count": 0,
      "file_size_bytes": 0,
      "can_open": true,
      "can_extract_native_text": true,
      "warnings": []
    }
  ]
}

Likely files/modules to inspect or create:

src/exam_bank/

src/exam_bank/advisory_evidence/

src/exam_bank/advisory_evidence/inventory.py

scripts/build_advisory_inventory.py

tests/test_advisory_inventory.py

Use existing project conventions for scripts, output writing, and tests.

Exact targets:

The inventory must detect all PDFs under both input folders.

The inventory must infer:

syllabus

year

session

document type

page count

source path

filename

file size

whether PDF can be opened

whether native text appears extractable

warnings

Session normalization should be stable, for example:

June 2025 → june_2025

November 2023 → november_2023

March 2021 → march_2021

Document type normalization should be stable:

examiner report → examiner_report

grade thresholds → grade_thresholds

Validation commands:

.venv/bin/python scripts/build_advisory_inventory.py --help

.venv/bin/python scripts/build_advisory_inventory.py --dry-run

.venv/bin/python scripts/build_advisory_inventory.py

.venv/bin/python -m pytest tests/test_advisory_inventory.py

Tests to add/update:

Test that known filenames infer correct syllabus/session/year/document type.

Test that inventory output includes schema version.

Test that PDFs with unreadable metadata produce warnings, not crashes.

Test that missing input folders produce empty inventory with warnings, not pipeline failure.

Test that --dry-run performs detection without writing output.

Test that --help exits successfully without reading/writing files.

Acceptance criteria:

Inventory files are generated deterministically.

Every discovered PDF appears exactly once.

Each inventory record includes normalized metadata.

Page counts are populated when PDFs can be opened.

Unreadable PDFs are represented with warnings, not hard failure.

No canonical question-bank output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 2 — Extract raw native PDF text safely

/goal

Create a safe raw-text extraction pipeline for examiner reports and grade thresholds.

Start with native PDF text only. Do not add OCR fallback yet. The purpose is to capture what the PDF already exposes as text and record extraction quality.

Target outputs:

output/advisory_evidence/extracted_text/examiner_reports/*.json

output/advisory_evidence/extracted_text/grade_thresholds/*.json

Recommended examiner report output shape:

{
  "schema": "exam_bank.advisory_evidence.extracted_text.v1",
  "source_path": "input/examiner_reports/9709 Mathematics June 2025 Examiner Report.pdf",
  "syllabus": "9709",
  "session": "june_2025",
  "document_type": "examiner_report",
  "page_count": 0,
  "extraction_method": "native_pdf_text",
  "text_length": 0,
  "raw_text": "",
  "page_text": [
    {
      "page_number": 1,
      "text": "",
      "text_length": 0
    }
  ],
  "detected_headers": [],
  "warnings": []
}

Grade threshold extraction can use the same shape for now, but must preserve page text because table parsing will happen later.

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/pdf_text.py

src/exam_bank/advisory_evidence/extraction.py

scripts/extract_advisory_text.py

tests/test_advisory_text_extraction.py

Exact targets:

Read inventory JSONs from Goal 1.

Extract native text from every inventoried PDF that can be opened.

Write one JSON per source PDF.

Preserve page-level text.

Record extraction method as native_pdf_text.

Record warnings for low text length, empty pages, or extraction failures.

Do not OCR.

Do not parse meaning.

Do not link to question-bank records yet.

Validation commands:

.venv/bin/python scripts/extract_advisory_text.py --help

.venv/bin/python scripts/extract_advisory_text.py --dry-run

.venv/bin/python scripts/extract_advisory_text.py

.venv/bin/python -m pytest tests/test_advisory_text_extraction.py

Tests to add/update:

Test that extraction creates one JSON per inventory document.

Test that page count matches inventory page count where available.

Test that raw_text equals or contains combined page text.

Test that empty extraction produces warnings, not crash.

Test that --dry-run does not write extracted-text files.

Test that canonical outputs remain untouched.

Acceptance criteria:

Raw text extraction works for both examiner reports and grade threshold PDFs.

Every output includes schema version, source path, page count, method, text length, page text, warnings.

Low text coverage is marked as warning only.

No OCR fallback is introduced.

No canonical question-bank output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 3 — Parse examiner reports into structured sections

/goal

Parse extracted examiner report text into structured report sections.

Turn raw examiner report text into:

session → component → section type → question comment → detectable part/comment fields if available.

Do not predict topics yet. Do not link to question-bank records yet. This goal is only about structure.

Target outputs:

output/advisory_evidence/parsed/examiner_reports/*.json

Recommended schema:

{
  "schema": "exam_bank.advisory_evidence.examiner_report_parsed.v1",
  "syllabus": "9709",
  "session": "june_2025",
  "document_type": "examiner_report",
  "source_path": "input/examiner_reports/9709 Mathematics June 2025 Examiner Report.pdf",
  "components": [
    {
      "component": "31",
      "paper_title": "Pure Mathematics 3",
      "section_headers": [],
      "key_messages": [],
      "general_comments": "",
      "questions": [
        {
          "question_number": 1,
          "parts": [],
          "comment_text": "",
          "evidence_level": "normal",
          "warnings": []
        }
      ],
      "warnings": []
    }
  ],
  "warnings": []
}

Important handling:

Examiner reports may include component sections like Paper 9709/11, Paper 9709/31, etc.

Question comments may appear as Question 1, Question 2, etc.

Some sections may include key messages and general comments without useful question-level evidence.

Some sections may say there were too few candidates for a meaningful report. These must be parsed but marked as low/no evidence.

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/examiner_reports.py

scripts/parse_examiner_reports.py

tests/test_examiner_report_parsing.py

Possible fixtures:

small extracted text fixture from June 2025 examiner report

small extracted text fixture from March 2021 examiner report with “too few candidates” language

Exact targets:

Detect component sections.

Detect paper titles where available.

Extract key messages.

Extract general comments.

Extract question comments.

Preserve comments as raw advisory text.

Mark low/no-evidence sections instead of forcing question evidence.

Do not infer topics yet.

Do not infer difficulty yet.

Validation commands:

.venv/bin/python scripts/parse_examiner_reports.py --help

.venv/bin/python scripts/parse_examiner_reports.py --dry-run

.venv/bin/python scripts/parse_examiner_reports.py

.venv/bin/python -m pytest tests/test_examiner_report_parsing.py

Tests to add/update:

Test component header parsing for 9709/31.

Test question comment parsing for numbered question sections.

Test key message extraction.

Test general comment extraction.

Test “too few candidates for a meaningful report” produces evidence_level: "none" or "low".

Test parser warnings are recorded for malformed or missing sections.

Test no question-bank files are modified.

Acceptance criteria:

Parsed examiner report JSONs are produced.

Components are detected.

Question-level comments are detected where present.

Low/no-evidence sections are not forced into fake question evidence.

Raw comment text is preserved.

No canonical question-bank output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 4 — Parse grade thresholds into structured tables

/goal

Build a deterministic parser for grade threshold PDFs using extracted native text.

The parser must capture both component-level thresholds and option-level thresholds.

Target outputs:

output/advisory_evidence/parsed/grade_thresholds/*.json

Recommended schema:

{
  "schema": "exam_bank.advisory_evidence.grade_thresholds_parsed.v1",
  "syllabus": "9709",
  "session": "june_2024",
  "document_type": "grade_thresholds",
  "source_path": "input/grade_thresholds/9709 Mathematics June 2024 Grade Thresholds.pdf",
  "components": [
    {
      "component": "31",
      "max_raw_mark": 75,
      "thresholds": {
        "A": 48,
        "B": 39,
        "C": 31,
        "D": 23,
        "E": 14
      },
      "warnings": []
    }
  ],
  "options": [
    {
      "option": "AX",
      "max_weighted_mark": 250,
      "components": ["11", "31", "41", "51"],
      "thresholds": {
        "A*": 199,
        "A": 174,
        "B": 149,
        "C": 116,
        "D": 83,
        "E": 51
      },
      "warnings": []
    }
  ],
  "warnings": []
}

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/grade_thresholds.py

scripts/parse_grade_thresholds.py

tests/test_grade_threshold_parsing.py

Exact targets:

Parse component rows such as:

11, 12, 13, 31, 32, 33, 41, 42, 43, 51, 52, 53, etc.

Capture:

component

max raw mark

A/B/C/D/E thresholds

Parse option rows such as:

AX, AY, AZ

AS-only options such as S1–S9, if present.

Capture:

option code

max weighted mark

included components

A*/A/B/C/D/E thresholds where available

Do not calculate difficulty yet.

Do not assign question-level difficulty.

Validation commands:

.venv/bin/python scripts/parse_grade_thresholds.py --help

.venv/bin/python scripts/parse_grade_thresholds.py --dry-run

.venv/bin/python scripts/parse_grade_thresholds.py

.venv/bin/python -m pytest tests/test_grade_threshold_parsing.py

Tests to add/update:

Test component table parsing with known row examples.

Test option table parsing with known row examples.

Test thresholds are integers.

Test max marks are integers.

Test malformed rows produce warnings, not crashes.

Test missing table sections produce warnings.

Test grade threshold parser does not emit question-level difficulty.

Acceptance criteria:

Parsed grade threshold JSONs are produced.

Component thresholds are captured.

Option thresholds are captured where present.

Rows are deterministic and stable.

Malformed rows are warned, not silently ignored.

No question-level difficulty is created.

No canonical question-bank output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 5 — Create stable linking IDs and link audits

/goal

Create stable linking between parsed advisory evidence and existing question-bank records.

Do not create predictions yet. This goal produces link audits only.

Read:

output/json/question_bank.json

output/advisory_evidence/parsed/examiner_reports/*.json

output/advisory_evidence/parsed/grade_thresholds/*.json

Target outputs:

output/advisory_evidence/linking/examiner_report_question_links.json

output/advisory_evidence/linking/grade_threshold_component_links.json

Recommended normalized advisory key:

9709_{year}_{session}_{component}_q{number}

Examples:

9709_2025_june_31_q06

9709_2023_november_12_q02

The project may already use IDs like:

33summer25_q06

Do not replace existing IDs. Build a bridge between advisory normalized keys and current question-bank IDs.

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/linking.py

scripts/link_advisory_evidence.py

tests/test_advisory_linking.py

Exact targets:

For examiner reports:

map report component/question → question-bank record candidates.

For grade thresholds:

map session/component → all question-bank records from that paper/component/session.

Each link record should include:

source document

syllabus

session

year

component

question number, if applicable

normalized advisory key

candidate question-bank IDs

match status: linked, ambiguous, unlinked, not_applicable

warnings

Recommended examiner link shape:

{
  "schema": "exam_bank.advisory_evidence.examiner_report_links.v1",
  "links": [
    {
      "normalized_key": "9709_2025_june_31_q06",
      "source_path": "...",
      "component": "31",
      "question_number": 6,
      "candidate_question_ids": ["31summer25_q06"],
      "match_status": "linked",
      "warnings": []
    }
  ]
}

Validation commands:

.venv/bin/python scripts/link_advisory_evidence.py --help

.venv/bin/python scripts/link_advisory_evidence.py --dry-run

.venv/bin/python scripts/link_advisory_evidence.py

.venv/bin/python -m pytest tests/test_advisory_linking.py

Tests to add/update:

Test normalized key generation.

Test examiner report component/question links to existing bank record.

Test missing question produces unlinked, not crash.

Test multiple candidates produce ambiguous.

Test grade threshold component/session links to all matching question records.

Test every linked candidate exists in question_bank.json.

Test linking script does not mutate question bank.

Acceptance criteria:

Link audit files are generated.

Every linked question ID exists in question_bank.json.

Ambiguous and unlinked cases are explicit.

Grade threshold links are component/session-level only.

No predictions are created.

No canonical question-bank output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 6 — Build advisory evidence sidecar skeleton

/goal

Create the first advisory evidence sidecar.

This sidecar should attach available examiner-report and grade-threshold evidence to question-bank records without making topic or difficulty predictions yet.

Target output:

output/json/question_bank.advisory_evidence.v1.json

Recommended top-level schema:

{
  "schema": "exam_bank.question_bank.advisory_evidence.v1",
  "generated_at": "...",
  "source_question_bank": "output/json/question_bank.json",
  "records_count": 0,
  "records": [
    {
      "question_id": "33summer25_q06",
      "advisory_evidence": {
        "examiner_report": {
          "available": true,
          "component": "33",
          "question_number": 6,
          "comment_text": "...",
          "evidence_level": "normal",
          "confidence": "medium",
          "source_path": "...",
          "warnings": []
        },
        "grade_threshold": {
          "available": true,
          "component": "33",
          "component_max_raw": 75,
          "thresholds": {
            "A": 54,
            "B": 45,
            "C": 36,
            "D": 27,
            "E": 18
          },
          "source_path": "...",
          "warnings": []
        }
      }
    }
  ],
  "warnings": []
}

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/sidecar.py

scripts/build_advisory_evidence_sidecar.py

tests/test_advisory_sidecar.py

Exact targets:

Create one advisory record per question-bank record, or only records with evidence if that better matches project convention. Whichever choice is made, document it in code/tests.

Attach examiner report comment where linked.

Attach grade threshold component context where linked.

Do not create predicted topics yet.

Do not create difficulty labels yet.

Do not alter canonical question-bank records.

Validation commands:

.venv/bin/python scripts/build_advisory_evidence_sidecar.py --help

.venv/bin/python scripts/build_advisory_evidence_sidecar.py --dry-run

.venv/bin/python scripts/build_advisory_evidence_sidecar.py

.venv/bin/python -m pytest tests/test_advisory_sidecar.py

Tests to add/update:

Test sidecar includes schema version.

Test every advisory question_id exists in question_bank.json.

Test examiner report evidence attaches only through link audit.

Test grade threshold evidence attaches only by component/session link.

Test missing evidence produces available: false or omitted evidence consistently.

Test sidecar does not contain canonical replacement fields.

Test sidecar does not modify question bank.

Acceptance criteria:

Sidecar builds deterministically.

Sidecar is advisory-only.

Every advisory record links to a real question-bank record.

Examiner report evidence and grade threshold evidence remain separate.

No topic or difficulty predictions are created yet.

No canonical output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 7 — Add deterministic topic evidence from examiner reports

/goal

Add rule-based topic evidence from parsed examiner report comments.

Do not use AI. Do not create canonical topic truth. This goal should produce advisory topic evidence only.

Read:

output/json/question_bank.advisory_evidence.v1.json

existing topic/taxonomy maps used by the project

parsed examiner report comments

Target output:

updated output/json/question_bank.advisory_evidence.v1.json

or, if cleaner for staging:

output/advisory_evidence/predictions/advisory_topic_evidence.v1.json

Use whichever pattern fits existing project conventions, but do not mutate canonical topic routing.

Recommended topic evidence shape:

{
  "question_id": "31summer25_q06",
  "topic_evidence": {
    "predicted_topic_ids": ["p3_complex_numbers"],
    "matched_terms": ["complex numbers in polar form"],
    "method": "rule_match_v1",
    "confidence": "high",
    "review_required": false,
    "warnings": []
  }
}

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/topic_evidence.py

src/exam_bank/advisory_evidence/topic_terms.py

scripts/build_advisory_topic_evidence.py

tests/test_advisory_topic_evidence.py

Exact targets:

Create deterministic phrase-to-topic matching.

Use only allowed existing topic IDs.

Support phrases such as:

complex numbers in polar form

vectors

integration by parts

numerical methods

volume of revolution

implicit differentiation

differential equations

binomial expansion

trigonometric equations

partial fractions

parametric equations

iteration

Do not infer precise topics from vague language only.

For example:

“algebraic errors” should become misconception/method evidence, not precise topic evidence.

“challenging” should not become topic evidence.

Validation commands:

.venv/bin/python scripts/build_advisory_topic_evidence.py --help

.venv/bin/python scripts/build_advisory_topic_evidence.py --dry-run

.venv/bin/python scripts/build_advisory_topic_evidence.py

.venv/bin/python -m pytest tests/test_advisory_topic_evidence.py

Tests to add/update:

Test known phrase maps to allowed topic ID.

Test invalid topic ID in matcher table fails validation.

Test vague phrase does not create topic prediction.

Test multiple matched terms can produce multiple advisory topic IDs.

Test topic evidence includes method and confidence.

Test topic evidence does not overwrite question_bank.topic_routing.v1.json.

Acceptance criteria:

Rule-based topic evidence is generated.

All predicted topic IDs are validated against allowed topic IDs.

Vague language does not produce fake precision.

Evidence includes matched terms.

Evidence is advisory and reviewable.

Canonical topic routing is unchanged.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 8 — Add examiner-report difficulty evidence

/goal

Extract cautious item-level difficulty signals from examiner report comments.

This should only use examiner report text. Do not use grade thresholds in this goal. Do not create final difficulty predictions yet.

Target output:

output/advisory_evidence/predictions/advisory_examiner_report_difficulty.v1.json

or updated advisory sidecar if that is the chosen project pattern.

Recommended shape:

{
  "question_id": "31summer25_q06",
  "examiner_report_difficulty": {
    "item_signal": "hard",
    "matched_terms": [
      "commonly gained no response",
      "few fully correct answers"
    ],
    "difficulty_reasons": [
      "question commonly gained no response"
    ],
    "method": "examiner_report_phrase_rules_v1",
    "confidence": "medium",
    "review_required": true,
    "warnings": []
  }
}

Allowed item signals:

easy

moderate

hard

mixed

unknown

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/difficulty_terms.py

src/exam_bank/advisory_evidence/examiner_difficulty.py

scripts/build_examiner_report_difficulty.py

tests/test_examiner_report_difficulty.py

Exact targets:

Detect positive/easier phrases, such as:

well answered

accessible

straightforward

most candidates

many correct

Detect harder phrases, such as:

challenging

few correct solutions

many omitted

high proportion blank

good discriminator

only strongest candidates

proved difficult

commonly gained no response

Detect mixed phrases where report includes both success and common errors.

Do not overstate confidence.

Do not infer difficulty from topic alone.

Do not use grade thresholds.

Validation commands:

.venv/bin/python scripts/build_examiner_report_difficulty.py --help

.venv/bin/python scripts/build_examiner_report_difficulty.py --dry-run

.venv/bin/python scripts/build_examiner_report_difficulty.py

.venv/bin/python -m pytest tests/test_examiner_report_difficulty.py

Tests to add/update:

Test hard phrase produces item_signal: "hard".

Test easy phrase produces item_signal: "easy" or "moderate" depending wording.

Test mixed wording produces item_signal: "mixed".

Test vague comments produce item_signal: "unknown".

Test “too few candidates” produces low/no evidence.

Test grade thresholds are not read or used.

Acceptance criteria:

Examiner report difficulty evidence is generated.

Evidence includes matched phrases and reasons.

Signals are cautious and reviewable.

No grade threshold data contributes to item-level difficulty.

No canonical output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 9 — Add grade-threshold paper/component difficulty context

/goal

Create paper/session/component-level difficulty context from grade thresholds.

Grade thresholds must not assign direct individual-question difficulty. This goal should calculate component-level context only.

Target output:

output/advisory_evidence/predictions/advisory_grade_threshold_context.v1.json

Recommended shape:

{
  "schema": "exam_bank.advisory_evidence.grade_threshold_context.v1",
  "contexts": [
    {
      "syllabus": "9709",
      "session": "june_2024",
      "component": "31",
      "max_raw_mark": 75,
      "threshold_ratios": {
        "A": 0.64,
        "B": 0.52,
        "C": 0.41,
        "D": 0.31,
        "E": 0.19
      },
      "component_context_label": "paper_context_harder_than_session_peers",
      "comparison_basis": [
        "same_session",
        "same_component_family"
      ],
      "confidence": "medium",
      "warnings": []
    }
  ]
}

Allowed context labels:

paper_context_harder_than_session_peers

paper_context_typical

paper_context_easier_than_session_peers

paper_context_unknown

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/threshold_context.py

scripts/build_grade_threshold_context.py

tests/test_grade_threshold_context.py

Exact targets:

Calculate:

A threshold ratio = A / max raw mark

B threshold ratio

C threshold ratio

D threshold ratio

E threshold ratio

Compare components within same session.

Compare components within same component family where reasonable, for example:

P1 family: 11/12/13

P3 family: 31/32/33

P4 family: 41/42/43

P5 family: 51/52/53

Do not label individual questions hard/easy.

Do not blend threshold context into examiner-report item signals yet.

Validation commands:

.venv/bin/python scripts/build_grade_threshold_context.py --help

.venv/bin/python scripts/build_grade_threshold_context.py --dry-run

.venv/bin/python scripts/build_grade_threshold_context.py

.venv/bin/python -m pytest tests/test_grade_threshold_context.py

Tests to add/update:

Test threshold ratio calculation.

Test component family comparison.

Test lower A ratio does not produce question-level difficulty.

Test missing A threshold produces paper_context_unknown.

Test zero or invalid max mark produces warning.

Test context labels are from allowed enum only.

Acceptance criteria:

Component-level threshold context is generated.

Ratios are calculated correctly.

Comparison basis is explicit.

No individual-question difficulty is produced.

No canonical output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 10 — Combine advisory topic and difficulty predictions

/goal

Combine advisory evidence into topic and difficulty prediction records.

This is still advisory only. Do not overwrite canonical topic routing. Do not enable strict filtering. Do not produce mastery decisions.

Inputs may include:

existing question-bank metadata

existing topic routing sidecar, read-only

trusted extracted question text hints, if already available and confidence-gated

examiner report topic evidence

examiner report difficulty evidence

grade threshold component context

Target output:

output/advisory_evidence/predictions/question_bank.advisory_predictions.v1.json

or integrated sidecar output if the project has settled on a single sidecar.

Recommended shape:

{
  "schema": "exam_bank.question_bank.advisory_predictions.v1",
  "records": [
    {
      "question_id": "31summer25_q06",
      "topic_prediction": {
        "predicted_topic_ids": [],
        "evidence_sources": [
          "examiner_report",
          "existing_topic_routing",
          "question_text_hint"
        ],
        "confidence": "medium",
        "review_required": true,
        "warnings": []
      },
      "difficulty_prediction": {
        "item_signal": "hard",
        "paper_context": "paper_context_harder_than_session_peers",
        "confidence": "medium",
        "review_required": true,
        "warnings": []
      }
    }
  ]
}

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/predictions.py

scripts/build_advisory_predictions.py

tests/test_advisory_predictions.py

Exact targets:

Keep topic prediction and difficulty prediction separate.

Difficulty must separate:

examiner-report item signal

grade-threshold paper/component context

combined advisory confidence

Topic prediction should include evidence sources and matched evidence.

Any conflict between existing routing and examiner report evidence should set review_required: true.

Any prediction with only weak evidence should set review_required: true.

Grade threshold context may influence confidence/context, but must not create item_signal: hard/easy by itself.

Validation commands:

.venv/bin/python scripts/build_advisory_predictions.py --help

.venv/bin/python scripts/build_advisory_predictions.py --dry-run

.venv/bin/python scripts/build_advisory_predictions.py

.venv/bin/python -m pytest tests/test_advisory_predictions.py

Tests to add/update:

Test topic and difficulty predictions are separate.

Test grade threshold context alone cannot create item difficulty.

Test conflict between evidence sources requires review.

Test invalid topic ID fails.

Test prediction record question IDs exist in question bank.

Test no canonical outputs are modified.

Acceptance criteria:

Advisory predictions are generated.

Topic and difficulty are separate.

Predictions include evidence sources.

Review-required logic is conservative.

Grade thresholds remain paper/component context only.

Canonical routing and question bank are unchanged.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 11 — Add advisory validation gates

/goal

Add validation gates for advisory evidence outputs.

Before advisory evidence can be consumed by Asterion or Content Lab, tests must enforce that it is linked, valid, advisory-only, and non-mutating.

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/validation.py

scripts/validate_advisory_evidence.py

tests/test_advisory_validation.py

Exact targets:

Validation must enforce:

Every advisory record links to a real question-bank record.

Every predicted topic ID is in the allowed topic/taxonomy map.

No advisory evidence overwrites canonical topic routing.

No grade threshold evidence is assigned directly as individual-question difficulty.

Missing examiner reports do not fail the pipeline.

Missing grade thresholds do not fail the pipeline.

“Too few candidates for meaningful report” sections are marked low/no evidence.

All advisory outputs include schema version.

All confidence labels come from allowed enums.

All context labels come from allowed enums.

Validation commands:

.venv/bin/python scripts/validate_advisory_evidence.py --help

.venv/bin/python scripts/validate_advisory_evidence.py

.venv/bin/python -m pytest tests/test_advisory_validation.py

Tests to add/update:

Test orphan advisory question ID fails validation.

Test invalid topic ID fails validation.

Test item difficulty created only from grade thresholds fails validation.

Test missing report input produces warning, not failure.

Test missing threshold input produces warning, not failure.

Test low/no-evidence report sections are allowed if explicitly marked.

Test schema missing fails validation.

Acceptance criteria:

Validation script exists and passes.

Invalid advisory data fails loudly.

Missing advisory sources warn but do not break canonical pipeline.

Validation proves advisory outputs remain separate from canonical outputs.

No canonical output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 12 — Create human-readable advisory review reports

/goal

Create Markdown review reports for advisory evidence extraction, linking, topic evidence, and difficulty evidence.

The reports should help a human quickly inspect whether this evidence is useful and where it is weak.

Target outputs:

output/advisory_evidence/reports/examiner_report_extraction_status.md

output/advisory_evidence/reports/grade_threshold_extraction_status.md

output/advisory_evidence/reports/advisory_topic_prediction_review.md

output/advisory_evidence/reports/advisory_difficulty_prediction_review.md

output/advisory_evidence/reports/unlinked_examiner_report_entries.md

output/advisory_evidence/reports/low_confidence_predictions.md

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/reports.py

scripts/build_advisory_review_reports.py

tests/test_advisory_review_reports.py

Reports should answer:

How many examiner reports were inventoried?

How many examiner reports extracted text successfully?

How many components were detected?

How many question comments were detected?

How many report question comments linked to bank records?

How many entries were unlinked or ambiguous?

How many grade threshold PDFs parsed?

How many component threshold rows parsed?

How many option threshold rows parsed?

How many topic predictions were high/medium/low confidence?

How many difficulty predictions were based on examiner report evidence?

How many records only had grade-threshold context?

How many predictions require review?

Validation commands:

.venv/bin/python scripts/build_advisory_review_reports.py --help

.venv/bin/python scripts/build_advisory_review_reports.py --dry-run

.venv/bin/python scripts/build_advisory_review_reports.py

.venv/bin/python -m pytest tests/test_advisory_review_reports.py

Tests to add/update:

Test report files are created.

Test report includes key counts.

Test unlinked report lists unlinked entries.

Test low-confidence report lists low-confidence predictions.

Test empty evidence still creates reports with zero counts.

Test reports do not require canonical output mutation.

Acceptance criteria:

Human-readable reports are generated.

Reports include useful counts and review queues.

Reports make weak/unlinked evidence visible.

Reports do not present advisory predictions as truth.

No canonical output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 13 — Add AI-assisted enrichment only for unresolved advisory cases

/goal

Add optional AI-assisted enrichment only after deterministic extraction, parsing, linking, rule-based topic evidence, difficulty evidence, validation, and review reports exist.

This goal should be optional and role-gated. AI should only run on unresolved or low-confidence cases.

Do not use AI to write canonical outputs.

Do not use AI to invent new topics.

Do not use AI to mark question text student-facing-safe.

Do not use AI to bypass validation.

Target output:

output/advisory_evidence/ai_suggestions/advisory_ai_suggestions.v1.json

Recommended shape:

{
  "schema": "exam_bank.advisory_evidence.ai_suggestions.v1",
  "records": [
    {
      "question_id": "31summer25_q06",
      "input_sources": [
        "examiner_report_comment",
        "allowed_topic_ids",
        "question_metadata"
      ],
      "suggested_topic_ids": [],
      "suggested_difficulty_signal": "hard",
      "rationale": "",
      "confidence": "low",
      "review_required": true,
      "model": "",
      "warnings": []
    }
  ]
}

Likely files/modules to inspect or create:

src/exam_bank/advisory_evidence/ai_enrichment.py

scripts/build_advisory_ai_suggestions.py

tests/test_advisory_ai_enrichment.py

Exact targets:

AI input must include:

examiner report question comment

allowed topic/subtopic/skill list

current question metadata

trusted extracted question text only if already confidence-gated

AI output must be structured.

AI output must be advisory.

AI output must validate against allowed topic IDs.

AI output must always set review_required: true.

AI should only process unresolved/low-confidence cases by default.

Validation commands:

.venv/bin/python scripts/build_advisory_ai_suggestions.py --help

.venv/bin/python scripts/build_advisory_ai_suggestions.py --dry-run

.venv/bin/python -m pytest tests/test_advisory_ai_enrichment.py

Do not require live AI calls in normal tests. Mock AI responses.

Tests to add/update:

Test AI suggestions reject invented topic IDs.

Test AI suggestions always require review.

Test AI suggestions cannot modify canonical outputs.

Test AI input builder includes allowed topic list.

Test unresolved-only filtering works.

Test mocked AI malformed response produces warning/failure, not silent acceptance.

Acceptance criteria:

AI enrichment is optional.

AI only runs on unresolved/low-confidence cases.

AI output is structured, advisory, and review-required.

AI cannot invent topic IDs.

AI cannot mutate canonical outputs.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Goal 14 — Document advisory sidecar contract and usage rules

/goal

Document the final advisory evidence contract.

This goal should explain the schema, usage rules, safety boundaries, and intended downstream consumers.

Target docs:

docs/ADVISORY_EVIDENCE_CONTRACT.md

Optionally update:

README.md

docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md

Only update README if the project already has a section for generated outputs or advisory evidence. Keep changes conservative.

The contract must state:

This sidecar is advisory evidence.

Canonical question images and mark-scheme images remain the source of truth.

Examiner reports can support:

topic hints

misconception tags

common error patterns

omission/no-response signals

method difficulty

review prioritization

item-level difficulty hints

Grade thresholds can support:

component/session difficulty context

relative paper difficulty

coarse difficulty calibration

review prioritization

Grade thresholds cannot directly prove individual question difficulty.

No strict filtering or mastery decision should depend only on this sidecar.

Asterion and Content Lab may consume this later only through role-gated review flows.

Do not use advisory evidence to overwrite canonical topic routing.

Do not use advisory evidence to replace mark schemes.

Do not use advisory evidence as student-facing question text.

Likely files/modules to inspect or create:

docs/ADVISORY_EVIDENCE_CONTRACT.md

tests/test_advisory_docs.py if the repo has docs/hygiene tests

Exact targets:

Document all generated output paths.

Document schema versions.

Document allowed confidence labels.

Document allowed difficulty/context labels.

Document validation command.

Document build order:

inventory → extract → parse → link → sidecar → topic evidence → examiner difficulty → threshold context → predictions → validation → reports → optional AI

Document what downstream projects may and may not consume.

Validation commands:

.venv/bin/python -m pytest

or targeted docs tests if available.

If full test suite is too expensive, run the relevant targeted tests and state that full suite was not run.

Tests to add/update:

If the repo has README/docs hygiene tests, update them.

If not, add a lightweight test that required advisory docs exist and mention key output files.

Acceptance criteria:

Contract document exists.

Contract clearly states advisory-only status.

Contract documents output files and schemas.

Contract documents safety boundaries.

Contract documents validation/build commands.

No canonical output is changed.

Required final summary:

Summarize files changed/created, validation run, findings, risks/concerns, and suggested next steps.

⸻

Recommended execution order

Run these exactly in order:

1. Inventory documents.
2. Extract raw native PDF text.
3. Parse examiner report sections.
4. Parse grade threshold tables.
5. Link evidence to question-bank records.
6. Build advisory sidecar skeleton.
7. Add deterministic topic evidence.
8. Add examiner-report difficulty evidence.
9. Add grade-threshold paper context.
10. Combine advisory predictions.
11. Add validation gates.
12. Create review reports.
13. Add optional AI-assisted suggestions only for unresolved cases.
14. Document advisory sidecar contract.

The important discipline is this:

Do not let the project become “AI difficulty engine.”

Keep it as:

extract → structure → link → predict cautiously → validate → review.

Canonical images stay the source of truth. Examiner reports and grade thresholds become useful, auditable educational evidence.