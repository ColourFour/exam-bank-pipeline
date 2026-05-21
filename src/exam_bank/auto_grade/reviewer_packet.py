from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.auto_grade import AUTO_GRADE_SCHEMA_VERSION
from exam_bank.auto_grade.constants import REVIEWED_RUBRIC_SCHEMA_VERSION, REVIEWED_RUBRICS_SCHEMA

DEFAULT_REVIEWER_PACKET_DIR = "reports/auto_grade/reviewer_packets/review_batch_0001"
DEFAULT_APPROVAL_TEMPLATE_DIR = "output/auto_grade/review_batches/approval_templates"
NOT_APPROVED_WARNING = "This candidate is not approved scoring evidence until reviewed_rubrics validation passes."


def build_reviewer_packet(
    *,
    review_batch_path: str | Path,
    question_bank_path: str | Path,
    output_dir: str | Path = DEFAULT_REVIEWER_PACKET_DIR,
    reviewed_rubrics_path: str | Path | None = None,
    include_html: bool = True,
) -> dict[str, Any]:
    review_batch_path = Path(review_batch_path)
    question_bank_path = Path(question_bank_path)
    output_dir = Path(output_dir)
    batch = _load_json(review_batch_path)
    question_bank = _load_json(question_bank_path)
    draft = _load_optional_json(reviewed_rubrics_path)
    questions = _records_by_question_id(_question_records(question_bank))
    draft_by_rubric_id = _records_by_key(draft.get("rubrics", []) if isinstance(draft, dict) else [], "rubric_id")
    candidates = [item for item in batch.get("candidates") or [] if isinstance(item, dict)]

    output_dir.mkdir(parents=True, exist_ok=True)
    page_records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        question_id = str(candidate.get("question_id") or "")
        rubric_id = str(candidate.get("proposed_rubric_id") or f"rr_{question_id}")
        filename = _candidate_page_name(index, question_id)
        page_path = output_dir / filename
        page_path.write_text(
            render_candidate_packet_page(
                candidate,
                index=index,
                question=questions.get(question_id, {}),
                draft_rubric=draft_by_rubric_id.get(rubric_id, {}),
            ),
            encoding="utf-8",
        )
        page_records.append(
            {
                "index": index,
                "question_id": question_id,
                "rubric_id": rubric_id,
                "page": filename,
                "question_image_path": _question_image_path(candidate, questions.get(question_id, {})),
                "mark_scheme_image_path": _mark_scheme_image_path(candidate, questions.get(question_id, {})),
            }
        )

    index_text = render_reviewer_packet_index(batch, page_records=page_records)
    (output_dir / "index.md").write_text(index_text, encoding="utf-8")
    html_path = None
    if include_html:
        html_path = output_dir / "index.html"
        html_path.write_text(render_reviewer_packet_html(batch, page_records=page_records), encoding="utf-8")
    return {
        "schema": "exam_bank.auto_grade.reviewer_packet",
        "schema_version": AUTO_GRADE_SCHEMA_VERSION,
        "review_batch": _rel_path(review_batch_path),
        "question_bank": _rel_path(question_bank_path),
        "output_dir": _rel_path(output_dir),
        "index": _rel_path(output_dir / "index.md"),
        "html_index": _rel_path(html_path) if html_path else None,
        "candidate_page_count": len(page_records),
        "candidate_pages": page_records,
    }


def build_approval_template(
    *,
    review_batch_path: str | Path,
    rubric_id: str | None = None,
    question_id: str | None = None,
    first: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    batch = _load_json(review_batch_path)
    candidate = _select_candidate(batch, rubric_id=rubric_id, question_id=question_id, first=first)
    template = approval_template_for_candidate(candidate)
    if output_path:
        write_atomic_json(template, output_path, sort_keys=True)
    return template


def approval_template_for_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    question_id = str(candidate.get("question_id") or "")
    rubric_id = str(candidate.get("proposed_rubric_id") or f"rr_{question_id}")
    events = []
    for index, event in enumerate(candidate.get("proposed_events") or [], start=1):
        if not isinstance(event, dict):
            continue
        events.append(
            {
                "event_id": f"{rubric_id}_e{index:04d}",
                "source_event_id": event.get("source_event_id"),
                "part_path": event.get("part_path") or [],
                "mark_code": event.get("mark_code"),
                "mark_type": event.get("mark_type"),
                "mark_value": event.get("mark_value"),
                "dependency": event.get("dependency") or "TODO_DEPENDENCY_OR_INDEPENDENT",
                "follow_through_policy": event.get("follow_through_policy") or "TODO_FOLLOW_THROUGH_POLICY_OR_NONE",
                "accepted_evidence": ["TODO_REWRITE_FROM_CANONICAL_MARK_SCHEME_IMAGE"],
                "common_errors": [],
                "alternative_methods": ["TODO_OR_EMPTY_LIST_AFTER_HUMAN_REVIEW"],
                "learning_target_ids": ["TODO_LEARNING_TARGET_ID"],
                "review_status": "needs_human_review",
                "review_notes": "TODO_REVIEW_NOTES",
                "advisory_source_material": {
                    "not_scoring_evidence": True,
                    "advisory_evidence": event.get("advisory_evidence") or {},
                    "mark_code_raw": event.get("mark_code_raw"),
                    "review_flags": event.get("review_flags") or [],
                },
            }
        )
    return {
        "schema": REVIEWED_RUBRICS_SCHEMA,
        "schema_version": REVIEWED_RUBRIC_SCHEMA_VERSION,
        "template_note": "Unapproved template only. A human must replace TODO values, verify canonical images, and run validation before use.",
        "rubric": {
            "rubric_id": rubric_id,
            "source_question_id": question_id,
            "source_question_image_path": candidate.get("canonical_question_artifact"),
            "source_mark_scheme_image_path": candidate.get("canonical_mark_scheme_artifact"),
            "source_mark_events_record_id": ((candidate.get("mark_event_source_reference") or {}).get("record_id")),
            "paper": candidate.get("paper"),
            "paper_family": candidate.get("paper_family"),
            "question_number": candidate.get("question_number"),
            "part_path": candidate.get("part_path") or [],
            "total_marks": candidate.get("total_marks"),
            "rubric_total_verified": False,
            "safe_for_auto_grade_lab": False,
            "safe_for_teacher_beta": False,
            "safe_for_student_self_check": False,
            "review_status": "needs_human_review",
            "reviewed_by": "TODO_REVIEWED_BY",
            "reviewed_at": "TODO_ISO_REVIEW_TIMESTAMP",
            "approval_scope": "none",
            "review_notes": "TODO_REVIEW_NOTES",
            "events": events,
            "advisory_source_material": {
                "not_scoring_evidence": True,
                "mark_codes_detected": candidate.get("mark_codes_detected") or [],
                "candidate_risk_flags": candidate.get("candidate_risk_flags") or [],
                "candidate_blockers": candidate.get("candidate_blockers") or [],
                "dependency_flags": candidate.get("dependency_flags") or [],
                "follow_through_flags": candidate.get("follow_through_flags") or [],
                "learning_target_ids_advisory": candidate.get("learning_target_ids_advisory") or [],
            },
        },
    }


def render_reviewer_packet_index(batch: dict[str, Any], *, page_records: list[dict[str, Any]]) -> str:
    lines = [
        "# Reviewer Packet: Review Batch 0001",
        "",
        NOT_APPROVED_WARNING,
        "",
        "## Summary",
        "",
        f"- Candidate rubrics: {len(page_records)}",
        f"- Advisory mark events: {batch.get('event_count', 0)}",
        f"- Source batch: `{batch.get('source_review_batch_path') or 'output/auto_grade/review_batches/review_batch_0001.v1.json'}`",
        "",
        "## Candidate Pages",
        "",
    ]
    for record in page_records:
        lines.append(
            f"- [{record['index']:03d} `{record['question_id']}` / `{record['rubric_id']}`]({record['page']})"
        )
    lines.extend(
        [
            "",
            "## Review Rule",
            "",
            "- Open the canonical question image and canonical mark-scheme image before editing any JSON.",
            "- Treat advisory mark events as review aids only.",
            "- Do not approve a rubric until the reviewed-rubrics validator passes.",
            "- Phase 2C must not produce student self-check or student-ready approvals.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_candidate_packet_page(
    candidate: dict[str, Any],
    *,
    index: int,
    question: dict[str, Any] | None = None,
    draft_rubric: dict[str, Any] | None = None,
) -> str:
    question = question or {}
    draft_rubric = draft_rubric or {}
    question_id = str(candidate.get("question_id") or "")
    question_image = _question_image_path(candidate, question)
    mark_scheme_image = _mark_scheme_image_path(candidate, question)
    proposed_events = candidate.get("proposed_events") or []
    lines = [
        f"# Rubric {index:03d}: {question_id}",
        "",
        f"**Warning:** {NOT_APPROVED_WARNING}",
        "",
        "## Candidate Metadata",
        "",
        f"- question_id: `{question_id}`",
        f"- paper: `{candidate.get('paper') or ''}`",
        f"- paper_family: `{candidate.get('paper_family') or ''}`",
        f"- question_number: `{candidate.get('question_number') or ''}`",
        f"- part_path: `{json.dumps(candidate.get('part_path') or [])}`",
        f"- total_marks: `{candidate.get('total_marks')}`",
        f"- proposed rubric_id: `{candidate.get('proposed_rubric_id') or ''}`",
        f"- detected mark codes: `{', '.join(str(code) for code in candidate.get('mark_codes_detected') or []) or 'none'}`",
        f"- risk flags: `{', '.join(str(flag) for flag in candidate.get('candidate_risk_flags') or []) or 'none'}`",
        f"- blockers: `{', '.join(str(flag) for flag in candidate.get('candidate_blockers') or []) or 'none'}`",
        "",
        "## Canonical Images",
        "",
        f"- Question image: [{question_image}]({_md_link_path(question_image)})" if question_image else "- Question image: missing",
        f"- Mark-scheme image: [{mark_scheme_image}]({_md_link_path(mark_scheme_image)})"
        if mark_scheme_image
        else "- Mark-scheme image: missing",
        "",
        "## Dependency And Follow-Through Notes",
        "",
        f"- Dependency flags: `{', '.join(str(value) for value in candidate.get('dependency_flags') or []) or 'none'}`",
        f"- Follow-through flags: `{', '.join(str(value) for value in candidate.get('follow_through_flags') or []) or 'none'}`",
        "",
        "## Advisory Mark Events",
        "",
    ]
    if not proposed_events:
        lines.append("- none")
    for event_index, event in enumerate(proposed_events, start=1):
        if not isinstance(event, dict):
            continue
        advisory = event.get("advisory_evidence") if isinstance(event.get("advisory_evidence"), dict) else {}
        lines.extend(
            [
                f"### Event {event_index}: `{event.get('source_event_id') or ''}`",
                "",
                f"- mark_code: `{event.get('mark_code') or ''}`",
                f"- mark_code_raw: `{event.get('mark_code_raw') or ''}`",
                f"- mark_type: `{event.get('mark_type') or ''}`",
                f"- mark_value: `{event.get('mark_value')}`",
                f"- dependency: `{event.get('dependency') or ''}`",
                f"- follow_through_policy: `{event.get('follow_through_policy') or ''}`",
                f"- advisory confidence: `{advisory.get('confidence') or ''}`",
                "",
                "Advisory text, not scoring evidence:",
                "",
                "```text",
                str(advisory.get("normalized_text") or advisory.get("raw_text") or "").strip(),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Reviewer Checklist",
            "",
            *_checklist_lines(candidate.get("reviewer_checklist") or []),
            "",
            "## Approval Checklist",
            "",
            "- [ ] Canonical question image verified.",
            "- [ ] Canonical mark-scheme image verified.",
            "- [ ] Total marks reconciled with event mark values.",
            "- [ ] Every advisory event rewritten as human-reviewed accepted evidence.",
            "- [ ] Mark codes verified and no `unknown` mark codes remain.",
            "- [ ] Dependencies and follow-through policies documented where applicable.",
            "- [ ] Learning target IDs assigned to every approved event.",
            "- [ ] Student self-check and student-ready remain false/not present.",
            "",
            "## JSON Fields Reviewer Must Complete",
            "",
            "- `source_question_image_path`",
            "- `source_mark_scheme_image_path`",
            "- `reviewed_by`",
            "- `reviewed_at`",
            "- `review_status`",
            "- `rubric_total_verified`",
            "- `safe_for_auto_grade_lab`",
            "- `safe_for_teacher_beta`",
            "- `safe_for_student_self_check`",
            "- `accepted_evidence` for every event",
            "- `dependency` for every dependent event",
            "- `follow_through_policy` for every follow-through event",
            "- `alternative_methods`",
            "- `learning_target_ids` for every approved event",
            "- `review_notes`",
            "",
            "## Draft Rubric Snapshot",
            "",
            "```json",
            json.dumps(draft_rubric or {"rubric_id": candidate.get("proposed_rubric_id")}, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_reviewer_packet_html(batch: dict[str, Any], *, page_records: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{record['index']:03d}</td>"
        f"<td>{html.escape(str(record['question_id']))}</td>"
        f"<td><a href=\"{html.escape(str(record['page']))}\">{html.escape(str(record['rubric_id']))}</a></td>"
        f"<td>{_html_image_link(record.get('question_image_path'))}</td>"
        f"<td>{_html_image_link(record.get('mark_scheme_image_path'))}</td>"
        "</tr>"
        for record in page_records
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reviewer Packet: Review Batch 0001</title>
  <style>
    body {{ font-family: system-ui, sans-serif; line-height: 1.4; margin: 2rem; max-width: 1100px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 0.4rem 0.5rem; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    .warning {{ border-left: 4px solid #d1242f; padding: 0.5rem 0.75rem; background: #fff8f8; }}
  </style>
</head>
<body>
  <h1>Reviewer Packet: Review Batch 0001</h1>
  <p class="warning">{html.escape(NOT_APPROVED_WARNING)}</p>
  <p>Candidate rubrics: {len(page_records)}. Advisory mark events: {html.escape(str(batch.get("event_count", 0)))}.</p>
  <table>
    <thead><tr><th>#</th><th>Question</th><th>Rubric page</th><th>Question image</th><th>Mark-scheme image</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
"""


def _select_candidate(
    batch: dict[str, Any], *, rubric_id: str | None, question_id: str | None, first: bool
) -> dict[str, Any]:
    candidates = [item for item in batch.get("candidates") or [] if isinstance(item, dict)]
    if first:
        if not candidates:
            raise ValueError("review batch has no candidates")
        return candidates[0]
    if not rubric_id and not question_id:
        raise ValueError("provide --first, --rubric-id, or --question-id")
    for candidate in candidates:
        if rubric_id and str(candidate.get("proposed_rubric_id") or "") == rubric_id:
            return candidate
        if question_id and str(candidate.get("question_id") or "") == question_id:
            return candidate
    raise ValueError("candidate not found in review batch")


def _candidate_page_name(index: int, question_id: str) -> str:
    return f"rubric_{index:03d}_{_slug(question_id)}.md"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug or "missing_question_id"


def _question_image_path(candidate: dict[str, Any], question: dict[str, Any]) -> str:
    return str(
        candidate.get("canonical_question_artifact")
        or question.get("canonical_question_artifact")
        or question.get("question_image_path")
        or ""
    )


def _mark_scheme_image_path(candidate: dict[str, Any], question: dict[str, Any]) -> str:
    return str(
        candidate.get("canonical_mark_scheme_artifact")
        or question.get("mark_scheme_image_path")
        or ((question.get("mark_scheme_image_paths") or [""])[0] if isinstance(question.get("mark_scheme_image_paths"), list) else "")
        or ""
    )


def _checklist_lines(values: list[str]) -> list[str]:
    if not values:
        return ["- [ ] canonical_question_image_verified", "- [ ] canonical_mark_scheme_image_verified"]
    return [f"- [ ] `{value}`" for value in values]


def _md_link_path(path: str) -> str:
    return path.replace(" ", "%20")


def _html_image_link(path: Any) -> str:
    if not path:
        return "missing"
    escaped = html.escape(str(path), quote=True)
    return f"<a href=\"{escaped}\">{escaped}</a>"


def _records_by_key(records: Any, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        return {}
    return {str(record.get(key) or ""): record for record in records if isinstance(record, dict) and record.get(key)}


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _records_by_question_id(records: Any) -> dict[str, dict[str, Any]]:
    return {str(record.get("question_id") or ""): record for record in records if record.get("question_id")}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_optional_json(path: str | Path | None) -> Any:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return _load_json(path)


def _rel_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
