from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
from typing import Any

from exam_bank.p3_exact_skill import DEFAULT_REVIEW_BATCH_DIR, DEFAULT_REVIEW_QUEUE_JSON_PATH
from exam_bank.p3_exact_skill.review_batch import ADVISORY_MARK_EVENT_WARNING

VISUAL_REVIEW_CHECKLIST = [
    "Inspect question image",
    "Inspect mark-scheme image",
    "Confirm exact P3 skill",
    "Confirm whether whole-question or part-level scope is safe",
    "Confirm whether P1 prerequisite/support-only material is involved",
    "Decide route_status",
    "Decide allowed use cases",
    "Write evidence_basis in project wording",
]


def build_p3_exact_skill_visual_review_packet(
    *,
    batch_dir: str | Path = DEFAULT_REVIEW_BATCH_DIR,
    batch_id: str = "batch_0001",
    repo_root: str | Path = ".",
    output_path: str | Path | None = None,
    dry_run: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    batch_dir = Path(batch_dir)
    repo_root = Path(repo_root).resolve()
    manifest_path = batch_dir / f"{batch_id}_manifest.v1.json"
    packet_path = batch_dir / f"{batch_id}_review_packet.md"
    template_path = batch_dir / f"{batch_id}_decision_template.v1.json"
    output_path = Path(output_path) if output_path else batch_dir / f"{batch_id}_visual_review.html"
    generated_at = generated_at or _utc_now_iso()

    manifest = _load_json(manifest_path)
    template = _load_json(template_path)
    packet_exists = packet_path.exists()
    queue_path = repo_root / str(manifest.get("source_queue_path") or DEFAULT_REVIEW_QUEUE_JSON_PATH)
    queue = _load_json(queue_path) if queue_path.exists() else {}

    records = [record for record in template.get("records", []) if isinstance(record, dict)]
    queue_items = _queue_items_by_id(queue)
    ordered_queue_ids = manifest.get("selected_queue_ids") if isinstance(manifest.get("selected_queue_ids"), list) else []
    records = _order_records(records, ordered_queue_ids)

    html = render_visual_review_html(
        records,
        manifest=manifest,
        queue_items=queue_items,
        batch_id=batch_id,
        generated_at=generated_at,
        repo_root=repo_root,
        output_path=output_path,
        manifest_path=manifest_path,
        packet_path=packet_path,
        template_path=template_path,
        packet_exists=packet_exists,
    )
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return {
        "ok": True,
        "dry_run": dry_run,
        "batch_id": batch_id,
        "selected_count": len(records),
        "output_path": str(output_path),
        "html": html if dry_run else None,
        "inputs": {
            "manifest": str(manifest_path),
            "packet": str(packet_path),
            "packet_exists": packet_exists,
            "template": str(template_path),
            "queue": str(queue_path),
            "queue_exists": queue_path.exists(),
        },
    }


def render_visual_review_html(
    records: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    queue_items: dict[str, dict[str, Any]],
    batch_id: str,
    generated_at: str,
    repo_root: Path,
    output_path: Path,
    manifest_path: Path,
    packet_path: Path,
    template_path: Path,
    packet_exists: bool,
) -> str:
    item_html = "\n".join(
        _render_item(
            index,
            record,
            queue_items.get(_text(record.get("queue_id")), {}),
            repo_root=repo_root,
            output_path=output_path,
        )
        for index, record in enumerate(records, start=1)
    )
    packet_status = "available" if packet_exists else "missing"
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{escape(batch_id)} P3 Visual Review</title>\n"
        f"  <style>{_css()}</style>\n"
        "</head>\n"
        "<body>\n"
        "  <main>\n"
        f"    <h1>P3 Exact-Skill Visual Review Packet: {escape(batch_id)}</h1>\n"
        '    <section class="summary">\n'
        "      <h2>Summary</h2>\n"
        f"      <p><strong>Batch ID:</strong> <code>{escape(batch_id)}</code></p>\n"
        f"      <p><strong>Selected count:</strong> <code>{len(records)}</code></p>\n"
        f"      <p><strong>Generated at:</strong> <code>{escape(generated_at)}</code></p>\n"
        f"      <p><strong>Manifest selected count:</strong> <code>{escape(str(manifest.get('selected_count', '')))}</code></p>\n"
        f"      <p><strong>Source packet:</strong> <code>{escape(str(packet_path))}</code> ({packet_status})</p>\n"
        f"      <p><strong>Decision template:</strong> <code>{escape(str(template_path))}</code></p>\n"
        f"      <p><strong>Manifest:</strong> <code>{escape(str(manifest_path))}</code></p>\n"
        '      <p class="warning">This is not reviewed evidence. It does not update the reviewed registry, does not promote any candidate, and does not create the Asterion sidecar.</p>\n'
        f'      <p class="warning">{escape(ADVISORY_MARK_EVENT_WARNING)}</p>\n'
        "    </section>\n"
        '    <section class="how-to">\n'
        "      <h2>How To Use This Packet</h2>\n"
        "      <ol>\n"
        "        <li>For each item, inspect the question image and mark-scheme image side by side.</li>\n"
        "        <li>Use the queue context only as review context; do not treat it as authority.</li>\n"
        "        <li>Record the final route status, allowed use cases, and evidence basis in the separate decision template or reviewed registry workflow.</li>\n"
        "        <li>Only manually reviewed and validated records may be copied into the reviewed-decision registry.</li>\n"
        "      </ol>\n"
        '      <div class="response-toolbar">\n'
        "        <h3>Response Capture</h3>\n"
        "        <p>Responses autosave in this browser only. To create a repo file, run the local save server and use the repo-save button, or download the JSON.</p>\n"
        f'        <p><strong>Browser save key:</strong> <code>p3ExactSkillReviewResponses:{escape(batch_id)}</code></p>\n'
        '        <button type="button" id="saveResponses">Save in this browser</button>\n'
        '        <button type="button" id="exportResponses">Download JSON</button>\n'
        '        <button type="button" id="submitResponses">Write repo JSON (server required)</button>\n'
        '        <span id="responseStatus" role="status"></span>\n'
        "      </div>\n"
        "    </section>\n"
        f"{item_html}\n"
        "  </main>\n"
        f"  <script>{_javascript(batch_id)}</script>\n"
        "</body>\n"
        "</html>\n"
    )


def _render_item(
    index: int,
    record: dict[str, Any],
    queue_item: dict[str, Any],
    *,
    repo_root: Path,
    output_path: Path,
) -> str:
    question_assets = record.get("source_question_asset_refs") if isinstance(record.get("source_question_asset_refs"), list) else []
    mark_assets = (
        record.get("source_mark_scheme_asset_refs") if isinstance(record.get("source_mark_scheme_asset_refs"), list) else []
    )
    mark_event_refs = record.get("mark_event_refs") if isinstance(record.get("mark_event_refs"), list) else []
    candidate_skill_ids = queue_item.get("candidate_p3_skill_ids") if isinstance(queue_item.get("candidate_p3_skill_ids"), list) else []
    suggested_skill_ids = (
        record.get("suggested_source_skill_ids") if isinstance(record.get("suggested_source_skill_ids"), list) else []
    )
    blockers = queue_item.get("proposed_blockers") if isinstance(queue_item.get("proposed_blockers"), list) else []
    cross_topic_status = _text(queue_item.get("cross_topic_status") or record.get("suggested_cross_topic_status") or "unknown")
    return (
        f'    <article class="review-item" id="item-{index}">\n'
        f"      <h2>{index}. {escape(_text(record.get('question_id')))} / {escape(_text(record.get('subpart_id')))}</h2>\n"
        '      <div class="meta-grid">\n'
        f"{_meta('Queue ID', record.get('queue_id'))}"
        f"{_meta('Question ID', record.get('question_id'))}"
        f"{_meta('Part / subpart', f'{_text(record.get('part_id'))} / {_text(record.get('subpart_id'))}')}"
        f"{_meta('Paper / session / variant', f'{_text(record.get('paper'))} / {_text(record.get('session'))} / {_text(record.get('variant'))}')}"
        f"{_meta('Candidate skill IDs', ', '.join(_texts(candidate_skill_ids)) or 'none')}"
        f"{_meta('Suggested source skill IDs', ', '.join(_texts(suggested_skill_ids)) or 'none')}"
        f"{_meta('Cross-topic status', cross_topic_status)}"
        f"{_meta('Recommended scope', queue_item.get('recommended_scope') or record.get('suggested_recommended_scope') or 'reviewer_decide')}"
        f"{_meta('Recommended review action', queue_item.get('recommended_review_action') or 'missing queue context')}"
        f"{_meta('Proposed blockers', ', '.join(_texts(blockers)) or 'none')}"
        "      </div>\n"
        '      <div class="image-grid">\n'
        f"{_asset_group('Question Image', question_assets, repo_root=repo_root, output_path=output_path)}"
        f"{_asset_group('Mark-Scheme Image', mark_assets, repo_root=repo_root, output_path=output_path)}"
        "      </div>\n"
        '      <details open>\n'
        "        <summary>Review Context</summary>\n"
        f"{_json_block('Candidate region/topic', queue_item.get('candidate_region_topic') or {})}"
        f"{_json_block('Topic-routing context', queue_item.get('topic_routing') or {})}"
        f"{_json_block('Content Lab blocker context', queue_item.get('asterion_candidate') or {})}"
        "      </details>\n"
        f"{_cross_topic_section(record, queue_item)}"
        '      <details>\n'
        "        <summary>Advisory-Only Mark-Event Refs</summary>\n"
        f'        <p class="warning small">{escape(ADVISORY_MARK_EVENT_WARNING)}</p>\n'
        f"{_json_list(mark_event_refs)}"
        "      </details>\n"
        '      <section class="checklist">\n'
        "        <h3>Reviewer Checklist</h3>\n"
        "        <ul>\n"
        f"{''.join(f'<li>{escape(item)}</li>' for item in VISUAL_REVIEW_CHECKLIST)}"
        "        </ul>\n"
        "      </section>\n"
        f"{_response_form(index, record)}"
        "    </article>\n"
    )


def _cross_topic_section(record: dict[str, Any], queue_item: dict[str, Any]) -> str:
    status = _text(queue_item.get("cross_topic_status") or record.get("suggested_cross_topic_status"))
    if not status or status in {"single_skill_candidate", "unknown"}:
        return ""
    primary = queue_item.get("primary_candidate_skill_ids") or record.get("suggested_primary_skill_ids") or []
    supporting = queue_item.get("supporting_candidate_skill_ids") or record.get("suggested_supporting_skill_ids") or []
    notes = queue_item.get("cross_topic_notes") if isinstance(queue_item.get("cross_topic_notes"), list) else []
    checklist = (
        queue_item.get("reviewer_cross_topic_checklist")
        if isinstance(queue_item.get("reviewer_cross_topic_checklist"), list)
        else []
    )
    warning_class = " warning" if status in {"cross_topic_split_needed", "conflict_needs_review"} else ""
    return (
        f'      <section class="cross-topic{warning_class}">\n'
        "        <h3>Cross-topic Review</h3>\n"
        f"        <p><strong>Status:</strong> <code>{escape(status)}</code></p>\n"
        f"        <p><strong>Primary candidate skill:</strong> <code>{escape(', '.join(_texts(primary)) or 'none')}</code></p>\n"
        f"        <p><strong>Supporting skill/topic context:</strong> <code>{escape(', '.join(_texts(supporting)) or 'none')}</code></p>\n"
        f"        <p><strong>Topic-routing topic:</strong> <code>{escape(', '.join(_texts(queue_item.get('topic_routing_topic_ids') or [])) or 'unknown')}</code></p>\n"
        f"        <p><strong>Topic-routing alignment:</strong> <code>{escape(_text(queue_item.get('topic_routing_alignment')) or 'unknown')}</code></p>\n"
        f"        <p><strong>Recommended scope:</strong> <code>{escape(_text(queue_item.get('recommended_scope') or record.get('suggested_recommended_scope')) or 'reviewer_decide')}</code></p>\n"
        "        <p class=\"warning small\">Supporting skills are not automatically reviewed source evidence. They need direct review before use as mastery evidence.</p>\n"
        f"{_json_block('Cross-topic notes', notes)}"
        "        <h4>Cross-topic checklist</h4>\n"
        "        <ul>\n"
        f"{''.join(f'<li>{escape(_text(item))}</li>' for item in checklist)}"
        "        </ul>\n"
        "      </section>\n"
    )


def _asset_group(title: str, refs: list[Any], *, repo_root: Path, output_path: Path) -> str:
    if not refs:
        return (
            '        <section class="asset-card missing">\n'
            f"          <h3>{escape(title)}</h3>\n"
            "          <p>Missing asset ref.</p>\n"
            "        </section>\n"
        )
    rendered = []
    for ref in refs:
        resolved = _resolve_asset_ref(ref if isinstance(ref, dict) else {}, repo_root=repo_root, output_path=output_path)
        if resolved["exists"]:
            rendered.append(
                f'          <a class="asset-link" href="{escape(resolved["href"])}" target="_blank" rel="noreferrer">'
                f'<img src="{escape(resolved["href"])}" alt="{escape(title)}"></a>\n'
                f'          <p><a href="{escape(resolved["href"])}" target="_blank" rel="noreferrer">Open original</a></p>\n'
                f'          <p><code>{escape(resolved["display_path"])}</code></p>\n'
            )
        else:
            rendered.append(
                '          <p class="missing-warning">Asset path is missing or cannot be resolved: '
                f'<code>{escape(resolved["display_path"])}</code></p>\n'
            )
    return (
        f'        <section class="asset-card{" missing" if all(not _resolve_asset_ref(ref if isinstance(ref, dict) else {}, repo_root=repo_root, output_path=output_path)["exists"] for ref in refs) else ""}">\n'
        f"          <h3>{escape(title)}</h3>\n"
        f"{''.join(rendered)}"
        "        </section>\n"
    )


def _response_form(index: int, record: dict[str, Any]) -> str:
    queue_id = _text(record.get("queue_id"))
    question_id = _text(record.get("question_id"))
    subpart_id = _text(record.get("subpart_id"))
    yes_no_review = ["not_reviewed", "yes", "no", "uncertain"]
    return (
        '      <section class="response-card">\n'
        "        <h3>Your Review Response</h3>\n"
        "        <p class=\"warning small\">These notes are draft review responses only. They do not update the reviewed registry or make any candidate clean.</p>\n"
        f'        <form class="review-response-form" data-item-number="{index}" data-queue-id="{escape(queue_id)}" data-question-id="{escape(question_id)}" data-subpart-id="{escape(subpart_id)}">\n'
        '          <div class="response-question-grid">\n'
        f"{_select('Have you inspected the question image?', 'inspected_question_image', yes_no_review)}"
        f"{_select('Have you inspected the mark-scheme image?', 'inspected_mark_scheme_image', yes_no_review)}"
        f"{_select('Does this confirm the suggested exact P3 skill?', 'exact_skill_confirmed', ['not_reviewed', 'yes', 'no_wrong_skill', 'uncertain', 'needs_split'])}"
        f"{_select('Is the current scope safe?', 'scope_decision', ['not_reviewed', 'whole_question_safe', 'part_level_needed', 'subpart_level_needed', 'unsafe_or_unclear'])}"
        f"{_select('Is P1/support-only material involved?', 'support_material_decision', ['not_reviewed', 'no', 'supporting_context_only', 'possible_target_skill', 'uncertain'])}"
        f"{_select('What route status would you choose?', 'route_status', ['review_needed', 'clean', 'thin', 'ambiguous', 'blocked', 'deferred', 'fallback_only'])}"
        f"{_select('What use-case allowance seems plausible after review?', 'allowed_use_case_summary', ['none', 'export_only', 'source_backed_examples', 'mastery_possible', 'guardian_possible', 'candidate_generation_possible', 'uncertain'])}"
        f"{_select('Is the evidence basis ready to write?', 'evidence_basis_status', ['not_started', 'drafted', 'needs_more_review', 'not_applicable'])}"
        "          </div>\n"
        f"{_textarea('Notes', 'reviewer_notes', 'Optional: evidence basis draft, blocker details, split suggestions, or follow-up questions.')}"
        "        </form>\n"
        "      </section>\n"
    )


def _select(label: str, field: str, options: list[str]) -> str:
    options_html = "".join(f'<option value="{escape(option)}">{escape(option)}</option>' for option in options)
    return (
        '            <label>'
        f"<span>{escape(label)}</span>"
        f'<select data-field="{escape(field)}">{options_html}</select>'
        "</label>\n"
    )


def _textarea(label: str, field: str, placeholder: str) -> str:
    return (
        '          <label class="wide-field">'
        f"<span>{escape(label)}</span>"
        f'<textarea data-field="{escape(field)}" placeholder="{escape(placeholder)}"></textarea>'
        "</label>\n"
    )


def _resolve_asset_ref(ref: dict[str, Any], *, repo_root: Path, output_path: Path) -> dict[str, Any]:
    path_text = _text(ref.get("path"))
    if not path_text:
        return {"exists": False, "href": "", "display_path": "missing path"}
    raw_path = Path(path_text)
    candidates = [raw_path] if raw_path.is_absolute() else [repo_root / raw_path, repo_root / "output" / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return {
                "exists": True,
                "href": _relative_href(candidate, output_path.parent),
                "display_path": str(candidate),
            }
    return {"exists": False, "href": "", "display_path": path_text}


def _relative_href(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def _meta(label: str, value: Any) -> str:
    return (
        '        <div class="meta-row">'
        f'<span>{escape(label)}</span><code>{escape(_text(value))}</code>'
        "</div>\n"
    )


def _json_block(title: str, value: Any) -> str:
    return (
        f"        <h3>{escape(title)}</h3>\n"
        f"        <pre>{escape(json.dumps(value, indent=2, sort_keys=True))}</pre>\n"
    )


def _json_list(values: list[Any]) -> str:
    if not values:
        return "        <p>None.</p>\n"
    return "        <pre>" + escape(json.dumps(values, indent=2, sort_keys=True)) + "</pre>\n"


def _order_records(records: list[dict[str, Any]], queue_ids: list[Any]) -> list[dict[str, Any]]:
    by_queue_id = {_text(record.get("queue_id")): record for record in records}
    ordered = [by_queue_id[_text(queue_id)] for queue_id in queue_ids if _text(queue_id) in by_queue_id]
    remaining = [record for record in records if record not in ordered]
    return ordered + remaining


def _queue_items_by_id(queue: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = queue.get("items") if isinstance(queue.get("items"), list) else []
    return {_text(item.get("queue_id")): item for item in items if isinstance(item, dict) and _text(item.get("queue_id"))}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _texts(values: list[Any]) -> list[str]:
    return [_text(value) for value in values if _text(value)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _javascript(batch_id: str) -> str:
    batch_json = json.dumps(batch_id)
    return f"""
(function () {{
  const batchId = {batch_json};
  const storageKey = `p3ExactSkillReviewResponses:${{batchId}}`;
  const status = document.getElementById('responseStatus');
  const forms = Array.from(document.querySelectorAll('.review-response-form'));

  function setStatus(message) {{
    if (!status) return;
    status.textContent = message;
  }}

  function readStored() {{
    try {{
      return JSON.parse(localStorage.getItem(storageKey) || '{{}}');
    }} catch (error) {{
      return {{}};
    }}
  }}

  function collectForm(form) {{
    const response = {{
      item_number: Number(form.dataset.itemNumber || 0),
      queue_id: form.dataset.queueId || '',
      question_id: form.dataset.questionId || '',
      subpart_id: form.dataset.subpartId || '',
      fields: {{}}
    }};
    form.querySelectorAll('[data-field]').forEach((control) => {{
      const field = control.dataset.field;
      response.fields[field] = control.type === 'checkbox' ? control.checked : control.value;
    }});
    return response;
  }}

  function collectPayload() {{
    return {{
      schema: 'exam_bank.p3_exact_skill.review_batch_responses',
      schema_version: 1,
      artifact_kind: 'human_review_response_draft',
      batch_id: batchId,
      saved_at: new Date().toISOString(),
      warning: 'Draft human review notes only. Not reviewed evidence and not the reviewed-decision registry.',
      responses: forms.map(collectForm)
    }};
  }}

  function saveLocal() {{
    const payload = collectPayload();
    localStorage.setItem(storageKey, JSON.stringify(payload));
    setStatus(`Saved in this browser at ${{new Date().toLocaleTimeString()}}; no repo file is written until you download JSON or use the save server.`);
    return payload;
  }}

  function restoreLocal() {{
    const payload = readStored();
    const byQueueId = new Map((payload.responses || []).map((response) => [response.queue_id, response]));
    forms.forEach((form) => {{
      const response = byQueueId.get(form.dataset.queueId);
      if (!response || !response.fields) return;
      form.querySelectorAll('[data-field]').forEach((control) => {{
        const value = response.fields[control.dataset.field];
        if (value === undefined) return;
        if (control.type === 'checkbox') {{
          control.checked = Boolean(value);
        }} else {{
          control.value = value;
        }}
      }});
    }});
    if ((payload.responses || []).length) {{
      setStatus(`Restored ${{payload.responses.length}} saved responses from this browser.`);
    }}
  }}

  function exportJson() {{
    const payload = saveLocal();
    const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${{batchId}}_review_responses.v1.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }}

  async function submitToServer() {{
    const payload = saveLocal();
    try {{
      const response = await fetch('/p3-exact-skill-review-responses', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
      }});
      if (!response.ok) {{
        throw new Error(`HTTP ${{response.status}}`);
      }}
      const result = await response.json();
      setStatus(`Saved to repo: ${{result.path || 'response file'}}`);
    }} catch (error) {{
      setStatus('No repo file written. Use Download JSON, or run scripts/serve_p3_exact_skill_visual_review.py and try the repo-save button again.');
    }}
  }}

  forms.forEach((form) => {{
    form.addEventListener('input', saveLocal);
    form.addEventListener('change', saveLocal);
  }});
  document.getElementById('saveResponses')?.addEventListener('click', saveLocal);
  document.getElementById('exportResponses')?.addEventListener('click', exportJson);
  document.getElementById('submitResponses')?.addEventListener('click', submitToServer);
  restoreLocal();
}})();
"""


def _css() -> str:
    return """
body {
  margin: 0;
  background: #f5f5f2;
  color: #1f2933;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
main {
  max-width: 1440px;
  margin: 0 auto;
  padding: 24px;
}
h1, h2, h3 {
  letter-spacing: 0;
}
code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.summary, .how-to, .review-item {
  background: #ffffff;
  border: 1px solid #d8d6cf;
  border-radius: 8px;
  margin: 0 0 20px;
  padding: 18px;
}
.warning {
  border-left: 4px solid #b45309;
  background: #fff7ed;
  padding: 10px 12px;
}
.warning.small {
  font-size: 0.95rem;
}
.cross-topic {
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  background: #eff6ff;
  padding: 12px;
  margin: 12px 0;
}
.cross-topic.warning {
  border-left: 4px solid #b45309;
  border-color: #fed7aa;
  background: #fff7ed;
}
.response-toolbar, .response-card {
  border: 1px solid #d8d6cf;
  border-radius: 8px;
  background: #fafaf8;
  padding: 12px;
  margin-top: 14px;
}
.response-toolbar button {
  border: 1px solid #334155;
  border-radius: 6px;
  background: #ffffff;
  color: #0f172a;
  cursor: pointer;
  font: inherit;
  margin: 4px 8px 4px 0;
  padding: 8px 10px;
}
#responseStatus {
  color: #475569;
  display: inline-block;
  margin-left: 8px;
}
.response-question-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}
.response-card label span,
.response-card legend {
  color: #475569;
  display: block;
  font-size: 0.86rem;
  font-weight: 700;
  margin-bottom: 4px;
}
.response-card input[type="text"],
.response-card select,
.response-card textarea {
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  box-sizing: border-box;
  font: inherit;
  padding: 8px;
  width: 100%;
}
.response-card textarea {
  min-height: 86px;
  resize: vertical;
}
.wide-field {
  display: block;
  margin: 10px 0;
}
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 8px;
}
.meta-row {
  border: 1px solid #e5e2da;
  border-radius: 6px;
  padding: 8px;
  min-width: 0;
}
.meta-row span {
  display: block;
  color: #59636e;
  font-size: 0.82rem;
  margin-bottom: 4px;
}
.meta-row code {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 16px;
  margin: 16px 0;
}
.asset-card {
  border: 1px solid #d8d6cf;
  border-radius: 8px;
  padding: 12px;
  background: #fafaf8;
}
.asset-card img {
  display: block;
  width: 100%;
  max-height: 920px;
  object-fit: contain;
  background: #fff;
  border: 1px solid #ece8df;
}
.asset-card.missing {
  background: #fff1f2;
  border-color: #fda4af;
}
.missing-warning {
  color: #9f1239;
  font-weight: 600;
}
details {
  margin: 12px 0;
}
summary {
  cursor: pointer;
  font-weight: 700;
}
pre {
  overflow: auto;
  white-space: pre-wrap;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 10px;
}
.checklist ul {
  columns: 2;
}
@media (max-width: 760px) {
  main {
    padding: 12px;
  }
  .image-grid {
    grid-template-columns: 1fr;
  }
  .checklist ul {
    columns: 1;
  }
}
"""
