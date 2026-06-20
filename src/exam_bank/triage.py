from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
from typing import Any
from urllib.parse import urlparse

from .audit import audit_current_output_integrity


ROOT_CAUSE_OPTIONS = [
    "question_crop_boundary",
    "mark_scheme_mapping",
    "paper_total_detection",
    "false_positive_validation_gate",
    "text_ocr_quality",
    "classification_only",
    "source_pdf_issue",
    "unknown",
]

ROOT_CAUSE_LABELS = {
    "question_crop_boundary": "Question crop/boundary",
    "mark_scheme_mapping": "Mark-scheme mapping",
    "paper_total_detection": "Paper-total detection",
    "false_positive_validation_gate": "False-positive validation gate",
    "text_ocr_quality": "Text/OCR quality",
    "classification_only": "Classification-only",
    "source_pdf_issue": "Source PDF issue",
    "unknown": "Unknown",
}

ISSUE_SET_HARD_FAILURES = "hard-failures"
ISSUE_SET_ALL_NON_READY = "all-non-ready"
ISSUE_SETS = {ISSUE_SET_HARD_FAILURES, ISSUE_SET_ALL_NON_READY}

STATUS_ORDERS = {
    "validation_status": {"pass": 0, "review": 1, "fail": 2},
    "mapping_status": {"pass": 0, "review": 1, "fail": 2},
    "visual_curation_status": {"ready": 0, "review": 1, "fail": 2},
}


def load_question_bank(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return payload, list(payload["questions"])
    if isinstance(payload, list):
        return {"questions": payload, "record_count": len(payload)}, list(payload)
    raise ValueError("Question bank input must be a question bank document or a list of question records.")


def is_hard_failure(record: dict[str, Any]) -> bool:
    return (
        _field(record, "validation_status") == "fail"
        or _field(record, "mapping_status") == "fail"
        or _field(record, "visual_curation_status") == "fail"
    )


def is_non_ready(record: dict[str, Any]) -> bool:
    return (
        _field(record, "validation_status") != "pass"
        or _field(record, "mapping_status") != "pass"
        or _field(record, "visual_curation_status") != "ready"
        or _field(record, "text_only_status") != "ready"
    )


def primary_issue(record: dict[str, Any]) -> str:
    validation_flags = _list_field(record, "validation_flags")
    if validation_flags:
        return validation_flags[0]
    if _field(record, "mapping_status") == "fail":
        reason = str(_field(record, "mapping_failure_reason") or "").strip()
        return f"mapping_failed:{reason}" if reason else "mapping_failed_no_validation_flag"
    if _field(record, "visual_curation_status") == "fail":
        return "visual_curation_failed"
    if _field(record, "validation_status") == "review":
        return "validation_review"
    if _field(record, "visual_curation_status") == "review":
        return "visual_curation_review"
    return "other"


def issue_counts(records: list[dict[str, Any]], *, issue_set: str = ISSUE_SET_HARD_FAILURES) -> dict[str, int]:
    return dict(Counter(primary_issue(record) for record in _issue_records(records, issue_set=issue_set)).most_common())


def select_sample_records(
    records: list[dict[str, Any]],
    *,
    issue_set: str = ISSUE_SET_HARD_FAILURES,
    sample_size: int = 30,
    target: str = "auto",
    seed: int = 1,
) -> tuple[str, list[dict[str, Any]]]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    counts = issue_counts(records, issue_set=issue_set)
    selected_target = _auto_target(counts) if target == "auto" else target
    candidates = [
        record
        for record in _issue_records(records, issue_set=issue_set)
        if _record_matches_target(record, selected_target)
    ]
    candidates = sorted(candidates, key=_record_sort_key)
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    sample = shuffled[: min(sample_size, len(shuffled))]
    return selected_target, sorted(sample, key=_record_sort_key)


def create_triage_iteration(
    input_path: str | Path,
    *,
    triage_root: str | Path | None = None,
    iteration: str | Path | None = None,
    issue_set: str = ISSUE_SET_HARD_FAILURES,
    sample_size: int = 30,
    target: str = "auto",
    seed: int = 1,
) -> dict[str, Any]:
    if issue_set not in ISSUE_SETS:
        raise ValueError(f"Unsupported issue set: {issue_set}")

    input_path = Path(input_path)
    output_root = _infer_output_root(input_path)
    triage_root = Path(triage_root) if triage_root is not None else output_root / "triage"
    iteration_dir = _iteration_dir(triage_root, iteration)
    if iteration_dir.exists():
        raise FileExistsError(f"Triage iteration already exists: {iteration_dir}")
    iteration_dir.mkdir(parents=True)

    _, records = load_question_bank(input_path)
    counts = issue_counts(records, issue_set=issue_set)
    selected_target, sample = select_sample_records(
        records,
        issue_set=issue_set,
        sample_size=sample_size,
        target=target,
        seed=seed,
    )

    baseline_path = iteration_dir / "baseline_question_bank.json"
    baseline_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")

    sample_payload = {
        "issue_set": issue_set,
        "target": selected_target,
        "seed": seed,
        "sample_size": sample_size,
        "sampled_count": len(sample),
        "questions": [_sample_record(record, iteration_dir, output_root) for record in sample],
    }
    sample_path = iteration_dir / "sample.json"
    _write_json(sample_path, sample_payload)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_root": str(output_root),
        "baseline_path": str(baseline_path),
        "sample_path": str(sample_path),
        "gallery_path": str(iteration_dir / "index.html"),
        "review_path": str(iteration_dir / "review.jsonl"),
        "issue_set": issue_set,
        "sample_size": sample_size,
        "seed": seed,
        "target": selected_target,
        "target_source": "auto" if target == "auto" else "explicit",
        "record_count": len(records),
        "hard_failure_count": sum(1 for record in records if is_hard_failure(record)),
        "issue_counts": counts,
        "target_issue_count": counts.get(selected_target, 0),
        "sampled_count": len(sample),
        "sample_question_ids": [str(record.get("question_id") or "") for record in sample],
        "root_cause_options": ROOT_CAUSE_OPTIONS,
    }
    _write_json(iteration_dir / "summary.json", summary)
    (iteration_dir / "review.jsonl").touch()
    write_gallery(iteration_dir / "index.html", sample_payload)
    return {"iteration_dir": str(iteration_dir), **summary}


def create_suspicious_crop_review_pack(
    input_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    review_root: str | Path | None = None,
    iteration: str | Path | None = None,
    sample_size: int = 30,
) -> dict[str, Any]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    input_path = Path(input_path)
    output_root = Path(artifact_root) if artifact_root is not None else _infer_output_root(input_path)
    review_root = Path(review_root) if review_root is not None else output_root / "triage"
    iteration_dir = _iteration_dir(review_root, iteration)
    if iteration_dir.exists():
        raise FileExistsError(f"Suspicious crop review iteration already exists: {iteration_dir}")
    iteration_dir.mkdir(parents=True)

    payload, records = load_question_bank(input_path)
    records_by_id = {str(record.get("question_id")): record for record in records if record.get("question_id")}
    audit = audit_current_output_integrity(
        input_path,
        artifact_root=output_root,
        example_limit=sample_size,
    )
    candidates = _suspicious_crop_review_candidates(audit)
    sample = candidates[: min(sample_size, len(candidates))]

    baseline_path = iteration_dir / "baseline_question_bank.json"
    baseline_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")

    sample_payload = {
        "issue_set": "output-integrity",
        "target": "suspicious_rendered_crop_artifacts",
        "seed": "",
        "sample_size": sample_size,
        "sampled_count": len(sample),
        "questions": [
            _suspicious_crop_sample_record(candidate, records_by_id.get(str(candidate.get("question_id")), {}), iteration_dir, output_root)
            for candidate in sample
        ],
    }
    sample_path = iteration_dir / "sample.json"
    _write_json(sample_path, sample_payload)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_root": str(output_root),
        "artifact_root": str(output_root),
        "baseline_path": str(baseline_path),
        "sample_path": str(sample_path),
        "gallery_path": str(iteration_dir / "index.html"),
        "review_path": str(iteration_dir / "review.jsonl"),
        "issue_set": "output-integrity",
        "target": "suspicious_rendered_crop_artifacts",
        "sample_size": sample_size,
        "record_count": len(records),
        "candidate_count": len(candidates),
        "dimension_candidate_count": int((audit.get("counts") or {}).get("suspicious_rendered_crop_dimension_count") or 0),
        "whitespace_candidate_count": int((audit.get("counts") or {}).get("suspicious_rendered_crop_whitespace_count") or 0),
        "sampled_count": len(sample),
        "sample_question_ids": [str(candidate.get("question_id") or "") for candidate in sample],
        "audit_summary": {
            "dimensions": audit.get("suspicious_rendered_crop_dimension_summary") or {},
            "whitespace": audit.get("suspicious_rendered_crop_whitespace_summary") or {},
        },
        "root_cause_options": ROOT_CAUSE_OPTIONS,
    }
    _write_json(iteration_dir / "summary.json", summary)
    (iteration_dir / "review.jsonl").touch()
    write_gallery(iteration_dir / "index.html", sample_payload)
    return {"iteration_dir": str(iteration_dir), **summary}


def compare_iteration(
    iteration_dir: str | Path,
    *,
    current_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    iteration_dir = Path(iteration_dir)
    summary_path = iteration_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    baseline_path = Path(summary.get("baseline_path") or iteration_dir / "baseline_question_bank.json")
    _, baseline_records = load_question_bank(baseline_path)
    _, current_records = load_question_bank(current_path)
    issue_set = str(summary.get("issue_set") or ISSUE_SET_HARD_FAILURES)
    target = str(summary.get("target") or _auto_target(issue_counts(baseline_records, issue_set=issue_set)))

    baseline_counts = issue_counts(baseline_records, issue_set=issue_set)
    current_counts = issue_counts(current_records, issue_set=issue_set)
    field_deltas = _issue_count_deltas(baseline_counts, current_counts)
    movement = _record_movements(baseline_records, current_records)
    report = {
        "iteration_dir": str(iteration_dir),
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "issue_set": issue_set,
        "target": target,
        "baseline_record_count": len(baseline_records),
        "current_record_count": len(current_records),
        "baseline_hard_failure_count": sum(1 for record in baseline_records if is_hard_failure(record)),
        "current_hard_failure_count": sum(1 for record in current_records if is_hard_failure(record)),
        "hard_failure_delta": sum(1 for record in current_records if is_hard_failure(record))
        - sum(1 for record in baseline_records if is_hard_failure(record)),
        "baseline_target_issue_count": baseline_counts.get(target, 0),
        "current_target_issue_count": current_counts.get(target, 0),
        "target_issue_delta": current_counts.get(target, 0) - baseline_counts.get(target, 0),
        "baseline_issue_counts": baseline_counts,
        "current_issue_counts": current_counts,
        "issue_count_deltas": field_deltas,
        **movement,
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report


def write_gallery(output_path: str | Path, sample_payload: dict[str, Any]) -> Path:
    output_path = Path(output_path)
    output_path.write_text(_gallery_html(sample_payload), encoding="utf-8")
    return output_path


def serve_iteration(iteration_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    iteration_dir = Path(iteration_dir).resolve()
    summary_path = iteration_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    output_root = Path(summary.get("output_root") or _infer_output_root_from_iteration(iteration_dir)).resolve()

    handler = partial(_TriageRequestHandler, directory=str(output_root), iteration_dir=iteration_dir)
    server = ThreadingHTTPServer((host, port), handler)
    index_path = os.path.relpath(iteration_dir / "index.html", output_root).replace(os.sep, "/")
    print(f"Serving triage gallery at http://{host}:{port}/{index_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _issue_records(records: list[dict[str, Any]], *, issue_set: str) -> list[dict[str, Any]]:
    if issue_set == ISSUE_SET_HARD_FAILURES:
        return [record for record in records if is_hard_failure(record)]
    if issue_set == ISSUE_SET_ALL_NON_READY:
        return [record for record in records if is_non_ready(record)]
    raise ValueError(f"Unsupported issue set: {issue_set}")


def _auto_target(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _record_matches_target(record: dict[str, Any], target: str) -> bool:
    if target == "none":
        return False
    return (
        primary_issue(record) == target
        or target in _list_field(record, "validation_flags")
        or target in _list_field(record, "review_flags")
        or target == _field(record, "mapping_failure_reason")
    )


def _sample_record(record: dict[str, Any], iteration_dir: Path, output_root: Path) -> dict[str, Any]:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    question_image_path = str(record.get("question_image_path") or _first(record.get("question_image_paths")) or "")
    mark_scheme_image_path = str(record.get("mark_scheme_image_path") or _first(record.get("mark_scheme_image_paths")) or "")
    return {
        "question_id": record.get("question_id"),
        "paper": record.get("paper"),
        "paper_family": record.get("paper_family"),
        "question_number": record.get("question_number"),
        "primary_issue": primary_issue(record),
        "validation_status": _field(record, "validation_status"),
        "mapping_status": _field(record, "mapping_status"),
        "visual_curation_status": _field(record, "visual_curation_status"),
        "text_only_status": _field(record, "text_only_status"),
        "question_text_trust": _field(record, "question_text_trust"),
        "topic": record.get("topic"),
        "topic_confidence": _field(record, "topic_confidence"),
        "question_image_path": question_image_path,
        "question_image_src": _gallery_asset_src(question_image_path, iteration_dir, output_root),
        "mark_scheme_image_path": mark_scheme_image_path,
        "mark_scheme_image_src": _gallery_asset_src(mark_scheme_image_path, iteration_dir, output_root),
        "source_pdf": notes.get("source_pdf"),
        "mark_scheme_source_pdf": notes.get("mark_scheme_source_pdf"),
        "mapping_failure_reason": _field(record, "mapping_failure_reason"),
        "question_crop_confidence": _field(record, "question_crop_confidence"),
        "mark_scheme_crop_confidence": _field(record, "mark_scheme_crop_confidence"),
        "question_total_detected": _field(record, "question_total_detected"),
        "mark_scheme_total_detected": _field(record, "mark_scheme_total_detected"),
        "paper_total_expected": _field(record, "paper_total_expected"),
        "paper_total_detected": _field(record, "paper_total_detected"),
        "validation_flags": _list_field(record, "validation_flags"),
        "review_flags": _list_field(record, "review_flags"),
        "visual_reason_flags": _list_field(record, "visual_reason_flags"),
        "extraction_quality_flags": _list_field(record, "extraction_quality_flags"),
        "question_text_snippet": str(record.get("question_text") or "")[:800],
    }


def _suspicious_crop_sample_record(
    candidate: dict[str, Any],
    record: dict[str, Any],
    iteration_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    sample = _sample_record(record, iteration_dir, output_root)
    candidate_path = str(candidate.get("path") or "")
    sample.update(
        {
            "primary_issue": str(candidate.get("primary_issue") or "suspicious_rendered_crop_dimensions"),
            "suspicious_crop": {
                "image_kind": candidate.get("image_kind"),
                "path": candidate_path,
                "width_px": candidate.get("width_px"),
                "height_px": candidate.get("height_px"),
                "aspect_ratio": candidate.get("aspect_ratio"),
                "content_bbox": candidate.get("content_bbox"),
                "blank_top_ratio": candidate.get("blank_top_ratio"),
                "blank_bottom_ratio": candidate.get("blank_bottom_ratio"),
                "content_area_ratio": candidate.get("content_area_ratio"),
                "reasons": candidate.get("reasons") if isinstance(candidate.get("reasons"), list) else [],
            },
            "suspicious_crop_image_src": _gallery_asset_src(candidate_path, iteration_dir, output_root),
        }
    )
    return sample


def _suspicious_crop_review_candidates(audit: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = [
        {**candidate, "primary_issue": "suspicious_rendered_crop_dimensions"}
        for candidate in audit.get("suspicious_rendered_crop_review_candidates") or []
        if isinstance(candidate, dict)
    ]
    whitespace = [
        {**candidate, "primary_issue": "suspicious_rendered_crop_whitespace"}
        for candidate in audit.get("suspicious_rendered_crop_whitespace_review_candidates") or []
        if isinstance(candidate, dict)
    ]
    return [*dimensions, *whitespace]


def _gallery_html(sample_payload: dict[str, Any]) -> str:
    questions = sample_payload.get("questions") if isinstance(sample_payload.get("questions"), list) else []
    cards = "\n".join(_gallery_card(record) for record in questions)
    root_options = "\n".join(
        f'<option value="{escape(value)}">{escape(ROOT_CAUSE_LABELS[value])}</option>' for value in ROOT_CAUSE_OPTIONS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Extraction Triage - {escape(str(sample_payload.get("target") or ""))}</title>
  <style>
    :root {{
      color-scheme: light;
      --border: #cfd6df;
      --ink: #17202a;
      --muted: #5f6c7a;
      --panel: #f7f9fb;
      --accent: #0b6bcb;
      --danger: #b42318;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 16px 24px;
      background: white;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      display: grid;
      gap: 24px;
      padding: 24px;
    }}
    article {{
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
    }}
    h2 {{
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .issue {{
      color: var(--danger);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    .body {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr) 360px;
      gap: 16px;
      padding: 16px;
    }}
    figure {{
      margin: 0;
      min-width: 0;
    }}
    figcaption {{
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 680px;
      object-fit: contain;
      border: 1px solid var(--border);
      background: white;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 180px;
      border: 1px dashed var(--border);
      color: var(--muted);
      background: var(--panel);
    }}
    dl {{
      display: grid;
      grid-template-columns: 125px minmax(0, 1fr);
      gap: 6px 10px;
      margin: 0 0 14px;
      font-size: 13px;
    }}
    dt {{
      color: var(--muted);
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .flags {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 14px;
    }}
    .flag {{
      padding: 3px 6px;
      border-radius: 4px;
      background: #eef2f6;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
    }}
    textarea, select {{
      width: 100%;
      box-sizing: border-box;
      margin: 5px 0 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px;
      font: inherit;
      font-size: 13px;
      background: white;
    }}
    textarea {{
      min-height: 88px;
      resize: vertical;
    }}
    button {{
      width: 100%;
      border: 0;
      border-radius: 6px;
      padding: 9px 10px;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
    }}
    .save-status {{
      min-height: 18px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      max-height: 180px;
      overflow: auto;
      background: var(--panel);
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      .body {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Extraction Triage</h1>
    <div class="summary">
      <span>Issue set: {escape(str(sample_payload.get("issue_set") or ""))}</span>
      <span>Target: {escape(str(sample_payload.get("target") or ""))}</span>
      <span>Seed: {escape(str(sample_payload.get("seed") or ""))}</span>
      <span>Sample: {escape(str(sample_payload.get("sampled_count") or 0))}</span>
    </div>
  </header>
  <main>
    {cards}
  </main>
  <template id="root-cause-options">{root_options}</template>
  <script>
    document.querySelectorAll("select[data-root-cause]").forEach((select) => {{
      select.innerHTML = document.getElementById("root-cause-options").innerHTML;
    }});
    document.querySelectorAll("button[data-save]").forEach((button) => {{
      button.addEventListener("click", async () => {{
        const card = button.closest("article");
        const payload = {{
          question_id: card.dataset.questionId,
          root_cause: card.querySelector("select[data-root-cause]").value,
          confidence: card.querySelector("select[data-confidence]").value,
          notes: card.querySelector("textarea").value,
          saved_at: new Date().toISOString()
        }};
        const status = card.querySelector(".save-status");
        status.textContent = "Saving...";
        try {{
          const response = await fetch("review", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify(payload)
          }});
          if (!response.ok) throw new Error(await response.text());
          status.textContent = "Saved";
        }} catch (error) {{
          status.textContent = "Save failed: " + error.message;
        }}
      }});
    }});
  </script>
</body>
</html>
"""


def _gallery_card(record: dict[str, Any]) -> str:
    qid = str(record.get("question_id") or "")
    question_img = _image_or_missing(record.get("question_image_src"), "Question crop")
    mark_img = _image_or_missing(record.get("mark_scheme_image_src"), "Mark-scheme crop")
    suspicious_crop = record.get("suspicious_crop") if isinstance(record.get("suspicious_crop"), dict) else {}
    flags = _flag_group(record.get("validation_flags"), "Validation flags")
    review_flags = _flag_group(record.get("review_flags"), "Review flags")
    visual_flags = _flag_group(record.get("visual_reason_flags"), "Visual flags")
    metadata = {
        "Validation": record.get("validation_status"),
        "Mapping": record.get("mapping_status"),
        "Visual": record.get("visual_curation_status"),
        "Text only": record.get("text_only_status"),
        "Question crop": record.get("question_crop_confidence"),
        "Mark crop": record.get("mark_scheme_crop_confidence"),
        "Q total": record.get("question_total_detected"),
        "MS total": record.get("mark_scheme_total_detected"),
        "Paper total": record.get("paper_total_detected"),
        "Mapping reason": record.get("mapping_failure_reason"),
        "Question image": record.get("question_image_path"),
        "Mark image": record.get("mark_scheme_image_path"),
        "Suspicious crop kind": suspicious_crop.get("image_kind"),
        "Suspicious crop path": suspicious_crop.get("path"),
        "Suspicious dimensions": _suspicious_dimensions_label(suspicious_crop),
        "Suspicious whitespace": _suspicious_whitespace_label(suspicious_crop),
        "Suspicious reasons": ", ".join(str(item) for item in suspicious_crop.get("reasons") or []),
        "Source PDF": record.get("source_pdf"),
    }
    metadata_html = _definition_list(metadata)
    text = escape(str(record.get("question_text_snippet") or ""))
    return f"""<article data-question-id="{escape(qid)}">
  <div class="card-head">
    <div>
      <h2>{escape(qid)} / Q{escape(str(record.get("question_number") or ""))}</h2>
      <div>{escape(str(record.get("paper_family") or ""))} {escape(str(record.get("paper") or ""))}</div>
    </div>
    <div class="issue">{escape(str(record.get("primary_issue") or ""))}</div>
  </div>
  <div class="body">
    <figure>
      <figcaption>Question crop</figcaption>
      {question_img}
    </figure>
    <figure>
      <figcaption>Mark-scheme crop</figcaption>
      {mark_img}
    </figure>
    <section>
      {metadata_html}
      {flags}
      {visual_flags}
      {review_flags}
      <pre>{text}</pre>
      <label>Root cause<select data-root-cause></select></label>
      <label>Confidence<select data-confidence>
        <option value="medium">Medium</option>
        <option value="high">High</option>
        <option value="low">Low</option>
      </select></label>
      <label>Notes<textarea placeholder="What is wrong, and what code path likely owns it?"></textarea></label>
      <button type="button" data-save>Save review</button>
      <div class="save-status"></div>
    </section>
  </div>
</article>"""


def _suspicious_dimensions_label(suspicious_crop: dict[str, Any]) -> str:
    width = suspicious_crop.get("width_px")
    height = suspicious_crop.get("height_px")
    aspect = suspicious_crop.get("aspect_ratio")
    if width is None or height is None:
        return ""
    label = f"{width}x{height}"
    if aspect is not None:
        label += f" aspect {aspect}"
    return label


def _suspicious_whitespace_label(suspicious_crop: dict[str, Any]) -> str:
    values = []
    for label, key in [
        ("top", "blank_top_ratio"),
        ("bottom", "blank_bottom_ratio"),
        ("content area", "content_area_ratio"),
    ]:
        value = suspicious_crop.get(key)
        if value is not None:
            values.append(f"{label} {value}")
    return ", ".join(values)


def _image_or_missing(src: Any, alt: str) -> str:
    if not src:
        return '<div class="missing">Missing image</div>'
    return f'<img src="{escape(str(src), quote=True)}" alt="{escape(alt, quote=True)}" loading="lazy">'


def _definition_list(values: dict[str, Any]) -> str:
    rows = []
    for key, value in values.items():
        if value is None or value == "" or value == []:
            continue
        rows.append(f"<dt>{escape(key)}</dt><dd>{escape(str(value))}</dd>")
    return "<dl>" + "".join(rows) + "</dl>"


def _flag_group(values: Any, label: str) -> str:
    if not isinstance(values, list) or not values:
        return ""
    flags = "".join(f'<span class="flag">{escape(str(value))}</span>' for value in values)
    return f"<div>{escape(label)}</div><div class=\"flags\">{flags}</div>"


def _record_movements(
    baseline_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    *,
    sample_limit: int = 30,
) -> dict[str, Any]:
    baseline_by_id = {str(record.get("question_id")): record for record in baseline_records if record.get("question_id")}
    current_by_id = {str(record.get("question_id")): record for record in current_records if record.get("question_id")}
    shared_ids = sorted(set(baseline_by_id) & set(current_by_id))
    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []
    for question_id in shared_ids:
        movement = _status_movement(current_by_id[question_id], baseline_by_id[question_id])
        if movement["improved"] and len(improved) < sample_limit:
            improved.append({"question_id": question_id, "fields": movement["improved"]})
        if movement["worsened"] and len(worsened) < sample_limit:
            worsened.append({"question_id": question_id, "fields": movement["worsened"]})
    return {
        "shared_record_count": len(shared_ids),
        "missing_from_current_count": len(set(baseline_by_id) - set(current_by_id)),
        "new_in_current_count": len(set(current_by_id) - set(baseline_by_id)),
        "improved_records": improved,
        "worsened_records": worsened,
    }


def _status_movement(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, list[str]]:
    improved: list[str] = []
    worsened: list[str] = []
    for field, order in STATUS_ORDERS.items():
        current_value = _field(current, field)
        baseline_value = _field(baseline, field)
        if current_value not in order or baseline_value not in order or current_value == baseline_value:
            continue
        if order[str(current_value)] < order[str(baseline_value)]:
            improved.append(field)
        else:
            worsened.append(field)
    return {"improved": improved, "worsened": worsened}


def _issue_count_deltas(baseline_counts: dict[str, int], current_counts: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(baseline_counts) | set(current_counts))
    return {key: current_counts.get(key, 0) - baseline_counts.get(key, 0) for key in keys}


def _iteration_dir(triage_root: Path, iteration: str | Path | None) -> Path:
    if iteration is not None and str(iteration):
        iteration_path = Path(iteration)
        if iteration_path.is_absolute() or iteration_path.parent != Path("."):
            return iteration_path
        return triage_root / iteration_path
    triage_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in triage_root.glob("iteration_[0-9][0-9][0-9]") if path.is_dir())
    next_number = 1
    if existing:
        next_number = max(int(name.rsplit("_", 1)[1]) for name in existing) + 1
    return triage_root / f"iteration_{next_number:03d}"


def _record_sort_key(record: dict[str, Any]) -> tuple[str, str, int, str]:
    question_number = str(record.get("question_number") or "")
    try:
        number = int(question_number)
    except ValueError:
        number = 999
    return (
        str(record.get("paper_family") or ""),
        str(record.get("paper") or ""),
        number,
        str(record.get("question_id") or ""),
    )


def _gallery_asset_src(image_path: str, iteration_dir: Path, output_root: Path) -> str:
    if not image_path:
        return ""
    absolute_path = output_root / image_path
    return os.path.relpath(absolute_path, iteration_dir).replace(os.sep, "/")


def _infer_output_root(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if input_path.parent.name == "json":
        return input_path.parent.parent
    return input_path.parent


def _infer_output_root_from_iteration(iteration_dir: Path) -> Path:
    if iteration_dir.parent.name == "triage":
        return iteration_dir.parent.parent
    return iteration_dir.parent


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _field(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record.get(key)
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(key)
    return None


def _list_field(record: dict[str, Any], key: str) -> list[str]:
    value = _field(record, key)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


class _TriageRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, iteration_dir: Path, **kwargs: Any) -> None:
        self.iteration_dir = iteration_dir
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not path.endswith("/review"):
            self.send_error(404, "Unsupported triage endpoint")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            self.send_error(413, "Review payload is too large")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Review payload must be JSON")
            return
        if not isinstance(payload, dict) or not payload.get("question_id"):
            self.send_error(400, "Review payload requires question_id")
            return
        if payload.get("root_cause") not in ROOT_CAUSE_OPTIONS:
            self.send_error(400, "Review payload has unsupported root_cause")
            return

        review_path = self.iteration_dir / "review.jsonl"
        with review_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        response = json.dumps({"ok": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)
