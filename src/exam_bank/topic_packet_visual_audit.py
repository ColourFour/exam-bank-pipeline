from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Sequence

import fitz

from .atomic_json import write_atomic_json


TOPIC_PACKET_VISUAL_AUDIT_BATCH_SCHEMA = "exam_bank.topic_packet_visual_audit.batch"
TOPIC_PACKET_VISUAL_AUDIT_BATCH_SCHEMA_VERSION = 1
TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION = "topic_packet_visual_audit_decision_v1"
TOPIC_PACKET_VISUAL_AUDIT_IMPORT_SCHEMA = "exam_bank.topic_packet_visual_audit.import"
TOPIC_PACKET_VISUAL_AUDIT_REGISTRY_SCHEMA = "exam_bank.topic_packet_visual_audit.registry"
TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION = "topic_packet_visual_audit_9709_v1"

DEFAULT_PACKETS_ROOT = Path("output/topic_packets")
DEFAULT_AUDIT_OUT_DIR = Path("data/review/topic_packet_visual_audit_2026_07_07")
DEFAULT_RENDER_ROOT = Path("output/audits/topic_packet_visual_audit_2026_07_07")
DEFAULT_ARTIFACT_ROOT = Path("output")

DECISION_STATUSES = {"pass", "bug", "needs_human", "not_reviewable"}
BUG_CATEGORIES = {
    "question_crop",
    "diagram_missing_or_clipped",
    "mark_scheme_crop",
    "wrong_or_missing_mark_scheme",
    "packet_layout",
    "downsampled_unreadable",
    "header_overlap",
    "wrong_topic_packet",
    "metadata_path_mismatch",
}
RESOLUTION_STATUSES = {"open", "fixed", "waived_with_reason", "not_applicable"}
FIX_OWNER_AREAS = {
    "question_png_regeneration",
    "mark_scheme_png_regeneration",
    "mark_scheme_path_promotion",
    "topic_routing_review",
    "topic_packet_layout",
    "packet_visual_audit",
    "unknown",
}
GENERALIZATION_DECISIONS = {"generalize", "targeted_exception", "needs_more_evidence", "not_applicable"}


DEFAULT_SEED_BUGS: tuple[dict[str, str], ...] = (
    {"type": "problem", "user_label": "2015 June P12 Q2", "question_id": "12summer15_q02"},
    {"type": "problem", "user_label": "2020 June P13 Q5", "question_id": "13summer20_q05"},
    {"type": "problem", "user_label": "2011 November P13 Q6", "question_id": "13winter11_q06"},
    {"type": "problem", "user_label": "2013 November P12 Q2", "question_id": "12winter13_q02"},
    {"type": "problem", "user_label": "2012 June P12 Q6", "question_id": "12summer12_q06"},
    {"type": "problem", "user_label": "2024 June P13 Q3", "question_id": "13summer24_q03"},
    {"type": "problem", "user_label": "2019 June P13 Q3", "question_id": "13summer19_q03"},
    {"type": "problem", "user_label": "2013 November P13 Q6", "question_id": "13winter13_q06"},
    {"type": "problem", "user_label": "2019 November P12 Q4", "question_id": "12winter19_q04"},
    {"type": "problem", "user_label": "2022 November P12 Q10", "question_id": "12winter22_q10"},
    {"type": "problem", "user_label": "2019 November P11 Q8", "question_id": "11winter19_q08"},
    {"type": "problem", "user_label": "2023 November P13 Q10", "question_id": "13winter23_q10"},
    {"type": "mark_scheme", "user_label": "2019 June P11 Q3 mark scheme", "question_id": "11summer19_q03"},
    {"type": "mark_scheme", "user_label": "2018 June P13 Q5 mark scheme", "question_id": "13summer18_q05"},
)


class TopicPacketVisualAuditError(RuntimeError):
    pass


def add_topic_packet_visual_audit_cli_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="topic_packet_visual_audit_command", required=True)

    build = subparsers.add_parser("build-batch", help="Render packet pages and build a page-level visual audit batch.")
    build.add_argument("--packets-root", type=Path, default=DEFAULT_PACKETS_ROOT)
    build.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    build.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    build.add_argument("--out-dir", type=Path, default=DEFAULT_AUDIT_OUT_DIR)
    build.add_argument("--dpi", type=int, default=144)
    build.add_argument("--paper-family", choices=["p1", "p3", "p4", "p5"], default=None)
    build.add_argument("--topic", default=None)
    build.add_argument("--limit-pages", type=int, default=None)
    build.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run", help="Run append/resumable visual decisions for a rendered page batch.")
    run.add_argument("--batch", type=Path, required=True)
    run.add_argument("--out", type=Path, default=None)
    run.add_argument("--max-records", type=int, default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--provider", choices=["codex", "openai"], default="codex")
    run.add_argument("--codex-bin", default=None, help="Codex CLI path for --provider codex. Defaults to PATH lookup.")
    run.add_argument("--codex-sandbox", choices=["read-only", "workspace-write"], default="read-only")
    run.add_argument("--dry-run", action="store_true")

    import_decisions = subparsers.add_parser(
        "import-decisions",
        help="Validate page visual decisions and write a consolidated bug registry.",
    )
    import_decisions.add_argument("--batch", type=Path, required=True)
    import_decisions.add_argument("--decisions", type=Path, required=True)
    import_decisions.add_argument("--out", type=Path, default=None)
    import_decisions.add_argument("--markdown-out", type=Path, default=None)
    import_decisions.add_argument("--allow-incomplete", action="store_true")
    import_decisions.add_argument("--dry-run", action="store_true")


def run_topic_packet_visual_audit_from_args(args: argparse.Namespace) -> dict[str, Any]:
    command = args.topic_packet_visual_audit_command
    if command == "build-batch":
        batch = build_topic_packet_visual_audit_batch(
            packets_root=args.packets_root,
            artifact_root=args.artifact_root,
            render_root=args.render_root,
            out_dir=args.out_dir,
            dpi=args.dpi,
            paper_family=args.paper_family,
            topic=args.topic,
            limit_pages=args.limit_pages,
            dry_run=bool(args.dry_run),
        )
        return _batch_console_summary(batch, out_dir=args.out_dir, dry_run=bool(args.dry_run))
    if command == "run":
        out_path = args.out or args.batch.parent / "topic_packet_visual_audit_decisions.jsonl"
        return run_topic_packet_visual_audit_reviews(
            batch_path=args.batch,
            out_path=out_path,
            max_records=args.max_records,
            model=args.model,
            provider=args.provider,
            codex_bin=args.codex_bin,
            codex_sandbox=args.codex_sandbox,
            dry_run=bool(args.dry_run),
        )
    if command == "import-decisions":
        out_path = args.out or args.batch.parent / "topic_packet_visual_bug_registry.v1.json"
        markdown_path = args.markdown_out or Path(out_path).with_suffix(".md")
        return import_topic_packet_visual_audit_decisions(
            batch_path=args.batch,
            decisions_path=args.decisions,
            out_path=out_path,
            markdown_out_path=markdown_path,
            allow_incomplete=bool(args.allow_incomplete),
            dry_run=bool(args.dry_run),
        )
    raise TopicPacketVisualAuditError(f"Unhandled topic packet visual audit command: {command}")


def build_topic_packet_visual_audit_batch(
    *,
    packets_root: str | Path = DEFAULT_PACKETS_ROOT,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    render_root: str | Path = DEFAULT_RENDER_ROOT,
    out_dir: str | Path = DEFAULT_AUDIT_OUT_DIR,
    dpi: int = 144,
    paper_family: str | None = None,
    topic: str | None = None,
    limit_pages: int | None = None,
    seed_bugs: Sequence[dict[str, str]] = DEFAULT_SEED_BUGS,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dpi <= 0:
        raise TopicPacketVisualAuditError("--dpi must be positive.")
    if limit_pages is not None and limit_pages < 0:
        raise TopicPacketVisualAuditError("--limit-pages must be zero or greater.")

    packets_root = Path(packets_root)
    artifact_root = Path(artifact_root)
    render_root = Path(render_root)
    out_dir = Path(out_dir)
    selected_family = str(paper_family or "").strip()
    selected_topic = str(topic or "").strip()
    seed_index = _seed_index(seed_bugs)

    rows: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for manifest_path in sorted(packets_root.glob("**/manifest.json")):
        manifest = _read_json(manifest_path)
        packet_family = str(manifest.get("paper_family") or "")
        packet_topic = str(manifest.get("topic_id") or manifest.get("topic") or "")
        if selected_family and packet_family != selected_family:
            continue
        if selected_topic and packet_topic != selected_topic:
            continue
        packet_id = _packet_id(manifest, manifest_path)
        pdf_path = _resolve_pdf_path(manifest_path, manifest)
        if not pdf_path.is_file():
            skipped.append({"manifest_path": str(manifest_path), "packet_id": packet_id, "reason": "missing_pdf", "pdf_path": str(pdf_path)})
            continue
        page_count = _pdf_page_count(pdf_path)
        packet_rows = []
        for page_number in range(1, page_count + 1):
            if limit_pages is not None and len(rows) >= limit_pages:
                break
            page_image = render_root / packet_id / f"page_{page_number:04d}.png"
            if not dry_run:
                _render_pdf_page(pdf_path, page_number=page_number, output_path=page_image, dpi=dpi)
            row = _page_row(
                manifest=manifest,
                manifest_path=manifest_path,
                pdf_path=pdf_path,
                packet_id=packet_id,
                page_number=page_number,
                page_image_path=page_image,
                artifact_root=artifact_root,
                seed_index=seed_index,
            )
            packet_rows.append(row)
            rows.append(row)
        packets.append(
            {
                "packet_id": packet_id,
                "paper_family": packet_family,
                "topic_id": packet_topic,
                "manifest_path": str(manifest_path),
                "pdf_path": str(pdf_path),
                "page_count": page_count,
                "selected_page_count": len(packet_rows),
                "seed_bug_page_count": sum(1 for row in packet_rows if row["seed_bug_refs"]),
            }
        )
        if limit_pages is not None and len(rows) >= limit_pages:
            break

    batch_id = _batch_id(rows)
    for index, row in enumerate(rows, start=1):
        row["batch_id"] = batch_id
        row["batch_index"] = index

    batch = {
        "schema_name": TOPIC_PACKET_VISUAL_AUDIT_BATCH_SCHEMA,
        "schema_version": TOPIC_PACKET_VISUAL_AUDIT_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": _utc_now_iso(),
        "dry_run": dry_run,
        "source_files": {
            "packets_root": str(packets_root),
            "artifact_root": str(artifact_root),
            "render_root": str(render_root),
        },
        "selection": {
            "paper_family": selected_family,
            "topic": selected_topic,
            "limit_pages": limit_pages,
            "packet_count": len(packets),
            "page_count": len(rows),
            "seed_bug_page_count": sum(1 for row in rows if row["seed_bug_refs"]),
            "skipped_packet_count": len(skipped),
            "page_section_counts": dict(Counter(str(row.get("page_section") or "") for row in rows)),
            "packet_page_counts": {packet["packet_id"]: packet["selected_page_count"] for packet in packets},
        },
        "review_policy": (
            "Review rendered topic-packet pages for readability, clipped diagrams, wrong/missing mark schemes, "
            "packet layout defects, and metadata/path mismatches. Source question and mark-scheme images remain "
            "canonical evidence; this audit identifies visual packet defects and routes repairs."
        ),
        "decision_version": TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION,
        "prompt_version": TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION,
        "decision_schema": topic_packet_visual_audit_decision_schema(),
        "seed_bugs": list(seed_bugs),
        "packets": packets,
        "rows": rows,
        "skipped_packets": skipped,
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_atomic_json(batch, out_dir / "topic_packet_visual_audit_batch.json", sort_keys=True)
        (out_dir / "topic_packet_visual_audit_batch.md").write_text(render_batch_markdown(batch), encoding="utf-8")
    return batch


def _batch_console_summary(batch: dict[str, Any], *, out_dir: Path, dry_run: bool) -> dict[str, Any]:
    selection = batch.get("selection") if isinstance(batch.get("selection"), dict) else {}
    return {
        "schema_name": batch.get("schema_name"),
        "schema_version": batch.get("schema_version"),
        "batch_id": batch.get("batch_id"),
        "dry_run": dry_run,
        "out_dir": "" if dry_run else str(out_dir),
        "batch_path": "" if dry_run else str(out_dir / "topic_packet_visual_audit_batch.json"),
        "markdown_path": "" if dry_run else str(out_dir / "topic_packet_visual_audit_batch.md"),
        "selection": selection,
        "skipped_packets": batch.get("skipped_packets") or [],
    }


def run_topic_packet_visual_audit_reviews(
    *,
    batch_path: str | Path,
    out_path: str | Path,
    max_records: int | None = None,
    model: str | None = None,
    provider: str = "codex",
    codex_bin: str | Path | None = None,
    codex_sandbox: str = "read-only",
    dry_run: bool = False,
) -> dict[str, Any]:
    if max_records is not None and max_records < 0:
        raise TopicPacketVisualAuditError("--max-records must be zero or greater.")
    batch_path = Path(batch_path)
    out_path = Path(out_path)
    batch = _read_json(batch_path)
    selected_model = model or ("gpt-5-mini" if provider == "openai" else "codex-default")
    rows = [row for row in batch.get("rows") or [] if isinstance(row, dict)]
    done = _existing_decision_row_ids(out_path)
    pending = [row for row in rows if str(row.get("row_id") or "") not in done]
    if max_records is not None:
        pending = pending[:max_records]
    manifest = {
        "schema_name": "exam_bank.topic_packet_visual_audit.runner_manifest",
        "provider": provider,
        "model": selected_model,
        "prompt_version": TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION,
        "dry_run": dry_run,
        "batch_path": str(batch_path),
        "out_path": str(out_path),
        "pending_count": len(pending),
        "resumed_count": len(done),
        "created_at": _utc_now_iso(),
    }
    if dry_run:
        if provider == "codex":
            manifest["codex_bin"] = str(codex_bin or _find_codex_bin(required=False) or "")
            manifest["codex_sandbox"] = codex_sandbox
        return manifest
    if provider == "codex":
        resolved_codex_bin = _find_codex_bin(codex_bin=codex_bin, required=True)
        manifest["codex_bin"] = str(resolved_codex_bin)
        manifest["codex_sandbox"] = codex_sandbox
        _run_codex_page_reviews(
            codex_bin=Path(str(resolved_codex_bin)),
            batch=batch,
            rows=pending,
            out_path=out_path,
            model=model or "",
            sandbox=codex_sandbox,
        )
        return manifest
    if provider != "openai":
        raise TopicPacketVisualAuditError("topic packet visual audit runner supports provider=codex or provider=openai.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise TopicPacketVisualAuditError("topic packet visual audit runner requires OPENAI_API_KEY when --provider openai is used.")

    from openai import OpenAI

    client = OpenAI()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in pending:
            try:
                decision = _request_openai_page_review(client=client, model=selected_model, row=row)
            except Exception as exc:
                decision = _not_reviewable_error_decision(row, model=selected_model, error=exc)
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return manifest


def import_topic_packet_visual_audit_decisions(
    *,
    batch_path: str | Path,
    decisions_path: str | Path,
    out_path: str | Path,
    markdown_out_path: str | Path | None = None,
    allow_incomplete: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    batch_path = Path(batch_path)
    decisions_path = Path(decisions_path)
    out_path = Path(out_path)
    markdown_out_path = Path(markdown_out_path) if markdown_out_path else None
    batch = _read_json(batch_path)
    rows = [row for row in batch.get("rows") or [] if isinstance(row, dict)]
    rows_by_id = {str(row.get("row_id") or ""): row for row in rows if str(row.get("row_id") or "")}
    decisions = _read_decisions(decisions_path)

    errors: list[str] = []
    warnings: list[str] = []
    accepted_by_row: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, decision in enumerate(decisions, start=1):
        if not isinstance(decision, dict):
            errors.append(f"decision:{index}:not_object")
            continue
        row_id = str(decision.get("row_id") or "").strip()
        if not row_id:
            errors.append(f"decision:{index}:missing_row_id")
            continue
        if row_id in seen:
            errors.append(f"decision:{index}:{row_id}:duplicate_decision")
            continue
        seen.add(row_id)
        validation_errors = validate_topic_packet_visual_audit_decision(decision, batch_rows=rows_by_id)
        if validation_errors:
            rejected.append(_decision_report_row(decision, validation_errors))
            errors.extend(f"decision:{index}:{row_id}:{error}" for error in validation_errors)
            continue
        accepted_by_row[row_id] = decision

    missing_ids = [row_id for row_id in rows_by_id if row_id not in accepted_by_row]
    if missing_ids:
        errors.extend(f"missing_decision:{row_id}" for row_id in missing_ids)
    complete = not missing_ids and len(accepted_by_row) == len(rows_by_id)
    hard_errors = [error for error in errors if not error.startswith("missing_decision:")]
    ok = not hard_errors and (complete or allow_incomplete)
    if allow_incomplete and missing_ids:
        warnings.extend(f"missing_decision:{row_id}" for row_id in missing_ids)

    registry = _registry_payload(
        batch=batch,
        batch_path=batch_path,
        decisions_path=decisions_path,
        out_path=out_path,
        accepted_by_row=accepted_by_row,
        rows=rows,
        complete=complete,
        allow_incomplete=allow_incomplete,
        errors=errors,
        warnings=warnings,
        rejected=rejected,
    )
    report = {
        "schema_name": TOPIC_PACKET_VISUAL_AUDIT_IMPORT_SCHEMA,
        "schema_version": 1,
        "ok": ok,
        "complete": complete,
        "dry_run": dry_run,
        "allow_incomplete": allow_incomplete,
        "batch_path": str(batch_path),
        "decisions_path": str(decisions_path),
        "out_path": str(out_path),
        "markdown_out_path": str(markdown_out_path or ""),
        "batch_page_count": len(rows_by_id),
        "decision_count": len(decisions),
        "accepted_count": len(accepted_by_row),
        "bug_record_count": len(registry["bug_records"]),
        "rejected_count": len(rejected),
        "missing_count": len(missing_ids),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "rejected": rejected,
    }
    if not dry_run and ok:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_json(registry, out_path, sort_keys=True)
        if markdown_out_path:
            markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_out_path.write_text(render_registry_markdown(registry), encoding="utf-8")
    return report


def validate_topic_packet_visual_audit_decision(
    decision: dict[str, Any],
    *,
    batch_rows: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if decision.get("decision_version") != TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION:
        errors.append("invalid_decision_version")
    if str(decision.get("prompt_version") or "").strip() not in {"", TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION}:
        errors.append("invalid_prompt_version")
    row_id = str(decision.get("row_id") or "").strip()
    row = batch_rows.get(row_id)
    if not row_id:
        errors.append("missing_row_id")
    elif row is None:
        errors.append("unknown_row_id")
    status = str(decision.get("status") or "").strip()
    if status not in DECISION_STATUSES:
        errors.append("invalid_status")
    categories = _string_list(decision.get("categories"))
    unknown_categories = sorted(set(categories) - BUG_CATEGORIES)
    errors.extend(f"unknown_category:{category}" for category in unknown_categories)
    if status in {"bug", "needs_human"} and not categories:
        errors.append("missing_categories_for_defect")
    resolution_status = str(decision.get("resolution_status") or _default_resolution_status(decision, row)).strip()
    if resolution_status not in RESOLUTION_STATUSES:
        errors.append("invalid_resolution_status")
    fix_owner = str(decision.get("fix_owner_area") or "unknown").strip()
    if fix_owner not in FIX_OWNER_AREAS:
        errors.append("invalid_fix_owner_area")
    generalization = str(decision.get("generalization_decision") or "needs_more_evidence").strip()
    if generalization not in GENERALIZATION_DECISIONS:
        errors.append("invalid_generalization_decision")
    if resolution_status == "waived_with_reason" and not str(decision.get("rationale") or "").strip():
        errors.append("waived_without_rationale")
    return errors


def topic_packet_visual_audit_decision_schema() -> dict[str, Any]:
    properties = {
        "decision_version": {"type": "string", "const": TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION},
        "prompt_version": {"type": "string", "const": TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION},
        "row_id": {"type": "string"},
        "packet_id": {"type": "string"},
        "page_number": {"type": "integer"},
        "status": {"type": "string", "enum": sorted(DECISION_STATUSES)},
        "categories": {"type": "array", "items": {"type": "string", "enum": sorted(BUG_CATEGORIES)}},
        "likely_root_cause": {"type": "string"},
        "fix_owner_area": {"type": "string", "enum": sorted(FIX_OWNER_AREAS)},
        "generalization_decision": {"type": "string", "enum": sorted(GENERALIZATION_DECISIONS)},
        "resolution_status": {"type": "string", "enum": sorted(RESOLUTION_STATUSES)},
        "rationale": {"type": "string"},
        "before_evidence": {"type": "array", "items": {"type": "string"}},
        "after_evidence": {"type": "array", "items": {"type": "string"}},
        "source": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def render_batch_markdown(batch: dict[str, Any]) -> str:
    lines = [
        "# Topic Packet Visual Audit Batch",
        "",
        f"- Batch ID: `{batch.get('batch_id')}`",
        f"- Packets: `{batch.get('selection', {}).get('packet_count')}`",
        f"- Pages: `{batch.get('selection', {}).get('page_count')}`",
        f"- Seed bug pages: `{batch.get('selection', {}).get('seed_bug_page_count')}`",
        "",
        "| Row | Packet | Page | Section | Problems | Questions | Seed | Image |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in batch.get("rows") or []:
        seed = ", ".join(ref.get("user_label", "") for ref in row.get("seed_bug_refs") or [])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('row_id')}`",
                    f"`{row.get('packet_id')}`",
                    str(row.get("page_number") or ""),
                    str(row.get("page_section") or ""),
                    ", ".join(str(item) for item in row.get("related_problem_numbers") or []),
                    ", ".join(str(item) for item in row.get("related_question_ids") or []),
                    seed,
                    str(row.get("page_image_path") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_registry_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# Topic Packet Visual Bug Registry",
        "",
        f"- Complete: `{registry.get('complete')}`",
        f"- Bug records: `{len(registry.get('bug_records') or [])}`",
        f"- Seed bugs tracked: `{registry.get('summary', {}).get('seed_bug_record_count')}`",
        "",
        "| Bug | Status | Resolution | Categories | Packet | Page | Root cause |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for record in registry.get("bug_records") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record.get('bug_id')}`",
                    str(record.get("decision_status") or ""),
                    str(record.get("resolution_status") or ""),
                    ", ".join(record.get("categories") or []),
                    f"`{record.get('packet_id')}`",
                    str(record.get("page_number") or ""),
                    str(record.get("likely_root_cause") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _page_row(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    pdf_path: Path,
    packet_id: str,
    page_number: int,
    page_image_path: Path,
    artifact_root: Path,
    seed_index: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    page_section = _page_section(manifest, page_number)
    related_records = _related_records(manifest, page_number=page_number, page_section=page_section, artifact_root=artifact_root)
    question_ids = _dedupe(str(record["question_id"]) for record in related_records if record.get("question_id"))
    seed_refs = _seed_refs(related_records, seed_index)
    return {
        "row_id": f"{packet_id}_page_{page_number:04d}",
        "packet_id": packet_id,
        "paper_family": str(manifest.get("paper_family") or ""),
        "topic_id": str(manifest.get("topic_id") or manifest.get("topic") or ""),
        "packet_level": str(manifest.get("packet_level") or ""),
        "manifest_path": str(manifest_path),
        "pdf_path": str(pdf_path),
        "page_number": page_number,
        "page_count": int(manifest.get("page_count") or 0),
        "page_section": page_section,
        "page_image_path": str(page_image_path),
        "related_problem_numbers": [record["problem_number"] for record in related_records if record.get("problem_number") is not None],
        "related_question_ids": question_ids,
        "related_records": related_records,
        "source_question_image_paths": _dedupe(path for record in related_records for path in record.get("question_image_paths") or []),
        "source_mark_scheme_image_paths": _dedupe(path for record in related_records for path in record.get("mark_scheme_image_paths") or []),
        "layout_warnings": _page_layout_warnings(manifest, related_records),
        "metadata_path_warnings": _metadata_path_warnings(related_records, artifact_root=artifact_root),
        "seed_bug_refs": seed_refs,
        "review_focus": _review_focus(page_section, related_records, seed_refs),
    }


def _related_records(manifest: dict[str, Any], *, page_number: int, page_section: str, artifact_root: Path) -> list[dict[str, Any]]:
    related: list[dict[str, Any]] = []
    for record in manifest.get("included_records") or []:
        if not isinstance(record, dict):
            continue
        problem_number = record.get("problem_number")
        question_id = str(record.get("question_id") or "")
        if record.get("question_start_page") == page_number:
            related.append(_related_record(record, kind="question", problem_number=problem_number, question_id=question_id, artifact_root=artifact_root))
        if record.get("answer_start_page") == page_number:
            related.append(_related_record(record, kind="answer", problem_number=problem_number, question_id=question_id, artifact_root=artifact_root))
    if related:
        return related
    if page_section == "Questions":
        return [
            _related_record(record, kind="question", problem_number=record.get("problem_number"), question_id=str(record.get("question_id") or ""), artifact_root=artifact_root)
            for record in manifest.get("included_records") or []
            if isinstance(record, dict) and record.get("question_start_page") == page_number
        ]
    return []


def _related_record(record: dict[str, Any], *, kind: str, problem_number: Any, question_id: str, artifact_root: Path) -> dict[str, Any]:
    mark_scheme_paths = _string_list(record.get("mark_scheme_image_paths"))
    question_paths = _string_list(record.get("question_image_paths"))
    return {
        "kind": kind,
        "problem_number": problem_number,
        "question_id": question_id,
        "source_label": str(record.get("source_label") or ""),
        "question_number": str(record.get("question_number") or ""),
        "section": str(record.get("section") or ""),
        "answer_available": bool(record.get("answer_available")),
        "question_image_paths": question_paths,
        "mark_scheme_image_paths": mark_scheme_paths,
        "question_image_paths_exist": [path for path in question_paths if (artifact_root / path).is_file()],
        "mark_scheme_image_paths_exist": [path for path in mark_scheme_paths if (artifact_root / path).is_file()],
        "review_reasons": _string_list(record.get("review_reasons")),
        "warnings": _string_list(record.get("warnings")),
        "question_block_height_estimate": record.get("question_block_height_estimate"),
        "answer_block_height_estimate": record.get("answer_block_height_estimate"),
    }


def _page_layout_warnings(manifest: dict[str, Any], related_records: Sequence[dict[str, Any]]) -> list[str]:
    problem_numbers = {str(record.get("problem_number")) for record in related_records if record.get("problem_number") is not None}
    paths = {Path(path).name for record in related_records for path in (record.get("question_image_paths") or []) + (record.get("mark_scheme_image_paths") or [])}
    warnings = _string_list(manifest.get("oversized_block_warnings"))
    pdf_outputs = manifest.get("pdf_outputs") if isinstance(manifest.get("pdf_outputs"), dict) else {}
    topic_packet = pdf_outputs.get("topic_packet") if isinstance(pdf_outputs.get("topic_packet"), dict) else {}
    warnings.extend(_string_list(topic_packet.get("warnings")))
    selected = []
    for warning in warnings:
        if any(f":{number}:" in warning or warning.endswith(f":{number}") for number in problem_numbers):
            selected.append(warning)
            continue
        if any(path and path in warning for path in paths):
            selected.append(warning)
    return _dedupe(selected)


def _metadata_path_warnings(related_records: Sequence[dict[str, Any]], *, artifact_root: Path) -> list[str]:
    warnings: list[str] = []
    for record in related_records:
        if record.get("kind") == "answer" and record.get("mark_scheme_image_paths") and not record.get("mark_scheme_image_paths_exist"):
            warnings.append(f"mark_scheme_paths_missing_on_disk:{record.get('question_id')}")
        if record.get("kind") == "question" and record.get("question_image_paths") and not record.get("question_image_paths_exist"):
            warnings.append(f"question_paths_missing_on_disk:{record.get('question_id')}")
    return _dedupe(warnings)


def _seed_refs(related_records: Sequence[dict[str, Any]], seed_index: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for record in related_records:
        for seed in seed_index.get(str(record.get("question_id") or ""), []):
            if seed.get("type") == "problem" and record.get("kind") == "question":
                refs.append(seed)
            elif seed.get("type") == "mark_scheme" and record.get("kind") == "answer":
                refs.append(seed)
    return _dedupe_dicts(refs)


def _review_focus(page_section: str, related_records: Sequence[dict[str, Any]], seed_refs: Sequence[dict[str, str]]) -> list[str]:
    focus = []
    if page_section == "Answers / Mark Schemes":
        focus.append("Verify mark-scheme crop is readable and belongs to the related problem.")
    else:
        focus.append("Verify question text, diagrams, labels, and headers are readable and unclipped.")
    if seed_refs:
        focus.append("This page contains a user-reported seed bug; review carefully even if the page appears acceptable.")
    if any(record.get("answer_block_height_estimate", 0) and float(record.get("answer_block_height_estimate") or 0) > 1500 for record in related_records):
        focus.append("Long answer block: check for downsampling or below-legibility scaling.")
    return focus


def _registry_payload(
    *,
    batch: dict[str, Any],
    batch_path: Path,
    decisions_path: Path,
    out_path: Path,
    accepted_by_row: dict[str, dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    complete: bool,
    allow_incomplete: bool,
    errors: Sequence[str],
    warnings: Sequence[str],
    rejected: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    bug_records = []
    decision_status_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    for row in rows:
        row_id = str(row.get("row_id") or "")
        decision = accepted_by_row.get(row_id)
        if not decision:
            continue
        decision_status = str(decision.get("status") or "")
        decision_status_counts[decision_status] += 1
        include = decision_status != "pass" or bool(row.get("seed_bug_refs"))
        if not include:
            continue
        resolution = str(decision.get("resolution_status") or _default_resolution_status(decision, row))
        resolution_counts[resolution] += 1
        bug_records.append(_bug_record(row=row, decision=decision, resolution_status=resolution))
    seed_bug_records = [record for record in bug_records if record.get("seed_bug_refs")]
    return {
        "schema_name": TOPIC_PACKET_VISUAL_AUDIT_REGISTRY_SCHEMA,
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "complete": complete,
        "allow_incomplete": allow_incomplete,
        "source_files": {
            "batch": str(batch_path),
            "decisions": str(decisions_path),
            "out": str(out_path),
        },
        "batch_id": batch.get("batch_id"),
        "decision_version": TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION,
        "prompt_version": TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION,
        "summary": {
            "batch_page_count": len(rows),
            "accepted_decision_count": len(accepted_by_row),
            "bug_record_count": len(bug_records),
            "seed_bug_record_count": len(seed_bug_records),
            "decision_status_counts": dict(sorted(decision_status_counts.items())),
            "resolution_status_counts": dict(sorted(resolution_counts.items())),
            "category_counts": dict(sorted(Counter(category for record in bug_records for category in record.get("categories") or []).items())),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "seed_bug_status": _seed_bug_status(batch.get("seed_bugs") or [], seed_bug_records, complete=complete),
        "bug_records": bug_records,
        "errors": list(errors),
        "warnings": list(warnings),
        "rejected": list(rejected),
    }


def _bug_record(*, row: dict[str, Any], decision: dict[str, Any], resolution_status: str) -> dict[str, Any]:
    row_id = str(row.get("row_id") or "")
    categories = _string_list(decision.get("categories"))
    return {
        "bug_id": f"tpva_{row_id}",
        "row_id": row_id,
        "packet_id": row.get("packet_id"),
        "paper_family": row.get("paper_family"),
        "topic_id": row.get("topic_id"),
        "page_number": row.get("page_number"),
        "page_section": row.get("page_section"),
        "page_image_path": row.get("page_image_path"),
        "pdf_path": row.get("pdf_path"),
        "manifest_path": row.get("manifest_path"),
        "related_problem_numbers": row.get("related_problem_numbers") or [],
        "related_question_ids": row.get("related_question_ids") or [],
        "seed_bug_refs": row.get("seed_bug_refs") or [],
        "decision_status": decision.get("status"),
        "resolution_status": resolution_status,
        "categories": categories,
        "likely_root_cause": str(decision.get("likely_root_cause") or ""),
        "fix_owner_area": str(decision.get("fix_owner_area") or "unknown"),
        "generalization_decision": str(decision.get("generalization_decision") or "needs_more_evidence"),
        "affected_artifacts": _affected_artifacts(row),
        "before_evidence": _string_list(decision.get("before_evidence")) or [str(row.get("page_image_path") or "")],
        "after_evidence": _string_list(decision.get("after_evidence")),
        "seed_resolution_status_by_question_id": _seed_resolution_status_overrides(decision),
        "verification_result": decision.get("verification_result") if isinstance(decision.get("verification_result"), dict) else {},
        "rationale": str(decision.get("rationale") or ""),
        "source": str(decision.get("source") or ""),
    }


def _seed_resolution_status_overrides(decision: dict[str, Any]) -> dict[str, str]:
    raw = decision.get("seed_resolution_status_by_question_id")
    if not isinstance(raw, dict):
        return {}
    return {
        str(question_id): str(status)
        for question_id, status in raw.items()
        if str(question_id).strip() and str(status) in RESOLUTION_STATUSES
    }


def _seed_bug_status(
    seed_bugs: Sequence[dict[str, Any]],
    seed_bug_records: Sequence[dict[str, Any]],
    *,
    complete: bool,
) -> list[dict[str, Any]]:
    by_question: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in seed_bug_records:
        for seed in record.get("seed_bug_refs") or []:
            question_id = str(seed.get("question_id") or "")
            resolution_overrides = record.get("seed_resolution_status_by_question_id")
            if isinstance(resolution_overrides, dict) and question_id in resolution_overrides:
                record = {**record, "resolution_status": str(resolution_overrides[question_id])}
            by_question.setdefault((str(seed.get("type") or ""), question_id), []).append(record)
    statuses = []
    for seed in seed_bugs:
        key = (str(seed.get("type") or ""), str(seed.get("question_id") or ""))
        records = by_question.get(key, [])
        resolution = "unreviewed"
        rationale = ""
        if records:
            resolutions = {str(record.get("resolution_status") or "") for record in records}
            if "open" in resolutions:
                resolution = "open"
            elif "waived_with_reason" in resolutions:
                resolution = "waived_with_reason"
            elif "fixed" in resolutions:
                resolution = "fixed"
            else:
                resolution = sorted(resolutions)[0]
        elif complete:
            resolution = "open"
            rationale = "No rendered page for this seed exists in the completed batch."
        status = {**seed, "resolution_status": resolution, "record_count": len(records)}
        if rationale:
            status["rationale"] = rationale
        statuses.append(status)
    return statuses


def _affected_artifacts(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pdf_path": row.get("pdf_path"),
        "manifest_path": row.get("manifest_path"),
        "page_image_path": row.get("page_image_path"),
        "question_image_paths": row.get("source_question_image_paths") or [],
        "mark_scheme_image_paths": row.get("source_mark_scheme_image_paths") or [],
    }


def _default_resolution_status(decision: dict[str, Any], row: dict[str, Any] | None) -> str:
    status = str(decision.get("status") or "")
    if status == "pass" and row and row.get("seed_bug_refs"):
        return "fixed"
    if status == "pass":
        return "not_applicable"
    return "open"


def _find_codex_bin(*, codex_bin: str | Path | None = None, required: bool = True) -> str | None:
    if codex_bin:
        path = Path(codex_bin)
        if path.is_file():
            return str(path)
        if required:
            raise TopicPacketVisualAuditError(f"Codex CLI not found at --codex-bin path: {codex_bin}")
        return None
    found = shutil.which("codex")
    if found:
        return found
    mac_app_path = Path("/Applications/Codex.app/Contents/Resources/codex")
    if mac_app_path.is_file():
        return str(mac_app_path)
    if required:
        raise TopicPacketVisualAuditError("Codex CLI not found. Install Codex or pass --codex-bin.")
    return None


def _run_codex_page_reviews(
    *,
    codex_bin: Path,
    batch: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    model: str,
    sandbox: str,
) -> None:
    if sandbox not in {"read-only", "workspace-write"}:
        raise TopicPacketVisualAuditError("--codex-sandbox must be read-only or workspace-write.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path = out_path.parent / "topic_packet_visual_audit_decision_schema.json"
    write_atomic_json(topic_packet_visual_audit_decision_schema(), schema_path, sort_keys=True)
    with tempfile.TemporaryDirectory(prefix="topic-packet-visual-audit-codex-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        with out_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                decision = _request_codex_page_review(
                    codex_bin=codex_bin,
                    schema_path=schema_path,
                    response_path=tmp_dir / f"{_slug(str(row.get('row_id') or 'row'))}.json",
                    model=model,
                    sandbox=sandbox,
                    row=row,
                    batch=batch,
                )
                handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()


def _request_codex_page_review(
    *,
    codex_bin: Path,
    schema_path: Path,
    response_path: Path,
    model: str,
    sandbox: str,
    row: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, Any]:
    prompt = (
        topic_packet_visual_audit_prompt()
        + "\n\nReturn exactly one JSON object matching the provided schema. Do not edit files."
        + "\n\nDecision schema:\n"
        + json.dumps(topic_packet_visual_audit_decision_schema(), indent=2, sort_keys=True)
        + "\n\nBatch context:\n"
        + json.dumps(_codex_batch_context(batch), indent=2, sort_keys=True)
        + "\n\nAudit row:\n"
        + json.dumps(row, indent=2, sort_keys=True)
    )
    command = [
        str(codex_bin),
        "exec",
        "--cd",
        str(Path.cwd()),
        "--sandbox",
        sandbox,
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
        "--color",
        "never",
    ]
    if model and model not in {"codex", "default"}:
        command.extend(["--model", model])
    page_image_path = Path(str(row.get("page_image_path") or ""))
    if page_image_path.is_file():
        command.extend(["--image", str(page_image_path.resolve())])
    command.append("-")
    try:
        completed = subprocess.run(command, input=prompt, text=True, capture_output=True, check=False)
    except Exception as exc:
        return _not_reviewable_error_decision(row, model=f"codex:{model}", error=exc)
    if completed.returncode != 0:
        return _not_reviewable_error_decision(
            row,
            model=f"codex:{model}",
            error=RuntimeError(_codex_failure_message(completed)),
        )
    try:
        parsed = _parse_json_object(response_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _not_reviewable_error_decision(row, model=f"codex:{model}", error=exc)
    parsed.setdefault("decision_version", TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION)
    parsed.setdefault("prompt_version", TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION)
    parsed.setdefault("row_id", row.get("row_id"))
    parsed.setdefault("packet_id", row.get("packet_id"))
    parsed.setdefault("page_number", row.get("page_number"))
    parsed.setdefault("source", f"codex_cli_topic_packet_visual_audit:{model}")
    return parsed


def _codex_batch_context(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": batch.get("batch_id"),
        "review_policy": batch.get("review_policy"),
        "decision_version": TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION,
        "prompt_version": TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION,
        "decision_statuses": sorted(DECISION_STATUSES),
        "bug_categories": sorted(BUG_CATEGORIES),
        "fix_owner_areas": sorted(FIX_OWNER_AREAS),
        "generalization_decisions": sorted(GENERALIZATION_DECISIONS),
        "resolution_statuses": sorted(RESOLUTION_STATUSES),
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise TopicPacketVisualAuditError("Codex output was not a JSON object.")
    return parsed


def _codex_failure_message(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    message = f"Codex CLI exited {completed.returncode}."
    if stderr:
        message += f" stderr: {stderr[-1200:]}"
    if stdout:
        message += f" stdout: {stdout[-1200:]}"
    return message


def _request_openai_page_review(*, client: Any, model: str, row: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                topic_packet_visual_audit_prompt()
                + "\n\nDecision schema:\n"
                + json.dumps(topic_packet_visual_audit_decision_schema(), indent=2, sort_keys=True)
                + "\n\nAudit row:\n"
                + json.dumps(row, indent=2, sort_keys=True)
            ),
        }
    ]
    path = str(row.get("page_image_path") or "")
    if path:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path))}})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    parsed.setdefault("decision_version", TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION)
    parsed.setdefault("prompt_version", TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION)
    parsed.setdefault("row_id", row.get("row_id"))
    parsed.setdefault("packet_id", row.get("packet_id"))
    parsed.setdefault("page_number", row.get("page_number"))
    parsed.setdefault("source", f"ai_assisted_topic_packet_visual_audit:{model}")
    return parsed


def topic_packet_visual_audit_prompt() -> str:
    return (
        "You are visually auditing rendered CAIE 9709 topic-packet PDF pages. "
        "Use the page image as the primary evidence. Choose pass only when the page is readable, diagrams are visible, "
        "headers do not overlap content, and mark schemes are appropriate for their related questions. "
        "Choose bug for clear defects, needs_human for ambiguity, and not_reviewable when the page image or metadata is unusable. "
        "Use only the provided category IDs. Prefer generalizable root causes only when the metadata supports a repeated pattern."
    )


def _not_reviewable_error_decision(row: dict[str, Any], *, model: str, error: Exception) -> dict[str, Any]:
    return {
        "decision_version": TOPIC_PACKET_VISUAL_AUDIT_DECISION_VERSION,
        "prompt_version": TOPIC_PACKET_VISUAL_AUDIT_PROMPT_VERSION,
        "row_id": row.get("row_id"),
        "packet_id": row.get("packet_id"),
        "page_number": row.get("page_number"),
        "status": "not_reviewable",
        "categories": ["packet_layout"],
        "likely_root_cause": "provider_call_failed",
        "fix_owner_area": "packet_visual_audit",
        "generalization_decision": "needs_more_evidence",
        "resolution_status": "open",
        "rationale": f"Provider call failed: {type(error).__name__}: {error}",
        "source": f"ai_assisted_topic_packet_visual_audit:{model}",
    }


def _render_pdf_page(pdf_path: Path, *, page_number: int, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        if page_number < 1 or page_number > doc.page_count:
            raise TopicPacketVisualAuditError(f"Page {page_number} outside PDF page count {doc.page_count}: {pdf_path}")
        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pixmap.save(output_path)


def _pdf_page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def _page_section(manifest: dict[str, Any], page_number: int) -> str:
    sections = manifest.get("page_sections")
    if isinstance(sections, list) and 1 <= page_number <= len(sections):
        return str(sections[page_number - 1] or "")
    question_range = manifest.get("questions_section_page_range")
    answer_range = manifest.get("answers_section_page_range")
    if _in_range(page_number, question_range):
        return "Questions"
    if _in_range(page_number, answer_range):
        return "Answers / Mark Schemes"
    return ""


def _in_range(page_number: int, value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and int(value[0]) <= page_number <= int(value[1])


def _resolve_pdf_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    pdf_outputs = manifest.get("pdf_outputs") if isinstance(manifest.get("pdf_outputs"), dict) else {}
    topic_packet = pdf_outputs.get("topic_packet") if isinstance(pdf_outputs.get("topic_packet"), dict) else {}
    raw_path = str(topic_packet.get("path") or manifest.get("pdf_path") or "").strip()
    if not raw_path:
        return manifest_path.parent / "topic_packet.pdf"
    pdf_path = Path(raw_path)
    if pdf_path.is_absolute() or pdf_path.is_file():
        return pdf_path
    sibling = manifest_path.parent / pdf_path.name
    if sibling.is_file():
        return sibling
    return pdf_path


def _packet_id(manifest: dict[str, Any], manifest_path: Path) -> str:
    family = str(manifest.get("paper_family") or manifest_path.parent.parent.name or "unknown")
    topic = str(manifest.get("topic_id") or manifest.get("topic") or manifest_path.parent.name or "unknown")
    subtopic = str(manifest.get("subtopic_id") or "")
    parts = [family, topic]
    if subtopic:
        parts.append(subtopic)
    return "_".join(_slug(part) for part in parts if part)


def _batch_id(rows: Sequence[dict[str, Any]]) -> str:
    digest = json.dumps(
        [(row.get("row_id"), row.get("page_image_path"), row.get("seed_bug_refs")) for row in rows],
        sort_keys=True,
        ensure_ascii=False,
    )
    import hashlib

    return f"topic_packet_visual_audit_{hashlib.sha256(digest.encode('utf-8')).hexdigest()[:12]}"


def _seed_index(seed_bugs: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    for seed in seed_bugs:
        question_id = str(seed.get("question_id") or "")
        if question_id:
            index.setdefault(question_id, []).append(dict(seed))
    return index


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_decisions(path: Path) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    if not path.exists():
        return decisions
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            decisions.append(json.loads(line))
    return decisions


def _existing_decision_row_ids(path: Path) -> set[str]:
    return {str(decision.get("row_id") or "") for decision in _read_decisions(path) if isinstance(decision, dict)}


def _decision_report_row(decision: dict[str, Any], errors: Sequence[str]) -> dict[str, Any]:
    return {
        "row_id": str(decision.get("row_id") or ""),
        "status": str(decision.get("status") or ""),
        "errors": list(errors),
        "rationale": str(decision.get("rationale") or ""),
    }


def _image_data_url(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def _dedupe(values: Iterable[Any]) -> list[Any]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _dedupe_dicts(values: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    output = []
    for value in values:
        key = tuple(sorted(value.items()))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(value))
    return output


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.strip().lower()).strip("_") or "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
