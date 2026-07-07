from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Sequence

from .atomic_json import write_atomic_json


TOPIC_DIFFICULTY_BATCH_SCHEMA = "exam_bank.topic_packet_difficulty_review.batch"
TOPIC_DIFFICULTY_BATCH_SCHEMA_VERSION = 1
TOPIC_DIFFICULTY_DECISION_VERSION = "topic_packet_difficulty_decision_v1"
TOPIC_DIFFICULTY_SIDECAR_SCHEMA = "exam_bank.topic_packet_difficulty_review"
TOPIC_DIFFICULTY_SIDECAR_SCHEMA_VERSION = 1
TOPIC_DIFFICULTY_PROMPT_VERSION = "topic_packet_difficulty_9709_v1"
DEFAULT_OUTPUT_ROOT = Path("data/review/topic_difficulty")
DEFAULT_REPORTS_DIR = Path("reports/topic_difficulty")
IMAGE_EVIDENCE_TYPES = {"canonical_question_image", "canonical_mark_scheme_image"}
CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}
VALID_STATUSES = {"accepted", "pending"}
VALID_CONFIDENCE = set(CONFIDENCE_ORDER)


class TopicDifficultyReviewError(RuntimeError):
    pass


def add_topic_difficulty_review_cli_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="topic_difficulty_review_command", required=True)

    build = subparsers.add_parser("build", help="Build an image-backed topic-packet difficulty review batch.")
    build.add_argument("--manifest", type=Path, required=True, help="Topic packet manifest.json.")
    build.add_argument("--artifact-root", type=Path, default=Path("output"))
    build.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    build.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run", help="Run append/resumable visual difficulty reviews for a batch.")
    run.add_argument("--batch", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--max-records", type=int, default=None)
    run.add_argument("--model", default="gpt-5-mini")
    run.add_argument("--provider", default="openai")
    run.add_argument("--dry-run", action="store_true")

    import_decisions = subparsers.add_parser(
        "import",
        help="Validate topic-packet difficulty decisions and write ranked sidecar/reports.",
    )
    import_decisions.add_argument("--batch", type=Path, required=True)
    import_decisions.add_argument("--decisions", type=Path, required=True)
    import_decisions.add_argument("--out", type=Path, default=None)
    import_decisions.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    import_decisions.add_argument("--artifact-root", type=Path, default=Path("output"))
    import_decisions.add_argument("--allow-incomplete", action="store_true")
    import_decisions.add_argument("--dry-run", action="store_true")


def run_topic_difficulty_review_from_args(args: argparse.Namespace) -> dict[str, Any]:
    command = args.topic_difficulty_review_command
    if command == "build":
        return build_topic_difficulty_batch(
            manifest_path=args.manifest,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            dry_run=bool(args.dry_run),
        )
    if command == "run":
        return run_topic_difficulty_reviews(
            batch_path=args.batch,
            out_path=args.out,
            max_records=args.max_records,
            model=args.model,
            provider=args.provider,
            dry_run=bool(args.dry_run),
        )
    if command == "import":
        return import_topic_difficulty_decisions(
            batch_path=args.batch,
            decisions_path=args.decisions,
            out_path=args.out,
            reports_dir=args.reports_dir,
            artifact_root=args.artifact_root,
            allow_incomplete=bool(args.allow_incomplete),
            dry_run=bool(args.dry_run),
        )
    raise TopicDifficultyReviewError(f"Unhandled topic difficulty review command: {command}")


def build_topic_difficulty_batch(
    *,
    manifest_path: str | Path,
    artifact_root: str | Path = Path("output"),
    out_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    artifact_root = Path(artifact_root)
    out_dir = Path(out_dir)
    manifest = _read_json(manifest_path)
    rows = [
        _batch_row(record, index=index, manifest=manifest, artifact_root=artifact_root, manifest_dir=manifest_path.parent)
        for index, record in enumerate(_included_records(manifest), start=1)
    ]
    packet_id = _packet_id(manifest, manifest_path)
    batch_id = _batch_id(packet_id, rows)
    for index, row in enumerate(rows, start=1):
        row["packet_id"] = packet_id
        row["batch_id"] = batch_id
        row["batch_index"] = index

    batch = {
        "schema_name": TOPIC_DIFFICULTY_BATCH_SCHEMA,
        "schema_version": TOPIC_DIFFICULTY_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "packet_id": packet_id,
        "created_at": _utc_now_iso(),
        "dry_run": dry_run,
        "source_files": {
            "manifest": str(manifest_path),
            "artifact_root": str(artifact_root),
        },
        "packet": _packet_metadata(manifest, manifest_path),
        "review_policy": (
            "Rank this packet only. Use canonical question and mark-scheme images as authoritative evidence. "
            "Rank 1 is hardest; higher difficulty percentile means harder."
        ),
        "prompt_version": TOPIC_DIFFICULTY_PROMPT_VERSION,
        "decision_version": TOPIC_DIFFICULTY_DECISION_VERSION,
        "reviewer_prompt": topic_difficulty_prompt(),
        "decision_schema": topic_difficulty_decision_schema(),
        "selection": {
            "question_count": len(rows),
            "missing_question_image_count": sum(1 for row in rows if not row["canonical_question_image_path"]),
            "missing_mark_scheme_image_count": sum(1 for row in rows if not row["canonical_mark_scheme_image_path"]),
            "section_counts": dict(Counter(str(row.get("section") or "") for row in rows)),
        },
        "rows": rows,
    }
    if not dry_run:
        packet_dir = out_dir / packet_id
        packet_dir.mkdir(parents=True, exist_ok=True)
        write_atomic_json(batch, packet_dir / "topic_packet_difficulty_batch.json", sort_keys=True)
        (packet_dir / "topic_packet_difficulty_batch.md").write_text(render_batch_markdown(batch), encoding="utf-8")
    return batch


def run_topic_difficulty_reviews(
    *,
    batch_path: str | Path,
    out_path: str | Path,
    max_records: int | None = None,
    model: str = "gpt-5-mini",
    provider: str = "openai",
    dry_run: bool = False,
) -> dict[str, Any]:
    batch_path = Path(batch_path)
    out_path = Path(out_path)
    if max_records is not None and max_records < 0:
        raise TopicDifficultyReviewError("--max-records must be zero or greater.")
    batch = _read_json(batch_path)
    rows = [row for row in batch.get("rows") or [] if isinstance(row, dict)]
    done = _existing_decision_question_ids(out_path)
    pending = [row for row in rows if str(row.get("question_id") or "") not in done]
    if max_records is not None:
        pending = pending[:max_records]
    manifest = {
        "schema_name": "exam_bank.topic_packet_difficulty_review.runner_manifest",
        "provider": provider,
        "model": model,
        "prompt_version": TOPIC_DIFFICULTY_PROMPT_VERSION,
        "dry_run": dry_run,
        "batch_path": str(batch_path),
        "out_path": str(out_path),
        "pending_count": len(pending),
        "resumed_count": len(done),
        "created_at": _utc_now_iso(),
    }
    if dry_run:
        return manifest
    if provider != "openai":
        raise TopicDifficultyReviewError("topic difficulty review runner supports provider=openai only.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise TopicDifficultyReviewError("topic difficulty review runner requires OPENAI_API_KEY.")

    from openai import OpenAI

    client = OpenAI()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in pending:
            try:
                decision = _request_openai_topic_difficulty_review(client=client, model=model, row=row)
            except Exception as exc:
                decision = _pending_error_decision(row, model=model, error=exc)
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return manifest


def import_topic_difficulty_decisions(
    *,
    batch_path: str | Path,
    decisions_path: str | Path,
    out_path: str | Path | None = None,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    artifact_root: str | Path = Path("output"),
    allow_incomplete: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    batch_path = Path(batch_path)
    decisions_path = Path(decisions_path)
    artifact_root = Path(artifact_root)
    reports_dir = Path(reports_dir)
    batch = _read_json(batch_path)
    rows = [row for row in batch.get("rows") or [] if isinstance(row, dict)]
    rows_by_id = {str(row.get("question_id") or ""): row for row in rows if str(row.get("question_id") or "")}
    decisions = _read_decisions(decisions_path)

    errors: list[str] = []
    warnings: list[str] = []
    pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for index, decision in enumerate(decisions, start=1):
        if not isinstance(decision, dict):
            errors.append(f"decision:{index}:not_object")
            continue
        question_id = str(decision.get("question_id") or "").strip()
        if not question_id:
            errors.append(f"decision:{index}:missing_question_id")
            continue
        if question_id in seen_ids:
            errors.append(f"decision:{index}:{question_id}:duplicate_decision")
            continue
        seen_ids.add(question_id)
        validation_errors = validate_topic_difficulty_decision(
            decision,
            batch_rows=rows_by_id,
            artifact_root=artifact_root,
        )
        if _decision_status(decision) == "pending":
            pending.append(_decision_report_row(decision, validation_errors))
            validation_errors = validation_errors or ["pending_decision"]
        if validation_errors:
            rejected.append(_decision_report_row(decision, validation_errors))
            errors.extend(f"decision:{index}:{question_id}:{error}" for error in validation_errors)
            continue
        accepted_by_id[question_id] = decision

    missing_ids = [question_id for question_id in rows_by_id if question_id not in accepted_by_id]
    if missing_ids:
        errors.extend(f"missing_decision:{question_id}" for question_id in missing_ids)
    complete = not errors and len(accepted_by_id) == len(rows_by_id)
    ok = complete or allow_incomplete
    if errors and allow_incomplete:
        warnings.extend(errors)

    sidecar = _sidecar_from_decisions(
        batch=batch,
        batch_path=batch_path,
        decisions_path=decisions_path,
        accepted_by_id=accepted_by_id,
        rows=rows,
        complete=complete,
        allow_incomplete=allow_incomplete,
        errors=errors,
        warnings=warnings,
        rejected=rejected,
        pending=pending,
    )
    output_path = Path(out_path) if out_path else _default_sidecar_path(batch_path, batch)
    report = {
        "schema_name": "exam_bank.topic_packet_difficulty_review.import_report",
        "schema_version": 1,
        "ok": ok,
        "complete": complete,
        "draft": not complete,
        "dry_run": dry_run,
        "allow_incomplete": allow_incomplete,
        "batch_path": str(batch_path),
        "decisions_path": str(decisions_path),
        "out_path": str(output_path),
        "reports_dir": str(reports_dir),
        "batch_question_count": len(rows_by_id),
        "decision_count": len(decisions),
        "accepted_count": len(accepted_by_id),
        "pending_count": len(pending),
        "rejected_count": len(rejected),
        "missing_count": len(missing_ids),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "pending": pending,
        "rejected": rejected,
    }
    if not dry_run and ok:
        write_atomic_json(sidecar, output_path, sort_keys=True)
        write_topic_difficulty_reports(sidecar, reports_dir=reports_dir)
    return report


def validate_topic_difficulty_decision(
    decision: dict[str, Any],
    *,
    batch_rows: dict[str, dict[str, Any]],
    artifact_root: Path,
) -> list[str]:
    errors: list[str] = []
    if decision.get("decision_version") != TOPIC_DIFFICULTY_DECISION_VERSION:
        errors.append("invalid_decision_version")
    if str(decision.get("prompt_version") or "").strip() not in {"", TOPIC_DIFFICULTY_PROMPT_VERSION}:
        errors.append("invalid_prompt_version")
    question_id = str(decision.get("question_id") or "").strip()
    row = batch_rows.get(question_id)
    if not question_id:
        errors.append("missing_question_id")
    elif row is None:
        errors.append("unknown_question_id")
    status = _decision_status(decision)
    if status not in VALID_STATUSES:
        errors.append("invalid_status")
    if status == "pending":
        return errors
    score = _float_or_none(decision.get("visual_difficulty_score_0_100"))
    if score is None or score < 0 or score > 100:
        errors.append("invalid_visual_difficulty_score_0_100")
    confidence = str(decision.get("confidence") or "").strip().lower()
    if confidence not in VALID_CONFIDENCE:
        errors.append("invalid_confidence")
    if not str(decision.get("rationale") or "").strip():
        errors.append("missing_rationale")
    if not str(decision.get("source") or "").strip():
        errors.append("missing_source")
    errors.extend(_validate_image_evidence_refs(decision.get("evidence_refs"), artifact_root=artifact_root))
    if row is not None and not row.get("image_evidence_available"):
        errors.append("batch_row_missing_image_evidence")
    return errors


def topic_difficulty_decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "decision_version",
            "question_id",
            "status",
            "visual_difficulty_score_0_100",
            "confidence",
            "rationale",
            "difficulty_factors",
            "risk_flags",
            "evidence_refs",
            "source",
            "prompt_version",
        ],
        "properties": {
            "decision_version": {"const": TOPIC_DIFFICULTY_DECISION_VERSION},
            "question_id": {"type": "string"},
            "status": {"enum": sorted(VALID_STATUSES)},
            "visual_difficulty_score_0_100": {"type": "number", "minimum": 0, "maximum": 100},
            "confidence": {"enum": sorted(VALID_CONFIDENCE)},
            "rationale": {"type": "string"},
            "difficulty_factors": {"type": "array", "items": {"type": "string"}},
            "risk_flags": {"type": "array", "items": {"type": "string"}},
            "evidence_refs": {"type": "array"},
            "source": {"type": "string"},
            "reviewer_model": {"type": "string"},
            "prompt_version": {"const": TOPIC_DIFFICULTY_PROMPT_VERSION},
        },
        "additionalProperties": True,
    }


def topic_difficulty_prompt() -> str:
    return (
        "You are ranking one CAIE 9709 topic packet by difficulty using canonical images. "
        "Inspect both the question image and mark-scheme image. Score difficulty within this topic packet only, "
        "not globally across the syllabus. Higher score means harder. Consider conceptual demand, algebraic load, "
        "method dependencies, number of steps, abstraction, unusual wording, and mark-scheme complexity. "
        "Return JSON matching the schema. Use status=pending if image evidence is unclear or incomplete."
    )


def render_batch_markdown(batch: dict[str, Any]) -> str:
    packet = batch.get("packet") if isinstance(batch.get("packet"), dict) else {}
    lines = [
        "# Topic Packet Difficulty Review Batch",
        "",
        f"- Packet: `{batch.get('packet_id')}`",
        f"- Topic: `{packet.get('paper_family')}/{packet.get('topic_id')}`",
        f"- Questions: `{len(batch.get('rows') or [])}`",
        "",
        "## Rows",
        "",
    ]
    for row in batch.get("rows") or []:
        lines.append(
            f"- `{row.get('question_id')}` problem `{row.get('problem_number')}` section `{row.get('section')}` "
            f"marks `{row.get('marks')}` images `{bool(row.get('image_evidence_available'))}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_topic_difficulty_reports(sidecar: dict[str, Any], *, reports_dir: str | Path = DEFAULT_REPORTS_DIR) -> dict[str, Any]:
    reports_dir = Path(reports_dir)
    packet_id = str(sidecar.get("packet_id") or "unknown_packet")
    packet_dir = reports_dir / packet_id
    outputs = {
        "summary": packet_dir / "summary.md",
        "ranking": packet_dir / "ranking.md",
        "review_queue": packet_dir / "review_queue.md",
    }
    rendered = {
        "summary": render_summary_report(sidecar),
        "ranking": render_ranking_report(sidecar),
        "review_queue": render_review_queue_report(sidecar),
    }
    packet_dir.mkdir(parents=True, exist_ok=True)
    for key, path in outputs.items():
        path.write_text(rendered[key], encoding="utf-8")
    return {"outputs": {key: str(path) for key, path in outputs.items()}}


def render_summary_report(sidecar: dict[str, Any]) -> str:
    records = [record for record in sidecar.get("records") or [] if isinstance(record, dict)]
    confidence_counts = Counter(str(record.get("confidence") or "") for record in records)
    section_counts = Counter(str(record.get("packet_section") or "") for record in records)
    risk_counts = Counter(flag for record in records for flag in record.get("risk_flags") or [])
    lines = [
        "# Topic Packet Difficulty Review Summary",
        "",
        f"- Packet: `{sidecar.get('packet_id')}`",
        f"- Generated: `{sidecar.get('generated_at')}`",
        f"- Complete: `{sidecar.get('complete')}`",
        f"- Draft: `{sidecar.get('draft')}`",
        f"- Safe for teacher filtering: `{sidecar.get('safe_for_teacher_filtering')}`",
        f"- Safe for student sequencing: `{sidecar.get('safe_for_student_sequencing')}`",
        f"- Ranked records: `{len(records)}`",
        "",
        "## Confidence",
        "",
        *_counter_lines(confidence_counts),
        "",
        "## Packet Sections",
        "",
        *_counter_lines(section_counts),
        "",
        "## Risk Flags",
        "",
        *_counter_lines(risk_counts),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_ranking_report(sidecar: dict[str, Any]) -> str:
    records = sorted(
        [record for record in sidecar.get("records") or [] if isinstance(record, dict)],
        key=lambda record: int(record.get("packet_rank") or 10**9),
    )
    lines = [
        "# Topic Packet Difficulty Ranking",
        "",
        "Rank 1 is hardest. Higher percentile means harder.",
        "",
        "| Rank | Percentile | Score | Confidence | Question | Problem | Marks | Section | Rationale |",
        "|---:|---:|---:|---|---|---:|---:|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.get('packet_rank')} | {record.get('difficulty_percentile_0_100')} | "
            f"{record.get('visual_difficulty_score_0_100')} | {record.get('confidence')} | "
            f"`{record.get('question_id')}` | {record.get('problem_number')} | {record.get('marks')} | "
            f"{record.get('packet_section')} | {_markdown_cell(record.get('rationale'))} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_review_queue_report(sidecar: dict[str, Any]) -> str:
    import_report = sidecar.get("import_report") if isinstance(sidecar.get("import_report"), dict) else {}
    rejected = import_report.get("rejected") or []
    pending = import_report.get("pending") or []
    warnings = import_report.get("warnings") or []
    lines = [
        "# Topic Packet Difficulty Review Queue",
        "",
        f"- Complete: `{sidecar.get('complete')}`",
        f"- Draft: `{sidecar.get('draft')}`",
        f"- Rejected decisions: `{len(rejected)}`",
        f"- Pending decisions: `{len(pending)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "| Question | Errors |",
        "|---|---|",
    ]
    for item in [*rejected, *pending]:
        lines.append(f"| `{item.get('question_id')}` | {_markdown_cell(', '.join(item.get('errors') or []))} |")
    if not rejected and not pending:
        lines.append("| - | - |")
    return "\n".join(lines).rstrip() + "\n"


def _sidecar_from_decisions(
    *,
    batch: dict[str, Any],
    batch_path: Path,
    decisions_path: Path,
    accepted_by_id: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    complete: bool,
    allow_incomplete: bool,
    errors: list[str],
    warnings: list[str],
    rejected: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> dict[str, Any]:
    row_by_id = {str(row.get("question_id") or ""): row for row in rows}
    accepted_rows = [
        _accepted_record(decision, row_by_id[str(question_id)])
        for question_id, decision in accepted_by_id.items()
        if str(question_id) in row_by_id
    ]
    ranked = _rank_records(accepted_rows)
    return {
        "schema_name": TOPIC_DIFFICULTY_SIDECAR_SCHEMA,
        "schema_version": TOPIC_DIFFICULTY_SIDECAR_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "packet_id": str(batch.get("packet_id") or ""),
        "batch_id": str(batch.get("batch_id") or ""),
        "complete": complete,
        "draft": not complete,
        "allow_incomplete": allow_incomplete,
        "safe_for_teacher_filtering": complete,
        "safe_for_student_sequencing": False,
        "interpretation": {
            "packet_rank": "Rank 1 is hardest within this topic packet; larger ranks are easier.",
            "difficulty_percentile_0_100": "Higher means harder within this topic packet.",
            "student_use": "v1 does not enable student-facing sequencing.",
        },
        "source_files": {
            "batch": str(batch_path),
            "decisions": str(decisions_path),
            "manifest": str((batch.get("source_files") or {}).get("manifest") or ""),
        },
        "packet": batch.get("packet") or {},
        "record_count": len(ranked),
        "expected_record_count": len(rows),
        "records": ranked,
        "import_report": {
            "errors": errors,
            "warnings": warnings,
            "rejected": rejected,
            "pending": pending,
        },
    }


def _accepted_record(decision: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": str(row.get("question_id") or decision.get("question_id") or ""),
        "packet_rank": None,
        "difficulty_percentile_0_100": None,
        "visual_difficulty_score_0_100": round(float(decision.get("visual_difficulty_score_0_100") or 0), 3),
        "confidence": str(decision.get("confidence") or "").strip().lower(),
        "rationale": str(decision.get("rationale") or ""),
        "difficulty_factors": _strings(decision.get("difficulty_factors")),
        "risk_flags": _strings(decision.get("risk_flags")),
        "evidence_refs": decision.get("evidence_refs") or [],
        "source": str(decision.get("source") or ""),
        "reviewer_model": str(decision.get("reviewer_model") or ""),
        "prompt_version": str(decision.get("prompt_version") or TOPIC_DIFFICULTY_PROMPT_VERSION),
        "problem_number": _int_or_none(row.get("problem_number")),
        "marks": _int_or_none(row.get("marks")),
        "packet_section": str(row.get("section") or ""),
        "paper": str(row.get("paper") or ""),
        "question_number": str(row.get("question_number") or ""),
        "source_label": str(row.get("source_label") or ""),
        "source_packet_metadata": {
            "batch_id": row.get("batch_id"),
            "batch_index": row.get("batch_index"),
            "packet_id": row.get("packet_id"),
            "primary_topic_id": row.get("primary_topic_id"),
            "coverage_topic_ids": row.get("coverage_topic_ids") or [],
        },
    }


def _rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda record: (
            -float(record.get("visual_difficulty_score_0_100") or 0),
            -CONFIDENCE_ORDER.get(str(record.get("confidence") or ""), 0),
            -(_int_or_none(record.get("marks")) or 0),
            -(_int_or_none(record.get("problem_number")) or 0),
            str(record.get("question_id") or ""),
        ),
    )
    n = len(ranked)
    for index, record in enumerate(ranked, start=1):
        record["packet_rank"] = index
        record["difficulty_percentile_0_100"] = 100.0 if n == 1 else round(100 * (n - index) / (n - 1), 2)
    return ranked


def _request_openai_topic_difficulty_review(*, client: Any, model: str, row: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                topic_difficulty_prompt()
                + "\n\nDecision schema:\n"
                + json.dumps(topic_difficulty_decision_schema(), indent=2, sort_keys=True)
                + "\n\nBatch row:\n"
                + json.dumps(row, indent=2, sort_keys=True)
            ),
        }
    ]
    for path_field in ("canonical_question_image_path", "canonical_mark_scheme_image_path"):
        path = str(row.get(path_field) or "")
        if path:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path))}})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    parsed.setdefault("decision_version", TOPIC_DIFFICULTY_DECISION_VERSION)
    parsed.setdefault("prompt_version", TOPIC_DIFFICULTY_PROMPT_VERSION)
    parsed.setdefault("source", f"ai_assisted_topic_difficulty_review:{model}")
    parsed.setdefault("reviewer_model", model)
    return parsed


def _pending_error_decision(row: dict[str, Any], *, model: str, error: Exception) -> dict[str, Any]:
    return {
        "decision_version": TOPIC_DIFFICULTY_DECISION_VERSION,
        "question_id": row.get("question_id"),
        "status": "pending",
        "visual_difficulty_score_0_100": 0,
        "confidence": "low",
        "rationale": f"Provider call failed: {type(error).__name__}: {error}",
        "difficulty_factors": [],
        "risk_flags": ["provider_error"],
        "evidence_refs": [],
        "source": f"ai_assisted_topic_difficulty_review:{model}",
        "reviewer_model": model,
        "prompt_version": TOPIC_DIFFICULTY_PROMPT_VERSION,
    }


def _batch_row(
    record: dict[str, Any],
    *,
    index: int,
    manifest: dict[str, Any],
    artifact_root: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    q_path = _first_existing_path(record.get("question_image_paths") or [], artifact_root, manifest_dir)
    ms_path = _first_existing_path(record.get("mark_scheme_image_paths") or [], artifact_root, manifest_dir)
    return {
        "question_id": str(record.get("question_id") or ""),
        "problem_number": _int_or_none(record.get("problem_number")) or index,
        "source_label": str(record.get("source_label") or ""),
        "paper": str(record.get("paper") or ""),
        "question_number": str(record.get("question_number") or ""),
        "marks": _int_or_none(record.get("marks")),
        "section": str(record.get("section") or ""),
        "primary_topic_id": str(record.get("primary_topic_id") or manifest.get("topic_id") or ""),
        "secondary_topic_ids": _strings(record.get("secondary_topic_ids")),
        "coverage_topic_ids": _strings(record.get("coverage_topic_ids")),
        "review_reasons": _strings(record.get("review_reasons")),
        "warnings": _strings(record.get("warnings")),
        "canonical_question_image_path": str(q_path) if q_path else "",
        "canonical_mark_scheme_image_path": str(ms_path) if ms_path else "",
        "image_evidence_available": bool(q_path and ms_path),
        "evidence_refs": _evidence_refs(q_path, ms_path),
        "packet_context": {
            "paper_family": str(manifest.get("paper_family") or ""),
            "topic_id": str(manifest.get("topic_id") or ""),
            "topic_label": str(manifest.get("topic_label") or ""),
            "subtopic_id": str(manifest.get("subtopic_id") or ""),
            "packet_level": str(manifest.get("packet_level") or ""),
            "packet_mode": str(manifest.get("packet_mode") or ""),
            "total_questions": manifest.get("total_questions"),
        },
    }


def _included_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("included_records")
    if not isinstance(records, list):
        raise TopicDifficultyReviewError("Topic packet manifest must contain included_records[].")
    return [record for record in records if isinstance(record, dict)]


def _packet_metadata(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "paper_family": str(manifest.get("paper_family") or ""),
        "topic_id": str(manifest.get("topic_id") or ""),
        "topic_label": str(manifest.get("topic_label") or ""),
        "subtopic_id": str(manifest.get("subtopic_id") or ""),
        "subtopic_label": str(manifest.get("subtopic_label") or ""),
        "packet_level": str(manifest.get("packet_level") or ""),
        "packet_mode": str(manifest.get("packet_mode") or ""),
        "pdf_path": str(manifest.get("pdf_path") or ""),
        "total_questions": manifest.get("total_questions"),
        "approved_count": manifest.get("approved_count"),
        "review_required_count": manifest.get("review_required_count"),
    }


def _packet_id(manifest: dict[str, Any], manifest_path: Path) -> str:
    parts = [
        str(manifest.get("paper_family") or "packet"),
        str(manifest.get("topic_id") or "topic"),
        str(manifest.get("subtopic_id") or ""),
    ]
    digest = hashlib.sha256(str(manifest_path).encode("utf-8")).hexdigest()[:8]
    return "_".join(_slug(part) for part in parts if part) + f"_{digest}"


def _batch_id(packet_id: str, rows: Sequence[dict[str, Any]]) -> str:
    digest = json.dumps([row.get("question_id") for row in rows], sort_keys=True, separators=(",", ":"))
    return f"topic_difficulty_{hashlib.sha256((packet_id + digest).encode('utf-8')).hexdigest()[:12]}"


def _default_sidecar_path(batch_path: Path, batch: dict[str, Any]) -> Path:
    if batch_path.name == "topic_packet_difficulty_batch.json":
        return batch_path.with_name("topic_packet_difficulty_review.v1.json")
    packet_id = str(batch.get("packet_id") or _slug(batch_path.stem))
    return DEFAULT_OUTPUT_ROOT / packet_id / "topic_packet_difficulty_review.v1.json"


def _existing_decision_question_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row.get("question_id") or "").strip()
        for row in _read_decisions(path)
        if isinstance(row, dict) and str(row.get("question_id") or "").strip()
    }


def _validate_image_evidence_refs(value: Any, *, artifact_root: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["missing_evidence_refs"]
    errors: list[str] = []
    found: set[str] = set()
    for ref in value:
        if not isinstance(ref, dict):
            errors.append("evidence_ref_not_object")
            continue
        ref_type = str(ref.get("type") or "")
        if ref_type not in IMAGE_EVIDENCE_TYPES:
            errors.append(f"unsupported_evidence_ref_type:{ref_type}")
            continue
        found.add(ref_type)
        raw_path = str(ref.get("path") or "")
        path = Path(raw_path)
        if not path.is_absolute():
            path = artifact_root / raw_path
        if not path.is_file():
            errors.append(f"evidence_path_not_found:{raw_path}")
    for missing_type in sorted(IMAGE_EVIDENCE_TYPES - found):
        errors.append(f"missing_evidence_ref_type:{missing_type}")
    return errors


def _evidence_refs(q_path: Path | None, ms_path: Path | None) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    if q_path:
        refs.append({"type": "canonical_question_image", "path": str(q_path)})
    if ms_path:
        refs.append({"type": "canonical_mark_scheme_image", "path": str(ms_path)})
    return refs


def _first_existing_path(paths: Any, artifact_root: Path, fallback_root: Path) -> Path | None:
    candidates = paths if isinstance(paths, list) else [paths]
    for raw in candidates:
        if not str(raw or "").strip():
            continue
        path = Path(str(raw))
        search = [path] if path.is_absolute() else [artifact_root / path, fallback_root / path, path]
        for candidate in search:
            if candidate.is_file():
                return candidate
    return None


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TopicDifficultyReviewError(f"Expected JSON object: {path}")
    return payload


def _read_decisions(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    text = candidate.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            rows = payload.get("decisions", payload.get("records"))
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
            return [payload]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _decision_status(decision: dict[str, Any]) -> str:
    return str(decision.get("status") or "").strip().lower()


def _decision_report_row(decision: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "question_id": str(decision.get("question_id") or ""),
        "status": _decision_status(decision),
        "visual_difficulty_score_0_100": decision.get("visual_difficulty_score_0_100"),
        "confidence": decision.get("confidence"),
        "errors": errors,
    }


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _counter_lines(values: Counter[str]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{key}`: {values[key]}" for key in sorted(values, key=str)]


def _markdown_cell(value: Any) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "packet"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
