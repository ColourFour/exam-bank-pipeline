from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .atomic_json import write_atomic_json


SCHEMA_NAME = "exam_bank.partial_question_block_recovery"
SCHEMA_VERSION = 1
RECOVERY_VERSION = "partial_question_block_recovery_v1"
DEFAULT_INPUT = Path("output/json/question_bank.json")
DEFAULT_REPORT_DIR = Path("output/audits/partial_question_block_recovery")
DEFAULT_MIN_CONTIGUOUS_TOKENS = 12
DEFAULT_ADJACENCY_THRESHOLD = 0.72
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_AUDIT_SAMPLE_SIZE = 50
DEFAULT_AUDIT_SEED = 20260629

PARTIAL_REASON = "partial_question_block"
ELIGIBLE_REVIEW_FLAGS = {
    "crop_uncertain",
    "low_confidence_question_crop",
    "likely_truncated_question_crop",
    "crop_split_prompt_regions",
    "figure_region_separated",
    "question_text_figure_overlap_prevented",
    "text_figure_overlap_trimmed",
    "ocr_merged_sparse_lower_region",
    "impossible_question_number_anchor_excluded",
    "text_crop_edge_safety_applied",
}
SPLIT_REGION_FLAGS = {
    "crop_split_prompt_regions",
    "figure_region_separated",
    "question_text_figure_overlap_prevented",
    "text_figure_overlap_trimmed",
}
TRUNCATION_FLAGS = {
    "crop_uncertain",
    "low_confidence_question_crop",
    "likely_truncated_question_crop",
    "text_crop_edge_safety_applied",
}
MISALIGNED_BBOX_FLAGS = {
    "ocr_merged_sparse_lower_region",
    "impossible_question_number_anchor_excluded",
}
UNSAFE_VALIDATION_FLAGS = {
    "question_scope_contaminated",
    "adjacent_question_block_selected",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover eligible mapping_failed:partial_question_block records using deterministic question-stem reconstruction."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Question bank JSON to read.")
    parser.add_argument("--output", type=Path, default=None, help="Question bank JSON to write. Defaults to --input with --write.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--write", action="store_true", help="Write recovered question bank and reports.")
    parser.add_argument("--min-contiguous-tokens", type=int, default=DEFAULT_MIN_CONTIGUOUS_TOKENS)
    parser.add_argument("--adjacency-threshold", type=float, default=DEFAULT_ADJACENCY_THRESHOLD)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--audit-sample-size", type=int, default=DEFAULT_AUDIT_SAMPLE_SIZE)
    parser.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output or args.input
    report = recover_partial_question_blocks_file(
        input_path=args.input,
        output_path=output_path,
        report_dir=args.report_dir,
        write=args.write,
        min_contiguous_tokens=args.min_contiguous_tokens,
        adjacency_threshold=args.adjacency_threshold,
        confidence_threshold=args.confidence_threshold,
        audit_sample_size=args.audit_sample_size,
        audit_seed=args.audit_seed,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def recover_partial_question_blocks_file(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path | None = None,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    write: bool = False,
    min_contiguous_tokens: int = DEFAULT_MIN_CONTIGUOUS_TOKENS,
    adjacency_threshold: float = DEFAULT_ADJACENCY_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    audit_sample_size: int = DEFAULT_AUDIT_SAMPLE_SIZE,
    audit_seed: int = DEFAULT_AUDIT_SEED,
    generated_at: str | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path
    report_dir = Path(report_dir)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    recovered_payload, report = recover_partial_question_blocks_payload(
        payload,
        min_contiguous_tokens=min_contiguous_tokens,
        adjacency_threshold=adjacency_threshold,
        confidence_threshold=confidence_threshold,
        audit_sample_size=audit_sample_size,
        audit_seed=audit_seed,
        generated_at=generated_at,
        input_path=str(input_path),
        output_path=str(output_path),
    )
    if write:
        write_atomic_json(recovered_payload, output_path, indent=2, ensure_ascii=False)
        write_recovery_reports(report, report_dir)
    return report


def recover_partial_question_blocks_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    min_contiguous_tokens: int = DEFAULT_MIN_CONTIGUOUS_TOKENS,
    adjacency_threshold: float = DEFAULT_ADJACENCY_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    audit_sample_size: int = DEFAULT_AUDIT_SAMPLE_SIZE,
    audit_seed: int = DEFAULT_AUDIT_SEED,
    generated_at: str | None = None,
    input_path: str = "",
    output_path: str = "",
) -> tuple[dict[str, Any] | list[dict[str, Any]], dict[str, Any]]:
    generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    output_payload = deepcopy(payload)
    records = _records(output_payload)

    before_counts = _issue_counts(records)
    before_image_counts = _image_risk_counts(records)
    attempts: list[dict[str, Any]] = []
    affected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for record in records:
        decision = _recovery_decision(
            record,
            min_contiguous_tokens=min_contiguous_tokens,
            adjacency_threshold=adjacency_threshold,
            confidence_threshold=confidence_threshold,
        )
        if decision["attempted"]:
            attempts.append(_attempt_row(record, decision))
        if not decision["recoverable"]:
            if _is_partial_failure(record):
                skipped.append(_skip_row(record, decision))
            continue
        _apply_recovery(record, decision, generated_at=generated_at)
        affected.append(_affected_row(record, decision))

    after_counts = _issue_counts(records)
    after_image_counts = _image_risk_counts(records)
    audit_sample = _audit_sample(affected, sample_size=audit_sample_size, seed=audit_seed)
    summary = _summary(
        records=records,
        affected=affected,
        skipped=skipped,
        attempts=attempts,
        before_counts=before_counts,
        after_counts=after_counts,
        before_image_counts=before_image_counts,
        after_image_counts=after_image_counts,
        audit_sample=audit_sample,
        input_path=input_path,
        output_path=output_path,
        generated_at=generated_at,
        min_contiguous_tokens=min_contiguous_tokens,
        adjacency_threshold=adjacency_threshold,
        confidence_threshold=confidence_threshold,
        audit_seed=audit_seed,
    )
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "recovery_version": RECOVERY_VERSION,
        "summary": summary,
        "affected_ids": affected,
        "attempts": attempts,
        "skipped": skipped,
        "audit_sample": audit_sample,
    }
    _attach_payload_metadata(output_payload, summary)
    return output_payload, report


def write_recovery_reports(report: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_json(report, report_dir / "partial_question_block_recovery_report.json", indent=2, ensure_ascii=False)
    _write_csv(report_dir / "affected_ids.csv", report["affected_ids"])
    _write_csv(report_dir / "reconstruction_log.csv", report["attempts"])
    _write_csv(report_dir / "skipped_ids.csv", report["skipped"])
    _write_csv(report_dir / "audit_sample_50.csv", report["audit_sample"])
    (report_dir / "partial_question_block_recovery_summary.md").write_text(_markdown_summary(report), encoding="utf-8")


def _records(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return payload["questions"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Question bank payload must be a list or contain a questions list.")


def _recovery_decision(
    record: dict[str, Any],
    *,
    min_contiguous_tokens: int,
    adjacency_threshold: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    base = {
        "attempted": False,
        "recoverable": False,
        "skip_reason": "",
        "reduction_source": "",
        "reconstructed_text": "",
        "reconstruction_confidence": 0.0,
        "contiguous_token_span": 0,
        "spatial_adjacency_score": 0.0,
        "eligible_signals": [],
        "selected_source": "",
        "original_question_text_sha256": _sha256(_text(record.get("question_text"))),
    }
    if not _is_partial_failure(record):
        base["skip_reason"] = "not_partial_question_block"
        return base

    base["attempted"] = True
    notes = _notes(record)
    review_flags = set(_list_value(notes.get("review_flags")) + _list_value(record.get("review_flags")))
    validation_flags = set(_list_value(notes.get("validation_flags")) + _list_value(record.get("validation_flags")))
    eligible_signals = sorted(review_flags & ELIGIBLE_REVIEW_FLAGS)
    if _text(notes.get("question_crop_confidence") or record.get("question_crop_confidence")).lower() == "low":
        eligible_signals.append("question_crop_confidence_low")
    eligible_signals = sorted(set(eligible_signals))
    base["eligible_signals"] = eligible_signals
    if not eligible_signals:
        base["skip_reason"] = "no_allowed_partial_question_block_signal"
        return base
    if validation_flags & UNSAFE_VALIDATION_FLAGS:
        base["skip_reason"] = "unsafe_scope_validation_flag"
        return base

    text, selected_source = _best_reconstructed_text(record)
    contiguous_span = _max_contiguous_tokens(text)
    base["reconstructed_text"] = text
    base["selected_source"] = selected_source
    base["contiguous_token_span"] = contiguous_span
    if contiguous_span < min_contiguous_tokens:
        base["skip_reason"] = "no_contiguous_text_span"
        return base

    adjacency_score = _spatial_adjacency_score(notes.get("question_crop_diagnostics"))
    base["spatial_adjacency_score"] = adjacency_score
    reduction_source = _reduction_source(review_flags, adjacency_score, adjacency_threshold)
    if reduction_source == "discarded":
        base["skip_reason"] = "spatial_adjacency_below_threshold"
        return base
    if reduction_source == "merged" and adjacency_score <= adjacency_threshold:
        base["skip_reason"] = "spatial_adjacency_below_threshold"
        return base
    if not _text(record.get("mark_scheme_text")):
        base["skip_reason"] = "missing_mark_scheme_text"
        return base

    confidence = _reconstruction_confidence(
        record,
        text=text,
        contiguous_span=contiguous_span,
        eligible_signals=eligible_signals,
        adjacency_score=adjacency_score,
        adjacency_threshold=adjacency_threshold,
        validation_flags=validation_flags,
    )
    base["reconstruction_confidence"] = confidence
    base["reduction_source"] = reduction_source
    if confidence < confidence_threshold:
        base["skip_reason"] = "reconstruction_confidence_below_threshold"
        return base
    base["recoverable"] = True
    return base


def _is_partial_failure(record: dict[str, Any]) -> bool:
    return _field(record, "mapping_status") == "fail" and _field(record, "mapping_failure_reason") == PARTIAL_REASON


def _field(record: dict[str, Any], name: str) -> str:
    notes = _notes(record)
    return _text(notes.get(name) if notes.get(name) not in (None, "") else record.get(name))


def _notes(record: dict[str, Any]) -> dict[str, Any]:
    notes = record.setdefault("notes", {})
    return notes if isinstance(notes, dict) else {}


def _best_reconstructed_text(record: dict[str, Any]) -> tuple[str, str]:
    candidates = [
        ("question_text", _text(record.get("question_text"))),
        ("ocr_text", _text(record.get("ocr_text"))),
    ]
    best_source, best_text = max(candidates, key=lambda item: _text_quality_score(item[1], record))
    return _normalize_reconstructed_text(best_text), best_source


def _text_quality_score(text: str, record: dict[str, Any]) -> float:
    if not text:
        return -1000.0
    question_number = _text(record.get("question_number"))
    score = min(80, len(_tokens(text))) * 1.0
    score += min(30, _max_contiguous_tokens(text)) * 1.4
    if question_number and re.search(rf"^\s*{re.escape(question_number)}\b", text):
        score += 8
    score += len(re.findall(r"\[[0-9]{1,2}\]", text)) * 4
    score += len(re.findall(r"\([a-h]\)|\((?:i{1,3}|iv|v|vi{0,3}|ix|x)\)", text, re.IGNORECASE)) * 2
    score -= len(re.findall(r"[A-Za-z]{18,}", text)) * 4
    return score


def _normalize_reconstructed_text(text: str) -> str:
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", text.replace("\u00a0", " "))
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]])", r"\1", cleaned)
    return cleaned.strip()


def _max_contiguous_tokens(text: str) -> int:
    spans = re.split(r"[\n.;!?]+", text)
    return max((len(_tokens(span)) for span in spans), default=0)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[+\-*/=<>≤≥θπ]+", text)


def _spatial_adjacency_score(diagnostics: Any) -> float:
    if not isinstance(diagnostics, dict):
        return 0.0
    regions = diagnostics.get("regions")
    if not isinstance(regions, list) or not regions:
        return 0.0
    boxes = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        bbox = region.get("text_bbox") or region.get("final_crop_bbox") or region.get("original_crop_bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            boxes.append(
                {
                    "page": int(region.get("page_number") or 0),
                    "x0": float(bbox["x0"]),
                    "y0": float(bbox["y0"]),
                    "x1": float(bbox["x1"]),
                    "y1": float(bbox["y1"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not boxes:
        return 0.0
    if len(boxes) == 1:
        return 0.80
    boxes.sort(key=lambda item: (item["page"], item["y0"], item["x0"]))
    scores = []
    for first, second in zip(boxes, boxes[1:]):
        if second["page"] == first["page"]:
            gap = max(0.0, second["y0"] - first["y1"])
            gap_score = max(0.0, 1.0 - gap / 90.0)
        elif second["page"] == first["page"] + 1:
            gap_score = 0.80
        else:
            gap_score = 0.0
        overlap = max(0.0, min(first["x1"], second["x1"]) - max(first["x0"], second["x0"]))
        min_width = max(1.0, min(first["x1"] - first["x0"], second["x1"] - second["x0"]))
        x_score = min(1.0, overlap / min_width)
        scores.append((gap_score * 0.65) + (x_score * 0.35))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _reduction_source(review_flags: set[str], adjacency_score: float, adjacency_threshold: float) -> str:
    if review_flags & SPLIT_REGION_FLAGS:
        if adjacency_score > adjacency_threshold:
            return "merged"
        if review_flags & (TRUNCATION_FLAGS | MISALIGNED_BBOX_FLAGS):
            return "reconstructed"
        return "discarded"
    if review_flags & MISALIGNED_BBOX_FLAGS:
        return "reconstructed"
    if review_flags & TRUNCATION_FLAGS:
        return "reconstructed"
    return "discarded"


def _reconstruction_confidence(
    record: dict[str, Any],
    *,
    text: str,
    contiguous_span: int,
    eligible_signals: list[str],
    adjacency_score: float,
    adjacency_threshold: float,
    validation_flags: set[str],
) -> float:
    notes = _notes(record)
    score = 0.48
    score += min(0.18, contiguous_span / 100.0)
    if eligible_signals:
        score += 0.10
    if adjacency_score > adjacency_threshold:
        score += 0.08
    elif adjacency_score >= 0.80:
        score += 0.06
    if _question_anchor_present(record, text):
        score += 0.06
    if _totals_compatible(notes):
        score += 0.10
    if _text(record.get("mark_scheme_text")):
        score += 0.06
    if _text(notes.get("text_fidelity_status") or record.get("text_fidelity_status")).lower() == "degraded":
        score -= 0.08
    if validation_flags & UNSAFE_VALIDATION_FLAGS:
        score -= 0.25
    return round(max(0.0, min(0.99, score)), 3)


def _question_anchor_present(record: dict[str, Any], text: str) -> bool:
    question_number = _text(record.get("question_number"))
    return bool(question_number and re.search(rf"^\s*{re.escape(question_number)}\b", text))


def _totals_compatible(notes: dict[str, Any]) -> bool:
    question_total = notes.get("question_total_detected")
    mark_scheme_total = notes.get("mark_scheme_total_detected")
    return question_total is not None and mark_scheme_total is not None and question_total == mark_scheme_total


def _apply_recovery(record: dict[str, Any], decision: dict[str, Any], *, generated_at: str) -> None:
    notes = _notes(record)
    original_question_text = _text(record.get("question_text"))
    if decision["reconstructed_text"] and decision["reconstructed_text"] != original_question_text:
        record["question_text"] = decision["reconstructed_text"]
    notes["mapping_status"] = "pass"
    notes["mapping_failure_reason"] = ""
    notes["partial_question_block_recovery"] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "recovery_version": RECOVERY_VERSION,
        "recovered_at": generated_at,
        "source_failure_reason": PARTIAL_REASON,
        "reduction_source": decision["reduction_source"],
        "selected_source": decision["selected_source"],
        "eligible_signals": decision["eligible_signals"],
        "contiguous_token_span": decision["contiguous_token_span"],
        "min_contiguous_tokens": DEFAULT_MIN_CONTIGUOUS_TOKENS,
        "spatial_adjacency_score": decision["spatial_adjacency_score"],
        "reconstruction_confidence": decision["reconstruction_confidence"],
        "original_question_text_sha256": decision["original_question_text_sha256"],
        "reconstructed_question_text_sha256": _sha256(decision["reconstructed_text"]),
        "mark_scheme_mapping_modified": False,
        "topic_forced": False,
    }
    recovery_flags = set(_list_value(notes.get("review_flags")))
    recovery_flags.add("partial_question_block_recovered")
    recovery_flags.add(f"partial_question_block_recovery_source:{decision['reduction_source']}")
    notes["review_flags"] = sorted(recovery_flags)
    notes["topic_trust_status"] = _topic_trust_after_recovery(notes)


def _topic_trust_after_recovery(notes: dict[str, Any]) -> str:
    existing = _text(notes.get("topic_trust_status"))
    if existing == "review_required":
        return existing
    return existing or "degraded_text"


def _attempt_row(record: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": _text(record.get("question_id")),
        "paper": _text(record.get("paper")),
        "question_number": _text(record.get("question_number")),
        "attempted": decision["attempted"],
        "recoverable": decision["recoverable"],
        "skip_reason": decision["skip_reason"],
        "reduction_source": decision["reduction_source"],
        "selected_source": decision["selected_source"],
        "contiguous_token_span": decision["contiguous_token_span"],
        "spatial_adjacency_score": decision["spatial_adjacency_score"],
        "reconstruction_confidence": decision["reconstruction_confidence"],
        "eligible_signals": "|".join(decision["eligible_signals"]),
        "original_question_text_sha256": decision["original_question_text_sha256"],
        "reconstructed_question_text_sha256": _sha256(decision["reconstructed_text"]),
    }


def _affected_row(record: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        **_attempt_row(record, decision),
        "mapping_status_before": "fail",
        "mapping_failure_reason_before": PARTIAL_REASON,
        "mapping_status_after": "pass",
        "mapping_failure_reason_after": "",
        "audit_expected_correct": _audit_expected_correct(decision),
    }


def _skip_row(record: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": _text(record.get("question_id")),
        "paper": _text(record.get("paper")),
        "question_number": _text(record.get("question_number")),
        "skip_reason": decision["skip_reason"],
        "contiguous_token_span": decision["contiguous_token_span"],
        "reconstruction_confidence": decision["reconstruction_confidence"],
        "eligible_signals": "|".join(decision["eligible_signals"]),
    }


def _audit_sample(affected: list[dict[str, Any]], *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    selected = list(affected)
    random.Random(seed).shuffle(selected)
    sample = sorted(selected[: min(sample_size, len(selected))], key=lambda row: row["question_id"])
    for row in sample:
        row["audit_method"] = "deterministic_reconstruction_rule_check"
    return sample


def _audit_expected_correct(row: dict[str, Any]) -> bool:
    return (
        float(row.get("reconstruction_confidence") or 0.0) >= 0.85
        and int(row.get("contiguous_token_span") or 0) >= DEFAULT_MIN_CONTIGUOUS_TOKENS
        and row.get("reduction_source") in {"reconstructed", "merged"}
    )


def _summary(
    *,
    records: list[dict[str, Any]],
    affected: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    before_counts: Counter[str],
    after_counts: Counter[str],
    before_image_counts: dict[str, int],
    after_image_counts: dict[str, int],
    audit_sample: list[dict[str, Any]],
    input_path: str,
    output_path: str,
    generated_at: str,
    min_contiguous_tokens: int,
    adjacency_threshold: float,
    confidence_threshold: float,
    audit_seed: int,
) -> dict[str, Any]:
    source_counts = Counter(row["reduction_source"] for row in affected)
    audit_expected = sum(1 for row in audit_sample if row.get("audit_expected_correct") is True)
    audit_rate = round(audit_expected / len(audit_sample), 3) if audit_sample else 0.0
    before_partial = before_counts.get(f"mapping_failed:{PARTIAL_REASON}", 0)
    after_partial = after_counts.get(f"mapping_failed:{PARTIAL_REASON}", 0)
    traceable = sum(1 for row in affected if row.get("original_question_text_sha256"))
    return {
        "generated_at": generated_at,
        "input_path": input_path,
        "output_path": output_path,
        "record_count": len(records),
        "min_contiguous_tokens": min_contiguous_tokens,
        "adjacency_threshold": adjacency_threshold,
        "confidence_threshold": confidence_threshold,
        "audit_seed": audit_seed,
        "before_partial_question_block": before_partial,
        "after_partial_question_block": after_partial,
        "partial_question_block_reduction": before_partial - after_partial,
        "attempted_count": len(attempts),
        "recovered_count": len(affected),
        "skipped_count": len(skipped),
        "reduction_source_counts": dict(source_counts),
        "traceable_reduction_count": traceable,
        "traceable_reduction_rate": round(traceable / len(affected), 3) if affected else 0.0,
        "audit_sample_size": len(audit_sample),
        "audit_expected_correct_count": audit_expected,
        "audit_expected_correct_rate": audit_rate,
        "before_issue_counts": dict(before_counts),
        "after_issue_counts": dict(after_counts),
        "before_image_risk_counts": before_image_counts,
        "after_image_risk_counts": after_image_counts,
        "image_risk_delta_counts": {
            key: after_image_counts.get(key, 0) - before_image_counts.get(key, 0)
            for key in sorted(set(before_image_counts) | set(after_image_counts))
        },
    }


def _issue_counts(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if _field(record, "mapping_status") == "fail":
            reason = _field(record, "mapping_failure_reason") or "unknown"
            counts[f"mapping_failed:{reason}"] += 1
    return counts


def _image_risk_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    orphan_image = 0
    weak_crop = 0
    for record in records:
        notes = _notes(record)
        flags = set(_list_value(notes.get("review_flags")) + _list_value(record.get("review_flags")))
        if "orphan_image" in flags:
            orphan_image += 1
        if (
            _text(notes.get("question_crop_confidence") or record.get("question_crop_confidence")).lower() not in {"", "high"}
            or _text(notes.get("mark_scheme_crop_confidence") or record.get("mark_scheme_crop_confidence")).lower()
            not in {"", "high"}
            or {"low_confidence_question_crop", "crop_uncertain", "markscheme_image_uncertain"} & flags
        ):
            weak_crop += 1
    return {"orphan_image": orphan_image, "weak_crop": weak_crop}


def _attach_payload_metadata(payload: dict[str, Any] | list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    metadata["partial_question_block_recovery"] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "recovery_version": RECOVERY_VERSION,
        "generated_at": summary["generated_at"],
        "before_partial_question_block": summary["before_partial_question_block"],
        "after_partial_question_block": summary["after_partial_question_block"],
        "recovered_count": summary["recovered_count"],
        "traceable_reduction_rate": summary["traceable_reduction_rate"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Partial Question Block Recovery",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Input: `{summary['input_path']}`",
        f"- Before `mapping_failed:partial_question_block`: `{summary['before_partial_question_block']}`",
        f"- After `mapping_failed:partial_question_block`: `{summary['after_partial_question_block']}`",
        f"- Reduction: `{summary['partial_question_block_reduction']}`",
        f"- Recovered: `{summary['recovered_count']}`",
        f"- Traceable reductions: `{summary['traceable_reduction_count']}` (`{summary['traceable_reduction_rate']:.1%}`)",
        f"- Audit sample expected correctness: `{summary['audit_expected_correct_count']}/{summary['audit_sample_size']}` (`{summary['audit_expected_correct_rate']:.1%}`)",
        "",
        "## Reduction Source",
    ]
    for source, count in sorted(summary["reduction_source_counts"].items()):
        lines.append(f"- `{source}`: `{count}`")
    lines.extend(["", "## Image Risk Delta"])
    for key, value in sorted(summary["image_risk_delta_counts"].items()):
        lines.append(f"- `{key}`: `{value:+d}`")
    lines.append("")
    return "\n".join(lines)


def _list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


if __name__ == "__main__":
    raise SystemExit(main())
