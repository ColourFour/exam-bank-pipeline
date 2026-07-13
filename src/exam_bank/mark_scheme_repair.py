from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .atomic_json import write_atomic_json
from .audit import _mark_scheme_text_foreign_question_labels
from .config import AppConfig, load_config
from .core.paper_identity import paper_identity_from_parts
from .exporters import _payload_qa_summary
from .mark_schemes import MarkSchemeImageResult, extract_mark_scheme_answers, render_mark_scheme_images


def repair_question_bank_mark_schemes(
    question_bank_path: str | Path,
    *,
    output_root: str | Path,
    config_path: str | Path | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Re-extract only papers with missing or cross-question mark-scheme data.

    Question, topic, difficulty, and review fields are deliberately left alone.
    The question bank is replaced atomically only after every attempted paper
    has been processed and only when ``write`` is explicitly enabled.
    """
    question_bank_path = Path(question_bank_path)
    output_root = Path(output_root)
    payload = json.loads(question_bank_path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("questions", []) if isinstance(row, dict)]
    known_by_paper = _known_question_numbers(rows)
    reasons_by_id = _repair_reasons(rows, known_by_paper)
    rows_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_paper[str(row.get("paper") or "")].append(row)

    config = load_config(Path(config_path)) if config_path else AppConfig()
    config.output.apply_root(output_root)
    attempts: list[dict[str, Any]] = []
    repaired_ids: list[str] = []

    for paper in sorted({str(row.get("paper") or "") for row in rows if str(row.get("question_id") or "") in reasons_by_id}):
        paper_rows = sorted(rows_by_paper[paper], key=lambda row: _question_sort_key(row.get("question_number")))
        source_pdf = _mark_scheme_source(paper_rows)
        target_rows = [row for row in paper_rows if str(row.get("question_id") or "") in reasons_by_id]
        attempt: dict[str, Any] = {
            "paper": paper,
            "source_pdf": str(source_pdf) if source_pdf else "",
            "target_question_ids": [str(row.get("question_id") or "") for row in target_rows],
            "target_count": len(target_rows),
            "repaired_question_ids": [],
            "failed_question_ids": [],
            "mapping_method_counts": {},
            "error": "",
        }
        if source_pdf is None or not source_pdf.is_file():
            attempt["error"] = "missing_mark_scheme_source_pdf"
            attempt["failed_question_ids"] = list(attempt["target_question_ids"])
            attempts.append(attempt)
            continue

        expected = [str(row.get("question_number") or "") for row in paper_rows]
        identities = {
            str(row.get("question_number") or ""): _identity_for_row(row)
            for row in paper_rows
        }
        try:
            answers = extract_mark_scheme_answers(source_pdf, config, expected)
            images = render_mark_scheme_images(
                source_pdf,
                config,
                expected,
                question_marks={str(row.get("question_number") or ""): _int_or_none(row.get("question_solution_marks")) for row in paper_rows},
                question_subparts={str(row.get("question_number") or ""): _string_list(row.get("subparts")) for row in paper_rows},
                question_identities=identities,
                clear_stale=False,
            )
        except Exception as exc:  # fail closed per paper; preserve the original payload
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempt["failed_question_ids"] = list(attempt["target_question_ids"])
            attempts.append(attempt)
            continue

        method_counts: Counter[str] = Counter()
        for row in target_rows:
            question_id = str(row.get("question_id") or "")
            number = str(row.get("question_number") or "")
            result = images.get(number)
            answer = str(answers.get(number) or "").strip()
            if result is None or not result.image_path:
                focused = render_mark_scheme_images(
                    source_pdf,
                    config,
                    [number],
                    question_marks={number: _int_or_none(row.get("question_solution_marks"))},
                    question_subparts={number: _string_list(row.get("subparts"))},
                    question_identities={number: identities[number]},
                    clear_stale=False,
                )
                result = focused.get(number)
            if result is None or not result.image_path or not answer:
                attempt["failed_question_ids"].append(question_id)
                continue
            foreign = _mark_scheme_text_foreign_question_labels(
                row | {"mark_scheme_text": answer},
                known_question_numbers=known_by_paper.get(paper, set()),
            )
            if foreign:
                attempt["failed_question_ids"].append(question_id)
                continue
            _apply_repaired_mark_scheme(row, result=result, answer=answer, output_root=output_root)
            method_counts.update([result.mapping_method or "unknown"])
            attempt["repaired_question_ids"].append(question_id)
            repaired_ids.append(question_id)
        attempt["mapping_method_counts"] = dict(sorted(method_counts.items()))
        attempts.append(attempt)

    residual_reasons = _repair_reasons(rows, known_by_paper)
    report = {
        "schema_name": "exam_bank.mark_scheme_targeted_repair",
        "schema_version": 1,
        "question_bank": str(question_bank_path),
        "output_root": str(output_root),
        "write": write,
        "initial_target_count": len(reasons_by_id),
        "initial_reason_counts": dict(sorted(Counter(reason for values in reasons_by_id.values() for reason in values).items())),
        "paper_count": len(attempts),
        "repaired_count": len(set(repaired_ids)),
        "residual_count": len(residual_reasons),
        "residual_reason_counts": dict(sorted(Counter(reason for values in residual_reasons.values() for reason in values).items())),
        "ok": not residual_reasons,
        "attempts": attempts,
    }
    if write:
        payload["record_count"] = len(rows)
        payload["questions"] = rows
        run_manifest = dict(payload.get("run_manifest") if isinstance(payload.get("run_manifest"), dict) else {})
        run_manifest["qa_summary"] = _payload_qa_summary(rows)
        payload["run_manifest"] = run_manifest
        write_atomic_json(payload, question_bank_path, sort_keys=False)
    return report


def _repair_reasons(
    rows: list[dict[str, Any]],
    known_by_paper: dict[str, set[str]],
) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        row_reasons: list[str] = []
        if not str(row.get("mark_scheme_image_path") or "").strip():
            row_reasons.append("missing_mark_scheme_image_path")
        if _mark_scheme_text_foreign_question_labels(
            row,
            known_question_numbers=known_by_paper.get(str(row.get("paper") or ""), set()),
        ):
            row_reasons.append("foreign_question_label")
        if row_reasons:
            reasons[question_id] = row_reasons
    return reasons


def _apply_repaired_mark_scheme(
    row: dict[str, Any],
    *,
    result: MarkSchemeImageResult,
    answer: str,
    output_root: Path,
) -> None:
    image_path = Path(result.image_path) if result.image_path else None
    if image_path is None:
        raise ValueError("repaired result has no image path")
    try:
        relative_path = image_path.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"mark-scheme image escaped output root: {image_path}") from exc

    row["canonical_mark_scheme_artifact"] = relative_path
    row["mark_scheme_image_path"] = relative_path
    row["mark_scheme_image_paths"] = [relative_path]
    row["mark_scheme_text"] = answer
    row["mark_scheme_block_ids"] = list(result.block_ids)
    row["mark_scheme_confidence_score"] = round(float(result.confidence_score or 0.0), 3)
    page_refs = dict(row.get("page_refs") if isinstance(row.get("page_refs"), dict) else {})
    page_refs["mark_scheme"] = list(result.page_numbers)
    row["page_refs"] = page_refs

    notes = dict(row.get("notes") if isinstance(row.get("notes"), dict) else {})
    notes["mapping_status"] = result.mapping_status
    notes["mapping_failure_reason"] = result.failure_reason
    notes["missing_mark_scheme_reason"] = ""
    notes["mark_scheme_crop_confidence"] = result.crop_confidence
    notes["mark_scheme_block_ids"] = list(result.block_ids)
    notes["mark_scheme_confidence_score"] = round(float(result.confidence_score or 0.0), 3)
    notes["mark_scheme_total_detected"] = result.markscheme_marks_total
    notes["mark_scheme_structure_detected"] = {
        "subparts": list(result.markscheme_subparts),
        "question_subparts": list(result.question_subparts),
        "question_total_detected": result.question_marks_total,
        "mark_scheme_total_detected": result.markscheme_marks_total,
        "mark_scheme_block_ids": list(result.block_ids),
        "mark_scheme_confidence_score": round(float(result.confidence_score or 0.0), 3),
        "missing_mark_scheme_reason": "",
        "asset_identity": {
            "question_id": result.question_id,
            "paper_id": result.paper_id,
            "component": result.component,
            "canonical_path": result.canonical_path or relative_path,
        },
    }
    review_flags = [
        flag
        for flag in _string_list(notes.get("review_flags"))
        if not flag.startswith("markscheme_") and flag not in {"unmatched_answer", "missing_mark_scheme_image_path"}
    ]
    notes["review_flags"] = sorted(set([*review_flags, *result.review_flags]))
    validation_flags = [
        flag
        for flag in _string_list(notes.get("validation_flags"))
        if flag not in {"missing_mark_scheme_image_path", "markscheme_image_missing"}
    ]
    notes["validation_flags"] = validation_flags
    if result.mapping_status != "pass":
        notes["validation_status"] = "fail"
        if result.failure_reason and result.failure_reason not in validation_flags:
            validation_flags.append(result.failure_reason)
        notes["validation_flags"] = validation_flags
    elif notes.get("validation_status") == "fail" and not validation_flags:
        notes["validation_status"] = "pass"
    row["notes"] = notes


def _known_question_numbers(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        paper = str(row.get("paper") or "")
        number = str(row.get("question_number") or "")
        if paper and number:
            known[paper].add(number)
    return dict(known)


def _identity_for_row(row: dict[str, Any]):
    notes = row.get("notes") if isinstance(row.get("notes"), dict) else {}
    return paper_identity_from_parts(
        syllabus="9709",
        subject_family=str(row.get("paper_family") or ""),
        year=str(row.get("canonical_year_folder") or ""),
        session=str(row.get("canonical_session") or ""),
        component=str(notes.get("source_paper_code") or ""),
        question_number=str(row.get("question_number") or ""),
        expected_question_id=str(row.get("question_id") or ""),
    )


def _mark_scheme_source(rows: list[dict[str, Any]]) -> Path | None:
    for row in rows:
        notes = row.get("notes") if isinstance(row.get("notes"), dict) else {}
        value = str(notes.get("mark_scheme_source_pdf") or "")
        if value:
            return Path(value)
    return None


def _question_sort_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    try:
        return int(text), text
    except ValueError:
        return 10_000, text


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def run_repair(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair only missing or cross-question mark-scheme records.")
    parser.add_argument("--input", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("reports/mark_scheme_targeted_repair.json"))
    parser.add_argument("--write", action="store_true", help="Atomically update the generated question bank.")
    args = parser.parse_args(argv)
    report = repair_question_bank_mark_schemes(
        args.input,
        output_root=args.output_root,
        config_path=args.config,
        write=args.write,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_json(report, args.report, sort_keys=True)
    print(json.dumps({key: value for key, value in report.items() if key != "attempts"}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(run_repair())
