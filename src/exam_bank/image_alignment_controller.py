from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .asset_manifest import deterministic_asset_id
from .atomic_json import write_atomic_json
from .core.paper_identity import IdentityError, build_question_id
from .image_alignment_metrics import (
    FAILURE_DUPLICATE_MAPPING,
    FAILURE_LEGACY_SEGMENTATION,
    FAILURE_MISALIGNED_IMAGE,
    FAILURE_MISSING_IMAGE,
    FAILURE_ORPHAN_IMAGE,
    FAILURE_WEAK_CROP,
    IMAGE_KINDS,
    AlignmentEvaluation,
    AlignmentFailure,
    AlignmentMapping,
    evaluate_image_alignment,
    image_slot_fields,
    mapping_from_record,
    parse_asset_path_metadata,
)


IMAGE_ALIGNMENT_LOOP_SCHEMA_NAME = "exam_bank.image_alignment_loop_report"
IMAGE_ALIGNMENT_LOOP_SCHEMA_VERSION = 1
IMAGE_BINDING_REPAIR_SCHEMA_NAME = "exam_bank.image_binding_repair_report"
IMAGE_BINDING_REPAIR_SCHEMA_VERSION = 1
DEFAULT_ALIGNMENT_THRESHOLD = 0.90
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_REPORT_PATH = Path("output/audits/image_alignment_loop_report.json")
DEFAULT_BINDING_REPORT_PATH = Path("output/audits/image_binding_repair_report.json")
UNASSIGNED_ASSET_POOL_NAME = "unassigned_asset_pool"

REPAIR_STRATEGIES = {
    FAILURE_MISSING_IMAGE: "rerun_pdf_page_scan_for_figure_detection",
    FAILURE_MISALIGNED_IMAGE: "reanchor_to_corrected_question_boundary",
    FAILURE_ORPHAN_IMAGE: "nearest_valid_question_match_by_bbox_proximity",
    FAILURE_WEAK_CROP: "expand_crop_region_and_reextract",
    FAILURE_DUPLICATE_MAPPING: "enforce_deterministic_asset_id_rule",
    FAILURE_LEGACY_SEGMENTATION: "switch_to_relaxed_page_segmentation_mode",
}


@dataclass(frozen=True)
class AlignmentLoopContext:
    iteration: int
    previous_score: float | None = None
    repair_actions: tuple[dict[str, Any], ...] = ()
    extraction_mode: str = "standard"
    repaired_question_bank: dict[str, Any] | None = None
    artifact_root: Path | None = None


@dataclass(frozen=True)
class AlignmentExtractionOutput:
    question_bank: dict[str, Any]
    question_bank_path: Path | None = None
    artifact_root: Path | None = None
    extraction_log: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentLoopResult:
    initial_alignment_score: float
    final_alignment_score: float
    iterations_run: int
    failure_breakdown: dict[str, int]
    reached_threshold: bool
    report_path: Path
    report: dict[str, Any]
    binding_report_path: Path
    binding_report: dict[str, Any]


@dataclass(frozen=True)
class ImageBindingContext:
    question_bank: dict[str, Any]
    artifact_root: Path
    image_kind: str = ""
    protected_mappings: dict[str, AlignmentMapping] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageBindingResolution:
    question_id: str | None
    method: str | None
    reason: str = ""
    score: float = 0.0


ExtractionRunner = Callable[[AlignmentLoopContext], AlignmentExtractionOutput | dict[str, Any] | str | Path | Any]


def run_image_alignment_loop(
    extraction_runner: ExtractionRunner,
    *,
    artifact_root: str | Path = "output",
    report_path: str | Path = DEFAULT_REPORT_PATH,
    binding_report_path: str | Path = DEFAULT_BINDING_REPORT_PATH,
    threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    expected_image_kinds: tuple[str, ...] = IMAGE_KINDS,
    scan_orphans: bool = True,
) -> AlignmentLoopResult:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in the range (0, 1]")

    root = Path(artifact_root)
    output_report_path = Path(report_path)
    output_binding_report_path = Path(binding_report_path)
    repair_actions: tuple[dict[str, Any], ...] = ()
    repaired_question_bank: dict[str, Any] | None = None
    protected_mappings: dict[str, AlignmentMapping] = {}
    history: list[dict[str, Any]] = []
    repair_log: list[dict[str, Any]] = []
    safety_events: list[dict[str, Any]] = []
    max_expected_images_seen = 0
    previous_referenced_image_count: int | None = None
    initial_score: float | None = None
    final_evaluation: AlignmentEvaluation | None = None
    final_question_bank: dict[str, Any] = {}
    initial_orphan_count: int | None = None

    for iteration in range(1, max_iterations + 1):
        context = AlignmentLoopContext(
            iteration=iteration,
            previous_score=final_evaluation.alignment_score if final_evaluation is not None else None,
            repair_actions=repair_actions,
            extraction_mode=_extraction_mode_for_actions(repair_actions),
            repaired_question_bank=repaired_question_bank,
            artifact_root=root,
        )
        extraction_output = _normalize_extraction_output(extraction_runner(context), default_artifact_root=root)
        evaluation_root = extraction_output.artifact_root or root
        question_bank, restore_actions, restore_events = _restore_protected_mappings(
            extraction_output.question_bank,
            protected_mappings,
        )
        repair_log.extend(restore_actions)
        safety_events.extend(restore_events)
        question_bank, binding_restore_actions = _restore_successful_image_bindings(
            question_bank,
            repair_actions,
            protected_mappings,
            artifact_root=evaluation_root,
        )
        repair_log.extend(binding_restore_actions)

        evaluation = evaluate_image_alignment(
            question_bank,
            artifact_root=evaluation_root,
            expected_image_kinds=expected_image_kinds,
            scan_orphans=scan_orphans,
        )
        final_evaluation = evaluation
        final_question_bank = question_bank
        if initial_score is None:
            initial_score = evaluation.alignment_score
        if initial_orphan_count is None:
            initial_orphan_count = evaluation.orphan_assets_count

        referenced_image_count = _referenced_image_count(question_bank, expected_image_kinds)
        iteration_safety_events: list[dict[str, Any]] = []
        if evaluation.expected_images < max_expected_images_seen:
            event = {
                "event": "expected_image_count_decreased",
                "iteration": iteration,
                "previous_max_expected_images": max_expected_images_seen,
                "current_expected_images": evaluation.expected_images,
                "reason": "image count stability is enforced across iterations",
            }
            safety_events.append(event)
            iteration_safety_events.append(event)
        max_expected_images_seen = max(max_expected_images_seen, evaluation.expected_images)

        if previous_referenced_image_count is not None and referenced_image_count < previous_referenced_image_count:
            event = {
                "event": "referenced_image_count_decreased",
                "iteration": iteration,
                "previous_referenced_image_count": previous_referenced_image_count,
                "current_referenced_image_count": referenced_image_count,
                "reason": "images must not be silently dropped during repair",
            }
            safety_events.append(event)
            iteration_safety_events.append(event)
        previous_referenced_image_count = referenced_image_count

        _protect_correct_mappings(protected_mappings, evaluation.correct_mappings)
        history.append(
            {
                "iteration": iteration,
                "alignment_score": evaluation.alignment_score,
                "correctly_mapped_images": evaluation.correctly_mapped_images,
                "expected_images": evaluation.expected_images,
                "referenced_image_count": referenced_image_count,
                "failure_distribution": dict(evaluation.failure_distribution),
                "unresolved_assets_count": evaluation.unresolved_assets_count,
                "orphan_assets_count": evaluation.orphan_assets_count,
                "repair_action_count": len(repair_actions),
                "extraction_mode": context.extraction_mode,
                "extraction": dict(extraction_output.extraction_log),
                "safety_events": iteration_safety_events,
            }
        )

        if evaluation.alignment_score >= threshold and evaluation.orphan_assets_count == 0:
            break
        repair_result = apply_alignment_repairs(
            question_bank,
            evaluation.failures,
            artifact_root=evaluation_root,
            protected_mappings=protected_mappings,
        )
        repaired_question_bank = repair_result["question_bank"]
        repair_actions = tuple(repair_result["actions"])
        repair_log.extend(repair_result["actions"])

    if final_evaluation is None or initial_score is None:
        raise RuntimeError("alignment loop did not run")

    image_count_stability_ok = not any(
        event["event"] in {"expected_image_count_decreased", "referenced_image_count_decreased"}
        for event in safety_events
    )
    persistent_orphan_free = final_evaluation.orphan_assets_count == 0
    reached_threshold = final_evaluation.alignment_score >= threshold and image_count_stability_ok and persistent_orphan_free
    binding_report = _build_image_binding_repair_report(
        initial_orphan_count=initial_orphan_count or 0,
        final_orphan_count=final_evaluation.orphan_assets_count,
        reached_threshold=reached_threshold,
        repair_log=repair_log,
        final_evaluation=final_evaluation,
        max_iterations=max_iterations,
    )
    report = {
        "schema_name": IMAGE_ALIGNMENT_LOOP_SCHEMA_NAME,
        "schema_version": IMAGE_ALIGNMENT_LOOP_SCHEMA_VERSION,
        "threshold": threshold,
        "max_iterations": max_iterations,
        "iterations_run": len(history),
        "initial_alignment_score": initial_score,
        "final_alignment_score": final_evaluation.alignment_score,
        "final_convergence_status": "converged" if reached_threshold else "fail",
        "reached_threshold": reached_threshold,
        "iteration_history": history,
        "failure_breakdown": dict(final_evaluation.failure_distribution),
        "unresolved_assets_count": final_evaluation.unresolved_assets_count,
        "orphan_images": final_evaluation.orphan_assets_count,
        "alignment_status": "PASS" if reached_threshold else "FAIL",
        "binding_report_path": str(output_binding_report_path),
        "safety": {
            "placeholder_generation_allowed": False,
            "image_count_stability_enforced": True,
            "image_count_stability_ok": image_count_stability_ok,
            "persistent_orphan_state_allowed": False,
            "persistent_orphan_free": persistent_orphan_free,
            "protected_correct_mapping_count": len(protected_mappings),
            "events": safety_events,
        },
        "repair_log": repair_log,
    }
    write_atomic_json(report, output_report_path, sort_keys=True)
    write_atomic_json(binding_report, output_binding_report_path, sort_keys=True)
    return AlignmentLoopResult(
        initial_alignment_score=initial_score,
        final_alignment_score=final_evaluation.alignment_score,
        iterations_run=len(history),
        failure_breakdown=dict(final_evaluation.failure_distribution),
        reached_threshold=reached_threshold,
        report_path=output_report_path,
        report=report,
        binding_report_path=output_binding_report_path,
        binding_report=binding_report,
    )


def apply_alignment_repairs(
    question_bank: dict[str, Any],
    failures: tuple[AlignmentFailure, ...] | list[AlignmentFailure],
    *,
    artifact_root: str | Path,
    protected_mappings: dict[str, AlignmentMapping] | None = None,
) -> dict[str, Any]:
    payload = deepcopy(question_bank)
    records = _records(payload)
    by_question_id = {str(record.get("question_id") or ""): record for record in records if record.get("question_id")}
    protected = protected_mappings or {}
    actions: list[dict[str, Any]] = []

    for index, failure in enumerate(failures, start=1):
        strategy = REPAIR_STRATEGIES[failure.failure_type]
        action = _repair_action(index, failure, strategy=strategy)
        if failure.failure_type == FAILURE_ORPHAN_IMAGE:
            matched = _apply_orphan_repair(action, failure, by_question_id, protected, artifact_root=Path(artifact_root))
            action.update(matched)
        elif failure.failure_type == FAILURE_DUPLICATE_MAPPING:
            action["after"] = {
                "asset_id_rule": deterministic_asset_id(
                    failure.image_kind,
                    failure.paper_id or "unknown_paper",
                    failure.question_id or "unknown_question",
                ),
            }
        elif failure.failure_type == FAILURE_MISSING_IMAGE:
            action["after"] = {"next_extraction_mode": "figure_detection_rescan"}
        elif failure.failure_type == FAILURE_MISALIGNED_IMAGE:
            action["after"] = {"next_extraction_mode": "question_boundary_reanchor"}
        elif failure.failure_type == FAILURE_WEAK_CROP:
            action["after"] = {"next_extraction_mode": "expanded_crop_reextract"}
        elif failure.failure_type == FAILURE_LEGACY_SEGMENTATION:
            action["after"] = {"next_extraction_mode": "relaxed_page_segmentation"}
        actions.append(action)

    return {"question_bank": payload, "actions": actions}


def _normalize_extraction_output(result: Any, *, default_artifact_root: Path) -> AlignmentExtractionOutput:
    if isinstance(result, AlignmentExtractionOutput):
        return result
    if isinstance(result, (str, Path)):
        path = Path(result)
        return AlignmentExtractionOutput(
            question_bank=json.loads(path.read_text(encoding="utf-8")),
            question_bank_path=path,
            artifact_root=_infer_artifact_root(path),
        )
    if isinstance(result, dict) and isinstance(result.get("questions"), list):
        return AlignmentExtractionOutput(question_bank=result, artifact_root=default_artifact_root)
    json_path = getattr(result, "json_path", None)
    if json_path:
        path = Path(json_path)
        return AlignmentExtractionOutput(
            question_bank=json.loads(path.read_text(encoding="utf-8")),
            question_bank_path=path,
            artifact_root=Path(getattr(result, "output_root", default_artifact_root)),
        )
    raise TypeError("extraction_runner must return AlignmentExtractionOutput, question-bank payload, path, or PipelineResult-like object")


def _restore_protected_mappings(
    question_bank: dict[str, Any],
    protected_mappings: dict[str, AlignmentMapping],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not protected_mappings:
        return question_bank, [], []
    payload = deepcopy(question_bank)
    records = _records(payload)
    by_question_id = {str(record.get("question_id") or ""): record for record in records if record.get("question_id")}
    actions: list[dict[str, Any]] = []
    safety_events: list[dict[str, Any]] = []
    for mapping in protected_mappings.values():
        record = by_question_id.get(mapping.question_id)
        if record is None:
            safety_events.append(
                {
                    "event": "protected_mapping_record_missing",
                    "question_id": mapping.question_id,
                    "image_kind": mapping.image_kind,
                    "reason": "previously correct mapping cannot be checked because the record is absent",
                }
            )
            continue
        current = mapping_from_record(record, mapping.image_kind)
        if current.path == mapping.path:
            continue
        _set_slot_path(record, mapping.image_kind, mapping.path)
        action = {
            "failure_type": "protected_correct_mapping_degraded",
            "strategy": "restore_previous_correct_mapping",
            "question_id": mapping.question_id,
            "paper_id": mapping.paper_id,
            "image_kind": mapping.image_kind,
            "before": {"path": current.path},
            "after": {"path": mapping.path},
            "reversible": True,
            "reason": "valid correct mappings must not be overwritten by later repair passes",
        }
        actions.append(action)
        safety_events.append(
            {
                "event": "protected_mapping_restored",
                "question_id": mapping.question_id,
                "image_kind": mapping.image_kind,
                "reason": action["reason"],
            }
        )
    return payload, actions, safety_events


def _protect_correct_mappings(
    protected_mappings: dict[str, AlignmentMapping],
    correct_mappings: tuple[AlignmentMapping, ...],
) -> None:
    for mapping in correct_mappings:
        protected_mappings.setdefault(mapping.slot_id, mapping)


def _repair_action(index: int, failure: AlignmentFailure, *, strategy: str) -> dict[str, Any]:
    return {
        "repair_id": f"image_alignment_repair:{index:04d}:{failure.failure_type}",
        "failure_type": failure.failure_type,
        "strategy": strategy,
        "question_id": failure.question_id,
        "paper_id": failure.paper_id,
        "question_number": failure.question_number,
        "image_kind": failure.image_kind,
        "before": {
            "path": failure.path,
            "field": failure.field,
            "asset_id": failure.asset_id,
            "reason": failure.reason,
            "details": dict(failure.details),
        },
        "after": {},
        "reversible": True,
    }


def _apply_orphan_repair(
    action: dict[str, Any],
    failure: AlignmentFailure,
    by_question_id: dict[str, dict[str, Any]],
    protected_mappings: dict[str, AlignmentMapping],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    context = ImageBindingContext(
        question_bank={"questions": list(by_question_id.values())},
        artifact_root=artifact_root,
        image_kind=failure.image_kind,
        protected_mappings=protected_mappings,
    )
    resolution = resolve_image_binding(failure, context)
    if not resolution.question_id:
        return {
            "after": {
                "matched": False,
                "reason": resolution.reason or "image could not be deterministically bound to PaperIdentity.question_id",
                "retry_queue": True,
                "unassigned_pool": UNASSIGNED_ASSET_POOL_NAME,
            }
        }
    question_id = resolution.question_id
    record = by_question_id.get(question_id)
    if record is None:
        return {"after": {"matched": False, "reason": "no matching question record for orphan asset"}}
    metadata = parse_asset_path_metadata(failure.path)
    image_kind = failure.image_kind or str(metadata.get("image_kind") or "")
    paper_id = failure.paper_id or str(metadata.get("paper_id") or record.get("paper") or record.get("canonical_paper_id") or "")
    if image_kind not in IMAGE_KINDS:
        return {"after": {"matched": False, "reason": "orphan image kind is missing or unsupported"}}
    current = mapping_from_record(record, image_kind)
    if current.slot_id in protected_mappings:
        return {"after": {"matched": False, "reason": "matching slot is protected as previously correct"}}
    if current.path and (artifact_root / current.path).is_file():
        return {"after": {"matched": False, "reason": "matching slot already has an existing image"}}
    _set_slot_path(record, image_kind, failure.path)
    action["question_id"] = question_id
    action["paper_id"] = paper_id
    action["image_kind"] = image_kind
    return {
        "after": {
            "matched": True,
            "path": failure.path,
            "field_updates": list(image_slot_fields(image_kind)),
            "match_method": resolution.method,
            "binding_score": resolution.score,
            "retry_queue": False,
            "identity_field": "PaperIdentity.question_id",
        }
    }


def resolve_image_to_question(image: AlignmentFailure | dict[str, Any], context: ImageBindingContext) -> str | None:
    return resolve_image_binding(image, context).question_id


def resolve_image_binding(image: AlignmentFailure | dict[str, Any], context: ImageBindingContext) -> ImageBindingResolution:
    image_payload = _image_payload(image)
    candidates = _records(context.question_bank)
    if not candidates:
        return ImageBindingResolution(None, None, "no question records available for binding")

    for resolver in (
        _resolve_by_bbox_overlap,
        _resolve_by_page_vertical_anchor,
        _resolve_by_textual_anchor,
        _resolve_by_paper_identity_fallback,
    ):
        resolution = resolver(image_payload, candidates, context)
        if resolution.question_id:
            return resolution
    return ImageBindingResolution(None, None, "no deterministic question binding found")


def _restore_successful_image_bindings(
    question_bank: dict[str, Any],
    repair_actions: tuple[dict[str, Any], ...],
    protected_mappings: dict[str, AlignmentMapping],
    *,
    artifact_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    successful = [
        action
        for action in repair_actions
        if action.get("failure_type") == FAILURE_ORPHAN_IMAGE and action.get("after", {}).get("matched") is True
    ]
    if not successful:
        return question_bank, []
    payload = deepcopy(question_bank)
    by_question_id = {str(record.get("question_id") or ""): record for record in _records(payload) if record.get("question_id")}
    actions: list[dict[str, Any]] = []
    for action in successful:
        question_id = str(action.get("question_id") or "")
        image_kind = str(action.get("image_kind") or "")
        path = str(action.get("after", {}).get("path") or action.get("before", {}).get("path") or "")
        if not question_id or image_kind not in IMAGE_KINDS or not path:
            continue
        record = by_question_id.get(question_id)
        if record is None:
            continue
        current = mapping_from_record(record, image_kind)
        if current.path == path:
            continue
        if current.slot_id in protected_mappings:
            continue
        if current.path and (artifact_root / current.path).is_file():
            continue
        _set_slot_path(record, image_kind, path)
        actions.append(
            {
                "failure_type": FAILURE_ORPHAN_IMAGE,
                "strategy": "restore_successful_image_binding",
                "question_id": question_id,
                "paper_id": action.get("paper_id"),
                "image_kind": image_kind,
                "before": {"path": current.path},
                "after": {
                    "matched": True,
                    "path": path,
                    "match_method": action.get("after", {}).get("match_method"),
                    "identity_field": "PaperIdentity.question_id",
                },
                "reversible": True,
            }
        )
    return payload, actions


def _resolve_by_bbox_overlap(
    image: dict[str, Any],
    candidates: list[dict[str, Any]],
    context: ImageBindingContext,
) -> ImageBindingResolution:
    image_bbox = _bbox_from_payload(image)
    image_page = _page_from_payload(image)
    if image_bbox is None:
        return ImageBindingResolution(None, "bbox_overlap", "image bbox is unavailable")
    scored: list[tuple[float, str]] = []
    for record in candidates:
        question_id = str(record.get("question_id") or "")
        question_bbox = _question_bbox(record)
        if not question_id or question_bbox is None:
            continue
        question_page = _page_from_record(record)
        if image_page is not None and question_page is not None and image_page != question_page:
            continue
        overlap = _overlap_ratio(image_bbox, question_bbox)
        if overlap > 0:
            scored.append((overlap, question_id))
    if not scored:
        return ImageBindingResolution(None, "bbox_overlap", "no question region overlaps image bbox")
    score, question_id = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
    return ImageBindingResolution(question_id, "bbox_overlap", score=round(score, 6))


def _resolve_by_page_vertical_anchor(
    image: dict[str, Any],
    candidates: list[dict[str, Any]],
    context: ImageBindingContext,
) -> ImageBindingResolution:
    image_page = _page_from_payload(image)
    image_y = _vertical_anchor_from_payload(image)
    if image_page is None or image_y is None:
        return ImageBindingResolution(None, "page_vertical_anchor_proximity", "image page or vertical anchor is unavailable")
    scored: list[tuple[float, str]] = []
    for record in candidates:
        question_id = str(record.get("question_id") or "")
        question_page = _page_from_record(record)
        question_y = _vertical_anchor_from_record(record)
        if not question_id or question_page is None or question_y is None or question_page != image_page:
            continue
        scored.append((abs(image_y - question_y), question_id))
    if not scored:
        return ImageBindingResolution(None, "page_vertical_anchor_proximity", "no same-page question anchor available")
    distance, question_id = sorted(scored, key=lambda item: (item[0], item[1]))[0]
    return ImageBindingResolution(question_id, "page_vertical_anchor_proximity", score=round(1.0 / (1.0 + distance), 6))


def _resolve_by_textual_anchor(
    image: dict[str, Any],
    candidates: list[dict[str, Any]],
    context: ImageBindingContext,
) -> ImageBindingResolution:
    image_question = _question_number_from_payload(image)
    if not image_question:
        return ImageBindingResolution(None, "textual_anchor_proximity", "image question-number anchor is unavailable")
    image_paper = _paper_from_payload(image)
    image_page = _page_from_payload(image)
    scored: list[tuple[int, int, str]] = []
    for record in candidates:
        question_id = str(record.get("question_id") or "")
        record_question = _normalized_question_number(record.get("question_number") or _question_number_from_question_id(question_id))
        if not question_id or not record_question:
            continue
        record_paper = str(record.get("paper") or record.get("canonical_paper_id") or "")
        if image_paper and record_paper and image_paper != record_paper:
            continue
        page_penalty = 0
        record_page = _page_from_record(record)
        if image_page is not None and record_page is not None and image_page != record_page:
            page_penalty = 1
        scored.append((abs(int(image_question) - int(record_question)), page_penalty, question_id))
    if not scored:
        return ImageBindingResolution(None, "textual_anchor_proximity", "no compatible question-number anchor found")
    distance, page_penalty, question_id = sorted(scored, key=lambda item: (item[0], item[1], item[2]))[0]
    return ImageBindingResolution(question_id, "textual_anchor_proximity", score=round(1.0 / (1.0 + distance + page_penalty), 6))


def _resolve_by_paper_identity_fallback(
    image: dict[str, Any],
    candidates: list[dict[str, Any]],
    context: ImageBindingContext,
) -> ImageBindingResolution:
    paper_id = _paper_from_payload(image)
    question_number = _question_number_from_payload(image)
    if not paper_id or not question_number:
        return ImageBindingResolution(None, "paper_identity_fallback", "paper id or question number is unavailable")
    try:
        question_id = build_question_id(paper_id, question_number)
    except IdentityError:
        return ImageBindingResolution(None, "paper_identity_fallback", "PaperIdentity.question_id could not be derived")
    valid_ids = {str(record.get("question_id") or "") for record in candidates}
    if question_id not in valid_ids:
        return ImageBindingResolution(None, "paper_identity_fallback", "derived PaperIdentity.question_id has no question record")
    return ImageBindingResolution(question_id, "paper_identity_fallback", score=1.0)


def _image_payload(image: AlignmentFailure | dict[str, Any]) -> dict[str, Any]:
    if isinstance(image, AlignmentFailure):
        metadata = parse_asset_path_metadata(image.path)
        payload: dict[str, Any] = {
            "image_kind": image.image_kind or metadata.get("image_kind"),
            "question_number": image.question_number or metadata.get("question_number"),
            "paper_id": image.paper_id or metadata.get("paper_id"),
            "path": image.path,
            "field": image.field,
            "asset_id": image.asset_id,
        }
        payload.update(image.details)
        return payload
    payload = dict(image)
    metadata = parse_asset_path_metadata(str(payload.get("path") or payload.get("canonical_path") or ""))
    for key in ("image_kind", "question_number", "paper_id"):
        payload.setdefault(key, metadata.get(key))
    return payload


def _question_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for value in (
        record.get("question_region"),
        record.get("question_bbox"),
        record.get("bbox"),
        record.get("crop_bbox"),
        record.get("final_crop_bbox"),
        _nested(record, "crop_diagnostics", "page_bbox"),
        _nested(record, "notes", "question_region"),
        _nested(record, "notes", "question_bbox"),
        _nested(record, "notes", "crop_diagnostics", "page_bbox"),
    ):
        bbox = _bbox_tuple(value)
        if bbox is not None:
            return bbox
    regions = _nested(record, "crop_diagnostics", "regions") or _nested(record, "notes", "crop_diagnostics", "regions")
    if isinstance(regions, list):
        boxes = [_bbox_tuple(region.get("final_crop_bbox") or region.get("bbox")) for region in regions if isinstance(region, dict)]
        boxes = [box for box in boxes if box is not None]
        if boxes:
            return (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
    return None


def _bbox_from_payload(payload: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ("bbox", "image_bbox", "asset_bbox", "crop_bbox", "final_crop_bbox"):
        bbox = _bbox_tuple(payload.get(key))
        if bbox is not None:
            return bbox
    return None


def _bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            try:
                return (float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"]))
            except (TypeError, ValueError):
                return None
        if "bbox" in value:
            return _bbox_tuple(value.get("bbox"))
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except (TypeError, ValueError):
            return None
    return None


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0:
        return 0.0
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    if a_area <= 0:
        return 0.0
    return intersection / a_area


def _page_from_payload(payload: dict[str, Any]) -> int | None:
    for key in ("page_number", "page", "source_page", "page_index"):
        page = _int_value(payload.get(key))
        if page is not None:
            return page + 1 if key == "page_index" else page
    return None


def _page_from_record(record: dict[str, Any]) -> int | None:
    for key in ("page_number", "source_page", "question_start_page", "start_page"):
        page = _int_value(record.get(key))
        if page is not None:
            return page
    page_refs = record.get("page_refs")
    if isinstance(page_refs, dict):
        for key in ("question", "questions", "page_numbers"):
            page = _first_int(page_refs.get(key))
            if page is not None:
                return page
    return _first_int(record.get("page_numbers"))


def _vertical_anchor_from_payload(payload: dict[str, Any]) -> float | None:
    for key in ("vertical_anchor", "anchor_y", "y", "y0"):
        value = _float_value(payload.get(key))
        if value is not None:
            return value
    bbox = _bbox_from_payload(payload)
    if bbox is not None:
        return (bbox[1] + bbox[3]) / 2
    return None


def _vertical_anchor_from_record(record: dict[str, Any]) -> float | None:
    for value in (
        record.get("vertical_anchor"),
        record.get("anchor_y"),
        _nested(record, "question_anchor", "y0"),
        _nested(record, "question_anchor", "y"),
        _nested(record, "notes", "question_anchor", "y0"),
        _nested(record, "notes", "question_anchor", "y"),
    ):
        anchor = _float_value(value)
        if anchor is not None:
            return anchor
    bbox = _question_bbox(record)
    if bbox is not None:
        return (bbox[1] + bbox[3]) / 2
    return None


def _paper_from_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("paper_id") or payload.get("paper") or payload.get("canonical_paper_id") or "").strip()


def _question_number_from_payload(payload: dict[str, Any]) -> str:
    for key in ("question_number", "question", "anchor_question_number"):
        value = _normalized_question_number(payload.get(key))
        if value:
            return value
    text = str(payload.get("text") or payload.get("anchor_text") or "")
    return _normalized_question_number(text)


def _normalized_question_number(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"\d{1,2}", text)
    if not match:
        return ""
    return str(int(match.group(0)))


def _question_number_from_question_id(question_id: str) -> str:
    match = re.search(r"(?:^|[_/\-])q(?P<number>\d{1,2})(?:[_\-.]|$)", str(question_id or ""), re.IGNORECASE)
    if not match:
        return ""
    return str(int(match.group("number")))


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_int(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _int_value(item)
            if parsed is not None:
                return parsed
        return None
    return _int_value(value)


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_image_binding_repair_report(
    *,
    initial_orphan_count: int,
    final_orphan_count: int,
    reached_threshold: bool,
    repair_log: list[dict[str, Any]],
    final_evaluation: AlignmentEvaluation,
    max_iterations: int,
) -> dict[str, Any]:
    orphan_actions = [
        action
        for action in repair_log
        if action.get("failure_type") == FAILURE_ORPHAN_IMAGE
        and action.get("strategy") != "restore_successful_image_binding"
    ]
    matched_actions = [action for action in orphan_actions if action.get("after", {}).get("matched") is True]
    unmatched_actions = [action for action in orphan_actions if action.get("after", {}).get("matched") is False]
    method_counts = Counter(
        str(action.get("after", {}).get("match_method") or "unknown")
        for action in matched_actions
    )
    failure_reasons = Counter(
        str(action.get("after", {}).get("reason") or action.get("before", {}).get("reason") or "unknown")
        for action in unmatched_actions
    )
    for failure in final_evaluation.failures:
        if failure.failure_type == FAILURE_ORPHAN_IMAGE:
            failure_reasons[failure.reason or "unresolved orphan image"] += 1
    primary_method = method_counts.most_common(1)[0][0] if method_counts else None
    success_rate = 1.0 if initial_orphan_count == 0 else len(matched_actions) / initial_orphan_count
    return {
        "schema_name": IMAGE_BINDING_REPAIR_SCHEMA_NAME,
        "schema_version": IMAGE_BINDING_REPAIR_SCHEMA_VERSION,
        "alignment_status": "PASS" if reached_threshold else "FAIL",
        "initial_orphan_count": initial_orphan_count,
        "final_orphan_count": final_orphan_count,
        "rebinding_success_count": len(matched_actions),
        "rebinding_success_rate": round(success_rate, 6),
        "primary_resolution_method_used": primary_method,
        "resolution_method_distribution": dict(sorted(method_counts.items())),
        "failure_reasons_distribution": dict(sorted(failure_reasons.items())),
        "retry": {
            "max_iterations": max_iterations,
            "retry_queue_final_count": final_orphan_count,
            "unassigned_pool": UNASSIGNED_ASSET_POOL_NAME,
            "persistent_orphan_state_allowed": False,
        },
    }


def _set_slot_path(record: dict[str, Any], image_kind: str, path: str) -> None:
    canonical_field, primary_field, list_field = image_slot_fields(image_kind)
    record[canonical_field] = path
    record[primary_field] = path
    record[list_field] = [path] if path else []


def _referenced_image_count(question_bank: dict[str, Any], expected_image_kinds: tuple[str, ...]) -> int:
    count = 0
    for record in _records(question_bank):
        for kind in expected_image_kinds:
            if mapping_from_record(record, kind).path:
                count += 1
    return count


def _extraction_mode_for_actions(repair_actions: tuple[dict[str, Any], ...]) -> str:
    modes = {str(action.get("after", {}).get("next_extraction_mode") or "") for action in repair_actions}
    modes.discard("")
    if not modes:
        return "standard"
    if "relaxed_page_segmentation" in modes:
        return "relaxed_page_segmentation"
    if "expanded_crop_reextract" in modes:
        return "expanded_crop_reextract"
    if "question_boundary_reanchor" in modes:
        return "question_boundary_reanchor"
    if "figure_detection_rescan" in modes:
        return "figure_detection_rescan"
    return sorted(modes)[0]


def _records(question_bank: dict[str, Any]) -> list[dict[str, Any]]:
    questions = question_bank.get("questions")
    if not isinstance(questions, list):
        return []
    return [record for record in questions if isinstance(record, dict)]


def _infer_artifact_root(path: Path) -> Path:
    if path.parent.name == "json":
        return path.parent.parent
    return path.parent
