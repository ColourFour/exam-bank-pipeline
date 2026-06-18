from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Iterable

from .asset_manifest import MARK_SCHEME_IMAGE_KIND, QUESTION_IMAGE_KIND, asset_id_for_record
from .core.paper_identity import IdentityError, build_paper_id


ALIGNMENT_METRICS_SCHEMA_NAME = "exam_bank.image_alignment_metrics"
ALIGNMENT_METRICS_SCHEMA_VERSION = 1

FAILURE_MISSING_IMAGE = "missing_image"
FAILURE_ORPHAN_IMAGE = "orphan_image"
FAILURE_MISALIGNED_IMAGE = "misaligned_image"
FAILURE_WEAK_CROP = "weak_crop"
FAILURE_DUPLICATE_MAPPING = "duplicate_mapping"
FAILURE_LEGACY_SEGMENTATION = "legacy_segmentation_failure"

FAILURE_TYPES = (
    FAILURE_MISSING_IMAGE,
    FAILURE_ORPHAN_IMAGE,
    FAILURE_MISALIGNED_IMAGE,
    FAILURE_WEAK_CROP,
    FAILURE_DUPLICATE_MAPPING,
    FAILURE_LEGACY_SEGMENTATION,
)

IMAGE_KINDS = (QUESTION_IMAGE_KIND, MARK_SCHEME_IMAGE_KIND)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
CANONICAL_IMAGE_ROOTS = {"pm1", "pm3", "stats", "mechanics", "p1", "p3", "m1", "s1"}

_QUESTION_TOKEN_RE = re.compile(r"(?:^|[_/\-])q(?P<number>\d{1,2})(?:[_\-.]|$)", re.IGNORECASE)
_CANONICAL_ASSET_RE = re.compile(
    r"^(?P<subject>pm1|pm3|stats|mechanics)_(?P<year>\d{4})_"
    r"(?P<session>[msw]\d{2})_(?P<component>\d{2})_"
    r"(?P<paper_type>qp|ms)_q(?P<question>\d{2})_"
    r"(?P<asset_type>question|markscheme)(?:_v\d+)?\.(?:png|jpg|jpeg|webp|tif|tiff)$",
    re.IGNORECASE,
)
_LEGACY_PAPER_PART_RE = re.compile(r"^\d{2}(?:spring|summer|autumn|winter)\d{2}$", re.IGNORECASE)
_PLACEHOLDER_FILENAMES = {"unknown.png", "placeholder.png", "missing.png", "image_missing.png"}


@dataclass(frozen=True)
class AlignmentMapping:
    slot_id: str
    image_kind: str
    question_id: str
    paper_id: str
    question_number: str
    path: str
    field: str
    asset_id: str


@dataclass(frozen=True)
class AlignmentFailure:
    failure_type: str
    image_kind: str
    question_id: str = ""
    paper_id: str = ""
    question_number: str = ""
    path: str = ""
    field: str = ""
    reason: str = ""
    asset_id: str = ""
    details: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentEvaluation:
    schema_name: str
    schema_version: int
    alignment_score: float
    correctly_mapped_images: int
    expected_images: int
    expected_image_count_by_kind: dict[str, int]
    failure_distribution: dict[str, int]
    failures: tuple[AlignmentFailure, ...]
    correct_mappings: tuple[AlignmentMapping, ...]
    unresolved_assets_count: int
    orphan_assets_count: int

    def to_dict(self, *, include_failures: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "alignment_score": self.alignment_score,
            "correctly_mapped_images": self.correctly_mapped_images,
            "expected_images": self.expected_images,
            "expected_image_count_by_kind": dict(self.expected_image_count_by_kind),
            "failure_distribution": dict(self.failure_distribution),
            "unresolved_assets_count": self.unresolved_assets_count,
            "orphan_assets_count": self.orphan_assets_count,
            "correct_mapping_count": len(self.correct_mappings),
        }
        if include_failures:
            payload["failures"] = [asdict(failure) for failure in self.failures]
        return payload


def _evaluate_image_alignment_uncached(
    question_bank: dict[str, Any] | list[dict[str, Any]],
    *,
    artifact_root: str | Path,
    expected_image_kinds: Iterable[str] = IMAGE_KINDS,
    scan_orphans: bool = True,
) -> AlignmentEvaluation:
    records = _records_from_question_bank(question_bank)
    root = Path(artifact_root)
    kinds = tuple(kind for kind in expected_image_kinds if kind in IMAGE_KINDS)
    slots = [_slot_for_record(record, kind) for record in records for kind in kinds]
    path_to_slots: dict[str, list[AlignmentMapping]] = defaultdict(list)
    for slot in slots:
        if slot.path:
            path_to_slots[_normalized_artifact_path(slot.path)].append(slot)

    duplicate_losers = _duplicate_loser_slot_ids(path_to_slots)
    failures: list[AlignmentFailure] = []
    correct_mappings: list[AlignmentMapping] = []
    expected_by_kind = Counter(slot.image_kind for slot in slots)
    referenced_paths: set[str] = set()

    for slot in slots:
        if slot.path:
            referenced_paths.add(_normalized_artifact_path(slot.path))
        failure = _classify_slot_failure(slot, root=root, duplicate_losers=duplicate_losers)
        if failure is None:
            correct_mappings.append(slot)
        else:
            failures.append(failure)

    orphan_failures: list[AlignmentFailure] = []
    if scan_orphans:
        orphan_failures = _orphan_asset_failures(root, referenced_paths=referenced_paths)
        failures.extend(orphan_failures)

    slot_failure_count = sum(1 for failure in failures if failure.failure_type != FAILURE_ORPHAN_IMAGE)
    expected_images = len(slots)
    correctly_mapped = max(0, expected_images - slot_failure_count)
    score = 1.0 if expected_images == 0 else correctly_mapped / expected_images
    distribution = Counter(failure.failure_type for failure in failures)
    return AlignmentEvaluation(
        schema_name=ALIGNMENT_METRICS_SCHEMA_NAME,
        schema_version=ALIGNMENT_METRICS_SCHEMA_VERSION,
        alignment_score=round(score, 6),
        correctly_mapped_images=correctly_mapped,
        expected_images=expected_images,
        expected_image_count_by_kind={kind: expected_by_kind.get(kind, 0) for kind in kinds},
        failure_distribution={failure_type: distribution.get(failure_type, 0) for failure_type in FAILURE_TYPES},
        failures=tuple(failures),
        correct_mappings=tuple(correct_mappings),
        unresolved_assets_count=len(failures),
        orphan_assets_count=len(orphan_failures),
    )


def classify_alignment_failures(
    question_bank: dict[str, Any] | list[dict[str, Any]],
    *,
    artifact_root: str | Path,
    expected_image_kinds: Iterable[str] = IMAGE_KINDS,
    scan_orphans: bool = True,
) -> tuple[AlignmentFailure, ...]:
    return evaluate_image_alignment(
        question_bank,
        artifact_root=artifact_root,
        expected_image_kinds=expected_image_kinds,
        scan_orphans=scan_orphans,
    ).failures


def image_slot_fields(image_kind: str) -> tuple[str, str, str]:
    if image_kind == QUESTION_IMAGE_KIND:
        return "canonical_question_artifact", "question_image_path", "question_image_paths"
    if image_kind == MARK_SCHEME_IMAGE_KIND:
        return "canonical_mark_scheme_artifact", "mark_scheme_image_path", "mark_scheme_image_paths"
    raise ValueError(f"Unsupported image kind: {image_kind}")


def mapping_from_record(record: dict[str, Any], image_kind: str) -> AlignmentMapping:
    return _slot_for_record(record, image_kind)


def parse_asset_path_metadata(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    name = path.name
    metadata: dict[str, Any] = {}
    canonical_match = _CANONICAL_ASSET_RE.match(name)
    if canonical_match:
        groups = canonical_match.groupdict()
        metadata.update(groups)
        metadata["question_number"] = str(int(groups["question"]))
        metadata["image_kind"] = QUESTION_IMAGE_KIND if groups["asset_type"].lower() == "question" else MARK_SCHEME_IMAGE_KIND
        try:
            metadata["paper_id"] = build_paper_id("9709", groups["session"].lower(), groups["component"])
        except IdentityError:
            metadata["paper_id"] = ""
        return metadata

    question_match = _QUESTION_TOKEN_RE.search(str(path).replace("\\", "/"))
    if question_match:
        metadata["question_number"] = str(int(question_match.group("number")))

    parts = path.parts
    for part in parts:
        if _LEGACY_PAPER_PART_RE.match(part):
            metadata["paper_id"] = part.lower()
            break

    lowered_parts = {part.lower() for part in parts}
    lowered_name = name.lower()
    if "questions" in lowered_parts or lowered_name.endswith("_question.png"):
        metadata["image_kind"] = QUESTION_IMAGE_KIND
    elif "mark_scheme" in lowered_parts or "markscheme" in lowered_name or lowered_name.endswith("_markscheme.png"):
        metadata["image_kind"] = MARK_SCHEME_IMAGE_KIND
    return metadata


def _records_from_question_bank(question_bank: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(question_bank, dict):
        questions = question_bank.get("questions")
        if isinstance(questions, list):
            return [record for record in questions if isinstance(record, dict)]
        return []
    return [record for record in question_bank if isinstance(record, dict)]


def _slot_for_record(record: dict[str, Any], image_kind: str) -> AlignmentMapping:
    field, path = _primary_path(record, image_kind)
    question_id = _clean_text(record.get("question_id"))
    paper_id = _clean_text(record.get("paper") or record.get("canonical_paper_id"))
    question_number = _clean_text(record.get("question_number")) or _question_number_from_question_id(question_id)
    asset_id = asset_id_for_record(image_kind, record, path) if path else ""
    return AlignmentMapping(
        slot_id=f"{question_id or '<missing_question_id>'}:{image_kind}",
        image_kind=image_kind,
        question_id=question_id,
        paper_id=paper_id,
        question_number=_normalize_question_number(question_number),
        path=path,
        field=field,
        asset_id=asset_id,
    )


def _primary_path(record: dict[str, Any], image_kind: str) -> tuple[str, str]:
    canonical_field, primary_field, list_field = image_slot_fields(image_kind)
    for field in (canonical_field, primary_field):
        value = _clean_text(record.get(field))
        if value:
            return field, value
    paths = _list_paths(record.get(list_field))
    if paths:
        return f"{list_field}[0]", paths[0]
    return primary_field, ""


def _classify_slot_failure(
    slot: AlignmentMapping,
    *,
    root: Path,
    duplicate_losers: set[str],
) -> AlignmentFailure | None:
    if slot.slot_id in duplicate_losers:
        return _failure(
            FAILURE_DUPLICATE_MAPPING,
            slot,
            reason="path is mapped to more than one image slot",
            details={"deterministic_asset_id": slot.asset_id},
        )
    if not slot.path:
        return _missing_or_legacy_failure(slot, reason="image path is empty")
    if _is_placeholder_path(slot.path):
        return _failure(FAILURE_MISSING_IMAGE, slot, reason="placeholder image path is not allowed")
    if _resolve_artifact_path(slot.path, root) is None:
        return _missing_or_legacy_failure(slot, reason="image file does not exist")

    metadata = parse_asset_path_metadata(slot.path)
    metadata_kind = str(metadata.get("image_kind") or "")
    if metadata_kind and metadata_kind != slot.image_kind:
        return _failure(
            FAILURE_MISALIGNED_IMAGE,
            slot,
            reason="image kind in path does not match expected slot kind",
            details={"path_image_kind": metadata_kind},
        )

    path_question = _clean_text(metadata.get("question_number"))
    if path_question and slot.question_number and path_question != slot.question_number:
        return _failure(
            FAILURE_MISALIGNED_IMAGE,
            slot,
            reason="question number in path does not match record",
            details={"path_question_number": path_question},
        )

    path_paper = _clean_text(metadata.get("paper_id"))
    if path_paper and slot.paper_id and path_paper != slot.paper_id:
        return _failure(
            FAILURE_MISALIGNED_IMAGE,
            slot,
            reason="paper id in path does not match record",
            details={"path_paper_id": path_paper},
        )

    crop_failure = _weak_crop_reason(slot)
    if crop_failure:
        return _failure(FAILURE_WEAK_CROP, slot, reason=crop_failure)

    mapping_failure = _mapping_failure_reason(slot)
    if mapping_failure:
        return _failure(FAILURE_MISALIGNED_IMAGE, slot, reason=mapping_failure)

    return None


def _missing_or_legacy_failure(slot: AlignmentMapping, *, reason: str) -> AlignmentFailure:
    legacy_reason = _legacy_segmentation_reason(slot)
    if legacy_reason:
        return _failure(FAILURE_LEGACY_SEGMENTATION, slot, reason=legacy_reason)
    return _failure(FAILURE_MISSING_IMAGE, slot, reason=reason)


def _weak_crop_reason(slot: AlignmentMapping) -> str:
    record = _slot_record_cache.get(slot.slot_id)
    if not record:
        return ""
    if slot.image_kind == QUESTION_IMAGE_KIND:
        confidence = _note_or_top(record, "question_crop_confidence")
        if str(confidence or "").lower() and str(confidence or "").lower() != "high":
            return f"question crop confidence is {confidence}"
        if _bool_value(_note_or_top(record, "crop_uncertain")):
            return "question crop is marked uncertain"
    else:
        confidence = _note_or_top(record, "mark_scheme_crop_confidence")
        if str(confidence or "").lower() and str(confidence or "").lower() != "high":
            return f"mark-scheme crop confidence is {confidence}"
    flags = set(_list_paths(_note_or_top(record, "review_flags")))
    if slot.image_kind == QUESTION_IMAGE_KIND and {"low_confidence_question_crop", "crop_uncertain"} & flags:
        return "question crop review flags indicate weak crop"
    if slot.image_kind == MARK_SCHEME_IMAGE_KIND and {"markscheme_image_uncertain"} & flags:
        return "mark-scheme review flags indicate weak crop"
    return ""


def _mapping_failure_reason(slot: AlignmentMapping) -> str:
    if slot.image_kind != MARK_SCHEME_IMAGE_KIND:
        return ""
    record = _slot_record_cache.get(slot.slot_id)
    if not record:
        return ""
    mapping_status = str(_note_or_top(record, "mapping_status") or "").lower()
    if mapping_status and mapping_status != "pass":
        return f"mark-scheme mapping status is {mapping_status}"
    return ""


def _legacy_segmentation_reason(slot: AlignmentMapping) -> str:
    if slot.image_kind != MARK_SCHEME_IMAGE_KIND:
        return ""
    record = _slot_record_cache.get(slot.slot_id)
    if not record:
        return ""
    reason = str(_note_or_top(record, "missing_mark_scheme_reason") or _note_or_top(record, "mapping_failure_reason") or "").lower()
    flags = set(_list_paths(_note_or_top(record, "review_flags")))
    if "segmentation_failure" in reason or "markscheme_segmentation_failure" in flags:
        return "mark-scheme segmentation failed in legacy layout"
    return ""


def _failure(failure_type: str, slot: AlignmentMapping, *, reason: str, details: dict[str, Any] | None = None) -> AlignmentFailure:
    return AlignmentFailure(
        failure_type=failure_type,
        image_kind=slot.image_kind,
        question_id=slot.question_id,
        paper_id=slot.paper_id,
        question_number=slot.question_number,
        path=slot.path,
        field=slot.field,
        reason=reason,
        asset_id=slot.asset_id,
        details=details or {},
    )


def _duplicate_loser_slot_ids(path_to_slots: dict[str, list[AlignmentMapping]]) -> set[str]:
    losers: set[str] = set()
    for path, slots in path_to_slots.items():
        unique_slots = {slot.slot_id: slot for slot in slots}
        if len(unique_slots) <= 1:
            continue
        winner = _duplicate_winner(path, list(unique_slots.values()))
        for slot in unique_slots.values():
            if slot.slot_id != winner.slot_id:
                losers.add(slot.slot_id)
    return losers


def _duplicate_winner(path: str, slots: list[AlignmentMapping]) -> AlignmentMapping:
    metadata = parse_asset_path_metadata(path)
    path_question = _clean_text(metadata.get("question_number"))
    path_paper = _clean_text(metadata.get("paper_id"))
    path_kind = _clean_text(metadata.get("image_kind"))

    def sort_key(slot: AlignmentMapping) -> tuple[int, str, str]:
        exact = (
            path_question
            and path_question == slot.question_number
            and (not path_paper or path_paper == slot.paper_id)
            and (not path_kind or path_kind == slot.image_kind)
        )
        return (0 if exact else 1, slot.asset_id, slot.slot_id)

    return sorted(slots, key=sort_key)[0]


def _orphan_asset_failures(root: Path, *, referenced_paths: set[str]) -> list[AlignmentFailure]:
    if not root.exists():
        return []
    failures: list[AlignmentFailure] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0].lower() not in CANONICAL_IMAGE_ROOTS:
            continue
        relative_text = str(relative)
        if _normalized_artifact_path(relative_text) in referenced_paths:
            continue
        metadata = parse_asset_path_metadata(relative_text)
        image_kind = str(metadata.get("image_kind") or "")
        failures.append(
            AlignmentFailure(
                failure_type=FAILURE_ORPHAN_IMAGE,
                image_kind=image_kind,
                question_number=_clean_text(metadata.get("question_number")),
                paper_id=_clean_text(metadata.get("paper_id")),
                path=relative_text,
                reason="canonical image file is not referenced by the question bank",
                details={"absolute_path": str(path)},
            )
        )
    return failures


def _resolve_artifact_path(path_value: str, root: Path) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [root / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _is_placeholder_path(path_value: str) -> bool:
    name = Path(path_value).name.lower()
    return name in _PLACEHOLDER_FILENAMES or name.startswith("unknown.")


def _normalized_artifact_path(path_value: str) -> str:
    return str(PureWindowsPath(str(path_value).replace("\\", "/"))).replace("\\", "/").strip()


def _question_number_from_question_id(question_id: str) -> str:
    match = _QUESTION_TOKEN_RE.search(question_id)
    if not match:
        return ""
    return str(int(match.group("number")))


def _normalize_question_number(value: str) -> str:
    match = re.search(r"\d{1,2}", str(value or ""))
    if not match:
        return ""
    return str(int(match.group(0)))


def _note_or_top(record: dict[str, Any], field: str) -> Any:
    if field in record and record[field] not in (None, ""):
        return record[field]
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(field)
    return None


def _list_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


# Slot-level checks need access to record notes while keeping failure payloads compact.
_slot_record_cache: dict[str, dict[str, Any]] = {}


def _cache_slot_records(records: Iterable[dict[str, Any]], kinds: tuple[str, ...]) -> None:
    _slot_record_cache.clear()
    for record in records:
        for kind in kinds:
            slot = _slot_for_record(record, kind)
            _slot_record_cache[slot.slot_id] = record


def evaluate_image_alignment(
    question_bank: dict[str, Any] | list[dict[str, Any]],
    *,
    artifact_root: str | Path,
    expected_image_kinds: Iterable[str] = IMAGE_KINDS,
    scan_orphans: bool = True,
) -> AlignmentEvaluation:
    records = _records_from_question_bank(question_bank)
    kinds = tuple(kind for kind in expected_image_kinds if kind in IMAGE_KINDS)
    _cache_slot_records(records, kinds)
    return _evaluate_image_alignment_uncached(
        records,
        artifact_root=artifact_root,
        expected_image_kinds=kinds,
        scan_orphans=scan_orphans,
    )
