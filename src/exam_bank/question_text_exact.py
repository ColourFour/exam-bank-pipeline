from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

SCHEMA_NAME = "exam_bank.question_text_exact_evaluation"
SCHEMA_VERSION = 1
NORMALIZATION_SCHEMA_NAME = "exam_bank.question_text_exact_normalization"
NORMALIZATION_SCHEMA_VERSION = 1
DEFAULT_MINIMUM_ACCURACY = 0.85


class QuestionTextExactError(ValueError):
    """Raised when an exact-text evaluation input violates the contract."""


def normalize_question_text_exact(text: str) -> str:
    """Apply presentation-only normalization for whole-question comparison.

    This intentionally does not case-fold, remove punctuation, map mathematical
    symbols, tokenize text, or infer missing content. Those operations could
    hide a semantic extraction error.
    """

    if not isinstance(text, str):
        raise QuestionTextExactError("Question text must be a string.")
    value = unicodedata.normalize("NFC", text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", value).strip()


def evaluate_question_text_exact(
    gold_registry: dict[str, Any],
    candidate_question_bank: dict[str, Any],
    *,
    cohort_sample: dict[str, Any] | None = None,
    minimum_accuracy: float = DEFAULT_MINIMUM_ACCURACY,
) -> dict[str, Any]:
    """Evaluate whole-question exact match over a complete fixed cohort.

    The gold registry must contain ``records`` and the candidate question bank
    must contain ``questions``. If no cohort is supplied, the gold registry is
    the cohort. Missing cohort records are reported as coverage failures and
    count as non-matches; duplicate identifiers and malformed inputs are
    rejected.
    """

    minimum_accuracy = _validated_minimum_accuracy(minimum_accuracy)
    gold_rows = _required_rows(gold_registry, "records", "gold registry")
    candidate_rows = _required_rows(candidate_question_bank, "questions", "candidate question bank")
    cohort_rows = (
        _required_rows(cohort_sample, "questions", "cohort sample")
        if cohort_sample is not None
        else gold_rows
    )

    gold_by_id = _index_rows(gold_rows, source="gold registry", require_question_text=True)
    candidate_by_id = _index_rows(candidate_rows, source="candidate question bank")
    cohort_by_id = _index_rows(cohort_rows, source="cohort sample" if cohort_sample is not None else "gold registry")
    cohort_ids = list(cohort_by_id)
    if not cohort_ids:
        raise QuestionTextExactError("Evaluation cohort must contain at least one question.")

    missing_gold_ids = [question_id for question_id in cohort_ids if question_id not in gold_by_id]
    missing_candidate_ids = [question_id for question_id in cohort_ids if question_id not in candidate_by_id]
    invalid_candidate_text_ids = [
        question_id
        for question_id in cohort_ids
        if question_id in candidate_by_id
        and not isinstance(candidate_by_id[question_id].get("question_text"), str)
    ]
    coverage_complete = not (missing_gold_ids or missing_candidate_ids or invalid_candidate_text_ids)

    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    mismatches: list[dict[str, Any]] = []
    exact_match_count = 0

    for question_id in cohort_ids:
        gold = gold_by_id.get(question_id)
        candidate = candidate_by_id.get(question_id)
        cohort = cohort_by_id[question_id]
        family = _paper_family(cohort, candidate, gold)
        family_counts[family]["total"] += 1

        reason: str | None = None
        gold_text: str | None = None
        candidate_text: str | None = None
        normalized_gold: str | None = None
        normalized_candidate: str | None = None

        if gold is None:
            reason = "missing_gold_record"
        else:
            gold_text = gold["question_text"]
            normalized_gold = normalize_question_text_exact(gold_text)

        if candidate is None:
            reason = reason or "missing_candidate_record"
        elif not isinstance(candidate.get("question_text"), str):
            reason = reason or "invalid_candidate_question_text"
        else:
            candidate_text = candidate["question_text"]
            normalized_candidate = normalize_question_text_exact(candidate_text)

        if reason is None and normalized_candidate == normalized_gold:
            exact_match_count += 1
            family_counts[family]["exact_match_count"] += 1
            continue
        if reason is None:
            reason = "text_mismatch"
        mismatches.append(
            {
                "question_id": question_id,
                "paper_family": family,
                "reason": reason,
                "gold_text": gold_text,
                "candidate_text": candidate_text,
                "normalized_gold_text": normalized_gold,
                "normalized_candidate_text": normalized_candidate,
            }
        )

    total = len(cohort_ids)
    overall = _metric(exact_match_count, total, minimum_accuracy)
    by_family = {
        family: _metric(counts["exact_match_count"], counts["total"], minimum_accuracy)
        for family, counts in sorted(family_counts.items())
    }
    passed = coverage_complete and overall["meets_minimum_accuracy"]
    cohort_set = set(cohort_ids)

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "normalization": {
            "schema_name": NORMALIZATION_SCHEMA_NAME,
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "operations": [
                "unicode_nfc",
                "line_endings_to_lf",
                "nonbreaking_spaces_to_space",
                "collapse_and_trim_whitespace",
            ],
            "semantic_transformations": False,
        },
        "minimum_accuracy": minimum_accuracy,
        "cohort": {
            "source": "provided_cohort_sample" if cohort_sample is not None else "gold_registry",
            "question_count": total,
        },
        "coverage": {
            "complete": coverage_complete,
            "gold_record_count": len(gold_by_id),
            "candidate_record_count": len(candidate_by_id),
            "cohort_question_count": total,
            "gold_covered_count": total - len(missing_gold_ids),
            "candidate_covered_count": total - len(missing_candidate_ids) - len(invalid_candidate_text_ids),
            "missing_gold_question_ids": missing_gold_ids,
            "missing_candidate_question_ids": missing_candidate_ids,
            "invalid_candidate_question_text_ids": invalid_candidate_text_ids,
            "gold_ids_outside_cohort": sorted(set(gold_by_id) - cohort_set),
            "candidate_ids_outside_cohort_count": len(set(candidate_by_id) - cohort_set),
        },
        "overall": overall,
        "by_family": by_family,
        "mismatches": mismatches,
        "passed": passed,
    }


def _required_rows(payload: Any, field: str, source: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise QuestionTextExactError(f"{source} must be a JSON object.")
    rows = payload.get(field)
    if not isinstance(rows, list):
        raise QuestionTextExactError(f"{source} must contain a {field!r} list.")
    if not all(isinstance(row, dict) for row in rows):
        raise QuestionTextExactError(f"Every item in {source} {field!r} must be an object.")
    return rows


def _index_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    require_question_text: bool = False,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for position, row in enumerate(rows):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise QuestionTextExactError(
                f"{source} item {position} must have a non-empty string question_id."
            )
        if question_id in indexed:
            duplicate_ids.add(question_id)
        indexed[question_id] = row
        if require_question_text and not isinstance(row.get("question_text"), str):
            raise QuestionTextExactError(
                f"Gold record {question_id!r} must have a string question_text."
            )
        if require_question_text and not row["question_text"].strip():
            raise QuestionTextExactError(
                f"Gold record {question_id!r} must have non-empty question_text."
            )
    if duplicate_ids:
        raise QuestionTextExactError(
            f"Duplicate question_id values in {source}: {sorted(duplicate_ids)}"
        )
    return indexed


def _paper_family(*rows: dict[str, Any] | None) -> str:
    for row in rows:
        if not row:
            continue
        value = row.get("paper_family")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _metric(exact_match_count: int, total: int, minimum_accuracy: float) -> dict[str, Any]:
    accuracy = exact_match_count / total
    return {
        "exact_match_count": exact_match_count,
        "mismatch_count": total - exact_match_count,
        "total": total,
        "accuracy": accuracy,
        "meets_minimum_accuracy": accuracy >= minimum_accuracy,
    }


def _validated_minimum_accuracy(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuestionTextExactError("minimum_accuracy must be a number from 0 to 1.")
    result = float(value)
    if not 0 <= result <= 1:
        raise QuestionTextExactError("minimum_accuracy must be a number from 0 to 1.")
    return result
