from __future__ import annotations

import pytest

from exam_bank.text_normalization import normalize_advisory_question_text
from exam_bank.text_normalized_candidate import (
    DETERMINISTIC_CREATED_AT,
    NormalizedTextCandidateContractError,
    build_advisory_normalized_text_candidate,
    source_text_hash,
    validate_advisory_normalized_text_candidate,
)


def make_candidate() -> dict:
    raw_text = "4 Find y = 3x^{3} ln x^{4}, for x20."
    normalized = normalize_advisory_question_text(raw_text)
    return build_advisory_normalized_text_candidate(
        record_id="35summer25_q04",
        source_text_kind="selected_text",
        source_text=raw_text,
        question_text_normalized=normalized.normalized_text,
        normalization_flags=normalized.flags,
        normalization_confidence=normalized.confidence,
        normalization_warnings=normalized.warnings,
        provenance={
            "source_report": "output/reports/text_fidelity_fixture_baseline_normalized.json",
            "normalizer": "exam_bank.text_normalization.normalize_advisory_question_text",
            "raw_text_preserved": True,
        },
    )


def test_build_candidate_defaults_to_advisory_report_only_contract() -> None:
    candidate = make_candidate()

    assert candidate["candidate_id"].startswith("normcand_")
    assert candidate["record_id"] == "35summer25_q04"
    assert candidate["source_text_kind"] == "selected_text"
    assert candidate["source_text_hash"] == source_text_hash("4 Find y = 3x^{3} ln x^{4}, for x20.")
    assert "for x > 0" in candidate["question_text_normalized"]
    assert candidate["normalization_is_advisory"] is True
    assert candidate["display_allowed"] is False
    assert candidate["export_allowed"] is False
    assert candidate["created_at"] == DETERMINISTIC_CREATED_AT
    assert candidate["provenance"]["raw_text_preserved"] is True
    assert isinstance(candidate["normalization_warnings"], list)


def test_candidate_id_is_deterministic_for_same_contract_inputs() -> None:
    first = make_candidate()
    second = make_candidate()

    assert first == second


@pytest.mark.parametrize(
    "field",
    [
        "normalization_is_advisory",
        "provenance",
        "normalization_warnings",
        "source_text_hash",
        "display_allowed",
        "export_allowed",
    ],
)
def test_candidate_records_cannot_omit_required_contract_fields(field: str) -> None:
    candidate = make_candidate()
    candidate.pop(field)

    with pytest.raises(NormalizedTextCandidateContractError, match="Missing required candidate fields"):
        validate_advisory_normalized_text_candidate(candidate)


def test_candidate_must_be_explicitly_advisory_with_non_empty_provenance() -> None:
    candidate = make_candidate()
    candidate["normalization_is_advisory"] = False

    with pytest.raises(NormalizedTextCandidateContractError, match="must be true"):
        validate_advisory_normalized_text_candidate(candidate)

    candidate = make_candidate()
    candidate["provenance"] = {}
    with pytest.raises(NormalizedTextCandidateContractError, match="provenance must be a non-empty object"):
        validate_advisory_normalized_text_candidate(candidate)


def test_candidate_must_reference_an_allowed_raw_source_kind() -> None:
    candidate = make_candidate()
    candidate["source_text_kind"] = "question_text"

    with pytest.raises(NormalizedTextCandidateContractError, match="Unsupported source_text_kind"):
        validate_advisory_normalized_text_candidate(candidate)
