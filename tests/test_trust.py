from exam_bank.trust import (
    CropConfidence,
    MappingStatus,
    ScopeQualityStatus,
    TextFidelityStatus,
    TopicTrustStatus,
    ValidationStatus,
    assess_text_fidelity,
    derive_scope_quality_status,
    derive_topic_trust_status,
    final_review_reasons,
    refine_validation_status,
    text_source_profile,
)


def test_status_vocabularies_expose_expected_export_values() -> None:
    assert ValidationStatus.VALUES == {"pass", "review", "fail"}
    assert ScopeQualityStatus.VALUES == {"clean", "review", "fail"}
    assert TextFidelityStatus.VALUES == {"clean", "degraded", "unusable"}
    assert TopicTrustStatus.VALUES == {"normal", "degraded_text", "review_required"}
    assert MappingStatus.VALUES == {"pass", "fail"}
    assert CropConfidence.VALUES == {"low", "medium", "high"}


def test_scope_text_and_topic_trust_derivation_preserves_current_semantics() -> None:
    assert derive_scope_quality_status(
        validation_flags=["question_start_uncertain"],
        review_flags=[],
        question_structure_detected={},
    ) == "review"
    assert text_source_profile(["ocr_merged_page"]) == "hybrid"

    fidelity_status, fidelity_flags = assess_text_fidelity(
        question_text="anne",
        extraction_quality_flags=["heavy_math_density"],
        review_flags=["weak_question_text"],
        validation_flags=[],
        question_structure_detected={},
        mapping_failure_reason="",
        text_source_profile="hybrid",
    )
    assert fidelity_status == "degraded"
    assert "hybrid_math_text_requires_review" in fidelity_flags

    assert derive_topic_trust_status(
        topic_confidence="high",
        topic_uncertain=False,
        text_fidelity_status=fidelity_status,
        validation_status="review",
        scope_quality_status="clean",
    ) == "degraded_text"


def test_validation_refinement_and_deepseek_review_reasons_share_status_constants() -> None:
    status, flags = refine_validation_status(
        base_status="review",
        base_validation_flags=[],
        mapping_status="fail",
        mapping_failure_reason="question_subparts_incomplete",
        crop_uncertain=False,
        extraction_quality_flags=[],
        review_flags=[],
        question_structure_detected={},
    )
    assert status == "fail"
    assert flags == ["question_subparts_incomplete"]

    reasons = final_review_reasons(
        model_review_required=True,
        validation_status=status,
        scope_quality_status="clean",
        text_fidelity_status="unusable",
        topic_trust_status="review_required",
        topic_reconciliation_status="mismatch",
        difficulty_reconciliation_status="match",
        local_difficulty_present=True,
    )
    assert reasons == [
        "llm_review_required",
        "validation_status:fail",
        "text_fidelity_status:unusable",
        "topic_trust_status:review_required",
        "topic_reconciliation_status:mismatch",
    ]
