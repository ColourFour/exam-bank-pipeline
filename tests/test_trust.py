from exam_bank.trust import (
    CurationStatus,
    CropConfidence,
    MappingStatus,
    QuestionTextRole,
    QuestionTextTrust,
    ScopeQualityStatus,
    TextFidelityStatus,
    TopicTrustStatus,
    ValidationStatus,
    assess_text_fidelity,
    derive_question_text_semantics,
    derive_scope_quality_status,
    derive_text_only_status,
    derive_topic_trust_status,
    derive_visual_curation_status,
    final_review_reasons,
    refine_validation_status,
    text_source_profile,
    visual_reason_flags,
)


def test_status_vocabularies_expose_expected_export_values() -> None:
    assert ValidationStatus.VALUES == {"pass", "review", "fail"}
    assert ScopeQualityStatus.VALUES == {"clean", "review", "fail"}
    assert TextFidelityStatus.VALUES == {"clean", "degraded", "unusable"}
    assert QuestionTextRole.VALUES == {"search_hint", "readable_text", "untrusted_math_text", "missing"}
    assert QuestionTextTrust.VALUES == {"high", "medium", "low", "unusable"}
    assert CurationStatus.VALUES == {"ready", "review", "fail"}
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


def _visual_flags(text: str, *, review_flags: list[str] | None = None, extraction_flags: list[str] | None = None) -> list[str]:
    return visual_reason_flags(
        question_text=text,
        extraction_quality_flags=extraction_flags or [],
        review_flags=review_flags or [],
        question_structure_detected={},
        text_source_profile="native_pdf",
    )


def test_visual_flags_catch_reordered_graph_equation_text() -> None:
    flags = _visual_flags("Sketch the graph of y x 3 6 = -.")
    role, trust, visual_required = derive_question_text_semantics(
        question_text="Sketch the graph of y x 3 6 = -.",
        text_fidelity_status="degraded",
        visual_reason_flags=flags,
    )

    assert "contains_graph_or_diagram_prompt" in flags
    assert "contains_math_text_corruption" in flags
    assert role == "untrusted_math_text"
    assert trust == "low"
    assert visual_required is True


def test_visual_flags_catch_scrambled_inequality_text() -> None:
    flags = _visual_flags("5 3 x x - - 1 3 6")
    assert "contains_math_text_corruption" in flags
    assert "text_order_unreliable" in flags


def test_visual_flags_catch_argand_region_question() -> None:
    flags = _visual_flags("On an Argand diagram shade the region satisfying |z - 2i| < 3 and arg(z) > pi/4.")
    assert "contains_complex_number_notation" in flags
    assert "contains_inequality_or_region_prompt" in flags


def test_visual_flags_catch_vector_line_question() -> None:
    flags = _visual_flags("The line l has vector equation r = 2i - j + 3k + lambda(i + j - k).")
    assert "contains_vector_notation" in flags
    assert "contains_equation_layout" in flags


def test_visual_flags_catch_integral_fraction_layout_question() -> None:
    flags = _visual_flags("Use the substitution u = 1 + x^2 to evaluate the integral from 0 to 1 of x / (1 + x^2) dx.")
    assert "contains_fraction_or_integral_layout" in flags
    assert "contains_equation_layout" in flags


def test_clean_natural_language_text_can_be_readable() -> None:
    text = "A committee has 5 members chosen from 8 people. Find the number of possible committees."
    flags = _visual_flags(text)
    role, trust, visual_required = derive_question_text_semantics(
        question_text=text,
        text_fidelity_status="clean",
        visual_reason_flags=flags,
    )

    assert flags == []
    assert role == "readable_text"
    assert trust == "high"
    assert visual_required is False


def test_page_furniture_and_barcode_garbage_demote_text() -> None:
    text = "DO NOT WRITE IN THIS MARGIN\n123456 789012 345678\n\ufffd\ufffd\ufffd\nBLANK PAGE"
    flags = _visual_flags(text)
    status, fidelity_flags = assess_text_fidelity(
        question_text=text,
        extraction_quality_flags=[],
        review_flags=[],
        validation_flags=[],
        question_structure_detected={},
        mapping_failure_reason="",
        text_source_profile="native_pdf",
    )
    role, trust, visual_required = derive_question_text_semantics(
        question_text=text,
        text_fidelity_status=status,
        visual_reason_flags=flags,
    )

    assert "contains_page_furniture" in flags
    assert "contains_pdf_control_garbage" in flags
    assert status == "degraded"
    assert "pdf_control_garbage_detected" in fidelity_flags
    assert role == "untrusted_math_text"
    assert trust == "low"
    assert visual_required is True


def test_good_image_bad_text_can_be_visual_ready_but_text_only_fail() -> None:
    visual_status = derive_visual_curation_status(
        validation_status="pass",
        scope_quality_status="clean",
        question_image_path="p1/12spring24/questions/q01.png",
        question_crop_confidence="high",
        mark_scheme_image_path="p1/12spring24/mark_scheme/q01.png",
        mark_scheme_crop_confidence="high",
    )
    text_status = derive_text_only_status(
        validation_status="pass",
        scope_quality_status="clean",
        question_text_role="untrusted_math_text",
        question_text_trust="low",
    )

    assert visual_status == "ready"
    assert text_status == "fail"


def test_repeated_trig_corruption_stays_text_only_fail_after_cleanup() -> None:
    text = "Show that sin cos θ θ + - 2 cos2 sin θ θ ≡ 5 cos^{2}4θ -4."
    flags = _visual_flags(
        text,
        extraction_flags=["flattened_display_math", "math_corruption_suspected", "likely_needs_visual_review"],
    )
    status, fidelity_flags = assess_text_fidelity(
        question_text=text,
        extraction_quality_flags=["flattened_display_math", "math_corruption_suspected", "likely_needs_visual_review"],
        review_flags=["weak_question_text"],
        validation_flags=[],
        question_structure_detected={},
        mapping_failure_reason="",
        text_source_profile="native_pdf",
    )
    role, trust, visual_required = derive_question_text_semantics(
        question_text=text,
        text_fidelity_status=status,
        visual_reason_flags=flags,
    )
    text_status = derive_text_only_status(
        validation_status="pass",
        scope_quality_status="clean",
        question_text_role=role,
        question_text_trust=trust,
    )

    assert "math_text_corruption_detected" in fidelity_flags
    assert role == "untrusted_math_text"
    assert visual_required is True
    assert text_status == "fail"
