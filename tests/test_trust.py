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


def test_topic_trust_downgrades_for_visual_search_hint_even_with_clean_text() -> None:
    assert (
        derive_topic_trust_status(
            topic_confidence="high",
            topic_uncertain=False,
            text_fidelity_status="clean",
            validation_status="pass",
            scope_quality_status="clean",
            question_text_role="search_hint",
            visual_required=True,
        )
        == "degraded_text"
    )


def _visual_flags(text: str, *, review_flags: list[str] | None = None, extraction_flags: list[str] | None = None) -> list[str]:
    return visual_reason_flags(
        question_text=text,
        extraction_quality_flags=extraction_flags or [],
        review_flags=review_flags or [],
        question_structure_detected={},
        text_source_profile="native_pdf",
    )


def _text_only_outcome(text: str) -> tuple[str, str, list[str], list[str]]:
    flags = _visual_flags(text)
    fidelity_status, fidelity_flags = assess_text_fidelity(
        question_text=text,
        extraction_quality_flags=[],
        review_flags=[],
        validation_flags=[],
        question_structure_detected={},
        mapping_failure_reason="",
        text_source_profile="native_pdf",
    )
    role, trust, _visual_required = derive_question_text_semantics(
        question_text=text,
        text_fidelity_status=fidelity_status,
        visual_reason_flags=flags,
    )
    text_status = derive_text_only_status(
        validation_status="pass",
        scope_quality_status="clean",
        question_text_role=role,
        question_text_trust=trust,
    )
    return text_status, fidelity_status, flags, fidelity_flags


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


def test_weak_crop_or_hybrid_flag_alone_does_not_degrade_clean_text() -> None:
    text = "8 A committee has 5 members chosen from 8 people. Find the number of possible committees. [3]"
    review_flags = ["weak_question_text", "low_confidence_question_crop", "ocr_merged_sparse_lower_region"]
    flags = visual_reason_flags(
        question_text=text,
        extraction_quality_flags=[],
        review_flags=review_flags,
        question_structure_detected={},
        text_source_profile="hybrid",
    )
    status, fidelity_flags = assess_text_fidelity(
        question_text=text,
        extraction_quality_flags=[],
        review_flags=review_flags,
        validation_flags=[],
        question_structure_detected={},
        mapping_failure_reason="",
        text_source_profile="hybrid",
    )
    role, trust, visual_required = derive_question_text_semantics(
        question_text=text,
        text_fidelity_status=status,
        visual_reason_flags=flags,
    )

    assert flags == []
    assert fidelity_flags == []
    assert status == "clean"
    assert role == "readable_text"
    assert trust == "high"
    assert visual_required is False
    assert (
        derive_text_only_status(
            validation_status="pass",
            scope_quality_status="clean",
            question_text_role=role,
            question_text_trust=trust,
        )
        == "ready"
    )


def test_merged_hybrid_prose_still_degrades_text_confidence() -> None:
    text = "5 In the diagram, Describefullythetwosingletransformationsofy = f(x) that give the result. [4]"
    flags = visual_reason_flags(
        question_text=text,
        extraction_quality_flags=[],
        review_flags=["weak_question_text", "ocr_merged_sparse_lower_region"],
        question_structure_detected={},
        text_source_profile="hybrid",
    )
    status, fidelity_flags = assess_text_fidelity(
        question_text=text,
        extraction_quality_flags=[],
        review_flags=["weak_question_text", "ocr_merged_sparse_lower_region"],
        validation_flags=[],
        question_structure_detected={},
        mapping_failure_reason="",
        text_source_profile="hybrid",
    )

    assert "ocr_text_sparse_or_merged" in flags
    assert status == "degraded"
    assert "sparse_or_merged_question_text" in fidelity_flags


def test_table_dependent_readable_text_is_review_not_ready() -> None:
    text = (
        "4 The heights of 15 players from each team are given in the table. "
        "Draw a back-to-back stem-and-leaf diagram and find the median. [4]"
    )
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

    assert status == "clean"
    assert fidelity_flags == []
    assert "contains_table_or_data_prompt" in flags
    assert role == "search_hint"
    assert trust == "medium"
    assert visual_required is True
    assert (
        derive_text_only_status(
            validation_status="pass",
            scope_quality_status="clean",
            question_text_role=role,
            question_text_trust=trust,
        )
        == "review"
    )


def test_instantaneously_is_not_treated_as_repeated_trig_corruption() -> None:
    text = (
        "A ball hits the ground and instantaneously loses 8 J of kinetic energy. "
        "Find the greatest height after hitting the ground."
    )
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

    assert "contains_math_text_corruption" not in flags
    assert "text_order_unreliable" not in flags
    assert fidelity_flags == []
    assert role == "readable_text"
    assert trust == "high"
    assert visual_required is False


def test_joined_trig_function_corruption_still_requires_visual_review() -> None:
    flags = _visual_flags("Prove that (sini + coscos_{2} ii)^{2} = 1.")
    assert "contains_math_text_corruption" in flags
    assert "text_order_unreliable" in flags


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


def test_compacted_native_math_tokens_block_ready() -> None:
    examples = [
        "1 (a) Expand@1 -21xA_{2}. [1]",
        "3 Find the term independent of x in @3x + x2_{2}A_{6}. [3]",
        "1 The coefficient of x^{3} in the expansion of@p + p1xA_{4} is 144. [4]",
    ]

    for text in examples:
        text_status, fidelity_status, flags, fidelity_flags = _text_only_outcome(text)
        assert text_status != "ready"
        assert fidelity_status == "degraded"
        assert "contains_native_compacted_math_corruption" in flags
        assert "native_compacted_math_corruption" in fidelity_flags


def test_valid_latex_like_subscript_does_not_trigger_compacted_math_corruption() -> None:
    text = "The sequence A_{n} is defined by A_{n+1} = A_{n} + 3. Find A_{5}. [3]"
    text_status, fidelity_status, flags, fidelity_flags = _text_only_outcome(text)

    assert "contains_native_compacted_math_corruption" not in flags
    assert "native_compacted_math_corruption" not in fidelity_flags
    assert fidelity_status == "clean"
    assert text_status == "review"


def test_merged_diagram_and_table_prompt_wording_blocks_ready() -> None:
    diagram_variants = [
        "5 Thediagramshowsasector OAB of a circle. Find the area of the shaded segment. [4]",
        "5 Thediagram shows a sector OAB of a circle. Find the area of the shaded segment. [4]",
        "5 The diagram shows a sector OAB of a circle. Find the area of the shaded segment. [4]",
        "12 Thepointtangent meets thediagram A with coordinatesshows y-axis. [2]",
        "5 Fig. 1 shows a shaded sector. Find its perimeter. [4]",
    ]
    for text in diagram_variants:
        text_status, fidelity_status, flags, _fidelity_flags = _text_only_outcome(text)
        assert text_status == "review"
        assert fidelity_status == "clean"
        assert "contains_graph_or_diagram_prompt" in flags

    table_text = "4 Thetableshows the values of x and P(X = x). Find E(X). [3]"
    text_status, _fidelity_status, flags, _fidelity_flags = _text_only_outcome(table_text)
    assert text_status == "review"
    assert "contains_table_or_data_prompt" in flags


def test_materially_flattened_math_blocks_ready() -> None:
    examples = [
        "1 Expand (1 + 3x)^{2}_{3} in ascending powers of x. [4]",
        "8 The first three terms are a, 3_{2}a and b respectively. [5]",
        "2 Show that the coefficient of friction is _{3}^{1}3. [3]",
        "2 Expand (6 - x)(1 - 2x)^{-} 2^{3} in ascending powers of x. [4]",
    ]

    for text in examples:
        text_status, fidelity_status, flags, fidelity_flags = _text_only_outcome(text)
        assert text_status != "ready"
        assert fidelity_status == "clean"
        assert "contains_flattened_math_structure" in flags
        assert "flattened_math_structure" not in fidelity_flags


def test_symbol_loss_for_sigma_radicals_theta_and_pi_blocks_ready() -> None:
    examples = [
        "2 The weights are normally distributed with mean 1.04 kg and standard deviation 3 kg. Find the value of 3. [4]",
        "2 A geometric progression has first term 3 + 4 2 and second term 5 - 2. Give your answer in exact form. [3]",
        "5 The square roots of - 4 + 6 5i can be expressed in exact Cartesian form. [5]",
        "4 Solve sin i + cos i = 1 for 0 < i < π. [4]",
    ]

    for text in examples:
        text_status, fidelity_status, flags, fidelity_flags = _text_only_outcome(text)
        assert text_status != "ready"
        assert fidelity_status == "degraded"
        assert "contains_symbol_loss" in flags
        assert "symbol_loss_detected" in fidelity_flags


def test_malformed_units_block_ready_but_clear_plain_text_units_are_allowed() -> None:
    malformed = [
        "4 The car moves at a constant speed of 36ms7!. Find the power. [2]",
        "5 The cyclist has speed 4ms~! and acceleration 0.3ms~?. Find the power. [3]",
        "2 A particle is projected vertically upwards with speed 10ms\"!. [2]",
        "4 The car travels at 32 ms™! up a hill. [2]",
    ]
    for text in malformed:
        text_status, fidelity_status, flags, fidelity_flags = _text_only_outcome(text)
        assert text_status != "ready"
        assert fidelity_status == "degraded"
        assert "contains_unit_corruption" in flags
        assert "malformed_unit_notation" in fidelity_flags

    valid = [
        "4 The car moves at a constant speed of 36 m s^-1. Find the power. [2]",
        "4 The car moves at a constant speed of 36ms^-1. Find the power. [2]",
        "4 The car moves at a constant speed of 36 m/s. Find the power. [2]",
        "4 The car moves at a constant speed of 36 m s^{-1}. Find the power. [2]",
    ]
    for text in valid:
        _text_status, fidelity_status, flags, fidelity_flags = _text_only_outcome(text)
        assert "contains_unit_corruption" not in flags
        assert "malformed_unit_notation" not in fidelity_flags
        assert fidelity_status == "clean"


def test_large_margin_ocr_structural_rejection_blocks_ready_without_marking_corrupt() -> None:
    text = "2 A particle moves with speed 6 m s^{-1}. Find the impulse. [3]"
    flags = _visual_flags(text, review_flags=["ocr_large_margin_blocked_by_structural_rejection"])
    fidelity_status, fidelity_flags = assess_text_fidelity(
        question_text=text,
        extraction_quality_flags=[],
        review_flags=["ocr_large_margin_blocked_by_structural_rejection"],
        validation_flags=[],
        question_structure_detected={},
        mapping_failure_reason="",
        text_source_profile="native_pdf",
    )
    role, trust, visual_required = derive_question_text_semantics(
        question_text=text,
        text_fidelity_status=fidelity_status,
        visual_reason_flags=flags,
    )
    text_status = derive_text_only_status(
        validation_status="pass",
        scope_quality_status="clean",
        question_text_role=role,
        question_text_trust=trust,
    )

    assert "ocr_large_margin_structural_rejection" in flags
    assert fidelity_status == "clean"
    assert fidelity_flags == []
    assert role == "search_hint"
    assert trust == "medium"
    assert visual_required is True
    assert text_status == "review"
