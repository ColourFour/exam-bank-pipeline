from __future__ import annotations

from exam_bank.text_normalization import normalize_advisory_question_text


def test_normalizes_fixture_fraction_power_derivative_and_inequality_artifacts() -> None:
    result = normalize_advisory_question_text(
        "1 Thecurve is such that ddyx = (x -43)^{3} forx > 3 and y = 3x^{3} ln x^{4}, for x20."
    )

    assert "dy/dx" in result.normalized_text
    assert "for x > 3" in result.normalized_text
    assert "ln(x^{4})" in result.normalized_text
    assert "for x > 0" in result.normalized_text
    assert "derivative_notation_normalized" in result.flags
    assert "inequality_notation_normalized" in result.flags
    assert result.confidence < 1.0
    assert result.warnings


def test_normalizes_stacked_fraction_and_exponential_power_forms() -> None:
    result = normalize_advisory_question_text(
        "a 9 The constant a is such that ∫_{0} xe^{-}2x dx = 1_{8}. "
        "(a) Show that a = 1_{2} ln(4a + 2). [5]"
    )

    assert "e^{-2x}" in result.normalized_text
    assert "1/8" in result.normalized_text
    assert "1/2 ln(4a + 2)" in result.normalized_text
    assert "fraction_notation_normalized" in result.flags
    assert "power_notation_normalized" in result.flags


def test_normalizes_trig_vector_and_matrix_glyphs_with_warnings() -> None:
    result = normalize_advisory_question_text(
        "5 Solve sin θ = 3 cos 21 + 2. Express the vectors---OM¿and---MN¿. "
        "translated by@ 41A."
    )

    assert "cos(2θ)" in result.normalized_text
    assert "vector(OM)" in result.normalized_text
    assert "vector(MN)" in result.normalized_text
    assert "column_vector(4,1)" in result.normalized_text
    assert "trig_log_notation_normalized" in result.flags
    assert "vector_matrix_notation_normalized" in result.flags
    assert any("image review" in warning for warning in result.warnings)


def test_normalizes_root_fixture_pattern_but_marks_it_ambiguous() -> None:
    result = normalize_advisory_question_text("3 The equation is y = (x - 3)vx + 1+3.")

    assert "(x - 3)sqrt(x + 1) + 3" in result.normalized_text
    assert "root_notation_normalized" in result.flags
    assert any("root span inferred" in warning for warning in result.warnings)


def test_false_positive_risk_does_not_rewrite_plain_question_marks_or_words() -> None:
    result = normalize_advisory_question_text("Why is x? Explain in words, not a formula.")

    assert result.normalized_text == "Why is x? Explain in words, not a formula."
    assert "power_notation_normalized" not in result.flags


def test_unchanged_plain_text_retains_full_confidence_and_no_flags() -> None:
    result = normalize_advisory_question_text("Find the probability that X is greater than 4. [2]")

    assert result.normalized_text == "Find the probability that X is greater than 4. [2]"
    assert result.flags == []
    assert result.warnings == []
    assert result.confidence == 1.0
