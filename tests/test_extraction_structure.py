from pathlib import Path

from exam_bank.config import AppConfig
from exam_bank.extraction_structure import build_structured_question_text
from exam_bank.models import BoundingBox, PageLayout, QuestionSpan, TextBlock


def _block(text: str, y: float, x: float = 60, width: float = 420, page: int = 1) -> TextBlock:
    return TextBlock(page_number=page, text=text, bbox=BoundingBox(x, y, x + width, y + 14))


def test_preserves_display_math_lines_in_body_and_math_lines() -> None:
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[],
    )
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="1",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("1 Find the values of x for which", 80),
            _block("x^2 - 5x + 6 = 0", 110, x=110, width=220),
            _block("and state the interval 0 < x < 6. [4]", 145),
        ],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "x^2 - 5x + 6 = 0" in structured.body_text_raw
    assert "x^2 - 5x + 6 = 0" in structured.math_lines
    assert structured.body_text_raw.count("\n") >= 2


def test_separates_diagram_labels_from_body_text() -> None:
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[],
        graphics=[BoundingBox(320, 160, 540, 360)],
    )
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="3",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("3 The diagram shows a sector ABC of a circle with centre O.", 80),
            _block("Find the perimeter of the shaded segment. [5]", 110),
            _block("A", 185, x=330, width=12),
            _block("B", 185, x=515, width=12),
            _block("O", 255, x=420, width=12),
            _block("8 cm", 300, x=360, width=35),
            _block("30°", 220, x=450, width=24),
        ],
        full_question_label="3",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "The diagram shows a sector ABC of a circle with centre O." in structured.body_text_normalized
    assert "A" in structured.diagram_text
    assert "8 cm" in structured.diagram_text
    assert "30°" in structured.diagram_text
    assert "8 cm" not in structured.combined_question_text


def test_preserves_part_boundaries_and_part_math_lines() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="5",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("5(a) Expand (1 + 2x)^4. [2]", 80),
            _block("(b) Hence find the coefficient of x^2 in", 120),
            _block("(1 + 2x)^4 (1 - x). [2]", 150, x=100, width=240),
        ],
        full_question_label="5(a)-(b)",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert len(structured.part_texts) == 2
    assert structured.part_texts[0]["part_label"] == "(a)"
    assert structured.part_texts[1]["part_label"] == "(b)"
    assert "(1 + 2x)^4 (1 - x). [2]" in structured.part_texts[1]["math_lines"]


def test_flags_malformed_power_or_symbol_runs() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="1",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("1 Solve the equation 4 sin 1 + tan 1 = 0", 80),
            _block("for 0Å < 1 < 180Å. [3]", 110),
        ],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "θ" in structured.combined_question_text
    assert "°" in structured.combined_question_text
    assert "broken_superscript_or_power" not in structured.extraction_quality_flags


def test_combined_question_text_prefers_clean_body_without_diagram_pollution() -> None:
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[],
        graphics=[BoundingBox(300, 180, 540, 340)],
    )
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="7",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("7 The graph of y = f(x) is shown.", 80),
            _block("(a) Describe the transformation. [2]", 110),
            _block("x", 320, x=520, width=10),
            _block("y", 190, x=305, width=10),
        ],
        full_question_label="7(a)",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert structured.combined_question_text == structured.body_text_normalized
    assert "x" in structured.diagram_text
    assert "y" in structured.diagram_text
    assert structured.combined_question_text.endswith("[2]")


def test_normalizes_pdf_control_glyphs_without_dropping_math_structure() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="1",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("1 Solve the equation ln\x00e^{2}x + 3\x01 = 2x + ln 3. [4]", 80),
            _block("2\x0e3x -1\x0e < \x0ex + 1\x0e. [4]", 110),
            _block("Find the exact value of Ó_{0}^{2}tan^{-}1\x101_{2}x\x11 dx. [5]", 140),
            _block("Solve cos\x001 -60Å\x01 = 3 sin 1. [5]", 170),
        ],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "ln(e^{2}x + 3)" in structured.combined_question_text
    assert "|3x -1|" in structured.combined_question_text
    assert "∫_{0}^{2} tan^{-}1(1_{2}x) dx" in structured.combined_question_text
    assert "cos(θ -60°) = 3 sin θ" in structured.combined_question_text
    assert "\x00" not in structured.combined_question_text
    assert "\x0e" not in structured.combined_question_text
    assert "Ó" not in structured.combined_question_text


def test_normalizes_common_ocr_math_substitutions_without_marking_clean_math_corrupt() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="2",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("2 Find the stationary point of y = e^{2}xsin2x for 0GxG ^{1}_{2}r. [5]", 80),
        ],
        full_question_label="2",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "e^{2}x sin 2x" in structured.combined_question_text
    assert "0 ≤ x ≤ ^{1}_{2}π" in structured.combined_question_text
    assert "flattened_display_math" in structured.extraction_quality_flags
    assert "math_corruption_suspected" not in structured.extraction_quality_flags
    assert "likely_needs_visual_review" not in structured.extraction_quality_flags


def test_repairs_selected_joined_prompt_words() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="3",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("3 Findthevalueofxforwhich3\x002^{1}-x\x01 = 7^{x}. Giveyouranswerintheform lnln b a, whereaandbare integers. [4]", 80),
        ],
        full_question_label="3",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "Find the value of x for which 3(2^{1}-x)" in structured.combined_question_text
    assert "Give your answer in the form" in structured.combined_question_text
    assert "where a and b are integers" in structured.combined_question_text


def test_repairs_long_joined_prose_tokens_from_pdf_spacing_failures() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="6",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block(
                "6 Thediagramshowsthecurvewithequationy = 9(x^{-}1_{2} -4x^{-}3_{2}). "
                "Thereisaconstantresistanceforceofmagnitude600 N. "
                "Useaniterativeformulabasedontheequationinpart(a)todetermine x. [6]",
                80,
            ),
            _block("Describefullythetwosingletransformations. Thewinchisusedtopullaloadofmass 50kg. [4]", 110),
            _block(
                "The particles are attached to theendsofalightinextensiblestring. "
                "Find theprobabilitythatarandomlychosenstudentpassesthewrittentestatthefirstattempt. "
                "Find theprobabilitythatarandomlychosenhouseholdhasgoodbroadbandservicegiven that the household is in Shan. [3]",
                140,
            ),
            _block(
                "Given that this power is suddenly increased by 12 kW, find theinstantaneousacceleration. "
                "The curve is reflected in the y-axisandthenstretchedbyscalefactor 1_{3}. [5]",
                170,
            ),
            _block(
                "Solve by factor is in g. No student is allowed m or e than one attempt. "
                "Each child keeps an egg, keep in g the sweet it contained. Water is be in g pumpedintothe tank. [4]",
                200,
            ),
        ],
        full_question_label="6",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "The diagram shows the curve with equation y" in structured.combined_question_text
    assert "There is a constant resistance force of magnitude 600 N" in structured.combined_question_text
    assert "Use an iterative formula based on the equation in part (a) to determine x" in structured.combined_question_text
    assert "Describe fully the two single transformations" in structured.combined_question_text
    assert "The winch is used to pull a load of mass 50kg" in structured.combined_question_text
    assert "the ends of a light inextensible string" in structured.combined_question_text
    assert "the probability that a randomly chosen student passes the written test at the first attempt" in structured.combined_question_text
    assert "the probability that a randomly chosen household has good broadband service given" in structured.combined_question_text
    assert "the instantaneous acceleration" in structured.combined_question_text
    assert "axis and then stretched by scale factor" in structured.combined_question_text
    assert "factorising" in structured.combined_question_text
    assert "more than one attempt" in structured.combined_question_text
    assert "m or e" not in structured.combined_question_text
    assert "keeping" in structured.combined_question_text
    assert "be in g" not in structured.combined_question_text
    assert "f u l l y" not in structured.combined_question_text


def test_repairs_trig_theta_placeholder_i_and_exclamation() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="5",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("5 Solve the equation 4sinitani = 1 + 5cosi for 0 < i < 180°. [6]", 80),
            _block("A plane is inclined at angle of ! to the horizontal, where sin ! = 0.1. [3]", 110),
            _block("Angle POQ = i radians. Find the exact value of i and -180°1i1180°. [2]", 140),
            _block("Angle ACB is i radians and BQP = i°. A van ascends a hill inclined at an angle °i to the horizontal. Find the value of i. [4]", 170),
        ],
        full_question_label="5",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "4 sin θ tan θ = 1 + 5 cos θ" in structured.combined_question_text
    assert "0 < θ < 180°" in structured.combined_question_text
    assert "angle of θ" in structured.combined_question_text
    assert "sin θ = 0.1" in structured.combined_question_text
    assert "Angle POQ = θ radians" in structured.combined_question_text
    assert "Angle ACB is θ radians" in structured.combined_question_text
    assert "BQP = θ°" in structured.combined_question_text
    assert "angle θ° to the horizontal" in structured.combined_question_text
    assert "value of θ" in structured.combined_question_text
    assert "-180° < θ < 180°" in structured.combined_question_text


def test_repairs_audit_regressions_without_splitting_expression() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="5",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block(
                "5 (a) Express 2x^{2}-8x + 14 in the form 2(x -a)^{2} + b\x03. [2] "
                "Describe fully a sequence of transformationsthatmapsthegraphof y = f(x)on to "
                "the graph of y = g(x). [4]",
                80,
            ),
            _block("Find an expression for f^{-}1(x). [2]", 110),
            _block("A geometric progression has common ratio cosi, where 01i1 ^{1}_{2}π.", 140),
            _block("Find the value of i. [3]", 170),
        ],
        full_question_label="5",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "\x03" not in structured.combined_question_text
    assert "transformations that maps the graph of y = f(x) onto the graph of y = g(x)" in structured.combined_question_text
    assert "expression for f^{-}1(x)" in structured.combined_question_text
    assert "express i on" not in structured.combined_question_text
    assert "cos θ" in structured.combined_question_text
    assert "0 < θ < ^{1}_{2}π" in structured.combined_question_text
    assert "value of θ" in structured.combined_question_text


def test_normal_function_spacing_does_not_create_broken_power_flag() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="4",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("4 The graph of lny against x is a straight line.", 80),
            _block("Find the values of k and c. [4]", 110),
        ],
        full_question_label="4",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "ln y" in structured.combined_question_text
    assert "broken_superscript_or_power" not in structured.extraction_quality_flags


def test_question_number_followed_by_george_is_not_ocr_inequality() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="2",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=700,
        page_numbers=[1],
        blocks=[
            _block("2 George has a fair 5-sided spinner.", 80),
            _block("Find the probability for 0GxG2r. [2]", 110),
        ],
        full_question_label="2",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "2 George" in structured.combined_question_text
    assert "≤ eorge" not in structured.combined_question_text
    assert "0 ≤ x ≤ 2π" in structured.combined_question_text
