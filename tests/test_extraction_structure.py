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


def test_text_only_graph_axis_cluster_is_ignored_by_combined_question_text() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="5",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=780,
        page_numbers=[1],
        blocks=[
            _block("5 A sprinter runs a race of 200 m.", 80),
            _block("(i) Hence sketch a displacement-time graph for the race. [6]", 120),
            _block("displacement (m)", 410, x=125, width=90),
            _block("200", 442, x=145, width=18),
            _block("0 time (s)", 725, x=155, width=360),
            _block("0 20", 738, x=165, width=295),
            _block("(ii) Find the value of V. [2]", 790),
        ],
        full_question_label="5(i)-(ii)",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "displacement (m)" not in structured.combined_question_text
    assert "0 time (s)" not in structured.combined_question_text
    assert "0 20" not in structured.combined_question_text
    assert "displacement (m)" in structured.diagram_text
    assert "0 time (s)" in structured.diagram_text
    assert "(ii) Find the value of V. [2]" in structured.combined_question_text


def test_marked_answer_line_near_diagram_stays_in_body_text() -> None:
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[],
        graphics=[BoundingBox(80, 90, 360, 260)],
    )
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="7",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=520,
        page_numbers=[1],
        blocks=[
            _block("7 y", 70, x=50, width=100),
            _block("O x", 250, x=100, width=300),
            _block("(iii) Obtain expressions to define f^{-1}, giving the values for which each expression is", 360),
            _block("valid. [4]", 388, x=96),
        ],
        full_question_label="7",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "valid. [4]" in structured.combined_question_text
    assert "valid. [4]" not in structured.diagram_text


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
    assert "∫_{0}^{2} tan^{-1}((1)/(2)x) dx" in structured.combined_question_text
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


def test_repairs_font_encoded_half_pi_interval_without_rewriting_ordinary_ones() -> None:
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
            _block("2 Show that there is one root in the interval 01x1^{1}_{2}r. [2]", 80),
        ],
        full_question_label="2",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "interval 0 ≤ x ≤ ^{1}_{2}π" in structured.combined_question_text


def test_repairs_high_signal_caie_math_delimiters_and_pi_theta_artifacts() -> None:
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
            _block("1 (a) Expand@1 -21xA_{2}. [1]", 80),
            _block("(b) Find the first four terms in the expansion of b2x - 3xl^{4}. [2]", 110),
            _block("(c) Solve the equation for - rGiGr. [3]", 140),
        ],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "Expand(1 -21x)^{2}" in structured.combined_question_text
    assert "(2x - 3x)^{4}" in structured.combined_question_text
    assert "-π ≤ θ ≤ π" in structured.combined_question_text
    assert "@" not in structured.combined_question_text
    assert "rGiGr" not in structured.combined_question_text


def test_repairs_targeted_probability_and_mechanics_joined_prose() -> None:
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
            _block("1 Afairspinnerhassidesnumbered1to5andisspun. Findthescoreon thesideonwhich it comestorest. [2]", 80),
            _block(
                "Thereisaresistancetothemotionoftheblock,whichthecranedoes10000Jofworkto overcome. "
                "Giventhattheaveragepowerexertedbythecraneis12.5kW, find thetotaltimeforwhichthe block is in motion. [2]",
                110,
            ),
        ],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "A fair spinner has sides numbered 1 to 5 and is spun" in structured.combined_question_text
    assert "Find the score on the side on which it comes to rest" in structured.combined_question_text
    assert "There is a resistance to the motion of the block" in structured.combined_question_text
    assert "which the crane does 10000 J of work to overcome" in structured.combined_question_text
    assert "Given that the average power exerted by the crane is 12.5 kW" in structured.combined_question_text
    assert "the total time for which the block is in motion" in structured.combined_question_text


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
    assert "The winch is used to pull a load of mass 50 kg" in structured.combined_question_text
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
    assert "expression for f^{-1}(x)" in structured.combined_question_text
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


def test_keeps_complex_variable_r_and_repairs_zmath_without_global_pi_guess() -> None:
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
        blocks=[_block("1 The area is 12r and the complex number z has modulus r. [2]", 80)],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert structured.combined_question_text == (
        "1 The area is 12r and the complex number z has modulus r. [2]"
    )


def test_drops_answer_lines_and_trailing_copyright_furniture() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="8",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=800,
        page_numbers=[1],
        blocks=[
            _block("8 Find the total amount recycled. [5]", 80),
            _block("Scheme A ................................................................", 110),
            _block("Permission to reproduce items where third-party material is included.", 700),
            _block("Every reasonable effort has been made by the publisher.", 730),
        ],
        full_question_label="8",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert structured.combined_question_text == "8 Find the total amount recycled. [5]"


def test_normalizes_ff_ligature_mapsto_and_vector_arrow_artifacts() -> None:
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
            _block("2 A diﬀerent map f : x ↦→ x + 1 has vector (--→)/(OA). [3]", 80),
        ],
        full_question_label="2",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert structured.combined_question_text == (
        r"2 A different map f : x ↦ x + 1 has vector \overrightarrow{OA}. [3]"
    )


def test_repairs_complex_binary_minus_without_corrupting_negative_values() -> None:
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
            _block(
                "1 The roots are -2160 and -3. For |z -4| = 1 and arg(z -u) = 0, "
                "find u. [4]",
                80,
            )
        ],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "roots are -2160 and -3" in structured.combined_question_text
    assert "|z - 4| = 1" in structured.combined_question_text
    assert "arg(z - u) = 0" in structured.combined_question_text


def test_repairs_cross_line_context_and_compact_fractional_powers() -> None:
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
            _block(
                "4 A plane is inclined at an angle to the\nhorizontal, where sin = (16)/(65).",
                80,
            ),
            _block("For -(1)/(2)π < t < (1)/(2)π, a = t(- 1)/(2).", 110),
            _block("Given that ggf -1(12) = 62, find a. [4]", 140),
        ],
        full_question_label="4",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "angle α to the horizontal, where sin α = (16)/(65)" in structured.combined_question_text
    assert "For -(1)/(2)π < t < (1)/(2)π" in structured.combined_question_text
    assert "a = t^{-(1)/(2)}" in structured.combined_question_text
    assert "ggf^{-1}(12) = 62" in structured.combined_question_text


def test_negative_exponent_does_not_consume_prose_and_gio_is_not_inequality() -> None:
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
            _block("4 Gio moves at 6 m s^{-}1when t = 0 and 0GxG2. [3]", 80),
        ],
        full_question_label="4",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "4 Gio moves at 6 m s^{-1} when t = 0" in structured.combined_question_text
    assert "4 ≤ io" not in structured.combined_question_text
    assert "0 ≤ x ≤ 2" in structured.combined_question_text


def test_keeps_lowercase_wrapped_prose_overlapping_a_broad_graphic() -> None:
    layout = PageLayout(
        page_number=1,
        width=595,
        height=842,
        blocks=[],
        graphics=[BoundingBox(0, 50, 595, 210)],
    )
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
            _block("6 The examination was taken by 234", 65, x=49),
            _block("students.", 80, x=72, width=50),
            _block("(i) Draw a histogram. [5]", 165, x=79),
        ],
        full_question_label="6",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "taken by 234\nstudents." in structured.body_text_raw
    assert "students." not in structured.diagram_text


def test_drops_page_turn_furniture_without_dropping_following_question_text() -> None:
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[])
    span = QuestionSpan(
        source_pdf=Path("paper.pdf"),
        paper_name="paper",
        question_number="5",
        start_page=1,
        start_y=40,
        end_page=1,
        end_y=800,
        page_numbers=[1],
        blocks=[
            _block("5 Find x. [2]", 80),
            _block("[Questions 6 and 7 are printed on the next page.]", 720),
        ],
        full_question_label="5",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert structured.combined_question_text == "5 Find x. [2]"


def test_repairs_split_exponents_trig_fraction_and_vector_dot_product() -> None:
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
            _block("3 Solve 3^{x}+2 = 9 and |2^{x}^{+}1 - 2| < 0.5.", 80),
            _block("Evaluate e^{-(1)/(2)}^{x} and sin (1)/(2θ).", 110),
            _block("Use 0 ≤ arg(z) ≤ (1)/(4π) and --→PN. --→PM. [5]", 140),
        ],
        full_question_label="3",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "3^{x + 2} = 9" in structured.combined_question_text
    assert "|2^{x+1} - 2| < 0.5" in structured.combined_question_text
    assert "e^{-(1)/(2)x}" in structured.combined_question_text
    assert "sin((1)/(2)θ)" in structured.combined_question_text
    assert "arg(z) ≤ (1)/(4)π" in structured.combined_question_text
    assert r"\overrightarrow{PN} · \overrightarrow{PM}" in structured.combined_question_text


def test_repairs_displaced_mechanics_unit_tokens() -> None:
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
            _block(
                "1 The speed is ms5 -1, the acceleration is m2 s -2 and the velocity is mv s -1. [3]",
                80,
            )
        ],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "speed is 5 m s^{-1}" in structured.combined_question_text
    assert "acceleration is 2 m s^{-2}" in structured.combined_question_text
    assert "velocity is v m s^{-1}" in structured.combined_question_text


def test_repairs_point_labels_split_exponents_subscripts_and_degree() -> None:
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
            _block("2 Points A (10, 2) and P (a, b) lie on re^{i}θ.", 80),
            _block("Use x_{n}_{+}1 with e^{-3}x and 4e^{-}x at α^{°}. [4]", 110),
        ],
        full_question_label="2",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "A(10, 2) and P(a, b)" in structured.combined_question_text
    assert "re^{iθ}" in structured.combined_question_text
    assert "x_{n+1}" in structured.combined_question_text
    assert "e^{-3x}" in structured.combined_question_text
    assert "4e^{-x}" in structured.combined_question_text
    assert "α°" in structured.combined_question_text


def test_repairs_prose_bound_minus_plain_unit_power_and_split_inverse_trig() -> None:
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
            _block("3 A constant_{-} speed is 16 m s 1.", 80),
            _block("The angle is sin^{-10} 15. to the horizontal. [3]", 110),
        ],
        full_question_label="3",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "constant speed is 16 m s^{-1}" in structured.combined_question_text
    assert "sin^{-1} 0.15 to the horizontal" in structured.combined_question_text


def test_repairs_flattened_fraction_power_and_trig_layouts() -> None:
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
            _block("4 The probability of throwing a head is 3^{2}.", 80),
            _block("Use ((1)/(x - 1 2)), k(2)/(x) and e^{2}_{-}t^{2}.", 110),
            _block("Show that A = 2πr(2 + 2000)/(r) and 2^{1}(sin x).", 140),
            _block("Evaluate cos 2^{1}x and tan (1)/(2)p. [6]", 170),
        ],
        full_question_label="4",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "probability of throwing a head is (2)/(3)" in structured.combined_question_text
    assert "(1)/((x - 1)^{2})" in structured.combined_question_text
    assert "(k^{2})/(x)" in structured.combined_question_text
    assert "e^{2 - t^{2}}" in structured.combined_question_text
    assert "A = 2πr^{2} + (2000)/(r)" in structured.combined_question_text
    assert "(1)/(2)(sin x)" in structured.combined_question_text
    assert "cos((1)/(2)x)" in structured.combined_question_text
    assert "tan((1)/(2)p)" in structured.combined_question_text


def test_repairs_vector_and_integral_geometry_flattening() -> None:
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
                "5 The points have position vectors given by16 --→OA and --→OB; "
                "also (--→)/(pOA).",
                80,
            ),
            _block("Evaluate ∫_{0}^{6} (x x + 1)/(x^{2} + 4)()dx. [5]", 110),
        ],
        full_question_label="5",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert r"\overrightarrow{OB}" in structured.combined_question_text
    assert r"p\overrightarrow{OA}" in structured.combined_question_text
    assert "given by16" not in structured.combined_question_text
    assert r"given by \overrightarrow{OA}" in structured.combined_question_text
    assert "(x(x + 1))/(x^{2} + 4) dx" in structured.combined_question_text


def test_repairs_inverse_trig_radical_and_split_equation_powers() -> None:
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
            _block("1 Solve y = -cos^{-1}((1√)/(2)3).", 80),
            _block("Given 8^{3} - 6x = 4#5^{-2} x, find x. [4]", 110),
        ],
        full_question_label="1",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "-cos^{-1}((1)/(2)√(3))" in structured.combined_question_text
    assert "8^{3 - 6x} = 4 × 5^{-2x}" in structured.combined_question_text


def test_repairs_legacy_mechanics_decimal_units_ranges_and_fractional_power() -> None:
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
            _block("6 The particle accelerates at.0 6tms -2 for 15GtG20.", 80),
            _block(
                "Its velocity msv -1 is v = (2t + 1)2^{3} - 2 t^{2}, where 0GtG 3. [5]",
                110,
            ),
        ],
        full_question_label="6",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "accelerates at 0.6t m s^{-2}" in structured.combined_question_text
    assert "15 ≤ t ≤ 20" in structured.combined_question_text
    assert "velocity v m s^{-1}" in structured.combined_question_text
    assert "(2t + 1)^{(3)/(2)} - 2t^{2}" in structured.combined_question_text
    assert "0 ≤ t ≤ 3" in structured.combined_question_text


def test_repairs_detached_unit_exponents_and_contextual_alpha() -> None:
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
            _block("4 The speed is 16ms^{-}, and its power is 40 kW.^{1}", 80),
            _block("It moves at an^{1} angle a°; find the value of a. [3]", 110),
        ],
        full_question_label="4",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "16 m s^{-1}" in structured.combined_question_text
    assert "40 kW." in structured.combined_question_text
    assert "an angle α°" in structured.combined_question_text
    assert "value of α" in structured.combined_question_text


def test_repairs_joined_projection_prose_and_relocates_acceleration() -> None:
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
                "5 P is projected (upwardsfromhorizontal)/(-) ground at 12 m s 1and "
                "Q at 7 m s 1respectively. [2]",
                80,
            ),
            _block(
                "There is a constant 0.15 m sresistance2.to motion of 20 N. "
                "At this instant his acceleration is [3]",
                110,
            ),
        ],
        full_question_label="5",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "upwards from horizontal ground" in structured.combined_question_text
    assert "12 m s^{-1} and Q at 7 m s^{-1} respectively" in structured.combined_question_text
    assert "constant resistance to motion of 20 N" in structured.combined_question_text
    assert "his acceleration is 0.15 m s^{-2}." in structured.combined_question_text


def test_repairs_iterative_formula_and_relocates_displaced_coordinate() -> None:
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
                "6 Use the iterative formula () a_{n+1} = (1)/(2) exp 1 + "
                "(ln 2)/(a) n to determine a. [3]",
                80,
            ),
            _block(
                "The curve crosses the(4, (189)/(16)). x-axis and passes through "
                "the point (i) Find its equation. [4]",
                110,
            ),
        ],
        full_question_label="6",
    )

    structured = build_structured_question_text(span, [layout], AppConfig())

    assert "iterative formula a_{n+1}" in structured.combined_question_text
    assert "exp(1 + (ln 2)/(a_{n})) to determine" in structured.combined_question_text
    assert "crosses the x-axis and passes through the point (4, (189)/(16))." in (
        structured.combined_question_text
    )
