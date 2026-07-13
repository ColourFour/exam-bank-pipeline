from exam_bank.mark_scheme_labels import top_level_mark_scheme_text_label
from exam_bank.mark_schemes import _trim_mark_scheme_text_to_question


def test_top_level_mark_scheme_label_rejects_formula_parenthetical() -> None:
    assert top_level_mark_scheme_text_label("7(o)/(8X - 8g sin 5) - F = 8a") is None
    assert top_level_mark_scheme_text_label("7 (i) Resolve parallel to the plane") == "7"
    assert top_level_mark_scheme_text_label("7(i) Resolve parallel to the plane") == "7"


def test_mark_scheme_text_is_trimmed_to_current_top_level_question() -> None:
    text = "4 (i) previous answer\nworking\n5 (i) target answer\nmore working\n6 (i) next answer"

    assert _trim_mark_scheme_text_to_question(text, "5", {"4", "5", "6"}) == "5 (i) target answer\nmore working"
