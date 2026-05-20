from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizationResult:
    normalized_text: str
    flags: list[str]
    confidence: float
    warnings: list[str]


def normalize_advisory_question_text(
    text: str,
    *,
    native_pdf_text_raw: str = "",
    ocr_text_raw: str = "",
    metadata: dict[str, Any] | None = None,
) -> NormalizationResult:
    """Build an advisory normalized text candidate without mutating raw text.

    This is intentionally conservative. It fixes repeated extraction artifacts
    seen in frozen fixtures and flags every class of change for review.
    """

    metadata = metadata or {}
    flags: list[str] = []
    warnings: list[str] = []
    value = text if isinstance(text, str) else ""
    original = value

    value = _normalize_unicode(value, flags)
    value = _normalize_common_spacing(value, flags)
    value = _normalize_caie_subparts(value, flags)
    value = _normalize_derivatives(value, flags)
    value = _normalize_fractions(value, flags, warnings)
    value = _normalize_powers(value, flags, warnings)
    value = _normalize_roots(value, flags, warnings)
    value = _normalize_trig_and_logs(value, flags, warnings)
    value = _normalize_vectors_and_matrices(value, flags, warnings)
    value = _normalize_inequalities(value, flags, warnings)
    value = _normalize_multiline_expressions(value, flags)
    value = _normalize_common_spacing(value, flags)

    raw_context = " ".join(part for part in (native_pdf_text_raw, ocr_text_raw) if part)
    _add_source_disagreement_warnings(original, value, raw_context, warnings)

    flags = sorted(set(flags))
    warnings = sorted(set(warnings))
    confidence = _confidence(flags, warnings)
    return NormalizationResult(
        normalized_text=value.strip(),
        flags=flags,
        confidence=confidence,
        warnings=warnings,
    )


def _normalize_unicode(value: str, flags: list[str]) -> str:
    replacements = {
        "—": "-",
        "–": "-",
        "−": "-",
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "’": "'",
        "′": "'",
        "¢": "t",
    }
    changed = value
    for source, target in replacements.items():
        changed = changed.replace(source, target)
    if changed != value:
        flags.append("unicode_math_glyphs_normalized")
    return changed


def _normalize_common_spacing(value: str, flags: list[str]) -> str:
    changed = value
    replacements = [
        (r"\bThecurve\b", "The curve"),
        (r"\bThepolynomial\b", "The polynomial"),
        (r"\bWhenp\(x\)is\b", "When p(x) is"),
        (r"\bwherex\b", "where x"),
        (r"\bforx\b", "for x"),
        (r"\bThetwo\b", "The two"),
        (r"\bAredspinnerhasfoursideslabelled\b", "A red spinner has four sides labelled "),
        (r"\bdiehas\b", "die has"),
        (r"\bnumbers(?=\d)", "numbers "),
        (r"\bnumbered(?=\d)", "numbered "),
        (r"\bonthe\b", "on the"),
        (r"\bonwhicha\b", "on which a"),
        (r"\bisnoted\b", "is noted"),
        (r"\bTherandom\b", "The random"),
        (r"\bMand\b", "M and"),
        (r"\bNis\b", "N is"),
        (r"\bGiventhat\b", "Given that"),
        (r"\bsolvethis\b", "solve this"),
        (r"\bdi(?:ff|ﬀ)erentialequationtofindtheequationofthecurve\b", "differential equation to find the equation of the curve"),
    ]
    for pattern, replacement in replacements:
        changed = re.sub(pattern, replacement, changed)
    changed = re.sub(r"\s+", " ", changed)
    changed = re.sub(r"\s+([,.;:])", r"\1", changed)
    changed = re.sub(r"([(])\s+", r"\1", changed)
    changed = re.sub(r"\s+([)])", r"\1", changed)
    changed = re.sub(r"(?<=\d)\s+kg\b", " kg", changed)
    if changed != value:
        flags.append("spacing_artifacts_normalized")
    return changed


def _normalize_caie_subparts(value: str, flags: list[str]) -> str:
    changed = re.sub(r"\s*\(([a-z])\)\s*", r" (\1) ", value)
    changed = re.sub(r"\s+\((i{1,3}|iv|v)\)\s+", r" (\1) ", changed)
    if changed != value:
        flags.append("subpart_labels_spaced")
    return changed


def _normalize_derivatives(value: str, flags: list[str]) -> str:
    changed = re.sub(r"\bdd?yx\b", "dy/dx", value)
    changed = re.sub(r"\bc?y\s*=\s*for x\b", "dy/dx = ... for x", changed)
    if changed != value:
        flags.append("derivative_notation_normalized")
    return changed


def _normalize_fractions(value: str, flags: list[str], warnings: list[str]) -> str:
    changed = value
    changed = re.sub(r"(?<![\w}])(\d+)_\{(\d+)\}", r"\1/\2", changed)
    changed = re.sub(r"\^\{\s*(\d+)\s*\}_\{(\d+)\}", r"\1/\2", changed)
    changed = re.sub(r"\(\s*([A-Za-z]{1,4})\s*\)/\(\s*([A-Za-z]{1,4})\s*\)", r"\1/\2", changed)
    changed = re.sub(r"(?<!\w)1\s+2\s+(dy)\s*/\s*2\s+(dx)", r"1/2 d\1/d\2", changed)
    if changed != value:
        flags.append("fraction_notation_normalized")
    if re.search(r"\d+_\{\d+\}|\^\{\s*\d+\s*\}_\{\d+\}", value):
        warnings.append("stacked fraction text was flattened as advisory a/b notation")
    return changed


def _normalize_powers(value: str, flags: list[str], warnings: list[str]) -> str:
    changed = value
    changed = re.sub(r"\?\s*(?=[+)=.,;]|$)", "^{2}", changed)
    changed = re.sub(r"\b([a-zA-Z])\s*\^\{\s*([+-]?\d+)\s*\}\s*([A-Za-z])", r"\1^{\2}\3", changed)
    changed = re.sub(r"e\^\{-\}\s*([0-9A-Za-z]+)", r"e^{-\1}", changed)
    changed = re.sub(r"\bms\^\{-\}\s*1\b", "m s^{-1}", changed)
    if changed != value:
        flags.append("power_notation_normalized")
    if "?" in value:
        warnings.append("question-mark-to-power repair is based on common PDF glyph loss and needs image review")
    return changed


def _normalize_roots(value: str, flags: list[str], warnings: list[str]) -> str:
    changed = value
    changed = re.sub(r"\b(v|√)\s*x\s*\+\s*1\s*\+\s*3\b", "sqrt(x + 1) + 3", changed)
    changed = re.sub(r"\b(v|√)\s*3\b", "sqrt(3)", changed)
    if changed != value:
        flags.append("root_notation_normalized")
        warnings.append("root span inferred from local fixture pattern; verify against image")
    return changed


def _normalize_trig_and_logs(value: str, flags: list[str], warnings: list[str]) -> str:
    changed = value
    changed = re.sub(r"\b(sin|cos|tan)\s+θ\b", r"\1 θ", changed)
    if "θ" in changed:
        changed = re.sub(r"\bcos\s+21\b", "cos(2θ)", changed)
    changed = re.sub(r"\bln\s+([A-Za-z])\^\{([^}]+)\}", r"ln(\1^{\2})", changed)
    changed = re.sub(r"\bIn(?=[a-zA-Z0-9(])", "ln", changed)
    changed = re.sub(r"\b1n(?=[a-zA-Z0-9(])", "ln", changed)
    changed = re.sub(r"\bln(?=\d)", "ln ", changed)
    if changed != value:
        flags.append("trig_log_notation_normalized")
    if "cos 21" in value:
        warnings.append("cos 21 was interpreted as cos(2θ) because theta appears elsewhere")
    return changed


def _normalize_vectors_and_matrices(value: str, flags: list[str], warnings: list[str]) -> str:
    changed = value
    changed = re.sub(r"-{2,}\s*([A-Z]{2})¿", r"vector(\1)", changed)
    changed = re.sub(r"\b([A-Z]{2})¿", r"vector(\1)", changed)
    changed = re.sub(r"@\s*([+-]?\d)\s*([+-]?\d)A\b", r"column_vector(\1,\2)", changed)
    if changed != value:
        flags.append("vector_matrix_notation_normalized")
    if "@" in value or "¿" in value:
        warnings.append("vector or matrix glyph substitutions were inferred and need image review")
    return changed


def _normalize_inequalities(value: str, flags: list[str], warnings: list[str]) -> str:
    changed = value
    changed = re.sub(r"\bin equal it y\b", "inequality", changed, flags=re.IGNORECASE)
    changed = re.sub(r"\bfor x20\b", "for x > 0", changed)
    changed = re.sub(r"\bx20\b", "x > 0", changed)
    if changed != value:
        flags.append("inequality_notation_normalized")
    if re.search(r"\bx20\b", value):
        warnings.append("x20 was interpreted as x > 0 from fixture context")
    return changed


def _normalize_multiline_expressions(value: str, flags: list[str]) -> str:
    changed = re.sub(r"\s+(\([a-z]\))\s+", r"\n\1 ", value)
    changed = re.sub(r"\s+(\((?:i{1,3}|iv|v)\))\s+", r"\n\1 ", changed)
    changed = re.sub(r"\s+(\[\d+\])(?=\s+\([a-z]\))", r" \1", changed)
    if changed != value:
        flags.append("subpart_line_breaks_inserted")
    return changed


def _add_source_disagreement_warnings(original: str, normalized: str, raw_context: str, warnings: list[str]) -> None:
    if not raw_context:
        return
    if normalized != original and len(normalized) > len(original) * 1.35:
        warnings.append("normalized candidate expanded substantially from selected text")
    if re.search(r"[?@¿]", original) and re.search(r"[?@¿]", raw_context):
        warnings.append("raw sources also contain uncertain glyphs; normalization remains low authority")


def _confidence(flags: list[str], warnings: list[str]) -> float:
    if not flags:
        return 1.0
    value = 0.92 - (0.04 * len(flags)) - (0.06 * len(warnings))
    if any("inferred" in warning or "needs image review" in warning for warning in warnings):
        value -= 0.08
    return round(max(0.25, min(0.95, value)), 2)
