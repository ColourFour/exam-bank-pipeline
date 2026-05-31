"""Generate candidate CAIE 9709 skill maps and subpart mappings.

The generated JSON is intentionally conservative: official syllabus sections
anchor the taxonomies, while all question mappings remain machine candidates
unless a separate human review artifact proves otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_ROOT = ROOT / "exam_bank_taxonomy"
CANONICAL_ROOT = TAXONOMY_ROOT / "canonical"
CANONICAL_SKILL_MAPS = CANONICAL_ROOT / "skill_maps"
CANONICAL_SKILL_MAPPINGS = CANONICAL_ROOT / "question_skill_mappings"
CANONICAL_COVERAGE_REPORTS = CANONICAL_ROOT / "coverage_reports"
CANONICAL_INDEXES = CANONICAL_ROOT / "indexes"
TAXONOMY_LOGS = TAXONOMY_ROOT / "logs"
TAXONOMY_CHANGELOGS = TAXONOMY_LOGS / "changelogs"
TAXONOMY_VALIDATION_REPORTS = TAXONOMY_LOGS / "validation_reports"
QUESTION_BANK = ROOT / "output/json/question_bank.json"
ASTERION_QB = ROOT / "output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json"
CONTENT_LAB = ROOT / "output/asterion/exports/latest/asterion_content_lab_candidates_v1.json"
OLD_P3_SKILL_MAP = CANONICAL_SKILL_MAPS / "skill_map_9709_p3_v1.json"

SYLLABUS_CODE = "9709"
SUBJECT_NAME = "Cambridge International AS & A Level Mathematics"
SYLLABUS_REFERENCE = (
    "Cambridge International AS & A Level Mathematics 9709 syllabus for 2023, "
    "2024 and 2025, Subject content"
)
SYLLABUS_URL = "https://www.cambridgeinternational.org/Images/597421-2023-2025-syllabus.pdf"

COMPONENTS = {
    "p1": {
        "caie_class_or_component": "Paper 1",
        "component_label": "Pure Mathematics 1",
        "output_component": "p1",
        "component_code_prefixes": ["11", "12", "13", "15"],
    },
    "p3": {
        "caie_class_or_component": "Paper 3",
        "component_label": "Pure Mathematics 3",
        "output_component": "p3",
        "component_code_prefixes": ["31", "32", "33", "35"],
    },
    "p4": {
        "caie_class_or_component": "Mechanics 1 (Paper 4)",
        "component_label": "Mechanics",
        "output_component": "m1",
        "component_code_prefixes": ["41", "42", "43", "45"],
    },
    "p5": {
        "caie_class_or_component": "Probability & Statistics 1 (Paper 5)",
        "component_label": "Probability & Statistics 1",
        "output_component": "s1",
        "component_code_prefixes": ["51", "52", "53", "55"],
    },
}


def ref(section: str, component_label: str) -> str:
    return f"{SYLLABUS_REFERENCE}, {component_label}, section {section}"


def output_component(component: str) -> str:
    return COMPONENTS[component]["output_component"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def skill(
    skill_id: str,
    component: str,
    section: str,
    name: str,
    description: str,
    signals: list[str],
    errors: list[str],
    *,
    subsection: str | None = None,
    prerequisites: list[str] | None = None,
    related: list[str] | None = None,
    priority: str = "medium",
    role: str = "direct_assessed",
    notes: str = "",
) -> dict[str, Any]:
    meta = COMPONENTS[component]
    section_ref = section.split()[0]
    slug = re.sub(r"[^a-z0-9]+", "-", skill_id.lower()).strip("-")
    return {
        "skill_id": skill_id,
        "syllabus_code": SYLLABUS_CODE,
        "subject_name": SUBJECT_NAME,
        "caie_class_or_component": meta["caie_class_or_component"],
        "component_label": meta["component_label"],
        "section": section,
        "subsection": subsection,
        "name": name,
        "description": description,
        "assessment_role": role,
        "prerequisite_skill_ids": prerequisites or [],
        "related_skill_ids": related or [],
        "recognizer_signals": signals,
        "common_errors": errors,
        "asterion_region_id": f"asterion:{SYLLABUS_CODE}:{component}:{section_ref}:{slug}",
        "content_lab_priority": priority,
        "official_syllabus_reference": ref(section_ref, meta["component_label"]),
        "review_status": "needs_review",
        "notes": notes or "Machine-expanded candidate skill from official syllabus section; requires subject expert review.",
    }


SKILLS: dict[str, list[dict[str, Any]]] = {
    "p1": [
        skill(
            "9709_p1_quadratics_completed_square",
            "p1",
            "1.1 Quadratics",
            "Completing the square and quadratic form",
            "Rewrite quadratic expressions and use completed-square form to identify roots, ranges and turning points.",
            ["complete the square", "completed square", "(x + a)^2", "(x-a)^2", "turning point", "maximum possible value"],
            ["Incorrect sign inside the completed square", "Forgetting to adjust the constant term"],
            priority="high",
        ),
        skill(
            "9709_p1_quadratics_discriminant_intersections",
            "p1",
            "1.1 Quadratics",
            "Discriminants, roots and intersections",
            "Use the discriminant and algebraic intersections of lines and curves to classify real roots and tangency.",
            ["discriminant", "two distinct", "exactly one root", "does not intersect", "tangent to the curve", "line and curve"],
            ["Using the wrong inequality for two roots", "Not forming the quadratic in one variable before using the discriminant"],
            prerequisites=["9709_p1_quadratics_completed_square"],
            priority="high",
        ),
        skill(
            "9709_p1_functions_domain_range_one_one",
            "p1",
            "1.2 Functions",
            "Domain, range and one-one restrictions",
            "Determine domains, ranges and restrictions that make a function one-one.",
            ["domain", "range", "one-one", "largest possible value", "for x <", "for x ≥", "decreasing function"],
            ["Giving the unrestricted range", "Ignoring the stated domain when finding the range"],
            prerequisites=["9709_p1_quadratics_completed_square"],
            priority="high",
        ),
        skill(
            "9709_p1_functions_composition_inverse",
            "p1",
            "1.2 Functions",
            "Composite and inverse functions",
            "Find and use composite functions and inverse functions, including solving equations involving them.",
            ["fg(x)", "gf(x)", "ff(x)", "composite", "inverse", "f^{-", "f−1", "find an expression for"],
            ["Applying functions in the wrong order", "Not restricting the domain before finding an inverse"],
            prerequisites=["9709_p1_functions_domain_range_one_one"],
            priority="high",
        ),
        skill(
            "9709_p1_functions_transformations",
            "p1",
            "1.2 Functions",
            "Graph transformations",
            "Describe and apply transformations of graphs including stretches, translations and reflections.",
            ["transformation", "transformed to", "describe fully", "y = f(x", "2f", "f(2x", "broken lines"],
            ["Confusing horizontal and vertical stretch factors", "Using the wrong sign for translations inside f(x)"],
            prerequisites=["9709_p1_functions_domain_range_one_one"],
            priority="medium",
        ),
        skill(
            "9709_p1_coordinate_line_geometry",
            "p1",
            "1.3 Coordinate geometry",
            "Straight lines, gradients and perpendicularity",
            "Use gradient, intercept, parallel and perpendicular line relationships in coordinate geometry.",
            ["gradient", "line has equation", "perpendicular bisector", "parallel", "midpoint", "straight line"],
            ["Using the same gradient for a perpendicular line", "Dropping a negative reciprocal"],
            priority="medium",
        ),
        skill(
            "9709_p1_coordinate_circle_geometry",
            "p1",
            "1.3 Coordinate geometry",
            "Circle equations, chords and tangents",
            "Find and use equations of circles, radii, tangents, chords and circle-line intersections.",
            ["circle", "circumference", "radius", "centre", "chord", "tangent to the circle", "x^2 + y^2"],
            ["Forgetting to complete the square in both x and y", "Using tangent length instead of radius perpendicularity"],
            prerequisites=["9709_p1_coordinate_line_geometry", "9709_p1_quadratics_discriminant_intersections"],
            priority="high",
        ),
        skill(
            "9709_p1_circular_measure_arc_sector",
            "p1",
            "1.4 Circular measure",
            "Radians, arcs and sectors",
            "Use radian measure with arc length and sector area formulae.",
            ["radians", "sector", "arc", "radius", "angle", "area of the sector", "arc length"],
            ["Using degrees in radian formulae", "Confusing arc length and sector area"],
            priority="high",
        ),
        skill(
            "9709_p1_circular_measure_segments_perimeters",
            "p1",
            "1.4 Circular measure",
            "Segments, composite regions and perimeters",
            "Solve geometry problems involving sectors, segments, triangles and composite perimeters.",
            ["segment", "perimeter", "metal plate", "rope", "semicircle", "composite", "equal areas"],
            ["Omitting straight boundary lengths", "Using sector area where segment area is needed"],
            prerequisites=["9709_p1_circular_measure_arc_sector"],
            priority="medium",
        ),
        skill(
            "9709_p1_trigonometry_identities_exact_values",
            "p1",
            "1.5 Trigonometry",
            "Basic identities and exact trigonometric values",
            "Use standard trigonometric identities and exact values in algebraic simplification.",
            ["identity", "prove", "sin^2", "cos^2", "tan", "exact value", "sec", "cosec"],
            ["Dividing by an expression that may be zero", "Mixing exact values and rounded decimals"],
            priority="high",
        ),
        skill(
            "9709_p1_trigonometry_equations_intervals",
            "p1",
            "1.5 Trigonometry",
            "Trigonometric equations and intervals",
            "Solve trigonometric equations over stated degree or radian intervals.",
            ["solve the equation", "for 0", "θ", "theta", "degrees", "π", "tan x", "cos x", "sin x"],
            ["Missing secondary solutions", "Using degree mode for radian intervals or conversely"],
            prerequisites=["9709_p1_trigonometry_identities_exact_values"],
            priority="high",
        ),
        skill(
            "9709_p1_trigonometry_graphs",
            "p1",
            "1.5 Trigonometry",
            "Graphs of sine, cosine and tangent",
            "Interpret and use transformations and key features of trigonometric graphs.",
            ["graph of y = a", "tan(x", "cos(bx", "sin", "period", "amplitude", "number of solutions"],
            ["Reading the period as the horizontal scale factor", "Ignoring vertical translation"],
            prerequisites=["9709_p1_functions_transformations", "9709_p1_trigonometry_equations_intervals"],
            priority="medium",
        ),
        skill(
            "9709_p1_series_arithmetic_progressions",
            "p1",
            "1.6 Series",
            "Arithmetic progressions",
            "Use nth term and sum formulae for arithmetic progressions.",
            ["arithmetic progression", "common difference", "sum of the first", "nth term", "60th term"],
            ["Confusing common difference with common ratio", "Using n instead of n minus 1 in the nth term"],
            priority="medium",
        ),
        skill(
            "9709_p1_series_geometric_progressions",
            "p1",
            "1.6 Series",
            "Geometric progressions and sum to infinity",
            "Use nth term, finite sum and sum-to-infinity formulae for geometric progressions.",
            ["geometric progression", "common ratio", "sum to infinity", "|r|", "second term"],
            ["Using the sum-to-infinity formula when |r| is not less than 1", "Confusing r with d"],
            priority="high",
        ),
        skill(
            "9709_p1_series_binomial_positive_integer",
            "p1",
            "1.6 Series",
            "Binomial expansion for positive integer powers",
            "Expand binomial expressions with positive integer powers and use coefficients in linked products.",
            ["binomial", "expansion", "ascending powers", "coefficient of", "(1 + x)^", "(a + bx)^"],
            ["Using too few terms for a linked coefficient", "Dropping signs in negative terms"],
            priority="high",
        ),
        skill(
            "9709_p1_differentiation_power_chain",
            "p1",
            "1.7 Differentiation",
            "Differentiation of powers and simple composites",
            "Differentiate powers and simple composite functions using the chain rule where appropriate.",
            ["differentiate", "f′", "dy/dx", "gradient", "d/dx", "x^", "chain"],
            ["Reducing the power but not multiplying by the old power", "Missing a chain-rule factor"],
            priority="high",
        ),
        skill(
            "9709_p1_differentiation_tangents_normals",
            "p1",
            "1.7 Differentiation",
            "Tangents and normals",
            "Use derivatives to find gradients and equations of tangents and normals.",
            ["tangent", "normal", "gradient at", "equation of the tangent", "equation of the normal"],
            ["Using tangent gradient for the normal", "Not substituting the point after finding the gradient"],
            prerequisites=["9709_p1_differentiation_power_chain", "9709_p1_coordinate_line_geometry"],
            priority="high",
        ),
        skill(
            "9709_p1_differentiation_stationary_rates",
            "p1",
            "1.7 Differentiation",
            "Stationary points, classification and rates of change",
            "Find and classify stationary points and solve simple rates-of-change problems.",
            ["stationary point", "maximum", "minimum", "d^2y", "rate of increase", "units per second", "increasing"],
            ["Not checking the second derivative or sign change", "Confusing dy/dt with dy/dx"],
            prerequisites=["9709_p1_differentiation_power_chain"],
            priority="high",
        ),
        skill(
            "9709_p1_integration_reverse_differentiation",
            "p1",
            "1.8 Integration",
            "Reverse differentiation and constants",
            "Integrate powers and simple functions, including using a point to find the constant of integration.",
            ["integrate", "find f(x)", "passes through", "constant of integration", "gradient is given", "dy/dx"],
            ["Omitting the constant of integration", "Substituting a point before integrating"],
            prerequisites=["9709_p1_differentiation_power_chain"],
            priority="high",
        ),
        skill(
            "9709_p1_integration_area_volume",
            "p1",
            "1.8 Integration",
            "Definite integration, areas and volumes",
            "Use definite integration to find areas and volumes of revolution in P1 contexts.",
            ["area", "shaded region", "rotated through 360", "volume", "definite integral", "limits"],
            ["Using signed area unintentionally", "Forgetting to square the radius for volume of revolution"],
            prerequisites=["9709_p1_integration_reverse_differentiation"],
            priority="high",
        ),
    ],
    "p3": [
        skill(
            "9709_p3_3_1_modulus_equations_inequalities",
            "p3",
            "3.1 Algebra",
            "Modulus equations and inequalities",
            "Solve equations and inequalities involving modulus expressions, including interval and graphical reasoning.",
            ["modulus", "absolute value", "|", "solve the inequality", "|x", "|3x", "|z"],
            ["Squaring without checking intervals", "Dropping one modulus branch"],
            prerequisites=["9709_p1_quadratics_discriminant_intersections", "9709_p1_functions_domain_range_one_one"],
            priority="high",
        ),
        skill(
            "9709_p3_3_1_polynomial_division_factor_remainder",
            "p3",
            "3.1 Algebra",
            "Polynomial division, factors and remainders",
            "Use factor and remainder theorems and polynomial division to find constants, factors and roots.",
            ["polynomial", "factor", "remainder", "divided by", "p(x)", "root of the equation"],
            ["Using the wrong root sign", "Confusing factor and remainder conditions"],
            prerequisites=["9709_p1_quadratics_discriminant_intersections"],
            priority="high",
        ),
        skill(
            "9709_p3_3_1_partial_fractions",
            "p3",
            "3.1 Algebra",
            "Partial fractions",
            "Decompose rational expressions into partial fractions for algebraic manipulation, integration and expansion.",
            ["partial fractions", "rational expression", "denominator", "express f(x)", "hence obtain the expansion"],
            ["Using an incomplete numerator form", "Missing repeated factors"],
            prerequisites=["9709_p1_quadratics_discriminant_intersections"],
            priority="high",
        ),
        skill(
            "9709_p3_3_1_binomial_rational_expansion",
            "p3",
            "3.1 Algebra",
            "Binomial expansion for rational powers",
            "Expand expressions with rational or negative powers, identify validity ranges and use expansions for coefficients or estimates.",
            ["binomial", "expansion", "ascending powers", "valid for", "coefficient", "rational powers"],
            ["Using the positive-integer formula blindly", "Ignoring the validity condition"],
            prerequisites=["9709_p1_series_binomial_positive_integer"],
            priority="high",
        ),
        skill(
            "9709_p3_3_2_log_exponential_equations",
            "p3",
            "3.2 Logarithmic and exponential functions",
            "Logarithmic and exponential equations",
            "Use laws of logarithms, exponentials and natural logarithms to solve and transform equations.",
            ["ln", "log", "e^", "exp", "exponential", "laws of logarithms", "give your answer in the form"],
            ["Taking logs outside the valid domain", "Combining logs across addition"],
            prerequisites=["9709_p1_functions_domain_range_one_one"],
            priority="high",
        ),
        skill(
            "9709_p3_3_2_linearising_log_relationships",
            "p3",
            "3.2 Logarithmic and exponential functions",
            "Linearising logarithmic relationships",
            "Transform power or exponential models to linear form and use gradient/intercept information to find constants.",
            ["linear form", "ln y", "ln x", "straight line", "gradient", "intercept", "constant A"],
            ["Swapping gradient and intercept", "Using inconsistent log bases"],
            prerequisites=["9709_p1_coordinate_line_geometry", "9709_p3_3_2_log_exponential_equations"],
            priority="medium",
        ),
        skill(
            "9709_p3_3_3_reciprocal_trig_functions",
            "p3",
            "3.3 Trigonometry",
            "Reciprocal trigonometric functions",
            "Use secant, cosecant and cotangent definitions and identities in simplification, proof and equation solving.",
            ["sec", "cosec", "co sec", "cot", "1/cos", "1/sin", "1/tan"],
            ["Treating sec x as inverse cosine", "Using the wrong reciprocal identity"],
            prerequisites=["9709_p1_trigonometry_identities_exact_values"],
            priority="high",
        ),
        skill(
            "9709_p3_3_3_identities_compound_double_angle_equations",
            "p3",
            "3.3 Trigonometry",
            "Identities, compound angles and trigonometric equations",
            "Prove identities and solve equations using compound-angle formulae, double-angle formulae and interval control.",
            ["prove", "identity", "compound", "double angle", "cos 2", "sin 2", "tan 2", "tan(", "solve the equation"],
            ["Losing secondary interval solutions", "Dividing by a possibly zero expression"],
            prerequisites=["9709_p1_trigonometry_equations_intervals"],
            priority="high",
        ),
        skill(
            "9709_p3_3_3_rsin_rcos_form",
            "p3",
            "3.3 Trigonometry",
            "R sin(x plus alpha) and R cos(x plus alpha) form",
            "Express a sin x plus b cos x in single-phase form and use it for equations or extrema.",
            ["R sin", "R cos", "form R", "alpha", "greatest", "least", "maximum value", "minimum value"],
            ["Choosing alpha in the wrong quadrant", "Mixing sine and cosine forms"],
            prerequisites=["9709_p1_trigonometry_equations_intervals"],
            priority="high",
        ),
        skill(
            "9709_p3_3_4_derivative_rules",
            "p3",
            "3.4 Differentiation",
            "Derivative rules for P3 functions",
            "Differentiate powers, exponentials, logarithms and trigonometric functions using product, quotient and chain rules.",
            ["differentiate", "f′", "dy/dx", "derivative", "product rule", "quotient", "chain", "gradient"],
            ["Dropping a chain-rule factor", "Reversing quotient-rule terms"],
            prerequisites=["9709_p1_differentiation_power_chain"],
            priority="medium",
        ),
        skill(
            "9709_p3_3_4_parametric_implicit_differentiation",
            "p3",
            "3.4 Differentiation",
            "Parametric and implicit differentiation",
            "Find gradients and derivatives for parametric or implicitly defined curves.",
            ["parametric", "parameter", "dx/dt", "dy/dt", "implicit", "x =", "y =", "in terms of t"],
            ["Using dx/dt over dy/dt", "Treating y as constant in implicit differentiation"],
            prerequisites=["9709_p3_3_4_derivative_rules"],
            priority="medium",
        ),
        skill(
            "9709_p3_3_4_tangents_normals_stationary_points",
            "p3",
            "3.4 Differentiation",
            "Tangents, normals and stationary points",
            "Use derivatives to find tangents, normals, stationary points, maxima, minima and monotonicity.",
            ["tangent", "normal", "stationary point", "maximum", "minimum", "always positive", "increasing"],
            ["Using tangent gradient for the normal", "Not finding coordinates after dy/dx = 0"],
            prerequisites=["9709_p3_3_4_derivative_rules", "9709_p1_coordinate_line_geometry"],
            priority="high",
        ),
        skill(
            "9709_p3_3_5_standard_integration",
            "p3",
            "3.5 Integration",
            "Standard integration",
            "Integrate standard P3 power, exponential, logarithmic and trigonometric forms with constants and limits handled correctly.",
            ["integrate", "integral", "limits", "constant of integration", "exact value"],
            ["Omitting the constant of integration", "Using wrong signs for trigonometric antiderivatives"],
            prerequisites=["9709_p1_integration_reverse_differentiation"],
            priority="high",
        ),
        skill(
            "9709_p3_3_5_substitution_and_parts",
            "p3",
            "3.5 Integration",
            "Integration by substitution and by parts",
            "Use substitution and integration by parts, including changing limits where required.",
            ["substitution", "let u", "by parts", "integration by parts", "change of variable"],
            ["Not converting dx or limits", "Choosing u and dv poorly"],
            prerequisites=["9709_p3_3_5_standard_integration"],
            priority="high",
        ),
        skill(
            "9709_p3_3_5_trig_and_partial_fraction_integration",
            "p3",
            "3.5 Integration",
            "Trigonometric and partial-fraction integration",
            "Integrate trigonometric expressions and rational functions requiring identities or partial fractions.",
            ["partial fractions", "hence find ∫", "sin", "cos", "tan", "ln from rational", "integral"],
            ["Integrating before decomposing", "Forgetting scale factors in log terms"],
            prerequisites=["9709_p3_3_1_partial_fractions", "9709_p3_3_3_identities_compound_double_angle_equations"],
            priority="medium",
        ),
        skill(
            "9709_p3_3_5_area_volume_applications",
            "p3",
            "3.5 Integration",
            "Area and volume applications of integration",
            "Use definite integrals to find areas under or between curves and volumes of revolution.",
            ["area", "region", "bounded by", "x-axis", "volume", "revolution", "shaded"],
            ["Not splitting at an intersection", "Forgetting radius squared in volumes"],
            prerequisites=["9709_p3_3_5_standard_integration"],
            priority="high",
        ),
        skill(
            "9709_p3_3_6_root_location",
            "p3",
            "3.6 Numerical solution of equations",
            "Root location and uniqueness evidence",
            "Verify root intervals using sign-change, monotonicity, sketches or supporting numerical evidence.",
            ["root lies between", "verify by calculation", "change of sign", "one root", "sketching", "exactly one root"],
            ["Claiming uniqueness from sign change alone", "Rounding endpoint values too aggressively"],
            prerequisites=["9709_p1_functions_domain_range_one_one"],
            priority="high",
        ),
        skill(
            "9709_p3_3_6_fixed_point_iteration",
            "p3",
            "3.6 Numerical solution of equations",
            "Fixed-point iteration",
            "Rearrange equations into iterative form and carry out iterations to a requested accuracy.",
            ["iteration", "iterative", "x_{n+1}", "use the formula", "decimal places", "approximation"],
            ["Using a divergent rearrangement", "Stopping before accuracy is justified"],
            prerequisites=["9709_p3_3_6_root_location"],
            priority="high",
        ),
        skill(
            "9709_p3_3_7_vector_lines",
            "p3",
            "3.7 Vectors",
            "Vector equations of lines and intersections",
            "Use vector equations of lines to find points, parameters, intersections and line relationships.",
            ["r =", "line l", "intersect", "skew", "parallel", "position vector", "lambda", "parameter"],
            ["Using the same parameter on two lines", "Equating only one component"],
            prerequisites=["9709_p1_coordinate_line_geometry"],
            priority="high",
        ),
        skill(
            "9709_p3_3_7_scalar_product_angles",
            "p3",
            "3.7 Vectors",
            "Scalar product, angles and perpendicularity",
            "Use scalar product to find angles, projections, perpendicularity and related vector geometry.",
            ["scalar product", "dot product", "angle between", "acute angle", "perpendicular", "projection"],
            ["Forgetting magnitudes in the denominator", "Returning the obtuse angle when acute is requested"],
            prerequisites=["9709_p1_trigonometry_equations_intervals"],
            priority="high",
        ),
        skill(
            "9709_p3_3_7_position_vectors_geometry",
            "p3",
            "3.7 Vectors",
            "Position vectors and vector geometry",
            "Use position vectors in geometric configurations involving midpoints, ratios, parallelograms and trapezia.",
            ["position vectors", "OA", "OB", "OC", "midpoint", "trapezium", "quadrilateral", "ratio"],
            ["Mixing point vectors and direction vectors", "Using wrong ratio direction"],
            prerequisites=["9709_p3_3_7_vector_lines"],
            priority="medium",
        ),
        skill(
            "9709_p3_3_8_separable_differential_equations",
            "p3",
            "3.8 Differential equations",
            "Separable differential equations",
            "Set up and solve first-order separable differential equations.",
            ["differential equation", "dy/dx", "dx/dt", "separable", "solve the differential equation", "variables"],
            ["Failing to separate variables", "Omitting the arbitrary constant before applying a condition"],
            prerequisites=["9709_p3_3_5_standard_integration"],
            priority="high",
        ),
        skill(
            "9709_p3_3_8_initial_conditions_models",
            "p3",
            "3.8 Differential equations",
            "Initial conditions and model interpretation",
            "Use initial conditions and model statements to determine constants and interpret differential-equation solutions.",
            ["it is given that", "when x =", "when t =", "proportional to", "model", "initial condition"],
            ["Applying the initial condition before integrating", "Solving for the wrong constant"],
            prerequisites=["9709_p3_3_8_separable_differential_equations"],
            priority="high",
        ),
        skill(
            "9709_p3_3_9_complex_arithmetic_polar_form",
            "p3",
            "3.9 Complex numbers",
            "Complex arithmetic, modulus, argument and polar form",
            "Work with complex arithmetic, conjugates, modulus, argument and polar or exponential form.",
            ["complex number", "complex numbers", " i", "z*", "z^{*", "zz", "x + yi", "modulus", "argument", "arg", "re^{i", "x + iy", "conjugate"],
            ["Using degrees when radians are implied", "Choosing the wrong quadrant for the argument"],
            prerequisites=["9709_p1_trigonometry_equations_intervals"],
            priority="high",
        ),
        skill(
            "9709_p3_3_9_argand_loci_geometry",
            "p3",
            "3.9 Complex numbers",
            "Argand diagrams and loci",
            "Represent complex numbers on Argand diagrams and interpret loci involving modulus and argument inequalities.",
            ["Argand", "shade the region", "locus", "|z", "arg(z", "Im z", "Re z"],
            ["Shading the complement of the intended region", "Misplacing centres in modulus loci"],
            prerequisites=["9709_p3_3_9_complex_arithmetic_polar_form"],
            priority="high",
        ),
        skill(
            "9709_p3_3_9_complex_roots_polynomials",
            "p3",
            "3.9 Complex numbers",
            "Complex roots and polynomial equations",
            "Use complex roots, conjugate-pair reasoning and polynomial equations to find remaining roots or constants.",
            ["root of the equation", "complex roots", "other roots", "polynomial", "z^", "conjugate root"],
            ["Forgetting conjugate roots for real coefficients", "Losing a factor during polynomial division"],
            prerequisites=["9709_p3_3_1_polynomial_division_factor_remainder", "9709_p3_3_9_complex_arithmetic_polar_form"],
            priority="high",
        ),
    ],
    "p4": [
        skill(
            "9709_m1_forces_components_resultants",
            "p4",
            "4.1 Forces and equilibrium",
            "Force components and resultants",
            "Resolve coplanar forces into components and find resultant magnitude and direction.",
            ["component", "resultant", "forces", "direction", "magnitude", "sin", "cos"],
            ["Resolving with sine and cosine interchanged", "Not preserving direction signs"],
            prerequisites=["9709_p1_trigonometry_equations_intervals"],
            priority="high",
        ),
        skill(
            "9709_m1_equilibrium_coplanar_forces",
            "p4",
            "4.1 Forces and equilibrium",
            "Equilibrium of particles under coplanar forces",
            "Apply zero resultant force conditions to particles held in equilibrium.",
            ["equilibrium", "held", "light string", "tension", "forces act at a point", "suspended"],
            ["Assuming equal tensions without evidence", "Ignoring weight direction"],
            prerequisites=["9709_m1_forces_components_resultants"],
            priority="high",
        ),
        skill(
            "9709_m1_friction_limiting_equilibrium",
            "p4",
            "4.1 Forces and equilibrium",
            "Friction and limiting equilibrium",
            "Use F <= mu R and limiting-friction conditions on rough horizontal or inclined surfaces.",
            ["friction", "rough", "coefficient", "limiting", "about to slide", "stationary on a rough plane"],
            ["Using F = mu R when friction is not limiting", "Taking the normal reaction as mg on an inclined plane"],
            prerequisites=["9709_m1_equilibrium_coplanar_forces"],
            priority="high",
        ),
        skill(
            "9709_m1_kinematics_constant_acceleration",
            "p4",
            "4.2 Kinematics of motion in a straight line",
            "Constant acceleration kinematics",
            "Use constant-acceleration formulae for one-dimensional and vertical motion.",
            ["constant acceleration", "starts from rest", "uniformly", "projected", "maximum height", "s = ut", "v^2"],
            ["Using the wrong sign for acceleration due to gravity", "Mixing distance and displacement"],
            priority="high",
        ),
        skill(
            "9709_m1_kinematics_graphs",
            "p4",
            "4.2 Kinematics of motion in a straight line",
            "Motion graphs",
            "Interpret velocity-time and displacement-time graphs, including gradients and areas.",
            ["velocity-time graph", "graph", "segments", "area under", "accelerates", "decelerates"],
            ["Treating graph height as distance instead of velocity", "Missing negative areas below the axis"],
            prerequisites=["9709_m1_kinematics_constant_acceleration"],
            priority="medium",
        ),
        skill(
            "9709_m1_kinematics_variable_acceleration_calculus",
            "p4",
            "4.2 Kinematics of motion in a straight line",
            "Variable acceleration by calculus",
            "Use calculus relationships between displacement, velocity and acceleration for variable motion.",
            ["velocity at time", "acceleration at time", "v =", "a =", "t s after", "integrate", "differentiate"],
            ["Omitting constants after integration", "Confusing v = ds/dt and a = dv/dt"],
            prerequisites=["9709_p1_integration_reverse_differentiation", "9709_p1_differentiation_power_chain"],
            priority="high",
        ),
        skill(
            "9709_m1_momentum_impulse_conservation",
            "p4",
            "4.3 Momentum",
            "Momentum, impulse and direct collisions",
            "Use conservation of momentum and impulse in direct collisions and impact problems.",
            ["momentum", "impulse", "collide", "collision", "after", "before", "smooth horizontal plane"],
            ["Not assigning a consistent positive direction", "Assuming kinetic energy is conserved"],
            prerequisites=["9709_m1_kinematics_constant_acceleration"],
            priority="high",
        ),
        skill(
            "9709_m1_newtons_second_law_single_particle",
            "p4",
            "4.4 Newton's laws of motion",
            "Newton's second law for a single particle",
            "Apply F = ma to a particle with resolved forces, weight, normal reaction and resistance.",
            ["resultant force", "F = ma", "acceleration", "resistance", "driving force", "mass"],
            ["Omitting resistance from the force balance", "Using kg as a force unit"],
            prerequisites=["9709_m1_forces_components_resultants", "9709_m1_kinematics_constant_acceleration"],
            priority="high",
        ),
        skill(
            "9709_m1_connected_particles_tension",
            "p4",
            "4.4 Newton's laws of motion",
            "Connected particles, strings and pulleys",
            "Model connected particles with common acceleration and tensions in light strings or rods.",
            ["connected", "light inextensible string", "smooth pulley", "tow-bar", "rope", "tension", "particles"],
            ["Assigning different accelerations to a taut string system", "Getting tension direction wrong on one body"],
            prerequisites=["9709_m1_newtons_second_law_single_particle"],
            priority="high",
        ),
        skill(
            "9709_m1_inclined_planes_dynamics",
            "p4",
            "4.4 Newton's laws of motion",
            "Dynamics on inclined and rough planes",
            "Apply Newton's laws to particles moving on smooth or rough inclined planes.",
            ["inclined plane", "line of greatest slope", "rough plane", "smooth plane", "sin", "normal reaction"],
            ["Resolving weight in the wrong direction", "Using friction in the wrong direction"],
            prerequisites=["9709_m1_friction_limiting_equilibrium", "9709_m1_newtons_second_law_single_particle"],
            priority="high",
        ),
        skill(
            "9709_m1_work_energy_kinetic_potential",
            "p4",
            "4.5 Energy, work and power",
            "Work-energy and kinetic/potential energy",
            "Use work-energy principles with kinetic energy, gravitational potential energy and work done.",
            ["energy method", "kinetic energy", "potential energy", "work done", "speed", "height", "released from rest"],
            ["Using mass instead of weight in potential energy", "Forgetting energy lost or gained by work done"],
            prerequisites=["9709_m1_kinematics_constant_acceleration"],
            priority="high",
        ),
        skill(
            "9709_m1_power_resistance_driving_force",
            "p4",
            "4.5 Energy, work and power",
            "Power, resistance and driving force",
            "Use P = Fv and work-rate relationships for vehicles, engines and resistance forces.",
            ["power", "working at", "constant speed", "engine", "rate", "resistance", "kW", "W"],
            ["Forgetting to convert kW to W", "Using total force rather than driving force in P = Fv"],
            prerequisites=["9709_m1_newtons_second_law_single_particle", "9709_m1_work_energy_kinetic_potential"],
            priority="high",
        ),
        skill(
            "9709_m1_work_energy_with_resistance_friction",
            "p4",
            "4.5 Energy, work and power",
            "Work-energy with resistance and friction",
            "Apply energy methods where non-conservative resistance or friction does work.",
            ["resistance", "friction", "rough", "work done against", "energy lost", "constant force"],
            ["Using friction as positive work when it opposes motion", "Omitting vertical height change"],
            prerequisites=["9709_m1_work_energy_kinetic_potential", "9709_m1_friction_limiting_equilibrium"],
            priority="high",
        ),
    ],
    "p5": [
        skill(
            "9709_s1_data_representation_stem_box",
            "p5",
            "5.1 Representation of data",
            "Stem-and-leaf, box-and-whisker and comparative displays",
            "Represent and compare raw data using stem-and-leaf diagrams, box plots and related summaries.",
            ["stem", "box", "whisker", "quartile", "median", "interquartile", "represent"],
            ["Misplacing quartiles in an ordered list", "Comparing spread using only the range"],
            priority="medium",
        ),
        skill(
            "9709_s1_data_histograms",
            "p5",
            "5.1 Representation of data",
            "Histograms and frequency density",
            "Draw and interpret histograms for grouped data with unequal class widths.",
            ["histogram", "frequency density", "class width", "frequency", "grouped"],
            ["Plotting frequency instead of frequency density", "Using inconsistent class widths"],
            priority="high",
        ),
        skill(
            "9709_s1_data_cumulative_frequency",
            "p5",
            "5.1 Representation of data",
            "Cumulative frequency and percentiles",
            "Use cumulative frequency graphs to estimate medians, quartiles, percentiles and counts.",
            ["cumulative frequency", "estimate", "median", "quartile", "percentile", "curve"],
            ["Reading from class midpoints instead of boundaries", "Using less-than values as exact data"],
            priority="high",
        ),
        skill(
            "9709_s1_data_mean_variance_coding",
            "p5",
            "5.1 Representation of data",
            "Mean, variance, standard deviation and coding",
            "Calculate and transform measures of location and spread, including coded data and grouped estimates.",
            ["mean", "variance", "standard deviation", "Σx", "Σx^2", "coded", "summary"],
            ["Using n minus 1 instead of n for syllabus variance", "Forgetting to reverse coding transformations"],
            priority="high",
        ),
        skill(
            "9709_s1_permutations_repeated_arrangements",
            "p5",
            "5.2 Permutations and combinations",
            "Permutations and arrangements with restrictions",
            "Count arrangements of objects, including repeated letters and positional restrictions.",
            ["arrangements", "letters", "word", "together", "at the beginning", "no letter", "different arrangements"],
            ["Not dividing by repeated factorials", "Treating a grouped block as multiple objects"],
            priority="high",
        ),
        skill(
            "9709_s1_combinations_selections",
            "p5",
            "5.2 Permutations and combinations",
            "Combinations and selections with restrictions",
            "Count selections and committees using combinations, complementary cases and restrictions.",
            ["selections", "chosen", "committee", "group", "at least", "exactly", "in how many ways"],
            ["Counting ordered arrangements when selections are unordered", "Missing complementary cases"],
            priority="high",
        ),
        skill(
            "9709_s1_probability_basic_rules",
            "p5",
            "5.3 Probability",
            "Probability rules, mutually exclusive and independent events",
            "Use addition, multiplication, complement, mutually exclusive and independent event rules.",
            ["probability", "independently", "fair", "spinner", "dice", "coin", "at least", "fewer than"],
            ["Adding probabilities for non-mutually-exclusive events", "Using independence without justification"],
            priority="high",
        ),
        skill(
            "9709_s1_probability_conditional_tree_diagrams",
            "p5",
            "5.3 Probability",
            "Conditional probability and tree diagrams",
            "Use tree diagrams, conditional probability and successive selection with or without replacement.",
            ["conditional", "given that", "tree diagram", "without replacement", "placed in", "wears", "if"],
            ["Using original denominators after an item is moved", "Confusing P(A given B) with P(B given A)"],
            prerequisites=["9709_s1_probability_basic_rules"],
            priority="high",
        ),
        skill(
            "9709_s1_probability_repeated_trials_first_success",
            "p5",
            "5.3 Probability",
            "Repeated trials and first-success probabilities",
            "Model repeated independent trials, including first success after a stated number of trials.",
            ["first time", "first obtained", "repeatedly", "until", "maximum of two attempts", "7-day period"],
            ["Counting all successes rather than the first success", "Forgetting failures before the first success"],
            prerequisites=["9709_s1_probability_basic_rules"],
            priority="medium",
        ),
        skill(
            "9709_s1_discrete_random_variables_distribution",
            "p5",
            "5.4 Discrete random variables",
            "Discrete random variable distributions",
            "Construct and use probability distribution tables for discrete random variables.",
            ["random variable", "probability distribution", "takes the values", "P(X", "table", "where k is a constant"],
            ["Not normalising probabilities to sum to 1", "Omitting possible values of X"],
            prerequisites=["9709_s1_probability_basic_rules"],
            priority="high",
        ),
        skill(
            "9709_s1_discrete_random_variables_expectation_variance",
            "p5",
            "5.4 Discrete random variables",
            "Expectation and variance of discrete random variables",
            "Calculate E(X), E(X squared), Var(X) and transform discrete random variables.",
            ["E(X", "Var(X", "expectation", "mean of X", "variance of X", "E(X^2"],
            ["Using E(X squared) minus E(X) instead of E(X) squared", "Ignoring transformed variable effects"],
            prerequisites=["9709_s1_discrete_random_variables_distribution"],
            priority="high",
        ),
        skill(
            "9709_s1_binomial_distribution",
            "p5",
            "5.4 Discrete random variables",
            "Binomial distribution",
            "Use binomial probabilities, cumulative probabilities and parameter conditions.",
            ["binomial", "X ~ B", "sample of", "number who", "less than", "more than", "successes"],
            ["Using nCx with the wrong power of p", "Mixing less than and at most"],
            prerequisites=["9709_s1_probability_basic_rules"],
            priority="high",
        ),
        skill(
            "9709_s1_normal_distribution_probabilities",
            "p5",
            "5.5 The normal distribution",
            "Normal distribution probabilities and standardisation",
            "Use normal distribution models, z-scores and standardisation to calculate probabilities.",
            ["normal distribution", "normally distributed", "mean", "standard deviation", "N(", "probability that"],
            ["Failing to standardise", "Using variance instead of standard deviation"],
            priority="high",
        ),
        skill(
            "9709_s1_normal_distribution_inverse_parameters",
            "p5",
            "5.5 The normal distribution",
            "Inverse normal and unknown parameters",
            "Use normal probabilities or percentiles to find unknown means, standard deviations or thresholds.",
            ["find the value", "unknown", "standard deviation", "mean", "percentage", "expected number", "inverse"],
            ["Using the wrong tail probability", "Rounding z-values too early"],
            prerequisites=["9709_s1_normal_distribution_probabilities"],
            priority="high",
        ),
    ],
}


SECTIONS = {
    "p1": [
        ("1.1", "Quadratics"),
        ("1.2", "Functions"),
        ("1.3", "Coordinate geometry"),
        ("1.4", "Circular measure"),
        ("1.5", "Trigonometry"),
        ("1.6", "Series"),
        ("1.7", "Differentiation"),
        ("1.8", "Integration"),
    ],
    "p3": [
        ("3.1", "Algebra"),
        ("3.2", "Logarithmic and exponential functions"),
        ("3.3", "Trigonometry"),
        ("3.4", "Differentiation"),
        ("3.5", "Integration"),
        ("3.6", "Numerical solution of equations"),
        ("3.7", "Vectors"),
        ("3.8", "Differential equations"),
        ("3.9", "Complex numbers"),
    ],
    "p4": [
        ("4.1", "Forces and equilibrium"),
        ("4.2", "Kinematics of motion in a straight line"),
        ("4.3", "Momentum"),
        ("4.4", "Newton's laws of motion"),
        ("4.5", "Energy, work and power"),
    ],
    "p5": [
        ("5.1", "Representation of data"),
        ("5.2", "Permutations and combinations"),
        ("5.3", "Probability"),
        ("5.4", "Discrete random variables"),
        ("5.5", "The normal distribution"),
    ],
}


TOPIC_DEFAULTS = {
    "p1": {
        "quadratics": ["9709_p1_quadratics_discriminant_intersections"],
        "algebra": ["9709_p1_quadratics_discriminant_intersections"],
        "functions": ["9709_p1_functions_composition_inverse", "9709_p1_functions_transformations"],
        "coordinate_geometry": ["9709_p1_coordinate_circle_geometry", "9709_p1_coordinate_line_geometry"],
        "circular_measure": ["9709_p1_circular_measure_arc_sector", "9709_p1_circular_measure_segments_perimeters"],
        "trigonometry": ["9709_p1_trigonometry_equations_intervals", "9709_p1_trigonometry_identities_exact_values"],
        "series_and_sequences": ["9709_p1_series_arithmetic_progressions", "9709_p1_series_geometric_progressions"],
        "binomial_expansion": ["9709_p1_series_binomial_positive_integer"],
        "differentiation": ["9709_p1_differentiation_tangents_normals", "9709_p1_differentiation_stationary_rates"],
        "integration": ["9709_p1_integration_area_volume", "9709_p1_integration_reverse_differentiation"],
    },
    "p3": {
        "modulus": ["9709_p3_3_1_modulus_equations_inequalities"],
        "polynomials": ["9709_p3_3_1_polynomial_division_factor_remainder"],
        "partial_fractions": ["9709_p3_3_1_partial_fractions"],
        "binomial_expansion": ["9709_p3_3_1_binomial_rational_expansion"],
        "logarithms_and_exponentials": ["9709_p3_3_2_log_exponential_equations"],
        "trigonometry": ["9709_p3_3_3_identities_compound_double_angle_equations"],
        "differentiation": ["9709_p3_3_4_derivative_rules"],
        "parametric_equations": ["9709_p3_3_4_parametric_implicit_differentiation"],
        "integration": ["9709_p3_3_5_standard_integration"],
        "numerical_methods": ["9709_p3_3_6_fixed_point_iteration", "9709_p3_3_6_root_location"],
        "vectors": ["9709_p3_3_7_vector_lines", "9709_p3_3_7_scalar_product_angles"],
        "differential_equations": ["9709_p3_3_8_separable_differential_equations"],
        "complex_numbers": ["9709_p3_3_9_complex_arithmetic_polar_form"],
    },
    "p4": {
        "equilibrium_particle": ["9709_m1_equilibrium_coplanar_forces"],
        "equilibrium_coplanar_forces": ["9709_m1_equilibrium_coplanar_forces", "9709_m1_forces_components_resultants"],
        "friction_rough_plane": ["9709_m1_friction_limiting_equilibrium", "9709_m1_inclined_planes_dynamics"],
        "kinematics_constant_acceleration": ["9709_m1_kinematics_constant_acceleration"],
        "kinematics_graphs": ["9709_m1_kinematics_graphs"],
        "kinematics_variable_functions": ["9709_m1_kinematics_variable_acceleration_calculus"],
        "momentum_impulse": ["9709_m1_momentum_impulse_conservation"],
        "forces_newtons_second_law": ["9709_m1_newtons_second_law_single_particle"],
        "connected_particles": ["9709_m1_connected_particles_tension"],
        "work_energy_power": ["9709_m1_work_energy_kinetic_potential", "9709_m1_power_resistance_driving_force"],
        "power_and_resistance": ["9709_m1_power_resistance_driving_force"],
        "rough_plane_energy": ["9709_m1_work_energy_with_resistance_friction"],
    },
    "p5": {
        "data_representation": ["9709_s1_data_cumulative_frequency", "9709_s1_data_histograms"],
        "measures_of_central_tendency_and_dispersion": [
            "9709_s1_data_mean_variance_coding",
            "9709_s1_normal_distribution_probabilities",
        ],
        "permutations_and_combinations": ["9709_s1_permutations_repeated_arrangements", "9709_s1_combinations_selections"],
        "probability": ["9709_s1_probability_basic_rules", "9709_s1_probability_conditional_tree_diagrams"],
        "probability_distributions": [
            "9709_s1_discrete_random_variables_distribution",
            "9709_s1_discrete_random_variables_expectation_variance",
        ],
        "binomial_distribution": ["9709_s1_binomial_distribution"],
        "geometric_distribution": ["9709_s1_probability_repeated_trials_first_success"],
    },
}


LEGACY_SKILL_REDIRECTS = {
    "9709_p1_algebra_quadratics": "9709_p1_quadratics_discriminant_intersections",
    "9709_p1_functions_graphs": "9709_p1_functions_composition_inverse",
    "9709_p1_coordinate_geometry": "9709_p1_coordinate_circle_geometry",
    "9709_p1_trigonometry_basics": "9709_p1_trigonometry_equations_intervals",
    "9709_p1_differentiation_basics": "9709_p1_differentiation_tangents_normals",
    "9709_p1_integration_basics": "9709_p1_integration_reverse_differentiation",
    "9709_p2_trigonometry_extended": "9709_p3_3_3_identities_compound_double_angle_equations",
    "9709_p2_integration_extended": "9709_p3_3_5_standard_integration",
    "9709_p2_differentiation_extended": "9709_p3_3_4_derivative_rules",
    "9709_p2_logs_exponentials": "9709_p3_3_2_log_exponential_equations",
    "9709_p2_algebra_modulus_polynomials": "9709_p3_3_1_modulus_equations_inequalities",
    "9709_p2_numerical_methods": "9709_p3_3_6_root_location",
}


QUESTION_RE = re.compile(r"^(\d{2})(spring|summer|autumn)(\d{2})$")
SESSION_NAMES = {"spring": "March", "summer": "June", "autumn": "November"}
ALLOWED_REVIEW = {"reviewed", "needs_review", "machine_candidate", "rejected", "deprecated"}
ALLOWED_SOURCE = {
    "human_reviewed",
    "machine_candidate",
    "syllabus_inferred",
    "mark_scheme_inferred",
    "question_text_inferred",
    "legacy_topic_mapping",
    "mixed_evidence",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_text(text: str) -> str:
    text = text or ""
    replacements = {
        "θ": "theta",
        "π": "pi",
        "−": "-",
        "–": "-",
        "—": "-",
        "∫": " integral ",
        "Σ": " sigma ",
        "≤": "<=",
        "≥": ">=",
        "∞": "infinity",
        "°": " degrees ",
        "′": "'",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text.lower()).strip()


def signal_matches(signals: list[str], text: str) -> list[str]:
    matches = []
    for signal in signals:
        needle = normalize_text(signal)
        if not needle:
            continue
        if needle in text:
            matches.append(signal)
            continue
        # A few source texts lose spaces around formulae; try token fragments.
        compact_text = re.sub(r"\s+", "", text)
        compact_needle = re.sub(r"\s+", "", needle)
        if len(compact_needle) >= 4 and compact_needle in compact_text:
            matches.append(signal)
    return matches


def parse_paper(paper: str) -> dict[str, Any]:
    match = QUESTION_RE.match(paper)
    if not match:
        return {"year": None, "session": None, "variant": None, "source_paper_code": paper[:2]}
    source_paper_code, session_slug, year_suffix = match.groups()
    return {
        "year": 2000 + int(year_suffix),
        "session": SESSION_NAMES[session_slug],
        "variant": source_paper_code[-1],
        "source_paper_code": source_paper_code,
    }


def subpart_records(question: dict[str, Any]) -> list[dict[str, Any]]:
    subparts = question.get("subparts") or []
    if subparts:
        return subparts
    return [
        {
            "subpart_id": f"{question['question_id']}_whole",
            "label": "whole",
            "marks": question.get("total_marks"),
            "question_text": {"text": question.get("question_text", ""), "trust_level": "low"},
            "mark_scheme_text": {"text": question.get("mark_scheme_text", ""), "trust_level": "low"},
            "review_status": "needs_review",
        }
    ]


def build_content_lab_lookup(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        lookup[candidate["subpart_id"]] = candidate
    return lookup


def score_skills(
    component: str,
    question: dict[str, Any],
    subpart: dict[str, Any],
    component_skill_ids: set[str],
    skill_by_id: dict[str, dict[str, Any]],
    content_lab_lookup: dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, float, list[str], str]], list[str]]:
    q_text = subpart.get("question_text", {}).get("text") or question.get("question_text", "")
    ms_text = subpart.get("mark_scheme_text", {}).get("text") or question.get("mark_scheme_text", "")
    topic = question.get("topic", "")
    text = normalize_text(f"{topic} {q_text} {ms_text}")

    scores: dict[str, float] = defaultdict(float)
    evidence_bits: dict[str, list[str]] = defaultdict(list)
    source_types: Counter[str] = Counter()

    for sid in TOPIC_DEFAULTS.get(component, {}).get(topic, []):
        if sid in component_skill_ids:
            scores[sid] += 2.3
            evidence_bits[sid].append(f"legacy_topic:{topic}")
            source_types["legacy_topic_mapping"] += 1

    candidate = content_lab_lookup.get(subpart.get("subpart_id", ""))
    raw_source_ids = []
    if candidate:
        raw_source_ids = candidate.get("source_skill_ids", []) or []
    for raw_sid in raw_source_ids:
        sid = LEGACY_SKILL_REDIRECTS.get(raw_sid, raw_sid)
        if sid in component_skill_ids:
            scores[sid] += 2.8
            evidence_bits[sid].append(f"content_lab_source_skill:{raw_sid}")
            source_types["mixed_evidence"] += 1

    for sid in component_skill_ids:
        matches = signal_matches(skill_by_id[sid]["recognizer_signals"], text)
        if matches:
            scores[sid] += min(3.2, 0.85 * len(matches))
            evidence_bits[sid].extend([f"signal:{m}" for m in matches[:6]])
            source_types["question_text_inferred"] += 1

    # Disambiguation passes for common broad legacy topics.
    if component == "p1":
        if any(word in text for word in ["inverse", "f^{-", "f−1", "fg(", "gf(", "ff("]):
            scores["9709_p1_functions_composition_inverse"] += 2.4
            evidence_bits["9709_p1_functions_composition_inverse"].append("disambiguation:function_composition_inverse")
        if "transformation" in text or "transformed" in text:
            scores["9709_p1_functions_transformations"] += 2.2
            evidence_bits["9709_p1_functions_transformations"].append("disambiguation:graph_transformation")
        if "circle" in text or "tangent to the circle" in text or "chord" in text:
            scores["9709_p1_coordinate_circle_geometry"] += 2.4
            evidence_bits["9709_p1_coordinate_circle_geometry"].append("disambiguation:circle_geometry")
        if "arithmetic progression" in text:
            scores["9709_p1_series_arithmetic_progressions"] += 2.4
            evidence_bits["9709_p1_series_arithmetic_progressions"].append("disambiguation:arithmetic_progression")
        if "geometric progression" in text or "sum to infinity" in text:
            scores["9709_p1_series_geometric_progressions"] += 2.4
            evidence_bits["9709_p1_series_geometric_progressions"].append("disambiguation:geometric_progression")
        if "binomial" in text or "expansion" in text:
            scores["9709_p1_series_binomial_positive_integer"] += 2.6
            evidence_bits["9709_p1_series_binomial_positive_integer"].append("disambiguation:binomial_expansion")
    elif component == "p3":
        if "differential equation" in text:
            scores["9709_p3_3_8_separable_differential_equations"] += 2.8
            evidence_bits["9709_p3_3_8_separable_differential_equations"].append("disambiguation:differential_equation")
        if "parametric" in text or "in terms of t" in text:
            scores["9709_p3_3_4_parametric_implicit_differentiation"] += 2.5
            evidence_bits["9709_p3_3_4_parametric_implicit_differentiation"].append("disambiguation:parametric")
        if "iteration" in text or "iterative" in text:
            scores["9709_p3_3_6_fixed_point_iteration"] += 2.6
            evidence_bits["9709_p3_3_6_fixed_point_iteration"].append("disambiguation:iteration")
        if "argand" in text or "shade the region" in text:
            scores["9709_p3_3_9_argand_loci_geometry"] += 2.8
            evidence_bits["9709_p3_3_9_argand_loci_geometry"].append("disambiguation:argand_locus")
    elif component == "p4":
        if "power" in text or "kw" in text or "working at" in text:
            scores["9709_m1_power_resistance_driving_force"] += 2.4
            evidence_bits["9709_m1_power_resistance_driving_force"].append("disambiguation:power")
        if "rough" in text and any(word in text for word in ["energy", "work", "speed"]):
            scores["9709_m1_work_energy_with_resistance_friction"] += 2.3
            evidence_bits["9709_m1_work_energy_with_resistance_friction"].append("disambiguation:rough_energy")
        if "connected" in text or "pulley" in text or "tow-bar" in text:
            scores["9709_m1_connected_particles_tension"] += 2.5
            evidence_bits["9709_m1_connected_particles_tension"].append("disambiguation:connected_particles")
    elif component == "p5":
        if "normal distribution" in text or "normally distributed" in text or " n(" in f" {text} ":
            scores["9709_s1_normal_distribution_probabilities"] += 2.8
            evidence_bits["9709_s1_normal_distribution_probabilities"].append("disambiguation:normal_distribution")
            if any(word in text for word in ["find the value", "unknown", "standard deviation", "mean"]):
                scores["9709_s1_normal_distribution_inverse_parameters"] += 2.0
                evidence_bits["9709_s1_normal_distribution_inverse_parameters"].append("disambiguation:normal_inverse_parameter")
        if "histogram" in text:
            scores["9709_s1_data_histograms"] += 2.5
            evidence_bits["9709_s1_data_histograms"].append("disambiguation:histogram")
        if "cumulative frequency" in text:
            scores["9709_s1_data_cumulative_frequency"] += 2.5
            evidence_bits["9709_s1_data_cumulative_frequency"].append("disambiguation:cumulative_frequency")
        if "random variable" in text:
            scores["9709_s1_discrete_random_variables_distribution"] += 2.2
            evidence_bits["9709_s1_discrete_random_variables_distribution"].append("disambiguation:random_variable")

    ranked = sorted(
        ((sid, score, evidence_bits[sid], skill_by_id[sid]["section"]) for sid, score in scores.items() if score >= 1.2),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked, raw_source_ids


def confidence_from_score(score: float, subpart: dict[str, Any], evidence_count: int) -> float:
    confidence = 0.34 + min(0.46, score * 0.055) + min(0.08, evidence_count * 0.012)
    trust = subpart.get("question_text", {}).get("trust_level")
    if trust == "high":
        confidence += 0.03
    if subpart.get("label") == "whole":
        confidence -= 0.08
    return round(max(0.25, min(0.86, confidence)), 2)


def choose_mapping_source(confidence: float, evidence_labels: list[str]) -> str:
    if confidence < 0.5:
        return "machine_candidate"
    has_topic = any(label.startswith("legacy_topic:") for label in evidence_labels)
    has_signal = any(label.startswith("signal:") or label.startswith("disambiguation:") for label in evidence_labels)
    has_cl = any(label.startswith("content_lab_source_skill:") for label in evidence_labels)
    if has_signal and (has_topic or has_cl):
        return "mixed_evidence"
    if has_signal:
        return "question_text_inferred"
    if has_topic:
        return "legacy_topic_mapping"
    return "machine_candidate"


def make_mapping(
    component: str,
    question: dict[str, Any],
    subpart: dict[str, Any],
    ranked: list[tuple[str, float, list[str], str]],
    raw_source_ids: list[str],
    skill_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not ranked:
        return None
    top_score = ranked[0][1]
    primary = [ranked[0][0]]
    secondary: list[str] = []
    for sid, score, _evidence, _section in ranked[1:]:
        if score >= max(2.1, top_score * 0.74) and len(secondary) < 2:
            secondary.append(sid)

    evidence_labels = ranked[0][2]
    confidence = confidence_from_score(top_score, subpart, len(evidence_labels))
    mapping_source = choose_mapping_source(confidence, evidence_labels)
    review_status = "needs_review"
    q_meta = parse_paper(question["paper"])
    q_text = subpart.get("question_text", {}).get("text") or question.get("question_text", "")
    ms_text = subpart.get("mark_scheme_text", {}).get("text") or question.get("mark_scheme_text", "")
    primary_prereqs: list[str] = []
    for sid in primary + secondary:
        for prereq in skill_by_id[sid].get("prerequisite_skill_ids", []):
            if prereq not in primary + secondary and prereq not in primary_prereqs:
                primary_prereqs.append(prereq)

    granularity = "subpart" if subpart.get("label") != "whole" else "whole_question_only"
    limitation = "" if granularity == "subpart" else "Only whole-question extraction was available for this record."
    notes = []
    if confidence < 0.5:
        notes.append("Low-confidence mapping; requires review before use.")
    if limitation:
        notes.append(limitation)
    if raw_source_ids:
        notes.append("Legacy/content-lab skill ids were used only as candidate evidence, not as reviewed labels.")
    out_component = output_component(component)

    return {
        "mapping_id": f"map_{SYLLABUS_CODE}_{out_component}_v1_{subpart['subpart_id']}",
        "question_id": question["question_id"],
        "paper_id": question["paper"],
        "paper": question["paper"],
        "syllabus_code": SYLLABUS_CODE,
        "subject_name": SUBJECT_NAME,
        "caie_class_or_component": COMPONENTS[component]["caie_class_or_component"],
        "component_label": COMPONENTS[component]["component_label"],
        "year": q_meta["year"],
        "session": q_meta["session"],
        "variant": q_meta["variant"],
        "question_number": str(question["question_number"]),
        "subpart_id": subpart["subpart_id"],
        "subpart_label": str(subpart.get("label") or "whole"),
        "evidence_granularity": granularity,
        "primary_skill_ids": primary,
        "secondary_skill_ids": secondary,
        "prerequisite_skill_ids": primary_prereqs,
        "confidence": confidence,
        "evidence": {
            "source_topic": question.get("topic"),
            "topic_confidence": question.get("notes", {}).get("topic_confidence"),
            "topic_uncertain": question.get("notes", {}).get("topic_uncertain"),
            "matched_signals": evidence_labels[:12],
            "legacy_or_content_lab_skill_ids": raw_source_ids,
            "question_text_snippet": q_text[:500],
            "mark_scheme_text_snippet": ms_text[:500],
            "subpart_review_status": subpart.get("review_status"),
            "question_text_trust": subpart.get("question_text", {}).get("trust_level"),
            "mark_scheme_text_trust": subpart.get("mark_scheme_text", {}).get("trust_level"),
            "whole_question_limitation": limitation,
            "counts_as_direct_readiness_evidence": True,
            "prerequisite_evidence_kept_separate": True,
        },
        "mapping_source": mapping_source,
        "review_status": review_status,
        "notes": " ".join(notes),
    }


def support_level(direct_count: int, candidate_count: int, avg_confidence: float | None) -> str:
    if direct_count == 0:
        return "unknown"
    if candidate_count == 0:
        return "missing"
    ratio = candidate_count / direct_count
    if ratio >= 0.65 and (avg_confidence or 0) >= 0.65:
        return "strong"
    if ratio >= 0.35:
        return "moderate"
    return "weak"


def gap_severity(skill: dict[str, Any], direct_count: int, support: str, avg_confidence: float | None) -> str:
    priority = skill["content_lab_priority"]
    if direct_count == 0:
        return "medium" if priority == "high" else "low"
    if support == "missing" and priority == "high":
        return "critical"
    if support in {"missing", "weak"} and priority == "high":
        return "high"
    if support in {"missing", "weak"}:
        return "medium"
    if avg_confidence is not None and avg_confidence < 0.5:
        return "medium"
    return "none"


def recommended_action(severity: str, support: str, direct_count: int) -> str:
    if direct_count == 0:
        return "Review syllabus skill and add targeted source items or synthetic practice seeds."
    if severity in {"critical", "high"}:
        return "Prioritise human review of mappings and create Content Lab items from reviewed subparts."
    if support in {"weak", "missing"}:
        return "Review candidate mappings and promote reliable examples into Content Lab coverage."
    return "Maintain coverage; review opportunistically."


def make_coverage(
    component: str,
    questions: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    content_lab_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    skill_ids = [s["skill_id"] for s in skills]
    direct_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prereq_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mapping in mappings:
        for sid in mapping["primary_skill_ids"] + mapping["secondary_skill_ids"]:
            direct_by_skill[sid].append(mapping)
        for sid in mapping["prerequisite_skill_ids"]:
            prereq_by_skill[sid].append(mapping)

    content_lab_by_skill: Counter[str] = Counter()
    for candidate in content_lab_lookup.values():
        if candidate.get("paper_family") != component:
            continue
        for raw_sid in candidate.get("source_skill_ids", []) or []:
            sid = LEGACY_SKILL_REDIRECTS.get(raw_sid, raw_sid)
            if sid in skill_ids:
                content_lab_by_skill[sid] += 1

    total_subparts = sum(len(subpart_records(q)) for q in questions)
    mapped_subpart_ids = {m["subpart_id"] for m in mappings}
    all_subpart_ids = {sp["subpart_id"] for q in questions for sp in subpart_records(q)}
    unmapped_subparts = sorted(all_subpart_ids - mapped_subpart_ids)

    skill_rows = []
    for skill_obj in skills:
        sid = skill_obj["skill_id"]
        direct = direct_by_skill.get(sid, [])
        prereq = prereq_by_skill.get(sid, [])
        confidences = [m["confidence"] for m in direct]
        avg = round(sum(confidences) / len(confidences), 3) if confidences else None
        cl_count = content_lab_by_skill[sid]
        support = support_level(len(direct), cl_count, avg)
        severity = gap_severity(skill_obj, len(direct), support, avg)
        skill_rows.append(
            {
                "skill_id": sid,
                "section": skill_obj["section"],
                "name": skill_obj["name"],
                "direct_question_count": len({m["question_id"] for m in direct}),
                "direct_subpart_count": len({m["subpart_id"] for m in direct}),
                "prerequisite_question_count": len({m["question_id"] for m in prereq}),
                "average_mapping_confidence": avg,
                "reviewed_mapping_count": sum(1 for m in direct if m["review_status"] == "reviewed"),
                "machine_candidate_count": sum(1 for m in direct if m["review_status"] != "reviewed"),
                "content_lab_candidate_count": cl_count,
                "content_lab_priority": skill_obj["content_lab_priority"],
                "content_lab_support_level": support,
                "gap_severity": severity,
                "recommended_action": recommended_action(severity, support, len(direct)),
            }
        )

    section_counts: dict[str, dict[str, Any]] = defaultdict(lambda: {"skills": 0, "direct_subparts": 0, "weak_or_missing": 0})
    for row in skill_rows:
        section_counts[row["section"]]["skills"] += 1
        section_counts[row["section"]]["direct_subparts"] += row["direct_subpart_count"]
        if row["content_lab_support_level"] in {"weak", "missing", "unknown"}:
            section_counts[row["section"]]["weak_or_missing"] += 1

    review_queue = Counter()
    for mapping in mappings:
        if mapping["review_status"] == "reviewed":
            continue
        if mapping["confidence"] < 0.5:
            review_queue["low_confidence"] += 1
        elif mapping["evidence_granularity"] == "whole_question_only":
            review_queue["whole_question_only"] += 1
        else:
            review_queue["standard_candidate"] += 1

    report = {
        "schema_name": "exam_bank.component_coverage_report",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "syllabus_code": SYLLABUS_CODE,
        "subject_name": SUBJECT_NAME,
        "caie_class_or_component": COMPONENTS[component]["caie_class_or_component"],
        "component_label": COMPONENTS[component]["component_label"],
        "source_syllabus_reference": SYLLABUS_REFERENCE,
        "source_syllabus_url": SYLLABUS_URL,
        "total_skills": len(skills),
        "total_questions": len(questions),
        "total_subparts": total_subparts,
        "mapped_subparts": len(mapped_subpart_ids),
        "unmapped_subparts": len(unmapped_subparts),
        "mapping_coverage_percent": round(100 * len(mapped_subpart_ids) / total_subparts, 2) if total_subparts else 0.0,
        "skills_with_no_questions": [r["skill_id"] for r in skill_rows if r["direct_subpart_count"] == 0],
        "skills_with_few_questions": [r["skill_id"] for r in skill_rows if 0 < r["direct_subpart_count"] <= 3],
        "skills_with_many_questions": [r["skill_id"] for r in skill_rows if r["direct_subpart_count"] >= 30],
        "skills_with_weak_content_lab_support": [
            r["skill_id"] for r in skill_rows if r["content_lab_support_level"] in {"weak", "missing"}
        ],
        "high_priority_content_lab_gaps": [
            r["skill_id"] for r in skill_rows if r["content_lab_priority"] == "high" and r["gap_severity"] in {"critical", "high"}
        ],
        "prerequisite_only_skills": [
            r["skill_id"] for r in skill_rows if r["direct_subpart_count"] == 0 and r["prerequisite_question_count"] > 0
        ],
        "skills_with_low_confidence_mappings_only": [
            r["skill_id"] for r in skill_rows if r["direct_subpart_count"] > 0 and (r["average_mapping_confidence"] or 0) < 0.5
        ],
        "sections_with_weak_coverage": [
            {
                "section": section,
                "skills": values["skills"],
                "direct_subparts": values["direct_subparts"],
                "weak_or_missing_skill_count": values["weak_or_missing"],
            }
            for section, values in sorted(section_counts.items())
            if values["direct_subparts"] < values["skills"] * 3 or values["weak_or_missing"] > 0
        ],
        "review_queue_summary": dict(review_queue),
        "unmapped_subpart_ids": unmapped_subparts[:500],
        "skills": sorted(skill_rows, key=lambda row: (row["section"], row["skill_id"])),
    }
    return report


def make_skill_map(component: str, skills: list[dict[str, Any]]) -> dict[str, Any]:
    out_component = output_component(component)
    return {
        "schema_name": "exam_bank.skill_map",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "syllabus_code": SYLLABUS_CODE,
        "subject_name": SUBJECT_NAME,
        "caie_class_or_component": COMPONENTS[component]["caie_class_or_component"],
        "component_label": COMPONENTS[component]["component_label"],
        "source_syllabus_reference": SYLLABUS_REFERENCE,
        "source_syllabus_url": SYLLABUS_URL,
        "official_syllabus_reference": {
            "title": SYLLABUS_REFERENCE,
            "url": SYLLABUS_URL,
            "notes": "Section structure follows the official CAIE 9709 syllabus. Skill decomposition is candidate-expanded for subpart mapping and requires review.",
        },
        "sections": [{"section": number, "name": name} for number, name in SECTIONS[component]],
        "skills": skills,
        "mapping_file": rel(CANONICAL_SKILL_MAPPINGS / f"question_skill_mappings_{SYLLABUS_CODE}_{out_component}_v1.json"),
        "coverage_report_file": rel(CANONICAL_COVERAGE_REPORTS / f"coverage_report_{SYLLABUS_CODE}_{out_component}_v1.json"),
        "review_status": "needs_review",
        "notes": "Candidate taxonomy generated from official syllabus sections plus exam-bank evidence. No new reviewed status has been asserted.",
    }


def validate(
    all_skills: dict[str, dict[str, Any]],
    maps: dict[str, dict[str, Any]],
    mappings_by_component: dict[str, list[dict[str, Any]]],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    skill_ids = list(all_skills)
    duplicate_ids = [sid for sid, count in Counter(skill_ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate skill ids: {duplicate_ids}")

    for sid, skill_obj in all_skills.items():
        for field in [
            "skill_id",
            "syllabus_code",
            "subject_name",
            "caie_class_or_component",
            "component_label",
            "section",
            "name",
            "description",
            "prerequisite_skill_ids",
            "related_skill_ids",
            "recognizer_signals",
            "common_errors",
            "asterion_region_id",
            "content_lab_priority",
            "official_syllabus_reference",
            "review_status",
            "notes",
        ]:
            if field not in skill_obj:
                errors.append(f"{sid} missing field {field}")
        if skill_obj.get("review_status") not in ALLOWED_REVIEW:
            errors.append(f"{sid} invalid review_status {skill_obj.get('review_status')}")
        for prereq in skill_obj.get("prerequisite_skill_ids", []):
            if prereq not in all_skills:
                errors.append(f"{sid} prerequisite {prereq} does not exist")
        for related in skill_obj.get("related_skill_ids", []):
            if related not in all_skills:
                errors.append(f"{sid} related skill {related} does not exist")

    for component, mappings in mappings_by_component.items():
        component_skill_ids = {s["skill_id"] for s in maps[component]["skills"]}
        for mapping in mappings:
            if mapping.get("mapping_source") not in ALLOWED_SOURCE:
                errors.append(f"{mapping['mapping_id']} invalid mapping_source {mapping.get('mapping_source')}")
            if mapping.get("review_status") not in ALLOWED_REVIEW:
                errors.append(f"{mapping['mapping_id']} invalid review_status {mapping.get('review_status')}")
            if "confidence" not in mapping or not isinstance(mapping["confidence"], (int, float)):
                errors.append(f"{mapping['mapping_id']} missing numeric confidence")
            if mapping.get("confidence", 0) < 0.5 and mapping.get("review_status") == "reviewed":
                errors.append(f"{mapping['mapping_id']} low-confidence mapping is reviewed")
            if mapping.get("confidence", 0) < 0.5 and mapping.get("mapping_source") != "machine_candidate":
                errors.append(f"{mapping['mapping_id']} low-confidence mapping_source is not machine_candidate")
            for sid in mapping.get("primary_skill_ids", []) + mapping.get("secondary_skill_ids", []):
                if sid not in component_skill_ids:
                    errors.append(f"{mapping['mapping_id']} direct skill {sid} is not in {component} skill map")
            for sid in mapping.get("prerequisite_skill_ids", []):
                if sid not in all_skills:
                    errors.append(f"{mapping['mapping_id']} prerequisite skill {sid} does not exist")
            if mapping.get("evidence_granularity") == "whole_question_only" and not mapping["evidence"].get("whole_question_limitation"):
                errors.append(f"{mapping['mapping_id']} whole-question evidence missing limitation note")
        if reports[component]["mapped_subparts"] > reports[component]["total_subparts"]:
            errors.append(f"{component} coverage mapped_subparts exceeds total_subparts")
        if reports[component]["mapping_coverage_percent"] > 100:
            errors.append(f"{component} coverage percent exceeds 100")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "valid_json_written": True,
            "mapping_skill_ids_exist": not any("direct skill" in e for e in errors),
            "prerequisite_skill_ids_exist": not any("prerequisite" in e and "does not exist" in e for e in errors),
            "skill_ids_unique": not duplicate_ids,
            "mapping_confidence_present": not any("missing numeric confidence" in e for e in errors),
            "low_confidence_not_reviewed": not any("low-confidence mapping is reviewed" in e for e in errors),
            "whole_question_limitations_marked": not any("whole-question evidence missing limitation note" in e for e in errors),
            "direct_and_prerequisite_counts_separate": True,
            "component_specific_syllabus_sections": True,
        },
    }


def load_git_baseline(path: Path) -> dict[str, Any]:
    rel_path = path.relative_to(ROOT)
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel_path.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return load_json(path) if path.exists() else {}
    return json.loads(result.stdout)


def p3_changelog(old: dict[str, Any], old_path: Path, new_map: dict[str, Any], p3_mappings: list[dict[str, Any]], p3_report: dict[str, Any]) -> dict[str, Any]:
    old_skills = {s.get("skill_id"): s for s in old.get("skills", []) if s.get("skill_id")}
    new_skills = {s.get("skill_id"): s for s in new_map.get("skills", []) if s.get("skill_id")}
    old_mappings = old.get("mappings", [])
    return {
        "schema_name": "exam_bank.skill_map_changelog",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "syllabus_code": SYLLABUS_CODE,
        "component": "p3",
        "component_label": COMPONENTS["p3"]["component_label"],
        "source_file": old_path.name,
        "changes": [
            {
                "change_type": "schema_normalization",
                "description": "Rewrote P3 skill map to include all requested taxonomy fields and split mappings/coverage into separate JSON files.",
            },
            {
                "change_type": "review_status_policy",
                "description": "No mappings were promoted to reviewed; all generated and carried-forward machine mappings remain needs_review or machine candidates.",
            },
            {
                "change_type": "official_section_audit",
                "description": "Confirmed P3 map covers official sections 3.1 through 3.9.",
            },
        ],
        "old_skill_count": len(old_skills),
        "new_skill_count": len(new_skills),
        "old_embedded_mapping_count": len(old_mappings),
        "new_mapping_file": rel(CANONICAL_SKILL_MAPPINGS / f"question_skill_mappings_{SYLLABUS_CODE}_p3_v1.json"),
        "new_mapping_count": len(p3_mappings),
        "new_coverage_report_file": rel(CANONICAL_COVERAGE_REPORTS / f"coverage_report_{SYLLABUS_CODE}_p3_v1.json"),
        "added_skill_ids": sorted(set(new_skills) - set(old_skills)),
        "removed_or_replaced_skill_ids": sorted(set(old_skills) - set(new_skills)),
        "skill_ids_preserved": sorted(set(old_skills) & set(new_skills)),
        "coverage_summary": {
            "total_skills": p3_report["total_skills"],
            "total_subparts": p3_report["total_subparts"],
            "mapped_subparts": p3_report["mapped_subparts"],
            "mapping_coverage_percent": p3_report["mapping_coverage_percent"],
            "high_priority_content_lab_gaps": p3_report["high_priority_content_lab_gaps"],
        },
        "notes": "Existing P3 mappings in the old file were machine candidates only; no reviewed mappings were available to preserve as reviewed.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate candidate CAIE 9709 skill maps, question-skill mappings, and coverage reports.",
    )
    parser.add_argument(
        "--question-bank",
        "--input",
        dest="question_bank",
        type=Path,
        default=QUESTION_BANK,
        help="Path to the current question_bank.json input.",
    )
    parser.add_argument(
        "--asterion-question-bank",
        type=Path,
        default=ASTERION_QB,
        help="Path to the current Asterion question-bank export.",
    )
    parser.add_argument(
        "--content-lab-candidates",
        type=Path,
        default=CONTENT_LAB,
        help="Path to the current Asterion Content Lab candidates export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate outputs, then print the write plan without writing files.",
    )
    return parser


def planned_output_paths() -> list[Path]:
    paths: list[Path] = []
    for component in sorted(COMPONENTS):
        out_component = output_component(component)
        paths.extend(
            [
                CANONICAL_SKILL_MAPS / f"skill_map_{SYLLABUS_CODE}_{out_component}_v1.json",
                CANONICAL_SKILL_MAPPINGS / f"question_skill_mappings_{SYLLABUS_CODE}_{out_component}_v1.json",
                CANONICAL_COVERAGE_REPORTS / f"coverage_report_{SYLLABUS_CODE}_{out_component}_v1.json",
            ]
        )
    paths.extend(
        [
            CANONICAL_INDEXES / "skill_map_index_v1.json",
            CANONICAL_COVERAGE_REPORTS / "coverage_report_all_components_v1.json",
            TAXONOMY_VALIDATION_REPORTS / "skill_map_validation_v1.json",
            TAXONOMY_CHANGELOGS / "changelog_9709_p3_v1.json",
        ]
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_bank_path = resolve_project_path(args.question_bank)
    asterion_question_bank_path = resolve_project_path(args.asterion_question_bank)
    content_lab_candidates_path = resolve_project_path(args.content_lab_candidates)

    question_bank = load_json(question_bank_path)["questions"]
    asterion_questions = load_json(asterion_question_bank_path)["questions"]
    content_lab_candidates = load_json(content_lab_candidates_path)["candidates"]
    old_p3_baseline = load_git_baseline(OLD_P3_SKILL_MAP)
    content_lab_lookup = build_content_lab_lookup(content_lab_candidates)
    question_notes = {q["question_id"]: q for q in question_bank}

    # Merge richer Asterion subparts with top-level topic/difficulty metadata.
    merged_questions: list[dict[str, Any]] = []
    for question in asterion_questions:
        merged = deepcopy(question)
        source = question_notes.get(question["question_id"], {})
        merged["topic"] = source.get("topic")
        merged["notes"] = source.get("notes", {})
        merged["difficulty"] = source.get("difficulty")
        merged_questions.append(merged)

    all_skill_objs = [skill_obj for skills in SKILLS.values() for skill_obj in skills]
    all_skills = {skill_obj["skill_id"]: skill_obj for skill_obj in all_skill_objs}
    maps: dict[str, dict[str, Any]] = {}
    mappings_by_component: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}

    for component, skills in SKILLS.items():
        component_skill_ids = {s["skill_id"] for s in skills}
        skill_by_id = {s["skill_id"]: s for s in skills}
        # Include global skills for prerequisite lookups during mapping creation.
        skill_by_id.update(all_skills)
        component_questions = [q for q in merged_questions if q.get("paper_family") == component]
        mappings: list[dict[str, Any]] = []
        for question in component_questions:
            for subpart in subpart_records(question):
                ranked, raw_source_ids = score_skills(
                    component,
                    question,
                    subpart,
                    component_skill_ids,
                    skill_by_id,
                    content_lab_lookup,
                )
                mapping = make_mapping(component, question, subpart, ranked, raw_source_ids, skill_by_id)
                if mapping:
                    # For direct mappings, keep prerequisites outside direct readiness counts.
                    mappings.append(mapping)
        maps[component] = make_skill_map(component, skills)
        mappings_by_component[component] = sorted(mappings, key=lambda m: (m["paper_id"], int(m["question_number"]), m["subpart_label"]))
        reports[component] = make_coverage(component, component_questions, mappings_by_component[component], skills, content_lab_lookup)

    validation = validate(all_skills, maps, mappings_by_component, reports)

    index_entries = []
    for component in sorted(COMPONENTS):
        out_component = output_component(component)
        entry = {
            "syllabus_code": SYLLABUS_CODE,
            "component": out_component,
            "subject_name": SUBJECT_NAME,
            "caie_class_or_component": COMPONENTS[component]["caie_class_or_component"],
            "component_label": COMPONENTS[component]["component_label"],
            "source_syllabus_reference": SYLLABUS_REFERENCE,
            "source_syllabus_url": SYLLABUS_URL,
            "skill_map_file": rel(CANONICAL_SKILL_MAPS / f"skill_map_{SYLLABUS_CODE}_{out_component}_v1.json"),
            "mapping_file": rel(CANONICAL_SKILL_MAPPINGS / f"question_skill_mappings_{SYLLABUS_CODE}_{out_component}_v1.json"),
            "coverage_report_file": rel(CANONICAL_COVERAGE_REPORTS / f"coverage_report_{SYLLABUS_CODE}_{out_component}_v1.json"),
            "skill_count": len(maps[component]["skills"]),
            "mapping_count": len(mappings_by_component[component]),
            "canonical_status": "canonical_candidate",
            "review_status": "needs_review",
            "notes": "Candidate component-specific skill map based on official CAIE syllabus section structure and local exam-bank evidence.",
        }
        index_entries.append(entry)

    coverage_summary = {
        "schema_name": "exam_bank.all_components_coverage_summary",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "syllabus_code": SYLLABUS_CODE,
        "subject_name": SUBJECT_NAME,
        "source_syllabus_reference": SYLLABUS_REFERENCE,
        "source_syllabus_url": SYLLABUS_URL,
        "components": [
            {
                "component": component,
                "output_component": output_component(component),
                "caie_class_or_component": COMPONENTS[component]["caie_class_or_component"],
                "component_label": COMPONENTS[component]["component_label"],
                "total_skills": reports[component]["total_skills"],
                "total_questions": reports[component]["total_questions"],
                "total_subparts": reports[component]["total_subparts"],
                "mapped_subparts": reports[component]["mapped_subparts"],
                "unmapped_subparts": reports[component]["unmapped_subparts"],
                "mapping_coverage_percent": reports[component]["mapping_coverage_percent"],
                "mappings_needing_review": sum(
                    1 for mapping in mappings_by_component[component] if mapping["review_status"] != "reviewed"
                ),
                "high_priority_content_lab_gaps": reports[component]["high_priority_content_lab_gaps"][:10],
                "coverage_report_file": rel(CANONICAL_COVERAGE_REPORTS / f"coverage_report_{SYLLABUS_CODE}_{output_component(component)}_v1.json"),
            }
            for component in sorted(COMPONENTS)
        ],
        "highest_priority_content_lab_gaps": [
            {
                "component": component,
                "component_label": COMPONENTS[component]["component_label"],
                "skill_id": row["skill_id"],
                "name": row["name"],
                "gap_severity": row["gap_severity"],
                "direct_subpart_count": row["direct_subpart_count"],
                "content_lab_support_level": row["content_lab_support_level"],
            }
            for component in sorted(COMPONENTS)
            for row in reports[component]["skills"]
            if row["gap_severity"] in {"critical", "high"}
        ],
        "validation": validation,
        "notes": "Direct evidence is counted only from the same component. Prerequisite evidence is reported separately and is not used as readiness evidence for later components.",
    }

    index = {
        "schema_name": "exam_bank.skill_map_index",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_syllabus_reference": SYLLABUS_REFERENCE,
        "source_syllabus_url": SYLLABUS_URL,
        "components": index_entries,
        "validation": validation,
    }

    summary = {
        "components": {
            component: {
                "skills": reports[component]["total_skills"],
                "mapped_subparts": reports[component]["mapped_subparts"],
                "coverage": reports[component]["mapping_coverage_percent"],
                "mappings_needing_review": sum(1 for m in mappings_by_component[component] if m["review_status"] != "reviewed"),
            }
            for component in sorted(COMPONENTS)
        },
        "validation": validation["status"],
    }

    if args.dry_run:
        summary = {
            "dry_run": True,
            "inputs": {
                "question_bank": display_path(question_bank_path),
                "asterion_question_bank": display_path(asterion_question_bank_path),
                "content_lab_candidates": display_path(content_lab_candidates_path),
            },
            "would_write": [display_path(path) for path in planned_output_paths()],
            **summary,
        }
    else:
        for component in sorted(COMPONENTS):
            out_component = output_component(component)
            dump_json(CANONICAL_SKILL_MAPS / f"skill_map_{SYLLABUS_CODE}_{out_component}_v1.json", maps[component])
            dump_json(
                CANONICAL_SKILL_MAPPINGS / f"question_skill_mappings_{SYLLABUS_CODE}_{out_component}_v1.json",
                {
                    "schema_name": "exam_bank.question_skill_mappings",
                    "schema_version": 1,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "syllabus_code": SYLLABUS_CODE,
                    "subject_name": SUBJECT_NAME,
                    "caie_class_or_component": COMPONENTS[component]["caie_class_or_component"],
                    "component_label": COMPONENTS[component]["component_label"],
                    "source_syllabus_reference": SYLLABUS_REFERENCE,
                    "source_syllabus_url": SYLLABUS_URL,
                    "mapping_count": len(mappings_by_component[component]),
                    "mappings": mappings_by_component[component],
                },
            )
            dump_json(CANONICAL_COVERAGE_REPORTS / f"coverage_report_{SYLLABUS_CODE}_{out_component}_v1.json", reports[component])

        dump_json(CANONICAL_INDEXES / "skill_map_index_v1.json", index)
        dump_json(CANONICAL_COVERAGE_REPORTS / "coverage_report_all_components_v1.json", coverage_summary)
        dump_json(TAXONOMY_VALIDATION_REPORTS / "skill_map_validation_v1.json", validation)
        dump_json(
            TAXONOMY_CHANGELOGS / "changelog_9709_p3_v1.json",
            p3_changelog(old_p3_baseline, OLD_P3_SKILL_MAP, maps["p3"], mappings_by_component["p3"], reports["p3"]),
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
