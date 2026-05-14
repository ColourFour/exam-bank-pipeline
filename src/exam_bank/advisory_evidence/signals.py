from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from statistics import median
from typing import Any

from exam_bank.advisory_evidence.common import dedupe_preserve_order, load_json, rel_path, utc_now_iso
from exam_bank.advisory_evidence.constants import (
    EXAMINER_DIFFICULTY_PATH,
    EXAMINER_DIFFICULTY_SCHEMA,
    EXAMINER_LINKS_PATH,
    EXAMINER_PARSED_DIR,
    GRADE_THRESHOLD_CONTEXT_PATH,
    GRADE_THRESHOLD_CONTEXT_SCHEMA,
    GRADE_THRESHOLD_PARSED_DIR,
    TOPIC_EVIDENCE_PATH,
    TOPIC_EVIDENCE_SCHEMA,
)
from exam_bank.advisory_evidence.linking import normalized_question_key
from exam_bank.atomic_json import write_atomic_json
from exam_bank.config import AppConfig


TOPIC_RULES: list[tuple[str, str, str]] = [
    ("complex_numbers", r"\bcomplex numbers?\b|argand|modulus|argument|polar form"),
    ("vectors", r"\bvectors?\b|scalar product|position vector"),
    ("integration", r"integration by parts|volume of revolution|integrat(?:e|ing|ion)"),
    ("numerical_methods", r"numerical methods?|iteration|newton[- ]raphson"),
    ("implicit_differentiation", r"implicit differentiation|differentiat(?:e|ing).{0,40}implicitly"),
    ("differential_equations", r"differential equations?|separat(?:e|ing) variables"),
    ("binomial_expansion", r"binomial expansion|ascending powers|coefficient of"),
    ("trigonometry", r"trig(?:onometric)? equations?|sine|cosine|\bsin\b|\bcos\b|\btan\b"),
    ("partial_fractions", r"partial fractions?"),
    ("parametric_equations", r"parametric equations?|parameter"),
    ("quadratics", r"quadratic|discriminant|completing the square"),
    ("coordinate_geometry", r"circle|tangent|normal|gradient"),
    ("series_and_sequences", r"arithmetic progression|geometric progression|\bAP\b|\bGP\b"),
    ("differentiation", r"differentiat(?:e|ing|ion)|dy\s*/\s*dx"),
    ("probability", r"\bprobability\b"),
    ("binomial_distribution", r"binomial distribution"),
    ("normal_distribution", r"normal distribution|standardi[sz]"),
    ("hypothesis_testing", r"hypothesis test|significance level|critical region"),
    ("kinematics_constant_acceleration", r"constant acceleration|suvat"),
    ("forces_newtons_second_law", r"newton'?s second law|resultant force"),
    ("equilibrium_coplanar_forces", r"resolve forces?|equilibrium"),
    ("friction_rough_plane", r"friction|rough plane"),
    ("momentum_impulse", r"momentum|impulse|collision"),
    ("work_energy_power", r"work energy|kinetic energy|potential energy|power"),
]

HARD_PATTERNS = [
    r"challenging",
    r"proved difficult",
    r"few (?:fully )?correct",
    r"many omitted",
    r"high proportion blank",
    r"good discriminator",
    r"only (?:the )?strongest candidates",
    r"commonly gained no response",
    r"generally unsuccessful",
]
EASY_PATTERNS = [
    r"well answered",
    r"accessible",
    r"straightforward",
    r"most candidates (?:were )?able",
    r"many candidates scored full marks",
    r"generally able",
]


def build_topic_evidence(
    *,
    parsed_dir: str | Path = EXAMINER_PARSED_DIR,
    links_path: str | Path = EXAMINER_LINKS_PATH,
    output_path: str | Path = TOPIC_EVIDENCE_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    parsed_by_key = _examiner_comments_by_key(parsed_dir)
    links = load_json(links_path, default={"links": []}).get("links", [])
    allowed = _allowed_topics_by_family(AppConfig())
    records: list[dict[str, Any]] = []
    for link in links:
        if link.get("match_status") != "linked":
            continue
        question_id = link["candidate_question_ids"][0]
        comment = parsed_by_key.get(str(link.get("normalized_key")), "")
        matches = _topic_matches(comment, _family_for_component(str(link.get("component") or "")), allowed)
        if not matches:
            continue
        records.append(
            {
                "question_id": question_id,
                "normalized_key": link.get("normalized_key", ""),
                "topic_evidence": {
                    "predicted_topic_ids": [match["topic_id"] for match in matches],
                    "matched_terms": dedupe_preserve_order(term for match in matches for term in match["matched_terms"]),
                    "method": "rule_match_v1",
                    "confidence": "high" if len(matches) == 1 else "medium",
                    "review_required": len(matches) > 1,
                    "warnings": [],
                },
            }
        )
    payload = {"schema": TOPIC_EVIDENCE_SCHEMA, "generated_at": utc_now_iso(), "records": records, "warnings": []}
    if not dry_run:
        write_atomic_json(payload, output_path)
    return payload


def build_examiner_difficulty(
    *,
    parsed_dir: str | Path = EXAMINER_PARSED_DIR,
    links_path: str | Path = EXAMINER_LINKS_PATH,
    output_path: str | Path = EXAMINER_DIFFICULTY_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    parsed_by_key = _examiner_comments_by_key(parsed_dir)
    parsed_levels = _examiner_levels_by_key(parsed_dir)
    links = load_json(links_path, default={"links": []}).get("links", [])
    records: list[dict[str, Any]] = []
    for link in links:
        if link.get("match_status") != "linked":
            continue
        question_id = link["candidate_question_ids"][0]
        normalized_key = str(link.get("normalized_key") or "")
        comment = parsed_by_key.get(normalized_key, "")
        evidence_level = parsed_levels.get(normalized_key, str(link.get("evidence_level") or "normal"))
        signal, matched, reasons, confidence = _difficulty_signal(comment, evidence_level)
        records.append(
            {
                "question_id": question_id,
                "normalized_key": normalized_key,
                "examiner_report_difficulty": {
                    "item_signal": signal,
                    "matched_terms": matched,
                    "difficulty_reasons": reasons,
                    "method": "examiner_report_phrase_rules_v1",
                    "confidence": confidence,
                    "review_required": signal in {"hard", "mixed", "unknown"} or confidence != "high",
                    "evidence_level": evidence_level,
                    "evidence_sources": ["examiner_report"],
                    "warnings": ["low_or_no_evidence"] if evidence_level in {"low", "none"} else [],
                },
            }
        )
    payload = {"schema": EXAMINER_DIFFICULTY_SCHEMA, "generated_at": utc_now_iso(), "records": records, "warnings": []}
    if not dry_run:
        write_atomic_json(payload, output_path)
    return payload


def build_grade_threshold_context(
    *,
    parsed_dir: str | Path = GRADE_THRESHOLD_PARSED_DIR,
    output_path: str | Path = GRADE_THRESHOLD_CONTEXT_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    parsed_dir = Path(parsed_dir)
    contexts: list[dict[str, Any]] = []
    for path in sorted(parsed_dir.glob("*.json")) if parsed_dir.exists() else []:
        parsed = load_json(path)
        for component in parsed.get("components", []):
            thresholds = component.get("thresholds") if isinstance(component.get("thresholds"), dict) else {}
            max_raw = component.get("max_raw_mark")
            warnings = list(component.get("warnings", []))
            ratios = _threshold_ratios(thresholds, max_raw, warnings)
            contexts.append(
                {
                    "syllabus": parsed.get("syllabus", ""),
                    "year": parsed.get("year", ""),
                    "session": parsed.get("session", ""),
                    "session_key": parsed.get("session_key", ""),
                    "component": component.get("component", ""),
                    "component_family": _family_for_component(str(component.get("component") or "")),
                    "max_raw_mark": max_raw,
                    "threshold_ratios": ratios,
                    "component_context_label": "paper_context_unknown",
                    "comparison_basis": [],
                    "confidence": "unknown",
                    "source_path": parsed.get("source_path", ""),
                    "warnings": warnings,
                }
            )
    _label_threshold_contexts(contexts)
    payload = {
        "schema": GRADE_THRESHOLD_CONTEXT_SCHEMA,
        "generated_at": utc_now_iso(),
        "contexts": contexts,
        "warnings": [] if parsed_dir.exists() else [f"missing_parsed_dir:{parsed_dir.as_posix()}"],
    }
    if not dry_run:
        write_atomic_json(payload, output_path)
    return payload


def _examiner_comments_by_key(parsed_dir: str | Path) -> dict[str, str]:
    comments: dict[str, str] = {}
    for parsed in _examiner_parsed_payloads(parsed_dir):
        for component in parsed.get("components", []):
            component_code = str(component.get("component") or "")
            for question in component.get("questions", []):
                key = normalized_question_key(
                    str(parsed.get("syllabus") or ""),
                    str(parsed.get("year") or ""),
                    str(parsed.get("session") or ""),
                    component_code,
                    int(question.get("question_number")),
                )
                comments[key] = str(question.get("comment_text") or "")
    return comments


def _examiner_levels_by_key(parsed_dir: str | Path) -> dict[str, str]:
    levels: dict[str, str] = {}
    for parsed in _examiner_parsed_payloads(parsed_dir):
        for component in parsed.get("components", []):
            component_code = str(component.get("component") or "")
            for question in component.get("questions", []):
                key = normalized_question_key(
                    str(parsed.get("syllabus") or ""),
                    str(parsed.get("year") or ""),
                    str(parsed.get("session") or ""),
                    component_code,
                    int(question.get("question_number")),
                )
                levels[key] = str(question.get("evidence_level") or "normal")
    return levels


def _examiner_parsed_payloads(parsed_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(parsed_dir)
    return [load_json(item) for item in sorted(path.glob("*.json"))] if path.exists() else []


def _allowed_topics_by_family(config: AppConfig) -> dict[str, set[str]]:
    return {family: set(topics) for family, topics in config.paper_family_taxonomy.items()}


def _topic_matches(text: str, family: str, allowed: dict[str, set[str]]) -> list[dict[str, Any]]:
    family_allowed = allowed.get(family, set())
    matches: list[dict[str, Any]] = []
    for topic_id, pattern in TOPIC_RULES:
        if topic_id not in family_allowed:
            continue
        found = [match.group(0) for match in re.finditer(pattern, text, re.IGNORECASE)]
        if found:
            matches.append({"topic_id": topic_id, "matched_terms": dedupe_preserve_order(found)})
    return matches


def _difficulty_signal(text: str, evidence_level: str) -> tuple[str, list[str], list[str], str]:
    if evidence_level in {"low", "none"}:
        return "unknown", [], ["low or no examiner-report evidence"], "low"
    hard = _matched_patterns(text, HARD_PATTERNS)
    easy = _matched_patterns(text, EASY_PATTERNS)
    if hard and easy:
        return "mixed", hard + easy, ["both harder and easier examiner phrases matched"], "medium"
    if hard:
        return "hard", hard, ["examiner report describes difficulty or low success"], "medium"
    if easy:
        return "easy", easy, ["examiner report describes broad candidate success"], "medium"
    if re.search(r"some candidates|many candidates|successful candidates|candidates who", text, re.IGNORECASE):
        return "moderate", [], ["examiner report includes usable but non-extreme performance evidence"], "low"
    return "unknown", [], ["no explicit examiner-report difficulty phrase matched"], "low"


def _matched_patterns(text: str, patterns: list[str]) -> list[str]:
    matched: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matched.append(match.group(0))
    return dedupe_preserve_order(matched)


def _threshold_ratios(thresholds: dict[str, Any], max_raw: Any, warnings: list[str]) -> dict[str, float]:
    if not isinstance(max_raw, int) or max_raw <= 0:
        warnings.append("invalid_max_raw_mark")
        return {}
    ratios: dict[str, float] = {}
    for grade in ["A", "B", "C", "D", "E"]:
        value = thresholds.get(grade)
        if isinstance(value, int):
            ratios[grade] = round(value / max_raw, 4)
    if "A" not in ratios:
        warnings.append("missing_A_threshold")
    return ratios


def _label_threshold_contexts(contexts: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        grouped[
            (
                str(context.get("syllabus") or ""),
                str(context.get("year") or ""),
                str(context.get("session") or ""),
                str(context.get("component_family") or ""),
            )
        ].append(context)
    for peers in grouped.values():
        a_ratios = [context["threshold_ratios"]["A"] for context in peers if "A" in context.get("threshold_ratios", {})]
        if not a_ratios:
            continue
        midpoint = median(a_ratios)
        for context in peers:
            ratio = context.get("threshold_ratios", {}).get("A")
            if ratio is None:
                continue
            diff = ratio - midpoint
            if diff <= -0.05:
                label = "paper_context_harder_than_session_peers"
            elif diff >= 0.05:
                label = "paper_context_easier_than_session_peers"
            else:
                label = "paper_context_typical"
            context["component_context_label"] = label
            context["comparison_basis"] = ["same_session", "same_component_family"]
            context["confidence"] = "medium" if len(a_ratios) >= 2 else "low"


def _family_for_component(component: str) -> str:
    if not component:
        return "unknown"
    return f"P{component[0]}"

