from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


SubjectFamily = Literal["pm1", "pm3", "mechanics", "stats"]
SyllabusCourseId = Literal["p1", "p3", "m1", "s1", "s2"]


@dataclass(frozen=True)
class PaperFamilyContract:
    paper_number: str
    subject_family: SubjectFamily
    course_id: SyllabusCourseId
    component_name: str


PAPER_FAMILY_CONTRACTS: dict[str, PaperFamilyContract] = {
    "1": PaperFamilyContract("1", "pm1", "p1", "Pure Mathematics 1"),
    "3": PaperFamilyContract("3", "pm3", "p3", "Pure Mathematics 3"),
    "4": PaperFamilyContract("4", "mechanics", "m1", "Mechanics 1"),
    "5": PaperFamilyContract("5", "stats", "s1", "Probability & Statistics 1"),
    "6": PaperFamilyContract("6", "stats", "s2", "Probability & Statistics 2"),
}

# CAIE 9709 changed the statistics paper numbering for the syllabus first
# examined in 2020.  Paper 6 is S1 through 2019, then S2 from 2020 onward.
# The storage subject remains `stats` in both eras; only course/taxonomy
# identity needs the year split.
P5_S1_FIRST_EXAM_YEAR = 2020
P6_S2_FIRST_EXAM_YEAR = 2020
LEGACY_P6_CONTRACT = PaperFamilyContract("6", "stats", "s1", "Probability & Statistics 1")

CANONICAL_SUBJECTS: tuple[SubjectFamily, ...] = ("pm1", "pm3", "mechanics", "stats")
SYLLABUS_COURSE_IDS: tuple[SyllabusCourseId, ...] = ("p1", "p3", "m1", "s1", "s2")

LEGACY_PAPER_FAMILY_TO_SUBJECT: dict[str, SubjectFamily] = {
    f"p{paper_number}": contract.subject_family
    for paper_number, contract in PAPER_FAMILY_CONTRACTS.items()
}

_SUBJECT_ALIASES: dict[str, SubjectFamily] = {
    "pm1": "pm1",
    "pure1": "pm1",
    "puremaths1": "pm1",
    "puremathematics1": "pm1",
    "p1": "pm1",
    "1": "pm1",
    "pm3": "pm3",
    "pure3": "pm3",
    "puremaths3": "pm3",
    "puremathematics3": "pm3",
    "p3": "pm3",
    "3": "pm3",
    "mechanics": "mechanics",
    "mech": "mechanics",
    "m1": "mechanics",
    "p4": "mechanics",
    "4": "mechanics",
    "stats": "stats",
    "statistics": "stats",
    "stat": "stats",
    "s1": "stats",
    "p5": "stats",
    "5": "stats",
    "s2": "stats",
    "p6": "stats",
    "6": "stats",
}

_COURSE_ALIASES: dict[str, SyllabusCourseId] = {
    "pm1": "p1",
    "pure1": "p1",
    "puremaths1": "p1",
    "puremathematics1": "p1",
    "p1": "p1",
    "1": "p1",
    "pm3": "p3",
    "pure3": "p3",
    "puremaths3": "p3",
    "puremathematics3": "p3",
    "p3": "p3",
    "3": "p3",
    "mechanics": "m1",
    "mech": "m1",
    "m1": "m1",
    "p4": "m1",
    "4": "m1",
    "s1": "s1",
    "s2": "s2",
}


def normalize_subject_family(value: object) -> SubjectFamily | None:
    return _SUBJECT_ALIASES.get(_normalize_token(value))


def paper_number_from_component(value: object) -> str | None:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if not digits:
        return None
    component = digits[-2:].zfill(2)
    paper_number = component[1] if component[0] == "0" else component[0]
    return paper_number if paper_number in PAPER_FAMILY_CONTRACTS else None


def exam_year_from_evidence(
    *,
    year: object = None,
    session: object = None,
    paper: object = None,
) -> int | None:
    explicit = str(year or "").strip()
    if explicit:
        if re.fullmatch(r"\d{4}", explicit):
            return int(explicit)
        if re.fullmatch(r"\d{1,2}", explicit):
            return 2000 + int(explicit)

    for value in (paper, session):
        text = str(value or "").strip().lower()
        if not text:
            continue
        full_year = re.search(r"(?<!\d)(20\d{2})(?!\d)", text)
        if full_year:
            return int(full_year.group(1))
        session_year = re.search(r"(?:spring|summer|winter|autumn|[msw])(\d{2})(?!\d)", text)
        if session_year:
            return 2000 + int(session_year.group(1))
    return None


def paper_contract_for_component(
    value: object,
    *,
    year: object = None,
    session: object = None,
    paper: object = None,
) -> PaperFamilyContract | None:
    paper_number = paper_number_from_component(value)
    if paper_number == "5":
        exam_year = exam_year_from_evidence(year=year, session=session, paper=paper)
        if exam_year is None or exam_year < P5_S1_FIRST_EXAM_YEAR:
            return None
    if paper_number == "6":
        exam_year = exam_year_from_evidence(year=year, session=session, paper=paper)
        if exam_year is None:
            return None
        if exam_year < P6_S2_FIRST_EXAM_YEAR:
            return LEGACY_P6_CONTRACT
    return PAPER_FAMILY_CONTRACTS.get(paper_number or "")


def subject_family_for_component(
    value: object,
    *,
    year: object = None,
    session: object = None,
    paper: object = None,
) -> SubjectFamily | None:
    paper_number = paper_number_from_component(value)
    # Storage identity is not era-sensitive for P6: both historical S1 and
    # current S2 live under `stats`.  Pre-2020 P5 is unsupported M2, while
    # current P5 is S1, so P5 requires year evidence even for storage identity.
    if paper_number == "6":
        return "stats"
    contract = paper_contract_for_component(
        value,
        year=year,
        session=session,
        paper=paper,
    )
    return contract.subject_family if contract else None


def course_id_for_identity(
    value: object,
    *,
    component: object = None,
    year: object = None,
    session: object = None,
    paper: object = None,
) -> SyllabusCourseId | None:
    component_paper_number = paper_number_from_component(component)
    if component_paper_number:
        contract = paper_contract_for_component(
            component,
            year=year,
            session=session,
            paper=paper,
        )
        return contract.course_id if contract else None
    token = _normalize_token(value)
    if token in {"p5", "5"}:
        contract = paper_contract_for_component(
            "5",
            year=year,
            session=session,
            paper=paper,
        )
        return contract.course_id if contract else None
    if token in {"p6", "6"}:
        contract = paper_contract_for_component(
            "6",
            year=year,
            session=session,
            paper=paper,
        )
        return contract.course_id if contract else None
    course_id = _COURSE_ALIASES.get(token)
    if course_id:
        return course_id
    # The canonical storage family combines S1 and S2. It is intentionally
    # ambiguous without a component and must not silently select either course.
    if token in {"stats", "statistics", "stat"}:
        return None
    return None


def subject_family_agrees_with_component(
    subject_family: object,
    component: object,
    *,
    year: object = None,
    session: object = None,
    paper: object = None,
) -> bool:
    family = normalize_subject_family(subject_family)
    component_family = subject_family_for_component(
        component,
        year=year,
        session=session,
        paper=paper,
    )
    return bool(family and component_family and family == component_family)


def default_component_for_subject(value: object) -> str | None:
    family = normalize_subject_family(value)
    if family == "pm1":
        return "11"
    if family == "pm3":
        return "31"
    if family == "mechanics":
        return "41"
    # `stats` covers both P5/S1 and P6/S2. S1 remains the compatibility default
    # only for filename helpers that have no paper/component evidence.
    if family == "stats":
        return "51"
    return None


def taxonomy_paper_family(
    value: object,
    *,
    component: object = None,
    year: object = None,
    session: object = None,
    paper: object = None,
) -> str:
    component_paper_number = paper_number_from_component(component)
    if component_paper_number:
        contract = paper_contract_for_component(
            component,
            year=year,
            session=session,
            paper=paper,
        )
        if contract is None:
            return ""
        return {
            "p1": "p1",
            "p3": "p3",
            "m1": "p4",
            "s1": "p5",
            "s2": "p6",
        }[contract.course_id]
    token = _normalize_token(value)
    if token in {"p5", "5"}:
        contract = paper_contract_for_component(
            "5",
            year=year,
            session=session,
            paper=paper,
        )
        return "p5" if contract else ""
    if token in {"p6", "6"}:
        contract = paper_contract_for_component(
            "6",
            year=year,
            session=session,
            paper=paper,
        )
        if contract is None:
            return ""
        return "p5" if contract.course_id == "s1" else "p6"
    course_id = course_id_for_identity(
        value,
        year=year,
        session=session,
        paper=paper,
    )
    return {
        "p1": "p1",
        "p3": "p3",
        "m1": "p4",
        "s1": "p5",
        "s2": "p6",
    }.get(course_id or "", _normalize_token(value))


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
