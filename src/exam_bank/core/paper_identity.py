from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from exam_bank.identifiers import parent_question_id
from exam_bank.shared.session_parser import SessionParseError
from exam_bank.shared.session_parser import parse_session as _parse_shared_session
from exam_bank.shared.session_parser import parse_session_from_parts as _parse_shared_session_from_parts


SubjectFamily = Literal["pm1", "pm3", "stats", "mechanics"]


class IdentityError(ValueError):
    """Raised when canonical paper identity cannot be derived or reconciled."""


@dataclass(frozen=True)
class ParsedSession:
    year: int
    session_code: str
    canonical_session: str
    season: Literal["s", "m", "w"]
    canonical_year_folder: str


@dataclass(frozen=True)
class PaperIdentity:
    syllabus: str
    subject_family: SubjectFamily
    year: int
    session_code: str
    canonical_session: str
    component: str
    paper_id: str
    question_id: str

    @classmethod
    def from_parts(
        cls,
        *,
        syllabus: str | int = "9709",
        subject_family: str = "",
        year: str | int = "",
        session: str = "",
        component: str | int = "",
        question_number: str | int | None = None,
        expected_paper_id: str = "",
        expected_question_id: str = "",
    ) -> "PaperIdentity":
        parsed = parse_session_from_parts(session, year)
        normalized_syllabus = _normalize_syllabus(syllabus)
        normalized_component = _normalize_component(component)
        normalized_subject = canonical_subject_family(subject_family or _subject_from_component(normalized_component))
        paper_id = build_paper_id(normalized_syllabus, parsed.session_code, normalized_component)
        question_id = build_question_id(paper_id, question_number) if question_number not in (None, "") else ""

        identity = cls(
            syllabus=normalized_syllabus,
            subject_family=normalized_subject,
            year=parsed.year,
            session_code=parsed.session_code,
            canonical_session=parsed.canonical_session,
            component=normalized_component,
            paper_id=paper_id,
            question_id=question_id,
        )
        validate_identity_agreement(
            identity,
            expected_paper_id=expected_paper_id,
            expected_question_id=expected_question_id,
        )
        return identity


def parse_session(code: str) -> ParsedSession:
    try:
        parsed = _parse_shared_session(code)
    except SessionParseError as exc:
        raise IdentityError(str(exc)) from exc
    return ParsedSession(
        year=int(parsed["year"]),
        session_code=str(parsed["component_year_key"]),
        canonical_session=str(parsed["canonical_session"]),
        season=parsed["season"],
        canonical_year_folder=str(parsed["canonical_year_folder"]),
    )


def parse_session_from_parts(session: str, year: str | int) -> ParsedSession:
    try:
        parsed = _parse_shared_session_from_parts(str(session), year)
    except SessionParseError as exc:
        raise IdentityError(str(exc)) from exc

    session_code = str(parsed["component_year_key"])
    explicit = parse_session(session_code)
    if explicit.year != int(parsed["year"]) or explicit.canonical_session != str(parsed["canonical_session"]):
        raise IdentityError(f"session parsing is inconsistent for session={session!r} year={year!r}")
    return explicit


def build_question_id(paper_id: str, question_number: str | int | None) -> str:
    paper = str(paper_id or "").strip()
    if not paper:
        raise IdentityError("paper_id cannot be empty")
    normalized_question = parent_question_id(question_number)
    if not normalized_question:
        raise IdentityError("question_number cannot be empty")
    match = re.fullmatch(r"\d{1,2}", normalized_question)
    if not match:
        raise IdentityError(f"question_number must resolve to a top-level number: {question_number!r}")
    return f"{paper}_q{int(normalized_question):02d}"


def build_paper_id(syllabus: str | int, session: str, component: str | int) -> str:
    _normalize_syllabus(syllabus)
    parsed = parse_session(session)
    normalized_component = _normalize_component(component)
    return f"{normalized_component}{parsed.canonical_session}"


def paper_identity_from_parts(
    *,
    syllabus: str | int = "9709",
    subject_family: str = "",
    year: str | int = "",
    session: str = "",
    component: str | int = "",
    question_number: str | int | None = None,
    expected_paper_id: str = "",
    expected_question_id: str = "",
) -> PaperIdentity:
    return PaperIdentity.from_parts(
        syllabus=syllabus,
        subject_family=subject_family,
        year=year,
        session=session,
        component=component,
        question_number=question_number,
        expected_paper_id=expected_paper_id,
        expected_question_id=expected_question_id,
    )


def session_for_source_path(path: str | Path, *, year: str | int, fallback_session: str) -> str:
    stem = Path(path).stem.lower()
    yy = str(year)[-2:]
    match = re.search(rf"(?:^|_)(?P<season>[msw]){re.escape(yy)}(?:_|$)", stem)
    if match:
        return f"{match.group('season')}{yy}"
    return fallback_session


def canonical_subject_family(value: str) -> SubjectFamily:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    mapping: dict[str, SubjectFamily] = {
        "pm1": "pm1",
        "puremaths1": "pm1",
        "puremathematics1": "pm1",
        "p1": "pm1",
        "1": "pm1",
        "pm3": "pm3",
        "puremaths3": "pm3",
        "puremathematics3": "pm3",
        "p3": "pm3",
        "3": "pm3",
        "stats": "stats",
        "statistics": "stats",
        "stat": "stats",
        "s1": "stats",
        "p4": "stats",
        "4": "stats",
        "p6": "stats",
        "6": "stats",
        "mechanics": "mechanics",
        "mech": "mechanics",
        "m1": "mechanics",
        "p5": "mechanics",
        "5": "mechanics",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise IdentityError(f"unknown subject family: {value!r}") from exc


def validate_identity_agreement(
    identity: PaperIdentity,
    *,
    expected_paper_id: str = "",
    expected_question_id: str = "",
) -> None:
    if not identity.paper_id:
        raise IdentityError("paper_id cannot be empty")
    if expected_paper_id and expected_paper_id != identity.paper_id:
        raise IdentityError(
            f"paper identity mismatch: expected {expected_paper_id!r}, derived {identity.paper_id!r}"
        )
    if expected_question_id and expected_question_id != identity.question_id:
        raise IdentityError(
            f"question identity mismatch: expected {expected_question_id!r}, derived {identity.question_id!r}"
        )


def _normalize_syllabus(value: str | int) -> str:
    text = str(value or "").strip()
    match = re.search(r"\d{4}", text)
    if not match:
        raise IdentityError(f"syllabus cannot be derived from {value!r}")
    return match.group(0)


def _normalize_component(value: str | int) -> str:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) >= 2:
        return digits[-2:]
    if len(digits) == 1:
        return digits.zfill(2)
    raise IdentityError(f"component cannot be derived from {value!r}")


def _subject_from_component(component: str) -> str:
    normalized = component.lstrip("0") or component
    if not normalized or not normalized[0].isdigit():
        raise IdentityError(f"subject family cannot be inferred from component {component!r}")
    return f"p{normalized[0]}"
