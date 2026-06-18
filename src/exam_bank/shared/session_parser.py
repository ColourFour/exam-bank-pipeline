from __future__ import annotations

import re
from typing import Literal, TypedDict


SeasonCode = Literal["s", "m", "w"]


class ParsedSession(TypedDict):
    year: int
    session: str
    canonical_session: str
    season: SeasonCode
    component_year_key: str
    canonical_year_folder: str


class SessionParseError(ValueError):
    pass


_SESSION_TO_SEASON: dict[str, SeasonCode] = {
    "m": "m",
    "mar": "m",
    "march": "m",
    "febmarch": "m",
    "februarymarch": "m",
    "s": "s",
    "summer": "s",
    "may": "s",
    "jun": "s",
    "june": "s",
    "mayjun": "s",
    "mayjune": "s",
    "mj": "s",
    "w": "w",
    "winter": "w",
    "autumn": "w",
    "oct": "w",
    "october": "w",
    "nov": "w",
    "november": "w",
    "octnov": "w",
    "octobernovember": "w",
    "on": "w",
}

_CANONICAL_SESSION_LABEL: dict[SeasonCode, str] = {
    "m": "summer",
    "s": "summer",
    "w": "winter",
}


def parse_session(code: str) -> ParsedSession:
    normalized = _normalize(code)
    if not normalized:
        raise SessionParseError("missing session code")

    season, year = _find_session_year(normalized)
    yy = f"{year % 100:02d}"
    session = _CANONICAL_SESSION_LABEL[season]
    canonical_session = f"{session}{yy}"
    return {
        "year": year,
        "session": session,
        "canonical_session": canonical_session,
        "season": season,
        "component_year_key": f"{season}{yy}",
        "canonical_year_folder": str(year),
    }


def parse_session_from_parts(session: str, year: str | int) -> ParsedSession:
    year_text = str(year).strip()
    if not year_text:
        raise SessionParseError(f"missing year for session {session!r}")
    try:
        parsed = parse_session(session)
        if str(parsed["year"]) == str(_normalize_year(year_text)):
            return parsed
    except SessionParseError:
        pass
    return parse_session(f"{session}{year_text[-2:] if len(year_text) == 4 else year_text}")


def canonical_paper_id(component: str, session: str, year: str | int) -> str:
    component_code = "".join(char for char in str(component) if char.isdigit())
    if len(component_code) >= 2:
        component_code = component_code[-2:]
    elif len(component_code) == 1:
        component_code = component_code.zfill(2)
    else:
        component_code = "xx"
    return f"{component_code}{parse_session_from_parts(session, year)['canonical_session']}"


def _find_session_year(normalized: str) -> tuple[SeasonCode, int]:
    compact = re.search(r"(?<![a-z0-9])([msw])(\d{2}|20\d{2})(?![a-z0-9])", normalized)
    if compact:
        return compact.group(1), _normalize_year(compact.group(2))

    aliases = sorted(_SESSION_TO_SEASON, key=len, reverse=True)
    alias_pattern = "|".join(re.escape(alias) for alias in aliases)
    session_then_year = re.search(rf"(?<![a-z0-9])({alias_pattern})(\d{{2}}|20\d{{2}})(?![a-z0-9])", normalized)
    if session_then_year:
        return _SESSION_TO_SEASON[session_then_year.group(1)], _normalize_year(session_then_year.group(2))

    year_then_session = re.search(rf"(?<![a-z0-9])(\d{{2}}|20\d{{2}})({alias_pattern})(?![a-z0-9])", normalized)
    if year_then_session:
        return _SESSION_TO_SEASON[year_then_session.group(2)], _normalize_year(year_then_session.group(1))

    raise SessionParseError(f"could not parse session code: {normalized!r}")


def _normalize_year(value: str) -> int:
    if len(value) == 4:
        return int(value)
    year = int(value)
    return 2000 + year if year < 80 else 1900 + year


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
