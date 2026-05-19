from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Any


ROMAN_LABELS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
ALPHA_LABELS = {"a", "b", "c", "d", "e", "f", "g", "h"}

MARK_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<dependent>D\s*)?(?P<kind>[MABCE])\s*(?P<value>[1-9]\d?)"
    r"(?P<suffix>\s*(?:ft|dep))?\b",
    re.IGNORECASE,
)
PART_HEADER_RE = re.compile(
    r"^\s*(?:\d{1,3}\s*)?(?P<labels>(?:\s*\((?:a|b|c|d|e|f|g|h|i|ii|iii|iv|v|vi|vii|viii|ix|x)\))+)",
    re.IGNORECASE,
)
LINE_LEADING_ALTERNATIVE_RE = re.compile(r"^\s*(?:or|alternative\s+method|alt(?:ernative)?\b)", re.IGNORECASE)


@dataclass
class ParsedMarkEvent:
    part_path: list[str]
    raw_text: str
    normalized_text: str
    mark_code_raw: str
    mark_type: str
    mark_value: int
    is_follow_through: bool
    is_dependent: bool
    depends_on_event_ids: list[str] = field(default_factory=list)
    alternative_group_id: str | None = None
    condition_text: str = ""
    answer_text: str = ""
    common_error_text: str = ""
    confidence: str = "medium"
    review_flags: list[str] = field(default_factory=list)

    def to_dict(self, event_id: str) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "part_path": list(self.part_path),
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "mark_code_raw": self.mark_code_raw,
            "mark_type": self.mark_type,
            "mark_value": self.mark_value,
            "is_follow_through": self.is_follow_through,
            "is_dependent": self.is_dependent,
            "depends_on_event_ids": list(self.depends_on_event_ids),
            "alternative_group_id": self.alternative_group_id,
            "condition_text": self.condition_text,
            "answer_text": self.answer_text,
            "common_error_text": self.common_error_text,
            "confidence": self.confidence,
            "review_flags": list(self.review_flags),
        }


@dataclass
class ParseResult:
    events: list[dict[str, Any]]
    unparsed_evidence: list[dict[str, Any]]
    review_flags: list[str]
    top_unparsed_patterns: dict[str, int]


def parse_mark_scheme_text(text: str | None, *, question_id: str) -> ParseResult:
    """Parse deterministic mark-event candidates from advisory mark-scheme text."""

    raw_text = str(text or "")
    lines = [_compact_text(line) for line in raw_text.splitlines()]
    current_part: list[str] = []
    events: list[dict[str, Any]] = []
    unparsed: list[dict[str, Any]] = []
    review_flags: list[str] = []
    previous_by_part: dict[tuple[str, ...], str] = {}
    alt_counter = 0
    unparsed_patterns: dict[str, int] = defaultdict(int)

    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        detected_part = detect_part_path(line, current_part)
        if detected_part is not None:
            current_part = detected_part

        matches = list(MARK_CODE_RE.finditer(line))
        if not matches:
            if _line_has_reviewable_semantics(line):
                pattern = _unparsed_pattern(line)
                unparsed_patterns[pattern] += 1
                unparsed.append(
                    {
                        "line_number": line_number,
                        "part_path": list(current_part),
                        "raw_text": line,
                        "reason": "semantic_line_without_deterministic_mark_code",
                        "pattern": pattern,
                    }
                )
            continue

        selected_matches, skipped_reason = _select_countable_matches(line, matches)
        if skipped_reason:
            review_flags.append(skipped_reason)
            pattern = _unparsed_pattern(line)
            unparsed_patterns[pattern] += 1
            unparsed.append(
                {
                    "line_number": line_number,
                    "part_path": list(current_part),
                    "raw_text": line,
                    "reason": skipped_reason,
                    "pattern": pattern,
                }
            )

        alternative_group_id = None
        if LINE_LEADING_ALTERNATIVE_RE.search(line):
            alt_counter += 1
            alternative_group_id = f"alt-{alt_counter:03d}"
        elif _has_inline_alternative(line):
            review_flags.append("possible_inline_alternative")
            pattern = _unparsed_pattern(line)
            unparsed_patterns[pattern] += 1
            unparsed.append(
                {
                    "line_number": line_number,
                    "part_path": list(current_part),
                    "raw_text": line,
                    "reason": "possible_inline_alternative",
                    "pattern": pattern,
                }
            )

        for match in selected_matches:
            parsed = _event_from_match(
                line,
                match,
                part_path=current_part,
                previous_event_id=previous_by_part.get(tuple(current_part)),
                alternative_group_id=alternative_group_id,
            )
            event_id = f"{question_id}_me{len(events) + 1:04d}"
            event = parsed.to_dict(event_id)
            events.append(event)
            previous_by_part[tuple(current_part)] = event_id
            if parsed.review_flags:
                review_flags.extend(parsed.review_flags)

    return ParseResult(
        events=events,
        unparsed_evidence=unparsed,
        review_flags=_dedupe(review_flags),
        top_unparsed_patterns=dict(sorted(unparsed_patterns.items(), key=lambda item: (-item[1], item[0]))[:20]),
    )


def detect_part_path(line: str, current_part: list[str] | None = None) -> list[str] | None:
    match = PART_HEADER_RE.match(line)
    if not match:
        return None
    labels = [label.lower() for label in re.findall(r"\(([^)]+)\)", match.group("labels"))]
    if not labels:
        return None
    current_part = current_part or []
    alpha = next((label for label in labels if label in ALPHA_LABELS), "")
    romans = [label for label in labels if label in ROMAN_LABELS]
    if alpha:
        return [alpha] + romans[:1]
    if romans and current_part and current_part[0] in ALPHA_LABELS:
        return [current_part[0], romans[-1]]
    if romans:
        return [romans[-1]]
    return None


def normalize_part_path(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    output: list[str] = []
    for item in value:
        token = str(item or "").strip().lower()
        if token not in ALPHA_LABELS and token not in ROMAN_LABELS:
            return None
        output.append(token)
    return output


def _event_from_match(
    line: str,
    match: re.Match[str],
    *,
    part_path: list[str],
    previous_event_id: str | None,
    alternative_group_id: str | None,
) -> ParsedMarkEvent:
    mark_code = _normalized_mark_code(match)
    value = int(match.group("value"))
    is_follow = _is_follow_through(match, line)
    is_dependent = _is_dependent(match, line)
    review_flags: list[str] = []
    depends_on: list[str] = []
    if is_dependent:
        if previous_event_id:
            depends_on.append(previous_event_id)
        else:
            review_flags.append("dependent_mark_without_deterministic_prior_event")
    if mark_code.startswith("D") and not mark_code.startswith("DM"):
        review_flags.append("unknown_dependent_mark_code")
    if mark_code.lstrip("D").startswith("C"):
        review_flags.append("unknown_mark_code")

    answer = _answer_text_from_line(line, match)
    condition = _condition_text(line)
    common_error = _common_error_text(line)
    mark_type = _mark_type(mark_code, is_follow=is_follow, condition_text=condition)
    confidence = _confidence(mark_type=mark_type, answer_text=answer, review_flags=review_flags, condition_text=condition)
    return ParsedMarkEvent(
        part_path=list(part_path),
        raw_text=line,
        normalized_text=_compact_text(line),
        mark_code_raw=mark_code,
        mark_type=mark_type,
        mark_value=value,
        is_follow_through=is_follow,
        is_dependent=is_dependent,
        depends_on_event_ids=depends_on,
        alternative_group_id=alternative_group_id,
        condition_text=condition,
        answer_text=answer,
        common_error_text=common_error,
        confidence=confidence,
        review_flags=review_flags,
    )


def _normalized_mark_code(match: re.Match[str]) -> str:
    dependent = "D" if match.group("dependent") else ""
    kind = match.group("kind").upper()
    value = str(int(match.group("value")))
    suffix = _compact_text(match.group("suffix") or "").lower()
    if suffix == "ft":
        return f"{dependent}{kind}{value}FT"
    return f"{dependent}{kind}{value}"


def _mark_type(mark_code: str, *, is_follow: bool, condition_text: str) -> str:
    if is_follow:
        return "follow_through"
    if re.search(r"\bAG\b|answer given", condition_text, re.IGNORECASE):
        return "answer_given"
    if mark_code.startswith("DM"):
        return "dependent_method"
    core = mark_code[1:] if mark_code.startswith("D") else mark_code
    if core.startswith("M"):
        return "method"
    if core.startswith("A"):
        return "accuracy"
    if core.startswith("B"):
        return "independent_statement"
    if core.startswith("E"):
        return "explanation"
    return "unknown"


def _is_follow_through(match: re.Match[str], line: str) -> bool:
    suffix = _compact_text(match.group("suffix") or "").lower()
    return suffix == "ft" or bool(re.search(r"\b(?:ft|follow\s*through|followed\s*through)\b", line, re.IGNORECASE))


def _is_dependent(match: re.Match[str], line: str) -> bool:
    suffix = _compact_text(match.group("suffix") or "").lower()
    return bool(match.group("dependent")) or suffix == "dep" or bool(re.search(r"\bdep(?:endent)?\b", line, re.IGNORECASE))


def _select_countable_matches(line: str, matches: list[re.Match[str]]) -> tuple[list[re.Match[str]], str | None]:
    if not matches:
        return [], None
    first = matches[0]
    first_value = int(first.group("value"))
    tail_after_first = line[first.end() :]
    if first_value > 1 and re.match(r"\s*,\s*\d{1,2}\s*,\s*\d{1,2}\b", tail_after_first):
        return [first], "condition_mark_codes_not_counted"
    if first_value > 1 and re.search(r"\b(?:all\s+correct|for\s+(?:one|two|three)|allow\s+\w+\s+for)\b", tail_after_first, re.IGNORECASE):
        return [first], "condition_mark_codes_not_counted"
    return matches, None


def _has_inline_alternative(line: str) -> bool:
    return bool(re.search(r"\s+Or\s+", line)) and not LINE_LEADING_ALTERNATIVE_RE.search(line)


def _answer_text_from_line(line: str, match: re.Match[str]) -> str:
    before = line[: match.start()]
    after = line[match.end() :]
    before = PART_HEADER_RE.sub(" ", before)
    before = re.sub(r"^\s*\d{1,3}\s*", " ", before)
    before = _compact_text(before).strip(" ;,:")
    if before:
        return before[:300]
    target = re.search(r"\b(?:answer|obtain|obtains|gives?|leading to|hence)\b\s*(?P<target>.+)$", after, re.IGNORECASE)
    if target:
        return _compact_text(target.group("target")).strip(" ;,:")[:300]
    return ""


def _condition_text(line: str) -> str:
    markers = [
        ("FT", r"\b(?:ft|follow\s*through|followed\s*through)\b"),
        ("AG", r"\bAG\b|answer\s+given"),
        ("cao", r"\bcao\b"),
        ("oe", r"\boe\b|or\s+equivalent"),
        ("www", r"\bwww\b"),
        ("isw", r"\bisw\b"),
        ("dependent", r"\bdep(?:endent)?\b"),
        ("independent", r"\bindep(?:endent)?\b"),
    ]
    found = [label for label, pattern in markers if re.search(pattern, line, re.IGNORECASE)]
    return "; ".join(found)


def _common_error_text(line: str) -> str:
    patterns = [
        r"\bdo\s+not\s+allow\b[^.;]*",
        r"\bdo\s+not\s+accept\b[^.;]*",
        r"\bnot\s+accepted\b[^.;]*",
        r"\bincorrect\b[^.;]*",
        r"\bwrong\b[^.;]*",
        r"\b(?:M0|A0|B0)\b[^.;]*",
        r"\bscores?\s+0\b[^.;]*",
        r"\bignore\b[^.;]*",
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, line, re.IGNORECASE):
            matches.append(_compact_text(match.group(0)).strip(" ;,:"))
    return "; ".join(_dedupe(matches))[:500]


def _confidence(*, mark_type: str, answer_text: str, review_flags: list[str], condition_text: str) -> str:
    if review_flags or mark_type == "unknown":
        return "low"
    if answer_text and condition_text:
        return "high"
    if answer_text:
        return "high"
    return "medium"


def _line_has_reviewable_semantics(line: str) -> bool:
    return bool(
        re.search(
            r"\b(?:or|alternative|allow|do\s+not|ignore|cao|oe|www|isw|answer\s+given|follow\s*through|common\s+error)\b",
            line,
            re.IGNORECASE,
        )
    )


def _unparsed_pattern(line: str) -> str:
    text = re.sub(r"\d+", "#", line.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]


def _compact_text(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output

