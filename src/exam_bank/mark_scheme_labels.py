from __future__ import annotations

import re

_VALID_PART_LABEL = r"(?:[a-h]|viii|vii|vi|iv|ix|iii|ii|i|v|x)"


def top_level_mark_scheme_text_label(line: str) -> str | None:
    """Return a leading question number only for a real CAIE part label."""
    match = re.match(r"^(?P<number>\d{1,2})(?P<rest>.*)$", line.strip())
    if not match:
        return None
    if not re.match(rf"^\s*\({_VALID_PART_LABEL}\)(?:\s|$)", match.group("rest"), re.IGNORECASE):
        return None
    return str(int(match.group("number")))
