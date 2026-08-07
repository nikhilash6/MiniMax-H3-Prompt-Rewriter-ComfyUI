"""Splitting a rewrite into the three fields the adapter is trained to emit."""

from __future__ import annotations

import re

from .constants import OUTPUT_FIELDS

_LABEL = re.compile(
    r"^[ \t]*[*_#>\s]*(" + "|".join(OUTPUT_FIELDS) + r")[*_ \t]*:[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)


def split_fields(text: str) -> dict[str, str]:
    """Return the three labelled sections of a rewrite.

    A rewrite that ignored the output format still yields something usable: the
    whole text lands in ``integrated_multimodal_description`` so downstream
    inputs are never silently empty.
    """
    result = {name: "" for name in OUTPUT_FIELDS}
    matches = list(_LABEL.finditer(text or ""))

    if not matches:
        result[OUTPUT_FIELDS[0]] = (text or "").strip()
        return result

    for index, match in enumerate(matches):
        name = match.group(1).lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[name] = text[match.end():end].strip()
    return result
