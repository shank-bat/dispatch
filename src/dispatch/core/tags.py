"""User-defined job labels.

Tags are the mechanism that makes several hundred runs navigable: ``tag:paper``,
``tag:re100000``, ``tag:naca0018``. See ``docs/ARCHITECTURE.md`` §4.5.

Normalisation is strict and happens at every entry point. The alternative -- accepting
whatever the user typed -- produces ``Paper``, ``paper``, and ``paper `` as three distinct
tags, and the feature quietly stops working around run number fifty.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

from dispatch.core.errors import ValidationError

__all__ = ["MAX_TAG_LENGTH", "TAG_PATTERN", "normalise_tag", "normalise_tags"]

MAX_TAG_LENGTH: Final = 64

TAG_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
"""Lowercase alphanumeric start, then alphanumerics and ``. _ -``.

Deliberately excludes ``:`` and whitespace, which are the search grammar's separators
(§5.2) -- a tag named ``foo:bar`` would be unsearchable by construction.
"""


def normalise_tag(raw: str) -> str:
    """Return the canonical form of a single tag.

    Trims surrounding whitespace, lowercases, and converts internal whitespace to ``-``
    so that a pasted ``"NACA 0018"`` becomes the usable ``naca-0018`` instead of being
    rejected for a reason the user considers pedantic.

    Args:
        raw: Tag as typed by the user.

    Returns:
        The normalised tag.

    Raises:
        ValidationError: If the result is empty, too long, or contains characters that
            would collide with the search grammar.
    """
    tag = "-".join(raw.strip().lower().split())
    if not tag:
        raise ValidationError("Tag is empty")
    if len(tag) > MAX_TAG_LENGTH:
        raise ValidationError(
            f"Tag is {len(tag)} characters; the maximum is {MAX_TAG_LENGTH}",
            detail={"tag": tag},
        )
    if not TAG_PATTERN.match(tag):
        raise ValidationError(
            f"Invalid tag {tag!r}: use letters, digits, and . _ - only, starting with a "
            "letter or digit",
            detail={"tag": tag},
        )
    return tag


def normalise_tags(raw: Iterable[str]) -> frozenset[str]:
    """Normalise a collection of tags, discarding duplicates.

    Blank entries are skipped rather than raising, so a trailing comma in ``a, b,`` is
    not an error the user has to go back and fix.
    """
    return frozenset(normalise_tag(item) for item in raw if item.strip())
