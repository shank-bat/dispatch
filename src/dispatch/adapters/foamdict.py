"""A minimal reader and writer for OpenFOAM dictionary files.

OpenFOAM's dictionary format is not JSON, INI, or YAML. It has ``//`` and ``/* */``
comments, ``#include`` directives, macro expansion (``$foo``), nested sub-dictionaries,
lists in three different syntaxes, and dimension brackets. Reaching for a regular
expression to pull ``application`` out of a ``controlDict`` works right up until a case has
the word in a comment, or a header banner, or a commented-out alternative -- and then it
silently runs the wrong solver.

This is a small proper tokeniser instead. It reads flat key/value pairs and nested blocks,
which is all Dispatch needs, and it ignores what it does not understand rather than
guessing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

__all__ = ["parse", "parse_file", "read_value", "set_value"]

log = logging.getLogger(__name__)

_COMMENT_BLOCK = re.compile(rb"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE = re.compile(rb"//[^\n]*")


def parse_file(path: Path) -> dict[str, Any]:
    """Parse a dictionary file. Returns an empty mapping if it cannot be read."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        log.debug("Could not read %s: %s", path, exc)
        return {}
    return parse(raw)


def parse(raw: bytes | str) -> dict[str, Any]:
    """Parse dictionary text into nested mappings.

    Values are returned as strings, with the trailing semicolon removed. Numeric conversion
    is the caller's business, because the same key can hold a number in one case and an
    expression in another.
    """
    if isinstance(raw, str):
        raw = raw.encode()
    text = _COMMENT_LINE.sub(b"", _COMMENT_BLOCK.sub(b"", raw)).decode("utf-8", errors="replace")
    tokens = _tokenise(text)
    result, _ = _parse_block(tokens, 0)
    return result


def _tokenise(text: str) -> list[str]:
    """Split into tokens, keeping quoted strings and brace/paren structure intact."""
    tokens: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char in "\"'":
            quote = char
            index += 1
            start = index
            while index < length and text[index] != quote:
                if text[index] == "\\":
                    index += 1
                index += 1
            tokens.append(text[start:index])
            index += 1
            continue

        if char in "{};()[]":
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()
            tokens.append(char)
            index += 1
            continue

        if char.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()
            index += 1
            continue

        buffer.append(char)
        index += 1

    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _parse_block(tokens: list[str], start: int) -> tuple[dict[str, Any], int]:
    """Parse tokens into a mapping until the block ends.

    Returns the mapping and the index just past it.
    """
    result: dict[str, Any] = {}
    index = start
    pending: list[str] = []

    while index < len(tokens):
        token = tokens[index]

        if token == "}":
            return result, index + 1

        if token == "{":
            key = pending[-1] if pending else ""
            block, index = _parse_block(tokens, index + 1)
            if key:
                result[key] = block
            pending.clear()
            continue

        if token == ";":
            if pending:
                key = pending[0]
                value = " ".join(pending[1:])
                if key and key not in result:
                    result[key] = value
            pending.clear()
            index += 1
            continue

        if token in "()[]":
            # List and dimension syntax. Kept as literal text: nothing Dispatch reads is a
            # list, and reconstructing them faithfully is not worth the code.
            pending.append(token)
            index += 1
            continue

        pending.append(token)
        index += 1

    return result, index


def read_value(path: Path, key: str, *, default: str | None = None) -> str | None:
    """Read one top-level key from a dictionary file."""
    value = parse_file(path).get(key, default)
    return value if isinstance(value, str) else default


def read_float(path: Path, key: str) -> float | None:
    """Read one key as a float, tolerating values that are not numbers.

    A ``controlDict`` may legitimately hold an expression where a number usually goes;
    that is a reason to record nothing, not to fail the job.
    """
    raw = read_value(path, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_int(path: Path, key: str) -> int | None:
    """Read one key as an integer."""
    value = read_float(path, key)
    return int(value) if value is not None else None


def set_value(path: Path, key: str, value: str) -> bool:
    """Set a top-level key in an existing dictionary file, in place.

    Rewrites only the matching entry and leaves the rest of the file -- including comments,
    the header banner, and formatting -- exactly as the user wrote it. Appends the entry if
    it is absent.

    Returns:
        Whether the file was modified.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("Could not read %s: %s", path, exc)
        return False

    pattern = re.compile(rf"^(\s*){re.escape(key)}(\s+)([^;\n]*);", re.MULTILINE)
    replacement = rf"\g<1>{key}\g<2>{value};"

    updated, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        if not updated.endswith("\n"):
            updated += "\n"
        updated += f"{key} {value};\n"

    if updated == text:
        return False
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        log.warning("Could not write %s: %s", path, exc)
        return False
    return True


GEOMETRIC_METHODS = frozenset({"simple", "hierarchical"})
"""Decomposition methods whose coefficients must agree with the subdomain count.

``scotch`` and ``metis`` compute their own partitioning, so changing the count alone is
enough. ``simple`` and ``hierarchical`` take an explicit ``n (nx ny nz)`` whose product
*is* the subdomain count -- leaving it stale makes ``decomposePar`` refuse to run.
"""

_N_VECTOR = re.compile(r"^(\s*)n(\s+)\(\s*\d+\s+\d+\s+\d+\s*\)\s*;", re.MULTILINE)


def balanced_factors(count: int) -> tuple[int, int, int]:
    """Split ``count`` into three factors for a geometric decomposition.

    Splits across x and y and leaves z at one. That is always valid, and it is the right
    answer for the 2D and quasi-2D cases that dominate CFD on a single workstation. A 3D
    mesh would usually prefer a genuine three-way split, but choosing one requires knowing
    the mesh's aspect ratios -- so Dispatch picks a safe split and says so, rather than
    guessing badly and silently.
    """
    best = (count, 1, 1)
    for a in range(1, int(count**0.5) + 1):
        if count % a == 0:
            best = (count // a, a, 1)
    return best


def set_decomposition(path: Path, subdomains: int, method: str | None = None) -> bool:
    """Bring a ``decomposeParDict`` in line with a new subdomain count.

    Updates ``numberOfSubdomains`` and, for geometric methods, the ``n`` vector that must
    multiply to it. Everything else -- the user's method, their comments, their formatting
    -- is left exactly as written.

    Returns:
        Whether the geometric coefficients had to be rewritten, so the caller can tell the
        user rather than changing their decomposition behind their back.
    """
    set_value(path, "numberOfSubdomains", str(subdomains))

    resolved = (method or read_value(path, "method") or "").strip()
    if resolved not in GEOMETRIC_METHODS:
        return False

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    match = _N_VECTOR.search(text)
    if match is None:
        return False

    x, y, z = balanced_factors(subdomains)
    updated = _N_VECTOR.sub(rf"\g<1>n\g<2>({x} {y} {z});", text, count=1)
    if updated == text:
        return False
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        log.warning("Could not update %s: %s", path, exc)
        return False
    log.info("Rewrote the %s coefficients in %s to (%d %d %d)", resolved, path, x, y, z)
    return True


HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     |                                                 |
|   \\\\  /    A nd           | Written by Dispatch                             |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {object};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def write_decompose_dict(path: Path, subdomains: int, method: str = "scotch") -> None:
    """Write a ``decomposeParDict`` for the requested number of subdomains.

    Only called when the case has none. An existing one is edited in place instead, so a
    user's hierarchical coefficients or manual decomposition survive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        HEADER.format(object="decomposeParDict")
        + f"numberOfSubdomains {subdomains};\n\n"
        + f"method          {method};\n\n"
        + "// Written by Dispatch because this case had no decomposeParDict.\n"
        + "// Edit it freely; Dispatch will only update numberOfSubdomains from now on.\n"
    )
    path.write_text(body, encoding="utf-8")
