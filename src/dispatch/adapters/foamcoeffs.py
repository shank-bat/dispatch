"""Reading lift and drag out of an OpenFOAM ``forceCoeffs`` function object.

A residual plot answers "is it converging". For anything with a wing on it the question is
"what is the lift", and OpenFOAM already writes the answer -- just not in the log. The
``forceCoeffs`` function object puts it in a file under ``postProcessing/``, and the awkward
part is finding that file rather than reading it.

**There is no single path to assume.** All of these occur, and three of them occur in one
real projects tree::

    postProcessing/forceCoeffs/0/coefficient.dat
    postProcessing/forceCoeffs/0/coefficient_0.dat
    postProcessing/forceCoeffs/0.0004319995976/coefficient.dat
    postProcessing/forceCoeffs1/0/forceCoeffs.dat

The function object's directory is named by the user, so it is only *conventionally* called
``forceCoeffs``. The time directory beneath it is the time the function object started, which
after a restart is not ``0`` and is not an integer. The filename gained its ``coefficient``
spelling in OpenFOAM v2012 and takes a ``_0`` suffix when the object restarts within the same
time directory. So the file is found by **searching and ranking**, not by construction:
every candidate that parses is a real answer, and the most recently written one is the
current run's.

Column names come from the file's own header rather than a fixed list, because the set
depends on the function object's settings -- ``CmPitch`` and the front/rear splits appear
only when asked for. Whatever is there is offered; ``Cl`` and ``Cd`` are simply the two that
are always there.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

from dispatch.core.series import PlotData, Series, series_from_records

__all__ = [
    "COEFFICIENT_PATTERNS",
    "FORCE_COEFF_DIRS",
    "find_coefficient_file",
    "latest_coefficients",
    "parse_coefficients",
]

log = logging.getLogger(__name__)

POST_PROCESSING: Final = "postProcessing"

FORCE_COEFF_DIRS: Final = ("forceCoeffs*", "forces*")
"""Function-object directory names that hold coefficient output.

Globs, because the directory is named by whatever the user called the function object in
``controlDict``: ``forceCoeffs``, ``forceCoeffs1``, ``forceCoeffsIncompressible`` are all
ordinary. ``forces*`` is included because a ``forces`` object with ``writeFields`` also
writes a coefficient file.
"""

COEFFICIENT_PATTERNS: Final = ("coefficient*.dat", "forceCoeffs*.dat")
"""Filenames within a time directory. ``coefficient`` since v2012, ``forceCoeffs`` before."""

TIME_COLUMN: Final = "Time"

LIFT_KEYS: Final = ("Cl", "CL", "cl")
DRAG_KEYS: Final = ("Cd", "CD", "cd")
"""Spellings of the two columns that matter, in the order they are looked for.

OpenFOAM writes ``Cl``/``Cd``; the others are here because a hand-edited or third-party
file is cheap to tolerate and expensive to debug.
"""

LABELS: Final[dict[str, str]] = {
    "Cl": "Cl (lift)",
    "Cd": "Cd (drag)",
    "CmPitch": "CmPitch",
    "CmRoll": "CmRoll",
    "CmYaw": "CmYaw",
    "Cs": "Cs (side)",
}
"""Friendlier names for the columns a reader has to recognise at a glance.

Only the ones worth expanding. Everything else keeps the name the file gave it, which is
what the user will grep for.
"""

MAX_BYTES: Final = 8 * 1024 * 1024
"""How much of a coefficient file to read.

Far smaller than a log limit and still generous: one row per write of perhaps 200 bytes is
tens of thousands of writes. A file larger than this is read from its end, so a very long
run still plots its recent history.
"""

_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def _candidates(case: Path) -> Iterator[Path]:
    """Every file that might hold coefficients, without reading any of them."""
    root = case / POST_PROCESSING
    if not root.is_dir():
        return
    for group in FORCE_COEFF_DIRS:
        try:
            directories = sorted(root.glob(group))
        except OSError:  # pragma: no cover - unreadable postProcessing
            continue
        for directory in directories:
            if not directory.is_dir():
                continue
            # One level of time directory. Globbing deeper would pick up unrelated output
            # from a differently-shaped function object.
            for time_dir in sorted(directory.iterdir()):
                if not time_dir.is_dir():
                    continue
                for pattern in COEFFICIENT_PATTERNS:
                    yield from (path for path in sorted(time_dir.glob(pattern)) if path.is_file())


def find_coefficient_file(case: Path) -> Path | None:
    """The most recently written coefficient file in a case, or ``None``.

    Ranked by modification time rather than by the time directory's name, and that is the
    point. A case restarted at ``t = 4.5`` has a ``0/`` directory holding the first run's
    output and a ``4.5/`` directory holding the current one; sorting the *names* means
    deciding whether ``0.0004319995976`` sorts above ``4.5`` as text or as a number, and
    getting it wrong shows a plot of the run before last. The filesystem already knows which
    file is being written, so it is asked.

    A candidate has to actually parse. An empty file -- which is what a function object that
    has started but not yet written looks like -- is skipped in favour of an older one that
    has content, because a plot of the previous run is more use than an empty chart.
    """
    best: tuple[float, Path] | None = None
    for path in _candidates(case):
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if best is not None and modified <= best[0]:
            continue
        if not _has_data(path):
            continue
        best = (modified, path)
    return None if best is None else best[1]


def _has_data(path: Path) -> bool:
    """Whether a candidate holds a readable header and at least one row.

    Cheap: the header is in the first kilobyte or it is not a coefficient file at all.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(4096).decode("utf-8", errors="replace")
    except OSError:
        return False
    if _header(head.splitlines()) is None:
        return False
    # A header alone is a function object that has written nothing yet.
    return any(line.strip() and not line.startswith("#") for line in head.splitlines())


def _header(lines: Sequence[str]) -> list[str] | None:
    """Column names from the last commented header line that names a time and a coefficient.

    *Last*, because the file opens with a block of commented metadata -- reference area,
    lift direction, free-stream speed -- and the column header is the final line of it. A
    restarted function object appends a second header, and the later one describes the rows
    that follow it.
    """
    found: list[str] | None = None
    for line in lines:
        if not line.startswith("#"):
            continue
        columns = line.lstrip("#").split()
        if len(columns) < 2 or columns[0] != TIME_COLUMN:
            continue
        if not any(name in LIFT_KEYS + DRAG_KEYS for name in columns):
            continue
        found = columns
    return found


def parse_coefficients(text: str, *, truncated: bool = False) -> PlotData:
    """Turn a coefficient file's contents into plottable series.

    Args:
        text: The file, or its end.
        truncated: Whether only the end was read, so the caller can say so.

    Returns:
        One series per column the file actually has, with ``Time`` marked as the axis.
        Empty when the text holds no recognisable header or no complete rows -- both of
        which are ordinary states for a file a solver is still writing.
    """
    lines = text.splitlines()
    columns = _header(lines)
    if columns is None:
        return PlotData(truncated=truncated)

    records: list[dict[str, float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        # A row shorter than the header is the one being written right now: the solver has
        # flushed part of a line and not the newline. Dropping it costs the newest point and
        # avoids a column being read from the wrong position.
        if len(fields) != len(columns):
            continue
        row: dict[str, float] = {}
        for name, raw in zip(columns, fields, strict=True):
            value = _number(raw)
            if value is not None:
                row[name] = value
        if TIME_COLUMN in row and len(row) > 1:
            records.append(row)

    if not records:
        return PlotData(truncated=truncated)

    seen = [name for name in columns if any(name in record for record in records)]
    series: Sequence[Series] = series_from_records(
        records,
        labels={name: LABELS.get(name, name) for name in seen},
        axes=[TIME_COLUMN],
        order=(TIME_COLUMN, *LIFT_KEYS, *DRAG_KEYS),
    )
    return PlotData(series=series, samples=len(records), truncated=truncated)


def latest_coefficients(data: PlotData) -> dict[str, float]:
    """The final value of every series, for a compact "where is it now" display.

    Keyed by the file's own column names, so a caller asking for ``Cl`` gets ``Cl``.
    """
    latest: dict[str, float] = {}
    for item in data.series:
        if item.values:
            latest[item.key] = item.values[-1]
    return latest


def pick(latest: dict[str, float], names: Sequence[str]) -> float | None:
    """The first of ``names`` present, so a caller can ask for lift without knowing the
    spelling this file used."""
    for name in names:
        if name in latest:
            return latest[name]
    return None


def _number(raw: str) -> float | None:
    """Parse one field, rejecting anything that is not a plain number.

    A regex first because ``float`` accepts ``nan`` and ``inf``, which a diverging run does
    produce and which poison every axis calculation downstream.
    """
    if not _NUMBER.match(raw):
        return None
    try:
        value = float(raw)
    except ValueError:  # pragma: no cover - the pattern already guaranteed this
        return None
    return value if value == value and abs(value) != float("inf") else None
