"""Reading numbers out of an OpenFOAM solver log.

Foam solvers print a fixed set of shapes, once per time step, interleaved with whatever
else the case's function objects have to say::

    Time = 0.005

    Courant Number mean: 0.0182 max: 0.4123
    smoothSolver:  Solving for Ux, Initial residual = 0.0234, Final residual = 1.2e-06, No It 3
    smoothSolver:  Solving for Uy, Initial residual = 0.0198, Final residual = 9.7e-07, No It 3
    GAMG:  Solving for p, Initial residual = 0.9012, Final residual = 0.0081, No Iterations 12
    time step continuity errors : sum local = 1.1e-09, global = -1.3e-19, cumulative = -1.3e-19
    ExecutionTime = 0.05 s  ClockTime = 0 s

Three decisions shape everything here.

**Time steps are the record boundary.** ``Time = `` opens a new record and everything up
to the next one belongs to it. That is what makes ``residual(p)`` and ``residual(Ux)``
comparable point by point without assuming they appear equally often.

**The first initial residual of each field per step is the one kept.** A PIMPLE run solves
for pressure several times within a step; the residual a person plots is the one at the
start of the step, which is what every hand-rolled ``foamLog`` script also takes. The
later inner-loop values are a different quantity and averaging them would smooth away
exactly the divergence somebody opened the plot to look for.

**Only what appeared is reported.** A 2-D case has no ``Uz``, so it gets no
``residual(Uz)`` series -- not an empty one, and not one full of zeros.

Nothing here can change a job's state. Parsing runs when a user asks for a plot, on a copy
of the log's text, and its worst failure is an empty chart.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from dispatch.core.series import PlotData, Series, series_from_records

__all__ = ["parse_foam_log"]

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

TIME_LINE: Final = re.compile(rf"^Time\s*=\s*({_NUMBER})\s*$")
RESIDUAL_LINE: Final = re.compile(
    rf"Solving for\s+([A-Za-z_][\w.()]*)\s*,\s*Initial residual\s*=\s*({_NUMBER})"
)
FINAL_RESIDUAL: Final = re.compile(rf"Final residual\s*=\s*({_NUMBER})")
COURANT_LINE: Final = re.compile(
    rf"Courant Number\s+mean:\s*({_NUMBER})\s+max:\s*({_NUMBER})"
)
CONTINUITY_LINE: Final = re.compile(rf"continuity errors\s*:.*?cumulative\s*=\s*({_NUMBER})")
EXECUTION_LINE: Final = re.compile(
    rf"^ExecutionTime\s*=\s*({_NUMBER})\s*s\s+ClockTime\s*=\s*({_NUMBER})\s*s"
)

RANK_PREFIX: Final = re.compile(r"^\[\d+\]\s?")
"""``mpirun`` labels every line of a parallel run with its rank.

Stripped before anything else for the same reason failure summaries strip it: in the
parallel case -- which is the case that matters -- nothing anchored to a line start
matches otherwise, and ``Time = `` is anchored.
"""

ITERATION: Final = "iteration"
TIME: Final = "time"
EXECUTION_TIME: Final = "execution_time"
CLOCK_TIME: Final = "clock_time"
CONTINUITY: Final = "continuity"
COURANT_MEAN: Final = "courant_mean"
COURANT_MAX: Final = "courant_max"

AXIS_KEYS: Final = (ITERATION, TIME, EXECUTION_TIME, CLOCK_TIME)
"""Quantities that increase monotonically through a run, so they read as an X axis.

Execution and clock time are on the list because "how long did the last thousand steps
take" is a real question, answered by plotting execution time against iteration -- and
occasionally by putting execution time on X instead.
"""

LABELS: Final[dict[str, str]] = {
    ITERATION: "Iteration",
    TIME: "Time",
    EXECUTION_TIME: "Execution time",
    CLOCK_TIME: "Clock time",
    CONTINUITY: "Cumulative continuity error",
    COURANT_MEAN: "Courant number (mean)",
    COURANT_MAX: "Courant number (max)",
}

UNITS: Final[dict[str, str]] = {TIME: "s", EXECUTION_TIME: "s", CLOCK_TIME: "s"}

ORDER: Final[tuple[str, ...]] = (
    ITERATION,
    TIME,
    EXECUTION_TIME,
    CLOCK_TIME,
    CONTINUITY,
    COURANT_MEAN,
    COURANT_MAX,
)
"""Preferred ordering. Residual series are unlisted, so they follow, sorted by field name."""


def residual_key(field: str) -> str:
    """The series key for one field's initial residual."""
    return f"residual({field})"


def parse_foam_log(text: str, *, truncated: bool = False) -> PlotData:
    """Extract every plottable series an OpenFOAM log actually contains.

    Args:
        text: The log, or its last portion.
        truncated: Whether ``text`` is only the end of a longer log, so the caller can
            say so rather than presenting part of a run as the whole of it.

    Returns:
        The available series. Empty when the log contains no recognisable time steps --
        a case that has not started yet, or a solver whose output shape is different.
        That is a valid answer and the interface reports it as one.
    """
    records: list[dict[str, float]] = []
    fields: list[str] = []
    current: dict[str, float] | None = None
    seen_in_step: set[str] = set()

    for raw in text.splitlines():
        line = RANK_PREFIX.sub("", raw).strip()
        if not line:
            continue

        start = TIME_LINE.match(line)
        if start is not None:
            value = _number(start.group(1))
            if value is None:
                # A malformed time is a reason to ignore the line, not to lose the run.
                continue
            current = {ITERATION: float(len(records) + 1), TIME: value}
            records.append(current)
            seen_in_step = set()
            continue

        if current is None:
            # Header, banner, or mesh statistics: everything before the first time step.
            continue

        _absorb(line, current, seen_in_step, fields)

    if not records:
        return PlotData(series=(), samples=0, truncated=truncated)

    labels = dict(LABELS)
    for field in fields:
        labels[residual_key(field)] = f"residual({field})"

    series: Sequence[Series] = series_from_records(
        records, labels=labels, units=UNITS, axes=AXIS_KEYS, order=ORDER
    )
    return PlotData(series=series, samples=len(records), truncated=truncated)


def _absorb(
    line: str, record: dict[str, float], seen: set[str], fields: list[str]
) -> None:
    """Fold one line of a time step into that step's record.

    Silent about anything it does not recognise: a Foam log is mostly lines this parser
    has no opinion about, and treating an unfamiliar one as an error would mean no case
    with a function object ever plotted.
    """
    residual = RESIDUAL_LINE.search(line)
    if residual is not None:
        field = residual.group(1)
        # First per step only. Later inner iterations of the same field are a different
        # quantity, and mixing them into one series makes a converging run look erratic.
        if field in seen:
            return
        value = _number(residual.group(2))
        if value is None:
            return
        seen.add(field)
        if field not in fields:
            fields.append(field)
        record[residual_key(field)] = value
        return

    courant = COURANT_LINE.search(line)
    if courant is not None:
        mean, peak = _number(courant.group(1)), _number(courant.group(2))
        # Later Courant lines within a step (interFoam prints one per PIMPLE loop) are
        # allowed to overwrite: the last one describes the step that was actually taken.
        if mean is not None:
            record[COURANT_MEAN] = mean
        if peak is not None:
            record[COURANT_MAX] = peak
        return

    continuity = CONTINUITY_LINE.search(line)
    if continuity is not None:
        value = _number(continuity.group(1))
        if value is not None:
            record[CONTINUITY] = value
        return

    timing = EXECUTION_LINE.match(line)
    if timing is not None:
        execution, clock = _number(timing.group(1)), _number(timing.group(2))
        if execution is not None:
            record[EXECUTION_TIME] = execution
        if clock is not None:
            record[CLOCK_TIME] = clock


def _number(raw: str) -> float | None:
    """Parse a float, returning ``None`` rather than raising.

    A log is the least trustworthy text in the system: it can be truncated mid-line by a
    kill, interleaved by ``mpirun``, or contain ``nan`` from a diverging run. Every one of
    those must produce a shorter plot, never an exception.
    """
    try:
        value = float(raw)
    except ValueError:
        return None
    # NaN and infinity are real outputs of a diverging solver, and they poison every
    # axis-scaling calculation downstream. Dropping the point leaves the divergence
    # visible in the residual that came before it, which is the informative part.
    return value if value == value and abs(value) != float("inf") else None
