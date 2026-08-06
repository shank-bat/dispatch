"""Job lifecycle states and the transitions between them.

The transition table is the single authority on what may happen to a job. It is enforced
in exactly one place (``JobRepository.transition``) using a conditional ``UPDATE``, so an
illegal move is impossible even if two coroutines race.

See ``docs/ARCHITECTURE.md`` §4.2 for the diagram.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

__all__ = [
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "ExitReason",
    "JobState",
    "can_transition",
]


class JobState(StrEnum):
    """Where a job is in its life.

    :class:`StrEnum` so the value stored in SQLite is the readable name. A ``sqlite3``
    session at 2am should show ``RUNNING``, not ``3``.
    """

    QUEUED = "QUEUED"
    """Waiting for resources. The only state from which a job may be admitted."""

    HELD = "HELD"
    """Withheld by the user. Skipped by the scheduler until released."""

    PREPARING = "PREPARING"
    """Running the plan's PREPARE steps (decomposePar, compilation, ...).

    Resources are already allocated: preparation uses the machine, and a job that is
    decomposing must not have its cores handed to somebody else.
    """

    RUNNING = "RUNNING"
    """The SOLVE step is executing."""

    COMPLETED = "COMPLETED"
    """Finished with exit code 0."""

    FAILED = "FAILED"
    """Finished with a non-zero exit code, died on a signal, or a PREPARE step aborted."""

    CANCELLED = "CANCELLED"
    """Stopped at the user's request."""

    REJECTED = "REJECTED"
    """Never ran: pre-flight validation failed and the user did not override."""

    UNKNOWN = "UNKNOWN"
    """The outcome is genuinely not knowable.

    Reached when a job was RUNNING, the daemon restarted, the process is gone, and no exit
    sentinel was written -- i.e. the machine was reset mid-run. Recording this as FAILED
    would be a plausible-looking lie, and a lie in the history is worse than a gap.
    """


TERMINAL_STATES: Final[frozenset[JobState]] = frozenset(
    {
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.REJECTED,
        JobState.UNKNOWN,
    }
)
"""States from which nothing further happens. Safe to delete, archive, or retag."""

ACTIVE_STATES: Final[frozenset[JobState]] = frozenset({JobState.PREPARING, JobState.RUNNING})
"""States that hold an allocation in the resource ledger."""

QUEUEABLE_STATES: Final[frozenset[JobState]] = frozenset({JobState.QUEUED, JobState.HELD})
"""States in which a job exists but has not yet consumed resources."""


_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset(
        {JobState.HELD, JobState.PREPARING, JobState.CANCELLED, JobState.REJECTED}
    ),
    JobState.HELD: frozenset({JobState.QUEUED, JobState.CANCELLED}),
    # PREPARING may reach RUNNING (normal), FAILED (a PREPARE step aborted),
    # CANCELLED (user), or UNKNOWN (daemon died mid-preparation).
    JobState.PREPARING: frozenset(
        {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED, JobState.UNKNOWN}
    ),
    JobState.RUNNING: frozenset(
        {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.UNKNOWN}
    ),
    # Terminal states have no outgoing edges. Re-running a case creates a new job with a
    # new id, which keeps history immutable and makes "which run produced this?" answerable.
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
    JobState.REJECTED: frozenset(),
    JobState.UNKNOWN: frozenset(),
}

TRANSITIONS: Final[Mapping[JobState, frozenset[JobState]]] = MappingProxyType(_TRANSITIONS)
"""Allowed state changes, as ``source -> {permitted targets}``. Read-only."""


def can_transition(source: JobState, target: JobState) -> bool:
    """Return whether a job may move from ``source`` to ``target``.

    A self-transition is always false: re-entering a state is not a state change, and
    treating it as one would append a misleading row to the event log.
    """
    return target in TRANSITIONS[source]


class ExitReason(StrEnum):
    """Why a job stopped. Complements the numeric exit code with an interpretation."""

    OK = "ok"
    """Exit code 0."""

    NONZERO = "nonzero"
    """The solver ran and returned a non-zero code."""

    SIGNAL = "signal"
    """Killed by a signal. The signal name is recorded alongside, e.g. ``signal:TERM``."""

    OOM = "oom"
    """Killed by the kernel out-of-memory killer. Distinguished from a plain SIGKILL
    because it means the RAM estimate was wrong, which is actionable."""

    PREPARE_FAILED = "prepare_failed"
    """A PREPARE step aborted; the solver never started."""

    CANCELLED = "cancelled"
    """Stopped on user request."""

    LOST = "lost"
    """The process vanished without leaving an exit sentinel (see :attr:`JobState.UNKNOWN`)."""

    TIMEOUT = "timeout"
    """A step exceeded its configured timeout."""
