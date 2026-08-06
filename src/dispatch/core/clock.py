"""Time, as an injected dependency.

Scheduling, retention, and runtime accounting are all time-dependent, and tests for them
must be instant and deterministic. Every component that needs the time takes a :class:`Clock`
rather than calling :func:`time.time` directly, so a test can hand it a :class:`FakeClock`
and drive a week of simulated uptime in a millisecond.

Two distinct notions of time are exposed, and confusing them causes real bugs:

``now()``
    Wall-clock Unix timestamp. Stored in the database, shown to users, compared against
    dates in search queries. Can jump backwards (NTP, DST is irrelevant for epochs but
    corrections are not).

``monotonic()``
    Elapsed time from an arbitrary origin. Used for timeouts and durations. Never jumps.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "FakeClock", "SystemClock"]


@runtime_checkable
class Clock(Protocol):
    """A source of the current time."""

    def now(self) -> float:
        """Return the current wall-clock time as a Unix timestamp."""
        ...

    def monotonic(self) -> float:
        """Return a monotonically increasing time in seconds, for measuring durations."""
        ...


class SystemClock:
    """The real clock. The only implementation used outside tests."""

    __slots__ = ()

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()


class FakeClock:
    """A manually advanced clock for deterministic tests.

    Args:
        start: Initial wall-clock value. Defaults to 2026-01-01T00:00:00Z, chosen because
            it is recent, round, and obviously artificial in test output.
    """

    __slots__ = ("_mono", "_now")

    def __init__(self, start: float = 1_767_225_600.0) -> None:
        self._now = start
        self._mono = 0.0

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        """Move both clocks forward by ``seconds``."""
        if seconds < 0:
            raise ValueError("cannot advance a clock backwards; use set() instead")
        self._now += seconds
        self._mono += seconds

    def set(self, wall: float) -> None:
        """Jump wall-clock time without touching the monotonic clock.

        Models an NTP correction, which is exactly the situation where code that used
        wall time for a duration reveals its bug.
        """
        self._now = wall
