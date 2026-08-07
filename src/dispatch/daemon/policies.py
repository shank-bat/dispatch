"""Scheduling policies.

The scheduler never names a concrete policy. It asks a :class:`SchedulingPolicy` which of
the queued jobs to start given the free capacity, and that object comes from configuration::

    [scheduler]
    policy = "backfill"

An unknown name is a startup error listing the valid ones, never a silent fallback to the
default. A typo that quietly changes how the machine schedules for six months is a far
worse outcome than a daemon that refuses to start and says why.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from dispatch.core.config import SchedulerConfig
from dispatch.core.errors import ConfigError
from dispatch.core.models import Job
from dispatch.daemon.resources import Capacity

__all__ = [
    "POLICIES",
    "FifoPolicy",
    "PriorityFifoBackfill",
    "SchedulingPolicy",
    "build_policy",
]


@runtime_checkable
class SchedulingPolicy(Protocol):
    """Chooses which queued jobs to start.

    Receives jobs already sorted into queue order (priority descending, then submission
    order) and the free capacity. Returns the subset to admit, in the order to start them.

    Implementations must be pure: no I/O, no clock, no database. That is what lets the
    scheduler tests run a week of simulated queue behaviour instantly.
    """

    name: str

    def select(self, queued: Sequence[Job], capacity: Capacity) -> Sequence[Job]: ...


class FifoPolicy:
    """Strict priority, then submission order. Stops at the first job that does not fit.

    Head-of-line blocking and all: if the front job wants 24 cores, nothing behind it runs
    until it does. Predictable, and occasionally what you want -- if the queue order you
    see is the order you expect things to run in, this is the policy that guarantees it.
    """

    name = "fifo"

    def select(self, queued: Sequence[Job], capacity: Capacity) -> Sequence[Job]:
        chosen: list[Job] = []
        remaining = capacity
        for job in queued:
            if not remaining.fits(job.resources):
                break
            chosen.append(job)
            remaining = remaining.reserve(job.resources)
        return chosen


class PriorityFifoBackfill:
    """Priority and FIFO order, with starvation-bounded backfill. The default.

    Walks the queue in order. A job that fits starts. A job that does not fit is *skipped*,
    and smaller jobs behind it may use the idle cores -- but only for a bounded number of
    passes. Once a job has been skipped :attr:`max_skips` times it claims everything that
    is left, and nothing behind it starts until it does.

    Both halves are load-bearing, and the naive version of each fails:

    * Reserving the blocked job's full requirement immediately (textbook conservative
      backfill) does nothing useful here. When an 8-core job is blocked on a machine with
      4 cores free, its reservation swallows all 4, so no small job ever backfills -- in
      exactly the situation backfill exists for.
    * Never reserving at all lets a steady trickle of small jobs starve the large one
      indefinitely, which on a workstation means the production run never starts.

    Counting skips bounds the wait explicitly: a blocked job gives way at most
    ``max_skips`` times, then takes precedence. Real EASY backfill would instead compute
    the blocked job's start time from expected completions, which needs runtime estimates
    Dispatch only begins accumulating once it has history. This is the honest approximation
    until then.

    The skip counter is the policy's only state. It is keyed by job id and pruned to the
    live queue on every pass, so it cannot grow.

    Args:
        max_skips: Passes a blocked job yields before claiming the machine.
    """

    name = "backfill"

    def __init__(self, max_skips: int = 3) -> None:
        self.max_skips = max(1, max_skips)
        self._skips: dict[str, int] = {}

    def select(self, queued: Sequence[Job], capacity: Capacity) -> Sequence[Job]:
        self._prune(queued)

        chosen: list[Job] = []
        remaining = capacity

        for job in queued:
            if remaining.fits(job.resources):
                chosen.append(job)
                remaining = remaining.reserve(job.resources)
                self._skips.pop(job.id, None)
                continue

            skips = self._skips.get(job.id, 0) + 1
            self._skips[job.id] = skips
            if skips >= self.max_skips:
                # This job has yielded often enough. Everything still free is now its.
                break

        return chosen

    def _prune(self, queued: Sequence[Job]) -> None:
        """Forget jobs that have left the queue, so the counter cannot grow."""
        live = {job.id for job in queued}
        for job_id in list(self._skips):
            if job_id not in live:
                del self._skips[job_id]

    def skips(self, job_id: str) -> int:
        """How many times a job has been passed over. Diagnostic, and asserted in tests."""
        return self._skips.get(job_id, 0)


class ShortestFirstPolicy:
    """Smallest core count first, within each priority band.

    Maximises the number of jobs running at once, at the cost of making queue order
    unpredictable. Offered because on a workstation the queue is often a mix of one big
    production run and several quick checks, and getting the quick ones through first is
    frequently what the user actually wants.

    Starvation is bounded by respecting priority bands: raising a large job's priority
    always gets it considered before anything smaller.
    """

    name = "shortest-first"

    def select(self, queued: Sequence[Job], capacity: Capacity) -> Sequence[Job]:
        chosen: list[Job] = []
        remaining = capacity
        for _, band in _by_priority(queued):
            for job in sorted(band, key=lambda j: (j.cores, j.seq)):
                if remaining.fits(job.resources):
                    chosen.append(job)
                    remaining = remaining.reserve(job.resources)
        return chosen


def _by_priority(queued: Sequence[Job]) -> Sequence[tuple[int, list[Job]]]:
    """Group jobs into descending priority bands, preserving order within each."""
    bands: dict[int, list[Job]] = {}
    for job in queued:
        bands.setdefault(job.priority, []).append(job)
    return sorted(bands.items(), key=lambda item: item[0], reverse=True)


POLICIES: Mapping[str, type] = {
    FifoPolicy.name: FifoPolicy,
    PriorityFifoBackfill.name: PriorityFifoBackfill,
    ShortestFirstPolicy.name: ShortestFirstPolicy,
}
"""Available policies by configuration name.

``easy-backfill`` is reserved for a future policy that uses recorded runtimes to compute
real reservation windows; it needs the runtime history that Dispatch is only now starting
to accumulate.
"""


def build_policy(config: SchedulerConfig) -> SchedulingPolicy:
    """Construct the configured policy.

    Raises:
        ConfigError: If the name is not registered, listing the ones that are.
    """
    name = config.policy.strip().lower()
    policy_cls = POLICIES.get(name)
    if policy_cls is None:
        raise ConfigError(
            f"Unknown scheduler policy {config.policy!r}. "
            f"Valid policies: {', '.join(sorted(POLICIES))}",
            detail={"requested": config.policy, "available": sorted(POLICIES)},
        )
    policy: SchedulingPolicy = policy_cls()
    return policy
