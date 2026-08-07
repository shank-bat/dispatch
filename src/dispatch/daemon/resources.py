"""The resource ledger.

Admission is decided against a *ledger* -- the sum of what running jobs asked for -- and
never against measured CPU load. This is the single most consequential scheduling decision
in Dispatch, so it is worth restating why:

A solver blocked on I/O reads as idle. Measurement-based admission would therefore
oversubscribe the machine at exactly the moment it is already struggling, and the resulting
oscillation is very hard to diagnose after the fact. Deterministic bookkeeping is
predictable, testable, and is what every real scheduler does.

The consequence is that ``htop`` and Dispatch will sometimes disagree about how busy the
machine is. That is correct, and the dashboard shows both numbers.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dispatch.core.config import SchedulerConfig
from dispatch.core.models import ResourceRequest

__all__ = ["Capacity", "MemoryProbe", "ResourceModel"]

log = logging.getLogger(__name__)


class MemoryProbe(Protocol):
    """Reports available system memory in megabytes.

    Injected so the scheduler can be tested without psutil and without depending on how
    much RAM the test machine happens to have free.
    """

    def __call__(self) -> int: ...


def _psutil_available_mb() -> int:
    import psutil

    return int(psutil.virtual_memory().available / (1024 * 1024))


@dataclass(frozen=True, slots=True)
class Capacity:
    """Free resources, as offered to a scheduling policy.

    Deliberately a value object: a policy receives capacity and returns choices, and can
    subtract from a local copy while planning without touching the real ledger.
    """

    cores: int
    ram_mb: int | None = None

    def fits(self, request: ResourceRequest) -> bool:
        """Whether ``request`` fits in this capacity.

        A job with no RAM estimate is admitted on cores alone -- which is what makes RAM
        awareness already present rather than merely possible.
        """
        if request.cores > self.cores:
            return False
        if request.ram_mb is not None and self.ram_mb is not None:
            return request.ram_mb <= self.ram_mb
        return True

    def reserve(self, request: ResourceRequest) -> Capacity:
        """Return the capacity remaining after notionally taking ``request``."""
        return Capacity(
            cores=max(0, self.cores - request.cores),
            ram_mb=(None if self.ram_mb is None else max(0, self.ram_mb - (request.ram_mb or 0))),
        )


class ResourceModel:
    """Tracks what Dispatch has handed out, and what is left.

    Args:
        config: Scheduler settings -- reservations, margins, disk floor.
        memory_probe: How to read available RAM. Defaults to psutil.
        log_dir: Filesystem checked for the free-space floor.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        *,
        memory_probe: MemoryProbe | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._probe = memory_probe or _psutil_available_mb
        self._log_dir = log_dir
        self.total_cores = config.resolve_total_cores()
        self.reserved_cores = min(config.reserved_cores, max(0, self.total_cores - 1))
        self._allocations: dict[str, ResourceRequest] = {}

    # -- ledger -------------------------------------------------------------------------

    @property
    def allocated_cores(self) -> int:
        """Cores currently handed out to PREPARING and RUNNING jobs."""
        return sum(request.cores for request in self._allocations.values())

    @property
    def allocated_ram_mb(self) -> int:
        """Sum of RAM estimates for jobs that declared one."""
        return sum(request.ram_mb or 0 for request in self._allocations.values())

    @property
    def free_cores(self) -> int:
        """Cores available to new jobs."""
        return max(0, self.total_cores - self.reserved_cores - self.allocated_cores)

    @property
    def schedulable_cores(self) -> int:
        """The largest core count any single job could ever be granted.

        A job requesting more than this can never start, which is a validation error at
        submission rather than a job that waits in the queue forever.
        """
        return max(1, self.total_cores - self.reserved_cores)

    def holds(self, job_id: str) -> bool:
        """Whether the ledger has an allocation for this job."""
        return job_id in self._allocations

    def capacity(self) -> Capacity:
        """Current free capacity, for a scheduling policy."""
        return Capacity(cores=self.free_cores, ram_mb=self.available_ram_mb())

    def available_ram_mb(self) -> int | None:
        """RAM available for new jobs, less the configured margin.

        Returns ``None`` if the probe fails -- an unreadable meminfo should degrade to
        core-only scheduling, not stop the queue.
        """
        try:
            available = self._probe()
        except Exception as exc:  # pragma: no cover - psutil failure is environmental
            log.warning("Could not read available memory: %s", exc)
            return None
        return max(0, available - self._config.ram_margin_mb)

    def acquire(self, job_id: str, request: ResourceRequest) -> None:
        """Record an allocation.

        Idempotent by job id, so replaying recovery over an already-restored allocation
        cannot double-count the machine.
        """
        self._allocations[job_id] = request

    def release(self, job_id: str) -> None:
        """Release a job's allocation. Silent if it holds none."""
        self._allocations.pop(job_id, None)

    def clear(self) -> None:
        """Drop every allocation. Used when rebuilding the ledger at startup."""
        self._allocations.clear()

    # -- admission gates -------------------------------------------------------------------

    def can_admit(self, request: ResourceRequest) -> tuple[bool, str | None]:
        """Whether a job could start right now, and if not, why not.

        The reason string is shown to the user in dry-run output and in the queue view,
        because "waiting" without a cause is the least useful thing a scheduler can say.
        """
        if request.cores > self.schedulable_cores:
            return False, (
                f"requests {request.cores} cores but only {self.schedulable_cores} are "
                f"schedulable ({self.total_cores} total, {self.reserved_cores} reserved)"
            )
        if request.cores > self.free_cores:
            return False, f"{self.free_cores} of {self.schedulable_cores} cores free"

        available = self.available_ram_mb()
        if request.ram_mb is not None and available is not None and request.ram_mb > available:
            return False, f"needs {request.ram_mb} MB but {available} MB is available"

        ok, reason = self.check_disk()
        if not ok:
            return False, reason
        return True, None

    def check_disk(self) -> tuple[bool, str | None]:
        """Whether the log filesystem has room to start another job.

        A solver that fills the disk mid-run loses its output and can leave its case in a
        state that needs manual repair. Not starting is strictly better.
        """
        if self._log_dir is None:
            return True, None
        try:
            usage = shutil.disk_usage(self._log_dir)
        except OSError as exc:
            log.warning("Could not check free space on %s: %s", self._log_dir, exc)
            return True, None
        free_mb = usage.free // (1024 * 1024)
        if free_mb < self._config.min_free_disk_mb:
            return False, (
                f"only {free_mb} MB free on {self._log_dir}; "
                f"{self._config.min_free_disk_mb} MB is required"
            )
        return True, None

    def load_average(self) -> tuple[float, float, float]:
        """The kernel's load average. Display only -- never used for admission."""
        try:
            one, five, fifteen = os.getloadavg()
        except OSError:  # pragma: no cover - unavailable on exotic kernels
            return (0.0, 0.0, 0.0)
        return (one, five, fifteen)
