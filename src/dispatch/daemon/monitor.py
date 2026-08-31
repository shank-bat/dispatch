"""Measuring the machine and the jobs on it.

Two samplers, both demand-driven, because measurement that nobody reads is pure waste on a
process expected to idle for months.

:class:`SystemMonitor`
    Samples CPU and memory for the dashboard. Its timer starts on the first subscriber and
    stops on the last unsubscribe. Nobody watching means nothing measured.

:class:`JobSampler`
    Samples each running job's process tree every 30 seconds for peak RSS and mean CPU,
    and reads the last few kilobytes of its log to extract progress. Runs only while jobs
    are running.

The log-tail reading is what recovers the one feature lost by writing solver output
straight to disk (§13.4): one small read per job per 30 seconds, instead of putting the
daemon in the path of every byte a solver prints for a week.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from dispatch.adapters.registry import AdapterRegistry
from dispatch.core.clock import Clock, SystemClock
from dispatch.core.config import DaemonConfig
from dispatch.core.errors import DispatchError
from dispatch.core.models import Sample, SystemSnapshot
from dispatch.daemon.events import EventBus
from dispatch.daemon.resources import ResourceModel
from dispatch.db.repository import JobRepository
from dispatch.ipc.protocol import Event, Topic, encode_snapshot

__all__ = ["JobSampler", "SystemMonitor"]

log = logging.getLogger(__name__)

LOG_TAIL_BYTES = 8192
"""How much of a log's end to read for progress. Enough for several output blocks."""


class SystemMonitor:
    """Samples the machine for the dashboard, only while somebody is watching.

    Args:
        resources: The ledger, so a snapshot can report allocated alongside measured.
        bus: Where snapshots are published.
        config: Sampling interval.
        clock: Time source.
    """

    def __init__(
        self,
        *,
        resources: ResourceModel,
        bus: EventBus,
        config: DaemonConfig,
        clock: Clock | None = None,
    ) -> None:
        self._resources = resources
        self._bus = bus
        self._config = config
        self._clock = clock or SystemClock()
        self._task: asyncio.Task[None] | None = None
        self._hostname = os.uname().nodename
        self._boot = _boot_time()

    @property
    def sampling(self) -> bool:
        """Whether the sampling timer is currently running."""
        return self._task is not None and not self._task.done()

    def on_demand_change(self, topics: set[str]) -> None:
        """Start or stop sampling as subscribers come and go.

        Wired to the event bus, which calls this whenever the set of subscribed topics
        changes. This is the mechanism behind "no subscribers means no sampling".
        """
        if str(Topic.SYSTEM) in topics:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        """Begin sampling, if not already."""
        if self.sampling:
            return
        self._task = asyncio.create_task(self._loop(), name="dispatch-system-monitor")

    def stop(self) -> None:
        """Stop sampling."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        try:
            self.snapshot()  # prime psutil's CPU deltas
            while True:
                await asyncio.sleep(self._config.sample_interval_s)
                self._bus.publish(Event.SYSTEM_STATS, encode_snapshot(self.snapshot()))
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            log.exception("System monitor stopped unexpectedly")

    def snapshot(self) -> SystemSnapshot:
        """Take one measurement of the machine.

        Reports the ledger's ``allocated_cores`` alongside the measured ``cpu_percent``.
        The two disagree whenever a solver blocks on I/O, and showing both is the honest
        presentation: one is what Dispatch has promised, the other is what the CPUs are
        doing.

        GPUs are reported from the ledger only. There is no measured counterpart, and
        inventing one would mean shelling out to a vendor tool on every dashboard tick --
        for a number that must not influence admission anyway (§13.2).
        """
        import psutil

        memory = psutil.virtual_memory()
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        return SystemSnapshot(
            timestamp=self._clock.now(),
            hostname=self._hostname,
            total_cores=self._resources.total_cores,
            allocated_cores=self._resources.allocated_cores,
            reserved_cores=self._resources.reserved_cores,
            cpu_percent=sum(per_core) / len(per_core) if per_core else 0.0,
            per_core_percent=tuple(per_core),
            total_ram_mb=int(memory.total / (1024 * 1024)),
            used_ram_mb=int((memory.total - memory.available) / (1024 * 1024)),
            available_ram_mb=int(memory.available / (1024 * 1024)),
            load_average=self._resources.load_average(),
            uptime_s=max(0.0, self._clock.now() - self._boot),
            total_gpus=self._resources.total_gpus,
            allocated_gpus=self._resources.allocated_gpus,
        )


class JobSampler:
    """Records resource usage and progress for running jobs.

    Args:
        repo: Where samples are stored.
        registry: Adapters, asked to parse progress out of log tails.
        bus: Where progress events are published.
        config: Sampling interval.
        clock: Time source.
        running_ids: Returns the ids the executor is currently supervising.
    """

    def __init__(
        self,
        *,
        repo: JobRepository,
        registry: AdapterRegistry,
        bus: EventBus,
        config: DaemonConfig,
        running_ids: Callable[[], list[str]],
        clock: Clock | None = None,
    ) -> None:
        self._repo = repo
        self._registry = registry
        self._bus = bus
        self._config = config
        self._clock = clock or SystemClock()
        self._running_ids = running_ids
        self._task: asyncio.Task[None] | None = None
        self._progress_task: asyncio.Task[None] | None = None
        self._cpu_totals: dict[str, tuple[float, float]] = {}

    def start(self) -> None:
        """Begin sampling running jobs."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="dispatch-job-sampler")
        if self._progress_task is None or self._progress_task.done():
            self._progress_task = asyncio.create_task(
                self._progress_loop(), name="dispatch-progress-sampler"
            )

    def stop(self) -> None:
        """Stop sampling."""
        for task in (self._task, self._progress_task):
            if task is not None:
                task.cancel()
        self._task = None
        self._progress_task = None

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._config.job_sample_interval_s)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self.sample_all)
        except asyncio.CancelledError:
            raise

    async def _progress_loop(self) -> None:
        """Publish each running job's current time step, often.

        Its own loop, at its own interval, because reading progress is a single small read
        at the end of a file while a resource sample walks a whole process tree. Sharing
        one timer would force a choice between a stale time step and needless work.
        """
        try:
            while True:
                await asyncio.sleep(self._config.progress_interval_s)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self.publish_progress_all)
        except asyncio.CancelledError:
            raise

    def publish_progress_all(self) -> None:
        """Read and publish progress for every running job, without measuring resources."""
        for job_id in self._running_ids():
            progress = self._progress(job_id)
            if progress is not None:
                self._bus.publish(Event.JOB_PROGRESS, {"id": job_id, "progress": progress})

    def sample_all(self) -> None:
        """Sample every active job once."""
        for job in self._repo.active():
            if job.pid is None:
                continue
            with contextlib.suppress(DispatchError):
                self.sample(job.id)

    def sample(self, job_id: str) -> Sample | None:
        """Measure one job's process tree and record the result."""
        job = self._repo.get_optional(job_id)
        if job is None or job.pid is None:
            return None

        usage = _tree_usage(job.pid)
        if usage is None:
            return None
        rss_mb, cpu_pct = usage

        sample = Sample(ts=self._clock.now(), rss_mb=rss_mb, cpu_pct=cpu_pct)
        self._repo.add_sample(job_id, sample)
        self._repo.update_metrics(job_id, peak_rss_mb=rss_mb, mean_cpu_pct=cpu_pct)

        progress = self._progress(job_id)
        self._bus.publish(
            Event.JOB_PROGRESS,
            {
                "id": job_id,
                "rss_mb": rss_mb,
                "cpu_pct": cpu_pct,
                "progress": progress,
            },
        )
        return sample

    def _progress(self, job_id: str) -> dict[str, object] | None:
        """Extract progress from the tail of a job's log, via its adapter."""
        job = self._repo.get_optional(job_id)
        if job is None or job.stdout_path is None:
            return None
        try:
            adapter = self._registry.get(job.solver)
        except DispatchError:
            return None

        tail = read_tail(job.stdout_path, LOG_TAIL_BYTES)
        if not tail:
            return None

        ctx = self._registry.context(job.workdir, cores=job.cores, adapter=job.solver)
        try:
            progress = adapter.parse_progress(tail, ctx)
        except Exception as exc:
            log.debug("Adapter %s could not parse progress: %s", job.solver, exc)
            return None
        if progress is None:
            return None
        return {
            "current": progress.current,
            "total": progress.total,
            "label": progress.label,
            "fraction": progress.fraction,
        }


def read_tail(path: Path, size: int) -> str:
    """Read the last ``size`` bytes of a file as text.

    One ``pread`` at the end of the file, so a multi-gigabyte log costs the same as a small
    one. Decoding is lenient because a truncated multi-byte character at the boundary is
    expected, not exceptional.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            start = max(0, handle.tell() - size)
            handle.seek(start)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _tree_usage(pid: int) -> tuple[int, float] | None:
    """Total RSS in MB and CPU percent for a process and its descendants.

    MPI jobs are trees: the launcher holds almost no memory and the ranks hold all of it,
    so measuring only the parent would report a 40 GB simulation as using nothing.
    """
    import psutil

    try:
        parent = psutil.Process(pid)
        members = [parent, *parent.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

    rss = 0
    cpu = 0.0
    for process in members:
        try:
            with process.oneshot():
                rss += process.memory_info().rss
                cpu += process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return int(rss / (1024 * 1024)), cpu


def _boot_time() -> float:
    """System boot time, for reporting uptime."""
    try:
        with Path("/proc/uptime").open(encoding="utf-8") as handle:
            return time.time() - float(handle.read().split()[0])
    except (OSError, ValueError, IndexError):  # pragma: no cover - environmental
        return time.time()
