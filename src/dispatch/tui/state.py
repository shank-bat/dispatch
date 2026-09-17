"""Client-side state, fed by daemon events.

The TUI holds no authoritative state. This is a cache of what the daemon last said,
updated by pushed events and rebuilt wholesale on reconnect. Nothing here is ever the
source of truth for a decision -- if the display and the daemon disagree, the daemon is
right and a resync fixes the display.

That is what makes the TUI genuinely disposable: killing it loses nothing.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

__all__ = ["AppState", "core_hours_used", "sweep_summary"]

HISTORY_LENGTH = 30
"""Readings kept per metric, for the header's sparklines.

The daemon pushes a system snapshot roughly every two seconds (see
``SystemMonitor`` on the daemon side), so thirty of them is about a minute -- long
enough for a trend to be visible, short enough that the line is still readable at the
dozen-odd cells a header sparkline actually has.
"""

HISTORY_METRICS = ("cores", "cpu", "mem", "gpu")

ACTIVE_STATES = ("QUEUED", "HELD", "PREPARING", "RUNNING")
TERMINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED", "REJECTED", "UNKNOWN")


@dataclass
class AppState:
    """Everything the interface renders."""

    connected: bool = False
    hostname: str = ""
    daemon_version: str = ""
    snapshot: dict[str, Any] = field(default_factory=dict)
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    progress: dict[str, dict[str, Any]] = field(default_factory=dict)
    sweeps: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Sweeps by id, as the daemon last reported them.

    Keyed by id and refreshed wholesale rather than event-driven: a sweep's configuration
    never changes after submission, and its live counts are re-derived from the jobs
    already cached here.
    """
    error: str | None = None

    history: dict[str, deque[float]] = field(
        default_factory=lambda: {name: deque(maxlen=HISTORY_LENGTH) for name in HISTORY_METRICS}
    )
    """Recent utilisation, as a percentage, per metric in :data:`HISTORY_METRICS`.

    Populated by :meth:`push_snapshot` alone -- never by a plain assignment to
    :attr:`snapshot`, which is why that method exists instead of screens setting the
    attribute directly. ``"gpu"`` stays empty on a machine with none, which a sparkline
    reads the same way :class:`~dispatch.tui.widgets.meters.HeaderStats` already reads an
    empty gauge: nothing to show, rather than a flat lie at zero.
    """

    def push_snapshot(self, data: dict[str, Any]) -> None:
        """Record a system snapshot, both as the current reading and as one more point of
        history for the header's sparklines.

        The single entry point for setting :attr:`snapshot`: every other write to it would
        let readers accumulate silently on connect and resync alike, which is what makes it
        safe to look at these from any picture of the machine the daemon ever hands over,
        not only the ones a live sampler happens to push.
        """
        self.snapshot = data
        total_cores = float(data.get("total_cores") or 0)
        if total_cores:
            self.history["cores"].append(float(data.get("allocated_cores", 0)) / total_cores * 100)
        self.history["cpu"].append(float(data.get("cpu_percent", 0.0)))
        total_ram = float(data.get("total_ram_mb") or 0)
        if total_ram:
            self.history["mem"].append(float(data.get("used_ram_mb", 0)) / total_ram * 100)
        total_gpus = float(data.get("total_gpus") or 0)
        if total_gpus:
            self.history["gpu"].append(float(data.get("allocated_gpus", 0)) / total_gpus * 100)

    def replace_jobs(self, jobs: Sequence[dict[str, Any]]) -> None:
        """Replace the whole job cache. Used on connect and on resync."""
        self.jobs = {job["id"]: job for job in jobs}

    def update_job(self, job: dict[str, Any]) -> None:
        """Apply a single job update from an event."""
        self.jobs[job["id"]] = job

    def update_progress(self, data: dict[str, Any]) -> None:
        """Record a progress sample for a running job.

        Merged into what is already known rather than replacing it. Two samplers publish
        here at different rates -- the frequent one carries only the solver's time step,
        the slower one carries memory and CPU -- and a plain assignment would let each
        erase the other's fields between updates, making the display flicker between
        halves of the truth.
        """
        job_id = data.get("id")
        if not job_id:
            return
        current = self.progress.setdefault(job_id, {})
        current.update(data)

    def forget(self, job_id: str) -> None:
        """Drop a deleted job."""
        self.jobs.pop(job_id, None)
        self.progress.pop(job_id, None)

    # -- views -------------------------------------------------------------------------

    def by_state(self, *states: str) -> list[dict[str, Any]]:
        """Jobs in any of the given states, in queue then submission order."""
        wanted = set(states)
        found = [job for job in self.jobs.values() if job["state"] in wanted]
        return sorted(found, key=_queue_order)

    @property
    def running(self) -> list[dict[str, Any]]:
        """Jobs currently using the machine."""
        return self.by_state("PREPARING", "RUNNING")

    @property
    def queued(self) -> list[dict[str, Any]]:
        """Jobs waiting, in the order they will be considered."""
        return self.by_state("QUEUED", "HELD")

    @property
    def finished(self) -> list[dict[str, Any]]:
        """Finished jobs, most recent first."""
        found = [job for job in self.jobs.values() if job["state"] in TERMINAL_STATES]
        return sorted(found, key=lambda job: job.get("finished_at") or 0, reverse=True)

    @property
    def next_job(self) -> dict[str, Any] | None:
        """The job at the head of the queue."""
        queued = [job for job in self.queued if job["state"] == "QUEUED"]
        return queued[0] if queued else None

    def get(self, job_id: str) -> dict[str, Any] | None:
        """One job by id."""
        return self.jobs.get(job_id)

    def replace_sweeps(self, sweeps: Sequence[dict[str, Any]]) -> None:
        """Replace the sweep cache. Used when the queue view opens or the queue changes."""
        self.sweeps = {sweep["id"]: sweep for sweep in sweeps}

    def sweep_for(self, job: dict[str, Any]) -> dict[str, Any] | None:
        """The sweep a job belongs to, if any."""
        sweep_id = job.get("sweep_id")
        return self.sweeps.get(sweep_id) if sweep_id else None

    def sweep_running(self, sweep_id: str) -> int:
        """How many of a sweep's members are running, counted from the cached jobs.

        Derived rather than read from the sweep record so the number tracks the job events
        the interface is already receiving, instead of going stale between refreshes.
        """
        return sum(
            1
            for job in self.jobs.values()
            if job.get("sweep_id") == sweep_id and job["state"] in ("PREPARING", "RUNNING")
        )

    def progress_for(self, job_id: str) -> dict[str, Any] | None:
        """The latest progress sample for a job, if any."""
        return self.progress.get(job_id)

    @property
    def counts(self) -> dict[str, int]:
        """How many jobs are in each state."""
        counts: dict[str, int] = {}
        for job in self.jobs.values():
            counts[job["state"]] = counts.get(job["state"], 0) + 1
        return counts


def _queue_order(job: dict[str, Any]) -> tuple[int, int, int]:
    """Sort key: queue position first, then priority, then submission order."""
    position = job.get("queue_position")
    return (
        position if position is not None else 10**6,
        -int(job.get("priority", 0)),
        int(job.get("seq", 0)),
    )


def sweep_summary(state: AppState, sweep: dict[str, Any]) -> str:
    """One line describing a sweep's progress and the limit it is running under.

    Both numbers, always. The progress alone leaves the reader wondering why only two of
    eight are moving; the cap alone does not say how far along it is. Shared by the queue
    and the dashboard so the two cannot word it differently.
    """
    running = state.sweep_running(sweep["id"])
    done = int(sweep.get("finished", 0))
    total = int(sweep.get("total", 0))
    return (
        f"sweep {sweep['name']}  {running} running, {done}/{total} done"
        f"  (max {sweep['concurrency']})"
    )


def core_hours_used(state: AppState) -> float:
    """Core-hours consumed by every job the interface currently has cached.

    ``cores * elapsed hours``, summed. Finished jobs use their recorded ``runtime_s``, the
    same figure the job table shows; a still-running job uses ``now - started_at``, so the
    total keeps climbing while it works rather than waiting for it to finish to count.

    "Cached" rather than "ever run": the interface only ever holds what one ``job.list``
    page and the events since have told it about (state.py's own docstring), so on a
    machine with a long history this is a total over recent jobs, not all of history. That
    is the same boundary the rest of the interface already lives with -- the job table
    shows the same jobs -- so this does not claim to know more than it does.
    """
    now = datetime.now().timestamp()
    total_seconds = 0.0
    for job in state.jobs.values():
        cores = job.get("cores")
        if not cores:
            continue
        runtime = (job.get("metrics") or {}).get("runtime_s")
        if not runtime:
            started = job.get("started_at")
            if started and job.get("state") in ("RUNNING", "PREPARING"):
                runtime = max(0.0, now - started)
        if runtime:
            total_seconds += cores * runtime
    return total_seconds / 3600
