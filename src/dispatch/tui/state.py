"""Client-side state, fed by daemon events.

The TUI holds no authoritative state. This is a cache of what the daemon last said,
updated by pushed events and rebuilt wholesale on reconnect. Nothing here is ever the
source of truth for a decision -- if the display and the daemon disagree, the daemon is
right and a resync fixes the display.

That is what makes the TUI genuinely disposable: killing it loses nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AppState"]

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
    error: str | None = None

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
