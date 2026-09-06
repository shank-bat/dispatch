"""Reconciling the database with reality at startup.

Runs before the socket is bound, so no client ever sees an inconsistent world.

For every job the database believes is active, there are exactly three possibilities, and
each has a defensible answer:

1. **An exit sentinel exists.** The job finished while the daemon was away. Its real exit
   code is on disk and its finish time is the file's mtime.
2. **The process is still alive and is still ours.** Re-adopt it. A daemon restart during
   a three-day simulation becomes a non-event.
3. **Neither, and the machine has rebooted since the job started.** The process cannot
   exist -- everything it was died with the power -- so the job returns to the queue to be
   restarted from whatever state the simulation itself last wrote (§6.7).
4. **Neither, and the machine has not rebooted.** Something killed the process without
   leaving a sentinel. The job becomes ``UNKNOWN`` with reason ``lost``, which is the
   honest answer and better than a confident, wrong ``FAILED``.

Identity in case 2 is the ``(pid, start time)`` pair, never the pid alone. After a reboot
that number may belong to something else entirely, and re-adopting a stranger would mean
signalling it on the next cancel.

Cases 3 and 4 are separated by the boot identity recorded with the job when it started,
compared against the machine's boot identity now. Boot against boot, from one source: the
tempting comparison -- did the machine boot after the job started? -- quietly assumes the
job's timestamps and the system clock share an epoch. A daemon restart is not a
reboot: its jobs' processes survive and are re-adopted by case 2, and nothing here changes
that. Restarting a simulation that is still running would be far worse than leaving it
alone, so the reboot branch is reached only when the process is genuinely gone *and* the
machine it was running on is no longer up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from dispatch.core.config import Config
from dispatch.core.errors import DispatchError
from dispatch.core.states import ExitReason, JobState
from dispatch.daemon.executor import JobExecutor
from dispatch.daemon.process import (
    ProcessManager,
    describe_exit,
    is_alive,
    matches_start_time,
    system_boot_time,
)
from dispatch.daemon.resources import ResourceModel
from dispatch.db.repository import JobRepository

__all__ = ["RecoveryReport", "recover"]

log = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    """What startup reconciliation found."""

    readopted: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    requeued: list[str] = field(default_factory=list)
    """Jobs the machine rebooted under, returned to the queue to be resumed."""

    @property
    def total(self) -> int:
        """Jobs examined."""
        return len(self.readopted) + len(self.finished) + len(self.lost) + len(self.requeued)

    def describe(self) -> str:
        """A one-line summary for the daemon log."""
        if not self.total:
            return "no jobs were active at shutdown"
        parts = []
        if self.readopted:
            parts.append(f"{len(self.readopted)} still running")
        if self.finished:
            parts.append(f"{len(self.finished)} finished while away")
        if self.requeued:
            parts.append(f"{len(self.requeued)} requeued after a reboot")
        if self.lost:
            parts.append(f"{len(self.lost)} lost")
        return ", ".join(parts)


def recover(
    *,
    repo: JobRepository,
    resources: ResourceModel,
    executor: JobExecutor,
    config: Config,
    boot_time: float | None = None,
) -> RecoveryReport:
    """Reconcile active jobs with what is actually on the machine.

    Also rebuilds the resource ledger from the survivors, so the first admission pass sees
    a true picture of what is already committed.

    Args:
        boot_time: This machine's current boot identity. Injected so a reboot can be
            simulated in tests; read from the kernel otherwise.
    """
    report = RecoveryReport()
    resources.clear()
    booted_at = system_boot_time() if boot_time is None else boot_time

    for job in repo.active():
        log_dir = config.paths.job_dir(job.id)
        exit_file = log_dir / "exit_code"

        code = ProcessManager.read_exit_file(exit_file)
        if code is not None:
            _finish_from_sentinel(repo, job.id, exit_file, code)
            report.finished.append(job.id)
            continue

        still_ours = (
            job.pid is not None
            and is_alive(job.pid)
            and matches_start_time(job.pid, job.pid_start_time)
        )
        if still_ours and executor.adopt(job):
            report.readopted.append(job.id)
            continue

        # The process is gone. Whether that is recoverable depends on *why*: a machine that
        # has rebooted since this job started took the process with it and nothing is wrong
        # with the case, whereas a process that vanished on a machine that has been up the
        # whole time was killed by something this daemon cannot account for.
        if _rebooted_since(job.boot_time, booted_at):
            _requeue_after_reboot(repo, job.id)
            report.requeued.append(job.id)
            continue

        _mark_lost(repo, job.id, job.state)
        report.lost.append(job.id)

    log.info("Startup recovery: %s", report.describe())
    return report


def _finish_from_sentinel(repo: JobRepository, job_id: str, exit_file: Path, code: int) -> None:
    """Record the outcome of a job that ended while the daemon was down.

    The finish time comes from the sentinel's mtime rather than from now, so a job that
    ended eight hours before the daemon restarted does not report an eight-hour runtime.
    """
    exit_code, signal_name = describe_exit(code)
    if exit_code == 0:
        state, reason = JobState.COMPLETED, ExitReason.OK
    elif signal_name:
        state, reason = JobState.FAILED, ExitReason.SIGNAL
    else:
        state, reason = JobState.FAILED, ExitReason.NONZERO

    try:
        finished_at = exit_file.stat().st_mtime
    except OSError:
        finished_at = None

    try:
        repo.add_event(job_id, "state", "recovered after a daemon restart")
        repo.mark_finished(
            job_id,
            state=state,
            exit_code=exit_code,
            reason=reason,
            signal_name=signal_name,
            finished_at=finished_at,
        )
    except DispatchError as exc:
        log.warning("Could not record recovered outcome for job %s: %s", job_id, exc)
    else:
        log.info("Job %s finished while the daemon was down: exit %s", job_id[:8], exit_code)


def _mark_lost(repo: JobRepository, job_id: str, state: JobState) -> None:
    """Record a job whose fate is genuinely unknown."""
    try:
        repo.add_event(
            job_id,
            "warn",
            f"was {state.value} at startup but its process is gone and left no exit code",
        )
        repo.mark_finished(job_id, state=JobState.UNKNOWN, reason=ExitReason.LOST)
    except DispatchError as exc:
        log.warning("Could not mark job %s as lost: %s", job_id, exc)
    else:
        log.warning("Job %s was lost (the machine most likely restarted mid-run)", job_id[:8])


def _rebooted_since(job_boot: float | None, booted_at: float | None) -> bool:
    """Whether the machine has rebooted since this job started.

    Separates "the power went" from "something killed it". Either unknown answers ``False``,
    which routes the job to the pre-existing ``lost`` outcome: a job with no recorded boot
    never reached RUNNING under a daemon that records one, and a machine that cannot report
    its boot is not evidence of a reboot. Guessing the other way restarts work on no
    evidence, which is the one outcome here worse than admitting ignorance.

    The second of tolerance absorbs the difference between two readings of a value that is
    nominally an integer but is not guaranteed to be one on every kernel.
    """
    if job_boot is None or booted_at is None:
        return False
    return abs(booted_at - job_boot) > 1.0


def _requeue_after_reboot(repo: JobRepository, job_id: str) -> None:
    """Return a job the machine rebooted under to the queue, flagged to resume.

    It keeps its ``seq``, so it keeps its place: the jobs that were queued behind it stay
    behind it, and a reboot does not silently reorder the queue. What state it resumes from
    is not decided here and is not stored -- the adapter reads that from the case when the
    job is next scheduled.
    """
    try:
        repo.add_event(
            job_id,
            "state",
            "the machine rebooted while this job was running; requeued to resume from the "
            "simulation's last saved state",
        )
        repo.requeue_for_resume(job_id, detail="requeued after a reboot")
    except DispatchError as exc:
        log.warning("Could not requeue job %s after a reboot: %s", job_id, exc)
    else:
        log.info("Job %s was interrupted by a reboot; requeued to resume", job_id[:8])
