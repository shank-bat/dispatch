"""Admission control.

The scheduler is a single coroutine parked on an :class:`asyncio.Event`. It has no tick.
It wakes when something happens that could change the answer -- a submission, a job
ending, a hold or release, a priority change, a configuration reload -- decides what to
start, and goes back to sleep.

A slow heartbeat runs alongside it, purely to catch drift: if a job somehow ends without
nudging, the queue recovers within half a minute instead of stalling until the next
submission. On an idle machine that is two wakeups a minute, which is why idle CPU
measures 0.0%.

The scheduler knows nothing about solvers, and a test asserts that: it manipulates jobs,
a ledger, and a policy, and calls ``executor.launch``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from dispatch.core.clock import Clock, SystemClock
from dispatch.core.config import SchedulerConfig
from dispatch.core.errors import DispatchError
from dispatch.core.models import Job
from dispatch.core.states import JobState
from dispatch.daemon.events import EventBus
from dispatch.daemon.executor import JobExecutor
from dispatch.daemon.policies import SchedulingPolicy
from dispatch.daemon.resources import ResourceModel
from dispatch.db.repository import JobRepository
from dispatch.ipc.protocol import Event

__all__ = ["Scheduler"]

log = logging.getLogger(__name__)


class Scheduler:
    """Decides which queued jobs start, and when.

    Args:
        repo: Persistence.
        resources: The ledger admission is decided against.
        executor: What actually runs a job.
        policy: The configured admission policy.
        config: Scheduler settings.
        bus: Event bus, for queue-changed notifications.
        clock: Time source.
    """

    def __init__(
        self,
        *,
        repo: JobRepository,
        resources: ResourceModel,
        executor: JobExecutor,
        policy: SchedulingPolicy,
        config: SchedulerConfig,
        bus: EventBus,
        clock: Clock | None = None,
    ) -> None:
        self._repo = repo
        self._resources = resources
        self._executor = executor
        self._policy = policy
        self._config = config
        self._bus = bus
        self._clock = clock or SystemClock()

        self._wake = asyncio.Event()
        self._stopping = False
        self._paused = False
        self._task: asyncio.Task[None] | None = None
        self.passes = 0
        """Completed admission passes. Diagnostic, and asserted on in tests."""

    # -- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        """Begin the scheduling loop."""
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="dispatch-scheduler")
            self.nudge()

    async def stop(self) -> None:
        """Stop scheduling. Does not touch running jobs."""
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def nudge(self) -> None:
        """Ask for an admission pass.

        Cheap and idempotent: several nudges before the loop wakes collapse into one pass,
        which is exactly right when four jobs finish at once.
        """
        self._wake.set()

    @property
    def paused(self) -> bool:
        """Whether admission is suspended."""
        return self._paused

    def pause(self) -> None:
        """Stop admitting new jobs. Running jobs are unaffected."""
        self._paused = True

    def resume(self) -> None:
        """Resume admitting, and reconsider the queue immediately."""
        self._paused = False
        self.nudge()

    async def run(self) -> None:
        """The scheduling loop. Sleeps until nudged or the heartbeat fires."""
        while not self._stopping:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._config.heartbeat_s)
            except TimeoutError:
                pass  # heartbeat: re-check for drift, then go back to sleep
            except asyncio.CancelledError:
                raise

            self._wake.clear()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A scheduling bug must not silently stop the queue for the next month.
                log.exception("Admission pass failed; will retry on the next nudge")

    # -- the pass -------------------------------------------------------------------------

    async def run_once(self) -> list[Job]:
        """Run one admission pass. Returns the jobs started.

        Exposed separately from :meth:`run` so tests can drive scheduling deterministically
        without a running loop.
        """
        self.passes += 1
        if self._paused:
            return []

        queued = [job for job in self._repo.queued() if not self._executor.is_running(job.id)]
        if not queued:
            return []

        ok, reason = self._resources.check_disk()
        if not ok:
            log.warning("Not admitting jobs: %s", reason)
            return []

        capacity = self._resources.capacity()
        selected = self._policy.select(queued, capacity)

        started: list[Job] = []
        for job in selected:
            if not self._admit(job):
                continue
            started.append(job)

        if started:
            self._bus.publish(
                Event.QUEUE_CHANGED,
                {"started": [job.id for job in started], "queued": len(queued) - len(started)},
            )
        return started

    def _admit(self, job: Job) -> bool:
        """Allocate for a job and hand it to the executor.

        The allocation is taken *before* the launch and released by the executor when the
        job ends. Doing it here means a second pass cannot see the cores as free while the
        first job is still starting up.
        """
        can, reason = self._resources.can_admit(job.resources)
        if not can:
            log.debug("Job %s not admitted: %s", job.id[:8], reason)
            return False

        self._resources.acquire(job.id, job.resources)
        try:
            self._executor.launch(job)
        except Exception:
            self._resources.release(job.id)
            log.exception("Could not launch job %s", job.id)
            return False
        log.info(
            "Admitted job %s (%s, %d cores); %d cores now free",
            job.id[:8],
            job.name,
            job.cores,
            self._resources.free_cores,
        )
        return True

    # -- queue operations -------------------------------------------------------------------

    def hold(self, job_id: str) -> Job:
        """Withhold a queued job from scheduling."""
        job = self._repo.transition(job_id, JobState.HELD, detail="held by the user")
        self._publish_queue()
        return job

    def release(self, job_id: str) -> Job:
        """Return a held job to the queue and reconsider immediately."""
        job = self._repo.transition(job_id, JobState.QUEUED, detail="released by the user")
        self._publish_queue()
        self.nudge()
        return job

    def set_priority(self, job_id: str, priority: int) -> Job:
        """Change a job's priority and reconsider the queue."""
        job = self._repo.set_priority(job_id, priority)
        self._publish_queue()
        self.nudge()
        return job

    def next_job(self) -> Job | None:
        """The job at the head of the queue, for the dashboard."""
        queued = self._repo.queued()
        return queued[0] if queued else None

    def explain(self, job: Job) -> str | None:
        """Why a queued job is not running, in the user's terms.

        "Waiting" without a cause is the least useful thing a scheduler can say, so the
        queue view shows this instead.
        """
        if job.state is JobState.HELD:
            return "held"
        if job.state is not JobState.QUEUED:
            return None
        can, reason = self._resources.can_admit(job.resources)
        if can:
            return "waiting for its turn"
        return reason

    def _publish_queue(self) -> None:
        with contextlib.suppress(DispatchError):
            self._bus.publish(Event.QUEUE_CHANGED, {"queued": len(self._repo.queued())})
