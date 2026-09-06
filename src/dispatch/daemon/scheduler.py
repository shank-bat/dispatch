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

        self._admitted_sweep_slots: dict[str, str] = {}
        """Sweep members admitted but not yet visible as active, as ``job id -> sweep id``.

        ``executor.launch`` returns before the job's transition to PREPARING is written, so
        for a moment a member occupies a slot that no query can see. Bounded by the number
        of in-flight admissions and pruned against the ledger on every pass.
        """

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

        # A job that was asked to run after another one is not a candidate until that job
        # has completed. Removing it from the queue *before* the policy sees it, rather
        # than teaching every policy about dependencies, is what keeps the effect local:
        # the rest of the queue is offered the same capacity it would have been offered
        # anyway, so nothing else waits and no core sits idle on its account.
        eligible = [job for job in queued if self._dependency_satisfied(job)]

        # A sweep caps how many of *its own* members may run at once. Applied here, after
        # dependencies and before the policy, for the same reason dependencies are: the
        # policy is offered only jobs that may actually start, so the cap cannot be
        # undone by a policy that sees spare capacity, and jobs outside the sweep are
        # offered exactly the capacity they would have been offered anyway.
        eligible = self._within_sweep_limits(eligible)
        if not eligible:
            return []

        ok, reason = self._resources.check_disk()
        if not ok:
            log.warning("Not admitting jobs: %s", reason)
            return []

        capacity = self._resources.capacity()
        selected = self._policy.select(eligible, capacity)

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

    def _within_sweep_limits(self, eligible: list[Job]) -> list[Job]:
        """Drop sweep members that would exceed their sweep's concurrency limit.

        The limit is a hard cap on running members, never a target and never a hint: free
        cores do not override it. A sweep configured for two concurrent jobs on a machine
        with ten spare cores runs two, and the rest of the machine stays available to
        unrelated work -- which is the entire reason the setting exists.

        Three sets of members occupy a slot, and all three have to be counted:

        * those already PREPARING or RUNNING in the database;
        * those admitted on an earlier pass whose transition has not landed yet --
          ``launch`` returns before the executor writes it, so there is a real window in
          which a running job is invisible to a query;
        * those admitted **earlier in this same pass**.

        They are unioned as ids rather than added as counts, because the first two sets
        overlap as a job's transition lands and adding them would double-count it.

        Jobs with no sweep are returned untouched, which is every ordinary job.
        """
        occupied = self._occupied_slots()
        allowed: list[Job] = []

        for job in eligible:
            if job.sweep_id is None:
                allowed.append(job)
                continue

            sweep = self._repo.get_sweep(job.sweep_id)
            if sweep is None:
                # The sweep row is gone but the member survived it (ON DELETE SET NULL has
                # not been applied, or this is a stale read). Scheduling it as an ordinary
                # job is the safe reading: no sweep, no sweep limit.
                allowed.append(job)
                continue

            taken = occupied.setdefault(job.sweep_id, set())
            if len(taken) >= sweep.concurrency:
                continue

            allowed.append(job)
            taken.add(job.id)

        return allowed

    def _occupied_slots(self) -> dict[str, set[str]]:
        """Which members of each sweep are currently holding one of its slots.

        The database knows about everything that has reached PREPARING. The ledger knows
        about everything this scheduler has admitted, synchronously, at the moment it was
        admitted -- which covers the gap before the executor's transition is written.
        """
        occupied = self._repo.active_members_by_sweep()

        for job_id, sweep_id in list(self._admitted_sweep_slots.items()):
            if not self._resources.holds(job_id):
                # The ledger released it: the job has ended, and whether it ever reached
                # the database as active no longer matters.
                del self._admitted_sweep_slots[job_id]
                continue
            occupied.setdefault(sweep_id, set()).add(job_id)

        return occupied

    def _sweep_blocked(self, job: Job) -> str | None:
        """Why a sweep member is waiting on its own sweep, if it is.

        Read-only, and used for the queue view's explanation. It intentionally re-derives
        the answer rather than caching what the admission pass decided: the pass runs on
        nudges, the user reads the queue whenever they like, and a stale reason is worse
        than none.
        """
        if job.sweep_id is None:
            return None
        sweep = self._repo.get_sweep(job.sweep_id)
        if sweep is None:
            return None
        if sweep.running >= sweep.concurrency:
            return f"waiting for its sweep ({sweep.running}/{sweep.concurrency} running)"
        return None

    def _dependency_satisfied(self, job: Job) -> bool:
        """Whether a job's ``run after`` dependency, if it has one, has been met."""
        return self._blocking_dependency(job) is None

    def _blocking_dependency(self, job: Job) -> str | None:
        """Why a job's dependency is not yet met, or ``None`` if nothing is blocking it.

        A dependency counts as met only once the other job has **completed**. A parent that
        failed, was cancelled, or ended unknowably leaves its dependent queued rather than
        starting it on the strength of an outcome the user did not ask to wait for; the
        reason says so, and the job can be cancelled or released by hand from there.
        """
        if job.depends_on_job_id is None:
            return None
        parent = self._repo.get_optional(job.depends_on_job_id)
        if parent is None:
            # Only reachable if the row went away without the foreign key clearing this
            # column. Blocked rather than started: the condition was never satisfied.
            return "waiting for a job that no longer exists"
        if parent.state is JobState.COMPLETED:
            return None
        return f"waiting for {parent.name} ({parent.state.value.lower()})"

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
        if job.sweep_id is not None:
            self._admitted_sweep_slots[job.id] = job.sweep_id
        try:
            self._executor.launch(job)
        except Exception:
            self._resources.release(job.id)
            self._admitted_sweep_slots.pop(job.id, None)
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
        blocking = self._blocking_dependency(job)
        if blocking is not None:
            return blocking
        held_by_sweep = self._sweep_blocked(job)
        if held_by_sweep is not None:
            return held_by_sweep
        can, reason = self._resources.can_admit(job.resources)
        if can:
            return "waiting for its turn"
        return reason

    def _publish_queue(self) -> None:
        with contextlib.suppress(DispatchError):
            self._bus.publish(Event.QUEUE_CHANGED, {"queued": len(self._repo.queued())})
