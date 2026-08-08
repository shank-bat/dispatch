"""Admission control: the ledger, the policies, and the scheduler loop.

Nothing here starts a process. The executor is faked, so a hundred scheduling decisions
run in a millisecond and every one of them is deterministic -- which is the payoff of
making policies pure functions of (queue, capacity).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dispatch.core.config import SchedulerConfig
from dispatch.core.errors import ConfigError
from dispatch.core.models import Job, JobSpec, ResourceRequest
from dispatch.core.states import ExitReason, JobState
from dispatch.daemon.events import EventBus
from dispatch.daemon.policies import (
    POLICIES,
    FifoPolicy,
    PriorityFifoBackfill,
    ShortestFirstPolicy,
    build_policy,
)
from dispatch.daemon.resources import Capacity, ResourceModel
from dispatch.daemon.scheduler import Scheduler
from dispatch.db.repository import JobRepository


class FakeExecutor:
    """Records launches instead of starting anything."""

    def __init__(self) -> None:
        self.launched: list[Job] = []
        self.fail_next = False

    def launch(self, job: Job) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("launch failed")
        self.launched.append(job)

    def is_running(self, job_id: str) -> bool:
        return any(job.id == job_id for job in self.launched)


@pytest.fixture
def executor() -> FakeExecutor:
    return FakeExecutor()


@pytest.fixture
def scheduler(repo: JobRepository, resources: ResourceModel, executor: FakeExecutor) -> Scheduler:
    return Scheduler(
        repo=repo,
        resources=resources,
        executor=executor,  # type: ignore[arg-type]
        policy=PriorityFifoBackfill(),
        config=SchedulerConfig(total_cores=8, reserved_cores=0),
        bus=EventBus(),
    )


def queue(repo: JobRepository, tmp_path: Path, *specs: tuple[str, int, int]) -> list[Job]:
    """Create jobs from ``(name, cores, priority)`` triples."""
    created = []
    for name, cores, priority in specs:
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        created.append(
            repo.create(
                JobSpec(
                    workdir=directory,
                    solver="fake",
                    resources=ResourceRequest(cores=cores),
                    name=name,
                    priority=priority,
                )
            )
        )
    return created


def queue_after(
    repo: JobRepository, tmp_path: Path, name: str, cores: int, parent_id: str | None
) -> Job:
    """Create one job that must wait for ``parent_id`` before it may start."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    return repo.create(
        JobSpec(
            workdir=directory,
            solver="fake",
            resources=ResourceRequest(cores=cores),
            name=name,
            depends_on_job_id=parent_id,
        )
    )


def start_running(repo: JobRepository, resources: ResourceModel, job: Job) -> Job:
    """Put a job into RUNNING with its cores allocated, as the executor would."""
    resources.acquire(job.id, job.resources)
    repo.mark_preparing(job.id)
    return repo.mark_started(job.id, pid=4242, pid_start_time=1.0)


def complete(repo: JobRepository, resources: ResourceModel, job: Job) -> Job:
    """Finish a running job successfully and give its cores back."""
    resources.release(job.id)
    return repo.mark_finished(job.id, state=JobState.COMPLETED, exit_code=0, reason=ExitReason.OK)


# -- the ledger ---------------------------------------------------------------------------


def test_free_cores_accounts_for_reservations() -> None:
    model = ResourceModel(SchedulerConfig(total_cores=24, reserved_cores=2))
    assert model.free_cores == 22
    assert model.schedulable_cores == 22


def test_reserved_cores_cannot_consume_the_whole_machine() -> None:
    """A misconfiguration must not make the machine unable to run anything."""
    model = ResourceModel(SchedulerConfig(total_cores=4, reserved_cores=99))
    assert model.schedulable_cores >= 1
    assert model.free_cores >= 1


def test_acquire_and_release(resources: ResourceModel) -> None:
    resources.acquire("a", ResourceRequest(cores=4))
    assert resources.allocated_cores == 4
    assert resources.free_cores == 4
    resources.release("a")
    assert resources.allocated_cores == 0


def test_acquire_is_idempotent_by_job(resources: ResourceModel) -> None:
    """Replaying recovery must not double-count the machine."""
    resources.acquire("a", ResourceRequest(cores=4))
    resources.acquire("a", ResourceRequest(cores=4))
    assert resources.allocated_cores == 4


def test_releasing_an_unknown_job_is_harmless(resources: ResourceModel) -> None:
    resources.release("never-existed")
    assert resources.allocated_cores == 0


def test_admission_is_refused_with_a_reason(resources: ResourceModel) -> None:
    resources.acquire("big", ResourceRequest(cores=8))
    can, reason = resources.can_admit(ResourceRequest(cores=4))
    assert not can
    assert reason is not None and "0 of 8 cores free" in reason


def test_a_job_larger_than_the_machine_is_refused_permanently(
    resources: ResourceModel,
) -> None:
    """This should be an error at submission, not a job that waits forever."""
    can, reason = resources.can_admit(ResourceRequest(cores=99))
    assert not can
    assert reason is not None and "never" not in reason  # the message names the numbers
    assert "99 cores" in reason


def test_ram_gates_admission_when_an_estimate_is_given() -> None:
    model = ResourceModel(
        SchedulerConfig(total_cores=8, reserved_cores=0, ram_margin_mb=0),
        memory_probe=lambda: 1000,
    )
    assert model.can_admit(ResourceRequest(cores=1, ram_mb=500))[0]
    assert not model.can_admit(ResourceRequest(cores=1, ram_mb=2000))[0]


def test_jobs_without_a_ram_estimate_are_admitted_on_cores_alone() -> None:
    model = ResourceModel(SchedulerConfig(total_cores=8, reserved_cores=0), memory_probe=lambda: 1)
    assert model.can_admit(ResourceRequest(cores=1))[0]


def test_a_failed_memory_probe_degrades_to_core_scheduling() -> None:
    """An unreadable meminfo should not stop the queue."""

    def broken() -> int:
        raise OSError("no meminfo")

    model = ResourceModel(SchedulerConfig(total_cores=8, reserved_cores=0), memory_probe=broken)
    assert model.available_ram_mb() is None
    assert model.can_admit(ResourceRequest(cores=1, ram_mb=999_999))[0]


def test_admission_stops_when_the_disk_is_nearly_full(tmp_path: Path) -> None:
    model = ResourceModel(
        SchedulerConfig(total_cores=8, reserved_cores=0, min_free_disk_mb=10**9),
        memory_probe=lambda: 64_000,
        log_dir=tmp_path,
    )
    can, reason = model.can_admit(ResourceRequest(cores=1))
    assert not can
    assert reason is not None and "free" in reason


# -- policies -------------------------------------------------------------------------------


def test_fifo_stops_at_the_first_job_that_does_not_fit(repo, tmp_path) -> None:
    jobs = queue(repo, tmp_path, ("small", 2, 0), ("huge", 16, 0), ("tiny", 1, 0))
    chosen = FifoPolicy().select(jobs, Capacity(cores=8))
    assert [j.name for j in chosen] == ["small"]


def test_backfill_lets_a_small_job_past_a_blocked_large_one(repo, tmp_path) -> None:
    """The case backfill exists for: idle cores while the head job cannot fit."""
    jobs = queue(repo, tmp_path, ("huge", 16, 0), ("small", 2, 0))
    chosen = PriorityFifoBackfill().select(jobs, Capacity(cores=8))
    assert [j.name for j in chosen] == ["small"]


def test_a_blocked_job_claims_the_machine_after_enough_skips(repo, tmp_path) -> None:
    """Bounded starvation: the large job yields three times, then takes precedence.

    Without this, a steady trickle of small jobs means the production run never starts.
    """
    jobs = queue(repo, tmp_path, ("huge", 16, 0), ("small", 2, 0))
    policy = PriorityFifoBackfill(max_skips=3)

    assert [j.name for j in policy.select(jobs, Capacity(cores=8))] == ["small"]
    assert [j.name for j in policy.select(jobs, Capacity(cores=8))] == ["small"]
    assert policy.select(jobs, Capacity(cores=8)) == []
    assert policy.skips(jobs[0].id) == 3


def test_the_skip_counter_resets_once_the_job_starts(repo, tmp_path) -> None:
    jobs = queue(repo, tmp_path, ("huge", 16, 0), ("small", 2, 0))
    policy = PriorityFifoBackfill()
    policy.select(jobs, Capacity(cores=8))
    assert policy.skips(jobs[0].id) == 1
    policy.select(jobs, Capacity(cores=32))  # now it fits
    assert policy.skips(jobs[0].id) == 0


def test_the_skip_counter_does_not_grow_without_bound(repo, tmp_path) -> None:
    """State keyed by job id must be pruned, or a long-lived daemon leaks it."""
    jobs = queue(repo, tmp_path, ("huge", 16, 0))
    policy = PriorityFifoBackfill()
    policy.select(jobs, Capacity(cores=1))
    assert policy.skips(jobs[0].id) == 1
    policy.select([], Capacity(cores=1))
    assert policy._skips == {}


def test_priority_is_respected_before_size(repo, tmp_path) -> None:
    jobs = sorted(
        queue(repo, tmp_path, ("low", 4, 0), ("high", 4, 10)),
        key=lambda j: (-j.priority, j.seq),
    )
    chosen = PriorityFifoBackfill().select(jobs, Capacity(cores=4))
    assert [j.name for j in chosen] == ["high"]


def test_shortest_first_packs_the_machine(repo, tmp_path) -> None:
    jobs = queue(repo, tmp_path, ("big", 8, 0), ("a", 2, 0), ("b", 2, 0))
    chosen = ShortestFirstPolicy().select(jobs, Capacity(cores=8))
    assert sorted(j.name for j in chosen) == ["a", "b"]


def test_shortest_first_still_respects_priority_bands(repo, tmp_path) -> None:
    """Otherwise raising a big job's priority would never help it."""
    jobs = queue(repo, tmp_path, ("small", 1, 0), ("urgent", 8, 5))
    jobs.sort(key=lambda j: (-j.priority, j.seq))
    chosen = ShortestFirstPolicy().select(jobs, Capacity(cores=8))
    assert [j.name for j in chosen] == ["urgent"]


def test_an_empty_queue_selects_nothing() -> None:
    assert PriorityFifoBackfill().select([], Capacity(cores=8)) == []


def test_policies_are_built_by_name() -> None:
    assert isinstance(build_policy(SchedulerConfig(policy="fifo")), FifoPolicy)
    assert isinstance(build_policy(SchedulerConfig(policy="backfill")), PriorityFifoBackfill)


def test_an_unknown_policy_is_a_startup_error_listing_the_valid_ones() -> None:
    """A silent fallback would mean scheduling differently than the config says."""
    with pytest.raises(ConfigError) as excinfo:
        build_policy(SchedulerConfig(policy="magic"))
    message = str(excinfo.value)
    assert "magic" in message
    for name in POLICIES:
        assert name in message


# -- the scheduler ----------------------------------------------------------------------------


async def test_a_pass_starts_what_fits(scheduler, executor, repo, tmp_path) -> None:
    queue(repo, tmp_path, ("a", 4, 0), ("b", 4, 0))
    started = await scheduler.run_once()
    assert len(started) == 2
    assert len(executor.launched) == 2


async def test_oversubscription_is_refused(scheduler, executor, repo, tmp_path) -> None:
    queue(repo, tmp_path, ("a", 8, 0), ("b", 8, 0))
    started = await scheduler.run_once()
    assert [j.name for j in started] == ["a"]


async def test_allocation_is_taken_before_launching(
    scheduler, executor, resources, repo, tmp_path
) -> None:
    """So a second pass cannot see the cores as free while the first job starts up."""
    queue(repo, tmp_path, ("a", 6, 0))
    await scheduler.run_once()
    assert resources.allocated_cores == 6
    assert resources.free_cores == 2


async def test_a_failed_launch_releases_its_allocation(
    scheduler, executor, resources, repo, tmp_path
) -> None:
    """Otherwise a launch bug would leak cores until the daemon restarted."""
    queue(repo, tmp_path, ("a", 4, 0))
    executor.fail_next = True
    started = await scheduler.run_once()
    assert started == []
    assert resources.allocated_cores == 0


async def test_held_jobs_are_never_started(scheduler, executor, repo, tmp_path) -> None:
    (job,) = queue(repo, tmp_path, ("a", 1, 0))
    scheduler.hold(job.id)
    assert await scheduler.run_once() == []


async def test_releasing_makes_a_job_schedulable(scheduler, executor, repo, tmp_path) -> None:
    (job,) = queue(repo, tmp_path, ("a", 1, 0))
    scheduler.hold(job.id)
    scheduler.release(job.id)
    started = await scheduler.run_once()
    assert [j.id for j in started] == [job.id]


async def test_priority_change_reorders_the_next_pass(scheduler, executor, repo, tmp_path) -> None:
    _, second = queue(repo, tmp_path, ("first", 8, 0), ("second", 8, 0))
    scheduler.set_priority(second.id, 10)
    started = await scheduler.run_once()
    assert [j.id for j in started] == [second.id]


async def test_a_pass_ignores_jobs_the_executor_already_has(
    scheduler, executor, repo, tmp_path
) -> None:
    """Guards against double-launching if a pass overlaps a slow transition."""
    queue(repo, tmp_path, ("a", 1, 0))
    await scheduler.run_once()
    await scheduler.run_once()
    assert len(executor.launched) == 1


async def test_pausing_stops_admission_without_touching_running_jobs(
    scheduler, executor, repo, tmp_path
) -> None:
    queue(repo, tmp_path, ("a", 1, 0))
    scheduler.pause()
    assert await scheduler.run_once() == []
    scheduler.resume()
    assert len(await scheduler.run_once()) == 1


async def test_a_pass_survives_a_repository_failure(scheduler, repo, tmp_path) -> None:
    """A scheduling bug must not silently stop the queue for the next month."""
    queue(repo, tmp_path, ("a", 1, 0))

    def explode() -> list[Job]:
        raise RuntimeError("database on fire")

    scheduler._repo.queued = explode
    with pytest.raises(RuntimeError):
        await scheduler.run_once()


def test_explain_says_why_a_job_is_waiting(scheduler, resources, repo, tmp_path) -> None:
    """ "Waiting" without a cause is the least useful thing a scheduler can say."""
    (job,) = queue(repo, tmp_path, ("a", 8, 0))
    resources.acquire("other", ResourceRequest(cores=8))
    reason = scheduler.explain(repo.get(job.id))
    assert reason is not None and "cores free" in reason


def test_explain_reports_a_hold(scheduler, repo, tmp_path) -> None:
    (job,) = queue(repo, tmp_path, ("a", 1, 0))
    scheduler.hold(job.id)
    assert scheduler.explain(repo.get(job.id)) == "held"


def test_next_job_is_the_head_of_the_queue(scheduler, repo, tmp_path) -> None:
    queue(repo, tmp_path, ("first", 1, 0), ("urgent", 1, 9))
    head = scheduler.next_job()
    assert head is not None and head.name == "urgent"


def test_next_job_is_none_on_an_empty_queue(scheduler) -> None:
    assert scheduler.next_job() is None


async def test_state_moves_to_preparing_only_via_the_executor(
    scheduler, executor, repo, tmp_path
) -> None:
    """The scheduler admits; the executor transitions. Keeps one owner per concern."""
    (job,) = queue(repo, tmp_path, ("a", 1, 0))
    await scheduler.run_once()
    assert repo.get(job.id).state is JobState.QUEUED


# -- run after --------------------------------------------------------------------------------


async def test_a_job_with_no_dependency_starts_as_soon_as_it_fits(
    scheduler, executor, repo, tmp_path
) -> None:
    """The default, and the behaviour every existing job keeps: purely opportunistic.

    Eight cores, two four-core jobs, no dependencies -- both start on the same pass.
    """
    queue(repo, tmp_path, ("a", 4, 0), ("b", 4, 0))
    started = await scheduler.run_once()
    assert sorted(j.name for j in started) == ["a", "b"]
    assert all(job.depends_on_job_id is None for job in started)


async def test_a_dependent_job_does_not_start_while_its_parent_runs(
    scheduler, executor, resources, repo, tmp_path
) -> None:
    """The point of the feature: free cores are not enough on their own.

    Four cores are genuinely idle and the job would fit in them, and it still waits.
    """
    (parent,) = queue(repo, tmp_path, ("parent", 4, 0))
    child = queue_after(repo, tmp_path, "child", 4, parent.id)
    start_running(repo, resources, parent)

    started = await scheduler.run_once()

    assert started == []
    assert resources.free_cores >= child.cores  # it would have fitted
    assert repo.get(child.id).state is JobState.QUEUED


async def test_a_dependent_job_starts_once_its_parent_has_completed(
    scheduler, executor, resources, repo, tmp_path
) -> None:
    (parent,) = queue(repo, tmp_path, ("parent", 4, 0))
    child = queue_after(repo, tmp_path, "child", 4, parent.id)
    start_running(repo, resources, parent)
    assert await scheduler.run_once() == []

    complete(repo, resources, parent)

    started = await scheduler.run_once()
    assert [j.id for j in started] == [child.id]


async def test_an_unrelated_job_runs_while_a_dependent_one_waits(
    scheduler, executor, resources, repo, tmp_path
) -> None:
    """A dependency constrains the job that asked for one, and nothing else."""
    (parent,) = queue(repo, tmp_path, ("parent", 4, 0))
    child = queue_after(repo, tmp_path, "child", 4, parent.id)
    (unrelated,) = queue(repo, tmp_path, ("unrelated", 2, 0))
    start_running(repo, resources, parent)

    started = await scheduler.run_once()

    assert [j.id for j in started] == [unrelated.id]
    assert repo.get(child.id).state is JobState.QUEUED


async def test_a_blocked_dependent_job_does_not_hold_up_the_queue_behind_it(
    scheduler, executor, resources, repo, tmp_path
) -> None:
    """No global sequential mode: the blocked job is skipped, not the rest of the queue.

    The dependent job sits at the head of the queue on priority, and the jobs behind it
    still get the whole machine.
    """
    (parent,) = queue(repo, tmp_path, ("parent", 4, 0))
    directory = tmp_path / "blocked"
    directory.mkdir(parents=True, exist_ok=True)
    repo.create(
        JobSpec(
            workdir=directory,
            solver="fake",
            resources=ResourceRequest(cores=1),
            name="blocked",
            priority=10,
            depends_on_job_id=parent.id,
        )
    )
    queue(repo, tmp_path, ("behind", 4, 0))
    start_running(repo, resources, parent)

    started = await scheduler.run_once()

    assert [j.name for j in started] == ["behind"]


async def test_a_dependency_on_a_queued_job_blocks_until_it_has_run(
    scheduler, executor, repo, tmp_path
) -> None:
    """Satisfied by completion, not by mere existence -- a parent still in the queue counts."""
    (parent,) = queue(repo, tmp_path, ("parent", 8, 0))
    child = queue_after(repo, tmp_path, "child", 1, parent.id)

    started = await scheduler.run_once()

    assert [j.id for j in started] == [parent.id]
    assert repo.get(child.id).state is JobState.QUEUED


async def test_a_dependent_job_stays_queued_when_its_parent_fails(
    scheduler, executor, resources, repo, tmp_path
) -> None:
    """The simplest consistent behaviour: it waits, visibly, rather than running anyway.

    Nothing propagates, no new state is invented, and the job can still be cancelled by
    hand -- which is the one thing a user might reasonably want to do about it.
    """
    (parent,) = queue(repo, tmp_path, ("parent", 4, 0))
    child = queue_after(repo, tmp_path, "child", 4, parent.id)
    start_running(repo, resources, parent)
    resources.release(parent.id)
    repo.mark_finished(parent.id, state=JobState.FAILED, exit_code=1, reason=ExitReason.NONZERO)

    assert await scheduler.run_once() == []
    assert repo.get(child.id).state is JobState.QUEUED
    reason = scheduler.explain(repo.get(child.id))
    assert reason is not None and "parent" in reason and "failed" in reason


def test_explain_names_the_job_being_waited_for(scheduler, resources, repo, tmp_path) -> None:
    (parent,) = queue(repo, tmp_path, ("parent", 4, 0))
    child = queue_after(repo, tmp_path, "child", 1, parent.id)
    start_running(repo, resources, parent)

    reason = scheduler.explain(repo.get(child.id))

    assert reason is not None and "waiting for parent" in reason


def test_explain_is_unchanged_for_a_job_with_no_dependency(
    scheduler, resources, repo, tmp_path
) -> None:
    (job,) = queue(repo, tmp_path, ("a", 8, 0))
    resources.acquire("other", ResourceRequest(cores=8))
    reason = scheduler.explain(repo.get(job.id))
    assert reason is not None and "cores free" in reason


async def test_a_dependency_on_a_job_that_no_longer_exists_blocks_rather_than_starts(
    scheduler, repo, tmp_path
) -> None:
    """Defensive: the condition was never met, so the safe answer is to keep waiting."""
    (job,) = queue(repo, tmp_path, ("orphan", 1, 0))
    dangling = replace(job, depends_on_job_id="00000000-0000-0000-0000-000000000000")

    assert not scheduler._dependency_satisfied(dangling)
    reason = scheduler.explain(dangling)
    assert reason is not None and "no longer exists" in reason
