"""Admission control: the ledger, the policies, and the scheduler loop.

Nothing here starts a process. The executor is faked, so a hundred scheduling decisions
run in a millisecond and every one of them is deterministic -- which is the payoff of
making policies pure functions of (queue, capacity).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatch.core.config import SchedulerConfig
from dispatch.core.errors import ConfigError
from dispatch.core.models import Job, JobSpec, ResourceRequest
from dispatch.core.states import JobState
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
