"""CPU and GPU as two separate resources.

The claim being tested is that the distinction is real rather than cosmetic: a GPU job
occupies GPUs and only the cores it declared, a CPU job cannot hold a GPU at all, and two
requests that cannot both fit are not both admitted. Everything here runs with a
configured GPU count, so the suite passes identically on a machine with four cards and on
one with none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatch.core.config import SchedulerConfig, installed_gpus
from dispatch.core.errors import ValidationError
from dispatch.core.models import ResourceKind, ResourceRequest
from dispatch.core.states import ExitReason, JobState
from dispatch.daemon.policies import PriorityFifoBackfill
from dispatch.daemon.resources import Capacity, ResourceModel


def model(*, cores: int = 8, gpus: int = 2) -> ResourceModel:
    return ResourceModel(
        SchedulerConfig(total_cores=cores, reserved_cores=0, total_gpus=gpus),
        memory_probe=lambda: 64_000,
    )


def cpu(cores: int = 1) -> ResourceRequest:
    return ResourceRequest(cores=cores)


def gpu(gpus: int = 1, cores: int = 1) -> ResourceRequest:
    return ResourceRequest(cores=cores, gpus=gpus, kind=ResourceKind.GPU)


# -- the request ------------------------------------------------------------------------


def test_a_plain_request_is_cpu_work() -> None:
    """The default has to be what every job before this feature existed was."""
    request = ResourceRequest(cores=4)
    assert request.kind is ResourceKind.CPU
    assert request.gpus == 0
    assert not request.is_gpu


def test_a_cpu_request_cannot_carry_gpus() -> None:
    """The invariant is on the type, so no call site can construct the contradiction."""
    with pytest.raises(ValidationError, match="cannot request"):
        ResourceRequest(cores=4, gpus=1, kind=ResourceKind.CPU)


def test_a_gpu_request_must_ask_for_a_gpu() -> None:
    with pytest.raises(ValidationError, match="at least one GPU"):
        ResourceRequest(cores=4, gpus=0, kind=ResourceKind.GPU)


def test_a_negative_gpu_count_is_refused() -> None:
    with pytest.raises(ValidationError, match="negative"):
        ResourceRequest(cores=1, gpus=-1, kind=ResourceKind.GPU)


@pytest.mark.parametrize(
    ("gpus", "resource", "expected_kind", "expected_gpus"),
    [
        (None, None, ResourceKind.CPU, 0),
        (0, None, ResourceKind.CPU, 0),
        (1, None, ResourceKind.GPU, 1),  # --gpus alone implies GPU work
        (None, "gpu", ResourceKind.GPU, 1),  # --resource alone implies one GPU
        (2, "gpu", ResourceKind.GPU, 2),
        (0, "cpu", ResourceKind.CPU, 0),
    ],
)
def test_build_fills_in_whichever_flag_was_omitted(
    gpus: int | None, resource: str | None, expected_kind: ResourceKind, expected_gpus: int
) -> None:
    """Either flag settles the question; the user should not have to give both."""
    request = ResourceRequest.build(cores=4, gpus=gpus, resource=resource)
    assert request.kind is expected_kind
    assert request.gpus == expected_gpus


def test_build_refuses_a_contradiction() -> None:
    """``--resource cpu --gpus 2`` is a mistake, not something to reinterpret."""
    with pytest.raises(ValidationError, match="cannot request"):
        ResourceRequest.build(cores=4, gpus=2, resource="cpu")


def test_build_rejects_an_unknown_resource_name() -> None:
    with pytest.raises(ValidationError, match="Unknown resource type"):
        ResourceRequest.build(cores=1, resource="tpu")


def test_a_request_describes_itself_for_the_user() -> None:
    assert ResourceRequest.build(cores=20).describe() == "20 cores"
    assert ResourceRequest.build(cores=1, gpus=1).describe() == "1 GPU, 1 core"
    assert ResourceRequest.build(cores=4, gpus=2).describe() == "2 GPUs, 4 cores"


# -- the ledger --------------------------------------------------------------------------


def test_gpus_are_counted_separately_from_cores() -> None:
    ledger = model(cores=8, gpus=2)
    ledger.acquire("train", gpu(gpus=1, cores=1))
    assert ledger.allocated_gpus == 1
    assert ledger.free_gpus == 1
    assert ledger.allocated_cores == 1
    assert ledger.free_cores == 7


def test_a_gpu_job_does_not_block_unrelated_cpu_work() -> None:
    """The requirement in one test: a one-core GPU job leaves the cores alone."""
    ledger = model(cores=8, gpus=1)
    ledger.acquire("pinn", gpu(gpus=1, cores=1))
    can, reason = ledger.can_admit(cpu(cores=7))
    assert can, reason


def test_a_cpu_job_never_reserves_a_gpu() -> None:
    """A CPU job takes cores and nothing else, however many of them it takes."""
    ledger = model(cores=8, gpus=1)
    ledger.acquire("cavity", cpu(cores=7))
    assert ledger.allocated_gpus == 0
    assert ledger.free_gpus == 1
    can, reason = ledger.can_admit(gpu(gpus=1, cores=1))
    assert can, reason


def test_gpu_exhaustion_refuses_a_second_gpu_job_with_a_reason() -> None:
    ledger = model(cores=32, gpus=1)
    ledger.acquire("first", gpu())
    can, reason = ledger.can_admit(gpu())
    assert not can
    assert reason is not None and "0 of 1 GPUs free" in reason


def test_incompatible_requests_cannot_both_be_admitted() -> None:
    """Two one-GPU jobs on a one-GPU machine: the ledger admits exactly one."""
    ledger = model(cores=32, gpus=1)
    first, second = gpu(), gpu()
    assert ledger.can_admit(first)[0]
    ledger.acquire("a", first)
    assert not ledger.can_admit(second)[0]
    ledger.release("a")
    assert ledger.can_admit(second)[0]


def test_a_job_wanting_more_gpus_than_exist_is_refused_permanently() -> None:
    ledger = model(cores=8, gpus=1)
    can, reason = ledger.can_admit(gpu(gpus=4))
    assert not can
    assert reason is not None and "only 1 exist" in reason


def test_a_machine_with_no_gpus_says_so() -> None:
    """The message has to name the remedy, or the job looks like it is merely waiting."""
    ledger = model(cores=8, gpus=0)
    can, reason = ledger.can_admit(gpu())
    assert not can
    assert reason is not None
    assert "no GPUs" in reason and "scheduler.total_gpus" in reason


def test_no_gpus_are_reserved_for_responsiveness() -> None:
    """Unlike cores: nothing about logging in over SSH needs a GPU."""
    ledger = ResourceModel(SchedulerConfig(total_cores=8, reserved_cores=4, total_gpus=1))
    assert ledger.schedulable_gpus == 1
    assert ledger.free_gpus == 1


def test_capacity_treats_gpus_as_their_own_dimension() -> None:
    capacity = Capacity(cores=8, gpus=0)
    assert capacity.fits(cpu(cores=8))
    assert not capacity.fits(gpu()), "eight free cores do not make a GPU appear"

    capacity = Capacity(cores=1, gpus=4)
    assert not capacity.fits(cpu(cores=2)), "four free GPUs do not make a core appear"


def test_reserving_subtracts_from_both_pools() -> None:
    remaining = Capacity(cores=8, gpus=2).reserve(gpu(gpus=1, cores=2))
    assert (remaining.cores, remaining.gpus) == (6, 1)


# -- scheduling ---------------------------------------------------------------------------


def test_a_policy_backfills_cpu_work_past_a_blocked_gpu_job(repo, tmp_path) -> None:
    """The two pools compose with the existing policy rather than needing a new one."""
    from tests.unit.test_scheduler import queue

    jobs = queue(repo, tmp_path, ("cpu-work", 4, 0))
    blocked = repo.create(
        _spec(tmp_path / "needs-gpu", ResourceRequest(cores=1, gpus=2, kind=ResourceKind.GPU))
    )

    chosen = PriorityFifoBackfill().select([blocked, *jobs], Capacity(cores=8, gpus=1))
    assert [job.name for job in chosen] == ["cpu-work"]


def _spec(workdir: Path, resources: ResourceRequest):
    from dispatch.core.models import JobSpec

    workdir.mkdir(parents=True, exist_ok=True)
    return JobSpec(
        workdir=workdir, solver="fake", resources=resources, name=workdir.name
    )


# -- persistence ---------------------------------------------------------------------------


def test_resource_information_survives_a_round_trip(repo, tmp_path) -> None:
    job = repo.create(_spec(tmp_path / "pinn", gpu(gpus=2, cores=4)))
    reloaded = repo.get(job.id)
    assert reloaded.resource_kind is ResourceKind.GPU
    assert reloaded.gpus == 2
    assert reloaded.cores == 4


def test_the_ledger_is_rebuilt_from_the_database_after_a_restart(repo, tmp_path) -> None:
    """What a daemon restart does: clear the ledger, then re-acquire from the rows."""
    job = repo.create(_spec(tmp_path / "training", gpu(gpus=1, cores=2)))
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1234, pid_start_time=1.0)

    ledger = model(cores=8, gpus=2)
    ledger.clear()
    for active in repo.active():
        ledger.acquire(active.id, active.resources)

    assert ledger.allocated_gpus == 1
    assert ledger.allocated_cores == 2
    assert ledger.free_gpus == 1


def test_a_gpu_job_is_searchable_by_kind(repo, tmp_path, clock) -> None:
    from dispatch.core.query import parse_query

    repo.create(_spec(tmp_path / "cavity", cpu(cores=8)))
    repo.create(_spec(tmp_path / "burgers", gpu(gpus=1)))

    found = repo.search(parse_query("resource:gpu", now=clock.now()))
    assert [job.name for job in found.items] == ["burgers"]

    found = repo.search(parse_query("gpus>=1", now=clock.now()))
    assert [job.name for job in found.items] == ["burgers"]


def test_finishing_a_gpu_job_returns_its_gpus(repo, tmp_path) -> None:
    ledger = model(cores=8, gpus=1)
    job = repo.create(_spec(tmp_path / "training", gpu()))
    ledger.acquire(job.id, job.resources)
    assert ledger.free_gpus == 0

    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    repo.mark_finished(job.id, state=JobState.COMPLETED, exit_code=0, reason=ExitReason.OK)
    ledger.release(job.id)
    assert ledger.free_gpus == 1


# -- detection -----------------------------------------------------------------------------


def test_gpu_detection_never_raises_and_never_guesses_negative() -> None:
    """Whatever this machine is, the probe returns a plausible count and does not throw."""
    count = installed_gpus()
    assert isinstance(count, int)
    assert count >= 0


def test_a_configured_zero_overrides_detection() -> None:
    """A machine whose GPUs belong to something else must be able to say so."""
    assert SchedulerConfig(total_gpus=0).resolve_total_gpus() == 0
    assert SchedulerConfig(total_gpus=3).resolve_total_gpus() == 3


def test_a_negative_configured_gpu_count_is_a_configuration_error() -> None:
    from dispatch.core.errors import ConfigError

    with pytest.raises(ConfigError, match="total_gpus"):
        SchedulerConfig(total_gpus=-1)
