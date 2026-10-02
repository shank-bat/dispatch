"""Starting a case from its own latest output, and changing its core count mid-run.

Two features that share one mechanism. Both end with a job planned to continue from
whatever the simulation itself last wrote, which is the thing reboot resume already knew
how to do (§6.7) -- so most of what is asserted here is that the existing machinery is
being *reused* rather than paralleled: the same ``startFrom latestTime`` edit, the same
``RUNNING -> QUEUED`` edge, the same ledger.

No solver is installed and none is needed: plans are asserted as command lists, and the
case is a handful of text files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatch.adapters import foamdict
from dispatch.adapters.base import CaseContext
from dispatch.adapters.openfoam import START_FROM_BACKUP, OpenFOAMAdapter
from dispatch.core.errors import ValidationError
from dispatch.core.models import JobSpec, ResourceRequest, SweepSpec
from dispatch.core.states import ExitReason, JobState
from tests.unit.test_adapters import foam_case


def ctx(case: Path, *, cores: int = 1, **kwargs) -> CaseContext:
    return CaseContext(workdir=case, cores=cores, env={"PATH": "/nonexistent"}, **kwargs)


def start_from(case: Path) -> str | None:
    return foamdict.read_value(case / "system" / "controlDict", "startFrom")


def write_time(case: Path, time: str) -> None:
    """Give the case a written time directory, as a solver would."""
    (case / time).mkdir(exist_ok=True)
    (case / time / "U").write_text("// field\n")


# == 1. START FROM THE LATEST WRITTEN TIME ================================================


def test_a_plain_run_starts_where_the_case_says(tmp_path: Path) -> None:
    """The default is untouched: this is what every job did before the option existed."""
    case = foam_case(tmp_path / "case")
    write_time(case, "0.3")

    OpenFOAMAdapter({}).plan(ctx(case))
    assert start_from(case) == "startTime"
    assert not (case / START_FROM_BACKUP).exists()


def test_continuing_points_the_case_at_its_latest_time(tmp_path: Path) -> None:
    """The OpenFOAM mechanism, reached through the flag rather than reimplemented."""
    case = foam_case(tmp_path / "case")
    write_time(case, "0.3")

    OpenFOAMAdapter({}).plan(ctx(case, resume=True))
    assert start_from(case) == "latestTime"


def test_the_original_start_is_restored_afterwards(tmp_path: Path) -> None:
    """The edit steers one run. Left behind it would move every later run's start."""
    case = foam_case(tmp_path / "case")
    write_time(case, "0.3")
    adapter = OpenFOAMAdapter({})

    adapter.plan(ctx(case, resume=True))
    assert start_from(case) == "latestTime"
    adapter.finalize(ctx(case))
    assert start_from(case) == "startTime"
    assert not (case / START_FROM_BACKUP).exists()


def test_the_plan_says_where_the_run_will_begin(tmp_path: Path) -> None:
    """Dry-run output has to reflect the setting, not merely obey it."""
    case = foam_case(tmp_path / "case")
    write_time(case, "0.3")

    plan = OpenFOAMAdapter({}).plan(ctx(case, resume=True))
    assert "continuing from t = 0.3" in plan.solve.description

    plain = OpenFOAMAdapter({}).plan(ctx(foam_case(tmp_path / "other")))
    assert "continuing" not in plain.solve.description


def test_a_case_with_nothing_written_says_so_rather_than_pretending(tmp_path: Path) -> None:
    """`startFrom latestTime` on such a case is correct -- it finds only the initial
    condition -- and the plan should state that instead of implying hours were preserved."""
    case = foam_case(tmp_path / "fresh")

    plan = OpenFOAMAdapter({}).plan(ctx(case, resume=True))
    assert "no saved time to continue from" in plan.solve.description


def test_the_initial_condition_is_not_a_saved_state(tmp_path: Path) -> None:
    """`0/` is where the case starts, so reporting it as preserved work would be a lie."""
    case = foam_case(tmp_path / "case")
    assert OpenFOAMAdapter({}).resume_point(ctx(case)) is None

    write_time(case, "0.25")
    assert OpenFOAMAdapter({}).resume_point(ctx(case)) == "t = 0.25"


def test_continuing_composes_with_re_decomposition(tmp_path: Path) -> None:
    """Both halves of a repartition at once: a new core count and a continued run."""
    case = foam_case(tmp_path / "case", processors=4)
    write_time(case, "0.3")

    plan = OpenFOAMAdapter({}).plan(ctx(case, cores=8, resume=True))
    programs = [step.program for step in plan.steps]
    assert programs == ["reconstructPar", "rm", "decomposePar", "mpirun"]
    assert start_from(case) == "latestTime"
    assert "continuing from t = 0.3" in plan.solve.description


# -- a preview must not touch the case -----------------------------------------------------


def test_a_preview_leaves_the_case_alone(tmp_path: Path) -> None:
    """`finalize` never runs for a preview, so an edit made under one would outlive it --
    silently changing where every later run of the case begins."""
    case = foam_case(tmp_path / "case")
    write_time(case, "0.3")

    plan = OpenFOAMAdapter({}).plan(ctx(case, resume=True, dry_run=True))
    assert start_from(case) == "startTime"
    assert not (case / START_FROM_BACKUP).exists()
    # Still described accurately: the preview is about what a real run would do.
    assert "continuing from t = 0.3" in plan.solve.description


def test_a_preview_does_not_rewrite_the_decomposition(tmp_path: Path) -> None:
    """Pre-existing: `--dry-run` was writing decomposeParDict into a case it only described."""
    case = foam_case(tmp_path / "case")
    plan = OpenFOAMAdapter({}).plan(ctx(case, cores=4, dry_run=True))

    assert not (case / "system" / "decomposeParDict").exists()
    assert ["decomposePar", "-force"] in [list(step.argv) for step in plan.steps]


def test_a_real_run_still_writes_the_decomposition(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case")
    OpenFOAMAdapter({}).plan(ctx(case, cores=4))
    assert (case / "system" / "decomposeParDict").is_file()


# -- the flag is persisted and sticky --------------------------------------------------------


def spec(tmp_path: Path, name: str = "case", *, cores: int = 4, **kwargs) -> JobSpec:
    workdir = tmp_path / name
    workdir.mkdir(parents=True, exist_ok=True)
    return JobSpec(
        workdir=workdir,
        solver="openfoam",
        resources=ResourceRequest(cores=cores),
        name=name,
        **kwargs,
    )


def test_the_request_survives_a_round_trip(repo, tmp_path) -> None:
    job = repo.create(spec(tmp_path, start_from_latest=True))
    assert repo.get(job.id).start_from_latest


def test_an_ordinary_job_does_not_carry_it(repo, tmp_path) -> None:
    assert repo.create(spec(tmp_path)).start_from_latest is False


def test_both_reasons_ask_the_adapter_for_the_same_thing(repo, tmp_path) -> None:
    """A reboot resume and an explicit request differ in the history, not in the plan."""
    explicit = repo.create(spec(tmp_path, "a", start_from_latest=True))
    assert explicit.continues_from_saved_state

    rebooted = repo.create(spec(tmp_path, "b"))
    repo.mark_preparing(rebooted.id)
    repo.mark_started(rebooted.id, pid=1, pid_start_time=1.0)
    rebooted = repo.requeue_for_resume(rebooted.id, detail="after a reboot")
    assert rebooted.continues_from_saved_state
    assert not rebooted.start_from_latest, "a reboot is not a configuration"


def test_the_setting_outlives_a_restart(repo, tmp_path) -> None:
    """Sticky, unlike the transient reboot flag: a case asked to continue still should."""
    job = repo.create(spec(tmp_path, start_from_latest=True))
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    repo.requeue_for_resume(job.id, detail="after a reboot")
    repo.clear_resume_request(job.id)

    after = repo.get(job.id)
    assert after.resume_requested is False
    assert after.start_from_latest is True
    assert after.continues_from_saved_state


# -- sweeps ------------------------------------------------------------------------------------


def test_a_sweep_records_the_setting_as_its_own_configuration(repo, tmp_path) -> None:
    """Stored on the sweep as well as its members, so it stays answerable after history
    has been pruned -- the same reason `cores_per_job` is."""
    cases = [tmp_path / "sweep" / name for name in ("a", "b")]
    for case in cases:
        case.mkdir(parents=True)
    sweep_spec = SweepSpec(
        root=tmp_path / "sweep",
        solver="openfoam",
        cases=cases,
        cores_per_job=2,
        concurrency=1,
        start_from_latest=True,
    )
    members = [
        JobSpec(
            workdir=case,
            solver="openfoam",
            resources=ResourceRequest(cores=2),
            name=case.name,
            sweep_id=sweep_spec.sweep_id,
            sweep_position=position,
            start_from_latest=True,
        )
        for position, case in enumerate(cases)
    ]

    sweep, jobs = repo.create_sweep(sweep_spec, members)
    assert sweep.start_from_latest is True
    assert all(job.start_from_latest for job in jobs)
    assert repo.get_sweep(sweep.id).start_from_latest is True


def test_a_sweep_defaults_to_starting_cases_afresh(repo, tmp_path) -> None:
    case = tmp_path / "sweep" / "a"
    case.mkdir(parents=True)
    sweep_spec = SweepSpec(
        root=tmp_path / "sweep", solver="openfoam", cases=[case], cores_per_job=1, concurrency=1
    )
    sweep, jobs = repo.create_sweep(
        sweep_spec,
        [
            JobSpec(
                workdir=case,
                solver="openfoam",
                resources=ResourceRequest(cores=1),
                name="a",
                sweep_id=sweep_spec.sweep_id,
                sweep_position=0,
            )
        ],
    )
    assert sweep.start_from_latest is False
    assert jobs[0].start_from_latest is False


# == 2. PAUSE AT THE NEXT WRITE AND CHANGE THE CORE COUNT =================================


def running(repo, tmp_path, name: str = "case", cores: int = 4):
    job = repo.create(spec(tmp_path, name, cores=cores))
    repo.mark_preparing(job.id)
    return repo.mark_started(job.id, pid=4242, pid_start_time=1.0)


def test_a_request_is_recorded_without_disturbing_the_run(repo, tmp_path) -> None:
    """Nothing changes until the solver reaches its next write."""
    job = running(repo, tmp_path, cores=4)
    after = repo.request_repartition(job.id, 8)

    assert after.state is JobState.RUNNING
    assert after.cores == 4, "the allocation it holds is unchanged"
    assert after.repartition_cores == 8
    assert after.repartition_pending


def test_the_request_is_visible_in_the_audit_trail(repo, tmp_path) -> None:
    job = running(repo, tmp_path)
    repo.request_repartition(job.id, 8)
    assert any("resume on 8 core" in event.detail for event in repo.events(job.id))


def test_a_job_that_is_not_running_cannot_be_repartitioned(repo, tmp_path) -> None:
    job = repo.create(spec(tmp_path))
    with pytest.raises(ValidationError, match="no run to repartition"):
        repo.request_repartition(job.id, 8)


def test_a_repartition_to_nothing_is_refused(repo, tmp_path) -> None:
    job = running(repo, tmp_path)
    with pytest.raises(ValidationError, match="at least one core"):
        repo.request_repartition(job.id, 0)


def test_a_request_can_be_withdrawn(repo, tmp_path) -> None:
    job = running(repo, tmp_path)
    repo.request_repartition(job.id, 8)
    assert repo.cancel_repartition(job.id).repartition_cores is None


def test_requeueing_changes_state_and_cores_together(repo, tmp_path) -> None:
    """One statement. There is no instant at which the job is queued on a core count that
    does not match what it asked for, or carries a request already honoured."""
    job = running(repo, tmp_path, cores=4)
    repo.request_repartition(job.id, 12)

    after = repo.requeue_for_repartition(job.id, cores=12, detail="repartitioned")
    assert after.state is JobState.QUEUED
    assert after.cores == 12
    assert after.repartition_cores is None
    assert after.resume_requested is True, "it must continue, not start over"
    assert after.pid is None and after.started_at is None


def test_a_repartitioned_job_keeps_its_place_in_the_queue(repo, tmp_path) -> None:
    """A core change is not a resubmission: it must not go behind later work."""
    first = running(repo, tmp_path, "first", cores=4)
    later = repo.create(spec(tmp_path, "later"))

    requeued = repo.requeue_for_repartition(first.id, cores=8, detail="repartitioned")
    assert requeued.seq < later.seq
    assert [job.name for job in repo.queued()] == ["first", "later"]


def test_the_new_core_count_is_what_gets_scheduled(repo, resources, tmp_path) -> None:
    """The ledger and the job row agree after the change, which is the whole point."""
    job = running(repo, tmp_path, cores=4)
    resources.acquire(job.id, job.resources)
    assert resources.allocated_cores == 4

    repo.request_repartition(job.id, 7)
    requeued = repo.requeue_for_repartition(job.id, cores=7, detail="repartitioned")
    resources.release(job.id)

    assert resources.allocated_cores == 0
    resources.acquire(requeued.id, requeued.resources)
    assert resources.allocated_cores == 7


def test_the_resume_flag_makes_the_restart_continue(repo, tmp_path) -> None:
    """End to end at the model level: a repartitioned job asks the adapter to continue."""
    job = running(repo, tmp_path, cores=4)
    repo.request_repartition(job.id, 8)
    requeued = repo.requeue_for_repartition(job.id, cores=8, detail="repartitioned")
    assert requeued.continues_from_saved_state


def test_a_finished_job_cannot_be_repartitioned(repo, tmp_path) -> None:
    job = running(repo, tmp_path)
    repo.mark_finished(job.id, state=JobState.COMPLETED, exit_code=0, reason=ExitReason.OK)
    with pytest.raises(ValidationError):
        repo.request_repartition(job.id, 8)
