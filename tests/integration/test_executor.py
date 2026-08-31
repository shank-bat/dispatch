"""Process supervision, exit codes, cancellation, and recovery.

These use **real subprocesses** -- shell scripts, not solvers. Mocking the process layer
here would test that the code calls the functions it calls; the behaviour that matters is
what the kernel does with sessions, signals, and reparented children, and only real
processes exhibit it.

This is the highest-risk area in Dispatch: everything else is comparatively mechanical.
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import pytest

from dispatch.core.config import Config
from dispatch.core.models import JobSpec, ResourceRequest
from dispatch.core.plan import CommandStep, StepKind
from dispatch.core.states import ExitReason, JobState
from dispatch.daemon.events import EventBus
from dispatch.daemon.executor import JobExecutor
from dispatch.daemon.joblog import assign_log_paths
from dispatch.daemon.process import (
    ProcessManager,
    describe_exit,
    is_alive,
    matches_start_time,
    process_start_time,
)
from dispatch.daemon.recovery import recover
from dispatch.daemon.resources import ResourceModel
from dispatch.db.repository import JobRepository
from tests.conftest_daemon import FakeAdapter, make_case


@pytest.fixture
def executor(
    repo: JobRepository,
    registry,
    resources: ResourceModel,
    dispatch_config: Config,
) -> JobExecutor:
    return JobExecutor(
        repo=repo,
        registry=registry,
        resources=resources,
        config=dispatch_config,
        bus=EventBus(),
    )


def submit(
    repo: JobRepository,
    config: Config,
    workdir: Path,
    *,
    cores: int = 1,
    resources: ResourceRequest | None = None,
):
    """Create a queued job with its log paths assigned, exactly as the server would.

    Shares :func:`~dispatch.daemon.joblog.assign_log_paths` with the real submit handler
    rather than reimplementing the layout, so a test cannot pass against a log location
    the daemon no longer uses.
    """
    job = repo.create(
        JobSpec(
            workdir=workdir,
            solver="fake",
            resources=resources or ResourceRequest(cores=cores),
            name=workdir.name,
        )
    )
    return assign_log_paths(repo, config, job, FakeAdapter.log_name)


async def run_to_completion(executor: JobExecutor, job, timeout: float = 20.0):
    """Launch a job and wait for its supervising task to finish."""
    executor.launch(job)
    entry = executor._running[job.id]
    await asyncio.wait_for(asyncio.shield(entry.task), timeout)


# -- happy paths ----------------------------------------------------------------------


async def test_a_successful_job_completes(executor, repo, dispatch_config, tmp_path) -> None:
    case = make_case(tmp_path / "ok", script="echo hello; exit 0")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    finished = repo.get(job.id)
    assert finished.state is JobState.COMPLETED
    assert finished.exit_code == 0
    assert finished.exit_reason is ExitReason.OK
    assert finished.metrics.runtime_s is not None


async def test_solver_output_lands_in_the_log_file(
    executor, repo, dispatch_config, tmp_path
) -> None:
    """The kernel writes it directly; the daemon never sees the bytes."""
    case = make_case(tmp_path / "chatty", script="echo 'Time = 0.5'; exit 0")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    assert job.stdout_path == case / "log.fake"
    assert "Time = 0.5" in (case / "log.fake").read_text()


async def test_a_nonzero_exit_fails_the_job(executor, repo, dispatch_config, tmp_path) -> None:
    case = make_case(tmp_path / "bad", script="exit 3")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    finished = repo.get(job.id)
    assert finished.state is JobState.FAILED
    assert finished.exit_code == 3
    assert finished.exit_reason is ExitReason.NONZERO


async def test_a_failure_records_why_from_the_log(
    executor, repo, dispatch_config, tmp_path
) -> None:
    """The point of the feature: the exit code is not an explanation, the log is.

    Modelled on the real report -- a launcher refusing the job before the solver starts, so
    stdout is empty and the whole story is on stderr.
    """
    case = make_case(
        tmp_path / "explained",
        script="echo 'There are not enough slots available in the system' >&2; exit 1",
    )
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    finished = repo.get(job.id)
    assert finished.state is JobState.FAILED
    assert finished.exit_detail is not None
    assert "not enough slots" in finished.exit_detail


async def test_a_failure_on_stdout_is_explained_too(
    executor, repo, dispatch_config, tmp_path
) -> None:
    case = make_case(tmp_path / "stdout-fail", script="echo 'FATAL: mesh is degenerate'; exit 2")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    detail = repo.get(job.id).exit_detail
    assert detail is not None and "mesh is degenerate" in detail


async def test_a_successful_job_records_no_explanation(
    executor, repo, dispatch_config, tmp_path
) -> None:
    """There is nothing to explain about a run that worked, and a note saying so is noise."""
    case = make_case(tmp_path / "fine", script="echo 'Time = 1'; exit 0")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    assert repo.get(job.id).exit_detail is None


async def test_the_adapter_gets_to_finalize_after_a_run(
    executor, repo, dispatch_config, tmp_path
) -> None:
    case = make_case(tmp_path / "tidied", script="exit 0")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    assert (case / "finalized").read_text() == "1"


async def test_the_adapter_finalizes_even_after_a_failure(
    executor, repo, dispatch_config, tmp_path
) -> None:
    """A case edited to steer the run must be restored however the run ended."""
    case = make_case(tmp_path / "failed-but-tidied", script="exit 9")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    assert (case / "finalized").exists()


async def test_a_signalled_solver_is_recorded_as_such(
    executor, repo, dispatch_config, tmp_path
) -> None:
    case = make_case(tmp_path / "signalled", script="kill -TERM $$")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    finished = repo.get(job.id)
    assert finished.state is JobState.FAILED
    assert finished.exit_signal == "TERM"
    assert finished.exit_reason in (ExitReason.SIGNAL, ExitReason.OOM)


async def test_resources_are_released_when_a_job_ends(
    executor, repo, resources, dispatch_config, tmp_path
) -> None:
    """A leak here would silently shrink the machine over months of uptime."""
    case = make_case(tmp_path / "release", script="exit 0")
    job = submit(repo, dispatch_config, case, cores=4)
    resources.acquire(job.id, job.resources)
    await run_to_completion(executor, job)
    assert resources.allocated_cores == 0


# -- preparation steps ----------------------------------------------------------------------


async def test_preparation_runs_before_the_solver(
    executor, repo, dispatch_config, tmp_path
) -> None:
    case = make_case(
        tmp_path / "prep",
        prepare="echo prepared > marker.txt",
        script="test -f marker.txt",
    )
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    assert repo.get(job.id).state is JobState.COMPLETED
    assert (case / "marker.txt").read_text().strip() == "prepared"


async def test_a_failed_preparation_step_aborts_the_job(
    executor, repo, dispatch_config, tmp_path
) -> None:
    """The solver must not run against a case that was never set up."""
    case = make_case(tmp_path / "badprep", prepare="exit 1", script="touch SHOULD_NOT_EXIST")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    finished = repo.get(job.id)
    assert finished.state is JobState.FAILED
    assert finished.exit_reason is ExitReason.PREPARE_FAILED
    assert not (case / "SHOULD_NOT_EXIST").exists()


async def test_step_output_is_captured_in_the_transcript(
    executor, repo, dispatch_config, tmp_path
) -> None:
    case = make_case(tmp_path / "transcript", prepare="echo decomposing", script="exit 0")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    transcript = dispatch_config.paths.job_dir(job.id) / "steps.log"
    assert "decomposing" in transcript.read_text()


async def test_step_descriptions_reach_the_audit_trail(
    executor, repo, dispatch_config, tmp_path
) -> None:
    case = make_case(tmp_path / "events", prepare="true", script="exit 0")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    details = [event.detail for event in repo.events(job.id)]
    assert any("Preparing" in detail for detail in details)


# -- exit sentinels -------------------------------------------------------------------------------


async def test_the_exit_code_is_written_to_disk(executor, repo, dispatch_config, tmp_path) -> None:
    """The mechanism that makes an exit code recoverable after a daemon crash."""
    case = make_case(tmp_path / "sentinel", script="exit 42")
    job = submit(repo, dispatch_config, case)
    await run_to_completion(executor, job)

    sentinel = dispatch_config.paths.job_dir(job.id) / "exit_code"
    assert sentinel.read_text().strip() == "42"


def test_reading_a_missing_sentinel_returns_none(tmp_path: Path) -> None:
    """Which is exactly what makes a job UNKNOWN rather than assumed."""
    assert ProcessManager.read_exit_file(tmp_path / "absent") is None


def test_reading_a_corrupt_sentinel_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "exit_code"
    path.write_text("not a number")
    assert ProcessManager.read_exit_file(path) is None


# -- cancellation ---------------------------------------------------------------------------------


async def test_cancelling_a_running_job_stops_it(executor, repo, dispatch_config, tmp_path) -> None:
    case = make_case(tmp_path / "long", script="sleep 60")
    job = submit(repo, dispatch_config, case)
    executor.launch(job)

    await _wait_for_state(repo, job.id, JobState.RUNNING)
    assert await executor.cancel(job.id)

    entry = executor._running.get(job.id)
    if entry is not None:
        await asyncio.wait_for(asyncio.shield(entry.task), 20)

    finished = repo.get(job.id)
    assert finished.state is JobState.CANCELLED
    assert finished.exit_reason is ExitReason.CANCELLED


async def test_a_cancelled_job_still_finalizes_its_case(
    executor, repo, dispatch_config, tmp_path
) -> None:
    """The path that most needs it, and the easiest one to lose.

    A graceful stop works by *editing the case* -- and the edit is only correct for the run
    it stopped. Skipping finalize here is what leaves a case that quietly does nothing on
    every subsequent run.
    """
    case = make_case(tmp_path / "cancelled-tidied", script="sleep 60")
    job = submit(repo, dispatch_config, case)
    executor.launch(job)

    await _wait_for_state(repo, job.id, JobState.RUNNING)
    await executor.cancel(job.id)

    entry = executor._running.get(job.id)
    if entry is not None:
        await asyncio.wait_for(asyncio.shield(entry.task), 20)

    assert (case / "finalized").exists()


async def test_cancelling_kills_the_whole_process_group(
    executor, repo, dispatch_config, tmp_path
) -> None:
    """An MPI launcher's children are the actual ranks; killing only the parent leaks them."""
    case = make_case(
        tmp_path / "tree",
        script="sleep 60 & echo $! > child.pid; wait",
    )
    job = submit(repo, dispatch_config, case)
    executor.launch(job)

    child_pid_file = case / "child.pid"
    for _ in range(200):
        if child_pid_file.exists() and child_pid_file.read_text().strip():
            break
        await asyncio.sleep(0.05)
    child_pid = int(child_pid_file.read_text().strip())
    assert is_alive(child_pid)

    await executor.cancel(job.id, force=True)
    entry = executor._running.get(job.id)
    if entry is not None:
        await asyncio.wait_for(asyncio.shield(entry.task), 20)

    for _ in range(100):
        if not is_alive(child_pid):
            break
        await asyncio.sleep(0.05)
    assert not is_alive(child_pid), "the grandchild survived cancellation"


async def test_cancelling_an_unknown_job_reports_false(executor) -> None:
    assert not await executor.cancel("never-existed")


async def test_a_graceful_stop_is_tried_first(executor, repo, dispatch_config, tmp_path) -> None:
    """For a CFD solver this means "write and exit", which leaves a usable result."""
    case = make_case(tmp_path / "graceful", script="sleep 30", graceful=True)
    job = submit(repo, dispatch_config, case)
    executor.launch(job)
    await _wait_for_state(repo, job.id, JobState.RUNNING)

    assert await executor.cancel(job.id)
    details = [event.detail for event in repo.events(job.id)]
    assert any("stop cleanly" in detail for detail in details)

    # The adapter claimed success, so no signal was sent and the job is still running.
    assert repo.get(job.id).state is JobState.RUNNING
    await executor.cancel(job.id, force=True)
    entry = executor._running.get(job.id)
    if entry is not None:
        await asyncio.wait_for(asyncio.shield(entry.task), 20)


# -- process primitives ----------------------------------------------------------------------------


async def test_spawned_processes_get_their_own_session(tmp_path: Path) -> None:
    """Detachment is what stops a stray Ctrl-C in the daemon's terminal reaching a solver."""
    manager = ProcessManager()
    step = CommandStep(
        argv=["/bin/sh", "-c", "sleep 5"],
        cwd=tmp_path,
        description="sleep",
        kind=StepKind.SOLVE,
    )
    handle = await manager.spawn(step)
    try:
        assert os.getpgid(handle.pid) == handle.pid
        assert os.getpgid(handle.pid) != os.getpgid(os.getpid())
    finally:
        manager.signal_group(handle.pid, signal.SIGKILL)
        await manager.wait(handle)


async def test_spawning_a_missing_program_is_reported_clearly(tmp_path: Path) -> None:
    from dispatch.daemon.process import SpawnError

    manager = ProcessManager()
    step = CommandStep(
        argv=["definitely-not-a-real-program"],
        cwd=tmp_path,
        description="nope",
        kind=StepKind.SOLVE,
    )
    with pytest.raises(SpawnError, match="command not found"):
        await manager.spawn(step)


async def test_a_missing_program_fails_the_job(
    executor, repo, dispatch_config, tmp_path, registry
) -> None:
    case = tmp_path / "noexec"
    case.mkdir()
    (case / "fake.job").write_text("exit 0")
    job = submit(repo, dispatch_config, case)

    # Replace the plan's program with one that does not exist.
    adapter = registry.get("fake")
    original = adapter.plan

    def broken_plan(ctx):
        plan = original(ctx)
        from dispatch.core.plan import ExecutionPlan

        return ExecutionPlan(
            steps=tuple(
                CommandStep(
                    argv=["definitely-not-a-real-program"],
                    cwd=step.cwd,
                    description=step.description,
                    kind=step.kind,
                    env=step.env,
                )
                for step in plan.steps
            )
        )

    adapter.plan = broken_plan
    try:
        await run_to_completion(executor, job)
    finally:
        adapter.plan = original

    assert repo.get(job.id).state is JobState.FAILED


def test_start_time_reads_from_proc() -> None:
    start = process_start_time(os.getpid())
    assert start is not None and start > 0


def test_start_time_of_a_dead_process_is_none() -> None:
    assert process_start_time(999_999_999) is None


def test_pid_reuse_is_detected() -> None:
    """The guard that stops re-adoption attaching to an unrelated process after a reboot."""
    assert matches_start_time(os.getpid(), process_start_time(os.getpid()))
    assert not matches_start_time(os.getpid(), 1.0)
    assert not matches_start_time(os.getpid(), None)


def test_describe_exit_splits_signals_from_codes() -> None:
    assert describe_exit(0) == (0, None)
    assert describe_exit(3) == (3, None)
    assert describe_exit(128 + int(signal.SIGKILL)) == (137, "KILL")
    assert describe_exit(None) == (None, None)


def test_signalling_a_dead_group_reports_false() -> None:
    assert not ProcessManager().signal_group(999_999_999, signal.SIGTERM)


# -- timeouts --------------------------------------------------------------------------------------


async def test_a_step_that_exceeds_its_timeout_is_killed(tmp_path: Path) -> None:
    manager = ProcessManager()
    step = CommandStep(
        argv=["/bin/sh", "-c", "sleep 30"],
        cwd=tmp_path,
        description="slow",
        kind=StepKind.SOLVE,
        timeout_s=0.3,
    )
    handle = await manager.spawn(step)
    assert await manager.wait(handle, timeout=step.timeout_s) is None
    await manager.terminate(handle, grace_s=0.1)
    assert not is_alive(handle.pid)


# -- recovery --------------------------------------------------------------------------------------


async def test_a_job_that_finished_while_the_daemon_was_down_is_recovered(
    repo, resources, executor, dispatch_config, tmp_path
) -> None:
    """Case 1: the sentinel exists, so the real exit code survives the restart."""
    case = make_case(tmp_path / "away", script="exit 0")
    job = submit(repo, dispatch_config, case)
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=999_999_998, pid_start_time=1.0)

    sentinel = dispatch_config.paths.job_dir(job.id) / "exit_code"
    sentinel.write_text("7")

    report = recover(repo=repo, resources=resources, executor=executor, config=dispatch_config)

    assert report.finished == [job.id]
    finished = repo.get(job.id)
    assert finished.state is JobState.FAILED
    assert finished.exit_code == 7


async def test_a_still_running_job_is_readopted(
    repo, resources, executor, dispatch_config, tmp_path
) -> None:
    """Case 2: a daemon restart during a three-day simulation is a non-event."""
    case = make_case(tmp_path / "survivor", script="sleep 30")
    job = submit(repo, dispatch_config, case)

    process = await asyncio.create_subprocess_exec(
        "/bin/sh", "-c", "sleep 30", start_new_session=True
    )
    try:
        repo.mark_preparing(job.id)
        repo.mark_started(
            job.id, pid=process.pid, pid_start_time=process_start_time(process.pid) or 0.0
        )

        report = recover(repo=repo, resources=resources, executor=executor, config=dispatch_config)

        assert report.readopted == [job.id]
        assert repo.get(job.id).state is JobState.RUNNING
        assert resources.allocated_cores == job.cores
        assert executor.is_running(job.id)
    finally:
        process.kill()
        await process.wait()
        await executor.shutdown()


async def test_a_lost_job_becomes_unknown(
    repo, resources, executor, dispatch_config, tmp_path
) -> None:
    """Case 3: the machine was reset mid-run. UNKNOWN is the honest answer."""
    case = make_case(tmp_path / "lost", script="sleep 1")
    job = submit(repo, dispatch_config, case)
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=999_999_997, pid_start_time=1.0)

    report = recover(repo=repo, resources=resources, executor=executor, config=dispatch_config)

    assert report.lost == [job.id]
    finished = repo.get(job.id)
    assert finished.state is JobState.UNKNOWN
    assert finished.exit_reason is ExitReason.LOST


async def test_recovery_refuses_a_reused_pid(
    repo, resources, executor, dispatch_config, tmp_path
) -> None:
    """Re-adopting a stranger would mean signalling it on the next cancel."""
    case = make_case(tmp_path / "reused", script="sleep 1")
    job = submit(repo, dispatch_config, case)
    repo.mark_preparing(job.id)
    # A live pid, but recorded with a start time that cannot match: this pid now belongs
    # to something else.
    repo.mark_started(job.id, pid=os.getpid(), pid_start_time=1.0)

    report = recover(repo=repo, resources=resources, executor=executor, config=dispatch_config)

    assert report.lost == [job.id]
    assert repo.get(job.id).state is JobState.UNKNOWN


async def test_recovery_with_nothing_to_do(repo, resources, executor, dispatch_config) -> None:
    report = recover(repo=repo, resources=resources, executor=executor, config=dispatch_config)
    assert report.total == 0
    assert "no jobs" in report.describe()


async def test_the_recovered_finish_time_comes_from_the_sentinel(
    repo, resources, executor, dispatch_config, tmp_path
) -> None:
    """So a job that ended eight hours ago does not report an eight-hour runtime."""
    case = make_case(tmp_path / "mtime", script="exit 0")
    job = submit(repo, dispatch_config, case)
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=999_999_996, pid_start_time=1.0)

    sentinel = dispatch_config.paths.job_dir(job.id) / "exit_code"
    sentinel.write_text("0")
    ended = repo.get(job.id).started_at
    assert ended is not None
    os.utime(sentinel, (ended + 30, ended + 30))

    recover(repo=repo, resources=resources, executor=executor, config=dispatch_config)
    assert repo.get(job.id).metrics.runtime_s == pytest.approx(30, abs=1)


# -- shutdown --------------------------------------------------------------------------------------


async def test_shutdown_leaves_simulations_running(
    executor, repo, dispatch_config, tmp_path
) -> None:
    """Stopping Dispatch and stopping a week-long run are different actions."""
    case = make_case(tmp_path / "keepgoing", script="sleep 30")
    job = submit(repo, dispatch_config, case)
    executor.launch(job)
    await _wait_for_state(repo, job.id, JobState.RUNNING)

    pid = repo.get(job.id).pid
    assert pid is not None

    await executor.shutdown(kill_jobs=False)
    assert is_alive(pid), "shutting down the daemon killed a running simulation"

    os.killpg(os.getpgid(pid), signal.SIGKILL)


async def test_shutdown_can_kill_jobs_when_explicitly_asked(
    executor, repo, dispatch_config, tmp_path
) -> None:
    case = make_case(tmp_path / "drain", script="sleep 30")
    job = submit(repo, dispatch_config, case)
    executor.launch(job)
    await _wait_for_state(repo, job.id, JobState.RUNNING)
    pid = repo.get(job.id).pid
    assert pid is not None

    await executor.shutdown(kill_jobs=True)
    for _ in range(100):
        if not is_alive(pid):
            break
        await asyncio.sleep(0.05)
    assert not is_alive(pid)


# -- helpers ---------------------------------------------------------------------------------------


async def _wait_for_state(
    repo: JobRepository, job_id: str, state: JobState, timeout: float = 20.0
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if repo.get(job_id).state is state:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"job never reached {state}; it is {repo.get(job_id).state}")
