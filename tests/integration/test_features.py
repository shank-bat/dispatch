"""The new surface, through a real daemon over a real socket.

Everything here goes through the same path the TUI and CLI use, because that is where the
pieces meet: an adapter names a log, the server records it, the executor opens it, the
sampler reads it, and a client asks for its contents back as numbers.

Nothing needs a solver, a GPU, or a machine learning framework: the fake adapter is a
shell script, and the GPU count is configuration.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

from dispatch.core.config import Config, ProjectsConfig
from dispatch.core.states import JobState
from dispatch.daemon.main import Daemon
from dispatch.daemon.recovery import recover
from dispatch.ipc.client import DaemonClient, RemoteError
from dispatch.ipc.protocol import Method
from tests.conftest_daemon import make_case


@pytest.fixture
def gpu_config(dispatch_config: Config, tmp_path: Path) -> Config:
    """A machine with two GPUs and a projects tree, both entirely inside ``tmp_path``."""
    projects = tmp_path / "projects"
    projects.mkdir(exist_ok=True)
    return replace(
        dispatch_config,
        scheduler=replace(dispatch_config.scheduler, total_gpus=2),
        projects=ProjectsConfig(root=projects),
    )


@pytest.fixture
async def daemon(gpu_config: Config, registry) -> AsyncIterator[Daemon]:
    instance = Daemon(gpu_config, load_plugins=False)
    instance.registry = registry
    instance.executor._registry = registry
    instance.inspector._registry = registry
    instance.server._registry = registry

    instance.scheduler.start()
    await instance.server.start()
    try:
        yield instance
    finally:
        await instance.server.stop()
        await instance.scheduler.stop()
        await instance.executor.shutdown(kill_jobs=True)
        instance.conn.close()


@pytest.fixture
async def client(daemon: Daemon) -> AsyncIterator[DaemonClient]:
    connection = DaemonClient(daemon.config.paths.socket, autostart=False)
    await connection.connect()
    try:
        yield connection
    finally:
        await connection.close()


async def await_state(
    client: DaemonClient, job_id: str, state: str, timeout: float = 20.0
) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    seen = "?"
    while loop.time() < deadline:
        detail = await client.call(Method.JOB_GET, id=job_id)
        seen = detail["job"]["state"]
        if seen == state:
            return detail
        await asyncio.sleep(0.05)
    raise AssertionError(f"job never reached {state}; last seen {seen}")


async def submit(client: DaemonClient, case: Path, **params) -> dict:
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), **params)
    return result["job"]


# -- logs land in the working directory ---------------------------------------------------


async def test_the_log_is_written_beside_the_case(client: DaemonClient, tmp_path: Path) -> None:
    """The requirement, end to end: a human browsing the case directory finds the log."""
    case = make_case(tmp_path / "cavity", script="echo 'Time = 0.5'; exit 0")
    job = await submit(client, case, cores=1)

    assert job["log_path"] == str(case / "log.fake")
    await await_state(client, job["id"], JobState.COMPLETED.value)
    assert "Time = 0.5" in (case / "log.fake").read_text()


async def test_stdout_and_stderr_are_both_captured(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(
        tmp_path / "both", script="echo to-stdout; echo to-stderr >&2; exit 0"
    )
    job = await submit(client, case, cores=1)
    await await_state(client, job["id"], JobState.COMPLETED.value)

    written = (case / "log.fake").read_text()
    assert "to-stdout" in written
    assert "to-stderr" in written


async def test_the_log_path_is_visible_before_the_job_runs(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Recorded at submission, so ``dispatch show`` can answer "where will this write"."""
    case = make_case(tmp_path / "queued", script="sleep 5")
    job = await submit(client, case, cores=1)
    detail = await client.call(Method.JOB_GET, id=job["id"])
    assert detail["job"]["log_path"] == str(case / "log.fake")


async def test_a_dry_run_names_the_log_it_would_write(
    client: DaemonClient, tmp_path: Path
) -> None:
    case = make_case(tmp_path / "planned")
    report = await client.call(Method.CASE_DRYRUN, workdir=str(case), cores=1)
    assert report["log_path"] == str(case / "log.fake")
    assert not (case / "log.fake").exists(), "a dry run writes nothing"


async def test_a_second_run_rotates_the_first_run_s_log(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Neither run's output may be lost, and neither job's record may point at the other's."""
    case = make_case(tmp_path / "twice", script="echo first; exit 0")
    first = await submit(client, case, cores=1)
    await await_state(client, first["id"], JobState.COMPLETED.value)

    (case / "fake.job").write_text("echo second; exit 0")
    second = await submit(client, case, cores=1)
    await await_state(client, second["id"], JobState.COMPLETED.value)

    assert (case / "log.fake").read_text().strip() == "second"
    assert (case / "log.fake.1").read_text().strip() == "first"

    detail = await client.call(Method.JOB_GET, id=first["id"])
    assert detail["job"]["log_path"] == str(case / "log.fake.1")
    assert Path(detail["job"]["log_path"]).read_text().strip() == "first"


async def test_the_step_transcript_still_goes_to_the_job_directory(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """Internal bookkeeping stays out of the user's case directory."""
    case = make_case(tmp_path / "prepared", script="exit 0", prepare="echo decomposing")
    job = await submit(client, case, cores=1)
    await await_state(client, job["id"], JobState.COMPLETED.value)

    transcript = daemon.config.paths.job_dir(job["id"]) / "steps.log"
    assert "decomposing" in transcript.read_text()
    assert not (case / "steps.log").exists()


async def test_the_exit_sentinel_still_goes_to_the_job_directory(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """It is what makes exit codes survive a daemon crash; it must not move."""
    case = make_case(tmp_path / "sentinel", script="exit 3")
    job = await submit(client, case, cores=1)
    await await_state(client, job["id"], JobState.FAILED.value)
    assert (daemon.config.paths.job_dir(job["id"]) / "exit_code").exists()


async def test_a_read_only_case_directory_still_runs(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """Falling back is right; refusing to run a job because of where its log goes is not."""
    case = make_case(tmp_path / "readonly", script="echo ran; exit 0")
    job = await submit(client, case, cores=1)
    case.chmod(0o500)
    try:
        detail = await await_state(client, job["id"], JobState.COMPLETED.value)
    finally:
        case.chmod(0o700)

    written = Path(detail["job"]["stdout_path"])
    assert "ran" in written.read_text()
    assert daemon.config.paths.job_dir(job["id"]) in written.parents


# -- resources -----------------------------------------------------------------------------


async def test_a_gpu_job_records_its_request(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "training", script="exit 0")
    job = await submit(client, case, cores=1, gpus=1, resource="gpu")
    assert job["resource_kind"] == "gpu"
    assert job["gpus"] == 1


async def test_gpus_alone_imply_gpu_work(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "implied", script="exit 0")
    job = await submit(client, case, cores=1, gpus=1)
    assert job["resource_kind"] == "gpu"


async def test_a_cpu_job_asking_for_gpus_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    case = make_case(tmp_path / "confused", script="exit 0")
    with pytest.raises(RemoteError, match="cannot request"):
        await submit(client, case, cores=1, gpus=2, resource="cpu")


async def test_asking_for_more_gpus_than_exist_is_refused_at_submission(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Rather than queued forever with no explanation."""
    case = make_case(tmp_path / "greedy", script="exit 0")
    with pytest.raises(RemoteError, match="never get"):
        await submit(client, case, cores=1, gpus=8, resource="gpu")


async def test_the_snapshot_reports_the_gpu_ledger(client: DaemonClient) -> None:
    snapshot = await client.call(Method.SYSTEM_SNAPSHOT)
    assert snapshot["total_gpus"] == 2
    assert snapshot["free_gpus"] == 2


async def test_a_running_gpu_job_holds_its_gpus_in_the_ledger(
    client: DaemonClient, tmp_path: Path
) -> None:
    case = make_case(tmp_path / "holding", script="sleep 3")
    job = await submit(client, case, cores=1, gpus=2, resource="gpu")
    await await_state(client, job["id"], JobState.RUNNING.value)

    snapshot = await client.call(Method.SYSTEM_SNAPSHOT)
    assert snapshot["allocated_gpus"] == 2
    assert snapshot["free_gpus"] == 0

    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_a_gpu_job_does_not_hold_the_machine_s_cores(
    client: DaemonClient, tmp_path: Path
) -> None:
    """A one-core GPU job must not stop a large CPU job from starting beside it."""
    gpu_case = make_case(tmp_path / "gpu-job", script="sleep 3")
    cpu_case = make_case(tmp_path / "cpu-job", script="sleep 3")

    gpu_job = await submit(client, gpu_case, cores=1, gpus=1, resource="gpu")
    await await_state(client, gpu_job["id"], JobState.RUNNING.value)
    cpu_job = await submit(client, cpu_case, cores=7)
    await await_state(client, cpu_job["id"], JobState.RUNNING.value)

    for job_id in (gpu_job["id"], cpu_job["id"]):
        await client.call(Method.JOB_CANCEL, id=job_id, force=True)


async def test_a_second_gpu_job_waits_for_the_first(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Two requests that cannot both fit are not both admitted."""
    first = await submit(
        client, make_case(tmp_path / "gpu-a", script="sleep 3"), cores=1, gpus=2, resource="gpu"
    )
    await await_state(client, first["id"], JobState.RUNNING.value)

    second = await submit(
        client, make_case(tmp_path / "gpu-b", script="exit 0"), cores=1, gpus=2, resource="gpu"
    )
    detail = await client.call(Method.JOB_GET, id=second["id"])
    assert detail["job"]["state"] == JobState.QUEUED.value
    assert "GPU" in (detail["waiting_because"] or "")

    await client.call(Method.JOB_CANCEL, id=first["id"], force=True)
    await await_state(client, second["id"], JobState.COMPLETED.value)


async def test_resources_survive_a_daemon_restart(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """Recovery rebuilds the ledger from the rows, GPUs included."""
    case = make_case(tmp_path / "long", script="sleep 30")
    job = await submit(client, case, cores=2, gpus=1, resource="gpu")
    await await_state(client, job["id"], JobState.RUNNING.value)

    # What a restart does: forget everything, then reconcile with the machine.
    daemon.executor._running.clear()
    daemon.resources.clear()
    report = recover(
        repo=daemon.repo,
        resources=daemon.resources,
        executor=daemon.executor,
        config=daemon.config,
    )

    assert report.readopted == [job["id"]]
    assert daemon.resources.allocated_gpus == 1
    assert daemon.resources.allocated_cores == 2

    reloaded = daemon.repo.get(job["id"])
    assert reloaded.gpus == 1
    assert reloaded.log_path == case / "log.fake"

    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_a_readopted_job_keeps_writing_to_the_same_log(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """Re-adoption must not rotate the log out from under a live process."""
    case = make_case(tmp_path / "surviving", script="echo alive; sleep 30")
    job = await submit(client, case, cores=1)
    await await_state(client, job["id"], JobState.RUNNING.value)
    await asyncio.sleep(0.3)

    daemon.executor._running.clear()
    daemon.resources.clear()
    recover(
        repo=daemon.repo,
        resources=daemon.resources,
        executor=daemon.executor,
        config=daemon.config,
    )

    assert not (case / "log.fake.1").exists()
    assert "alive" in (case / "log.fake").read_text()
    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


# -- plot data -------------------------------------------------------------------------------


async def test_a_job_s_series_come_back_over_the_socket(
    client: DaemonClient, tmp_path: Path
) -> None:
    """The full chain: log on disk, adapter parses, client receives numbers."""
    case = make_case(
        tmp_path / "curve",
        script="for i in 1 2 3 4 5; do echo \"step $i\"; done; exit 0",
    )
    job = await submit(client, case, cores=1)
    await await_state(client, job["id"], JobState.COMPLETED.value)

    payload = await client.call(Method.JOB_SERIES, id=job["id"])
    assert payload["path"] == str(case / "log.fake")
    assert payload["samples"] == 0, "the fake adapter declares no series"
    assert payload["series"] == []


async def test_a_job_with_no_log_yet_reports_no_series(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Queued, so nothing has been written. An empty answer, not an error."""
    case = make_case(tmp_path / "waiting", script="sleep 5")
    job = await submit(client, case, cores=1, gpus=2, resource="gpu")
    blocked = await submit(client, make_case(tmp_path / "blocked"), cores=1, gpus=2,
                           resource="gpu")
    del job

    payload = await client.call(Method.JOB_SERIES, id=blocked["id"])
    assert payload["series"] == []
    assert payload["samples"] == 0


async def test_series_extraction_uses_the_real_adapter(
    daemon: Daemon, client: DaemonClient, tmp_path: Path
) -> None:
    """Swap in an adapter that parses, and the same request returns its series."""
    from dispatch.adapters.foamlog import parse_foam_log
    from dispatch.core.series import PlotData
    from tests.conftest_daemon import FakeAdapter

    class ParsingAdapter(FakeAdapter):
        name = "fake"

        def parse_series(self, text: str, ctx) -> PlotData:
            return parse_foam_log(text)

    daemon.registry._instances["fake"] = ParsingAdapter({})

    case = make_case(
        tmp_path / "residuals",
        script=(
            "printf 'Time = 0.005\\n'; "
            "printf 'smoothSolver:  Solving for Ux, Initial residual = 0.02, x\\n'; "
            "printf 'Time = 0.01\\n'; "
            "printf 'smoothSolver:  Solving for Ux, Initial residual = 0.002, x\\n'; "
            "exit 0"
        ),
    )
    job = await submit(client, case, cores=1)
    await await_state(client, job["id"], JobState.COMPLETED.value)

    payload = await client.call(Method.JOB_SERIES, id=job["id"])
    keys = {series["key"] for series in payload["series"]}
    assert payload["samples"] == 2
    assert {"iteration", "time", "residual(Ux)"} <= keys

    residual = next(s for s in payload["series"] if s["key"] == "residual(Ux)")
    assert residual["values"] == [0.02, 0.002]
    assert residual["samples"] == [0, 1]


async def test_plotting_never_changes_a_job_s_state(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Reading a log is a read. It must not be able to reclassify a run."""
    case = make_case(tmp_path / "failed", script="echo 'FOAM FATAL ERROR' ; exit 1")
    job = await submit(client, case, cores=1)
    await await_state(client, job["id"], JobState.FAILED.value)

    await client.call(Method.JOB_SERIES, id=job["id"])
    detail = await client.call(Method.JOB_GET, id=job["id"])
    assert detail["job"]["state"] == JobState.FAILED.value


# -- project search ----------------------------------------------------------------------------


async def test_projects_are_searchable_over_the_socket(
    client: DaemonClient, gpu_config: Config
) -> None:
    root = gpu_config.projects.root
    (root / "openfoam" / "cavity").mkdir(parents=True)
    (root / "paper1" / "cavityRe100").mkdir(parents=True)
    (root / "notes").mkdir(parents=True)

    payload = await client.call(Method.PROJECTS_SEARCH, query="cavity")
    found = [entry["relative"] for entry in payload["results"]]
    assert found[0] == "openfoam/cavity"
    assert "paper1/cavityRe100" in found
    assert "notes" not in found
    assert payload["error"] is None


async def test_search_results_carry_the_case_marker(
    client: DaemonClient, gpu_config: Config
) -> None:
    """The mark comes from the real adapters, which is why the search runs daemon-side."""
    root = gpu_config.projects.root
    make_case(root / "cases" / "cavity")
    (root / "cases" / "cavity-notes").mkdir(parents=True)

    payload = await client.call(Method.PROJECTS_SEARCH, query="cavity")
    marks = {entry["relative"]: entry["case"] for entry in payload["results"]}
    assert marks["cases/cavity"] == "fake"
    assert marks["cases/cavity-notes"] is None


async def test_a_selected_project_can_be_submitted_directly(
    client: DaemonClient, gpu_config: Config
) -> None:
    """The definition of done: search, select, submit, and the log appears in that folder."""
    root = gpu_config.projects.root
    case = make_case(root / "paper" / "cavity", script="echo running; exit 0")

    payload = await client.call(Method.PROJECTS_SEARCH, query="cavity")
    chosen = payload["results"][0]["path"]
    assert chosen == str(case)

    job = await submit(client, Path(chosen), cores=1)
    await await_state(client, job["id"], JobState.COMPLETED.value)
    assert (case / "log.fake").read_text().strip() == "running"
