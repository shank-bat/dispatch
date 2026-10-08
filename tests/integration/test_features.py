"""The new surface, through a real daemon over a real socket.

Everything here goes through the same path the TUI and CLI use, because that is where the
pieces meet: an adapter names a log, the server records it, the executor opens it, the
sampler reads it, and a client asks for its contents back as numbers.

Nothing needs a solver, a GPU, or a machine learning framework: the fake adapter is a
shell script, and the GPU count is configuration.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

from dispatch.core.config import Config, ProjectsConfig
from dispatch.core.states import ExitReason, JobState
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


# -- sweep folders, end to end ---------------------------------------------------------------


def sweep_folder(root: Path, count: int, *, script: str = "exit 0") -> Path:
    """A directory of identically-shaped cases, all detected as the same adapter."""
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        make_case(root / f"case_{index + 1:03d}", script=script)
    return root


async def test_a_folder_of_cases_is_offered_as_a_sweep(
    client: DaemonClient, tmp_path: Path
) -> None:
    """The browser learns about the sweep from the listing it already fetches."""
    root = sweep_folder(tmp_path / "study", 4)
    listing = await client.call(Method.FS_LIST, path=str(root.parent))

    assert listing["sweep"] is None  # the parent holds one directory, not four cases

    inside = await client.call(Method.FS_LIST, path=str(root))
    assert inside["sweep"] is not None
    assert inside["sweep"]["count"] == 4
    assert inside["sweep"]["cases"] == [f"case_{i:03d}" for i in range(1, 5)]


async def test_submitting_a_sweep_queues_every_case_in_order(
    client: DaemonClient, tmp_path: Path
) -> None:
    root = sweep_folder(tmp_path / "aoa", 5, script="sleep 5")
    result = await client.call(
        Method.SWEEP_SUBMIT, root=str(root), cores_per_job=2, concurrency=2
    )

    assert result["sweep"]["total"] == 5
    assert result["sweep"]["cores_per_job"] == 2
    assert result["sweep"]["concurrency"] == 2
    assert [job["name"] for job in result["jobs"]] == [
        f"case_{i:03d}" for i in range(1, 6)
    ]
    assert [job["sweep_position"] for job in result["jobs"]] == [0, 1, 2, 3, 4]
    assert all(job["cores"] == 2 for job in result["jobs"])


async def test_a_sweep_runs_only_its_configured_number_at_once(
    client: DaemonClient, tmp_path: Path
) -> None:
    """The hard cap, through the real daemon rather than a hand-driven pass.

    Eight cores are free and each case wants one, so a scheduler that treated the limit as
    advice would start all five.
    """
    root = sweep_folder(tmp_path / "capped", 5, script="sleep 5")
    await client.call(Method.SWEEP_SUBMIT, root=str(root), cores_per_job=1, concurrency=2)

    await asyncio.sleep(1.0)
    page = await client.call(Method.JOB_LIST, states=["RUNNING", "PREPARING"], limit=50)

    assert len(page["items"]) == 2
    assert {job["name"] for job in page["items"]} == {"case_001", "case_002"}


async def test_a_normal_job_still_starts_beside_a_capped_sweep(
    client: DaemonClient, tmp_path: Path
) -> None:
    """The cores a sweep is not using stay available to unrelated work."""
    root = sweep_folder(tmp_path / "sharing", 5, script="sleep 5")
    await client.call(Method.SWEEP_SUBMIT, root=str(root), cores_per_job=1, concurrency=2)
    other = await submit(client, make_case(tmp_path / "other", script="sleep 5"), cores=4)

    await await_state(client, other["id"], "RUNNING")

    page = await client.call(Method.JOB_LIST, states=["RUNNING", "PREPARING"], limit=50)
    names = {job["name"] for job in page["items"]}
    assert names == {"case_001", "case_002", "other"}


async def test_the_next_case_starts_when_one_finishes(
    client: DaemonClient, tmp_path: Path
) -> None:
    root = sweep_folder(tmp_path / "rolling", 4, script="exit 0")
    result = await client.call(
        Method.SWEEP_SUBMIT, root=str(root), cores_per_job=1, concurrency=2
    )

    for job in result["jobs"]:
        await await_state(client, job["id"], "COMPLETED")

    listed = await client.call(Method.SWEEP_LIST)
    sweep = listed["sweeps"][0]
    assert sweep["finished"] == 4
    assert sweep["running"] == 0


async def test_a_directory_that_is_not_a_sweep_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Conservative detection, enforced at the point of submission too."""
    root = sweep_folder(tmp_path / "mixed", 2)
    (root / "notes").mkdir()

    with pytest.raises(RemoteError, match="not a sweep folder"):
        await client.call(
            Method.SWEEP_SUBMIT, root=str(root), cores_per_job=1, concurrency=1
        )


async def test_a_sweep_asking_for_impossible_cores_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Refused once, rather than queueing forty jobs that can never start."""
    root = sweep_folder(tmp_path / "toobig", 3)

    with pytest.raises(RemoteError, match="can never get"):
        await client.call(
            Method.SWEEP_SUBMIT, root=str(root), cores_per_job=999, concurrency=1
        )


async def test_a_sweep_and_its_order_survive_a_daemon_restart(
    daemon, client: DaemonClient, tmp_path: Path
) -> None:
    """Reopening the database is enough: nothing about a sweep lives in memory."""
    root = sweep_folder(tmp_path / "durable", 4, script="sleep 5")
    result = await client.call(
        Method.SWEEP_SUBMIT, root=str(root), cores_per_job=1, concurrency=2
    )
    sweep_id = result["sweep"]["id"]

    restored = daemon.repo.get_sweep(sweep_id)
    assert restored is not None
    assert (restored.cores_per_job, restored.concurrency, restored.total) == (1, 2, 4)
    assert [job.name for job in daemon.repo.sweep_members(sweep_id)] == [
        f"case_{i:03d}" for i in range(1, 5)
    ]


# == pausing a run to change its core count (§6.13) =========================================
#
# The whole feature, through a real daemon and real processes. The fake solver watches for a
# sentinel the adapter's clean stop touches, which is the same shape as OpenFOAM's
# `stopAt writeNow`: the adapter writes into the case and the solver decides when to act, at
# the end of a step it has finished writing.

WATCHES_FOR_STOP = (
    "echo running; "
    "while [ ! -f stop ]; do sleep 0.05; done; "
    "echo 'wrote the timestep'; exit 0"
)


async def await_cores(
    client: DaemonClient, job_id: str, cores: int, timeout: float = 25.0
) -> dict:
    """Wait until a job is running on exactly ``cores``.

    The resize passes through QUEUED so briefly that polling for that state is a race: the
    scheduler re-admits on the same nudge that requeued the job. What matters is the
    outcome, so that is what is waited for; `test_a_paused_job_waits_for_its_new_cores`
    pins the queued state deliberately by making the new count unavailable.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    seen: object = None
    while loop.time() < deadline:
        detail = await client.call(Method.JOB_GET, id=job_id)
        seen = (detail["job"]["state"], detail["job"]["cores"])
        if seen == (JobState.RUNNING.value, cores):
            return detail
        await asyncio.sleep(0.05)
    raise AssertionError(f"job never ran on {cores} cores; last seen {seen}")


async def test_a_running_job_can_be_paused_and_resumed_on_more_cores(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """The headline: stop at a write, change the count, come back and continue."""
    case = make_case(tmp_path / "resize", script=WATCHES_FOR_STOP, graceful=True)
    job = await submit(client, case, cores=2)
    await await_state(client, job["id"], JobState.RUNNING.value)

    result = await client.call(Method.JOB_REPARTITION, id=job["id"], cores=5)
    assert result["pausing"] is True and result["cores"] == 5

    detail = await await_cores(client, job["id"], 5)
    assert detail["job"]["repartition_cores"] is None, "the request was honoured and cleared"
    assert any("requeued on 5" in event["detail"] for event in detail["events"])
    assert any(
        "last saved state" in event["detail"] or "no saved state" in event["detail"]
        for event in detail["events"]
    ), "the restart continued rather than silently starting over"
    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_a_paused_job_waits_for_its_new_cores(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Requeued, not special-cased: it waits for capacity like any other queued job."""
    blocker = await submit(client, make_case(tmp_path / "blocker", script="sleep 30"), cores=6)
    await await_state(client, blocker["id"], JobState.RUNNING.value)

    case = make_case(tmp_path / "waits", script=WATCHES_FOR_STOP, graceful=True)
    job = await submit(client, case, cores=2)
    await await_state(client, job["id"], JobState.RUNNING.value)

    await client.call(Method.JOB_REPARTITION, id=job["id"], cores=5)
    detail = await await_state(client, job["id"], JobState.QUEUED.value)
    assert detail["job"]["cores"] == 5
    assert detail["job"]["resume_requested"] is True
    assert detail["job"]["repartition_cores"] is None
    assert "core" in (detail["waiting_because"] or "")

    for job_id in (blocker["id"], job["id"]):
        await client.call(Method.JOB_CANCEL, id=job_id, force=True)


async def test_the_timestep_is_written_before_the_process_stops(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Not a kill: the solver reaches its own exit after finishing the step."""
    case = make_case(tmp_path / "clean", script=WATCHES_FOR_STOP, graceful=True)
    job = await submit(client, case, cores=1)
    await await_state(client, job["id"], JobState.RUNNING.value)

    await client.call(Method.JOB_REPARTITION, id=job["id"], cores=3)
    await await_cores(client, job["id"], 3)

    # The first run's own words, written after the stop was requested and before it exited.
    assert "wrote the timestep" in (case / "log.fake.1").read_text()
    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_the_ledger_follows_the_new_core_count(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """Resource accounting changes with the job's state, not independently of it."""
    case = make_case(tmp_path / "ledger", script=WATCHES_FOR_STOP, graceful=True)
    job = await submit(client, case, cores=2)
    await await_state(client, job["id"], JobState.RUNNING.value)
    assert daemon.resources.allocated_cores == 2

    await client.call(Method.JOB_REPARTITION, id=job["id"], cores=5)
    await await_cores(client, job["id"], 5)
    assert daemon.resources.allocated_cores == 5

    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_a_paused_job_keeps_its_place_in_the_queue(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """A core change is not a resubmission; it must not go behind later work."""
    first = await submit(
        client,
        make_case(tmp_path / "first", script=WATCHES_FOR_STOP, graceful=True),
        cores=2,
    )
    await await_state(client, first["id"], JobState.RUNNING.value)
    later = await submit(client, make_case(tmp_path / "later", script="sleep 30"), cores=8)

    # Admission is paused for the duration, so the queued state can be observed at all:
    # otherwise the scheduler re-admits on the same nudge that requeued the job.
    daemon.scheduler.pause()
    await client.call(Method.JOB_REPARTITION, id=first["id"], cores=8)
    await await_state(client, first["id"], JobState.QUEUED.value)

    assert first["seq"] < later["seq"]
    # Queue *position*, which is the derived truth (§13.3), not the listing order.
    page = await client.call(Method.JOB_LIST, states=["QUEUED"], limit=10)
    positions = {item["id"]: item["queue_position"] for item in page["items"]}
    assert positions[first["id"]] < positions[later["id"]]

    daemon.scheduler.resume()
    for job_id in (first["id"], later["id"]):
        await client.call(Method.JOB_CANCEL, id=job_id, force=True)


async def test_a_pending_pause_is_visible_while_it_is_pending(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Otherwise the interface looks like the keypress did nothing."""
    case = make_case(tmp_path / "pending", script="sleep 30", graceful=True)
    job = await submit(client, case, cores=2)
    await await_state(client, job["id"], JobState.RUNNING.value)

    await client.call(Method.JOB_REPARTITION, id=job["id"], cores=4)
    detail = await client.call(Method.JOB_GET, id=job["id"])
    # The solver ignores the stop, so the job is still running with the request showing.
    assert detail["job"]["state"] == JobState.RUNNING.value
    assert detail["job"]["repartition_cores"] == 4

    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_a_solver_with_no_clean_stop_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Killing a solver mid-timestep to resize it would corrupt the very write this
    feature exists to preserve."""
    case = make_case(tmp_path / "nostop", script="sleep 30")  # graceful=False
    job = await submit(client, case, cores=2)
    await await_state(client, job["id"], JobState.RUNNING.value)

    with pytest.raises(RemoteError, match="no clean stop"):
        await client.call(Method.JOB_REPARTITION, id=job["id"], cores=4)

    detail = await client.call(Method.JOB_GET, id=job["id"])
    assert detail["job"]["state"] == JobState.RUNNING.value
    assert detail["job"]["repartition_cores"] is None, "the request was withdrawn"
    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def blocked(client: DaemonClient, tmp_path: Path, **params) -> tuple[dict, dict]:
    """A running eight-core job, and a job queued behind it that cannot start yet."""
    blocker = await submit(client, make_case(tmp_path / "blocker", script="sleep 30"), cores=8)
    await await_state(client, blocker["id"], JobState.RUNNING.value)
    waiting = await submit(client, make_case(tmp_path / "waiting", **params), cores=8)
    return blocker, waiting


async def test_a_queued_job_is_re_cored_at_once(client: DaemonClient, tmp_path: Path) -> None:
    """Nothing to pause: it has not started, so the new count is simply what it starts on."""
    blocker, waiting = await blocked(client, tmp_path)
    result = await client.call(Method.JOB_REPARTITION, id=waiting["id"], cores=4)

    assert result["pausing"] is False
    assert result["job"]["state"] == "QUEUED"
    assert result["job"]["cores"] == 4
    assert result["job"]["repartition_cores"] is None, "no pending pause is recorded"
    await client.call(Method.JOB_CANCEL, id=blocker["id"], force=True)


async def test_a_re_cored_job_starts_on_its_new_count(
    daemon: Daemon, client: DaemonClient, tmp_path: Path
) -> None:
    blocker, waiting = await blocked(client, tmp_path, script="sleep 30")
    await client.call(Method.JOB_REPARTITION, id=waiting["id"], cores=3)
    await client.call(Method.JOB_CANCEL, id=blocker["id"], force=True)

    await await_state(client, waiting["id"], JobState.RUNNING.value)
    assert daemon.resources.allocated_cores == 3
    await client.call(Method.JOB_CANCEL, id=waiting["id"], force=True)


async def test_shrinking_a_queued_job_can_let_it_start_now(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Four of eight cores busy: an eight-core job waits, and on four it starts at once."""
    blocker = await submit(client, make_case(tmp_path / "half", script="sleep 30"), cores=4)
    await await_state(client, blocker["id"], JobState.RUNNING.value)
    waiting = await submit(client, make_case(tmp_path / "big", script="sleep 30"), cores=8)
    await asyncio.sleep(0.2)
    assert (await client.call(Method.JOB_GET, id=waiting["id"]))["job"]["state"] == "QUEUED"

    await client.call(Method.JOB_REPARTITION, id=waiting["id"], cores=4)
    await await_state(client, waiting["id"], JobState.RUNNING.value)
    for job in (blocker, waiting):
        await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_a_held_job_can_be_re_cored(client: DaemonClient, tmp_path: Path) -> None:
    blocker, waiting = await blocked(client, tmp_path)
    await client.call(Method.JOB_HOLD, id=waiting["id"])
    result = await client.call(Method.JOB_REPARTITION, id=waiting["id"], cores=2)
    assert result["job"]["state"] == "HELD" and result["job"]["cores"] == 2
    await client.call(Method.JOB_CANCEL, id=blocker["id"], force=True)


async def test_a_queued_job_keeps_its_place_when_re_cored(
    client: DaemonClient, tmp_path: Path
) -> None:
    blocker, first = await blocked(client, tmp_path)
    second = await submit(client, make_case(tmp_path / "second"), cores=8)
    await client.call(Method.JOB_REPARTITION, id=first["id"], cores=6)

    listing = await client.call(Method.JOB_LIST, states=["QUEUED"])
    order = [job["id"] for job in sorted(listing["items"], key=lambda j: j["queue_position"])]
    assert order == [first["id"], second["id"]]
    await client.call(Method.JOB_CANCEL, id=blocker["id"], force=True)


async def test_re_coring_a_queued_job_validates_the_new_count(
    client: DaemonClient, tmp_path: Path
) -> None:
    """As a submission would. Forcing it through is still possible."""
    blocker, waiting = await blocked(client, tmp_path)
    (Path(waiting["workdir"]) / "invalid").touch()

    with pytest.raises(RemoteError, match="does not pass validation") as excinfo:
        await client.call(Method.JOB_REPARTITION, id=waiting["id"], cores=2)
    assert excinfo.value.detail and "validation" in excinfo.value.detail
    assert (await client.call(Method.JOB_GET, id=waiting["id"]))["job"]["cores"] == 8

    forced = await client.call(Method.JOB_REPARTITION, id=waiting["id"], cores=2, force=True)
    assert forced["job"]["cores"] == 2
    await client.call(Method.JOB_CANCEL, id=blocker["id"], force=True)


async def test_re_coring_a_queued_job_beyond_the_machine_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    blocker, waiting = await blocked(client, tmp_path)
    with pytest.raises(RemoteError, match="can never be scheduled"):
        await client.call(Method.JOB_REPARTITION, id=waiting["id"], cores=64)
    await client.call(Method.JOB_CANCEL, id=blocker["id"], force=True)


async def test_a_finished_job_cannot_be_re_cored(client: DaemonClient, tmp_path: Path) -> None:
    job = await submit(client, make_case(tmp_path / "done"), cores=2)
    await await_state(client, job["id"], JobState.COMPLETED.value)
    with pytest.raises(RemoteError, match="completed"):
        await client.call(Method.JOB_REPARTITION, id=job["id"], cores=4)


async def test_resizing_to_more_cores_than_exist_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    """It would stop the run and then never resume it."""
    case = make_case(tmp_path / "toobig", script="sleep 30", graceful=True)
    job = await submit(client, case, cores=2)
    await await_state(client, job["id"], JobState.RUNNING.value)

    with pytest.raises(RemoteError, match="never be scheduled"):
        await client.call(Method.JOB_REPARTITION, id=job["id"], cores=9999)
    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_resizing_to_the_same_count_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    case = make_case(tmp_path / "same", script="sleep 30", graceful=True)
    job = await submit(client, case, cores=3)
    await await_state(client, job["id"], JobState.RUNNING.value)

    with pytest.raises(RemoteError, match="already set to 3"):
        await client.call(Method.JOB_REPARTITION, id=job["id"], cores=3)
    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_a_run_that_does_not_stop_cleanly_is_not_requeued(
    client: DaemonClient, tmp_path: Path
) -> None:
    """A crash while stopping may have left a half-written timestep, so resuming from
    "the latest time" could resume from a corrupt one. It fails honestly instead."""
    case = make_case(
        tmp_path / "crashes",
        script="while [ ! -f stop ]; do sleep 0.05; done; exit 7",
        graceful=True,
    )
    job = await submit(client, case, cores=2)
    await await_state(client, job["id"], JobState.RUNNING.value)

    await client.call(Method.JOB_REPARTITION, id=job["id"], cores=4)
    detail = await await_state(client, job["id"], JobState.FAILED.value)
    assert detail["job"]["exit_code"] == 7
    assert detail["job"]["repartition_cores"] is None
    assert any("not resized" in event["detail"] for event in detail["events"])


async def test_cancelling_still_ends_a_job_that_was_being_resized(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Cancel and resize want opposite outcomes from the same stop; cancel must win."""
    case = make_case(tmp_path / "both", script=WATCHES_FOR_STOP, graceful=True)
    job = await submit(client, case, cores=2)
    await await_state(client, job["id"], JobState.RUNNING.value)

    await client.call(Method.JOB_REPARTITION, id=job["id"], cores=4)
    await client.call(Method.JOB_CANCEL, id=job["id"], force=True)
    detail = await await_state(client, job["id"], JobState.CANCELLED.value)
    assert detail["job"]["state"] == JobState.CANCELLED.value


# == the case information view and rendering (§9.8, §8.10) ==================================


def foam_case_at(root: Path) -> Path:
    """A structurally valid OpenFOAM case, built from text files."""
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (root / "0").mkdir(exist_ok=True)
    (root / "system" / "controlDict").write_text(
        "FoamFile { object controlDict; }\napplication icoFoam;\nstartFrom startTime;\n"
        "startTime 0;\nendTime 0.5;\ndeltaT 0.01;\nwriteControl timeStep;\nwriteInterval 20;\n"
    )
    (root / "system" / "fvSchemes").touch()
    (root / "system" / "fvSolution").touch()
    (root / "constant" / "polyMesh" / "owner").write_text(
        'FoamFile { object owner; note "nPoints:8  nCells:400  nFaces:6"; }\n'
    )
    (root / "constant" / "polyMesh" / "boundary").write_text(
        "FoamFile { object boundary; }\n2\n(\n"
        "    walls { type wall; nFaces 3; startFace 0; }\n"
        "    frontAndBack { type empty; nFaces 2; startFace 3; }\n)\n"
    )
    (root / "0" / "p").write_text(
        "FoamFile { object p; }\ndimensions [0 2 -2 0 0 0 0];\n"
    )
    return root


@pytest.fixture
def foam_client(daemon: Daemon, client: DaemonClient) -> DaemonClient:
    """A client talking to a daemon that also has the real OpenFOAM adapter."""
    from dispatch.adapters.openfoam import OpenFOAMAdapter

    daemon.registry.register(OpenFOAMAdapter)
    return client


async def test_a_case_describes_itself_over_the_socket(
    foam_client: DaemonClient, tmp_path: Path
) -> None:
    """The whole point of the layering: the adapter reads the dictionaries, not the screen."""
    case = foam_case_at(tmp_path / "cavity")
    report = await foam_client.call(Method.CASE_INFO, path=str(case))

    assert report["solver"] == "openfoam"
    assert report["title"] == "cavity"
    sections = {item["title"]: item for item in report["sections"]}
    fields = {f["label"]: f["value"] for f in sections["solver"]["fields"]}
    assert fields["Application"] == "icoFoam"
    assert {f["label"]: f["value"] for f in sections["mesh"]["fields"]}["Cells"] == "400"
    assert "2D" in {f["label"]: f["value"] for f in sections["mesh"]["fields"]}["Dimensionality"]


async def test_the_description_works_on_a_directory_never_submitted(
    foam_client: DaemonClient, tmp_path: Path
) -> None:
    """Which is when it is most useful -- deciding whether to submit at all."""
    case = foam_case_at(tmp_path / "unsubmitted")
    report = await foam_client.call(Method.CASE_INFO, path=str(case))
    assert report["path"] == str(case)
    assert not any(item["title"] == "dispatch history" for item in report["sections"])


async def test_the_description_includes_what_dispatch_has_run_there(
    daemon: Daemon, client: DaemonClient, tmp_path: Path
) -> None:
    """Not in any of the solver's own files, and often the most useful thing on the page."""
    from dispatch.adapters.openfoam import OpenFOAMAdapter

    case = foam_case_at(tmp_path / "ran")
    # The job row is written directly rather than run: a directory holding both a
    # controlDict and a fake.job is claimed by two adapters at once, which detection
    # correctly refuses to resolve. What is under test is the history section, not running.
    from dispatch.core.models import JobSpec, ResourceRequest

    created = daemon.repo.create(
        JobSpec(
            workdir=case,
            solver="openfoam",
            resources=ResourceRequest(cores=4),
            name="ran",
        )
    )
    daemon.repo.mark_preparing(created.id)
    daemon.repo.mark_started(created.id, pid=1, pid_start_time=1.0)
    daemon.repo.mark_finished(
        created.id, state=JobState.COMPLETED, exit_code=0, reason=ExitReason.OK
    )

    daemon.registry.register(OpenFOAMAdapter)
    report = await client.call(Method.CASE_INFO, path=str(case))
    history = next(
        item for item in report["sections"] if item["title"] == "dispatch history"
    )
    values = {f["label"]: f["value"] for f in history["fields"]}
    assert values["Runs"] == "1"
    assert values["Latest"] == "completed"


async def test_describing_something_that_is_not_a_case_is_refused(
    foam_client: DaemonClient, tmp_path: Path
) -> None:
    plain = tmp_path / "notes"
    plain.mkdir()
    with pytest.raises(RemoteError, match="No solver recognises"):
        await foam_client.call(Method.CASE_INFO, path=str(plain))


async def test_a_render_is_planned_without_running_it(
    foam_client: DaemonClient, tmp_path: Path, monkeypatch
) -> None:
    """``--dry-run`` on a render: the command and the camera, and nothing touched."""
    binary = tmp_path / "bin" / "pvbatch"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")

    case = foam_case_at(tmp_path / "wing")
    result = await foam_client.call(
        Method.CASE_RENDER, path=str(case), kind="mesh", preset="front", dry_run=True
    )

    assert result["rendered"] is False
    assert result["command"][0].endswith("pvbatch")
    assert "--force-offscreen-rendering" in result["command"]
    # The case declares an empty patch, so it is planar; it has no blockMeshDict, so its
    # extent is unknown and the camera is settled inside the script instead.
    assert any("2D" in note for note in result["notes"])
    assert not (case / "postProcessing").exists(), "a preview touches nothing"


async def test_rendering_without_paraview_says_so(
    foam_client: DaemonClient, tmp_path: Path, monkeypatch
) -> None:
    """Rather than reporting a command that failed for reasons nobody can act on."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    case = foam_case_at(tmp_path / "wing")
    with pytest.raises(RemoteError, match=r"ParaView .* was not found"):
        await foam_client.call(Method.CASE_RENDER, path=str(case), kind="mesh")


async def test_a_render_failure_is_reported_with_its_output(
    foam_client: DaemonClient, tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "bin" / "pvbatch"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 'cannot open display'\nexit 3\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")

    case = foam_case_at(tmp_path / "wing")
    with pytest.raises(RemoteError, match="exit 3"):
        # `wait`, as the CLI does: the failure is then the response rather than an event.
        await foam_client.call(Method.CASE_RENDER, path=str(case), kind="mesh", wait=True)


async def test_a_successful_render_reports_what_it_wrote(
    foam_client: DaemonClient, tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "bin" / "pvbatch"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho rendered\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")

    case = foam_case_at(tmp_path / "wing")
    result = await foam_client.call(
        Method.CASE_RENDER, path=str(case), kind="mesh", preset="top", wait=True
    )
    assert result["rendered"] is True
    assert result["produced"] == ["rendered"]
    # The case is planar and has no blockMeshDict, so the script aims along the plane's
    # normal itself, and the name says so rather than claiming the angle that was asked for.
    assert result["outputs"][0].endswith("mesh-plane.png")
    # The generated script and the reader stub are the only things written into the case.
    assert (case / "postProcessing" / "dispatch" / "mesh-plane.py").is_file()
    assert (case / "wing.foam").is_file()


# == choosing a field, and renders that run in the background (§8.10) ========================


def fake_tool(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / name
    binary.write_text(f"#!/bin/sh\n{body}\n")
    binary.chmod(0o755)
    return binary


def foam_with_results(root: Path) -> Path:
    case = foam_case_at(root)
    for time in ("0", "0.2"):
        (case / time).mkdir(exist_ok=True)
        (case / time / "p").write_text("FoamFile { class volScalarField; object p; }\n")
        (case / time / "U").write_text("FoamFile { class volVectorField; object U; }\n")
    return case


async def test_a_case_lists_what_it_can_be_coloured_by(
    foam_client: DaemonClient, tmp_path: Path
) -> None:
    case = foam_with_results(tmp_path / "wing")
    result = await foam_client.call(Method.CASE_FIELDS, path=str(case))
    assert [item["name"] for item in result["fields"]] == ["U", "p"]
    assert result["fields"][0]["label"] == "U (velocity, magnitude)"


async def test_a_render_runs_in_the_background_and_reports_when_done(
    daemon: Daemon, foam_client: DaemonClient, tmp_path: Path, monkeypatch
) -> None:
    """The request returns at once, other requests are answered meanwhile, and the result
    arrives as an event."""
    from dispatch.adapters.paraview import video_encoder
    from dispatch.ipc.protocol import Event, Topic

    tools = tmp_path / "bin"
    fake_tool(tools, "pvbatch", 'echo "frame 1/2  t = 0"; sleep 1; echo "frame 2/2  t = 0.2"')
    fake_tool(tools, "ffmpeg", 'echo " V....D libx264  H.264"')
    video_encoder.cache_clear()
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")

    events: list = []
    watcher = DaemonClient(
        daemon.config.paths.socket, autostart=False, on_event=lambda note: events.append(note)
    )
    await watcher.connect()
    try:
        await watcher.subscribe([str(Topic.RENDERS)])
        case = foam_with_results(tmp_path / "wing")
        started = await foam_client.call(
            Method.CASE_RENDER, path=str(case), kind="animation", field="U", preset="front"
        )
        assert started["started"] is True and started["render_id"]
        # Planar with no blockMeshDict: aimed along the normal by the script, and named so.
        assert started["outputs"][0].endswith("animation-U-plane.mp4")

        listed = await foam_client.call(Method.RENDER_LIST)
        assert [entry["id"] for entry in listed["renders"]] == [started["render_id"]]

        for _ in range(200):
            if any(note.event == str(Event.RENDER_FINISHED) for note in events):
                break
            await asyncio.sleep(0.05)
        done = next(note.data for note in events if note.event == str(Event.RENDER_FINISHED))
        assert done["ok"] is True, done
        assert done["steps"] == 2, "render, then encode"
        assert any(note.event == str(Event.RENDER_PROGRESS) for note in events)
        assert (await foam_client.call(Method.RENDER_LIST))["renders"] == []
    finally:
        await watcher.close()


async def test_a_background_render_can_be_cancelled(
    foam_client: DaemonClient, tmp_path: Path, monkeypatch
) -> None:
    fake_tool(tmp_path / "bin", "pvbatch", "exec sleep 30")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    case = foam_with_results(tmp_path / "wing")
    started = await foam_client.call(Method.CASE_RENDER, path=str(case), kind="mesh")

    await foam_client.call(Method.RENDER_CANCEL, id=started["render_id"][:8])
    for _ in range(100):
        if not (await foam_client.call(Method.RENDER_LIST))["renders"]:
            break
        await asyncio.sleep(0.05)
    assert (await foam_client.call(Method.RENDER_LIST))["renders"] == []
    with pytest.raises(RemoteError, match="No running render"):
        await foam_client.call(Method.RENDER_CANCEL, id="nonsense")


async def test_an_unknown_field_is_refused_before_anything_runs(
    foam_client: DaemonClient, tmp_path: Path, monkeypatch
) -> None:
    fake_tool(tmp_path / "bin", "pvbatch", "exit 0")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    case = foam_with_results(tmp_path / "wing")
    with pytest.raises(RemoteError, match="velocity"):
        await foam_client.call(Method.CASE_RENDER, path=str(case), kind="mesh", field="velocity")


# == switching what a core means, from the interface (§4.3.2) =================================


@pytest.fixture
def smt_config(dispatch_config: Config, monkeypatch) -> Config:
    """A 16-core, 24-thread machine with no explicit core count, so the mode matters."""
    monkeypatch.setattr("dispatch.core.config.physical_cores", lambda: 16)
    monkeypatch.setattr("dispatch.core.config.logical_cpus", lambda: 24)
    return replace(dispatch_config, scheduler=replace(dispatch_config.scheduler, total_cores=None))


@pytest.fixture
async def smt_daemon(smt_config: Config, registry) -> AsyncIterator[Daemon]:
    instance = Daemon(smt_config, load_plugins=False)
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
async def smt_client(smt_daemon: Daemon) -> AsyncIterator[DaemonClient]:
    connection = DaemonClient(smt_daemon.config.paths.socket, autostart=False)
    await connection.connect()
    try:
        yield connection
    finally:
        await connection.close()


async def test_the_preview_says_what_a_switch_would_do_and_does_nothing(
    smt_daemon: Daemon, smt_client: DaemonClient
) -> None:
    preview = await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="toggle", preview=True)
    assert (preview["previous"], preview["mode"]) == ("physical", "logical")
    assert (preview["previous_total"], preview["total_cores"]) == (16, 24)
    assert preview["applied"] is False and preview["relabel_only"] is False
    assert smt_daemon.resources.total_cores == 16
    assert smt_daemon.repo.get_setting("scheduler.cpu_mode") is None


async def test_switching_changes_the_snapshot_the_dashboard_draws(
    smt_daemon: Daemon, smt_client: DaemonClient
) -> None:
    result = await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="logical")
    assert result["applied"] is True
    snapshot = await smt_client.call(Method.SYSTEM_SNAPSHOT)
    assert snapshot["cpu_mode"] == "logical"
    assert snapshot["cpu_mode_source"] == "interface"
    assert snapshot["total_cores"] == 24


async def test_the_choice_survives_a_daemon_restart(
    smt_config: Config, smt_daemon: Daemon, smt_client: DaemonClient
) -> None:
    await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="logical")
    restarted = Daemon(smt_config, load_plugins=False)
    try:
        assert restarted.resources.cpu_mode.value == "logical"
        assert restarted.resources.cpu_mode_source == "interface"
        assert restarted.resources.total_cores == 24
    finally:
        restarted.conn.close()


async def test_choosing_the_config_s_own_mode_clears_the_override(
    smt_daemon: Daemon, smt_client: DaemonClient
) -> None:
    await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="logical")
    await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="toggle")
    assert smt_daemon.repo.get_setting("scheduler.cpu_mode") is None
    assert smt_daemon.resources.cpu_mode_source == "config"
    assert smt_daemon.resources.total_cores == 16


async def test_an_unreadable_stored_mode_does_not_stop_the_daemon(
    smt_config: Config, smt_daemon: Daemon
) -> None:
    smt_daemon.repo.set_setting("scheduler.cpu_mode", "quantum")
    restarted = Daemon(smt_config, load_plugins=False)
    try:
        assert restarted.resources.cpu_mode.value == "physical"
    finally:
        restarted.conn.close()


async def test_the_preview_names_jobs_a_switch_would_strand(
    smt_daemon: Daemon, smt_client: DaemonClient, tmp_path: Path
) -> None:
    """A queued job asking for more than the new total would never start."""
    await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="logical")
    # Twelve threads busy, so the twenty-thread job waits rather than starting.
    busy = await submit(smt_client, make_case(tmp_path / "busy", script="sleep 30"), cores=12)
    await await_state(smt_client, busy["id"], JobState.RUNNING.value)
    job = await submit(smt_client, make_case(tmp_path / "wide", script="exit 0"), cores=20)

    preview = await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="physical", preview=True)
    assert [entry["id"] for entry in preview["stranded"]] == [job["id"]]
    for each in (busy, job):
        await smt_client.call(Method.JOB_CANCEL, id=each["id"], force=True)


async def test_a_running_job_over_the_new_total_is_reported_over_committed(
    smt_daemon: Daemon, smt_client: DaemonClient, tmp_path: Path
) -> None:
    await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="logical")
    job = await submit(smt_client, make_case(tmp_path / "big", script="sleep 30"), cores=20)
    await await_state(smt_client, job["id"], JobState.RUNNING.value)

    preview = await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="physical", preview=True)
    assert preview["over_committed"] is True
    await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="physical")
    assert smt_daemon.resources.allocated_cores == 20, "the running job keeps its cores"
    await smt_client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_more_capacity_admits_waiting_work_at_once(
    smt_daemon: Daemon, smt_client: DaemonClient, tmp_path: Path
) -> None:
    """No wait for the next heartbeat: the heartbeat in these tests is an hour."""
    first = await submit(smt_client, make_case(tmp_path / "a", script="sleep 30"), cores=12)
    await await_state(smt_client, first["id"], JobState.RUNNING.value)
    second = await submit(smt_client, make_case(tmp_path / "b", script="sleep 30"), cores=8)
    await asyncio.sleep(0.3)
    assert (await smt_client.call(Method.JOB_GET, id=second["id"]))["job"]["state"] == "QUEUED"

    await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="logical")
    await await_state(smt_client, second["id"], JobState.RUNNING.value)
    for job in (first, second):
        await smt_client.call(Method.JOB_CANCEL, id=job["id"], force=True)


async def test_an_unknown_mode_is_refused_with_the_valid_ones(smt_client: DaemonClient) -> None:
    with pytest.raises(RemoteError, match="physical, logical"):
        await smt_client.call(Method.SCHEDULER_CPU_MODE, mode="hyper")
