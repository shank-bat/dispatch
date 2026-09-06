"""The whole daemon, over a real socket.

A real ``IpcServer`` bound to a real Unix socket in ``tmp_path``, a real
:class:`~dispatch.ipc.client.DaemonClient`, and real subprocesses behind the fake adapter.
This is the test that would catch a protocol change breaking the TUI.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dispatch.core.config import Config
from dispatch.core.states import JobState
from dispatch.daemon.main import Daemon
from dispatch.ipc.client import DaemonClient, DaemonUnavailable, RemoteError
from dispatch.ipc.protocol import PROTOCOL_VERSION, Event, Method, Topic
from tests.conftest_daemon import make_case


@pytest.fixture
async def daemon(dispatch_config: Config, registry) -> AsyncIterator[Daemon]:
    """A fully wired daemon serving on a temporary socket."""
    instance = Daemon(dispatch_config, load_plugins=False)
    # Swap in the fake adapter so tests need no solver installed.
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
    """Poll a job until it reaches ``state``, then return its detail payload."""
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


# -- handshake ------------------------------------------------------------------------


async def test_hello_reports_the_daemon(client: DaemonClient) -> None:
    assert client.server_info["protocol"] == PROTOCOL_VERSION
    assert "fake" in client.server_info["adapters"]


async def test_a_protocol_mismatch_says_to_restart_the_daemon(daemon: Daemon) -> None:
    """The realistic skew: you upgraded and forgot to restart the daemon."""
    other = DaemonClient(daemon.config.paths.socket, autostart=False)
    await other._open()
    try:
        with pytest.raises(RemoteError) as excinfo:
            await other.call(Method.HELLO, protocol=PROTOCOL_VERSION + 99, client="test")
        assert "restart the daemon" in str(excinfo.value).lower()
    finally:
        await other.close()


async def test_requests_before_hello_are_refused(daemon: Daemon) -> None:
    other = DaemonClient(daemon.config.paths.socket, autostart=False)
    await other._open()
    try:
        with pytest.raises(RemoteError, match="hello"):
            await other.call(Method.JOB_LIST)
    finally:
        await other.close()


async def test_an_unknown_method_lists_the_known_ones(client: DaemonClient) -> None:
    with pytest.raises(RemoteError) as excinfo:
        await client.call("job.teleport")
    assert "job.submit" in str(excinfo.value)


# -- the submit-to-completion path -------------------------------------------------------


async def test_a_job_runs_end_to_end(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "e2e", script="echo hello; exit 0")
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
    job_id = result["job"]["id"]

    detail = await await_state(client, job_id, "COMPLETED")
    assert detail["job"]["exit_code"] == 0
    assert detail["events"]


async def test_submission_returns_the_queue_position(client: DaemonClient, tmp_path: Path) -> None:
    for index in range(3):
        make_case(tmp_path / f"case{index}", script="sleep 5")
    first = await client.call(
        Method.JOB_SUBMIT, workdir=str(tmp_path / "case0"), cores=8, name="first"
    )
    second = await client.call(
        Method.JOB_SUBMIT, workdir=str(tmp_path / "case1"), cores=8, name="second"
    )
    assert first["job"]["queue_position"] in (None, 1)
    assert second["job"]["queue_position"] is not None


async def test_a_job_id_prefix_is_enough(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "prefix", script="sleep 5")
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
    job_id = result["job"]["id"]
    detail = await client.call(Method.JOB_GET, id=job_id[:8])
    assert detail["job"]["id"] == job_id


# -- validation and force ------------------------------------------------------------------


async def test_a_failing_case_is_refused(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "bad", invalid=True)
    with pytest.raises(RemoteError) as excinfo:
        await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
    assert excinfo.value.code == "VALIDATION_FAILED"


async def test_force_overrides_validation(client: DaemonClient, tmp_path: Path) -> None:
    """The user sometimes knows things the validator does not."""
    case = make_case(tmp_path / "forced", invalid=True, script="exit 0")
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1, force=True)
    assert result["job"]["state"] in ("QUEUED", "PREPARING", "RUNNING")


async def test_warnings_do_not_block(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "warned", warn=True, script="exit 0")
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
    assert result["validation"]["passed"]


async def test_an_unrecognised_directory_is_reported(client: DaemonClient, tmp_path: Path) -> None:
    plain = tmp_path / "not-a-case"
    plain.mkdir()
    with pytest.raises(RemoteError) as excinfo:
        await client.call(Method.JOB_SUBMIT, workdir=str(plain), cores=1)
    assert "No solver recognised" in str(excinfo.value)


async def test_a_job_larger_than_the_machine_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Better an error at submission than a job that waits forever."""
    case = make_case(tmp_path / "toobig")
    with pytest.raises(RemoteError, match="never get"):
        await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=9999)


# -- dry run ----------------------------------------------------------------------------------


async def test_dry_run_reports_the_plan_and_changes_nothing(
    client: DaemonClient, tmp_path: Path
) -> None:
    case = make_case(tmp_path / "dry", prepare="true", script="exit 0")
    report = await client.call(Method.CASE_DRYRUN, workdir=str(case), cores=2)

    assert report["would_submit"]
    kinds = [step["kind"] for step in report["plan"]["steps"]]
    assert kinds == ["PREPARE", "SOLVE"]
    assert report["projection"]["would_start_immediately"]
    assert "fake" in report["suggested_tags"]

    page = await client.call(Method.JOB_LIST)
    assert page["total"] == 0, "a dry run created a job"


async def test_dry_run_shows_why_a_job_would_wait(client: DaemonClient, tmp_path: Path) -> None:
    blocker = make_case(tmp_path / "blocker", script="sleep 10")
    await client.call(Method.JOB_SUBMIT, workdir=str(blocker), cores=8)
    await asyncio.sleep(0.5)

    case = make_case(tmp_path / "waiter")
    report = await client.call(Method.CASE_DRYRUN, workdir=str(case), cores=8)
    assert not report["projection"]["would_start_immediately"]
    assert "cores free" in report["projection"]["blocking_reason"]


# -- queue operations ----------------------------------------------------------------------------


async def test_hold_and_release(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "held", script="sleep 10")
    blocker = make_case(tmp_path / "blocker", script="sleep 10")
    await client.call(Method.JOB_SUBMIT, workdir=str(blocker), cores=8)
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=8)
    job_id = result["job"]["id"]

    held = await client.call(Method.JOB_HOLD, id=job_id)
    assert held["state"] == "HELD"
    released = await client.call(Method.JOB_RELEASE, id=job_id)
    assert released["state"] == "QUEUED"


async def test_priority_changes_the_order(client: DaemonClient, tmp_path: Path) -> None:
    blocker = make_case(tmp_path / "blocker", script="sleep 10")
    await client.call(Method.JOB_SUBMIT, workdir=str(blocker), cores=8)
    first = await client.call(
        Method.JOB_SUBMIT, workdir=str(make_case(tmp_path / "a")), cores=8, name="a"
    )
    second = await client.call(
        Method.JOB_SUBMIT, workdir=str(make_case(tmp_path / "b")), cores=8, name="b"
    )

    await client.call(Method.JOB_PRIORITY, id=second["job"]["id"], priority=10)
    page = await client.call(Method.JOB_LIST, states=["QUEUED"])
    positions = {job["name"]: job["queue_position"] for job in page["items"]}
    assert positions["b"] < positions["a"]
    assert first["job"]["id"]


async def test_cancelling_a_queued_job(client: DaemonClient, tmp_path: Path) -> None:
    blocker = make_case(tmp_path / "blocker", script="sleep 10")
    await client.call(Method.JOB_SUBMIT, workdir=str(blocker), cores=8)
    result = await client.call(
        Method.JOB_SUBMIT, workdir=str(make_case(tmp_path / "doomed")), cores=8
    )
    job_id = result["job"]["id"]

    outcome = await client.call(Method.JOB_CANCEL, id=job_id)
    assert not outcome["cancelling"]
    detail = await client.call(Method.JOB_GET, id=job_id)
    assert detail["job"]["state"] == "CANCELLED"


async def test_cancelling_a_finished_job_is_an_error(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "quick", script="exit 0")
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
    job_id = result["job"]["id"]
    await await_state(client, job_id, "COMPLETED")

    with pytest.raises(RemoteError, match="already finished"):
        await client.call(Method.JOB_CANCEL, id=job_id)


async def test_a_job_can_be_asked_to_run_after_another(
    client: DaemonClient, tmp_path: Path
) -> None:
    """The whole feature, over the real socket, on a machine with cores to spare.

    Eight cores, two one-core jobs: without the dependency the second would start beside
    the first immediately.
    """
    first = await client.call(
        Method.JOB_SUBMIT,
        workdir=str(make_case(tmp_path / "first", script="sleep 1")),
        cores=1,
    )
    first_id = first["job"]["id"]
    await await_state(client, first_id, "RUNNING")

    second = await client.call(
        Method.JOB_SUBMIT,
        workdir=str(make_case(tmp_path / "second", script="exit 0")),
        cores=1,
        depends_on_job_id=first_id,
    )
    second_id = second["job"]["id"]
    assert second["job"]["depends_on_job_id"] == first_id

    detail = await client.call(Method.JOB_GET, id=second_id)
    assert detail["job"]["state"] == "QUEUED"
    assert "waiting for" in (detail["waiting_because"] or "")
    snapshot = await client.call(Method.SYSTEM_SNAPSHOT)
    assert snapshot["free_cores"] >= 1, "the cores were there; only the dependency held it back"

    await await_state(client, first_id, "COMPLETED")
    await await_state(client, second_id, "COMPLETED")


async def test_submitting_after_an_unknown_job_is_refused(
    client: DaemonClient, tmp_path: Path
) -> None:
    """Better than queueing a job whose condition can never be met."""
    with pytest.raises(RemoteError):
        await client.call(
            Method.JOB_SUBMIT,
            workdir=str(make_case(tmp_path / "orphan")),
            cores=1,
            depends_on_job_id="no-such-job",
        )


async def test_a_job_without_a_dependency_is_unaffected(
    client: DaemonClient, tmp_path: Path
) -> None:
    """The default path: nothing to wait for, so it starts on submission as it always has."""
    result = await client.call(
        Method.JOB_SUBMIT, workdir=str(make_case(tmp_path / "plain")), cores=1
    )
    assert result["job"]["depends_on_job_id"] is None
    await await_state(client, result["job"]["id"], "COMPLETED")


async def test_the_queue_explains_why_a_job_waits(client: DaemonClient, tmp_path: Path) -> None:
    blocker = make_case(tmp_path / "blocker", script="sleep 10")
    await client.call(Method.JOB_SUBMIT, workdir=str(blocker), cores=8)
    await asyncio.sleep(0.4)
    result = await client.call(
        Method.JOB_SUBMIT, workdir=str(make_case(tmp_path / "waiting")), cores=8
    )
    detail = await client.call(Method.JOB_GET, id=result["job"]["id"])
    assert detail["waiting_because"]


# -- tags, notes, search ---------------------------------------------------------------------------


async def test_tags_and_notes_round_trip(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "tagged", script="exit 0")
    result = await client.call(
        Method.JOB_SUBMIT, workdir=str(case), cores=1, tags=["paper", "NACA 0018"]
    )
    job_id = result["job"]["id"]
    assert sorted(result["job"]["tags"]) == ["naca-0018", "paper"]

    await client.call(Method.JOB_NOTE, id=job_id, body="the run behind figure 4")
    updated = await client.call(Method.JOB_TAG, id=job_id, add=["final"], remove=["paper"])
    assert "final" in updated["tags"] and "paper" not in updated["tags"]

    found = await client.call(Method.HISTORY_SEARCH, query="tag:final")
    assert [job["id"] for job in found["items"]] == [job_id]

    by_note = await client.call(Method.HISTORY_SEARCH, query="figure")
    assert [job["id"] for job in by_note["items"]] == [job_id]


async def test_search_reports_a_bad_query_clearly(client: DaemonClient) -> None:
    with pytest.raises(RemoteError) as excinfo:
        await client.call(Method.HISTORY_SEARCH, query="taag:paper")
    assert excinfo.value.code == "QUERY_INVALID"


async def test_tags_list_counts_usage(client: DaemonClient, tmp_path: Path) -> None:
    for index in range(2):
        case = make_case(tmp_path / f"t{index}", script="exit 0")
        await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1, tags=["shared"])
    tags = await client.call(Method.TAGS_LIST)
    assert {"name": "shared", "count": 2} in tags


# -- filesystem browsing ---------------------------------------------------------------------------


async def test_fs_list_marks_recognised_cases(client: DaemonClient, tmp_path: Path) -> None:
    """The marking comes from the daemon, so the TUI stays solver-ignorant."""
    root = tmp_path / "projects"
    root.mkdir()
    make_case(root / "a-case", script="exit 0")
    (root / "not-a-case").mkdir()

    listing = await client.call(Method.FS_LIST, path=str(root))
    marks = {entry["name"]: entry["case"] for entry in listing["entries"]}
    assert marks["a-case"] == "fake"
    assert marks["not-a-case"] is None
    assert listing["parent"] == str(root.parent)


async def test_fs_list_rejects_a_file(client: DaemonClient, tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("hello")
    with pytest.raises(RemoteError, match="not a directory"):
        await client.call(Method.FS_LIST, path=str(target))


# -- events ----------------------------------------------------------------------------------------


async def test_state_changes_are_pushed_to_subscribers(daemon: Daemon, tmp_path: Path) -> None:
    """The mechanism behind the TUI never polling."""
    received: list[str] = []
    connection = DaemonClient(
        daemon.config.paths.socket,
        autostart=False,
        on_event=lambda note: received.append(note.event),
    )
    await connection.connect()
    try:
        await connection.subscribe([str(Topic.JOBS), str(Topic.QUEUE)])
        case = make_case(tmp_path / "watched", script="exit 0")
        await connection.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)

        for _ in range(400):
            if str(Event.JOB_STATE) in received:
                break
            await asyncio.sleep(0.05)
        assert str(Event.JOB_STATE) in received
    finally:
        await connection.close()


async def test_sampling_only_runs_while_somebody_is_watching(daemon: Daemon) -> None:
    """No subscribers means the machine is not measured at all."""
    assert not daemon.monitor.sampling

    connection = DaemonClient(daemon.config.paths.socket, autostart=False)
    await connection.connect()
    await connection.subscribe([str(Topic.SYSTEM)])
    await asyncio.sleep(0.1)
    sampling_while_watched = daemon.monitor.sampling

    await connection.close()
    await asyncio.sleep(0.2)
    sampling_after = daemon.monitor.sampling

    assert sampling_while_watched
    assert not sampling_after


async def test_a_stalled_client_is_dropped_not_tolerated(daemon: Daemon) -> None:
    """Unbounded queues are how long-lived daemons die."""
    subscription = daemon.bus.subscribe(maxsize=4)
    daemon.bus.set_topics(subscription, [str(Topic.JOBS)])

    for _ in range(50):
        daemon.bus.publish(Event.JOB_STATE, {"id": "x"})

    assert subscription.dropped > 0
    assert subscription.queue.qsize() <= 4
    events = [subscription.queue.get_nowait().event for _ in range(subscription.queue.qsize())]
    assert str(Event.RESYNC) in events


# -- daemon information ----------------------------------------------------------------------------


async def test_daemon_info_describes_itself(client: DaemonClient) -> None:
    info = await client.call(Method.DAEMON_INFO)
    assert info["protocol"] == PROTOCOL_VERSION
    assert info["policy"] == "backfill"
    assert any(spec["adapter"] == "fake" for spec in info["adapters"])
    assert set(info["counts"]) == {state.value for state in JobState}


async def test_the_system_snapshot_separates_ledger_from_measurement(
    client: DaemonClient,
) -> None:
    """Two different notions of busy, both shown, never conflated."""
    snapshot = await client.call(Method.SYSTEM_SNAPSHOT)
    assert "allocated_cores" in snapshot
    assert "cpu_percent" in snapshot
    assert snapshot["free_cores"] <= snapshot["total_cores"]


async def test_provenance_is_recorded_for_a_run_job(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "prov", script="exit 0")
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
    job_id = result["job"]["id"]
    await await_state(client, job_id, "COMPLETED")

    provenance = await client.call(Method.JOB_PROVENANCE, id=job_id)
    assert provenance is not None
    assert provenance["dispatch_version"]
    assert provenance["hostname"]
    assert provenance["solver_version"] == "fake 1.0"
    assert provenance["argv"] == ["/bin/sh", "fake.job"]
    assert provenance["env_hash"]


async def test_deleting_a_finished_job(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "gone", script="exit 0")
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
    job_id = result["job"]["id"]
    await await_state(client, job_id, "COMPLETED")

    await client.call(Method.JOB_DELETE, id=job_id)
    page = await client.call(Method.JOB_LIST)
    assert job_id not in [job["id"] for job in page["items"]]


async def test_a_running_job_cannot_be_deleted(client: DaemonClient, tmp_path: Path) -> None:
    case = make_case(tmp_path / "busy", script="sleep 10")
    result = await client.call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
    with pytest.raises(RemoteError):
        await client.call(Method.JOB_DELETE, id=result["job"]["id"])


# -- autostart ------------------------------------------------------------------------

async def test_autostart_forwards_the_config_file(tmp_path: Path) -> None:
    """A client started with --config must spawn a daemon using the same config.

    Without this the autostarted daemon reads the default configuration, binds a different
    socket, and the client waits forever on one that will never appear.
    """
    config_file = tmp_path / "conf.toml"
    config_file.write_text("[scheduler]\ntotal_cores = 2\n")

    client = DaemonClient(
        tmp_path / "run" / "daemon.sock", autostart=True, config_path=config_file
    )

    recorded: list[list[str]] = []

    def fake_run(argv, **kwargs):
        recorded.append(list(argv))
        raise OSError("not actually starting a daemon")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("subprocess.run", fake_run)
    try:
        with pytest.raises(DaemonUnavailable):
            await client.connect()
    finally:
        monkeypatch.undo()

    assert recorded, "no daemon spawn was attempted"
    argv = recorded[0]
    assert "--config" in argv
    assert str(config_file) in argv
    assert "--detach" in argv


# -- concurrency -----------------------------------------------------------------------------------


async def test_requests_are_pipelined(client: DaemonClient, tmp_path: Path) -> None:
    """A slow search must not delay a cancel queued behind it."""
    results = await asyncio.gather(
        client.call(Method.SYSTEM_SNAPSHOT),
        client.call(Method.JOB_LIST),
        client.call(Method.DAEMON_INFO),
        client.call(Method.TAGS_LIST),
    )
    assert len(results) == 4
    assert results[2]["protocol"] == PROTOCOL_VERSION


async def test_several_clients_share_one_daemon(daemon: Daemon, tmp_path: Path) -> None:
    clients = [DaemonClient(daemon.config.paths.socket, autostart=False) for _ in range(3)]
    for connection in clients:
        await connection.connect()
    try:
        case = make_case(tmp_path / "shared", script="exit 0")
        await clients[0].call(Method.JOB_SUBMIT, workdir=str(case), cores=1)
        pages = await asyncio.gather(*(c.call(Method.JOB_LIST) for c in clients))
        assert {page["total"] for page in pages} == {1}
    finally:
        for connection in clients:
            await connection.close()


# -- responses larger than a default stream buffer ------------------------------------------
#
# asyncio's StreamReader defaults to a 64 KiB buffer, and `readuntil` treats a line that
# outruns it as the stream being unusable. Left implicit, that gives the protocol two
# ceilings 64x apart -- the 4 MiB it documents and enforces, and the 64 KiB that actually
# fires -- and a reply merely larger than 64 KiB ends the connection. The daemon logs
# nothing, because the daemon did nothing, so it reads as "the daemon is unreachable".
#
# It is a threshold bug: it appears only once the machine has enough history for one reply
# to cross 64 KiB, which is roughly forty-odd jobs.


def test_the_stream_buffer_matches_the_enforced_message_limit() -> None:
    """One ceiling, not two. The buffer is what silently truncates; the check is what reports."""
    from dispatch.ipc.protocol import MAX_MESSAGE_BYTES
    from dispatch.ipc.socketpath import STREAM_LIMIT

    assert STREAM_LIMIT == MAX_MESSAGE_BYTES


async def test_a_reply_larger_than_64k_does_not_drop_the_connection(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """The exact failure: a big `job.list` looked like the daemon hanging up.

    Enough jobs are queued for the reply to cross the old 64 KiB buffer, then the whole
    page is fetched in one call -- which is what the interface does on every connect.
    """
    for index in range(60):
        await client.call(
            Method.JOB_SUBMIT,
            workdir=str(make_case(tmp_path / f"case_{index:03d}", script="exit 0")),
            cores=1,
        )

    page = await client.call(Method.JOB_LIST, limit=500)

    assert len(page["items"]) >= 60
    assert client.connected, "a large reply must not be reported as a disconnection"
    # Still usable afterwards: the read loop is the thing that used to die.
    assert await client.call(Method.SYSTEM_SNAPSHOT)


async def test_the_interface_caches_a_full_queue_on_connect(
    client: DaemonClient, daemon: Daemon, tmp_path: Path
) -> None:
    """What the user actually saw: the TUI's own startup call, at its own page size."""
    for index in range(60):
        await client.call(
            Method.JOB_SUBMIT,
            workdir=str(make_case(tmp_path / f"job_{index:03d}", script="exit 0")),
            cores=1,
        )

    snapshot = await client.call(Method.SYSTEM_SNAPSHOT)
    page = await client.call(Method.JOB_LIST, limit=500)
    sweeps = await client.call(Method.SWEEP_LIST)

    assert snapshot and page["items"] and sweeps is not None
    assert client.connected
