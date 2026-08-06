"""The repository: persistence, transitions, tags, and derived indexes.

These run against a real (in-memory) SQLite, not a mock. The behaviour under test is
largely *SQL* behaviour -- conditional updates, cascades, partial indexes -- and a mock
would assert that the code calls the queries it calls, which proves nothing.
"""

from __future__ import annotations

import sqlite3

import pytest

from dispatch.core.clock import FakeClock
from dispatch.core.errors import IllegalTransition, JobNotFound, ValidationError
from dispatch.core.models import JobSpec, ResourceRequest
from dispatch.core.states import ExitReason, JobState
from dispatch.db.repository import MAX_LIMIT, JobRepository

# -- creation ---------------------------------------------------------------------------


def test_create_returns_a_queued_job_with_defaults(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec("cavity", cores=8))
    assert job.state is JobState.QUEUED
    assert job.name == "cavity"
    assert job.cores == 8
    assert job.seq == 1
    assert job.started_at is None
    assert job.exit_code is None


def test_seq_increments_and_is_the_fifo_tiebreak(repo: JobRepository, make_spec) -> None:
    first = repo.create(make_spec("a"))
    second = repo.create(make_spec("b"))
    assert second.seq > first.seq


def test_job_ids_are_unique(repo: JobRepository, make_spec) -> None:
    ids = {repo.create(make_spec(f"case{i}")).id for i in range(20)}
    assert len(ids) == 20


def test_name_defaults_to_the_directory_name(repo: JobRepository, tmp_path) -> None:
    workdir = tmp_path / "foamacoustic"
    workdir.mkdir()
    spec = JobSpec(workdir=workdir, solver="openfoam", resources=ResourceRequest(cores=1))
    assert repo.create(spec).name == "foamacoustic"


def test_create_records_an_event(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    assert [e.kind for e in repo.events(job.id)] == ["state"]
    assert "created in QUEUED" in repo.events(job.id)[0].detail


def test_a_job_cannot_be_created_already_running(repo: JobRepository, make_spec) -> None:
    with pytest.raises(ValidationError):
        repo.create(make_spec(), state=JobState.RUNNING)


def test_submission_note_is_stored(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec(note="first attempt at Re=100000"))
    assert [n.body for n in repo.notes(job.id)] == ["first attempt at Re=100000"]


# -- reads ------------------------------------------------------------------------------


def test_get_missing_job_raises(repo: JobRepository) -> None:
    with pytest.raises(JobNotFound):
        repo.get("nope")


def test_get_optional_returns_none(repo: JobRepository) -> None:
    assert repo.get_optional("nope") is None


def test_resolve_id_expands_a_unique_prefix(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    assert repo.resolve_id(job.id[:8]) == job.id


def test_resolve_id_rejects_an_ambiguous_prefix(
    repo: JobRepository, monkeypatch, make_spec
) -> None:
    """Two ids sharing a prefix must be an error, never a guess."""
    ids = iter(["abc11111-0000-0000-0000-000000000000", "abc22222-0000-0000-0000-000000000000"])
    monkeypatch.setattr("dispatch.db.repository.new_job_id", lambda: next(ids))
    repo.create(make_spec("a"))
    repo.create(make_spec("b"))
    with pytest.raises(ValidationError, match="ambiguous"):
        repo.resolve_id("abc")


def test_resolve_id_missing_raises_not_found(repo: JobRepository) -> None:
    with pytest.raises(JobNotFound):
        repo.resolve_id("zzzz")


def test_queued_is_ordered_by_priority_then_submission(repo: JobRepository, make_spec) -> None:
    low = repo.create(make_spec("low", priority=0))
    high = repo.create(make_spec("high", priority=10))
    also_low = repo.create(make_spec("also_low", priority=0))
    assert [j.id for j in repo.queued()] == [high.id, low.id, also_low.id]


def test_queue_positions_are_derived_and_one_based(repo: JobRepository, make_spec) -> None:
    first = repo.create(make_spec("a"))
    second = repo.create(make_spec("b"))
    assert repo.queue_positions() == {first.id: 1, second.id: 2}


def test_queue_positions_renumber_themselves_when_a_job_leaves(
    repo: JobRepository, make_spec
) -> None:
    """The point of deriving them: nothing has to be rewritten."""
    first = repo.create(make_spec("a"))
    second = repo.create(make_spec("b"))
    repo.mark_preparing(first.id)
    assert repo.queue_positions() == {second.id: 1}


def test_held_jobs_are_not_in_the_queue(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    repo.transition(job.id, JobState.HELD)
    assert repo.queued() == ()
    assert repo.queue_positions() == {}


def test_active_returns_preparing_and_running(repo: JobRepository, make_spec) -> None:
    preparing = repo.create(make_spec("p"))
    running = repo.create(make_spec("r"))
    queued = repo.create(make_spec("q"))
    repo.mark_preparing(preparing.id)
    repo.mark_preparing(running.id)
    repo.mark_started(running.id, pid=1, pid_start_time=1.0)
    assert {j.id for j in repo.active()} == {preparing.id, running.id}
    assert queued.id not in {j.id for j in repo.active()}


def test_count_by_state_covers_every_state(repo: JobRepository, make_spec) -> None:
    repo.create(make_spec())
    counts = repo.count_by_state()
    assert set(counts) == set(JobState)
    assert counts[JobState.QUEUED] == 1
    assert counts[JobState.FAILED] == 0


def test_list_jobs_pages(repo: JobRepository, make_spec) -> None:
    for i in range(5):
        repo.create(make_spec(f"case{i}"))
    page = repo.list_jobs(limit=2, offset=0)
    assert page.total == 5
    assert page.limit == 2
    assert page.has_more


def test_list_jobs_last_page_has_no_more(repo: JobRepository, make_spec) -> None:
    for i in range(3):
        repo.create(make_spec(f"case{i}"))
    assert not repo.list_jobs(limit=2, offset=2).has_more


def test_list_jobs_filtered_by_state(repo: JobRepository, make_spec) -> None:
    running = repo.create(make_spec("r"))
    repo.create(make_spec("q"))
    repo.mark_preparing(running.id)
    page = repo.list_jobs(states=[JobState.PREPARING])
    assert [j.id for j in page.items] == [running.id]


def test_list_jobs_with_no_states_matches_nothing(repo: JobRepository, make_spec) -> None:
    """An empty filter means "none of them", not "all of them"."""
    repo.create(make_spec())
    assert repo.list_jobs(states=[]).total == 0


def test_page_size_is_clamped(repo: JobRepository, make_spec) -> None:
    """A caller asking for a million rows must not be able to spike daemon memory."""
    repo.create(make_spec())
    assert repo.list_jobs(limit=10_000_000).limit <= MAX_LIMIT


def test_recent_returns_finished_jobs_newest_first(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    first = _run_to_completion(repo, make_spec("first"), clock)
    clock.advance(100)
    second = _run_to_completion(repo, make_spec("second"), clock)
    assert [j.id for j in repo.recent()] == [second.id, first.id]


# -- transitions --------------------------------------------------------------------------


def test_illegal_transition_is_refused(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    with pytest.raises(IllegalTransition):
        repo.transition(job.id, JobState.COMPLETED)


def test_illegal_transition_leaves_the_job_untouched(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    with pytest.raises(IllegalTransition):
        repo.transition(job.id, JobState.RUNNING)
    assert repo.get(job.id).state is JobState.QUEUED


def test_transition_on_a_missing_job_raises(repo: JobRepository) -> None:
    with pytest.raises(JobNotFound):
        repo.transition("nope", JobState.CANCELLED)


def test_transition_rejects_unknown_columns(repo: JobRepository, make_spec) -> None:
    """A typo'd keyword must be an error, not a silently discarded update."""
    job = repo.create(make_spec())
    with pytest.raises(ValidationError, match="Cannot set"):
        repo.transition(job.id, JobState.HELD, hostname="eddy")


def test_transition_is_atomic_against_a_concurrent_change(
    repo: JobRepository, conn: sqlite3.Connection, make_spec
) -> None:
    """The conditional UPDATE is what makes the state machine a guarantee, not a diagram.

    The transition reads the current state, checks the table, then writes with
    ``WHERE state = <what I read>``. Here the state is changed behind the repository's
    back in between -- exactly what a racing writer would do -- and the write must match
    zero rows and raise rather than silently overwriting the other change.

    The whole transition runs in one transaction, so the injected change is rolled back
    along with it. That is the correct outcome and the thing worth asserting: the job did
    not move to PREPARING, and no allocation was made against a job somebody else claimed.
    """
    job = repo.create(make_spec())
    real = repo.connection
    proxy = _RacingConnection(real, job_id=job.id)
    repo._conn = proxy  # type: ignore[assignment]
    try:
        with pytest.raises(IllegalTransition):
            repo.transition(job.id, JobState.PREPARING)
    finally:
        repo._conn = real

    assert proxy.injected, "the race was never simulated; the test proved nothing"
    assert repo.get(job.id).state is not JobState.PREPARING


def test_entering_preparing_stamps_started_at(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    job = repo.create(make_spec())
    clock.advance(50)
    job = repo.mark_preparing(job.id)
    assert job.started_at == clock.now()


def test_started_at_covers_preparation_not_just_the_solver(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    """Decomposing a large mesh can take twenty minutes; the runtime must include it."""
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    clock.advance(1200)  # decomposePar
    repo.mark_started(job.id, pid=42, pid_start_time=1.0)
    clock.advance(3600)  # the solve
    job = repo.mark_finished(job.id, state=JobState.COMPLETED, exit_code=0, reason=ExitReason.OK)
    assert job.metrics.runtime_s == 4800


def test_mark_started_records_pid_and_start_time(repo: JobRepository, make_spec) -> None:
    """The pair, not the pid alone: a pid is not a stable identity across a restart."""
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    job = repo.mark_started(job.id, pid=4242, pid_start_time=123.5)
    assert (job.pid, job.pid_start_time) == (4242, 123.5)


def test_runtime_is_recorded_on_any_terminal_path(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    """Including a plain transition that bypasses the mark_finished helper."""
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    clock.advance(90)
    job = repo.transition(job.id, JobState.CANCELLED)
    assert job.metrics.runtime_s == 90
    assert job.finished_at == clock.now()


def test_a_job_cancelled_from_the_queue_has_no_runtime(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    """It never started, so a runtime of 0 would be a fabrication."""
    job = repo.create(make_spec())
    clock.advance(500)
    job = repo.transition(job.id, JobState.CANCELLED)
    assert job.metrics.runtime_s is None
    assert job.finished_at is not None


def test_mark_finished_accepts_an_explicit_finish_time(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    """Recovery needs this: the sentinel file's mtime is when the job really ended."""
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    started = repo.get(job.id).started_at
    assert started is not None
    clock.advance(10_000)  # the daemon was down for hours
    job = repo.mark_finished(
        job.id, state=JobState.COMPLETED, exit_code=0, finished_at=started + 60
    )
    assert job.finished_at == started + 60
    assert job.metrics.runtime_s == 60


def test_mark_finished_rejects_a_non_terminal_state(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    with pytest.raises(ValidationError, match="not a terminal state"):
        repo.mark_finished(job.id, state=JobState.RUNNING)


def test_exit_reason_round_trips(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    job = repo.mark_finished(
        job.id, state=JobState.FAILED, exit_code=137, reason=ExitReason.OOM, signal_name="KILL"
    )
    assert job.exit_reason is ExitReason.OOM
    assert job.exit_signal == "KILL"
    assert job.exit_code == 137


def test_unknown_state_is_reachable_for_a_lost_job(repo: JobRepository, make_spec) -> None:
    """The honest outcome when the machine was reset mid-run."""
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    job = repo.mark_finished(job.id, state=JobState.UNKNOWN, reason=ExitReason.LOST)
    assert job.state is JobState.UNKNOWN
    assert job.exit_code is None


def test_set_priority_reorders_the_queue(repo: JobRepository, make_spec) -> None:
    first = repo.create(make_spec("a"))
    second = repo.create(make_spec("b"))
    repo.set_priority(second.id, 5)
    assert [j.id for j in repo.queued()] == [second.id, first.id]


def test_set_priority_on_a_missing_job_raises(repo: JobRepository) -> None:
    with pytest.raises(JobNotFound):
        repo.set_priority("nope", 1)


# -- metrics ---------------------------------------------------------------------------------


def test_peak_rss_only_increases(repo: JobRepository, make_spec) -> None:
    """A late-arriving low sample must not erase a peak that really happened."""
    job = repo.create(make_spec())
    repo.update_metrics(job.id, peak_rss_mb=4096)
    repo.update_metrics(job.id, peak_rss_mb=1024)
    assert repo.get(job.id).metrics.peak_rss_mb == 4096


def test_mean_cpu_is_replaced_not_maximised(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    repo.update_metrics(job.id, mean_cpu_pct=90.0)
    repo.update_metrics(job.id, mean_cpu_pct=50.0)
    assert repo.get(job.id).metrics.mean_cpu_pct == 50.0


def test_update_metrics_with_nothing_is_a_no_op(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    repo.update_metrics(job.id)
    assert repo.get(job.id).metrics.is_empty


# -- deletion ----------------------------------------------------------------------------------


def test_a_running_job_cannot_be_deleted(repo: JobRepository, make_spec) -> None:
    """Deleting the row would orphan a live simulation the daemon can no longer find."""
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    with pytest.raises(ValidationError, match="cannot be deleted"):
        repo.delete(job.id)


def test_a_queued_job_cannot_be_deleted(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    with pytest.raises(ValidationError, match="Cancel it first"):
        repo.delete(job.id)


def test_delete_removes_attached_rows(
    repo: JobRepository, conn: sqlite3.Connection, make_spec, clock: FakeClock
) -> None:
    job = _run_to_completion(repo, make_spec(tags={"paper"}, note="hello"), clock)
    repo.add_sample(job.id, _sample(clock.now()))
    repo.delete(job.id)

    for table in ("jobs", "notes", "job_events", "job_tags", "job_samples", "jobs_fts"):
        column = "job_id" if table != "jobs" else "id"
        row = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (job.id,)
        ).fetchone()
        count = row[0]
        assert count == 0, f"{table} still holds rows for the deleted job"


def test_delete_leaves_the_tag_vocabulary_alone(repo: JobRepository, make_spec, clock) -> None:
    """The tag itself survives; only the association goes."""
    keeper = repo.create(make_spec("keeper", tags={"paper"}))
    doomed = _run_to_completion(repo, make_spec("doomed", tags={"paper"}), clock)
    repo.delete(doomed.id)
    assert repo.all_tags() == (("paper", 1),)
    assert "paper" in repo.get(keeper.id).tags


# -- helpers ---------------------------------------------------------------------------------------


class _RacingConnection:
    """Wraps a connection and changes a job's state just before the conditional UPDATE.

    ``sqlite3.Connection.execute`` is read-only and cannot be monkeypatched, so the
    connection itself is proxied. Everything else delegates untouched.
    """

    def __init__(self, real: sqlite3.Connection, *, job_id: str) -> None:
        self._real = real
        self._job_id = job_id
        self.injected = False

    def execute(self, sql: str, *args, **kwargs):
        if not self.injected and sql.strip().startswith("UPDATE jobs SET state"):
            self.injected = True
            self._real.execute("UPDATE jobs SET state = 'CANCELLED' WHERE id = ?", (self._job_id,))
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def _run_to_completion(repo: JobRepository, spec: JobSpec, clock: FakeClock):
    job = repo.create(spec)
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    clock.advance(60)
    return repo.mark_finished(job.id, state=JobState.COMPLETED, exit_code=0, reason=ExitReason.OK)


def _sample(ts: float):
    from dispatch.core.models import Sample

    return Sample(ts=ts, rss_mb=100, cpu_pct=50.0)
