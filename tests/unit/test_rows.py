"""Row mapping, and its deliberate tolerance for data it did not write.

These parsers read rows written by *past* versions of Dispatch. A job that cannot be
loaded because one column holds something unexpected is a hole in the history, appearing
at exactly the moment the history is being consulted. So the rule is: degrade the field,
never refuse the row.

The tests below therefore write deliberately malformed values straight into SQLite,
bypassing the repository, and assert that the job still loads.
"""

from __future__ import annotations

import sqlite3

from dispatch.core.clock import FakeClock
from dispatch.core.provenance import GitInfo, Provenance
from dispatch.core.states import ExitReason, JobState
from dispatch.db.repository import JobRepository


def test_unrecognised_exit_reason_degrades_to_none(
    repo: JobRepository, conn: sqlite3.Connection, make_spec
) -> None:
    """A reason written by a future version must not make the job unreadable."""
    job = repo.create(make_spec())
    conn.execute("UPDATE jobs SET exit_reason = 'quantum_tunnelled' WHERE id = ?", (job.id,))
    loaded = repo.get(job.id)
    assert loaded.exit_reason is None
    assert loaded.name == job.name


def test_the_database_refuses_to_store_invalid_json(
    repo: JobRepository, conn: sqlite3.Connection, make_spec
) -> None:
    """The CHECK constraint is the first line of defence, and it holds.

    Which means the tolerant parser below can only ever fire on genuine file corruption,
    not on anything Dispatch itself could write.
    """
    import pytest

    job = repo.create(make_spec())
    with pytest.raises(sqlite3.IntegrityError, match="json_valid"):
        conn.execute("UPDATE jobs SET metadata = '{not json' WHERE id = ?", (job.id,))


def test_the_parser_still_degrades_gracefully_on_unparseable_json() -> None:
    """Second line of defence, for a row that arrived some other way."""
    from dispatch.db.rows import _json_array, _json_object

    assert _json_object("{not json") == {}
    assert _json_object(None) == {}
    assert _json_object("[1,2]") == {}  # valid JSON, wrong shape
    assert _json_array("nope") == []
    assert _json_array('{"a": 1}') == []


def test_metadata_that_is_valid_json_but_the_wrong_shape(
    repo: JobRepository, conn: sqlite3.Connection, make_spec
) -> None:
    job = repo.create(make_spec())
    conn.execute("UPDATE jobs SET metadata = '[1, 2, 3]' WHERE id = ?", (job.id,))
    assert repo.get(job.id).metadata.case == {}


def test_null_log_paths_load_as_none(repo: JobRepository, make_spec) -> None:
    """A job recorded before its log paths were assigned is still a valid job."""
    job = repo.create(make_spec())
    assert job.stdout_path is None and job.stderr_path is None


def test_log_paths_round_trip(repo: JobRepository, make_spec, tmp_path) -> None:
    out = tmp_path / "stdout.log"
    err = tmp_path / "stderr.log"
    job = repo.create(make_spec(), stdout_path=out, stderr_path=err)
    loaded = repo.get(job.id)
    assert loaded.stdout_path == out
    assert loaded.stderr_path == err


# -- provenance persistence ----------------------------------------------------------------


def test_provenance_round_trips(repo: JobRepository, make_spec, clock: FakeClock) -> None:
    job = repo.create(make_spec())
    original = Provenance(
        captured_at=clock.now(),
        dispatch_version="0.1.0",
        python_version="3.13.14",
        hostname="eddy",
        kernel_version="6.16.3",
        os_release="Debian GNU/Linux 13",
        cpu_model="AMD Ryzen 9 7950X",
        total_ram_mb=32000,
        solver_name="openfoam",
        solver_version="OpenFOAM-v2312",
        adapter_version=3,
        git=GitInfo(commit="a" * 40, branch="main", dirty=True, remote="git@example:me/case.git"),
        env_snapshot={"WM_PROJECT_DIR": "/opt/openfoam2312", "MPI_ARCH_PATH": "/usr"},
        env_hash="deadbeef",
        argv=("mpirun", "-np", "20", "interFoam", "-parallel"),
    )
    repo.save_provenance(job.id, original)
    restored = repo.provenance(job.id)

    assert restored is not None
    assert restored.solver_version == "OpenFOAM-v2312"
    assert restored.git.dirty is True
    assert restored.git.remote == "git@example:me/case.git"
    assert restored.env_snapshot == dict(original.env_snapshot)
    assert restored.argv == ("mpirun", "-np", "20", "interFoam", "-parallel")


def test_provenance_is_absent_for_a_job_that_never_started(
    repo: JobRepository, make_spec
) -> None:
    assert repo.provenance(repo.create(make_spec()).id) is None


def test_saving_provenance_twice_replaces_it(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    """Re-running recovery must not fail on a job whose record already exists."""
    job = repo.create(make_spec())
    base = Provenance(
        captured_at=clock.now(),
        dispatch_version="0.1.0",
        python_version="3.13.14",
        hostname="eddy",
        kernel_version="6.16.3",
    )
    repo.save_provenance(job.id, base)
    repo.save_provenance(job.id, base)
    restored = repo.provenance(job.id)
    assert restored is not None and restored.hostname == "eddy"


def test_provenance_is_deleted_with_its_job(
    repo: JobRepository, conn: sqlite3.Connection, make_spec, clock: FakeClock
) -> None:
    job = repo.create(make_spec())
    repo.save_provenance(
        job.id,
        Provenance(
            captured_at=clock.now(),
            dispatch_version="0.1.0",
            python_version="3.13.14",
            hostname="eddy",
            kernel_version="6.16.3",
        ),
    )
    repo.transition(job.id, JobState.CANCELLED)
    repo.delete(job.id)
    remaining = conn.execute(
        "SELECT COUNT(*) FROM job_provenance WHERE job_id = ?", (job.id,)
    ).fetchone()[0]
    assert remaining == 0


# -- samples and events -----------------------------------------------------------------------


def test_samples_are_returned_oldest_first(repo: JobRepository, make_spec) -> None:
    from dispatch.core.models import Sample

    job = repo.create(make_spec())
    for ts in (300.0, 100.0, 200.0):
        repo.add_sample(job.id, Sample(ts=ts, rss_mb=int(ts), cpu_pct=50.0))
    assert [s.ts for s in repo.samples(job.id)] == [100.0, 200.0, 300.0]


def test_samples_are_limited_to_the_most_recent(repo: JobRepository, make_spec) -> None:
    """A week-long job accumulates thousands; the caller asked for a window."""
    from dispatch.core.models import Sample

    job = repo.create(make_spec())
    for ts in range(10):
        repo.add_sample(job.id, Sample(ts=float(ts), rss_mb=1, cpu_pct=1.0))
    assert [s.ts for s in repo.samples(job.id, limit=3)] == [7.0, 8.0, 9.0]


def test_duplicate_sample_timestamps_replace_rather_than_raise(
    repo: JobRepository, make_spec
) -> None:
    from dispatch.core.models import Sample

    job = repo.create(make_spec())
    repo.add_sample(job.id, Sample(ts=1.0, rss_mb=100, cpu_pct=10.0))
    repo.add_sample(job.id, Sample(ts=1.0, rss_mb=200, cpu_pct=20.0))
    assert [s.rss_mb for s in repo.samples(job.id)] == [200]


def test_events_accumulate_in_order(repo: JobRepository, make_spec) -> None:
    job = repo.create(make_spec())
    repo.add_event(job.id, "warn", "could not read solver version")
    repo.mark_preparing(job.id)
    kinds = [e.kind for e in repo.events(job.id)]
    assert kinds == ["state", "warn", "state"]


def test_notes_can_be_added_after_a_job_finishes(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    """Which is usually when you learn a run mattered."""
    job = repo.create(make_spec())
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    repo.mark_finished(job.id, state=JobState.COMPLETED, exit_code=0, reason=ExitReason.OK)
    clock.advance(86400 * 30)
    note = repo.add_note(job.id, "this is the run behind figure 4")
    assert note.ts == clock.now()
    assert len(repo.notes(job.id)) == 1


def test_an_empty_note_is_rejected(repo: JobRepository, make_spec) -> None:
    import pytest

    from dispatch.core.errors import ValidationError

    job = repo.create(make_spec())
    with pytest.raises(ValidationError, match="cannot be empty"):
        repo.add_note(job.id, "   ")


def test_integrity_check_passes_on_a_healthy_database(repo: JobRepository) -> None:
    assert repo.integrity_check() == "ok"
