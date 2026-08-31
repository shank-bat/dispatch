"""Schema evolution.

The promise is narrow and absolute: upgrading Dispatch must not make a single existing job
unreadable, and must not require deleting the database. So these tests build a database at
an *old* schema version, put real rows in it, migrate forward, and read them back through
the same repository the daemon uses.

Building the old schema means running the migration files themselves, up to a version --
not a hand-written copy of what the schema used to be, which would drift.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dispatch.core.clock import FakeClock
from dispatch.core.errors import MigrationError
from dispatch.core.models import ResourceKind
from dispatch.core.states import JobState
from dispatch.db.connection import SCHEMA_VERSION, connect, migrate
from dispatch.db.connection import _pending as pending_migrations
from dispatch.db.repository import JobRepository


def at_version(path: Path | str, version: int) -> sqlite3.Connection:
    """A database migrated only as far as ``version``, as an older Dispatch left it."""
    conn = connect(path)
    for number, sql in pending_migrations(0):
        if number > version:
            break
        conn.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version = {number};\nCOMMIT;")
    return conn


def legacy_job(conn: sqlite3.Connection, job_id: str, *, name: str, cores: int) -> None:
    """Insert a job the way a pre-upgrade Dispatch would have, naming no new column."""
    conn.execute(
        """
        INSERT INTO jobs (id, seq, name, workdir, solver, cores, state, created_at,
                          stdout_path, stderr_path, metadata)
        VALUES (?, ?, ?, ?, 'openfoam', ?, 'COMPLETED', 1000.0, ?, ?, '{}')
        """,
        (
            job_id,
            abs(hash(job_id)) % 10_000,
            name,
            f"/home/shu/projects/{name}",
            cores,
            f"/home/shu/.local/share/dispatch/logs/jobs/{job_id}/stdout.log",
            f"/home/shu/.local/share/dispatch/logs/jobs/{job_id}/stderr.log",
        ),
    )
    conn.commit()


# -- the runner --------------------------------------------------------------------------


def test_a_fresh_database_reaches_the_current_version(tmp_path: Path) -> None:
    conn = connect(tmp_path / "fresh.db")
    assert migrate(conn) == SCHEMA_VERSION
    conn.close()


def test_migrating_twice_is_a_no_op(tmp_path: Path) -> None:
    conn = connect(tmp_path / "twice.db")
    migrate(conn)
    assert migrate(conn) == SCHEMA_VERSION
    conn.close()


def test_a_database_from_the_future_is_refused_rather_than_corrupted(tmp_path: Path) -> None:
    conn = connect(tmp_path / "future.db")
    migrate(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    with pytest.raises(MigrationError, match="newer version"):
        migrate(conn)
    conn.close()


def test_every_migration_file_is_numbered_in_order() -> None:
    numbers = [number for number, _ in pending_migrations(0)]
    assert numbers == sorted(numbers)
    assert numbers[-1] == SCHEMA_VERSION


# -- upgrading a populated database ----------------------------------------------------------


def test_existing_jobs_survive_the_resource_migration(tmp_path: Path) -> None:
    """The whole point: a user upgrades and their history is still there."""
    path = tmp_path / "history.db"
    conn = at_version(path, SCHEMA_VERSION - 1)
    legacy_job(conn, "11111111-1111-1111-1111-111111111111", name="cavity", cores=20)
    legacy_job(conn, "22222222-2222-2222-2222-222222222222", name="wing", cores=8)

    assert migrate(conn) == SCHEMA_VERSION

    repo = JobRepository(conn, clock=FakeClock())
    jobs = {job.name: job for job in repo.list_jobs(limit=10).items}
    assert set(jobs) == {"cavity", "wing"}
    assert jobs["cavity"].cores == 20
    conn.close()


def test_migrated_jobs_are_cpu_jobs_holding_no_gpus(tmp_path: Path) -> None:
    """Which is what they were. Defaults must not invent a resource claim."""
    conn = at_version(tmp_path / "upgrade.db", SCHEMA_VERSION - 1)
    legacy_job(conn, "33333333-3333-3333-3333-333333333333", name="cavity", cores=4)
    migrate(conn)

    job = JobRepository(conn, clock=FakeClock()).list_jobs(limit=1).items[0]
    assert job.resource_kind is ResourceKind.CPU
    assert job.gpus == 0
    assert job.state is JobState.COMPLETED
    conn.close()


def test_migrated_jobs_keep_their_original_log_paths(tmp_path: Path) -> None:
    """No ``log_path``, because their output never was in the case directory."""
    job_id = "44444444-4444-4444-4444-444444444444"
    conn = at_version(tmp_path / "logs.db", SCHEMA_VERSION - 1)
    legacy_job(conn, job_id, name="cavity", cores=4)
    migrate(conn)

    job = JobRepository(conn, clock=FakeClock()).get(job_id)
    assert job.log_path is None
    assert job.stdout_path is not None
    assert job.stdout_path.name == "stdout.log"
    assert job.output_path == job.stdout_path, "history stays readable via the old field"
    conn.close()


def test_an_upgraded_database_accepts_a_gpu_job(tmp_path: Path) -> None:
    """Old rows and new rows coexist in one table."""
    from dispatch.core.models import JobSpec, ResourceRequest

    conn = at_version(tmp_path / "mixed.db", SCHEMA_VERSION - 1)
    legacy_job(conn, "55555555-5555-5555-5555-555555555555", name="cavity", cores=4)
    migrate(conn)

    repo = JobRepository(conn, clock=FakeClock())
    workdir = tmp_path / "pinn"
    workdir.mkdir()
    repo.create(
        JobSpec(
            workdir=workdir,
            solver="pinn",
            resources=ResourceRequest(cores=1, gpus=1, kind=ResourceKind.GPU),
            name="pinn",
        )
    )
    kinds = {job.name: job.resource_kind for job in repo.list_jobs(limit=10).items}
    assert kinds == {"cavity": ResourceKind.CPU, "pinn": ResourceKind.GPU}
    conn.close()


def test_the_database_enforces_the_cpu_gpu_agreement(tmp_path: Path) -> None:
    """The invariant is in the schema as well as the model, so no writer can break it."""
    conn = connect(tmp_path / "check.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO jobs (id, seq, name, workdir, solver, cores, state, created_at,
                              metadata, resource_kind, gpus)
            VALUES ('x', 1, 'x', '/tmp', 'ml', 1, 'QUEUED', 1.0, '{}', 'cpu', 2)
            """
        )
    conn.close()


def test_a_row_with_an_unknown_resource_kind_still_loads() -> None:
    """Reading is tolerant: a row written by a future version must render, not raise.

    The database's own CHECK stops such a row being written *here*, which is the right
    behaviour; the case this covers is a database written by a later Dispatch and then
    opened by this one, where the constraint has already been satisfied by a schema this
    code has never seen. So the row is built directly and handed to the mapper.
    """
    from dispatch.db.rows import job_from_row

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    columns = [
        "id", "seq", "name", "workdir", "solver", "solver_binary", "cores",
        "ram_estimate_mb", "priority", "state", "created_at", "started_at", "finished_at",
        "exit_code", "exit_reason", "exit_signal", "exit_detail", "stdout_path",
        "stderr_path", "log_path", "pid", "pid_start_time", "metadata", "runtime_s",
        "peak_rss_mb", "mean_cpu_pct", "depends_on_job_id", "resource_kind", "gpus",
    ]
    conn.execute(f"CREATE TABLE loose ({', '.join(columns)})")
    values = dict.fromkeys(columns)
    values.update(
        id="y", seq=1, name="y", workdir="/tmp", solver="tpu", cores=1,
        priority=0, state="COMPLETED", created_at=1.0, metadata="{}",
        resource_kind="quantum", gpus=1,
    )
    conn.execute(
        f"INSERT INTO loose VALUES ({', '.join(':' + name for name in columns)})", values
    )
    row = conn.execute("SELECT * FROM loose").fetchone()

    job = job_from_row(row)
    assert job.resource_kind is ResourceKind.GPU, "falls back to what the gpu count implies"
    assert job.gpus == 1
    conn.close()


def test_a_row_with_nonsensical_resources_still_loads() -> None:
    """Never let one odd column make a job unopenable; that is the failure to avoid."""
    from dispatch.db.rows import job_from_row

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    columns = [
        "id", "seq", "name", "workdir", "solver", "solver_binary", "cores",
        "ram_estimate_mb", "priority", "state", "created_at", "started_at", "finished_at",
        "exit_code", "exit_reason", "exit_signal", "exit_detail", "stdout_path",
        "stderr_path", "log_path", "pid", "pid_start_time", "metadata", "runtime_s",
        "peak_rss_mb", "mean_cpu_pct", "depends_on_job_id", "resource_kind", "gpus",
    ]
    conn.execute(f"CREATE TABLE loose ({', '.join(columns)})")
    values = dict.fromkeys(columns)
    values.update(
        id="z", seq=1, name="z", workdir="/tmp", solver="ml", cores=4,
        priority=0, state="COMPLETED", created_at=1.0, metadata="{}",
        resource_kind="cpu", gpus=3,  # contradictory: a CPU job holding GPUs
    )
    conn.execute(
        f"INSERT INTO loose VALUES ({', '.join(':' + name for name in columns)})", values
    )
    job = job_from_row(conn.execute("SELECT * FROM loose").fetchone())

    assert job.cores == 4
    assert job.resource_kind is ResourceKind.CPU
    assert job.gpus == 0, "read as the CPU job it most likely was"
    conn.close()
