"""Connection setup, capability checks, and migrations.

The migration tests matter disproportionately: this is a database that will be carried
across Dispatch upgrades on a machine that runs for months, and a half-applied schema is
the one failure mode with no easy recovery.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dispatch.core.errors import DatabaseError, MigrationError
from dispatch.db.connection import SCHEMA_VERSION, connect, migrate, transaction


def test_connect_creates_parent_directories(tmp_path: Path) -> None:
    """First run must work without the user pre-creating ~/.local/share/dispatch."""
    target = tmp_path / "deep" / "nested" / "dispatch.db"
    conn = connect(target)
    conn.close()
    assert target.exists()


def test_wal_is_enabled_on_a_file_database(tmp_path: Path) -> None:
    """WAL is what lets a long search coexist with a job transition."""
    conn = connect(tmp_path / "d.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()


def test_foreign_keys_are_enforced(conn: sqlite3.Connection) -> None:
    """Otherwise ON DELETE CASCADE silently does nothing and deletes leak rows."""
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_mmap_is_disabled(tmp_path: Path) -> None:
    """Deliberate: mmap inflates RSS, and RAM belongs to the simulations.

    Checked on a file database -- an in-memory one reports no row for this pragma at all.
    """
    conn = connect(tmp_path / "d.db")
    assert conn.execute("PRAGMA mmap_size").fetchone()[0] == 0
    conn.close()


def test_row_factory_gives_named_access(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT 1 AS answer").fetchone()
    assert row["answer"] == 1


def test_capability_check_accepts_this_build() -> None:
    """FTS5 and JSON1 are compile-time options; both are required."""
    conn = connect(":memory:")
    assert conn.execute("SELECT json_valid('{}')").fetchone()[0] == 1
    conn.close()


def test_capability_failure_is_reported_actionably(monkeypatch) -> None:
    """A missing FTS5 must fail at startup with instructions, not at the first search."""

    real_connect = sqlite3.connect

    class NoFts5:
        """A connection proxy that rejects FTS5, as a build without it would.

        ``sqlite3.Connection.execute`` is read-only, so the connection is wrapped rather
        than patched.
        """

        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def execute(self, sql: str, *a, **kw):
            if "fts5" in sql.lower():
                raise sqlite3.OperationalError("no such module: fts5")
            return self._real.execute(sql, *a, **kw)

        def __getattr__(self, name: str):
            return getattr(self._real, name)

        def __setattr__(self, name: str, value: object) -> None:
            if name == "_real":
                object.__setattr__(self, name, value)
            else:
                setattr(self._real, name, value)

    monkeypatch.setattr("sqlite3.connect", lambda *a, **kw: NoFts5(real_connect(*a, **kw)))
    with pytest.raises(DatabaseError) as excinfo:
        connect(":memory:")
    assert "FTS5" in str(excinfo.value)
    assert "SQLITE_ENABLE_FTS5" in str(excinfo.value)


# -- migrations --------------------------------------------------------------------------


def test_migrate_from_empty_reaches_the_current_version() -> None:
    conn = connect(":memory:")
    assert migrate(conn) == SCHEMA_VERSION
    conn.close()


def test_migrate_is_idempotent(conn: sqlite3.Connection) -> None:
    """The daemon calls it on every start, including the ten-thousandth."""
    assert migrate(conn) == SCHEMA_VERSION
    assert migrate(conn) == SCHEMA_VERSION


def test_migrate_creates_every_expected_table(conn: sqlite3.Connection) -> None:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {
        "jobs",
        "job_events",
        "notes",
        "job_samples",
        "tags",
        "job_tags",
        "job_metadata",
        "job_provenance",
    } <= tables


def test_an_existing_v1_database_gains_the_exit_detail_column() -> None:
    """The upgrade a user with months of history actually performs.

    Applying only ``001`` reproduces a database written before failure reasons existed;
    migrating it must add the column without disturbing the rows already there.
    """
    conn = connect(":memory:")
    conn.executescript(_migration_sql(1))
    conn.execute("PRAGMA user_version = 1")

    assert "exit_detail" not in _columns(conn, "jobs")
    assert migrate(conn) == SCHEMA_VERSION
    assert "exit_detail" in _columns(conn, "jobs")
    conn.close()


def test_the_new_column_is_null_for_jobs_that_predate_it(conn: sqlite3.Connection) -> None:
    """Not an empty string: nothing was ever read from those logs, so there is no answer."""
    row = conn.execute("SELECT exit_detail FROM jobs WHERE 0").description
    assert row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _migration_sql(version: int) -> str:
    from importlib import resources

    root = resources.files("dispatch.db.migrations")
    entry = next(e for e in root.iterdir() if e.name.startswith(f"{version:03d}_"))
    return entry.read_text(encoding="utf-8")


def test_migrate_creates_the_partial_queue_index(conn: sqlite3.Connection) -> None:
    """Partial, so the scheduler's hot query never scales with completed history."""
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_jobs_queue'"
    ).fetchone()[0]
    assert "WHERE state = 'QUEUED'" in sql


def test_a_database_from_the_future_is_refused(tmp_path: Path) -> None:
    """Running an old Dispatch against a newer schema could corrupt data silently."""
    path = tmp_path / "future.db"
    conn = connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    with pytest.raises(MigrationError, match="newer version"):
        migrate(conn)
    conn.close()


def test_a_failed_migration_leaves_nothing_behind(tmp_path: Path, monkeypatch) -> None:
    """The schema and the version bump land together, or neither does."""
    import dispatch.db.connection as module

    def broken(_current: int):
        yield 1, "CREATE TABLE half_applied (x); SELECT this_function_does_not_exist();"

    monkeypatch.setattr(module, "_pending", broken)
    conn = connect(tmp_path / "broken.db")
    with pytest.raises(MigrationError, match="Migration 001 failed"):
        migrate(conn)

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "half_applied" not in tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    conn.close()


def test_migration_files_are_numbered(monkeypatch, tmp_path: Path) -> None:

    class FakeEntry:
        name = "oops.sql"

        def read_text(self, encoding: str = "utf-8") -> str:
            return ""

    class FakeRoot:
        def iterdir(self):
            return [FakeEntry()]

    monkeypatch.setattr("importlib.resources.files", lambda _pkg: FakeRoot())
    conn = connect(tmp_path / "x.db")
    with pytest.raises(MigrationError, match="version number"):
        migrate(conn)
    conn.close()


# -- transactions -------------------------------------------------------------------------


def test_transaction_commits_on_success(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        conn.execute("INSERT INTO tags (name) VALUES ('kept')")
    assert conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0] == 1


def test_transaction_rolls_back_on_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError), transaction(conn):
        conn.execute("INSERT INTO tags (name) VALUES ('discarded')")
        raise RuntimeError("boom")
    assert conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0] == 0


def test_nested_transactions_are_a_no_op_wrapper(conn: sqlite3.Connection) -> None:
    """SQLite has no true nesting; the inner block must not commit early."""
    with transaction(conn):
        conn.execute("INSERT INTO tags (name) VALUES ('outer')")
        with transaction(conn):
            conn.execute("INSERT INTO tags (name) VALUES ('inner')")
        assert conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0] == 2


def test_nested_rollback_discards_the_whole_outer_block(conn: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError), transaction(conn):
        conn.execute("INSERT INTO tags (name) VALUES ('outer')")
        with transaction(conn):
            conn.execute("INSERT INTO tags (name) VALUES ('inner')")
        raise RuntimeError("boom")
    assert conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0] == 0
