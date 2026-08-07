"""Database connection setup, capability checks, and migrations.

Three responsibilities, all of which must happen before any query runs:

1. Apply the PRAGMAs that make SQLite behave the way this design assumes (WAL, foreign
   keys, a busy timeout).
2. Verify the SQLite build actually has the features the schema depends on. FTS5 and JSON1
   are compile-time options; discovering their absence at the first search -- months in --
   would be far worse than refusing to start.
3. Run migrations forward, in a transaction, refusing to touch a database written by a
   newer Dispatch.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Final

from dispatch.core.errors import DatabaseError, MigrationError

__all__ = ["SCHEMA_VERSION", "connect", "migrate", "transaction"]

log = logging.getLogger(__name__)

SCHEMA_VERSION: Final = 2
"""The schema version this code expects. Bump alongside a new migration file."""

_MIGRATIONS_PACKAGE: Final = "dispatch.db.migrations"


def connect(
    path: Path | str, *, read_only: bool = False, timeout: float = 5.0
) -> sqlite3.Connection:
    """Open a Dispatch database, applying PRAGMAs and verifying capabilities.

    Args:
        path: Database file. Parent directories are created. Pass ``":memory:"`` for tests.
        read_only: Open without write access. Used by nothing in the daemon -- it is the
            sole writer -- but useful for inspection tools.
        timeout: Seconds to wait on a locked database before raising.

    Returns:
        A configured connection with :class:`sqlite3.Row` row factory.

    Raises:
        DatabaseError: If the file cannot be opened or the SQLite build lacks FTS5 or JSON1.
    """
    is_memory = str(path) == ":memory:"
    if not is_memory:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        uri = f"file:{target}?mode=ro" if read_only else str(target)
    else:
        uri = ":memory:"

    try:
        conn = sqlite3.connect(
            uri,
            timeout=timeout,
            uri=not is_memory and read_only,
            isolation_level=None,  # explicit transactions; see `transaction()`
            check_same_thread=False,
        )
    except sqlite3.Error as exc:
        raise DatabaseError(f"Cannot open database {path}: {exc}") from exc

    conn.row_factory = sqlite3.Row
    try:
        _apply_pragmas(conn, read_only=read_only, timeout=timeout)
        _require_capabilities(conn)
    except BaseException:
        # Refusing to start must not also leak the handle: the caller has no reference to
        # close, and on a daemon that retries this would accumulate.
        conn.close()
        raise
    return conn


def _apply_pragmas(conn: sqlite3.Connection, *, read_only: bool, timeout: float) -> None:
    """Configure the connection. See docs/ARCHITECTURE.md §5 for the rationale."""
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    # Memory-mapped I/O would speed up large reads but inflates RSS, which is directly
    # opposed to leaving RAM free for simulations (§13.6). The database is small.
    conn.execute("PRAGMA mmap_size = 0")
    if not read_only:
        # WAL lets readers proceed without blocking the writer, which is what allows a
        # long-running search to coexist with a job transition.
        conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL fsyncs at checkpoints rather than every commit. Safe under WAL: a power
        # loss can lose the last few transactions but cannot corrupt the database.
        conn.execute("PRAGMA synchronous = NORMAL")


def missing_capabilities(conn: sqlite3.Connection | None = None) -> list[str]:
    """Return the SQLite features this build lacks, if any.

    Exposed publicly so ``dispatch doctor`` can report the same answer without writing its
    own SQL -- the probe belongs here with the rest of the database knowledge.

    Args:
        conn: Connection to test. A temporary in-memory one is used when omitted.
    """
    owned = conn is None
    connection = conn if conn is not None else sqlite3.connect(":memory:")
    missing: list[str] = []
    try:
        try:
            connection.execute("CREATE VIRTUAL TABLE temp.__fts_probe USING fts5(x)")
            connection.execute("DROP TABLE temp.__fts_probe")
        except sqlite3.Error:
            missing.append("FTS5 (full-text search)")

        try:
            connection.execute("SELECT json_valid('{}')")
        except sqlite3.Error:
            missing.append("JSON1 (json_valid, json_extract)")
    finally:
        if owned:
            connection.close()
    return missing


def _require_capabilities(conn: sqlite3.Connection) -> None:
    """Fail loudly, now, if this SQLite build cannot support the schema."""
    missing = missing_capabilities(conn)
    if missing:
        raise DatabaseError(
            "This SQLite build is missing "
            + " and ".join(missing)
            + f" (SQLite {sqlite3.sqlite_version}). Dispatch needs both. Install a Python "
            "built against a full SQLite, or rebuild SQLite with "
            "-DSQLITE_ENABLE_FTS5 -DSQLITE_ENABLE_JSON1.",
            detail={"missing": missing, "sqlite_version": sqlite3.sqlite_version},
        )


@contextmanager
def transaction(
    conn: sqlite3.Connection, *, immediate: bool = True
) -> Iterator[sqlite3.Connection]:
    """Run a block inside a transaction, committing on success and rolling back on error.

    Args:
        conn: The connection.
        immediate: Take the write lock up front with ``BEGIN IMMEDIATE`` rather than
            escalating on first write. This turns lock contention into an immediate,
            clearly-attributable error instead of a failure partway through a multi-table
            write -- which matters because job creation touches five tables.

    Nested use is a no-op wrapper: SQLite has no true nesting, and savepoints would add
    complexity for a codebase where the daemon is the only writer.
    """
    if conn.in_transaction:
        yield conn
        return

    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the database up to :data:`SCHEMA_VERSION`.

    Migrations are numbered ``.sql`` files applied in order inside one transaction each,
    tracked with ``PRAGMA user_version``. Forward only: a rollback path would need to be
    written and tested for a case that has never occurred on a single-user workstation,
    and the honest recovery for a bad migration is a backup.

    Returns:
        The version the database is now at.

    Raises:
        MigrationError: If the database is newer than this code understands, or a
            migration fails.
    """
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])

    if current > SCHEMA_VERSION:
        raise MigrationError(
            f"This database is at schema version {current}, but this Dispatch understands "
            f"only up to {SCHEMA_VERSION}. It was written by a newer version. Upgrade "
            "Dispatch rather than running this one -- continuing could corrupt data.",
            detail={"database_version": current, "code_version": SCHEMA_VERSION},
        )

    if current == SCHEMA_VERSION:
        return current

    for version, sql in _pending(current):
        log.info("Applying database migration %03d", version)
        # The BEGIN/COMMIT go inside the script rather than around it: executescript()
        # implicitly commits any transaction that is already open, so an outer `with
        # transaction(conn)` would be silently discarded. `PRAGMA user_version` is
        # transactional, so bumping it inside the same script makes "schema applied" and
        # "version recorded" atomic -- a migration cannot half-land.
        script = f"BEGIN;\n{sql}\nPRAGMA user_version = {version};\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error as exc:
            conn.rollback()
            raise MigrationError(
                f"Migration {version:03d} failed: {exc}", detail={"version": version}
            ) from exc

    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _pending(current: int) -> Iterator[tuple[int, str]]:
    """Yield ``(version, sql)`` for every migration above ``current``, in order."""
    root = resources.files(_MIGRATIONS_PACKAGE)
    files = sorted(
        (entry for entry in root.iterdir() if entry.name.endswith(".sql")),
        key=lambda entry: entry.name,
    )
    for entry in files:
        try:
            version = int(entry.name.split("_", 1)[0])
        except ValueError as exc:
            raise MigrationError(
                f"Migration file {entry.name!r} does not start with a version number"
            ) from exc
        if version > current:
            yield version, entry.read_text(encoding="utf-8")
