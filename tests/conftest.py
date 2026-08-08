"""Shared fixtures.

Every fixture here is in-memory or under ``tmp_path``: the suite must never touch the
user's real database, log directory, or socket.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from dispatch.adapters.registry import AdapterRegistry
from dispatch.core.clock import FakeClock
from dispatch.core.config import Config, DaemonConfig, PathsConfig, SchedulerConfig
from dispatch.core.metadata import FieldType, MetadataField, MetadataSpec, SpecRef
from dispatch.core.models import JobSpec, ResourceRequest
from dispatch.daemon.resources import ResourceModel
from dispatch.db.connection import connect, migrate
from dispatch.db.repository import JobRepository
from tests.conftest_daemon import FakeAdapter, make_case

__all__ = ["FakeAdapter", "make_case"]


@pytest.fixture
def clock() -> FakeClock:
    """A manually advanced clock, so time-dependent behaviour is deterministic."""
    return FakeClock()


@pytest.fixture
def dispatch_config(tmp_path: Path) -> Config:
    """A configuration confined entirely to ``tmp_path``.

    Every daemon test uses this, so nothing can touch the developer's real database,
    logs, or socket -- including when a test fails halfway through.
    """
    return Config(
        paths=PathsConfig(
            database=tmp_path / "state" / "dispatch.db",
            log_dir=tmp_path / "logs",
            runtime_dir=tmp_path / "run",
        ),
        scheduler=SchedulerConfig(total_cores=8, reserved_cores=0, heartbeat_s=3600.0),
        daemon=DaemonConfig(autostart=False, cancel_grace_s=0.2),
    )


@pytest.fixture
def registry() -> AdapterRegistry:
    """A registry holding only the fake adapter."""
    built = AdapterRegistry({})
    built.register(FakeAdapter)
    return built


@pytest.fixture
def resources(dispatch_config: Config) -> ResourceModel:
    """A ledger over eight cores, with memory reported as plentiful.

    The memory probe is injected so scheduling behaviour does not depend on how much RAM
    the machine running the tests happens to have free.
    """
    return ResourceModel(
        dispatch_config.scheduler,
        memory_probe=lambda: 64_000,
        log_dir=dispatch_config.paths.log_dir,
    )


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """A migrated in-memory database."""
    connection = connect(":memory:")
    migrate(connection)
    yield connection
    connection.close()


@pytest.fixture
def foam_spec() -> MetadataSpec:
    """A stand-in for the OpenFOAM adapter's metadata spec.

    Defined here rather than imported so that the database tests do not depend on the
    adapter package, which does not exist until Phase 4.
    """
    return MetadataSpec(
        ref=SpecRef(adapter="openfoam", version=1),
        fields=(
            MetadataField("application", FieldType.STR, "Application", display_order=1),
            MetadataField("endTime", FieldType.FLOAT, "End time", unit="s", display_order=2),
            MetadataField("deltaT", FieldType.FLOAT, "Time step", unit="s", display_order=3),
            MetadataField("decomposition", FieldType.INT, "Subdomains", display_order=4),
            MetadataField("mesh_cells", FieldType.INT, "Cells", display_order=5),
            MetadataField("notes_blob", FieldType.STR, "Blob", searchable=False, display_order=9),
        ),
    )


@pytest.fixture
def repo(conn: sqlite3.Connection, clock: FakeClock, foam_spec: MetadataSpec) -> JobRepository:
    """A repository on the in-memory database, wired to the fake clock."""
    return JobRepository(conn, clock=clock, specs={foam_spec.ref.adapter: foam_spec})


@pytest.fixture
def make_spec(tmp_path: Path):
    """Factory for :class:`JobSpec` values with sensible defaults."""

    def _make(
        name: str = "case",
        *,
        solver: str = "openfoam",
        cores: int = 4,
        ram_mb: int | None = None,
        priority: int = 0,
        tags: frozenset[str] | set[str] = frozenset(),
        note: str | None = None,
        metadata=None,
        solver_binary: str | None = None,
        workdir: Path | None = None,
        depends_on: str | None = None,
    ) -> JobSpec:
        directory = workdir or (tmp_path / name)
        directory.mkdir(parents=True, exist_ok=True)
        return JobSpec(
            workdir=directory,
            solver=solver,
            solver_binary=solver_binary,
            resources=ResourceRequest(cores=cores, ram_mb=ram_mb),
            name=name,
            priority=priority,
            tags=frozenset(tags),
            note=note,
            metadata=metadata,
            depends_on_job_id=depends_on,
        )

    return _make
