"""The domain objects Dispatch schedules, stores, and displays.

Everything here is a frozen, slotted dataclass. Frozen because these are snapshots handed
across layer boundaries -- the scheduler receives a :class:`Job`, and it must be impossible
for it to mutate one and have the change appear somewhere else. Slotted because with a
history of thousands of jobs, the per-instance ``__dict__`` is real memory that a
simulation could have used (§10).

Mutation happens exactly one way: through the repository, which writes to SQLite and hands
back a new snapshot.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from dispatch.core.errors import ValidationError
from dispatch.core.metadata import CaseMetadata
from dispatch.core.provenance import Provenance
from dispatch.core.states import ACTIVE_STATES, TERMINAL_STATES, ExitReason, JobState
from dispatch.core.tags import normalise_tags

__all__ = [
    "Detection",
    "Job",
    "JobEvent",
    "JobMetrics",
    "JobSpec",
    "Note",
    "Page",
    "ResourceRequest",
    "Sample",
    "SystemSnapshot",
    "new_job_id",
]

MAX_NAME_LENGTH = 200


def new_job_id() -> str:
    """Return a fresh job identifier.

    UUID4 in canonical hyphenated form. Stored as TEXT: a job id appears in log paths, in
    ``ps`` output, and in things the user types, so it needs to be greppable rather than
    compact.
    """
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """What a job asks the machine for.

    Attributes:
        cores: Logical CPU cores. Always required -- a job that does not know how parallel
            it is cannot be scheduled safely.
        ram_mb: Optional estimate. When given, it gates admission; when absent, the job is
            admitted on cores alone. This is what makes RAM-aware scheduling already
            present rather than merely possible (§4.3).
    """

    cores: int
    ram_mb: int | None = None

    def __post_init__(self) -> None:
        if self.cores < 1:
            raise ValidationError(f"A job must request at least one core, got {self.cores}")
        if self.ram_mb is not None and self.ram_mb <= 0:
            raise ValidationError(f"RAM estimate must be positive, got {self.ram_mb}")


@dataclass(frozen=True, slots=True)
class Detection:
    """An adapter's claim that a directory is a case it can handle.

    Attributes:
        solver: Adapter name, e.g. ``openfoam``.
        confidence: 0.0-1.0. Ranks competing claims. OpenFOAM scores 0.95 on
            ``system/controlDict`` because that file means one thing; Basilisk scores 0.7
            on a ``.c`` file including a Basilisk header, because ``.c`` files are common
            and the evidence is weaker. Ambiguity is the *only* case that prompts the user.
        solver_binary: The specific application, when detection can determine it
            (``interFoam``, ``SU2_CFD``).
        label: Human-readable summary for the submit wizard.
        entry: The specific file the detection keyed on -- a config path for SU2, a source
            file for Basilisk. Carried into the plan so a case with two configs does not
            silently run the wrong one.
        detail: Adapter-specific extras for the wizard, e.g. an existing decomposition count.
    """

    solver: str
    confidence: float
    solver_binary: str | None = None
    label: str = ""
    entry: Path | None = None
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(f"Detection confidence must be in [0,1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class JobSpec:
    """A request to create a job, before it exists.

    Separate from :class:`Job` because the fields differ in kind: a spec is what the user
    chose, a job additionally carries what the system decided. Merging them would mean a
    ``Job`` with a meaningless ``id`` and ``state`` at submission time.
    """

    workdir: Path
    solver: str
    resources: ResourceRequest
    name: str = ""
    solver_binary: str | None = None
    priority: int = 0
    tags: frozenset[str] = frozenset()
    note: str | None = None
    metadata: CaseMetadata | None = None

    def __post_init__(self) -> None:
        if not self.solver:
            raise ValidationError("A job spec must name a solver")
        if len(self.name) > MAX_NAME_LENGTH:
            raise ValidationError(
                f"Job name is {len(self.name)} characters; the maximum is {MAX_NAME_LENGTH}"
            )
        # Normalise in the constructor so that every construction path -- IPC, CLI, tests --
        # produces the same canonical form, rather than trusting each caller to remember.
        object.__setattr__(self, "workdir", Path(self.workdir).expanduser())
        object.__setattr__(self, "tags", normalise_tags(self.tags))
        if not self.name:
            object.__setattr__(self, "name", self.workdir.name or str(self.workdir))


@dataclass(frozen=True, slots=True)
class JobMetrics:
    """Measured resource usage, accumulated while a job runs (§6.6)."""

    peak_rss_mb: int | None = None
    mean_cpu_pct: float | None = None
    runtime_s: float | None = None

    @property
    def is_empty(self) -> bool:
        """Whether nothing has been measured yet."""
        return self.peak_rss_mb is None and self.mean_cpu_pct is None and self.runtime_s is None


@dataclass(frozen=True, slots=True)
class Job:
    """A simulation, at some point in its life.

    See ``docs/ARCHITECTURE.md`` §4.1. Note the absence of a ``queue_position`` field: it
    is derived at read time from ``(priority, seq)`` rather than stored, because a stored
    position must be rewritten on every insert, hold, and priority change, and drifts after
    a crash (§13.3).
    """

    id: str
    seq: int
    name: str
    workdir: Path
    solver: str
    resources: ResourceRequest
    state: JobState
    created_at: float

    solver_binary: str | None = None
    priority: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    exit_reason: ExitReason | None = None
    exit_signal: str | None = None
    exit_detail: str | None = None
    """Why the job failed, quoted from its own output. ``None`` when it did not, or when
    the log said nothing an adapter could make sense of."""
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    pid: int | None = None
    pid_start_time: float | None = None
    tags: frozenset[str] = frozenset()
    metadata: CaseMetadata = field(default_factory=CaseMetadata.empty)
    provenance: Provenance | None = None
    metrics: JobMetrics = field(default_factory=JobMetrics)

    # -- convenience -----------------------------------------------------------------

    @property
    def cores(self) -> int:
        """Requested core count."""
        return self.resources.cores

    @property
    def ram_estimate_mb(self) -> int | None:
        """Requested RAM estimate, if any."""
        return self.resources.ram_mb

    @property
    def is_terminal(self) -> bool:
        """Whether the job has finished, one way or another."""
        return self.state in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        """Whether the job currently holds an allocation in the resource ledger."""
        return self.state in ACTIVE_STATES

    def elapsed(self, now: float) -> float | None:
        """Seconds since the job started, or its total runtime if it has finished.

        Returns ``None`` for a job that has not started. Uses ``finished_at`` when
        present so that the value stops advancing once the job is done.
        """
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else now
        return max(0.0, end - self.started_at)

    def with_state(self, state: JobState) -> Job:
        """Return a copy in a new state. Does not validate the transition.

        Validation belongs to the repository, which is the only component that can make a
        transition atomic with respect to other writers.
        """
        return replace(self, state=state)


@dataclass(frozen=True, slots=True)
class JobEvent:
    """One entry in a job's audit trail.

    Every state change appends one of these, so "what happened to this job, in order" is
    always answerable -- including for jobs that finished months ago.
    """

    id: int
    job_id: str
    ts: float
    kind: str
    """``state``, ``step``, ``signal``, ``warn``, or ``note``."""

    detail: str


@dataclass(frozen=True, slots=True)
class Note:
    """A free-text annotation attached to a job.

    Notes are added at any time, including long after completion -- which is when you
    usually learn that a run mattered.
    """

    id: int
    job_id: str
    ts: float
    body: str


@dataclass(frozen=True, slots=True)
class Sample:
    """One resource measurement of a running job."""

    ts: float
    rss_mb: int
    cpu_pct: float


@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    """The machine's state at an instant, for the dashboard.

    Distinguishes two different notions of "busy" that must not be conflated (§4.3):
    :attr:`allocated_cores` is Dispatch's ledger and is what admission decisions use;
    :attr:`cpu_percent` is what the CPUs are actually doing and is display-only. They
    disagree whenever a solver blocks on I/O, and that is correct.
    """

    timestamp: float
    hostname: str
    total_cores: int
    allocated_cores: int
    reserved_cores: int
    cpu_percent: float
    per_core_percent: Sequence[float]
    total_ram_mb: int
    used_ram_mb: int
    available_ram_mb: int
    load_average: tuple[float, float, float]
    uptime_s: float

    @property
    def free_cores(self) -> int:
        """Cores available to new jobs: total, less reserved, less allocated."""
        return max(0, self.total_cores - self.reserved_cores - self.allocated_cores)

    @property
    def ram_percent(self) -> float:
        """Used RAM as a percentage of total."""
        return 100.0 * self.used_ram_mb / self.total_ram_mb if self.total_ram_mb else 0.0


@dataclass(frozen=True, slots=True)
class Page[T]:
    """A slice of a larger result set.

    Paging is not a nicety here: it is what bounds the size of an IPC message, and an
    unbounded ``job.list`` on a machine with three years of history would be a memory
    spike in the daemon at exactly the wrong moment (§7.5).
    """

    items: Sequence[T]
    total: int
    offset: int

    @property
    def limit(self) -> int:
        """Number of items in this page."""
        return len(self.items)

    @property
    def has_more(self) -> bool:
        """Whether further pages exist after this one."""
        return self.offset + len(self.items) < self.total
