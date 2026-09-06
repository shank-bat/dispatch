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
from enum import StrEnum
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
    "ResourceKind",
    "ResourceRequest",
    "Sample",
    "Sweep",
    "SweepSpec",
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


class ResourceKind(StrEnum):
    """Which of the machine's two schedulable pools a job draws its work from.

    Not a cosmetic label. CPU cores and GPUs are counted in separate ledgers (§4.3), so a
    GPU job occupies GPUs and only the cores it actually asked for, and a CPU job can
    never hold a GPU. The kind is what makes that guarantee checkable at the point a
    request is built, rather than at the point a job mysteriously fails to start.

    :class:`StrEnum` so the stored value is readable in a ``sqlite3`` session.
    """

    CPU = "cpu"
    """Work that consumes cores. The default, and what every job predating this field was."""

    GPU = "gpu"
    """Work that consumes GPUs, plus however many cores it declares to drive them."""


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """What a job asks the machine for.

    Attributes:
        cores: Logical CPU cores. Always required -- a job that does not know how parallel
            it is cannot be scheduled safely. A GPU job still declares cores, because the
            host process driving the GPU is real work; it simply usually declares one.
        ram_mb: Optional estimate. When given, it gates admission; when absent, the job is
            admitted on cores alone. This is what makes RAM-aware scheduling already
            present rather than merely possible (§4.3).
        gpus: GPUs to reserve. Zero for CPU work.
        kind: Which pool this request draws from. Constrained against ``gpus`` below so
            the two can never disagree.
    """

    cores: int
    ram_mb: int | None = None
    gpus: int = 0
    kind: ResourceKind = ResourceKind.CPU

    def __post_init__(self) -> None:
        if self.cores < 1:
            raise ValidationError(f"A job must request at least one core, got {self.cores}")
        if self.ram_mb is not None and self.ram_mb <= 0:
            raise ValidationError(f"RAM estimate must be positive, got {self.ram_mb}")
        if self.gpus < 0:
            raise ValidationError(f"GPU count cannot be negative, got {self.gpus}")
        # The two fields are kept consistent here rather than at each call site, so that
        # "a CPU job cannot accidentally hold a GPU" is a property of the type instead of
        # a convention every caller has to remember.
        if self.kind is ResourceKind.CPU and self.gpus:
            raise ValidationError(
                f"A CPU job cannot request {self.gpus} GPU(s). "
                "Submit it with --resource gpu if it is GPU work."
            )
        if self.kind is ResourceKind.GPU and self.gpus < 1:
            raise ValidationError("A GPU job must request at least one GPU")

    @classmethod
    def build(
        cls,
        *,
        cores: int = 1,
        ram_mb: int | None = None,
        gpus: int | None = None,
        resource: str | None = None,
    ) -> ResourceRequest:
        """Build a request from what a user typed, filling in the obvious.

        The two flags overlap, so one may be omitted whenever the other settles the
        question: ``--gpus 1`` is GPU work, ``--resource gpu`` wants at least one GPU. A
        combination that genuinely contradicts itself -- ``--resource cpu --gpus 2`` -- is
        an error rather than a silent reinterpretation of what was asked for.

        Args:
            cores: Requested cores.
            ram_mb: Optional RAM estimate.
            gpus: Requested GPUs, or ``None`` when the user did not say.
            resource: ``"cpu"``, ``"gpu"``, or ``None`` when the user did not say.

        Raises:
            ValidationError: On an unknown resource name or a contradictory combination.
        """
        count = int(gpus or 0)
        if resource is None:
            chosen = ResourceKind.GPU if count > 0 else ResourceKind.CPU
        else:
            try:
                chosen = ResourceKind(str(resource).strip().lower())
            except ValueError as exc:
                valid = ", ".join(k.value for k in ResourceKind)
                raise ValidationError(
                    f"Unknown resource type {resource!r}. Valid types: {valid}"
                ) from exc
        if chosen is ResourceKind.GPU and count == 0:
            count = 1
        return cls(cores=cores, ram_mb=ram_mb, gpus=count, kind=chosen)

    @property
    def is_gpu(self) -> bool:
        """Whether this request draws on the GPU ledger."""
        return self.kind is ResourceKind.GPU

    def describe(self) -> str:
        """A compact human summary, e.g. ``20 cores`` or ``1 GPU, 4 cores``."""
        cores = f"{self.cores} core{'' if self.cores == 1 else 's'}"
        if not self.gpus:
            return cores
        return f"{self.gpus} GPU{'' if self.gpus == 1 else 's'}, {cores}"


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
    depends_on_job_id: str | None = None
    """Run only after this job has completed. ``None`` -- the default -- means the job is
    scheduled as soon as it fits, which is how every job behaves unless asked otherwise."""

    sweep_id: str | None = None
    """The sweep this case belongs to. ``None`` for an ordinary standalone submission."""

    sweep_position: int | None = None
    """Index within the sweep, from zero. ``None`` when the job is not part of one."""

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
    log_path: Path | None = None
    """The solver's output log, inside the job's own working directory (§6.4).

    This is the file a human browsing the case will find -- ``log.foam`` beside the
    ``system/`` directory rather than a UUID under ``~/.local/share``. ``None`` for jobs
    written before the convention existed, and for jobs whose case directory turned out
    not to be writable; :attr:`stdout_path` is the authority in both cases and is what
    every reader already uses.
    """

    pid: int | None = None
    pid_start_time: float | None = None
    tags: frozenset[str] = frozenset()
    metadata: CaseMetadata = field(default_factory=CaseMetadata.empty)
    provenance: Provenance | None = None
    metrics: JobMetrics = field(default_factory=JobMetrics)
    depends_on_job_id: str | None = None
    """The job this one was asked to run after, if any (§6.2).

    ``None`` for every job that did not ask, which is the default and the overwhelming
    majority: such a job is scheduled purely on resources, exactly as before this field
    existed.
    """

    sweep_id: str | None = None
    """The sweep this job belongs to, if any (§6.12).

    ``None`` for every ordinary job, which is what makes sweeps additive: the scheduler's
    sweep rule reads this field, finds nothing, and leaves such a job exactly as
    opportunistic as it has always been.
    """

    sweep_position: int | None = None
    """This job's index within its sweep, from zero, in the order the cases were found.

    Stored rather than derived so the sweep's order is a fact about the submission and not
    a re-reading of a directory that may have changed since. Queue order still comes from
    ``seq``; this is what lets the interface say "case 3 of 40" and mean it.
    """

    boot_time: float | None = None
    """Which boot of this machine the job started on, as the kernel's ``btime`` (§6.7).

    Recorded when the job starts and compared against the machine's current boot identity
    at the next daemon startup. That comparison -- boot against boot, from one source -- is
    what distinguishes a reboot from a daemon restart without assuming anything about
    clocks. ``None`` for jobs that never started, and on machines that cannot report it.
    """

    resume_requested: bool = False
    """Whether this job must restart from the simulation's own last saved state (§6.7).

    Set only by reboot recovery, and cleared once the job starts. The flag says *that* a
    resume is wanted; **where** to resume from is never stored here -- it is read from the
    case by the adapter at plan time, because the simulation's files are the only honest
    authority on what it actually finished writing.
    """

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
    def gpus(self) -> int:
        """Requested GPU count. Zero for CPU work."""
        return self.resources.gpus

    @property
    def resource_kind(self) -> ResourceKind:
        """Which pool this job draws from."""
        return self.resources.kind

    @property
    def output_path(self) -> Path | None:
        """Where to read this job's solver output.

        The working-directory log when there is one, falling back to the recorded stdout
        path -- which is what every job predating the convention has, and is why history
        stays readable across the change.
        """
        return self.log_path or self.stdout_path

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

    total_gpus: int = 0
    """GPUs the machine has, as the ledger counts them. Zero on most machines."""

    allocated_gpus: int = 0
    """GPUs handed out to PREPARING and RUNNING jobs."""

    @property
    def free_cores(self) -> int:
        """Cores available to new jobs: total, less reserved, less allocated."""
        return max(0, self.total_cores - self.reserved_cores - self.allocated_cores)

    @property
    def free_gpus(self) -> int:
        """GPUs available to new jobs.

        No reservation is subtracted: the responsiveness argument that holds a core back
        for SSH (§4.3) has no GPU equivalent, and a machine with one GPU that permanently
        reserved it would be a machine with no GPU.
        """
        return max(0, self.total_gpus - self.allocated_gpus)

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


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """A request to submit a directory of cases as one sweep, before it exists."""

    root: Path
    solver: str
    cases: Sequence[Path]
    cores_per_job: int
    concurrency: int
    name: str = ""
    sweep_id: str = field(default_factory=new_job_id)
    """Generated up front so member specs can name the sweep before it is inserted."""

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValidationError("A sweep must contain at least one case")
        if self.cores_per_job < 1:
            raise ValidationError(
                f"A sweep needs at least one core per job, got {self.cores_per_job}"
            )
        if self.concurrency < 1:
            raise ValidationError(
                f"A sweep must be allowed to run at least one job at a time, got "
                f"{self.concurrency}"
            )
        object.__setattr__(self, "root", Path(self.root).expanduser())
        if not self.name:
            object.__setattr__(self, "name", self.root.name or str(self.root))


@dataclass(frozen=True, slots=True)
class Sweep:
    """A group of independent cases submitted together, scheduled under one limit.

    A sweep is emphatically **not** a job. Every case in it is an ordinary job holding its
    own ordinary allocation, and the sweep contributes exactly one extra scheduling rule:
    no more than :attr:`concurrency` of its members may run at once. Modelling it as one
    large job would reserve ``cores_per_job * concurrency`` cores as a block, which is both
    a lie to the ledger and the opposite of what the setting is for -- the whole point is
    that the cores a sweep is *not* using stay available to unrelated work.

    See ``docs/ARCHITECTURE.md`` §6.12.
    """

    id: str
    name: str
    root: Path
    solver: str
    cores_per_job: int
    concurrency: int
    """Hard cap on simultaneously running members. Free cores never override it."""

    created_at: float
    total: int = 0
    """How many cases were submitted as part of this sweep."""

    running: int = 0
    """How many members are currently PREPARING or RUNNING. Derived, not stored."""

    finished: int = 0
    """How many members have reached a terminal state. Derived, not stored."""

    def describe(self) -> str:
        """A short progress line for the queue view, e.g. ``2/8 running``."""
        return f"{self.running}/{self.total} running"
