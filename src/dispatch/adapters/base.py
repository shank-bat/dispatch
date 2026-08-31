"""The solver adapter interface.

This is the boundary that keeps the scheduler ignorant of solvers. An adapter answers four
questions about a directory -- is this mine, is it valid, what should I record about it, and
what commands would run it -- and never executes anything itself.

That last constraint is the important one. Adapters return an
:class:`~dispatch.core.plan.ExecutionPlan`, which is data. The executor runs it, logs it,
times it out, and cancels it, once, for every solver that will ever exist. It is also why
``--dry-run`` is free: building a plan and running one are already separate operations.

See ``docs/ARCHITECTURE.md`` §8.
"""

from __future__ import annotations

import logging
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from dispatch.core.metadata import EMPTY_SPEC, CaseMetadata, MetadataSpec
from dispatch.core.models import Detection
from dispatch.core.plan import ExecutionPlan
from dispatch.core.series import PlotData
from dispatch.core.validation import ValidationReport

__all__ = [
    "ADAPTER_API_VERSION",
    "DEFAULT_LOG_NAME",
    "BaseAdapter",
    "CaseContext",
    "Progress",
    "SolverAdapter",
    "generic_failure_summary",
]

log = logging.getLogger(__name__)

DEFAULT_LOG_NAME = "log.job"
"""What an adapter's output log is called when the adapter does not say.

The name of a job's log is a solver convention -- ``log.foam`` beside a ``system/``
directory is instantly recognisable to somebody who has never heard of Dispatch -- so the
adapter chooses it and the executor merely opens it. That is also what keeps the daemon
free of solver names: grep it for ``foam`` and there is nothing to find (§1.1).
"""

ADAPTER_API_VERSION = 1
"""The adapter contract version.

Covers the method set below and the meaning of :class:`CaseContext`,
:class:`~dispatch.core.models.Detection`,
:class:`~dispatch.core.validation.ValidationReport`,
:class:`~dispatch.core.plan.ExecutionPlan`, and
:class:`~dispatch.core.metadata.CaseMetadata`.

Additive changes -- a new optional method with a default -- do not bump it. Removing
anything, or changing what it means, does. ``log_name``, :meth:`SolverAdapter.parse_series`
and :attr:`CaseContext.gpus` were all added this way: an adapter written against the
original version keeps working, with a generic log name, no plottable series, and a GPU
count it does not read. An adapter declaring a different version is
refused at registration with a message naming both, while the daemon starts normally with
the remaining adapters: one stale third-party plugin must not take three months of queued
work down with it.
"""


@dataclass(frozen=True, slots=True)
class CaseContext:
    """Everything an adapter needs to know about one case.

    Passed to every adapter method, so adapters are testable with a temporary directory
    and no daemon anywhere in sight.

    Attributes:
        workdir: The case directory.
        cores: Cores the user requested. Drives decomposition decisions.
        ram_mb: Optional RAM estimate.
        entry: The specific file detection keyed on -- a config for SU2, a source file for
            Basilisk. Carried from detection so a directory with two configs cannot
            silently run the wrong one.
        env: The environment the job's commands will run in.
        settings: This adapter's section of ``config.toml``.
        job_name: The job's display name, for constructing output names.
        metadata: Metadata collected so far, if any.
    """

    workdir: Path
    cores: int = 1
    ram_mb: int | None = None
    entry: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    job_name: str = ""
    metadata: CaseMetadata | None = None
    gpus: int = 0
    """GPUs the scheduler has reserved for this job. Zero for CPU work.

    A count, not a set of device indices. Handing out specific devices is deliberately
    deferred (§13.24); what an adapter does with this is decide whether to ask its
    framework for a GPU at all -- and, for CPU work, to say so explicitly rather than
    letting a library help itself to whatever it finds.
    """

    def path(self, *parts: str) -> Path:
        """A path inside the case directory."""
        return self.workdir.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        """Whether a path inside the case directory exists."""
        return self.path(*parts).exists()

    @property
    def uses_gpu(self) -> bool:
        """Whether the scheduler reserved any GPU for this job."""
        return self.gpus > 0

    def which(self, program: str) -> str | None:
        """Locate ``program`` on the PATH this job will actually use.

        Uses the job's own ``env``, not the daemon's -- an OpenFOAM binary exists only
        after its environment has been sourced, and checking the daemon's PATH would
        report every solver as missing.
        """
        path = self.env.get("PATH") if self.env else None
        return shutil.which(program, path=path)


@dataclass(frozen=True, slots=True)
class Progress:
    """How far along a running job is, as read from its log tail."""

    current: float
    """Current simulated time or iteration number."""

    total: float | None = None
    """Target, when known from the case configuration."""

    label: str = ""
    """What ``current`` counts, e.g. ``"Time"`` or ``"Iteration"``."""

    @property
    def fraction(self) -> float | None:
        """Completed fraction in [0, 1], or ``None`` when the target is unknown.

        A solver with no declared end time reports ``None`` rather than a fabricated
        percentage; the TUI shows elapsed time instead.
        """
        if self.total is None or self.total <= 0:
            return None
        return max(0.0, min(1.0, self.current / self.total))


@runtime_checkable
class SolverAdapter(Protocol):
    """What the daemon requires of a solver adapter.

    :class:`BaseAdapter` implements the optional parts; most adapters subclass it rather
    than satisfying this protocol from scratch.
    """

    api_version: ClassVar[int]
    name: ClassVar[str]
    display_name: ClassVar[str]
    adapter_version: ClassVar[int]
    metadata_spec: ClassVar[MetadataSpec]
    env_keys: ClassVar[Sequence[str]]
    log_name: ClassVar[str]

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        """Decide whether ``path`` is a case this adapter handles."""
        ...

    def validate(self, ctx: CaseContext) -> ValidationReport:
        """Pre-flight checks, before the job is queued."""
        ...

    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        """Build the ordered commands that prepare and run this case."""
        ...

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        """Extract declared case settings."""
        ...

    def prepare_environment(self, ctx: CaseContext) -> Mapping[str, str]:
        """Return the environment the job's commands should run in."""
        ...

    def solver_version(self, ctx: CaseContext) -> str | None:
        """Identify the solver build, for the provenance record."""
        ...

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        """Tags to offer the user at submission. Never applied silently."""
        ...

    def parse_progress(self, tail: str, ctx: CaseContext) -> Progress | None:
        """Read progress out of the last chunk of a job's log."""
        ...

    def parse_series(self, text: str, ctx: CaseContext) -> PlotData:
        """Extract plottable numerical series from a job's output."""
        ...

    def stop_gracefully(self, ctx: CaseContext) -> bool:
        """Attempt a solver-native clean stop. ``False`` falls back to signals."""
        ...

    def finalize(self, ctx: CaseContext) -> None:
        """Undo anything done to the case to control the run itself."""
        ...

    def explain_failure(self, tail: str, ctx: CaseContext) -> str | None:
        """Summarise why a run failed, from the end of its output."""
        ...


MAX_SUMMARY_LINES = 6
MAX_SUMMARY_CHARS = 600

_NOISE = re.compile(r"^[\s\-=*_~#/\\.]*$")
"""Rule-off lines. MPI in particular brackets its errors in seventy-odd dashes."""

_ERROR_MARKER = re.compile(
    r"""
      FOAM\ FATAL
    | MPI_ABORT | \bnot\ enough\ slots\b
    | \bfatal\ error\b | \berror:      # compiler and library style, with the colon
    | \bcannot\ (?:find|open|read|create)\b
    | \bno\ such\ file\b | \bcommand\ not\ found\b | \bpermission\ denied\b
    | \bsegmentation\ fault\b | \bbus\ error\b | \bterminate\ called\b
    | \bassertion\b.*\bfailed\b | \bundefined\ reference\b
    | \bout\ of\ memory\b | \bkilled\b
    | ^Traceback\ \(most\ recent\ call\ last\)
    | \bdiverg (?:ed|ence|ing)\b
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)
"""What "the interesting part" looks like across solvers, compilers, and launchers.

Deliberately strict, and every loosening of it has to survive one test: a healthy OpenFOAM
log prints ``time step continuity errors`` on **every single time step**, so a pattern as
innocent-looking as ``\\berror\\b`` matches thousands of lines of a perfectly good run and
would confidently offer one of them as the reason the job died. A marker earns its place
here only if its presence means something actually went wrong.
"""


def generic_failure_summary(tail: str) -> str | None:
    """Reduce the end of a failed job's output to the part worth reading.

    Solvers do not agree on how to report an error, but they do agree on where: the end.
    The heuristic is to find the last line that looks like a complaint and keep it with the
    few lines after it, which is where the detail usually sits. Failing that, the last few
    non-blank lines are still a far better answer than "exit code 1".
    """
    lines = [line.rstrip() for line in tail.splitlines()]
    meaningful = [line for line in lines if line.strip() and not _NOISE.match(line)]
    if not meaningful:
        return None

    marked = [i for i, line in enumerate(meaningful) if _ERROR_MARKER.search(line)]
    start = marked[-1] if marked else max(0, len(meaningful) - MAX_SUMMARY_LINES)
    chosen = meaningful[start : start + MAX_SUMMARY_LINES]

    summary = "\n".join(line.strip() for line in chosen).strip()
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"
    return summary or None


class BaseAdapter(ABC):
    """Convenience base implementing every optional part of :class:`SolverAdapter`.

    Subclasses must provide :meth:`detect`, :meth:`validate`, and :meth:`plan`. Everything
    else has a defensible default, so a minimal adapter is three methods.
    """

    api_version: ClassVar[int] = ADAPTER_API_VERSION
    name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    adapter_version: ClassVar[int] = 1
    metadata_spec: ClassVar[MetadataSpec] = EMPTY_SPEC
    env_keys: ClassVar[Sequence[str]] = ()
    log_name: ClassVar[str] = DEFAULT_LOG_NAME
    """Filename for this solver's output log inside the case directory.

    Follow the convention the solver's own users already have: ``log.foam``,
    ``log.su2``. It has to be recognisable to somebody browsing the directory who has
    never heard of Dispatch.
    """

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        """Args:
        settings: This adapter's section of ``config.toml``.
        """
        self.settings: Mapping[str, Any] = dict(settings or {})

    # -- required ---------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def detect(cls, path: Path) -> Detection | None:
        """Return a :class:`Detection` if ``path`` is a case this adapter handles.

        Must be cheap and read-only: this runs against every directory the user browses
        to, for every registered adapter.
        """

    @abstractmethod
    def validate(self, ctx: CaseContext) -> ValidationReport:
        """Check the case before it is queued."""

    @abstractmethod
    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        """Build the execution plan. Describes side effects; performs none."""

    # -- optional, with defaults --------------------------------------------------------

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        """Extract declared case settings. Defaults to an empty envelope."""
        return CaseMetadata.empty(self.name)

    def prepare_environment(self, ctx: CaseContext) -> Mapping[str, str]:
        """Return the job's environment. Defaults to inheriting the daemon's."""
        return ctx.env

    def solver_version(self, ctx: CaseContext) -> str | None:
        """Identify the solver build. Defaults to unknown, which is recorded as NULL."""
        return None

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        """Tags to offer at submission. Defaults to none."""
        return ()

    def parse_progress(self, tail: str, ctx: CaseContext) -> Progress | None:
        """Read progress from a log tail. Defaults to none, which is a valid answer."""
        return None

    def parse_series(self, text: str, ctx: CaseContext) -> PlotData:
        """Extract plottable numerical series from a job's output.

        Called with the job's log -- the whole of it, or its end when it is very large --
        only when a user asks to plot the job. Never on the hot path, never by the
        scheduler, and never automatically: nothing about a run's *state* depends on what
        this returns, so a parser that misreads a line cannot make a completed job look
        failed.

        Defaults to nothing, which is a valid and common answer: the interface then says
        this job has no plottable data rather than inventing some.

        Args:
            text: The job's output, or its last portion.
            ctx: The case, for anything the log does not say -- a target end time, say.

        Returns:
            The series this log actually contains. Emit only quantities that appeared:
            a log with no ``Uy`` residual must not offer an empty ``residual(Uy)``.
        """
        return PlotData()

    def stop_gracefully(self, ctx: CaseContext) -> bool:
        """Attempt a clean solver-native stop.

        Returning ``False`` -- the default -- tells the executor to use the
        SIGINT/SIGTERM/SIGKILL ladder instead.
        """
        return False

    def finalize(self, ctx: CaseContext) -> None:
        """Undo anything :meth:`stop_gracefully` or :meth:`plan` did to *control* the run.

        Called after the solver exits, however it exited, including on cancellation. This
        is for edits to the case that exist to steer this one run and would be wrong to
        leave behind -- OpenFOAM's ``stopAt writeNow`` is the motivating example.

        It is emphatically **not** for cleaning up results. A cancelled run's output is the
        user's, and deleting any of it is not Dispatch's decision to make.

        Failures here are logged and swallowed by the executor: a job's recorded outcome
        must not depend on tidying that comes after it.
        """
        return None

    def explain_failure(self, tail: str, ctx: CaseContext) -> str | None:
        """Pull the reason a run failed out of the end of its output.

        Args:
            tail: The last few kilobytes of the job's stderr and stdout, stderr first --
                which is where solvers and MPI launchers put the thing worth reading.
            ctx: The case.

        Returns:
            A short explanation to show beside the exit code, or ``None`` when the output
            says nothing useful. The default returns the last non-empty, non-noise lines,
            which is a surprisingly good answer for most solvers; adapters override it
            where the solver has a recognisable error format.
        """
        return generic_failure_summary(tail)

    def setting(self, key: str, default: Any = None) -> Any:
        """Read one value from this adapter's configuration section."""
        return self.settings.get(key, default)
