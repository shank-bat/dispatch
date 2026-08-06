"""Execution plans: what an adapter *describes*, and what the executor *does*.

This module is the seam that keeps the scheduler ignorant of solvers. An adapter never
spawns a process; it returns an :class:`ExecutionPlan`, an ordered list of commands. The
executor runs them, logs them, times them out, and cancels them -- once, in one place, for
every solver that will ever exist.

Two consequences worth naming, because they are the reason for the design:

* ``decomposePar``, ``reconstructPar``, and ``qcc`` are all just PREPARE steps. The
  executor cannot tell them apart, so adding a solver that must be *compiled* before it
  runs required no scheduler change at all.
* Dry run (§6.11) is free. Building a plan and running one are already separate
  operations, so "show me what would happen" is the real code path minus the last call --
  it cannot drift from reality the way a parallel explain-only implementation would.

See ``docs/ARCHITECTURE.md`` §6.3.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from dispatch.core.errors import ValidationError
from dispatch.core.models import Detection
from dispatch.core.validation import ValidationReport

__all__ = [
    "CommandStep",
    "DryRunReport",
    "ExecutionPlan",
    "FailureAction",
    "StepKind",
    "StepOutcome",
]


class StepKind(StrEnum):
    """What role a step plays in a job."""

    PREPARE = "PREPARE"
    """Runs before the solver, with the job in state PREPARING.

    Mesh decomposition, source compilation, field initialisation. Resources are already
    allocated: preparation uses the machine.
    """

    SOLVE = "SOLVE"
    """The simulation itself. Exactly one per plan.

    Its lifetime defines the RUNNING state, its pid is what gets supervised, and its exit
    code becomes the job's exit code.
    """

    CLEANUP = "CLEANUP"
    """Runs after the solver, regardless of the solver's exit status.

    Note that OpenFOAM's ``reconstructPar`` is deliberately *not* a cleanup step: the
    decomposed case is left exactly as the solver left it (§8.3).
    """


class FailureAction(StrEnum):
    """What the executor does when a step exits non-zero."""

    ABORT = "ABORT"
    """Fail the job. The correct default: a failed ``decomposePar`` means the solver would
    run on a case that does not exist."""

    WARN = "WARN"
    """Record the failure and continue. For steps that are best-effort."""


@dataclass(frozen=True, slots=True)
class CommandStep:
    """One command to execute.

    ``argv`` is a list, never a shell string: no quoting bugs, no injection surface, and
    a case directory containing a space stays a case directory containing a space.

    Attributes:
        argv: Program and arguments. Must be non-empty.
        cwd: Working directory. Solver-relative paths in case files depend on this being
            the case directory, not the daemon's.
        description: Human-readable summary shown in the TUI and dry-run output, e.g.
            "Decomposing case into 20 subdomains".
        kind: Role in the job.
        env: Complete environment for the process, or ``None`` to inherit the daemon's.
            Adapters that need a sourced environment (OpenFOAM) supply it whole.
        on_failure: What a non-zero exit means.
        timeout_s: Wall-clock limit, or ``None`` for unlimited. SOLVE steps are normally
            unlimited -- a simulation legitimately runs for a week.
    """

    argv: Sequence[str]
    cwd: Path
    description: str
    kind: StepKind = StepKind.PREPARE
    env: Mapping[str, str] | None = None
    on_failure: FailureAction = FailureAction.ABORT
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValidationError("CommandStep.argv must not be empty")
        if any(not isinstance(a, str) for a in self.argv):
            raise ValidationError(f"CommandStep.argv must be all strings: {self.argv!r}")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValidationError(f"CommandStep.timeout_s must be positive: {self.timeout_s}")

    @property
    def program(self) -> str:
        """The executable name, for PATH checks and display."""
        return self.argv[0]

    def render(self) -> str:
        """Shell-quoted command line, for logs and dry-run output.

        For display only. The command is never executed through a shell.
        """
        return shlex.join(self.argv)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """The complete ordered set of commands that prepare and run one job.

    Invariant: exactly one step has kind SOLVE. A plan with none would leave a job with no
    process to supervise; a plan with two would leave the executor with no defensible
    answer to "which exit code is the job's". Enforced at construction, so a buggy adapter
    fails at plan time -- before any of its PREPARE steps have touched the user's case.
    """

    steps: Sequence[CommandStep]

    def __post_init__(self) -> None:
        solves = [s for s in self.steps if s.kind is StepKind.SOLVE]
        if len(solves) != 1:
            raise ValidationError(
                f"An execution plan needs exactly one SOLVE step, found {len(solves)}",
                detail={"kinds": [str(s.kind) for s in self.steps]},
            )

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def solve(self) -> CommandStep:
        """The SOLVE step. Guaranteed to exist by the constructor invariant."""
        return next(s for s in self.steps if s.kind is StepKind.SOLVE)

    def of_kind(self, kind: StepKind) -> Sequence[CommandStep]:
        """Steps of one kind, in plan order."""
        return tuple(s for s in self.steps if s.kind is kind)

    @property
    def prepare(self) -> Sequence[CommandStep]:
        """PREPARE steps, in order."""
        return self.of_kind(StepKind.PREPARE)

    @property
    def cleanup(self) -> Sequence[CommandStep]:
        """CLEANUP steps, in order."""
        return self.of_kind(StepKind.CLEANUP)


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """The result of running one step."""

    step: CommandStep
    exit_code: int | None
    """``None`` when the step was killed before reporting, e.g. on timeout."""

    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Whether the step succeeded."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def fatal(self) -> bool:
        """Whether this outcome should abort the job."""
        return not self.ok and self.step.on_failure is FailureAction.ABORT


@dataclass(frozen=True, slots=True)
class ResourceProjection:
    """What admitting a hypothetical job would mean for the machine. Dry-run only."""

    cores_requested: int
    cores_free: int
    cores_total: int
    would_start_immediately: bool
    blocking_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DryRunReport:
    """Everything ``--dry-run`` needs to print, and nothing was executed to produce it.

    Assembled from the same detect -> validate -> plan sequence a real submission runs
    (§6.11). No database row is written, no process is spawned, and no file in the case
    is touched.
    """

    workdir: Path
    solver: str
    solver_binary: str | None
    cores: int
    detections: Sequence[Detection] = field(default_factory=tuple)
    validation: ValidationReport = field(default_factory=ValidationReport)
    plan: ExecutionPlan | None = None
    projection: ResourceProjection | None = None
    suggested_tags: Sequence[str] = ()
    stdout_path: Path | None = None

    @property
    def would_submit(self) -> bool:
        """Whether a real submission would be accepted without an explicit override."""
        return self.validation.passed and self.plan is not None
