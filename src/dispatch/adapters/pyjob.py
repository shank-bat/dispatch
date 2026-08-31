"""Shared machinery for Python-based workloads: ML training and PINN solving.

Two adapters use this, and they differ in almost nothing: what makes a directory theirs,
and what their log is called. Everything else -- finding the interpreter, resolving the
entrypoint, building a one-command plan, reading loss curves back out of the output -- is
identical, and duplicating it into two files would guarantee the two drift.

**Detection is the hard part, and the answer is to ask.** A directory with a
``pyproject.toml`` and a ``main.py`` is a Python project; it is not evidence of a training
run, and an adapter that claimed it would attach itself to every repository the user ever
browsed to. So the primary signal is an explicit declaration the user writes once::

    # dispatch.toml, in the project directory
    [job]
    adapter    = "pinn"
    entrypoint = "train_burgers.py"
    args       = ["--config", "configs/burgers.yaml"]
    venv       = ".venv"

That file is unambiguous, it is version-controlled next to the code it describes, it
answers "which script" -- which no heuristic can -- and it extends the convention
Dispatch already had rather than inventing a parallel one: it is the same relationship
``system/controlDict`` has to an OpenFOAM case, chosen by the user instead of by the
solver. The heuristics each adapter adds on top are strictly secondary and deliberately
narrow.

See ``docs/adapters.md`` for the full file format.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final

from dispatch.adapters import gpuenv
from dispatch.adapters.base import BaseAdapter, CaseContext, Progress
from dispatch.core.plan import CommandStep, ExecutionPlan, StepKind
from dispatch.core.series import PlotData, series_from_records
from dispatch.core.validation import ReportBuilder, ValidationReport

__all__ = ["DECLARATION_FILE", "Declaration", "PythonJobAdapter", "read_declaration"]

log = logging.getLogger(__name__)

DECLARATION_FILE: Final = "dispatch.toml"
"""The explicit "this is a Dispatch job, and here is how to run it" file."""

MAX_DECLARATION_BYTES: Final = 64 * 1024
"""A declaration is a dozen lines. Anything larger is not one, and is not read."""

ENTRYPOINT_CANDIDATES: Final[tuple[str, ...]] = ("train.py", "main.py", "run.py")
"""Scripts looked for when a declaration does not name one.

Ordered by how strongly each implies "this is the thing to run". ``train.py`` is a
near-certain training entrypoint; ``main.py`` is a Python convention that means nothing in
particular, which is why detection never rests on it alone -- see each adapter's
``detect``.
"""

VENV_CANDIDATES: Final[tuple[str, ...]] = (".venv", "venv", "env")


@dataclass(frozen=True, slots=True)
class Declaration:
    """A parsed ``dispatch.toml``.

    Attributes:
        adapter: Which adapter the user says owns this directory. Empty when unstated,
            which leaves detection to the heuristics.
        entrypoint: Script or module to run. Empty means "work it out".
        args: Arguments appended after the entrypoint, verbatim.
        python: Interpreter to use when no virtual environment is found.
        venv: Virtual environment directory, relative to the project.
        module: Run the entrypoint with ``python -m`` rather than as a file path.
        framework: What this project is built on, recorded as metadata. Free text.
        source: The file this came from.
    """

    adapter: str = ""
    entrypoint: str = ""
    args: Sequence[str] = ()
    python: str = ""
    venv: str = ""
    module: bool = False
    framework: str = ""
    source: Path | None = None

    @property
    def declared(self) -> bool:
        """Whether the file actually named an adapter."""
        return bool(self.adapter)


def read_declaration(path: Path) -> Declaration | None:
    """Read ``dispatch.toml`` from a project directory.

    Never raises. A malformed declaration is logged and treated as absent: this runs
    during detection, against every directory the user browses to, and a syntax error in
    one project's file must not make the browser unusable.

    Returns:
        The declaration, or ``None`` when there is no readable one.
    """
    candidate = path / DECLARATION_FILE
    try:
        if not candidate.is_file() or candidate.stat().st_size > MAX_DECLARATION_BYTES:
            return None
        with candidate.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        log.debug("Ignoring %s: %s", candidate, exc)
        return None

    job = data.get("job")
    if not isinstance(job, Mapping):
        return None

    raw_args = job.get("args")
    args = tuple(str(item) for item in raw_args) if isinstance(raw_args, list) else ()
    return Declaration(
        adapter=str(job.get("adapter") or "").strip().lower(),
        entrypoint=str(job.get("entrypoint") or "").strip(),
        args=args,
        python=str(job.get("python") or "").strip(),
        venv=str(job.get("venv") or "").strip(),
        module=bool(job.get("module", False)),
        framework=str(job.get("framework") or "").strip(),
        source=candidate,
    )


# -- log reading ----------------------------------------------------------------------

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

METRIC: Final = re.compile(rf"\b([A-Za-z_][\w.\-/]*)\s*[=:]\s*({_NUMBER})\b")
"""``loss: 0.02``, ``val_loss=1.4e-3``, ``lr = 0.001``.

Requires a separator. ``loss 0.02`` is not matched, because ``Epoch 3`` and ``GPU 0`` have
the same shape and a parser that accepted them would invent series out of prose.
"""

COUNTER: Final = re.compile(
    r"\b(epoch|step|iter|iteration)\b\s*[=:]?\s*(\d+)\s*(?:/\s*(\d+))?", re.IGNORECASE
)
"""``Epoch 3/50``, ``step=1000``, ``iteration: 42``.

The counter is what makes a record boundary, so it is matched separately and more
loosely than the metrics: every training loop in existence prints one of these words.
"""

COUNTER_KEYS: Final = frozenset({"epoch", "step", "iter", "iteration"})

LABELS: Final[dict[str, str]] = {
    "epoch": "Epoch",
    "step": "Step",
    "iteration": "Iteration",
    "loss": "loss",
}


def parse_metric_log(text: str, *, truncated: bool = False) -> PlotData:
    """Extract training curves from a Python job's output.

    Deliberately generic, because there is no standard: every project prints its own
    metrics, and the one thing they agree on is a counter and some ``name: number`` pairs.
    A line carrying a counter opens a record; metrics on that line and the lines after it
    belong to it, until the next counter.

    This is a heuristic, and it is allowed to be, because of where it sits: it runs only
    when somebody presses ``p``, it feeds a chart, and **nothing about a job's state
    depends on it** (§13.25). Misreading a line costs a wrong point on a plot, never a
    misreported run. A log it makes no sense of yields no series at all, and the interface
    says so rather than drawing an empty axis.

    Args:
        text: The job's output, or its end.
        truncated: Whether only the end was read.

    Returns:
        Whatever series the log actually contained.
    """
    records: list[dict[str, float]] = []
    metrics: list[str] = []
    counters: list[str] = []
    current: dict[str, float] | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        counter = COUNTER.search(line)
        if counter is not None:
            key = counter.group(1).lower()
            if key not in counters:
                counters.append(key)
            current = {key: float(counter.group(2))}
            records.append(current)

        if current is None:
            continue

        for name, value in METRIC.findall(line):
            key = name.lower()
            if key in COUNTER_KEYS:
                # Already captured as the counter; recording it twice would offer the
                # same quantity under two names in the axis list.
                continue
            number = _number(value)
            if number is None:
                continue
            if key not in metrics:
                metrics.append(key)
            current[key] = number

    if not records:
        return PlotData(series=(), samples=0, truncated=truncated)

    series = series_from_records(
        records,
        labels={key: LABELS.get(key, key) for key in [*counters, *metrics]},
        axes=counters,
        order=(*counters, "loss", *metrics),
    )
    return PlotData(series=series, samples=len(records), truncated=truncated)


def _number(raw: str) -> float | None:
    """Parse a float, dropping the values that would poison an axis."""
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None


# -- the shared adapter ----------------------------------------------------------------


class PythonJobAdapter(BaseAdapter):
    """Base for adapters that run a Python program in a project directory.

    Subclasses provide :meth:`detect` and their identity; everything below is shared.
    """

    entrypoint_candidates: ClassVar[Sequence[str]] = ENTRYPOINT_CANDIDATES
    """Scripts to look for when the declaration does not name one."""

    # -- resolution -------------------------------------------------------------------

    def declaration(self, ctx: CaseContext) -> Declaration:
        """The project's declaration, or an empty one."""
        return read_declaration(ctx.workdir) or Declaration()

    def entrypoint(self, ctx: CaseContext) -> tuple[str, bool] | None:
        """What to run, and whether it is a module rather than a file.

        Declaration first, then the detection's own choice, then the conventional names.
        The declaration is checked first precisely so that a project whose training script
        is called ``fit_operator.py`` needs no guessing at all -- which is the whole reason
        the file exists.

        Returns:
            ``(target, is_module)``, or ``None`` when nothing plausible was found.
        """
        declared = self.declaration(ctx)
        if declared.entrypoint:
            return declared.entrypoint, declared.module
        if ctx.entry is not None and ctx.entry.is_file():
            return ctx.entry.name, False
        for name in self.entrypoint_candidates:
            if (ctx.workdir / name).is_file():
                return name, False
        return None

    def interpreter(self, ctx: CaseContext) -> str:
        """The Python to run with.

        A virtual environment in the project wins, because a project that ships one means
        it. Then the declaration's ``python``, then the adapter's configured default, then
        the interpreter Dispatch itself is running under -- which at least exists.
        """
        declared = self.declaration(ctx)
        for name in ([declared.venv] if declared.venv else VENV_CANDIDATES):
            candidate = ctx.workdir / name / "bin" / "python"
            if candidate.is_file():
                return str(candidate)
        for choice in (declared.python, str(self.setting("python", "") or "")):
            if choice:
                return choice
        return str(self.setting("fallback_python", "") or sys.executable)

    # -- validation --------------------------------------------------------------------

    def validate(self, ctx: CaseContext) -> ValidationReport:
        """Check that there is something to run, and something to run it with."""
        builder = ReportBuilder()

        target = self.entrypoint(ctx)
        if target is None:
            builder.error(
                "no entrypoint found in this project",
                hint=(
                    f"Name one in {DECLARATION_FILE} as [job] entrypoint = \"train.py\", "
                    f"or add one of: {', '.join(self.entrypoint_candidates)}."
                ),
                code="missing_entrypoint",
            )
        elif not target[1] and not (ctx.workdir / target[0]).is_file():
            builder.error(
                f"the declared entrypoint {target[0]!r} does not exist",
                path=str(ctx.workdir / target[0]),
                code="missing_entrypoint",
            )

        interpreter = self.interpreter(ctx)
        if not Path(interpreter).is_file() and shutil.which(interpreter) is None:
            builder.error(
                f"the interpreter {interpreter!r} was not found",
                hint=(
                    "Create a virtual environment in the project, or set [job] python "
                    f"in {DECLARATION_FILE}."
                ),
                code="missing_interpreter",
            )

        self._check_resources(ctx, builder)
        return builder.build()

    def _check_resources(self, ctx: CaseContext, builder: ReportBuilder) -> None:
        """Say plainly what the job will and will not be allowed to touch.

        Both directions are worth a note. A GPU job on a machine with none never starts,
        and the scheduler already says so; a CPU job that expected a GPU starts happily
        and trains a hundred times too slowly, which is much harder to notice.
        """
        if ctx.uses_gpu:
            builder.info(f"{ctx.gpus} GPU(s) reserved for this job")
            return
        builder.info(
            "no GPU was requested, so this job runs on CPU only: "
            f"{gpuenv.GPU_VISIBILITY_VARS[0]} is set empty for it",
            hint="Submit with --resource gpu --gpus 1 if it should train on a GPU.",
            code="cpu_only",
        )

    # -- planning ----------------------------------------------------------------------

    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        """One command: the interpreter, the entrypoint, and the declared arguments.

        No preparation step. Installing dependencies is deliberately not Dispatch's job --
        a scheduler that silently ran ``pip install`` into a user's environment as a side
        effect of queueing a run would be doing something nobody asked for, and it would
        do it hours later when the job was finally admitted.
        """
        target = self.entrypoint(ctx)
        if target is None:
            raise FileNotFoundError(
                f"No entrypoint found in {ctx.workdir}; name one in {DECLARATION_FILE}"
            )
        name, is_module = target
        declared = self.declaration(ctx)

        argv = [self.interpreter(ctx)]
        argv.extend(["-m", name] if is_module else [name])
        argv.extend(declared.args)

        env = gpuenv.apply_gpu_visibility(dict(ctx.env), ctx)
        env.setdefault("PYTHONUNBUFFERED", "1")
        """Without it a training script's prints sit in a pipe buffer for minutes, and the
        live log viewer -- and the progress and plot parsers that read the same file --
        show nothing at all until it flushes."""

        return ExecutionPlan(
            steps=(
                CommandStep(
                    argv=argv,
                    cwd=ctx.workdir,
                    description=self.describe_run(ctx, name),
                    kind=StepKind.SOLVE,
                    env=env,
                ),
            )
        )

    def describe_run(self, ctx: CaseContext, entrypoint: str) -> str:
        """One line naming what will run and on what."""
        where = f"{ctx.gpus} GPU(s)" if ctx.uses_gpu else f"{ctx.cores} core(s)"
        return f"Running {entrypoint} on {where}"

    # -- reading the output -------------------------------------------------------------

    def parse_progress(self, tail: str, ctx: CaseContext) -> Progress | None:
        """Read the current epoch or step out of the log tail."""
        matches = COUNTER.findall(tail)
        if not matches:
            return None
        label, current, total = matches[-1]
        try:
            value = float(current)
        except ValueError:
            return None
        try:
            target = float(total) if total else None
        except ValueError:
            target = None
        return Progress(current=value, total=target, label=label.capitalize())

    def parse_series(self, text: str, ctx: CaseContext) -> PlotData:
        """Extract training curves. See :func:`parse_metric_log`."""
        return parse_metric_log(text)

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        """Offer the framework, when the project declared one."""
        framework = self.declaration(ctx).framework.lower()
        return (framework,) if framework else ()

    # -- helpers -------------------------------------------------------------------------

    def base_metadata(self, ctx: CaseContext) -> dict[str, Any]:
        """The metadata fields every Python job has, for subclasses to extend."""
        declared = self.declaration(ctx)
        target = self.entrypoint(ctx)
        values: dict[str, Any] = {
            "gpus": ctx.gpus,
            "python": self.interpreter(ctx),
        }
        if target is not None:
            values["entrypoint"] = target[0]
        if declared.args:
            values["args"] = " ".join(declared.args)
        if declared.framework:
            values["framework"] = declared.framework
        return values


@dataclass(frozen=True, slots=True)
class ImportScan:
    """What a shallow look at a project's Python files and manifests turned up.

    Attributes:
        modules: Top-level module names seen in ``import`` statements.
        requirements: Distribution names seen in requirement and project manifests.
    """

    modules: frozenset[str] = field(default_factory=frozenset)
    requirements: frozenset[str] = field(default_factory=frozenset)

    def mentions(self, names: Sequence[str]) -> str | None:
        """The first of ``names`` this project references, if any."""
        for name in names:
            if name in self.modules or name in self.requirements:
                return name
        return None


IMPORT_LINE: Final = re.compile(
    r"^\s*(?:import\s+([\w.]+)|from\s+([\w.]+)\s+import\b)", re.MULTILINE
)
REQUIREMENT_LINE: Final = re.compile(r"^\s*([A-Za-z][\w.\-]*)", re.MULTILINE)

MANIFEST_FILES: Final[frozenset[str]] = frozenset(
    {"requirements.txt", "requirements-dev.txt", "environment.yml", "pyproject.toml"}
)
"""Files that list what a project depends on. Scanned for framework names, not parsed.

A regex over line starts, not a TOML or YAML reader: this needs to answer "does the word
``deepxde`` appear as a dependency", and a parser for four file formats would be a great
deal of code for a question a substring answers.
"""

MAX_SCAN_FILES: Final = 40
MAX_SCAN_BYTES: Final = 128 * 1024
"""Bounds on the scan. Detection runs on every directory the user browses to, so it reads
the top of a few dozen files and stops -- never a recursive walk, never a whole file."""


def scan_project(path: Path) -> ImportScan:
    """Look for framework evidence in a project's top-level files.

    Non-recursive and bounded: this is a detection helper, and detection must stay cheap
    enough to run against every adapter for every directory in a file browser.
    """
    modules: set[str] = set()
    requirements: set[str] = set()

    try:
        children = sorted(path.iterdir())
    except OSError:
        return ImportScan()

    scanned = 0
    for child in children:
        if scanned >= MAX_SCAN_FILES or not child.is_file():
            continue
        if child.suffix == ".py":
            scanned += 1
            for direct, indirect in IMPORT_LINE.findall(_head(child)):
                name = (direct or indirect).split(".")[0]
                if name:
                    modules.add(name.lower())
        elif child.name in MANIFEST_FILES:
            scanned += 1
            requirements |= {name.lower() for name in REQUIREMENT_LINE.findall(_head(child))}

    return ImportScan(modules=frozenset(modules), requirements=frozenset(requirements))


def _head(path: Path) -> str:
    """The first :data:`MAX_SCAN_BYTES` of a file, decoded leniently."""
    try:
        with path.open("rb") as handle:
            return handle.read(MAX_SCAN_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""
