"""The Basilisk adapter.

Basilisk is a *compiler*, not a solver binary: you write a ``.c`` file including Basilisk
headers, compile it with ``qcc``, and run the resulting executable. That makes it the most
interesting test of the adapter design -- and the answer is that compilation is simply a
preparation step, exactly like mesh decomposition.

**The scheduler required no changes to support a solver that must be built first.** The
executor already knows how to run preparation steps, log them, and fail the job if one
aborts. That is the concrete evidence that the boundary is in the right place.

Detection confidence is deliberately lower than the other adapters'. A ``.c`` file is weak
evidence compared with ``system/controlDict``, which is exactly why detections carry a
confidence at all.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from dispatch.adapters import gpuenv, mpi
from dispatch.adapters.base import BaseAdapter, CaseContext, Progress
from dispatch.core.metadata import (
    CaseMetadata,
    FieldType,
    MetadataField,
    MetadataSpec,
    SpecRef,
)
from dispatch.core.models import Detection
from dispatch.core.plan import CommandStep, ExecutionPlan, StepKind
from dispatch.core.validation import ReportBuilder, ValidationReport

__all__ = ["BasiliskAdapter"]

log = logging.getLogger(__name__)

BASILISK_INCLUDES = re.compile(
    r'#\s*include\s+[<"]('
    r"navier-stokes/|grid/|embed\.h|two-phase\.h|tension\.h|vof\.h|"
    r"axi\.h|utils\.h|run\.h|poisson\.h|diffusion\.h|saint-venant\.h|"
    r"tracer\.h|curvature\.h|view\.h|fractions\.h|reduced\.h"
    r')[^>"]*[>"]'
)
TIME_LINE = re.compile(r"\bt\s*=\s*([0-9.eE+-]+)")
MAX_SOURCE_SCAN_BYTES = 256 * 1024


class BasiliskAdapter(BaseAdapter):
    """Compiles and runs Basilisk simulations."""

    name: ClassVar[str] = "basilisk"
    display_name: ClassVar[str] = "Basilisk"
    adapter_version: ClassVar[int] = 1
    log_name: ClassVar[str] = "log.basilisk"
    """The simulation's output log, beside its source file."""

    metadata_spec: ClassVar[MetadataSpec] = MetadataSpec(
        ref=SpecRef(adapter="basilisk", version=1),
        fields=(
            MetadataField("source", FieldType.PATH, "Source file", display_order=1),
            MetadataField("executable", FieldType.STR, "Executable", display_order=2),
            MetadataField("grid", FieldType.STR, "Grid", display_order=3),
            MetadataField("dimension", FieldType.INT, "Dimensions", display_order=4),
            MetadataField("parallel", FieldType.BOOL, "MPI", display_order=5),
            MetadataField("compile_flags", FieldType.STR, "Compiler flags", display_order=6),
        ),
    )

    env_keys: ClassVar[Sequence[str]] = ("BASILISK", "CC99", "OPENGLIBS", "MPI_ARCH_PATH")

    # -- detection ---------------------------------------------------------------------

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        """Recognise a project by a ``.c`` file that includes Basilisk headers.

        Confidence 0.70. ``.c`` files are common and this evidence is genuinely weaker
        than the other adapters', so an ambiguous directory prompts the user rather than
        guessing -- which is the correct outcome.
        """
        sources = find_sources(path)
        if not sources:
            return None

        chosen = sources[0]
        label = f"Basilisk project: {chosen.name}"
        if len(sources) > 1:
            label += f", {len(sources)} candidate sources"

        return Detection(
            solver=cls.name,
            confidence=0.70,
            solver_binary="qcc",
            label=label,
            entry=chosen,
            detail={"sources": [str(s) for s in sources]},
        )

    # -- validation --------------------------------------------------------------------------

    def validate(self, ctx: CaseContext) -> ValidationReport:
        """Check that there is exactly one source, and that ``qcc`` can build it."""
        builder = ReportBuilder()
        sources = find_sources(ctx.workdir)

        if not sources:
            builder.error(
                "no Basilisk source file found",
                hint="A Basilisk source includes headers such as navier-stokes/centered.h.",
                code="missing_source",
            )
            return builder.build()

        source = self._source(ctx)
        if len(sources) > 1 and ctx.entry is None:
            names = ", ".join(sorted(s.name for s in sources))
            builder.error(
                f"several Basilisk sources are present ({names}); Dispatch cannot tell "
                "which to build",
                hint="Keep one source per directory, or submit with the file selected.",
                code="ambiguous_source",
            )
            return builder.build()

        if source is not None and len(sources) > 1:
            builder.info(f"building {source.name}")

        if ctx.which("qcc") is None:
            builder.error(
                "qcc is not on the PATH",
                hint="Add $BASILISK to your PATH, or set [adapters.basilisk] qcc in the "
                "Dispatch configuration.",
                code="qcc_not_found",
            )
        if not ctx.env.get("BASILISK"):
            builder.warning(
                "the BASILISK environment variable is not set",
                hint="qcc usually needs it to locate its headers.",
                code="basilisk_unset",
            )
        if ctx.cores > 1 and ctx.which("mpirun") is None:
            builder.error(
                "mpirun is not on the PATH, so this case cannot run in parallel",
                code="no_mpi",
            )
        mpi.check_slots(ctx, builder)
        return builder.build()

    # -- planning -----------------------------------------------------------------------------

    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        """Compile, then run.

        The compile is an ordinary preparation step. If it fails the job fails before the
        solver ever starts, and the compiler's output is in the step transcript -- all of
        which the executor already did for mesh decomposition.
        """
        source = self._source(ctx)
        if source is None:
            raise FileNotFoundError(f"No Basilisk source file in {ctx.workdir}")

        executable = source.stem
        env = gpuenv.apply_gpu_visibility(dict(ctx.env), ctx)
        qcc = str(self.setting("qcc", "qcc"))
        parallel = ctx.cores > 1

        compile_argv = [qcc, *self._compile_flags(ctx, parallel), "-o", executable, source.name]
        compile_argv.extend(str(flag) for flag in self.setting("libs", ["-lm"]))

        steps: list[CommandStep] = [
            CommandStep(
                argv=compile_argv,
                cwd=ctx.workdir,
                description=f"Compiling {source.name} with qcc",
                kind=StepKind.PREPARE,
                env=env,
                timeout_s=float(self.setting("compile_timeout_s", 600)),
            )
        ]

        if parallel:
            run_argv = mpi.launch_argv(ctx.cores, f"./{executable}")
            description = f"Running {executable} on {ctx.cores} cores"
        else:
            run_argv = [f"./{executable}"]
            description = f"Running {executable}"

        steps.append(
            CommandStep(
                argv=run_argv,
                cwd=ctx.workdir,
                description=description,
                kind=StepKind.SOLVE,
                env=env,
            )
        )
        return ExecutionPlan(steps=tuple(steps))

    def _compile_flags(self, ctx: CaseContext, parallel: bool) -> list[str]:
        """Assemble qcc flags, adding MPI only when the job is parallel.

        ``-D_MPI=N`` is how Basilisk enables its MPI paths, and it must match the rank
        count the job will actually launch with.
        """
        configured = self.setting("flags")
        flags = (
            [str(flag) for flag in configured]
            if isinstance(configured, (list, tuple))
            else ["-O2", "-Wall"]
        )
        if parallel:
            flags.append(f"-D_MPI={ctx.cores}")
        return flags

    # -- metadata -------------------------------------------------------------------------------

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        """Record what was built, and how."""
        source = self._source(ctx)
        if source is None:
            return CaseMetadata.empty(self.name)

        parallel = ctx.cores > 1
        values: dict[str, object] = {
            "source": source.name,
            "executable": source.stem,
            "parallel": parallel,
            "compile_flags": " ".join(self._compile_flags(ctx, parallel)),
        }
        grid, dimension = _grid_and_dimension(source)
        if grid:
            values["grid"] = grid
        if dimension:
            values["dimension"] = dimension
        return self.metadata_spec.build(values)

    def solver_version(self, ctx: CaseContext) -> str | None:
        """Report the qcc build. Basilisk has no release version, so the path is recorded."""
        qcc = ctx.which(str(self.setting("qcc", "qcc")))
        if qcc is None:
            return None
        try:
            result = subprocess.run(
                [qcc, "--version"],
                capture_output=True,
                timeout=10,
                check=False,
                env=dict(ctx.env),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        first = (result.stdout + result.stderr).decode(errors="replace").strip().splitlines()
        version = first[0].strip() if first else ""
        basilisk = ctx.env.get("BASILISK", "")
        if version and basilisk:
            return f"{version} (BASILISK={basilisk})"
        return version or (f"BASILISK={basilisk}" if basilisk else None)

    def parse_progress(self, tail: str, ctx: CaseContext) -> Progress | None:
        """Try to read simulated time from output that is entirely user-defined.

        Basilisk prints whatever the case author chose to print. ``t = <float>`` is the
        overwhelmingly common convention, so it is worth trying -- and returning ``None``
        is a perfectly good answer, after which the interface shows elapsed time instead.
        """
        matches = TIME_LINE.findall(tail)
        if not matches:
            return None
        try:
            return Progress(current=float(matches[-1]), total=None, label="t")
        except ValueError:
            return None

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        """Offer the source name as a tag."""
        source = self._source(ctx)
        return (source.stem.lower(),) if source else ()

    # -- helpers -----------------------------------------------------------------------------------

    def _source(self, ctx: CaseContext) -> Path | None:
        """The source file to build, preferring the one detection selected."""
        if ctx.entry is not None and ctx.entry.is_file():
            return ctx.entry
        sources = find_sources(ctx.workdir)
        return sources[0] if sources else None


def find_sources(path: Path) -> list[Path]:
    """Every ``.c`` file in ``path`` that includes a Basilisk header.

    Files under ``_``-prefixed directories are skipped: qcc writes its generated
    intermediate sources there, and offering to compile the output of a previous compile
    would be a confusing loop.
    """
    try:
        candidates = sorted(child for child in path.glob("*.c") if child.is_file())
    except OSError:
        return []
    return [
        candidate
        for candidate in candidates
        if not candidate.name.startswith("_") and looks_like_basilisk(candidate)
    ]


def looks_like_basilisk(path: Path) -> bool:
    """Whether a C source includes Basilisk headers."""
    try:
        with path.open("rb") as handle:
            head = handle.read(MAX_SOURCE_SCAN_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return False
    return bool(BASILISK_INCLUDES.search(head))


def _grid_and_dimension(source: Path) -> tuple[str | None, int | None]:
    """Extract the grid type and dimensionality from a source's includes."""
    try:
        text = source.read_text(encoding="utf-8", errors="replace")[:MAX_SOURCE_SCAN_BYTES]
    except OSError:
        return None, None

    grid = None
    match = re.search(r'#\s*include\s+[<"]grid/([\w-]+)\.h[>"]', text)
    if match:
        grid = match.group(1)

    dimension = None
    if re.search(r"^\s*#\s*define\s+dimension\s+3", text, re.MULTILINE):
        dimension = 3
    elif re.search(r'#\s*include\s+[<"]axi\.h[>"]', text):
        dimension = 2
    elif grid in ("octree", "multigrid3D"):
        dimension = 3
    elif grid in ("quadtree", "multigrid", "cartesian", "bitree"):
        dimension = 2
    return grid, dimension
