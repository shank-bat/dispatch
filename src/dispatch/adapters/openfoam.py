"""The OpenFOAM adapter.

Handles the thing that makes OpenFOAM tedious to run by hand: matching the case's
decomposition to the number of cores you actually want. The user asks for 20 cores on a
case decomposed for 8, and the adapter emits reconstruct, remove, re-decompose, run --
without the user ever typing ``reconstructPar``.

The one deliberate omission: **the case is not reconstructed after the solve.**
Reconstruction of a large case can take longer than the run itself, and post-processing
with ``paraFoam -builtin`` or ``foamToVTK`` on the decomposed case is the normal workflow.
The case is left exactly as the solver left it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from dispatch.adapters import foamdict, foamlog, gpuenv, mpi
from dispatch.adapters.base import (
    BaseAdapter,
    CaseContext,
    Progress,
    generic_failure_summary,
)
from dispatch.adapters.shellenv import capture, find_first
from dispatch.core.metadata import (
    CaseMetadata,
    FieldType,
    MetadataField,
    MetadataSpec,
    SpecRef,
)
from dispatch.core.models import Detection
from dispatch.core.plan import CommandStep, ExecutionPlan, FailureAction, StepKind
from dispatch.core.series import PlotData
from dispatch.core.validation import ReportBuilder, ValidationReport

__all__ = ["OpenFOAMAdapter"]

log = logging.getLogger(__name__)

PROCESSOR_DIR = re.compile(r"^processor(\d+)$")
TIME_LINE = re.compile(r"^Time = ([0-9.eE+-]+)\s*$", re.MULTILINE)

FOAM_FATAL = re.compile(r"^-->\s*FOAM FATAL (?:IO )?ERROR", re.MULTILINE)
RANK_PREFIX = re.compile(r"^\[\d+\]\s?")
"""``mpirun`` labels every line of a parallel run with its rank.

Without stripping these, nothing anchored to the start of a line ever matches in the
parallel case -- which is to say, in the case that matters, since a job large enough to
worry about is a job running on more than one core.
"""

END_OF_MESSAGE = re.compile(
    r"^(?:\[stack trace\]|#\d+\s|From\s|in file\s|FOAM (?:parallel run )?exiting)"
)
"""Where a fatal block stops being the message and starts being the C++ it came from."""

FATAL_BLOCK_LINES = 6
"""Enough for the banner, the message, and the dictionary path it complains about."""

BASHRC_CANDIDATES = [
    "/usr/lib/openfoam/openfoam*/etc/bashrc",
    "/opt/openfoam*/etc/bashrc",
    "/opt/OpenFOAM/OpenFOAM-*/etc/bashrc",
    "~/OpenFOAM/OpenFOAM-*/etc/bashrc",
    "/usr/share/openfoam/etc/bashrc",
]


class OpenFOAMAdapter(BaseAdapter):
    """Runs OpenFOAM cases, managing decomposition automatically."""

    name: ClassVar[str] = "openfoam"
    display_name: ClassVar[str] = "OpenFOAM"
    adapter_version: ClassVar[int] = 1
    log_name: ClassVar[str] = "log.foam"
    """The solver's output log, beside ``system/`` where a Foam user looks for it.

    ``log.foam`` rather than ``log.interFoam``: the application can change between runs of
    the same case, and a name that moves is a name nobody can tail from memory.
    """

    metadata_spec: ClassVar[MetadataSpec] = MetadataSpec(
        ref=SpecRef(adapter="openfoam", version=1),
        fields=(
            MetadataField("application", FieldType.STR, "Application", display_order=1),
            MetadataField("startTime", FieldType.FLOAT, "Start time", unit="s", display_order=2),
            MetadataField("endTime", FieldType.FLOAT, "End time", unit="s", display_order=3),
            MetadataField("deltaT", FieldType.FLOAT, "Time step", unit="s", display_order=4),
            MetadataField("writeInterval", FieldType.FLOAT, "Write interval", display_order=5),
            MetadataField("writeControl", FieldType.STR, "Write control", display_order=6),
            MetadataField("startFrom", FieldType.STR, "Start from", display_order=7),
            MetadataField("decomposition", FieldType.INT, "Subdomains", display_order=8),
            MetadataField("decomposition_method", FieldType.STR, "Decomposition", display_order=9),
            MetadataField("mesh_cells", FieldType.INT, "Mesh cells", display_order=10),
        ),
    )

    env_keys: ClassVar[Sequence[str]] = (
        "WM_PROJECT",
        "WM_PROJECT_VERSION",
        "WM_PROJECT_DIR",
        "WM_OPTIONS",
        "FOAM_USER_LIBBIN",
        "FOAM_USER_APPBIN",
        "FOAM_RUN",
        "MPI_ARCH_PATH",
        "WM_MPLIB",
    )

    # -- detection ---------------------------------------------------------------------

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        """Recognise a case by ``system/controlDict``.

        Confidence 0.95: that file means one thing and appears nowhere else. The
        application name is read from it with a real parser, not a regular expression,
        because a commented-out alternative solver in a ``controlDict`` is common and
        matching it would silently run the wrong one.
        """
        control = path / "system" / "controlDict"
        if not control.is_file():
            return None

        settings = foamdict.parse_file(control)
        application = settings.get("application")
        application = application if isinstance(application, str) else None

        subdomains = count_processor_dirs(path)
        label = f"OpenFOAM case: {application or 'unknown application'}"
        if subdomains:
            label += f", decomposed into {subdomains}"

        return Detection(
            solver=cls.name,
            confidence=0.95,
            solver_binary=application,
            label=label,
            entry=control,
            detail={"decomposition": subdomains},
        )

    # -- environment ------------------------------------------------------------------------

    def prepare_environment(self, ctx: CaseContext) -> Mapping[str, str]:
        """Source the OpenFOAM ``etc/bashrc`` once and reuse the result.

        Without this, no OpenFOAM binary is on the PATH and every validation reports the
        solver as missing. The path comes from ``[adapters.openfoam] bashrc``; when unset,
        the usual install locations are probed.
        """
        bashrc = self._bashrc()
        if bashrc is None:
            return ctx.env
        return capture(bashrc, base=dict(ctx.env))

    def _bashrc(self) -> Path | None:
        """Locate the OpenFOAM setup script."""
        configured = self.setting("bashrc")
        if configured:
            path = Path(str(configured)).expanduser()
            return path if path.exists() else None

        for pattern in BASHRC_CANDIDATES:
            expanded = Path(pattern).expanduser()
            matches = sorted(Path(expanded.anchor or "/").glob(str(expanded).lstrip("/")))
            found = find_first(matches)
            if found is not None:
                return found
        return None

    def solver_version(self, ctx: CaseContext) -> str | None:
        """Report the OpenFOAM build, e.g. ``OpenFOAM-v2312``."""
        project = ctx.env.get("WM_PROJECT")
        version = ctx.env.get("WM_PROJECT_VERSION")
        if project and version:
            return f"{project}-{version}"
        return version or None

    # -- validation ---------------------------------------------------------------------------

    def validate(self, ctx: CaseContext) -> ValidationReport:
        """Check that this is a runnable case, and explain what will happen to it."""
        builder = ReportBuilder()
        case = ctx.workdir

        for directory in ("system", "constant"):
            if not (case / directory).is_dir():
                builder.error(
                    f"{directory}/ is missing",
                    path=str(case / directory),
                    code=f"missing_{directory}",
                )

        for dictionary in ("controlDict", "fvSchemes", "fvSolution"):
            if not (case / "system" / dictionary).is_file():
                builder.error(
                    f"system/{dictionary} is missing",
                    path=str(case / "system" / dictionary),
                    code=f"missing_{dictionary.lower()}",
                )

        self._check_initial_conditions(ctx, builder)
        self._check_mesh(ctx, builder)
        self._check_application(ctx, builder)
        self._check_decomposition(ctx, builder)

        return builder.build()

    def _check_initial_conditions(self, ctx: CaseContext, builder: ReportBuilder) -> None:
        case = ctx.workdir
        if (case / "0").is_dir():
            return
        if latest_time_dir(case) is not None:
            builder.info("no 0/ directory, but the case has later time directories to start from")
            return
        if (case / "0.orig").is_dir():
            builder.error(
                "there is a 0.orig/ but no 0/ -- the case has not been set up yet",
                hint="Copy 0.orig to 0, or run the case's Allrun script first.",
                code="needs_setup",
            )
            return
        builder.error("no 0/ directory and no time directories", code="missing_initial")

    def _check_mesh(self, ctx: CaseContext, builder: ReportBuilder) -> None:
        """A case with no mesh cannot run, decomposed or not."""
        case = ctx.workdir
        serial = (case / "constant" / "polyMesh" / "owner").is_file()
        parallel = (case / "processor0" / "constant" / "polyMesh" / "owner").is_file()
        if serial or parallel:
            return
        builder.error(
            "no mesh found (constant/polyMesh or processor0/constant/polyMesh)",
            hint="Run blockMesh, snappyHexMesh, or import a mesh first.",
            code="missing_mesh",
        )

    def _check_application(self, ctx: CaseContext, builder: ReportBuilder) -> None:
        application = self._application(ctx)
        if not application:
            builder.error(
                "system/controlDict does not name an application",
                code="missing_application",
            )
            return
        if ctx.which(application) is None:
            hint = (
                "Set [adapters.openfoam] bashrc in the Dispatch configuration to your "
                "OpenFOAM etc/bashrc."
                if self._bashrc() is None
                else "Check that this solver is part of your OpenFOAM installation."
            )
            builder.error(
                f"the solver {application!r} is not on the PATH", hint=hint, code="solver_not_found"
            )

    def _check_decomposition(self, ctx: CaseContext, builder: ReportBuilder) -> None:
        """Explain the decomposition work the job will do before it does it."""
        existing = count_processor_dirs(ctx.workdir)
        wanted = ctx.cores

        if wanted == 1:
            if existing:
                builder.info(
                    f"the case is decomposed into {existing}; it will be reconstructed to "
                    "run in serial"
                )
            return

        if existing == wanted:
            builder.info(f"reusing the existing decomposition into {existing} subdomains")
            return

        if existing:
            builder.info(
                f"the existing decomposition is {existing} but {wanted} cores were "
                f"requested, so the case will be reconstructed and re-decomposed"
            )
        else:
            builder.info(f"the case will be decomposed into {wanted} subdomains")

        decompose_dict = ctx.workdir / "system" / "decomposeParDict"
        if not decompose_dict.is_file():
            builder.warning(
                "system/decomposeParDict is missing; one will be written using the scotch method",
                hint="Provide your own if you need a specific decomposition.",
                code="no_decomposepardict",
            )
        else:
            self._check_geometric_coefficients(decompose_dict, wanted, builder)

        if ctx.which("mpirun") is None:
            builder.error(
                "mpirun is not on the PATH, so this case cannot run in parallel",
                code="no_mpi",
            )
        mpi.check_slots(ctx, builder)

    def _check_geometric_coefficients(
        self, decompose_dict: Path, wanted: int, builder: ReportBuilder
    ) -> None:
        """Warn when a geometric decomposition's ``n`` vector will have to be rewritten.

        ``hierarchical`` and ``simple`` take an explicit ``n (nx ny nz)`` whose product is
        the subdomain count. Changing the core count without it makes ``decomposePar``
        fail -- so Dispatch fixes it, and says so rather than altering the user's
        decomposition silently.
        """
        method = (foamdict.read_value(decompose_dict, "method") or "").strip()
        if method not in foamdict.GEOMETRIC_METHODS:
            return

        current = foamdict.parse_file(decompose_dict)
        coeffs = current.get("coeffs") or current.get(f"{method}Coeffs")
        existing = _product_of_vector(coeffs.get("n") if isinstance(coeffs, dict) else None)
        if existing is None or existing == wanted:
            return

        x, y, z = foamdict.balanced_factors(wanted)
        builder.warning(
            f"the {method} decomposition is set to {existing} subdomains "
            f"(n {coeffs.get('n') if isinstance(coeffs, dict) else '?'}) but {wanted} cores "
            f"were requested, so n will be rewritten to ({x} {y} {z})",
            hint="Set your own n, or use the scotch method, if the split matters.",
            code="geometric_coeffs_rewritten",
        )

    # -- planning -------------------------------------------------------------------------------

    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        """Build the steps that bring the decomposition in line and run the solver.

        =========================  ==============================================
        Situation                  Steps
        =========================  ==============================================
        1 core, no processor dirs  solve
        N cores, N dirs exist      solve in parallel, reusing the decomposition
        N cores, M dirs, M != N    reconstruct, remove, decompose, solve
        N cores, no dirs           decompose, solve
        1 core, M dirs exist       reconstruct, solve in serial
        =========================  ==============================================
        """
        case = ctx.workdir
        # Heal a case whose cancellation never got to run its finalize -- a daemon killed
        # between the two would otherwise leave `stopAt writeNow` in place forever, and
        # every run from then on would stop at its first time step.
        _restore_stop_at(case)

        env = gpuenv.apply_gpu_visibility(dict(ctx.env), ctx)
        application = self._application(ctx) or "foamRun"
        existing = count_processor_dirs(case)
        wanted = max(1, ctx.cores)

        steps: list[CommandStep] = []

        if wanted == 1:
            if existing:
                steps.append(self._reconstruct(case, env))
            steps.append(
                CommandStep(
                    argv=[application],
                    cwd=case,
                    description=f"Running {application}",
                    kind=StepKind.SOLVE,
                    env=env,
                )
            )
            return ExecutionPlan(steps=tuple(steps))

        if existing and existing != wanted:
            steps.append(self._reconstruct(case, env))
            steps.append(
                CommandStep(
                    argv=["rm", "-rf", *[f"processor{i}" for i in range(existing)]],
                    cwd=case,
                    description=f"Removing {existing} old processor directories",
                    kind=StepKind.PREPARE,
                    env=env,
                )
            )

        if existing != wanted:
            self._ensure_decompose_dict(ctx, wanted)
            steps.append(
                CommandStep(
                    argv=["decomposePar", "-force"],
                    cwd=case,
                    description=f"Decomposing the case into {wanted} subdomains",
                    kind=StepKind.PREPARE,
                    env=env,
                )
            )

        steps.append(
            CommandStep(
                argv=mpi.launch_argv(wanted, application, "-parallel"),
                cwd=case,
                description=f"Running {application} on {wanted} cores",
                kind=StepKind.SOLVE,
                env=env,
            )
        )
        return ExecutionPlan(steps=tuple(steps))

    def _reconstruct(self, case: Path, env: Mapping[str, str]) -> CommandStep:
        """Reconstruct the latest time before changing the decomposition.

        Only ``-latestTime``: reconstructing every written time on a large case can take
        hours, and the latest is what a re-decomposition needs to preserve.

        ``WARN`` rather than ``ABORT`` because a case decomposed but never run has nothing
        to reconstruct, and ``reconstructPar`` reports that as a failure. Losing the job
        over it would be wrong.
        """
        return CommandStep(
            argv=["reconstructPar", "-latestTime"],
            cwd=case,
            description="Reconstructing the latest time before re-decomposing",
            kind=StepKind.PREPARE,
            env=dict(env),
            on_failure=FailureAction.WARN,
        )

    def _ensure_decompose_dict(self, ctx: CaseContext, subdomains: int) -> None:
        """Make ``decomposeParDict`` agree with the requested core count.

        An existing file is edited in place, preserving the user's method, comments, and
        formatting. Geometric methods also need their ``n (nx ny nz)`` vector rewritten,
        because its product *is* the subdomain count -- a stale one makes ``decomposePar``
        refuse to run, which is precisely the manual step this adapter exists to remove.
        """
        path = ctx.workdir / "system" / "decomposeParDict"
        if path.is_file():
            foamdict.set_decomposition(path, subdomains)
            return
        method = str(self.setting("decomposition_method", "scotch"))
        foamdict.write_decompose_dict(path, subdomains, method)
        log.info("Wrote %s for %d subdomains", path, subdomains)

    # -- metadata ---------------------------------------------------------------------------------

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        """Read the case settings worth remembering."""
        control = ctx.workdir / "system" / "controlDict"
        settings = foamdict.parse_file(control)
        decompose = foamdict.parse_file(ctx.workdir / "system" / "decomposeParDict")

        values: dict[str, object] = {
            "application": settings.get("application"),
            "startFrom": settings.get("startFrom"),
            "writeControl": settings.get("writeControl"),
            "decomposition": count_processor_dirs(ctx.workdir) or None,
            "decomposition_method": decompose.get("method"),
        }
        for key in ("startTime", "endTime", "deltaT", "writeInterval"):
            values[key] = _as_float(settings.get(key))

        cells = mesh_cell_count(ctx.workdir)
        if cells is not None:
            values["mesh_cells"] = cells

        return self.metadata_spec.build({k: v for k, v in values.items() if v is not None})

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        """Offer the application name as a tag. Never applied without confirmation."""
        application = self._application(ctx)
        return (application.lower(),) if application else ()

    def parse_progress(self, tail: str, ctx: CaseContext) -> Progress | None:
        """Read the current simulated time out of the solver's output."""
        matches = TIME_LINE.findall(tail)
        if not matches:
            return None
        try:
            current = float(matches[-1])
        except ValueError:
            return None

        total = foamdict.read_float(ctx.workdir / "system" / "controlDict", "endTime")
        return Progress(current=current, total=total, label="Time")

    def parse_series(self, text: str, ctx: CaseContext) -> PlotData:
        """Extract residuals, Courant numbers, and timings from the solver's log.

        Delegated to :mod:`~dispatch.adapters.foamlog` rather than written inline, because
        this is the one part of the adapter that reads a format instead of describing a
        command, and it is worth being able to test it against a page of real log text
        with no adapter, no context, and no case directory in sight.
        """
        return foamlog.parse_foam_log(text)

    def stop_gracefully(self, ctx: CaseContext) -> bool:
        """Ask the solver to write and stop at the end of the current step.

        Setting ``stopAt writeNow`` in ``controlDict`` is the correct way to stop an
        OpenFOAM run: the solver notices at the next time step and exits after writing, so
        the result is usable rather than truncated mid-write.

        The previous value is saved first, because this edit **must not outlive the
        cancellation**. ``stopAt`` is a property of the case, not of the run: left behind,
        it makes every later run of that case write once and exit at the first time step.
        That failure is silent and it looks like the case is broken -- the solver exits 0
        having done nothing, so nothing reports an error. :meth:`finalize` puts the
        original value back.
        """
        control = ctx.workdir / "system" / "controlDict"
        if not control.is_file():
            return False

        previous = foamdict.read_value(control, "stopAt")
        if not foamdict.set_value(control, "stopAt", "writeNow"):
            return False

        # Written after the edit lands, so a crash between the two leaves the case
        # untouched rather than pointing at a restore that never happened.
        _save_stop_at(ctx.workdir, previous or "endTime")
        log.info("Set stopAt writeNow in %s (was %s)", control, previous or "endTime")
        return True

    def finalize(self, ctx: CaseContext) -> None:
        """Undo the ``stopAt`` edit a cancellation made, if there was one.

        Runs after the solver exits however it exited, so a cancelled case is left as its
        owner wrote it.
        """
        _restore_stop_at(ctx.workdir)

    def explain_failure(self, tail: str, ctx: CaseContext) -> str | None:
        """Extract the reason an OpenFOAM run failed.

        OpenFOAM's fatal errors have a shape worth exploiting: a ``--> FOAM FATAL ERROR``
        banner, the message, then a ``From ...`` line and a stack trace. The message is the
        useful part and the stack trace is not, so the block is cut where the C++ begins.

        In parallel -- which is to say, in practice -- every rank reports the same failure
        at the same moment, and ``mpirun`` interleaves them into one stream with ``[N]``
        rank labels. So the labels come off first, and identical lines are collapsed:
        otherwise the answer to "why did it fail" is the same sentence twenty times, half
        of it spliced through the middle of the other half.

        Anything without a Foam banner -- a launcher refusing the job, a missing library, a
        segfault -- falls through to the generic reading, which handles those well.
        """
        clean = "\n".join(RANK_PREFIX.sub("", line) for line in tail.splitlines())

        match = FOAM_FATAL.search(clean)
        if match is None:
            return generic_failure_summary(clean)

        block: list[str] = []
        for line in clean[match.start() :].splitlines():
            stripped = line.strip()
            if END_OF_MESSAGE.match(stripped):
                break
            # Normalise before the duplicate check, not after: two ranks' banners differ
            # only in the build stamp being trimmed off, so comparing the raw lines lets
            # both through and prints the banner twice.
            normalised = _normalise_banner(stripped)
            if normalised and normalised not in block:
                block.append(normalised)
            if len(block) >= FATAL_BLOCK_LINES:
                break
        return "\n".join(block).strip() or generic_failure_summary(clean)

    # -- helpers -----------------------------------------------------------------------------------

    def _application(self, ctx: CaseContext) -> str | None:
        """The solver named in ``controlDict``."""
        return foamdict.read_value(ctx.workdir / "system" / "controlDict", "application")


def _normalise_banner(line: str) -> str:
    """Trim the build stamp off a fatal banner.

    ``--> FOAM FATAL ERROR: (openfoam-2406 patch=260127)`` says nothing the provenance
    record does not already say, and interleaved ranks routinely tear the parenthetical in
    half -- so the half-line becomes noise sitting directly above the actual message.
    """
    if not line.startswith("-->"):
        return line
    head, sep, _ = line.partition(" (")
    return head.rstrip() if sep else line


STOP_AT_BACKUP = "system/.dispatch-stopAt"
"""Where the pre-cancellation ``stopAt`` is parked while the solver winds down.

A file in the case rather than memory on the adapter: adapter instances are shared between
concurrent jobs, and the value has to survive a daemon restart to be worth saving at all.
"""


def _save_stop_at(case: Path, value: str) -> None:
    """Record the ``stopAt`` that was in force before a cancellation edited it."""
    try:
        (case / STOP_AT_BACKUP).write_text(value.strip(), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unwritable case directory
        log.warning("Could not save the original stopAt for %s: %s", case, exc)


def _restore_stop_at(case: Path) -> bool:
    """Put back the ``stopAt`` a cancellation replaced. Returns whether it did.

    A no-op when there is no backup, which is the overwhelmingly common path: only a
    cancelled job leaves one behind.
    """
    backup = case / STOP_AT_BACKUP
    try:
        value = backup.read_text(encoding="utf-8").strip()
    except OSError:
        return False

    control = case / "system" / "controlDict"
    if value and control.is_file():
        foamdict.set_value(control, "stopAt", value)
        log.info("Restored stopAt %s in %s after cancellation", value, control)
    backup.unlink(missing_ok=True)
    return True


def count_processor_dirs(case: Path) -> int:
    """Count ``processorN`` directories, verifying they are numbered contiguously.

    A gap means an interrupted decomposition or a partial delete, and reusing that would
    hand ``mpirun`` a case it cannot read. Reporting zero makes the adapter re-decompose,
    which is the repair.
    """
    try:
        found = {
            int(match.group(1))
            for child in case.iterdir()
            if child.is_dir() and (match := PROCESSOR_DIR.match(child.name))
        }
    except OSError:
        return 0
    if not found:
        return 0
    count = len(found)
    if found != set(range(count)):
        log.warning("%s has non-contiguous processor directories; ignoring them", case)
        return 0
    return count


def latest_time_dir(case: Path) -> Path | None:
    """The highest-numbered time directory, if any."""
    best: tuple[float, Path] | None = None
    try:
        children = list(case.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if value > 0 and (best is None or value > best[0]):
            best = (value, child)
    return best[1] if best else None


def mesh_cell_count(case: Path) -> int | None:
    """Read the cell count from the mesh's ``owner`` header.

    The ``note`` line in the FoamFile header carries ``nCells``, which is far cheaper than
    counting entries and works for both serial and decomposed meshes.
    """
    for candidate in (
        case / "constant" / "polyMesh" / "owner",
        case / "processor0" / "constant" / "polyMesh" / "owner",
    ):
        try:
            with candidate.open("rb") as handle:
                head = handle.read(2048).decode("utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"nCells:\s*(\d+)", head)
        if match:
            return int(match.group(1))
    return None


def _product_of_vector(raw: object) -> int | None:
    """Multiply out an OpenFOAM vector like ``( 3 3 1 )``.

    The dictionary reader keeps list syntax as literal text, which is enough here: this
    only needs the product, not a general list parser.
    """
    if not isinstance(raw, str):
        return None
    numbers = re.findall(r"\d+", raw)
    if not numbers:
        return None
    total = 1
    for number in numbers:
        total *= int(number)
    return total or None


def _as_float(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return float(value)
    except ValueError:
        return None
