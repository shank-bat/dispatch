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
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from dispatch.adapters import (
    foamcoeffs,
    foamdict,
    foamlog,
    foammesh,
    gpuenv,
    mpi,
    paraview,
)
from dispatch.adapters.base import (
    BaseAdapter,
    CaseContext,
    Progress,
    generic_failure_summary,
)
from dispatch.adapters.shellenv import capture, find_first
from dispatch.core.caseinfo import CaseReport
from dispatch.core.caseinfo import ReportBuilder as InfoBuilder
from dispatch.core.errors import ValidationError
from dispatch.core.geometry import CaseGeometry
from dispatch.core.metadata import (
    CaseMetadata,
    FieldType,
    MetadataField,
    MetadataSpec,
    SpecRef,
)
from dispatch.core.models import Detection
from dispatch.core.plan import CommandStep, ExecutionPlan, FailureAction, StepKind
from dispatch.core.series import Dataset, PlotData
from dispatch.core.validation import ReportBuilder, ValidationReport
from dispatch.core.visual import VisualKind, VisualPlan, VisualRequest

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

    case_markers: ClassVar[Sequence[str]] = ("system/controlDict",)
    renderer: ClassVar[str] = "ParaView (pvbatch)"
    """One stat. The same file :meth:`detect` then parses for the application."""

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
        if not ctx.dry_run:
            # Heal a case whose cancellation never got to run its finalize -- a daemon
            # killed between the two would otherwise leave `stopAt writeNow` in place
            # forever, and every run from then on would stop at its first time step.
            _restore_stop_at(case)
            _restore_start_from(case)

        latest = latest_written_time(case) if ctx.resume else None

        if ctx.resume and not ctx.dry_run:
            # Continuing rather than starting: either the machine went down under this run
            # (§6.7) or the case was explicitly asked to pick up where it stopped (§6.12).
            # OpenFOAM's own mechanism for both is `startFrom latestTime`, which makes the
            # solver read the highest time written in the case -- so the restart point comes
            # from the case's files, not from anything Dispatch believes about it. The
            # previous value is parked first and put back by `finalize`, exactly as the
            # graceful-stop edit is: it steers this run only, and a `startFrom` left behind
            # would silently change where every later run of the case begins.
            #
            # Skipped under `dry_run` precisely *because* `finalize` undoes it: a preview
            # never reaches `finalize`, so the edit would outlive the preview.
            _request_latest_time(case)

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
                    description=f"Running {application}{_from_clause(ctx, latest)}",
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
            if not ctx.dry_run:
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
                description=(
                    f"Running {application} on "
                    f"{wanted} {'thread' if ctx.counts_threads else 'core'}"
                    f"{'' if wanted == 1 else 's'}{_from_clause(ctx, latest)}"
                ),
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

    def case_datasets(self, ctx: CaseContext) -> Sequence[Dataset]:
        """Force coefficients, when the case has a ``forceCoeffs`` function object.

        Read from ``postProcessing/`` rather than the log, which is why it is a dataset of
        its own: the function object writes on its own schedule, so its rows do not line up
        with the solver's time steps and must not be paired with them by position.

        Finding the file is the substance -- see :mod:`~dispatch.adapters.foamcoeffs`. The
        path is searched and ranked by modification time, because the directory is named by
        the user, the time directory is named by the restart, and the filename changed
        spelling in v2012.
        """
        found = foamcoeffs.find_coefficient_file(ctx.workdir)
        if found is None:
            return ()
        text, truncated = _read_tail(found, foamcoeffs.MAX_BYTES)
        data = foamcoeffs.parse_coefficients(text, truncated=truncated)
        if not data:
            return ()
        return (
            Dataset(
                key="forceCoeffs",
                label="Force coefficients",
                data=data,
                source=str(found),
            ),
        )

    def geometry(self, ctx: CaseContext) -> CaseGeometry:
        """Whether the case is planar, and how big it is.

        Both read from the case's own files -- see :mod:`~dispatch.adapters.foammesh`. The
        planarity is exact rather than inferred: OpenFOAM has no 2-D meshes, so a planar case
        declares itself by giving its front and back faces the ``empty`` patch type, which is
        how the user tells the solver to skip that direction.
        """
        return foammesh.read_geometry(ctx.workdir)

    def describe_case(self, ctx: CaseContext) -> CaseReport:
        """Everything worth reading about the case, grouped for a human (§9.8).

        Reads only headers and small dictionaries, so it is cheap enough for a keypress on a
        directory being browsed. Every section tolerates its source being absent, because the
        most common reason to open this page is that something about the case is incomplete.
        """
        case = ctx.workdir
        control = case / "system" / "controlDict"
        settings = foamdict.parse_file(control)
        builder = InfoBuilder(case.name or str(case), self.name)

        # -- solver -------------------------------------------------------------------
        builder.section("solver")
        application = settings.get("application")
        builder.field(
            "Application",
            application if isinstance(application, str) else None,
            important=True,
        )
        builder.field("Case", str(case))
        if not control.is_file():
            builder.warn("system/controlDict is missing: this is not a runnable case yet")

        # -- time ---------------------------------------------------------------------
        builder.section("time")
        builder.field("Start from", settings.get("startFrom"))
        builder.field("Start time", _fmt(_as_float(settings.get("startTime"))), note="s")
        builder.field(
            "End time", _fmt(_as_float(settings.get("endTime"))), note="s", important=True
        )
        builder.field("Time step", _fmt(_as_float(settings.get("deltaT"))), note="s")
        builder.field("Stop at", settings.get("stopAt"))
        latest = latest_written_time(case)
        builder.field(
            "Latest written",
            _fmt(latest),
            note="continue from here" if latest is not None else "",
            important=latest is not None,
        )
        if latest is None:
            builder.field("Latest written", "nothing beyond the initial condition")

        adjustable = settings.get("adjustTimeStep")
        builder.field("Adjustable step", adjustable)
        builder.field("Max Courant", _fmt(_as_float(settings.get("maxCo"))))

        # -- output -------------------------------------------------------------------
        builder.section("output")
        builder.field("Write control", settings.get("writeControl"))
        builder.field("Write interval", _fmt(_as_float(settings.get("writeInterval"))))
        builder.field("Purge write", settings.get("purgeWrite"))
        builder.field("Format", settings.get("writeFormat"))
        times = _written_times(case)
        if times:
            builder.field(
                "Times written",
                len(times),
                note=f"{_fmt(times[0])} … {_fmt(times[-1])}" if len(times) > 1 else "",
            )

        # -- mesh ---------------------------------------------------------------------
        counts = foammesh.mesh_counts(case)
        builder.section(
            "mesh", missing="" if counts else "no mesh has been generated yet"
        )
        builder.field("Cells", _thousands(counts.cells), important=True)
        builder.field("Points", _thousands(counts.points))
        builder.field("Faces", _thousands(counts.faces))
        builder.field("Internal faces", _thousands(counts.internal_faces))

        shape = self.geometry(ctx)
        builder.field(
            "Dimensionality",
            "2D (planar)" if shape.is_planar else "3D",
            note=shape.source or "",
        )
        if shape.bounds is not None:
            size = shape.bounds.size
            builder.field(
                "Extent",
                " x ".join(_fmt(value) or "?" for value in size),
                note="m, from blockMeshDict",
            )

        # -- patches ------------------------------------------------------------------
        patches = foammesh.boundary_patches(case)
        builder.section(
            "boundary patches", missing="" if patches else "no boundary file to read"
        )
        for patch in patches:
            faces = patch.get("faces")
            builder.field(
                str(patch["name"]),
                str(patch["type"]),
                note=f"{_thousands(int(faces))} faces" if isinstance(faces, int) else "",
            )

        # -- fields -------------------------------------------------------------------
        fields = _initial_fields(case)
        builder.section(
            "fields", missing="" if fields else "no time directory with fields to read"
        )
        if fields:
            names, directory = fields
            builder.field("Present", ", ".join(names), note=f"in {directory}/")
            for name in names:
                # The dimension vector, which is the field's units and the thing that makes
                # `p` either a pressure or a kinematic pressure.
                dimensions = _field_dimensions(case / directory / name)
                builder.field(name, dimensions, note="dimensions" if dimensions else "")

        # -- parallel -----------------------------------------------------------------
        decomposed = count_processor_dirs(case)
        decompose_dict = case / "system" / "decomposeParDict"
        builder.section("parallel")
        builder.field(
            "Decomposed into",
            f"{decomposed} subdomains" if decomposed else "not decomposed",
            important=bool(decomposed),
        )
        if decompose_dict.is_file():
            declared = foamdict.read_int(decompose_dict, "numberOfSubdomains")
            builder.field("decomposeParDict", f"{declared} subdomains" if declared else "present")
            builder.field("Method", foamdict.read_value(decompose_dict, "method"))
            if declared and decomposed and declared != decomposed:
                builder.warn(
                    f"the case is decomposed into {decomposed} but decomposeParDict asks "
                    f"for {declared}; submitting will re-decompose it"
                )

        # -- post-processing ----------------------------------------------------------
        found = foamcoeffs.find_coefficient_file(case)
        builder.section("post-processing")
        if found is not None:
            text, truncated = _read_tail(found, foamcoeffs.MAX_BYTES)
            data = foamcoeffs.parse_coefficients(text, truncated=truncated)
            latest_values = foamcoeffs.latest_coefficients(data)
            builder.field(
                "Force coefficients",
                f"{data.samples} writes",
                note=str(found.relative_to(case)),
            )
            for label, keys in (
                ("Cl (lift)", foamcoeffs.LIFT_KEYS),
                ("Cd (drag)", foamcoeffs.DRAG_KEYS),
            ):
                builder.field(
                    label, _fmt(foamcoeffs.pick(latest_values, keys)), important=True
                )
        functions = settings.get("functions")
        if isinstance(functions, dict) and functions:
            builder.field("Function objects", ", ".join(sorted(functions)))

        # -- setup warnings -----------------------------------------------------------
        if (case / "0.orig").is_dir() and not (case / "0").is_dir():
            builder.warn("there is a 0.orig/ but no 0/: the case has not been set up yet")

        return builder.build()

    def visualise(self, ctx: CaseContext, request: VisualRequest) -> VisualPlan | None:
        """Render the case with ParaView, headlessly (§8.10).

        Returns ``None`` when ParaView is not installed, so the caller can say exactly that
        rather than reporting a command that failed. Dispatch runs on machines reached over
        SSH, and the GUI binary is deliberately never used -- see
        :mod:`~dispatch.adapters.paraview`.

        The camera is derived from the case's own bounding box, and a planar case is never
        rendered edge-on: for a 2-D OpenFOAM mesh -- a 3-D mesh one cell thick -- four of the
        seven angles would correctly produce a picture of a line, so they are substituted and
        the substitution is reported in the plan's notes.

        Writes two things into the case: the generated script, and a ``.foam`` stub if none
        exists. The stub is unavoidable -- it is how ParaView's reader is pointed at an
        OpenFOAM case -- and both land under ``postProcessing/dispatch/`` or are reused if
        already present, so nothing of the simulation is touched.
        """
        binary = paraview.find_paraview(ctx.env)
        if binary is None:
            return None

        try:
            preset = paraview.CameraPreset(request.preset.strip().lower())
        except ValueError as exc:
            valid = ", ".join(paraview.presets())
            raise ValidationError(
                f"Unknown camera preset {request.preset!r}. Valid presets: {valid}"
            ) from exc

        case = ctx.workdir
        output = case / paraview.OUTPUT_DIR
        shape = self.geometry(ctx)
        # Resolved before the camera, because the angle has to be honoured -- and a
        # substitution reported -- whether or not the bounds happen to be readable.
        angle_preset, substituted = paraview.resolve_preset(preset, shape)
        camera = paraview.camera_for(
            shape.bounds if shape else None, preset, geometry=shape
        )

        decomposed = count_processor_dirs(case) > 0
        # Always names the angle, including when the bounds were unknown and ParaView framed
        # the mesh itself: a predictable filename is one that does not depend on whether a
        # blockMeshDict happened to be readable.
        name = f"{request.kind.value}-{angle_preset.value}"
        spec = paraview.RenderRequest(
            reader=self._reader_stub(case, write=not ctx.dry_run),
            output=output,
            name=name,
            preset=angle_preset,
            size=(request.width, request.height),
            field=request.field,
            decomposed=decomposed,
            frames=request.frames,
            planar=bool(shape and shape.is_planar),
        )
        body = (
            paraview.animation_script(spec, camera)
            if request.kind is VisualKind.ANIMATION
            else paraview.screenshot_script(spec, camera)
        )

        script = output / f"{name}.py"
        if not ctx.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            script.write_text(body, encoding="utf-8")

        notes: list[str] = []
        if substituted:
            notes.append(
                f"this case is 2D, so {preset.value} would have looked edge-on at it; "
                f"rendering from {angle_preset.value} instead"
            )
        if camera is None and shape is not None and shape.is_planar:
            notes.append(
                "this case is 2D but its extent could not be read, so the camera will be "
                "aimed along the plane's normal from the mesh ParaView opens"
            )
        elif camera is None:
            notes.append(
                "the mesh bounds could not be read from the case, so ParaView will frame it"
            )
        if decomposed:
            notes.append("reading the decomposed case from its processor directories")

        outputs: list[Path] = [output / f"{name}.png"]
        if request.kind is VisualKind.ANIMATION:
            outputs = [output]

        return VisualPlan(
            steps=(
                CommandStep(
                    argv=paraview.render_argv(binary, script),
                    cwd=case,
                    description=(
                        f"Rendering {request.kind.value} of {case.name} "
                        f"from {angle_preset.value}"
                    ),
                    kind=StepKind.SOLVE,
                    env=dict(ctx.env),
                    timeout_s=paraview.RENDER_TIMEOUT_S,
                ),
            ),
            outputs=tuple(outputs),
            tool=binary,
            notes=tuple(notes),
        )

    def _reader_stub(self, case: Path, *, write: bool) -> Path:
        """The ``.foam`` file ParaView's reader opens, creating one if the case has none.

        An empty file whose *directory* is the case; that is the whole convention. An existing
        one is reused, so a user who already keeps ``case.foam`` in their case keeps it.

        **Dispatch's own log is excluded explicitly.** The working-directory log is called
        ``log.foam`` (§6.4), which matches ``*.foam``; handing it to ParaView would open a
        text file as a case and fail in a way nobody would connect to the log convention.
        """
        existing = sorted(
            path
            for path in _safe_iterdir(case)
            if path.is_file() and path.suffix == ".foam" and not path.name.startswith("log.")
        )
        if existing:
            return existing[0]

        stub = case / f"{case.name or 'case'}.foam"
        if write and not stub.exists():
            try:
                stub.touch()
            except OSError as exc:  # pragma: no cover - unwritable case directory
                log.warning("Could not create the ParaView reader stub %s: %s", stub, exc)
        return stub

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
        _restore_start_from(ctx.workdir)

    def resume_point(self, ctx: CaseContext) -> str | None:
        """The latest time this case has actually written, or ``None`` if it wrote none.

        Read off the disk, because the disk is the only thing that knows. A decomposed run
        writes its times inside ``processorN/``, and a serial one writes them beside
        ``system/``, so both are consulted -- the decomposed copy first, since that is
        where a parallel run (the case worth resuming) puts them.

        The ``0`` directory is deliberately not a resume point: it is the initial
        condition, so "resuming" from it is starting again, and reporting it as a saved
        state would tell the user work had been preserved when none had.
        """
        latest = latest_written_time(ctx.workdir)
        return None if latest is None else f"t = {latest:g}"

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


def _written_times(case: Path) -> list[float]:
    """Every time this case has written, in order, excluding the initial condition.

    Looks in ``processor0`` first for the same reason the resume point does: a parallel run
    writes its times there, and a parallel run is the one worth asking about.
    """
    for root in (case / "processor0", case):
        found: list[float] = []
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            try:
                value = float(child.name)
            except ValueError:
                continue
            if value > 0:
                found.append(value)
        if found:
            return sorted(found)
    return []


def _initial_fields(case: Path) -> tuple[list[str], str] | None:
    """Field names in the earliest time directory, and which directory that was.

    The earliest rather than the latest, because the question this answers is "what does
    this case solve for", and the initial condition is where the full set is declared.
    """
    candidates: list[tuple[float, Path]] = []
    for child in _safe_iterdir(case):
        if not child.is_dir():
            continue
        try:
            candidates.append((float(child.name), child))
        except ValueError:
            continue
    if not candidates:
        return None
    _, directory = min(candidates, key=lambda item: item[0])
    names = sorted(
        entry.name
        for entry in _safe_iterdir(directory)
        if entry.is_file() and not entry.name.startswith(".")
    )
    return (names, directory.name) if names else None


FIELD_HEADER_BYTES = 2048
"""Enough for a FoamFile header and the ``dimensions`` line that follows it."""

_DIMENSIONS = re.compile(r"\bdimensions\s+(\[[^\]]*\])\s*;")


def _field_dimensions(path: Path) -> str | None:
    """A field's dimension vector, e.g. ``[0 2 -2 0 0 0 0]``.

    Read from the top of the file only. These are the field's units, and they are what
    distinguish a pressure from a kinematic pressure -- a distinction that has cost people
    whole studies.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(FIELD_HEADER_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None
    match = _DIMENSIONS.search(head)
    return match.group(1) if match else None


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _fmt(value: float | None) -> str | None:
    """Render a number the way a solver's own output would: ``%g``.

    A case's time step is as likely to be ``1e-05`` as ``0.01``, and neither a fixed number
    of decimals nor ``str`` renders both readably.
    """
    return None if value is None else f"{value:g}"


def _thousands(value: int | None) -> str | None:
    """Group a count, because ``6908400`` and ``6,908,400`` are not equally readable."""
    return None if value is None else f"{value:,}"


def _read_tail(path: Path, limit: int) -> tuple[str, bool]:
    """Read up to ``limit`` bytes from the end of a file, with whether it was truncated.

    From the end, for the same reason the log reader is: when only part of a long history
    can be read, the recent part is the one worth having. The first partial line after a
    mid-file seek is dropped, since a half row parsed as a whole one puts a column's value
    under a different column's name.
    """
    try:
        size = path.stat().st_size
        truncated = size > limit
        with path.open("rb") as handle:
            if truncated:
                handle.seek(-limit, os.SEEK_END)
                handle.readline()
            data = handle.read()
    except OSError as exc:
        log.debug("Cannot read %s: %s", path, exc)
        return "", False
    return data.decode("utf-8", errors="replace"), truncated


def _from_clause(ctx: CaseContext, latest: float | None) -> str:
    """The " from ..." tail of a SOLVE description, when the run is not starting afresh.

    Part of the step description rather than a separate report field so that every surface
    which shows a plan -- the dry run, the TUI's plan view, the step transcript -- says the
    same thing without any of them having to know what ``startFrom`` is.

    A case asked to continue that has written nothing says so plainly. ``startFrom
    latestTime`` on such a case is not an error: OpenFOAM finds only the initial condition
    and starts there, which is the right behaviour and worth stating rather than implying.
    """
    if not ctx.resume:
        return ""
    if latest is None:
        return " from the beginning (no saved time to continue from)"
    return f" continuing from t = {latest:g}"


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


START_FROM_BACKUP = "system/.dispatch-startFrom"
"""Where the pre-resume ``startFrom`` is parked while a restarted run is in flight.

Beside :data:`STOP_AT_BACKUP`, and for the same reason: the edit steers one run, adapter
instances are shared between concurrent jobs, and the value has to survive a daemon
restart -- which, for a resume after a reboot, is precisely the situation.
"""


def _request_latest_time(case: Path) -> bool:
    """Point ``controlDict`` at the case's latest written time. Returns whether it did.

    Idempotent by way of the backup file: if a resume is already in force the original
    value has been saved once, and saving the current ``latestTime`` over it would lose
    what the user actually wrote.
    """
    control = case / "system" / "controlDict"
    if not control.is_file():
        return False
    if (case / START_FROM_BACKUP).exists():
        return True

    previous = foamdict.read_value(control, "startFrom") or "startTime"
    if previous.strip() == "latestTime":
        # Already what the user asked for. Nothing to change, and nothing to restore --
        # writing a backup here would make finalize "restore" a value it never replaced.
        return True
    if not foamdict.set_value(control, "startFrom", "latestTime"):
        return False

    try:
        (case / START_FROM_BACKUP).write_text(previous.strip(), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unwritable case directory
        log.warning("Could not save the original startFrom for %s: %s", case, exc)
    log.info("Set startFrom latestTime in %s (was %s) to resume", control, previous)
    return True


def _restore_start_from(case: Path) -> bool:
    """Put back the ``startFrom`` a resume replaced. Returns whether it did.

    A no-op when there is no backup, which is every run that was not a resume.
    """
    backup = case / START_FROM_BACKUP
    try:
        value = backup.read_text(encoding="utf-8").strip()
    except OSError:
        return False

    control = case / "system" / "controlDict"
    if value and control.is_file():
        foamdict.set_value(control, "startFrom", value)
        log.info("Restored startFrom %s in %s after a resume", value, control)
    backup.unlink(missing_ok=True)
    return True


def latest_written_time(case: Path) -> float | None:
    """The highest time this case has written, decomposed or serial.

    Returns ``None`` when nothing beyond the initial condition has been written, which is
    what a run interrupted before its first write looks like. ``0`` is excluded on purpose:
    it is the initial condition, not a saved state, and treating it as one would report
    preserved progress where there is none.
    """
    for root in (case / "processor0", case):
        latest = _max_time_dir(root)
        if latest is not None:
            return latest
    return None


def _max_time_dir(root: Path) -> float | None:
    """The largest positive numeric directory name directly under ``root``."""
    best: float | None = None
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if value > 0 and (best is None or value > best):
            best = value
    return best


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
