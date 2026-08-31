"""The SU2 adapter.

Simpler than OpenFOAM in one important way: SU2 partitions the mesh internally, so there
is no decomposition to manage. The whole plan is one command.

The interesting problem here is *which config*. A case directory frequently holds several
``.cfg`` files -- a baseline, a variation, an adjoint run -- and picking the wrong one
would silently run the wrong simulation. So every plausible config produces its own
detection, and the user chooses the config rather than the solver.
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

__all__ = ["SU2Adapter"]

log = logging.getLogger(__name__)

SIGNATURE_KEYS = ("SOLVER", "MESH_FILENAME", "MATH_PROBLEM", "PHYSICAL_PROBLEM")
"""Keys that identify a file as an SU2 configuration.

``MESH_FILENAME`` alone would match other tools' configs; requiring one of these in a
``KEY= value`` line is specific enough in practice.
"""

PREFERRED_NAMES = ("config.cfg", "turb_SA.cfg", "inv.cfg")
ITERATION_LINE = re.compile(r"^\|?\s*(\d+)\s*\|", re.MULTILINE)
MAX_CONFIG_SCAN_BYTES = 64 * 1024


class SU2Adapter(BaseAdapter):
    """Runs SU2 cases."""

    name: ClassVar[str] = "su2"
    display_name: ClassVar[str] = "SU2"
    adapter_version: ClassVar[int] = 1
    log_name: ClassVar[str] = "log.su2"
    """The solver's output log, in the case directory."""

    metadata_spec: ClassVar[MetadataSpec] = MetadataSpec(
        ref=SpecRef(adapter="su2", version=1),
        fields=(
            MetadataField("solver_type", FieldType.STR, "Solver", display_order=1),
            MetadataField("math_problem", FieldType.STR, "Problem", display_order=2),
            MetadataField("mesh_filename", FieldType.PATH, "Mesh", display_order=3),
            MetadataField("mesh_format", FieldType.STR, "Mesh format", display_order=4),
            MetadataField("iterations", FieldType.INT, "Iterations", display_order=5),
            MetadataField("mach", FieldType.FLOAT, "Mach number", display_order=6),
            MetadataField("aoa", FieldType.FLOAT, "Angle of attack", unit="deg", display_order=7),
            MetadataField("reynolds", FieldType.FLOAT, "Reynolds number", display_order=8),
            MetadataField("restart", FieldType.BOOL, "Restart", display_order=9),
            MetadataField("config", FieldType.PATH, "Config file", display_order=10),
        ),
    )

    env_keys: ClassVar[Sequence[str]] = ("SU2_RUN", "SU2_HOME", "MPI_ARCH_PATH")

    # -- detection -----------------------------------------------------------------------

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        """Recognise a case by an SU2 configuration file.

        Confidence 0.85, below OpenFOAM's: a ``.cfg`` is a less specific signal than
        ``system/controlDict``, so the content is checked for SU2 keys before claiming it.
        """
        configs = find_configs(path)
        if not configs:
            return None

        chosen = _preferred(configs)
        settings = read_config(chosen)
        solver_type = settings.get("SOLVER") or settings.get("PHYSICAL_PROBLEM")

        label = f"SU2 case: {chosen.name}"
        if solver_type:
            label += f" ({solver_type})"
        if len(configs) > 1:
            label += f", {len(configs)} configs present"

        return Detection(
            solver=cls.name,
            confidence=0.85,
            solver_binary="SU2_CFD",
            label=label,
            entry=chosen,
            detail={"configs": [str(c) for c in configs]},
        )

    # -- validation --------------------------------------------------------------------------

    def validate(self, ctx: CaseContext) -> ValidationReport:
        """Check the config, the mesh it names, and the solver binary."""
        builder = ReportBuilder()
        config = self._config(ctx)

        if config is None:
            builder.error(
                "no SU2 configuration file found in this directory",
                hint="An SU2 config contains keys such as SOLVER= and MESH_FILENAME=.",
                code="missing_config",
            )
            return builder.build()

        others = find_configs(ctx.workdir)
        if len(others) > 1:
            names = ", ".join(sorted(c.name for c in others))
            builder.info(f"using {config.name}; this directory also has: {names}")

        settings = read_config(config)
        if not settings:
            builder.error(f"{config.name} could not be read", code="unreadable_config")
            return builder.build()

        self._check_mesh(ctx, config, settings, builder)
        self._check_restart(ctx, settings, builder)

        if ctx.which("SU2_CFD") is None:
            builder.error(
                "SU2_CFD is not on the PATH",
                hint="Add SU2's bin directory to your PATH, or set SU2_RUN.",
                code="solver_not_found",
            )
        if ctx.cores > 1 and ctx.which("mpirun") is None:
            builder.error(
                "mpirun is not on the PATH, so this case cannot run in parallel",
                code="no_mpi",
            )
        mpi.check_slots(ctx, builder)
        return builder.build()

    def _check_mesh(
        self, ctx: CaseContext, config: Path, settings: dict[str, str], builder: ReportBuilder
    ) -> None:
        mesh_name = settings.get("MESH_FILENAME")
        if not mesh_name:
            builder.error(f"{config.name} does not set MESH_FILENAME", code="missing_mesh_filename")
            return
        mesh = (ctx.workdir / mesh_name).resolve()
        if not mesh.is_file():
            builder.error(
                f"the mesh {mesh_name} named in {config.name} does not exist",
                path=str(mesh),
                code="missing_mesh",
            )

    def _check_restart(
        self, ctx: CaseContext, settings: dict[str, str], builder: ReportBuilder
    ) -> None:
        """Catch the classic silent failure: restart requested, no restart file present."""
        if settings.get("RESTART_SOL", "NO").upper() not in ("YES", "TRUE"):
            return
        restart_name = settings.get("SOLUTION_FILENAME", "solution_flow.dat")
        candidates = [
            ctx.workdir / restart_name,
            ctx.workdir / Path(restart_name).with_suffix(".csv").name,
            ctx.workdir / Path(restart_name).with_suffix(".dat").name,
        ]
        if not any(candidate.is_file() for candidate in candidates):
            builder.warning(
                f"RESTART_SOL is YES but no restart file ({restart_name}) was found",
                hint="SU2 will fail at startup, or silently begin from freestream.",
                code="missing_restart",
            )

    # -- planning -------------------------------------------------------------------------------

    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        """One command. SU2 partitions internally, so there is nothing to prepare."""
        config = self._config(ctx)
        name = config.name if config else "config.cfg"
        env = gpuenv.apply_gpu_visibility(dict(ctx.env), ctx)

        if ctx.cores > 1:
            argv = mpi.launch_argv(ctx.cores, "SU2_CFD", name)
            description = f"Running SU2_CFD on {ctx.cores} cores with {name}"
        else:
            argv = ["SU2_CFD", name]
            description = f"Running SU2_CFD with {name}"

        return ExecutionPlan(
            steps=(
                CommandStep(
                    argv=argv,
                    cwd=ctx.workdir,
                    description=description,
                    kind=StepKind.SOLVE,
                    env=env,
                ),
            )
        )

    # -- metadata --------------------------------------------------------------------------------

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        """Read the case parameters worth searching on later."""
        config = self._config(ctx)
        if config is None:
            return CaseMetadata.empty(self.name)

        settings = read_config(config)
        values: dict[str, object] = {
            "solver_type": settings.get("SOLVER") or settings.get("PHYSICAL_PROBLEM"),
            "math_problem": settings.get("MATH_PROBLEM"),
            "mesh_filename": settings.get("MESH_FILENAME"),
            "mesh_format": settings.get("MESH_FORMAT"),
            "config": config.name,
            "iterations": _as_number(settings.get("ITER") or settings.get("EXT_ITER")),
            "mach": _as_number(settings.get("MACH_NUMBER")),
            "aoa": _as_number(settings.get("AOA")),
            "reynolds": _as_number(settings.get("REYNOLDS_NUMBER")),
            "restart": settings.get("RESTART_SOL", "NO").upper() in ("YES", "TRUE"),
        }
        return self.metadata_spec.build({k: v for k, v in values.items() if v is not None})

    def solver_version(self, ctx: CaseContext) -> str | None:
        """Read SU2's version from its ``--help`` banner.

        ``--help`` rather than a bare invocation: an MPI-linked ``SU2_CFD`` started with no
        arguments aborts inside ``MPI_Win_create`` and prints a wall of MPI errors instead
        of a version, which is a noisy way to learn nothing.
        """
        binary = ctx.which("SU2_CFD")
        if binary is None:
            return None
        try:
            result = subprocess.run(
                [binary, "--help"],
                capture_output=True,
                timeout=10,
                check=False,
                env=dict(ctx.env),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        text = (result.stdout + result.stderr).decode(errors="replace")
        match = re.search(r"SU2\s+v?([\w.]+)\s+\"?(\w+)?\"?", text)
        if match:
            release = match.group(2)
            return f"SU2 {match.group(1)}" + (f" {release}" if release else "")
        return None

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        """Offer the solver type as a tag."""
        config = self._config(ctx)
        if config is None:
            return ()
        solver_type = read_config(config).get("SOLVER")
        return (solver_type.lower().replace("_", "-"),) if solver_type else ()

    def parse_progress(self, tail: str, ctx: CaseContext) -> Progress | None:
        """Read the iteration number from SU2's convergence table."""
        matches = ITERATION_LINE.findall(tail)
        if not matches:
            return None
        try:
            current = float(matches[-1])
        except ValueError:
            return None

        total: float | None = None
        config = self._config(ctx)
        if config is not None:
            settings = read_config(config)
            total = _as_number(settings.get("ITER") or settings.get("EXT_ITER"))
        return Progress(current=current, total=total, label="Iteration")

    # -- helpers -----------------------------------------------------------------------------------

    def _config(self, ctx: CaseContext) -> Path | None:
        """The config this job should use.

        Prefers the one detection chose and carried in the context, so a case with three
        configs runs the one the user actually picked.
        """
        if ctx.entry is not None and ctx.entry.is_file():
            return ctx.entry
        configs = find_configs(ctx.workdir)
        return _preferred(configs) if configs else None


def find_configs(path: Path) -> list[Path]:
    """Every file in ``path`` that looks like an SU2 configuration."""
    try:
        candidates = sorted(child for child in path.glob("*.cfg") if child.is_file())
    except OSError:
        return []
    return [candidate for candidate in candidates if looks_like_su2(candidate)]


def looks_like_su2(path: Path) -> bool:
    """Whether a ``.cfg`` contains SU2 configuration keys.

    Only the head of the file is read: SU2 configs put their solver keys near the top, and
    a mesh accidentally named ``.cfg`` should not cost a full read.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(MAX_CONFIG_SCAN_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(
        re.search(rf"^\s*{key}\s*=", head, re.MULTILINE | re.IGNORECASE) for key in SIGNATURE_KEYS
    )


def read_config(path: Path) -> dict[str, str]:
    """Parse an SU2 configuration into a flat mapping.

    The format is ``KEY= value`` with ``%`` comments and values that may span parentheses.
    Multi-line bracketed values are joined so that a list does not truncate at the newline.
    """
    settings: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return settings

    text = re.sub(r"%[^\n]*", "", text)
    text = re.sub(r"\\\s*\n", " ", text)

    depth = 0
    buffer = ""
    for line in text.splitlines():
        buffer += (" " if buffer else "") + line.strip()
        depth += line.count("(") - line.count(")")
        if depth > 0:
            continue
        if "=" in buffer:
            key, _, value = buffer.partition("=")
            key = key.strip().upper()
            if key and key not in settings:
                settings[key] = value.strip()
        buffer = ""
        depth = 0
    return settings


def _preferred(configs: list[Path]) -> Path:
    """Choose among several configs: a conventional name first, else the first found."""
    by_name = {config.name: config for config in configs}
    for name in PREFERRED_NAMES:
        if name in by_name:
            return by_name[name]
    return configs[0]


def _as_number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None
