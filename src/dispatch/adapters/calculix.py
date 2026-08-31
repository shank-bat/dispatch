"""The CalculiX adapter.

The one that was designed before it was built, as a check that the adapter interface was
actually complete. It needed no new interface surface at all -- which is the result the
design was hoping for.

CalculiX differs from the CFD adapters in a way that would have been awkward if the
interface had assumed MPI: ``ccx`` parallelises with OpenMP threads, not ranks. So the
"parallelism" of a job is an environment variable rather than a launcher, and
:class:`~dispatch.core.plan.CommandStep` already carries a per-step environment.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from dispatch.adapters import gpuenv
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

__all__ = ["CalculiXAdapter"]

log = logging.getLogger(__name__)

STEP_KEYWORD = re.compile(r"^\s*\*STEP\b", re.MULTILINE | re.IGNORECASE)
ANALYSIS_KEYWORDS = ("*STATIC", "*DYNAMIC", "*FREQUENCY", "*HEAT TRANSFER", "*BUCKLE", "*MODAL")
INCREMENT_LINE = re.compile(r"increment\s+(\d+)", re.IGNORECASE)
MAX_DECK_SCAN_BYTES = 256 * 1024


class CalculiXAdapter(BaseAdapter):
    """Runs CalculiX finite-element analyses."""

    name: ClassVar[str] = "calculix"
    display_name: ClassVar[str] = "CalculiX"
    adapter_version: ClassVar[int] = 1
    log_name: ClassVar[str] = "log.calculix"
    """The analysis log, beside the deck.

    Not ``<deck>.log``: ``ccx`` writes its own ``.dat``, ``.sta`` and ``.cvg`` files next
    to the deck, and colliding with that family would be a genuine hazard.
    """

    metadata_spec: ClassVar[MetadataSpec] = MetadataSpec(
        ref=SpecRef(adapter="calculix", version=1),
        fields=(
            MetadataField("deck", FieldType.PATH, "Input deck", display_order=1),
            MetadataField("analysis", FieldType.STR, "Analysis", display_order=2),
            MetadataField("steps", FieldType.INT, "Steps", display_order=3),
            MetadataField("nodes", FieldType.INT, "Nodes", display_order=4),
            MetadataField("elements", FieldType.INT, "Elements", display_order=5),
            MetadataField("threads", FieldType.INT, "OpenMP threads", display_order=6),
        ),
    )

    env_keys: ClassVar[Sequence[str]] = ("OMP_NUM_THREADS", "CCX_NPROC_EQUATION_SOLVER")

    # -- detection ---------------------------------------------------------------------

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        """Recognise a deck by a ``.inp`` file containing ``*STEP``.

        Confidence 0.80: ``.inp`` is used by other FE tools, but the ``*STEP`` keyword in
        Abaqus-style syntax is a strong signal, so the content is checked rather than the
        extension alone.
        """
        decks = find_decks(path)
        if not decks:
            return None

        chosen = decks[0]
        label = f"CalculiX deck: {chosen.name}"
        if len(decks) > 1:
            label += f", {len(decks)} candidate decks"

        return Detection(
            solver=cls.name,
            confidence=0.80,
            solver_binary="ccx",
            label=label,
            entry=chosen,
            detail={"decks": [str(d) for d in decks]},
        )

    # -- validation --------------------------------------------------------------------------

    def validate(self, ctx: CaseContext) -> ValidationReport:
        """Check the deck, its includes, and the solver binary."""
        builder = ReportBuilder()
        decks = find_decks(ctx.workdir)

        if not decks:
            builder.error(
                "no CalculiX input deck found",
                hint="A deck is a .inp file containing a *STEP keyword.",
                code="missing_deck",
            )
            return builder.build()

        if len(decks) > 1 and ctx.entry is None:
            names = ", ".join(sorted(d.name for d in decks))
            builder.error(
                f"several input decks are present ({names}); Dispatch cannot tell which to run",
                hint="Keep one deck per directory, or submit with the file selected.",
                code="ambiguous_deck",
            )
            return builder.build()

        deck = self._deck(ctx)
        if deck is not None:
            self._check_includes(ctx, deck, builder)

        binary = str(self.setting("binary", "ccx"))
        if ctx.which(binary) is None:
            builder.error(
                f"{binary} is not on the PATH",
                hint="Install calculix-ccx, or set [adapters.calculix] binary.",
                code="solver_not_found",
            )

        if ctx.cores > 1:
            builder.info(
                f"CalculiX parallelises with OpenMP, so this job will run with "
                f"OMP_NUM_THREADS={ctx.cores} rather than under MPI"
            )
        return builder.build()

    def _check_includes(self, ctx: CaseContext, deck: Path, builder: ReportBuilder) -> None:
        """Verify every ``*INCLUDE`` the deck references exists.

        A missing include makes ccx fail immediately with a message that does not always
        name the file, so catching it here saves a confusing round trip.
        """
        try:
            text = deck.read_text(encoding="utf-8", errors="replace")[:MAX_DECK_SCAN_BYTES]
        except OSError as exc:
            builder.error(f"{deck.name} could not be read: {exc}", code="unreadable_deck")
            return

        for match in re.finditer(
            r"^\s*\*INCLUDE\s*,\s*INPUT\s*=\s*([^\s,]+)", text, re.MULTILINE | re.IGNORECASE
        ):
            included = (deck.parent / match.group(1).strip()).resolve()
            if not included.is_file():
                builder.error(
                    f"{deck.name} includes {match.group(1)}, which does not exist",
                    path=str(included),
                    code="missing_include",
                )

    # -- planning -----------------------------------------------------------------------------

    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        """One command, with thread count set in its environment.

        ``ccx`` takes the deck name without its ``.inp`` extension. Passing the full
        filename is the most common CalculiX mistake, and the adapter exists partly so the
        user never has to remember.
        """
        deck = self._deck(ctx)
        if deck is None:
            raise FileNotFoundError(f"No CalculiX input deck in {ctx.workdir}")

        env = gpuenv.apply_gpu_visibility(dict(ctx.env), ctx)
        env["OMP_NUM_THREADS"] = str(ctx.cores)
        env["CCX_NPROC_EQUATION_SOLVER"] = str(ctx.cores)

        binary = str(self.setting("binary", "ccx"))
        return ExecutionPlan(
            steps=(
                CommandStep(
                    argv=[binary, "-i", deck.stem],
                    cwd=ctx.workdir,
                    description=(
                        f"Running {binary} on {deck.name} with {ctx.cores} OpenMP thread(s)"
                    ),
                    kind=StepKind.SOLVE,
                    env=env,
                ),
            )
        )

    # -- metadata -------------------------------------------------------------------------------

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        """Read the analysis type and model size from the deck."""
        deck = self._deck(ctx)
        if deck is None:
            return CaseMetadata.empty(self.name)

        values: dict[str, object] = {"deck": deck.name, "threads": ctx.cores}
        try:
            text = deck.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return self.metadata_spec.build(values)

        upper = text.upper()
        for keyword in ANALYSIS_KEYWORDS:
            if keyword in upper:
                values["analysis"] = keyword.lstrip("*").title()
                break

        values["steps"] = len(STEP_KEYWORD.findall(text))
        nodes, elements = _count_entities(text)
        if nodes:
            values["nodes"] = nodes
        if elements:
            values["elements"] = elements
        return self.metadata_spec.build(values)

    def solver_version(self, ctx: CaseContext) -> str | None:
        """Read the CalculiX version from ccx's banner."""
        binary = ctx.which(str(self.setting("binary", "ccx")))
        if binary is None:
            return None
        try:
            result = subprocess.run(
                [binary, "-v"], capture_output=True, timeout=10, check=False, env=dict(ctx.env)
            )
        except (OSError, subprocess.SubprocessError):
            return None
        text = (result.stdout + result.stderr).decode(errors="replace")
        match = re.search(r"Version\s+([\w.]+)", text)
        return f"CalculiX {match.group(1)}" if match else None

    def parse_progress(self, tail: str, ctx: CaseContext) -> Progress | None:
        """Read the increment number from ccx's output.

        The total is not knowable in general -- increments adapt -- so only the current
        value is reported and the interface shows elapsed time alongside it.
        """
        matches = INCREMENT_LINE.findall(tail)
        if not matches:
            return None
        try:
            return Progress(current=float(matches[-1]), total=None, label="Increment")
        except ValueError:
            return None

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        """Offer the deck name as a tag."""
        deck = self._deck(ctx)
        return (deck.stem.lower(),) if deck else ()

    # -- helpers ---------------------------------------------------------------------------------

    def _deck(self, ctx: CaseContext) -> Path | None:
        """The deck to run, preferring the one detection selected."""
        if ctx.entry is not None and ctx.entry.is_file():
            return ctx.entry
        decks = find_decks(ctx.workdir)
        return decks[0] if decks else None


def find_decks(path: Path) -> list[Path]:
    """Every ``.inp`` file in ``path`` that looks like a CalculiX deck."""
    try:
        candidates = sorted(child for child in path.glob("*.inp") if child.is_file())
    except OSError:
        return []
    return [candidate for candidate in candidates if looks_like_deck(candidate)]


def looks_like_deck(path: Path) -> bool:
    """Whether an ``.inp`` contains a ``*STEP`` keyword."""
    try:
        with path.open("rb") as handle:
            head = handle.read(MAX_DECK_SCAN_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return False
    return bool(STEP_KEYWORD.search(head))


def _count_entities(text: str) -> tuple[int | None, int | None]:
    """Count nodes and elements defined inline in a deck.

    Counts data lines following ``*NODE`` and ``*ELEMENT`` blocks. Approximate by design:
    decks that pull their mesh from includes will under-report, and a rough model size is
    still far more useful than none when comparing runs a year later.
    """
    nodes = elements = 0
    mode: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("**"):
            continue
        if stripped.startswith("*"):
            upper = stripped.upper()
            if upper.startswith("*NODE") and "PRINT" not in upper and "FILE" not in upper:
                mode = "node"
            elif upper.startswith("*ELEMENT") and "OUTPUT" not in upper:
                mode = "element"
            else:
                mode = None
            continue
        if mode == "node":
            nodes += 1
        elif mode == "element":
            elements += 1

    return nodes or None, elements or None
