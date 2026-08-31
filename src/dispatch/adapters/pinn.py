"""The physics-informed neural network adapter.

Mechanically this is the ML adapter: a Python program, one command, the same training
curves in the log. What differs is the evidence, and PINN work happens to leave much
better evidence than generic ML does.

A PINN library is a specific dependency. ``import deepxde`` is not something a project
does by accident, and no ordinary Python program imports ``modulus`` or ``sciann`` for
some other reason. So unlike :mod:`~dispatch.adapters.ml` -- which cannot use imports as
evidence, because ``import torch`` proves nothing -- this adapter can read the project's
own imports and manifests and be right.

What it deliberately will **not** do is guess from names. A directory called ``pinn`` or a
file called ``pinn_burgers.py`` is suggestive and nothing more, and the moment a name-based
rule exists somebody's ``spinning_disk`` case gets claimed by a neural-network adapter. The
explicit declaration already covers every case a name would have covered, and covers it
correctly::

    # dispatch.toml
    [job]
    adapter    = "pinn"
    entrypoint = "solve_burgers.py"

Confidence sits at 0.75 for the import signal: above the ML adapter's 0.6, so a PINN
project that also has a ``train.py`` resolves to this one without asking, and below the
0.95 of a case file that means exactly one thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Final

from dispatch.adapters.base import CaseContext
from dispatch.adapters.pyjob import (
    DECLARATION_FILE,
    PythonJobAdapter,
    read_declaration,
    scan_project,
)
from dispatch.core.metadata import (
    CaseMetadata,
    FieldType,
    MetadataField,
    MetadataSpec,
    SpecRef,
)
from dispatch.core.models import Detection

__all__ = ["PINNAdapter"]

PINN_LIBRARIES: Final[tuple[str, ...]] = (
    "deepxde",
    "modulus",
    "physicsnemo",
    "sciann",
    "pinnstorch",
    "neurodiffeq",
    "neuromancer",
    "idrlnet",
    "nvidia-modulus",
)
"""Libraries whose presence means physics-informed work and essentially nothing else.

This is the whole detection rule, and every entry has to earn its place by that standard.
``torch``, ``jax`` and ``tensorflow`` are conspicuously absent: they are how PINNs are
usually built, and they are also how everything else is built.
"""

ENTRYPOINT_CANDIDATES: Final[tuple[str, ...]] = ("train.py", "main.py", "run.py", "solve.py")
"""As for ML, plus ``solve.py`` -- a name PINN projects use and ML projects do not."""


class PINNAdapter(PythonJobAdapter):
    """Runs physics-informed neural network training."""

    name: ClassVar[str] = "pinn"
    display_name: ClassVar[str] = "PINN"
    adapter_version: ClassVar[int] = 1
    log_name: ClassVar[str] = "log.pinn"
    """The training log, beside the script that wrote it."""

    entrypoint_candidates: ClassVar[Sequence[str]] = ENTRYPOINT_CANDIDATES

    metadata_spec: ClassVar[MetadataSpec] = MetadataSpec(
        ref=SpecRef(adapter="pinn", version=1),
        fields=(
            MetadataField("entrypoint", FieldType.STR, "Entrypoint", display_order=1),
            MetadataField("framework", FieldType.STR, "Framework", display_order=2),
            MetadataField("library", FieldType.STR, "PINN library", display_order=3),
            MetadataField("args", FieldType.STR, "Arguments", display_order=4),
            MetadataField("gpus", FieldType.INT, "GPUs", display_order=5),
            MetadataField("python", FieldType.PATH, "Interpreter", display_order=6),
            MetadataField("declared", FieldType.BOOL, "Declared", display_order=7),
        ),
    )

    env_keys: ClassVar[Sequence[str]] = (
        "CUDA_VISIBLE_DEVICES",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "PYTHONPATH",
        "DDE_BACKEND",
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    """``DDE_BACKEND`` in particular: the same DeepXDE script gives different numbers on
    PyTorch and on TensorFlow, and six months later nothing else records which ran."""

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        """Claim a directory on a declaration, or on a PINN library it actually uses."""
        declared = read_declaration(path)
        if declared is not None and declared.adapter == cls.name:
            entry = declared.entrypoint or "the declared entrypoint"
            return Detection(
                solver=cls.name,
                confidence=0.95,
                solver_binary="python",
                label=f"PINN project: {entry}",
                entry=path / DECLARATION_FILE,
                detail={"declared": True, "entrypoint": declared.entrypoint},
            )
        if declared is not None and declared.declared:
            return None

        library = scan_project(path).mentions(PINN_LIBRARIES)
        if library is None:
            return None

        runnable = next(
            (name for name in ENTRYPOINT_CANDIDATES if (path / name).is_file()), None
        )
        if runnable is None:
            # The library is here but nothing looks runnable. Claiming the directory would
            # mean a job that fails validation for a reason the user cannot act on; saying
            # nothing lets them point Dispatch at the right place, or write a declaration.
            return None

        return Detection(
            solver=cls.name,
            confidence=0.75,
            solver_binary="python",
            label=f"PINN project: {runnable} ({library})",
            entry=path / runnable,
            detail={"declared": False, "library": library, "entrypoint": runnable},
        )

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        """Record the entrypoint, the PINN library, and whether a GPU was reserved."""
        values = self.base_metadata(ctx)
        values["declared"] = self.declaration(ctx).declared
        library = scan_project(ctx.workdir).mentions(PINN_LIBRARIES)
        if library:
            values["library"] = library
            values.setdefault("framework", library)
        return self.metadata_spec.build(values)

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        """Offer ``pinn`` and the library, for a history that is searchable by kind."""
        tags = ["pinn", *super().suggest_tags(ctx)]
        library = scan_project(ctx.workdir).mentions(PINN_LIBRARIES)
        if library:
            tags.append(library)
        return tuple(dict.fromkeys(tags))
