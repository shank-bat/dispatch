"""The generic machine-learning adapter.

Runs a Python training script in a project directory. The interesting question is not how
-- that is one ``python train.py`` and lives in :mod:`~dispatch.adapters.pyjob` -- but
**when to claim a directory**, and this adapter's answer is deliberately, almost
frustratingly, conservative.

A ``pyproject.toml`` means somebody wrote Python. A ``requirements.txt`` means the same
thing. ``main.py`` is a filename convention that predates machine learning by decades. If
any of those were treated as evidence, this adapter would attach itself to every
repository on the machine, and the submit wizard would start marking the user's dotfiles
as a training run. Detection therefore needs one of exactly two things:

1. **A declaration.** ``dispatch.toml`` saying ``[job] adapter = "ml"``. Unambiguous, and
   it also answers "which script", which no heuristic can.
2. **A file literally called ``train.py``, in a directory that is recognisably a Python
   *project*** -- one with a dependency manifest. A script by that name, sitting beside a
   ``pyproject.toml``, is about as close to a self-describing training run as a filesystem
   gets, and it is still only claimed at confidence 0.6 so that anything with a stronger
   claim wins outright.

``main.py`` and ``run.py`` remain in :data:`~dispatch.adapters.pyjob.ENTRYPOINT_CANDIDATES`
-- once a directory *is* an ML job, they are perfectly good entrypoints. They are simply
not evidence that it is one.
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

__all__ = ["MLAdapter"]

TRAINING_SCRIPT: Final = "train.py"
"""The one filename this adapter treats as evidence rather than merely as a candidate."""

PROJECT_MARKERS: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "Pipfile",
)
"""What makes a directory a Python *project* rather than a directory containing Python.

Required alongside ``train.py``: a stray ``train.py`` in a scratch folder is not a job,
and a directory with a dependency manifest is one somebody set up on purpose.
"""

KNOWN_FRAMEWORKS: Final[tuple[str, ...]] = (
    "torch",
    "pytorch_lightning",
    "lightning",
    "tensorflow",
    "keras",
    "jax",
    "flax",
    "sklearn",
    "scikit-learn",
    "xgboost",
    "transformers",
)
"""Recorded as metadata when present. **Not** used for detection.

Importing ``torch`` says nothing about whether a directory is a job to be scheduled --
plenty of libraries import it -- so this only ever enriches the record of a job that was
identified some other way.
"""


class MLAdapter(PythonJobAdapter):
    """Runs a Python machine-learning training script."""

    name: ClassVar[str] = "ml"
    display_name: ClassVar[str] = "Machine learning"
    adapter_version: ClassVar[int] = 1
    log_name: ClassVar[str] = "log.ml"
    """The training log, beside the script that wrote it."""

    metadata_spec: ClassVar[MetadataSpec] = MetadataSpec(
        ref=SpecRef(adapter="ml", version=1),
        fields=(
            MetadataField("entrypoint", FieldType.STR, "Entrypoint", display_order=1),
            MetadataField("framework", FieldType.STR, "Framework", display_order=2),
            MetadataField("args", FieldType.STR, "Arguments", display_order=3),
            MetadataField("gpus", FieldType.INT, "GPUs", display_order=4),
            MetadataField("python", FieldType.PATH, "Interpreter", display_order=5),
            MetadataField("declared", FieldType.BOOL, "Declared", display_order=6),
        ),
    )

    env_keys: ClassVar[Sequence[str]] = (
        "CUDA_VISIBLE_DEVICES",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "TORCH_HOME",
        "HF_HOME",
    )
    """Recorded verbatim in provenance.

    ``CUDA_VISIBLE_DEVICES`` earns its place: six months later, "did this run see a GPU"
    is exactly the question, and the answer is not inferable from anything else.
    """

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        """Claim a directory only on a declaration or on ``train.py`` in a real project."""
        declared = read_declaration(path)
        if declared is not None and declared.adapter == cls.name:
            entry = declared.entrypoint or TRAINING_SCRIPT
            return Detection(
                solver=cls.name,
                confidence=0.95,
                solver_binary="python",
                label=f"ML project: {entry}",
                entry=path / DECLARATION_FILE,
                detail={"declared": True, "entrypoint": entry},
            )
        if declared is not None and declared.declared:
            # Declared, but as somebody else's. Saying nothing is the whole point of the
            # file: the user has already answered the question this method asks.
            return None

        script = path / TRAINING_SCRIPT
        if not script.is_file():
            return None
        if not any((path / marker).is_file() for marker in PROJECT_MARKERS):
            return None

        return Detection(
            solver=cls.name,
            confidence=0.6,
            solver_binary="python",
            label=f"ML project: {TRAINING_SCRIPT}",
            entry=script,
            detail={"declared": False, "entrypoint": TRAINING_SCRIPT},
        )

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        """Record what will run, on what, and with which framework."""
        values = self.base_metadata(ctx)
        values["declared"] = self.declaration(ctx).declared
        if "framework" not in values:
            found = _framework(ctx.workdir)
            if found:
                values["framework"] = found
        return self.metadata_spec.build(values)


def _framework(path: Path) -> str | None:
    """Which known framework this project references, for the record only."""
    return scan_project(path).mentions(KNOWN_FRAMEWORKS)
