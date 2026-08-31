"""Fixtures for daemon tests, imported by ``tests/conftest.py``.

The important piece is :class:`FakeAdapter`: a solver that is just ``/bin/sh``. It lets the
whole daemon -- scheduler, executor, process supervision, recovery, IPC -- be exercised end
to end with real subprocesses and real signals, without OpenFOAM installed anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

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

__all__ = ["FakeAdapter", "make_case"]


class FakeAdapter(BaseAdapter):
    """A solver that is a shell script, for testing everything around solvers.

    A case is a directory containing ``fake.job``, whose contents are the shell commands to
    run. Optional files change the shape of the plan:

    ``prepare.sh``   becomes a PREPARE step, so preparation failure paths are testable.
    ``invalid``      makes validation report an ERROR.
    ``warn``         makes validation report a WARNING.
    """

    name: ClassVar[str] = "fake"
    display_name: ClassVar[str] = "Fake Solver"
    adapter_version: ClassVar[int] = 7
    log_name: ClassVar[str] = "log.fake"

    metadata_spec: ClassVar[MetadataSpec] = MetadataSpec(
        ref=SpecRef(adapter="fake", version=1),
        fields=(
            MetadataField("script", FieldType.STR, "Script", display_order=1),
            MetadataField("iterations", FieldType.INT, "Iterations", display_order=2),
        ),
    )

    env_keys: ClassVar[Sequence[str]] = ("FAKE_SOLVER_HOME",)

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        marker = path / "fake.job"
        if not marker.is_file():
            return None
        return Detection(
            solver=cls.name,
            confidence=0.9,
            solver_binary="fakeSolver",
            label="Fake case",
            entry=marker,
            detail={},
        )

    def validate(self, ctx: CaseContext) -> ValidationReport:
        builder = ReportBuilder()
        if (ctx.workdir / "invalid").exists():
            builder.error("this case is marked invalid", code="marked_invalid")
        if (ctx.workdir / "warn").exists():
            builder.warning("this case is marked suspicious", code="marked_warn")
        return builder.build()

    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        steps: list[CommandStep] = []
        prepare = ctx.workdir / "prepare.sh"
        if prepare.is_file():
            steps.append(
                CommandStep(
                    argv=["/bin/sh", "prepare.sh"],
                    cwd=ctx.workdir,
                    description="Preparing the fake case",
                    kind=StepKind.PREPARE,
                    env=dict(ctx.env),
                )
            )
        steps.append(
            CommandStep(
                argv=["/bin/sh", "fake.job"],
                cwd=ctx.workdir,
                description=f"Running the fake solver on {ctx.cores} core(s)",
                kind=StepKind.SOLVE,
                env=dict(ctx.env),
            )
        )
        return ExecutionPlan(steps=tuple(steps))

    def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
        return self.metadata_spec.build({"script": "fake.job", "iterations": ctx.cores * 10})

    def solver_version(self, ctx: CaseContext) -> str | None:
        return "fake 1.0"

    def suggest_tags(self, ctx: CaseContext) -> Sequence[str]:
        return ("fake",)

    def parse_progress(self, tail: str, ctx: CaseContext) -> Progress | None:
        for line in reversed(tail.splitlines()):
            if line.startswith("step "):
                try:
                    return Progress(current=float(line.split()[1]), total=100.0, label="step")
                except (IndexError, ValueError):
                    return None
        return None

    def stop_gracefully(self, ctx: CaseContext) -> bool:
        return (ctx.workdir / "graceful").exists()

    def finalize(self, ctx: CaseContext) -> None:
        """Leave a mark, so tests can assert the executor really does call this.

        Real adapters use it to undo case edits made to steer the run; if the executor ever
        stops calling it, those edits start outliving their jobs.
        """
        marker = ctx.workdir / "finalized"
        marker.write_text(str(int(marker.read_text() or 0) + 1) if marker.exists() else "1")


def make_case(
    directory: Path,
    *,
    script: str = "exit 0",
    prepare: str | None = None,
    invalid: bool = False,
    warn: bool = False,
    graceful: bool = False,
) -> Path:
    """Create a fake case directory.

    Args:
        script: Shell body for the solve step.
        prepare: Shell body for a preparation step, if any.
        invalid: Make validation fail.
        warn: Make validation warn.
        graceful: Make ``stop_gracefully`` claim success.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fake.job").write_text(script)
    if prepare is not None:
        (directory / "prepare.sh").write_text(prepare)
    if invalid:
        (directory / "invalid").touch()
    if warn:
        (directory / "warn").touch()
    if graceful:
        (directory / "graceful").touch()
    return directory
