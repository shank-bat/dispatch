"""Asking an adapter to produce a picture of a case.

Same shape as everything else an adapter is asked for: a **declarative plan** rather than an
action (§13.9). The adapter decides which tool renders its cases, where the camera goes and
what the output is called; something else runs the command, supervises it, times it out and
reports what happened. The payoff is the same too -- the interesting parts become pure
functions that can be asserted on with no renderer installed.

Deliberately thin. There is one renderer and one solver that uses it today, and a generic
visualisation framework for a feature with one implementation would be abstraction bought on
credit. What is here is the minimum that keeps the solver-specific half inside the adapter:
a request naming what sort of picture is wanted, and a plan carrying the commands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from dispatch.core.plan import CommandStep

__all__ = ["VisualKind", "VisualPlan", "VisualRequest"]


class VisualKind(StrEnum):
    """What sort of picture is wanted."""

    MESH = "mesh"
    """A still image of the mesh, from one camera angle."""

    ANIMATION = "animation"
    """A frame per written time step, over the whole run."""


@dataclass(frozen=True, slots=True)
class VisualRequest:
    """What to render.

    Attributes:
        kind: Still or animated.
        preset: Camera angle, by name. Which names exist is the renderer's business; an
            unknown one should be refused with the list rather than silently defaulted.
        field: Field to colour by, or ``None`` for the plain mesh.
        frames: Cap on animation frames. ``None`` renders every written time.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    kind: VisualKind = VisualKind.MESH
    preset: str = "isometric"
    field: str | None = None
    frames: int | None = None
    width: int = 1600
    height: int = 1000


@dataclass(frozen=True, slots=True)
class VisualPlan:
    """How to produce the requested picture, and what it will produce.

    Attributes:
        steps: Commands to run, in order. Ordinary
            :class:`~dispatch.core.plan.CommandStep` values, so whatever runs them already
            knows how to log, time out and cancel them.
        outputs: Files or directories the render will write. Reported up front so the caller
            can tell the user where to look without having to parse the renderer's output.
        tool: The renderer that will be used, for a user asking what produced this.
        notes: Things worth saying about the plan -- most importantly, that a requested
            camera angle was substituted because it would have looked edge-on at a planar
            mesh. A substitution that is not reported is a wrong answer.
    """

    steps: Sequence[CommandStep]
    outputs: Sequence[Path] = field(default_factory=tuple)
    tool: str = ""
    notes: Sequence[str] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return bool(self.steps)
