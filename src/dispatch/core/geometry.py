"""A case's spatial extent, and how many dimensions it meaningfully has.

Needed by anything that has to point a camera at a mesh: where to put it, how far back, and
which directions are worth looking from. All three answers come from the geometry, and the
third one also depends on a fact only the solver knows -- whether the case is
two-dimensional.

That matters more than it sounds. A 2-D OpenFOAM case is a 3-D mesh one cell thick, so its
bounding box is perfectly valid and a camera placed to the side of it sees a line. Asking
"is this 2-D" is therefore not a nicety for labelling; it decides which camera angles
produce an image at all.

Kept in ``core`` and free of any solver vocabulary, so the adapter that knows how to answer
and the tool that needs the answer can meet here without either importing the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = ["Bounds", "CaseGeometry", "Dimensionality"]

FLATNESS_RATIO: Final = 0.02
"""Below this fraction of the largest extent, an axis is treated as flat.

Only a fallback. The authority on whether a case is two-dimensional is its adapter, which
reads the solver's own declaration; this exists for a mesh whose bounds are known but whose
case files are not, where "one of these three extents is two per cent of the others" is the
best available evidence and is nearly always right.
"""


class Dimensionality(StrEnum):
    """How many dimensions a case meaningfully occupies."""

    TWO_D = "2d"
    """Planar: a mesh one cell thick in one direction, however its bounds read."""

    THREE_D = "3d"
    """Genuinely volumetric, and the assumption when nothing says otherwise."""


@dataclass(frozen=True, slots=True)
class Bounds:
    """An axis-aligned bounding box.

    Attributes:
        minimum: Lowest corner, as ``(x, y, z)``.
        maximum: Highest corner.
    """

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.minimum) != 3 or len(self.maximum) != 3:
            raise ValueError("Bounds take three coordinates each")

    @property
    def size(self) -> tuple[float, float, float]:
        """Extent along each axis. Never negative."""
        return tuple(  # type: ignore[return-value]
            max(0.0, high - low) for low, high in zip(self.minimum, self.maximum, strict=True)
        )

    @property
    def centre(self) -> tuple[float, float, float]:
        """The middle of the box, which is what a camera should look at."""
        return tuple(  # type: ignore[return-value]
            (low + high) / 2 for low, high in zip(self.minimum, self.maximum, strict=True)
        )

    @property
    def diagonal(self) -> float:
        """Corner-to-corner distance.

        The basis for every camera distance here: it is the one length that bounds the whole
        mesh whatever direction it is viewed from, so a distance derived from it cannot crop
        the geometry no matter which preset was chosen.
        """
        return math.sqrt(sum(extent * extent for extent in self.size))

    @property
    def largest(self) -> float:
        """The longest single extent."""
        return max(self.size)

    @property
    def flat_axis(self) -> int | None:
        """Which axis is negligibly thin, if one is. ``0``/``1``/``2`` for x/y/z.

        The geometric fallback described in :data:`FLATNESS_RATIO`.
        """
        largest = self.largest
        if largest <= 0:
            return None
        thin = [
            index
            for index, extent in enumerate(self.size)
            if extent / largest < FLATNESS_RATIO
        ]
        # Exactly one: a box thin in two directions is a line, and nothing sensible can be
        # concluded about which way to look at it.
        return thin[0] if len(thin) == 1 else None

    @property
    def thinnest_axis(self) -> int | None:
        """The axis with the smallest extent, with no threshold applied.

        For a case already *known* to be planar this is the plane's normal, and knowing that
        is what makes the threshold in :attr:`flat_axis` unnecessary here. Real 2-D meshes
        are routinely a few per cent thick rather than a thousandth -- one measured case is
        17.5 x 10 x 1 -- so a ratio test would call them volumetric and point the camera
        edge-on at them. Once the case has declared itself planar, the thinnest direction is
        the answer without qualification.
        """
        if self.is_degenerate:
            return None
        extents = self.size
        return min(range(3), key=lambda axis: extents[axis])

    @property
    def is_degenerate(self) -> bool:
        """Whether the box has no size at all, so no camera distance can be derived."""
        return self.diagonal <= 0.0


@dataclass(frozen=True, slots=True)
class CaseGeometry:
    """What a case occupies in space, and how many dimensions of it matter.

    Attributes:
        bounds: The mesh's bounding box, when it is known. ``None`` when the extent could
            not be read but the dimensionality still could -- a case whose 2-D nature is
            declared in its own files but whose mesh has not been generated yet.
        dimensionality: Planar or volumetric.
        normal: For a planar case, the axis the plane is normal to, when known. This is the
            direction from which the mesh is *visible* and perpendicular to which it is a
            line, so a camera that ignores it produces a blank image.
        source: What the answer was read from, for a user asking why.
    """

    bounds: Bounds | None = None
    dimensionality: Dimensionality = Dimensionality.THREE_D
    normal: int | None = None
    source: str | None = None

    @property
    def is_planar(self) -> bool:
        """Whether this case is two-dimensional."""
        return self.dimensionality is Dimensionality.TWO_D

    @property
    def resolved_normal(self) -> int | None:
        """The plane's normal axis, falling back to the geometrically thin one.

        A declaration is preferred over a measurement: a case can say it is 2-D before its
        mesh exists, and a mesh can be thin without the case being planar.
        """
        if self.normal is not None:
            return self.normal
        if self.bounds is None:
            return None
        # A declared planar case is thin in exactly one direction, so no threshold is
        # needed; an undeclared one has to earn the conclusion geometrically.
        return self.bounds.thinnest_axis if self.is_planar else self.bounds.flat_axis
