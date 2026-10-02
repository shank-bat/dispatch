"""Reading an OpenFOAM mesh's shape without loading the mesh.

Two questions, both answerable from headers and small files:

**Is the case two-dimensional?** OpenFOAM has no 2-D meshes. A planar case is a 3-D mesh one
cell thick whose front and back faces carry the ``empty`` patch type, and that patch type is
the declaration -- it is how the user tells the solver to skip the third direction. So the
answer is in ``constant/polyMesh/boundary``, it is exact, and it is available before any
camera has been pointed anywhere.

**How big is it?** The ``points`` file holds every vertex, which on a real mesh is hundreds
of megabytes, so it is not read. ``blockMeshDict`` holds the vertices of the *blocks* when
the mesh was built with ``blockMesh``, which is a handful of numbers and bounds the mesh
exactly. When that is unavailable the bounds are simply unknown, and the caller that needs
them -- the screenshot tool -- asks ParaView, which has the mesh open anyway.

Counts come from the ``note`` line OpenFOAM writes into the ``owner`` header, which records
``nPoints``, ``nCells``, ``nFaces`` and ``nInternalFaces``. Far cheaper than counting, and
it works for a decomposed case too.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from dispatch.core.geometry import Bounds, CaseGeometry, Dimensionality

__all__ = [
    "MeshCounts",
    "boundary_patches",
    "mesh_counts",
    "read_bounds",
    "read_geometry",
]

log = logging.getLogger(__name__)

EMPTY_PATCH: Final = "empty"
"""The patch type that declares a direction the solver should ignore -- i.e. a 2-D case."""

HEADER_BYTES: Final = 4096
BOUNDARY_BYTES: Final = 256 * 1024
"""A boundary file is a few hundred lines even for a complicated mesh."""

BLOCKMESH_BYTES: Final = 512 * 1024

_NOTE = re.compile(r"note\s+\"([^\"]*)\"")
_COUNT = re.compile(r"n([A-Za-z]+)\s*:\s*(\d+)")
_PATCH = re.compile(r"([A-Za-z_][\w.\-]*)\s*\{([^{}]*)\}", re.DOTALL)
"""A named block and its body.

The name is *not* anchored to its own line. OpenFOAM writes it that way, but a hand-edited
or generated boundary file may put the brace alongside, and a parser that only accepts the
house style silently reports a mesh as having no patches at all.

``[^{}]*`` rather than a non-greedy ``.*?``: a patch body is flat, and refusing to cross a
brace means a nested block cannot be swallowed into its parent's body. Blocks with no
``type`` -- the ``FoamFile`` header, chiefly -- are dropped by the caller.
"""
_TYPE = re.compile(r"\btype\s+([A-Za-z_]+)\s*;")
_N_FACES = re.compile(r"\bnFaces\s+(\d+)\s*;")
_VERTEX = re.compile(r"\(\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\)")


class MeshCounts:
    """Entity counts from a mesh's own header note.

    A plain object rather than a dataclass because what it holds depends on what the header
    recorded, and an absent count is meaningfully different from zero.
    """

    __slots__ = ("values",)

    def __init__(self, values: dict[str, int] | None = None) -> None:
        self.values = dict(values or {})

    def __bool__(self) -> bool:
        return bool(self.values)

    def get(self, name: str) -> int | None:
        """One count by its header name, e.g. ``Cells``."""
        return self.values.get(name)

    @property
    def cells(self) -> int | None:
        return self.get("Cells")

    @property
    def points(self) -> int | None:
        return self.get("Points")

    @property
    def faces(self) -> int | None:
        return self.get("Faces")

    @property
    def internal_faces(self) -> int | None:
        return self.get("InternalFaces")


def _owner_files(case: Path) -> Sequence[Path]:
    """Where a mesh's ``owner`` file may be, decomposed copy first.

    The decomposed copy first because a case worth asking about is usually one that has run,
    and a parallel run's mesh lives in ``processor0``.
    """
    return (
        case / "processor0" / "constant" / "polyMesh" / "owner",
        case / "constant" / "polyMesh" / "owner",
    )


def mesh_counts(case: Path) -> MeshCounts:
    """Entity counts, read from the ``owner`` header's ``note`` line.

    Empty when there is no mesh, which is an ordinary state for a case that has been set up
    but not meshed.
    """
    for candidate in _owner_files(case):
        try:
            with candidate.open("rb") as handle:
                head = handle.read(HEADER_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue
        note = _NOTE.search(head)
        if note is None:
            continue
        found = {name: int(value) for name, value in _COUNT.findall(note.group(1))}
        if found:
            return MeshCounts(found)
    return MeshCounts()


def boundary_patches(case: Path) -> list[dict[str, str | int]]:
    """Every boundary patch, with its type and face count, in file order.

    Order is kept because it is the order the solver reports and the order the user wrote,
    which is what makes the list recognisable.
    """
    for parent in (case / "processor0", case):
        path = parent / "constant" / "polyMesh" / "boundary"
        try:
            with path.open("rb") as handle:
                text = handle.read(BOUNDARY_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue

        patches: list[dict[str, str | int]] = []
        for name, body in _PATCH.findall(text):
            kind = _TYPE.search(body)
            if kind is None:
                # A block with no type is the FoamFile header or a coefficients sub-block,
                # not a patch.
                continue
            entry: dict[str, str | int] = {"name": name, "type": kind.group(1)}
            faces = _N_FACES.search(body)
            if faces is not None:
                entry["faces"] = int(faces.group(1))
            patches.append(entry)
        if patches:
            return patches
    return []


def read_bounds(case: Path) -> Bounds | None:
    """The mesh's extent, from ``blockMeshDict``'s vertices when there is one.

    Not from ``constant/polyMesh/points``: that file holds every vertex of the real mesh and
    runs to hundreds of megabytes, which is not a thing to read to decide where to put a
    camera. ``blockMeshDict`` holds the block corners, which bound the mesh exactly and
    number a dozen.

    ``None`` when the mesh was not built by ``blockMesh`` -- a snappyHexMesh or imported
    mesh -- in which case the caller should ask whatever already has the mesh open.
    """
    for relative in ("system/blockMeshDict", "constant/polyMesh/blockMeshDict"):
        path = case / relative
        try:
            with path.open("rb") as handle:
                text = handle.read(BLOCKMESH_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue

        vertices = _vertices(text)
        if len(vertices) < 2:
            continue
        scale = _scale(text)
        lows = tuple(min(vertex[axis] for vertex in vertices) * scale for axis in range(3))
        highs = tuple(max(vertex[axis] for vertex in vertices) * scale for axis in range(3))
        return Bounds(minimum=lows, maximum=highs)  # type: ignore[arg-type]
    return None


def _vertices(text: str) -> list[tuple[float, float, float]]:
    """Vertex triples from a ``blockMeshDict``'s ``vertices`` list.

    Only that list: ``blocks``, ``edges`` and ``boundary`` also contain parenthesised groups,
    and a ``hex (0 1 2 3 4 5 6 7)`` read as a coordinate would put the bounding box around
    vertex indices instead of around the mesh.
    """
    start = text.find("vertices")
    if start < 0:
        return []
    opening = text.find("(", start)
    if opening < 0:
        return []
    depth = 0
    end = opening
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                end = index
                break
    found: list[tuple[float, float, float]] = []
    for match in _VERTEX.finditer(text[opening : end + 1]):
        try:
            found.append((float(match.group(1)), float(match.group(2)), float(match.group(3))))
        except ValueError:  # pragma: no cover - the pattern already guaranteed this
            continue
    return found


def _scale(text: str) -> float:
    """The dict's ``scale`` (or legacy ``convertToMeters``), defaulting to 1.

    Omitting it would report a mesh in millimetres as if it were in metres, which is only
    wrong by a thousand and would place a camera nowhere near the geometry.
    """
    for key in ("scale", "convertToMeters"):
        match = re.search(rf"\b{key}\s+([-\d.eE+]+)\s*;", text)
        if match is None:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value > 0:
            return value
    return 1.0


def read_geometry(case: Path) -> CaseGeometry:
    """The case's extent and dimensionality, from whatever of each can be read.

    The two halves are independent, and that is deliberate: a case declares itself planar in
    its boundary file whether or not its mesh has been generated, so the dimensionality is
    often knowable when the bounds are not.
    """
    patches = boundary_patches(case)
    empty = [patch for patch in patches if patch.get("type") == EMPTY_PATCH]

    dimensionality = Dimensionality.TWO_D if empty else Dimensionality.THREE_D
    source = (
        f"{EMPTY_PATCH} patch {empty[0]['name']!r} in constant/polyMesh/boundary"
        if empty
        else ("constant/polyMesh/boundary" if patches else None)
    )

    bounds = read_bounds(case)
    # The thinnest axis, not the "flat" one: a declared 2-D mesh is one cell thick in exactly
    # one direction, but that thickness is routinely a few per cent of the domain rather
    # than a thousandth, so a ratio test would miss it.
    normal = bounds.thinnest_axis if (empty and bounds is not None) else None
    return CaseGeometry(
        bounds=bounds, dimensionality=dimensionality, normal=normal, source=source
    )


FIELD_CLASSES: Final[dict[str, str]] = {
    "volScalarField": "scalar",
    "volVectorField": "vector",
    "volSymmTensorField": "tensor",
    "volTensorField": "tensor",
    "volSphericalTensorField": "tensor",
    "pointScalarField": "scalar",
    "pointVectorField": "vector",
}
"""FoamFile classes ParaView's reader turns into colourable arrays.

Surface fields (``phi``) are absent on purpose: the reader does not load them onto the
internal mesh, so offering one would produce a render that fails with "no such array".
"""

FIELD_LABELS: Final[dict[str, str]] = {
    "U": "velocity",
    "p": "pressure",
    "p_rgh": "pressure minus hydrostatic",
    "T": "temperature",
    "k": "turbulent kinetic energy",
    "omega": "specific dissipation rate",
    "epsilon": "dissipation rate",
    "nut": "turbulent viscosity",
    "nuTilda": "Spalart-Allmaras variable",
    "rho": "density",
    "alpha.water": "water fraction",
    "vorticity": "vorticity",
    "Q": "Q-criterion",
}
"""Plain-language names for the fields CFD users meet most, shown beside the solver's name."""

_CLASS = re.compile(r"\bclass\s+(\w+)\s*;")


def _field_class(path: Path) -> str | None:
    """A field file's FoamFile ``class``, reading only its header (gzipped or not)."""
    try:
        if path.suffix == ".gz":
            import gzip

            with gzip.open(path, "rb") as handle:
                head = handle.read(HEADER_BYTES)
        else:
            with path.open("rb") as handle:
                head = handle.read(HEADER_BYTES)
    except OSError:
        return None
    match = _CLASS.search(head.decode("utf-8", errors="replace"))
    return match.group(1) if match else None


def _time_dirs(root: Path) -> list[tuple[float, Path]]:
    found: list[tuple[float, Path]] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return found
    for child in children:
        if not child.is_dir():
            continue
        try:
            found.append((float(child.name), child))
        except ValueError:
            continue
    return sorted(found)


def render_fields(case: Path) -> list[tuple[str, str]]:
    """``(name, kind)`` for every field a render of this case can be coloured by.

    Read from the **latest** written time rather than ``0/``: a field the solver derives as it
    runs (``vorticity``, ``Q``, ``yPlus``) exists only in written times, and those are what
    an animation shows. Falls back to the earliest directory for a case that has not run.
    A decomposed case's times live in ``processor0``.
    """
    for root in (case / "processor0", case):
        times = _time_dirs(root)
        if not times:
            continue
        written = [entry for entry in times if entry[0] > 0]
        directory = (written or times)[-1][1]
        fields: list[tuple[str, str]] = []
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file() or entry.name.startswith("."):
                continue
            kind = FIELD_CLASSES.get(_field_class(entry) or "")
            if kind is None:
                continue
            name = entry.name[:-3] if entry.name.endswith(".gz") else entry.name
            fields.append((name, kind))
        if fields:
            return fields
    return []
