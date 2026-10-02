"""Rendering OpenFOAM cases with ParaView, headlessly.

Dispatch runs on a workstation that is reached over SSH, so there is no display and
``paraview`` the GUI is not an option. ``pvbatch`` is: it is ParaView's batch interpreter,
it ships with every ParaView install, and it renders off-screen. Everything here therefore
produces a **Python script and an argv**, which is the same declarative shape adapters use
for solvers (§13.9) and for the same reasons: the hard parts become pure functions that can
be asserted on without ParaView installed, and running them stays in one supervised place.

Two outputs, sharing all of the camera work:

* an **animation** over the case's written time steps;
* **mesh screenshots** from named camera angles.

The camera is the substance. A fixed distance works for exactly one mesh, so the position is
derived from the case's own bounding box; and a two-dimensional case -- which in OpenFOAM is
a 3-D mesh one cell thick -- has four of the seven preset angles pointing edge-on at it,
where a correct render is a blank image. Both of those are handled here, from facts the
adapter supplies.

See ``docs/ARCHITECTURE.md`` §8.10.
"""

from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from dispatch.core.geometry import Bounds, CaseGeometry

__all__ = [
    "Camera",
    "CameraPreset",
    "RenderRequest",
    "animation_script",
    "camera_for",
    "find_paraview",
    "render_argv",
    "screenshot_script",
]

log = logging.getLogger(__name__)

PARAVIEW_BINARIES: Final = ("pvbatch", "pvpython")
"""Batch interpreters, in preference order.

``pvbatch`` first because it is built for exactly this -- no GUI, no display, renders
off-screen -- and because it is what a headless install has. ``pvpython`` is the fallback and
behaves the same for a script that never opens a window. The GUI ``paraview`` binary is
deliberately absent: a machine reached over SSH has no display to put it on.
"""

OUTPUT_DIR: Final = "postProcessing/dispatch"
"""Where renders are written, inside the case.

Under ``postProcessing`` because that is where a case already keeps things derived from its
results, and in a ``dispatch`` subdirectory so nothing of ours is mistaken for a function
object's output.
"""

MARGIN: Final = 1.15
"""How much room to leave around the mesh, as a multiple of what would just fit.

Fifteen per cent. Enough that the geometry is not touching the frame edge, little enough
that the mesh still fills the image.
"""

ISOMETRIC_DIRECTION: Final = (1.0, 1.0, 1.0)

DEFAULT_SIZE: Final = (1600, 1000)
"""Pixels. A 16:10 frame large enough to read a mesh in, small enough to send over SSH."""

RENDER_TIMEOUT_S: Final = 900.0
"""Wall-clock limit on one render.

Generous: an animation over a thousand time steps of a seven-million-cell mesh is real work.
Bounded all the same, because a ParaView that cannot find a GL context does not fail -- it
waits.
"""


class CameraPreset(StrEnum):
    """Named viewing directions.

    The six axis-aligned faces plus an isometric three-quarter view, which is the one that
    shows a 3-D mesh as a shape rather than as a silhouette.
    """

    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ISOMETRIC = "isometric"


# Which way the camera looks *from*, and which way is up. Right-handed, z up, so "front"
# looks along -y at the xz plane -- the convention a CFD user reading a 2-D case in the xy
# plane expects, with "top" looking down the z axis at it.
_Vector = tuple[float, float, float]

_DIRECTIONS: Final[Mapping[CameraPreset, tuple[_Vector, _Vector]]] = {
    CameraPreset.FRONT: ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    CameraPreset.BACK: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    CameraPreset.LEFT: ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    CameraPreset.RIGHT: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    CameraPreset.TOP: ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    CameraPreset.BOTTOM: ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    CameraPreset.ISOMETRIC: (ISOMETRIC_DIRECTION, (0.0, 0.0, 1.0)),
}

# For a planar case, the axis each preset looks along. A preset whose axis is *not* the
# plane's normal is looking edge-on at a mesh one cell thick, and renders a line.
_PRESET_AXIS: Final[Mapping[CameraPreset, int | None]] = {
    CameraPreset.FRONT: 1,
    CameraPreset.BACK: 1,
    CameraPreset.LEFT: 0,
    CameraPreset.RIGHT: 0,
    CameraPreset.TOP: 2,
    CameraPreset.BOTTOM: 2,
    CameraPreset.ISOMETRIC: None,
}


@dataclass(frozen=True, slots=True)
class Camera:
    """A resolved camera, in the terms ParaView's render view takes.

    Attributes:
        position: Where the camera is.
        focal_point: What it looks at -- the centre of the mesh.
        up: Which way is up in the image.
        parallel_scale: Half the vertical extent the view covers, for parallel projection.
        preset: The preset this came from, after any substitution for a planar case.
        substituted: Whether the requested preset was replaced because it would have looked
            edge-on at a planar mesh. Reported so the caller can say so rather than silently
            returning a different view than was asked for.
    """

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    up: tuple[float, float, float]
    parallel_scale: float
    preset: CameraPreset
    substituted: bool = False


def find_paraview(env: Mapping[str, str] | None = None) -> str | None:
    """The ParaView batch interpreter on this machine, or ``None``.

    Searched on the job's own ``PATH`` when one is given, for the same reason adapter
    validation uses ``ctx.which``: a tool that exists only after an environment has been
    sourced is not found by looking at the daemon's path.
    """
    path = (env or {}).get("PATH")
    for binary in PARAVIEW_BINARIES:
        found = shutil.which(binary, path=path)
        if found is not None:
            return found
    return None


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 0:
        return (0.0, 0.0, 1.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _axis_direction(axis: int) -> tuple[float, float, float]:
    return tuple(1.0 if index == axis else 0.0 for index in range(3))  # type: ignore[return-value]


def _up_for(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    """An up vector that is not parallel to the viewing direction.

    A camera whose up vector points the way it is looking has no defined orientation, and
    ParaView renders nothing useful from it.
    """
    candidate = (0.0, 0.0, 1.0)
    if abs(direction[2]) > 0.9:
        candidate = (0.0, 1.0, 0.0)
    return candidate


def resolve_preset(
    preset: CameraPreset, geometry: CaseGeometry | None
) -> tuple[CameraPreset, bool]:
    """Swap a preset that would look edge-on at a planar mesh for one that can see it.

    A 2-D OpenFOAM case is a 3-D mesh one cell thick. Its bounding box is perfectly valid, so
    nothing *fails* when the camera is placed to the side of it -- the render simply contains
    a line. Four of the seven presets do that on any given planar case, which would make the
    feature look broken for exactly the cases CFD users have most of.

    So a planar case whose requested view is not along its normal is rendered along the
    normal instead, and the substitution is reported rather than hidden.

    Returns:
        The preset to use, and whether it was substituted.
    """
    if geometry is None or not geometry.is_planar:
        return preset, False
    normal = geometry.resolved_normal
    if normal is None:
        return preset, False
    if _PRESET_AXIS[preset] == normal:
        return preset, False

    # Look along the plane's normal. Which of the two opposite presets is chosen keeps the
    # sense the user asked for where that is meaningful -- "back"/"left"/"bottom" stay
    # negative-facing -- so repeating the request is at least consistent.
    negative = preset in (CameraPreset.BACK, CameraPreset.LEFT, CameraPreset.BOTTOM)
    along = {
        0: (CameraPreset.LEFT if negative else CameraPreset.RIGHT),
        1: (CameraPreset.BACK if negative else CameraPreset.FRONT),
        2: (CameraPreset.BOTTOM if negative else CameraPreset.TOP),
    }[normal]
    return along, True


def camera_for(
    bounds: Bounds | None,
    preset: CameraPreset = CameraPreset.ISOMETRIC,
    *,
    geometry: CaseGeometry | None = None,
    margin: float = MARGIN,
) -> Camera | None:
    """Place a camera so the whole mesh fits in frame.

    The distance comes from the bounding box's **diagonal**, not from any single extent: the
    diagonal bounds the mesh from every direction, so a distance derived from it cannot crop
    the geometry whichever preset was chosen. That is the alternative to the fixed numbers
    that work for one mesh and clip or shrink every other one.

    Args:
        bounds: The mesh's extent. ``None`` -- an imported mesh whose bounds Dispatch cannot
            read -- yields ``None``, and the caller should let ParaView frame it instead.
        preset: Requested viewing direction.
        geometry: The case's shape, so a planar case is not viewed edge-on.
        margin: Room to leave around the mesh.

    Returns:
        The camera, or ``None`` when there is no geometry to aim at.
    """
    if bounds is None or bounds.is_degenerate:
        return None

    chosen, substituted = resolve_preset(preset, geometry)
    direction, up = _DIRECTIONS[chosen]
    direction = _unit(direction)

    # Half the diagonal is the radius of a sphere containing the mesh; the margin opens the
    # frame around it. Distance is from that radius rather than from a field of view, so the
    # framing is identical under parallel and perspective projection.
    radius = bounds.diagonal / 2 * margin
    centre = bounds.centre
    position = tuple(
        centre[axis] + direction[axis] * radius * 2 for axis in range(3)
    )

    if abs(sum(direction[axis] * up[axis] for axis in range(3))) > 0.99:
        up = _up_for(direction)

    return Camera(
        position=position,  # type: ignore[arg-type]
        focal_point=centre,
        up=_unit(up),
        parallel_scale=radius,
        preset=chosen,
        substituted=substituted,
    )


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """What to render, and where to put it.

    Attributes:
        reader: The file ParaView should open -- for OpenFOAM, the ``.foam`` stub.
        output: Directory for the images. Created by the script.
        name: Base filename, without extension or frame number.
        preset: Camera angle. Ignored when ``camera`` is ``None``.
        size: Image size in pixels.
        field: Field to colour by, or ``None`` for the plain mesh.
        decomposed: Whether to read the case as a decomposed (``processorN``) case.
        frames: Cap on animation frames, or ``None`` for every written time.
        planar: Whether the case is two-dimensional. Only consulted when no camera could be
            computed, in which case the script works the plane's normal out for itself.
    """

    reader: Path
    output: Path
    name: str = "mesh"
    preset: CameraPreset = CameraPreset.ISOMETRIC
    size: tuple[int, int] = DEFAULT_SIZE
    field: str | None = None
    decomposed: bool = False
    frames: int | None = None
    planar: bool = False


def render_argv(binary: str, script: Path) -> list[str]:
    """The command that runs a generated script headlessly.

    ``--force-offscreen-rendering`` so a ParaView built against a windowing system does not
    try to find a display that is not there. Harmless on a build without it.
    """
    return [binary, "--force-offscreen-rendering", str(script)]


_PREAMBLE = '''\
# Generated by Dispatch. Safe to delete.
#
# Run headlessly:  pvbatch --force-offscreen-rendering <this file>
import os

from paraview.simple import (
    GetActiveViewOrCreate,
    OpenFOAMReader,
    SaveScreenshot,
    Show,
    UpdatePipeline,
)

reader = OpenFOAMReader(registrationName="case", FileName={reader!r})
reader.CaseType = {case_type!r}
reader.MeshRegions = ["internalMesh"]
UpdatePipeline(proxy=reader)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [{width:d}, {height:d}]
view.UseLight = 1
view.OrientationAxesVisibility = 0
# The palette has to be switched off before Background is honoured. Without this ParaView
# renders its default grey whatever Background says, which is a legible image but not the
# one asked for -- and a white background is what goes into a paper.
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:  # pragma: no cover - older ParaView has no palette override
    pass
view.Background = [1.0, 1.0, 1.0]
view.BackgroundColorMode = "Single Color" if hasattr(view, "BackgroundColorMode") else None

display = Show(reader, view)
display.Representation = {representation!r}
os.makedirs({output!r}, exist_ok=True)
'''

_CAMERA = '''\
view.CameraPosition = [{px!r}, {py!r}, {pz!r}]
view.CameraFocalPoint = [{fx!r}, {fy!r}, {fz!r}]
view.CameraViewUp = [{ux!r}, {uy!r}, {uz!r}]
view.CameraParallelProjection = 1
view.CameraParallelScale = {scale!r}
'''

_RESET_CAMERA = """\
# Bounds were not available from the case -- an imported or snappyHexMesh mesh Dispatch
# cannot measure -- so the *direction* comes from the requested angle and the *distance*
# from ParaView, which has the mesh open and can measure it exactly. ResetCamera keeps the
# direction it is given and only moves along it.
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.CameraPosition = [{dx!r}, {dy!r}, {dz!r}]
view.CameraViewUp = [{ux!r}, {uy!r}, {uz!r}]
view.ResetCamera()
view.CameraParallelProjection = 1
"""

_PLANAR_RESET_CAMERA = """\
# This case is two-dimensional -- its front and back faces carry the `empty` patch type --
# but Dispatch could not read its extent, so it cannot say which way the plane faces. A
# camera pointed the wrong way at a mesh one cell thick renders a line.
#
# ParaView has the mesh open and therefore knows. The thinnest axis of the data's own
# bounding box is the plane's normal, and that is the only direction worth looking from.
_bounds = reader.GetDataInformation().GetBounds()
_extents = [_bounds[1] - _bounds[0], _bounds[3] - _bounds[2], _bounds[5] - _bounds[4]]
_normal = _extents.index(min(_extents))

view.CameraFocalPoint = [
    (_bounds[0] + _bounds[1]) / 2,
    (_bounds[2] + _bounds[3]) / 2,
    (_bounds[4] + _bounds[5]) / 2,
]
view.CameraPosition = [
    view.CameraFocalPoint[axis] + ({sense!r} if axis == _normal else 0.0)
    for axis in range(3)
]
view.CameraViewUp = [0.0, 1.0, 0.0] if _normal == 2 else [0.0, 0.0, 1.0]
view.ResetCamera()
view.CameraParallelProjection = 1
print("2D case: rendering along axis %d" % _normal)
"""


def _preamble(request: RenderRequest, *, representation: str) -> str:
    return _PREAMBLE.format(
        reader=str(request.reader),
        case_type="Decomposed Case" if request.decomposed else "Reconstructed Case",
        width=request.size[0],
        height=request.size[1],
        representation=representation,
        output=str(request.output),
    )


def _camera_block(
    camera: Camera | None, fallback: CameraPreset, *, planar: bool = False
) -> str:
    """The camera section of a script.

    Without bounds there is still an *angle* to honour -- including one substituted because
    the case is planar -- so the direction is set and ParaView is left to work out how far
    back to stand.

    A planar case whose extent Dispatch could not read is the one situation where the angle
    cannot be settled here at all, and it is a common one: a snappyHexMesh case declares
    itself 2-D in its boundary file but has no ``blockMeshDict`` to measure. Rather than
    guessing or honouring a request that would render a line, the decision is deferred into
    the script, where the mesh is open and its bounds are exact.
    """
    if camera is None and planar:
        # Negative sense, so the view is from "below" the plane in whichever axis it turns
        # out to be -- consistent between runs, which an arbitrary choice would not be.
        return _PLANAR_RESET_CAMERA.format(sense=-1.0)
    if camera is None:
        direction, up = _DIRECTIONS[fallback]
        direction = _unit(direction)
        if abs(sum(direction[axis] * up[axis] for axis in range(3))) > 0.99:
            up = _up_for(direction)
        up = _unit(up)
        return _RESET_CAMERA.format(
            dx=direction[0], dy=direction[1], dz=direction[2],
            ux=up[0], uy=up[1], uz=up[2],
        )
    return _CAMERA.format(
        px=camera.position[0], py=camera.position[1], pz=camera.position[2],
        fx=camera.focal_point[0], fy=camera.focal_point[1], fz=camera.focal_point[2],
        ux=camera.up[0], uy=camera.up[1], uz=camera.up[2],
        scale=camera.parallel_scale,
    )


def _colour_block(field: str | None) -> str:
    if not field:
        return ""
    # Rescaled over the whole run rather than per frame, so a feature does not appear to
    # pulse because the colour bar moved underneath it between time steps.
    return f'''
from paraview.simple import ColorBy, GetColorTransferFunction

ColorBy(display, ("POINTS", {field!r}))
display.RescaleTransferFunctionToDataRange(True, False)
GetColorTransferFunction({field!r}).ApplyPreset("Cool to Warm", True)
display.SetScalarBarVisibility(view, True)
'''


def screenshot_script(request: RenderRequest, camera: Camera | None) -> str:
    """A pvbatch script that saves one image of the mesh.

    ``Surface With Edges`` because the point of a mesh screenshot is the mesh: a plain
    surface shows the shape and tells you nothing about the grid on it.
    """
    target = request.output / f"{request.name}.png"
    return (
        _preamble(request, representation="Surface With Edges")
        + _colour_block(request.field)
        + _camera_block(camera, request.preset, planar=request.planar)
        + f'\nSaveScreenshot({str(target)!r}, view, ImageResolution=['
        f"{request.size[0]:d}, {request.size[1]:d}])\n"
        f"print({str(target)!r})\n"
    )


_ANIMATION = """
from paraview.simple import GetAnimationScene

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(reader.TimestepValues or [0.0]){slice}

for index, time in enumerate(times):
    view.ViewTime = time
    scene.AnimationTime = time
    UpdatePipeline(time=time, proxy=reader)
    SaveScreenshot(
        {pattern!r} % index,
        view,
        ImageResolution=[{width:d}, {height:d}],
    )

print("%d frame(s) in %s" % (len(times), {output!r}))
print("ffmpeg -framerate 24 -i {pattern} -pix_fmt yuv420p {video}")
"""


def animation_script(request: RenderRequest, camera: Camera | None) -> str:
    """A pvbatch script that saves one image per written time step.

    A numbered PNG series rather than a video file, deliberately. ``SaveAnimation`` can write
    ``.avi`` or ``.ogv``, but only if that ParaView was built with the encoder, and a feature
    that works on one install and fails on the next with a message about codecs is worse than
    one that always produces frames. Frames are also what anybody wanting a video will feed
    to ``ffmpeg`` anyway, so the script prints the command that does it.
    """
    pattern = request.output / f"{request.name}.%04d.png"
    return (
        _preamble(request, representation="Surface")
        + _colour_block(request.field or "p")
        + _camera_block(camera, request.preset, planar=request.planar)
        + _ANIMATION.format(
            slice="" if request.frames is None else f"[:{request.frames:d}]",
            pattern=str(pattern),
            width=request.size[0],
            height=request.size[1],
            output=str(request.output),
            video=str(request.output / f"{request.name}.mp4"),
        )
    )


def presets() -> Sequence[str]:
    """Every preset name, for a chooser."""
    return tuple(preset.value for preset in CameraPreset)
