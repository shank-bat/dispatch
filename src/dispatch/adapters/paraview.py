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

import functools
import logging
import math
import re
import shutil
import subprocess
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
    "encode_argv",
    "find_ffmpeg",
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

PNG_COMPRESSION: Final = "0"
"""zlib level for every PNG ParaView writes: always zero, i.e. stored uncompressed.

Required, not a tuning choice. Compression in ParaView's PNG writer is single-threaded and
sits inside the per-frame loop, so on a long animation it is a large share of the render
time; and the frames are an intermediate that ffmpeg reads straight back. Every
``SaveScreenshot`` call goes through :func:`_save_screenshot`, so no template can forget it.
"""

FFMPEG_BINARY: Final = "ffmpeg"

VIDEO_ENCODERS: Final = (
    ("libx264", ("-c:v", "libx264", "-preset", "medium", "-crf", "18")),
    ("libopenh264", ("-c:v", "libopenh264", "-b:v", "12M")),
    ("mpeg4", ("-c:v", "mpeg4", "-q:v", "2")),
)
"""H.264 encoders in preference order, with a widely-built MPEG-4 fallback.

``libx264`` at CRF 18 is visually lossless for a CFD render. Distribution builds that cannot
ship it usually have ``libopenh264``; ``mpeg4`` is in essentially every ffmpeg ever built.
"""

DEFAULT_FPS: Final = 24

ANIMATION_TIMEOUT_S: Final = 6 * 3600.0
"""Rendering every written time of a large case is hours of work, not minutes."""

ENCODE_TIMEOUT_S: Final = 2 * 3600.0

BYTES_PER_PIXEL: Final = 3
"""An uncompressed RGB PNG is essentially width x height x 3 bytes; used to warn about disk."""

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
        frames_dir: Where an animation's frames go. A directory of their own, cleared before
            each render, so a re-render that produces fewer frames cannot leave stale ones
            behind for ffmpeg to splice onto the end of the new video.
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
    frames_dir: Path | None = None

    @property
    def frame_dir(self) -> Path:
        """The animation's frame directory, defaulting beside the video."""
        return self.frames_dir or self.output / f"{self.name}.frames"

    @property
    def frame_pattern(self) -> Path:
        """``frame.%05d.png`` inside :attr:`frame_dir` -- the form both ParaView and ffmpeg use.

        Five digits: four overflows at 10,000 frames, which a long transient run reaches.
        """
        return self.frame_dir / "frame.%05d.png"


def render_argv(binary: str, script: Path) -> list[str]:
    """The command that runs a generated script headlessly.

    ``--force-offscreen-rendering`` so a ParaView built against a windowing system does not
    try to find a display that is not there. Harmless on a build without it.
    """
    return [binary, "--force-offscreen-rendering", str(script)]


def find_ffmpeg(env: Mapping[str, str] | None = None) -> str | None:
    """ffmpeg on the job's own ``PATH``, or ``None``."""
    return shutil.which(FFMPEG_BINARY, path=(env or {}).get("PATH"))


@functools.lru_cache(maxsize=8)
def video_encoder(ffmpeg: str) -> tuple[str, tuple[str, ...]]:
    """The best H.264 (or MPEG-4) encoder this ffmpeg was built with.

    Asked once per binary and cached: the answer cannot change while the daemon runs, and
    probing is a subprocess. A probe that fails falls back to ``mpeg4``, which every build
    has, rather than refusing to encode at all.
    """
    try:
        listing = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        listing = ""
    available = {line.split()[1] for line in listing.splitlines() if len(line.split()) > 1}
    for name, arguments in VIDEO_ENCODERS:
        if name in available:
            return name, arguments
    return VIDEO_ENCODERS[-1]


def encode_argv(
    ffmpeg: str,
    request: RenderRequest,
    video: Path,
    *,
    fps: int = DEFAULT_FPS,
    encoder: tuple[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """The ffmpeg command that joins an animation's frames into a video.

    ``yuv420p`` because it is the only pixel format every player decodes; the ``pad`` filter
    because yuv420p needs even dimensions and a user-chosen size may be odd. ``+faststart``
    puts the index at the front so the file plays while it is still being copied off the
    machine.
    """
    _, arguments = encoder or video_encoder(ffmpeg)
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(max(1, fps)),
        "-start_number",
        "0",
        "-i",
        str(request.frame_pattern),
        *arguments,
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
        str(video),
    ]


def frame_bytes(size: tuple[int, int]) -> int:
    """Disk taken by one uncompressed frame, for warning before an animation starts."""
    return size[0] * size[1] * BYTES_PER_PIXEL


def safe_name(text: str) -> str:
    """A field name made safe for a filename: ``alpha.water`` stays, ``grad(p)`` -> ``grad_p_``."""
    return re.sub(r"[^\w.\-]", "_", text) or "field"


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
# Include the initial condition: the reader skips time 0 by default. Setting the property is
# not enough on its own -- the reader listed its time steps when it was created, and only
# re-reads the case directory when told to reload, so without this every animation would
# still silently start one write into the run.
reader.SkipZeroTime = 0
try:
    from paraview.simple import ReloadFiles

    ReloadFiles(reader)
except Exception:  # pragma: no cover - very old ParaView
    reader.Refresh()
reader.UpdatePipelineInformation()
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
if hasattr(view, "BackgroundColorMode"):
    view.BackgroundColorMode = "Single Color"

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


_COLOUR = '''
from paraview.simple import ColorBy, GetColorTransferFunction

# Point or cell data, decided here rather than assumed: the OpenFOAM reader offers fields as
# cell arrays, and only interpolates to points when asked, so a hard-coded "POINTS" fails on
# exactly the fields a user picks.
_field = {field!r}
if _field in reader.PointData.keys():
    _association = "POINTS"
elif _field in reader.CellData.keys():
    _association = "CELLS"
else:
    raise SystemExit(
        "field %r is not in this case; it has %s"
        % (_field, ", ".join(sorted(set(reader.CellData.keys()) | set(reader.PointData.keys()))))
    )

ColorBy(display, (_association, _field))
_lut = GetColorTransferFunction(_field)
_lut.ApplyPreset("Cool to Warm", True)
{rescale}
display.SetScalarBarVisibility(view, True)
'''

_RESCALE_NOW = "display.RescaleTransferFunctionToDataRange(True, False)"

_RESCALE_OVER_TIME = '''\
# One colour scale for the whole run, then frozen. Rescaled per frame, the bar moves under
# the flow and every feature appears to pulse; ParaView's default also grows the range as it
# goes, so the first frames are coloured against a scale the later ones then change.
if hasattr(display, "RescaleTransferFunctionToDataRangeOverTime"):
    display.RescaleTransferFunctionToDataRangeOverTime()
else:  # pragma: no cover - ParaView before 5.6
    display.RescaleTransferFunctionToDataRange(True, False)
_lut.AutomaticRescaleRangeMode = "Never"'''


def _colour_block(field: str | None, *, over_time: bool = False) -> str:
    """Colour the surface by ``field``, or nothing for a plain mesh.

    Vectors are coloured by magnitude, which is ParaView's default for a vector array and
    the reading people want from velocity.
    """
    if not field:
        return ""
    return _COLOUR.format(field=field, rescale=_RESCALE_OVER_TIME if over_time else _RESCALE_NOW)


def _save_screenshot(target: str, size: tuple[int, int]) -> str:
    """One ``SaveScreenshot`` call, always uncompressed. See :data:`PNG_COMPRESSION`."""
    return (
        f"SaveScreenshot({target}, view, ImageResolution=[{size[0]:d}, {size[1]:d}], "
        f"CompressionLevel={PNG_COMPRESSION!r})"
    )


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
        + "\n"
        + _save_screenshot(repr(str(target)), request.size)
        + f"\nprint({str(target)!r})\n"
    )


_ANIMATION = """
import glob

from paraview.simple import GetAnimationScene

# A directory of its own, emptied first: a re-render that produces fewer frames must not
# leave the old tail behind for ffmpeg to splice onto the new video.
os.makedirs({frames_dir!r}, exist_ok=True)
for _stale in glob.glob(os.path.join({frames_dir!r}, "frame.*.png")):
    os.remove(_stale)

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(reader.TimestepValues or [0.0]){slice}

for index, time in enumerate(times):
    view.ViewTime = time
    scene.AnimationTime = time
    UpdatePipeline(time=time, proxy=reader)
    {save}
    print("frame %d/%d  t = %g" % (index + 1, len(times), time), flush=True)

print("%d frame(s) in %s" % (len(times), {frames_dir!r}))
"""


def animation_script(request: RenderRequest, camera: Camera | None) -> str:
    """A pvbatch script that saves one uncompressed frame per written time step.

    Frames only: joining them into a video is ffmpeg's job, as a separate step of the same
    plan (:func:`encode_argv`). Keeping the two apart means a failed encode -- a missing
    codec, a full disk -- leaves the frames, which took the hours, intact.
    """
    return (
        _preamble(request, representation="Surface")
        + _colour_block(request.field, over_time=True)
        + _camera_block(camera, request.preset, planar=request.planar)
        + _ANIMATION.format(
            slice="" if request.frames is None else f"[:{request.frames:d}]",
            frames_dir=str(request.frame_dir),
            save=_save_screenshot(f"{str(request.frame_pattern)!r} % index", request.size),
        )
    )


def presets() -> Sequence[str]:
    """Every preset name, for a chooser."""
    return tuple(preset.value for preset in CameraPreset)
