"""Rendering a case with ParaView: cameras, scripts and commands.

ParaView is not assumed to be installed, so nothing here runs it. What is asserted is the
part that would otherwise only be testable by looking at pictures: where the camera ends up,
which angle is actually used for a two-dimensional case, and what the generated script and
command line say.

The camera arithmetic is pure, which is why it is worth having as a function at all -- a
fixed distance works for exactly one mesh, and "it looked right on my case" is not a test.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dispatch.adapters.base import CaseContext
from dispatch.adapters.openfoam import OpenFOAMAdapter
from dispatch.adapters.paraview import (
    MARGIN,
    OUTPUT_DIR,
    PARAVIEW_BINARIES,
    Camera,
    CameraPreset,
    RenderRequest,
    animation_script,
    camera_for,
    find_paraview,
    presets,
    render_argv,
    resolve_preset,
    screenshot_script,
)
from dispatch.core.errors import ValidationError
from dispatch.core.geometry import Bounds, CaseGeometry, Dimensionality
from dispatch.core.visual import VisualKind, VisualPlan, VisualRequest
from tests.unit.test_caseinfo import BOUNDARY_3D, case

WIDE = Bounds((0.0, 0.0, 0.0), (20.0, 10.0, 1.0))
CUBE = Bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

PLANAR = CaseGeometry(bounds=WIDE, dimensionality=Dimensionality.TWO_D, normal=2)
SOLID = CaseGeometry(bounds=CUBE, dimensionality=Dimensionality.THREE_D)


def distance(camera: Camera) -> float:
    return math.dist(camera.position, camera.focal_point)


# -- finding the renderer --------------------------------------------------------------


def test_the_gui_binary_is_never_used() -> None:
    """Dispatch runs on a machine reached over SSH; there is no display to put it on."""
    assert "paraview" not in PARAVIEW_BINARIES
    assert PARAVIEW_BINARIES[0] == "pvbatch"


def test_a_missing_paraview_is_reported_rather_than_guessed() -> None:
    assert find_paraview({"PATH": "/nonexistent"}) is None


def test_the_job_s_own_path_is_searched(tmp_path: Path) -> None:
    """A tool that exists only after an environment is sourced is not on the daemon's path."""
    fake = tmp_path / "pvbatch"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    assert find_paraview({"PATH": str(tmp_path)}) == str(fake)


def test_the_command_renders_offscreen() -> None:
    argv = render_argv("/usr/bin/pvbatch", Path("/tmp/x.py"))
    assert argv[0] == "/usr/bin/pvbatch"
    assert "--force-offscreen-rendering" in argv
    assert argv[-1] == "/tmp/x.py"


# -- camera placement ----------------------------------------------------------------------


def test_the_camera_looks_at_the_centre_of_the_mesh() -> None:
    camera = camera_for(WIDE, CameraPreset.TOP, geometry=PLANAR)
    assert camera is not None
    assert camera.focal_point == pytest.approx((10.0, 5.0, 0.5))


def test_the_distance_comes_from_the_bounding_diagonal() -> None:
    """Not from a single extent: the diagonal bounds the mesh from every direction, so the
    same distance cannot crop it whichever angle was chosen."""
    camera = camera_for(CUBE, CameraPreset.TOP, geometry=SOLID)
    assert camera is not None
    assert distance(camera) == pytest.approx(CUBE.diagonal * MARGIN)


def test_the_whole_mesh_fits_whichever_angle_is_used() -> None:
    """The property that matters, asserted directly rather than by eye."""
    for preset in CameraPreset:
        camera = camera_for(CUBE, preset, geometry=SOLID)
        assert camera is not None
        # Every corner of the box is inside the half-extent the view covers.
        for corner_x in (CUBE.minimum[0], CUBE.maximum[0]):
            for corner_y in (CUBE.minimum[1], CUBE.maximum[1]):
                for corner_z in (CUBE.minimum[2], CUBE.maximum[2]):
                    offset = math.dist((corner_x, corner_y, corner_z), camera.focal_point)
                    assert offset <= camera.parallel_scale + 1e-9


def test_a_bigger_mesh_gets_a_bigger_distance() -> None:
    """The alternative -- a fixed distance -- works for exactly one mesh."""
    small = camera_for(CUBE, CameraPreset.TOP, geometry=SOLID)
    large = camera_for(
        Bounds((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)), CameraPreset.TOP
    )
    assert small is not None and large is not None
    assert distance(large) > distance(small) * 100


def test_the_margin_leaves_room_around_the_mesh() -> None:
    camera = camera_for(CUBE, CameraPreset.TOP, geometry=SOLID)
    assert camera is not None
    assert camera.parallel_scale > CUBE.diagonal / 2


def test_the_up_vector_is_never_parallel_to_the_view() -> None:
    """A camera looking along its own up vector has no defined orientation."""
    for preset in CameraPreset:
        camera = camera_for(CUBE, preset, geometry=SOLID)
        assert camera is not None
        direction = [
            camera.focal_point[axis] - camera.position[axis] for axis in range(3)
        ]
        length = math.sqrt(sum(value * value for value in direction))
        dot = sum(direction[axis] / length * camera.up[axis] for axis in range(3))
        assert abs(dot) < 0.99


def test_unknown_bounds_yield_no_camera() -> None:
    """An imported mesh Dispatch cannot measure: ParaView frames it instead."""
    assert camera_for(None, CameraPreset.TOP) is None


def test_a_degenerate_box_yields_no_camera() -> None:
    point = Bounds((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert camera_for(point, CameraPreset.TOP) is None


# -- two-dimensional cases ------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset", [CameraPreset.FRONT, CameraPreset.BACK, CameraPreset.LEFT, CameraPreset.RIGHT]
)
def test_an_edge_on_angle_is_substituted_for_a_planar_case(preset: CameraPreset) -> None:
    """A 2-D OpenFOAM case is a mesh one cell thick: four of the seven angles see a line."""
    chosen, substituted = resolve_preset(preset, PLANAR)
    assert substituted
    assert chosen in (CameraPreset.TOP, CameraPreset.BOTTOM)


def test_isometric_is_substituted_for_a_planar_case() -> None:
    """An isometric view of a flat plate is a sliver."""
    chosen, substituted = resolve_preset(CameraPreset.ISOMETRIC, PLANAR)
    assert substituted and chosen is CameraPreset.TOP


@pytest.mark.parametrize("preset", [CameraPreset.TOP, CameraPreset.BOTTOM])
def test_an_angle_that_already_faces_the_plane_is_kept(preset: CameraPreset) -> None:
    chosen, substituted = resolve_preset(preset, PLANAR)
    assert not substituted and chosen is preset


def test_the_sense_of_the_request_is_preserved() -> None:
    """Asking for a negative-facing view twice should give the same answer both times."""
    assert resolve_preset(CameraPreset.BACK, PLANAR)[0] is CameraPreset.BOTTOM
    assert resolve_preset(CameraPreset.FRONT, PLANAR)[0] is CameraPreset.TOP


def test_a_volumetric_case_is_never_substituted() -> None:
    for preset in CameraPreset:
        assert resolve_preset(preset, SOLID) == (preset, False)


def test_a_planar_case_with_an_unknown_normal_is_left_alone() -> None:
    """Substituting on a guess would be worse than honouring the request."""
    unknown = CaseGeometry(dimensionality=Dimensionality.TWO_D)
    assert resolve_preset(CameraPreset.FRONT, unknown) == (CameraPreset.FRONT, False)


def test_the_camera_reports_that_it_was_substituted() -> None:
    """A substitution that is not reported is a wrong answer."""
    camera = camera_for(WIDE, CameraPreset.FRONT, geometry=PLANAR)
    assert camera is not None
    assert camera.substituted and camera.preset is CameraPreset.TOP

    kept = camera_for(CUBE, CameraPreset.FRONT, geometry=SOLID)
    assert kept is not None and not kept.substituted


def test_a_planar_case_normal_to_x_substitutes_towards_x() -> None:
    sideways = CaseGeometry(
        bounds=Bounds((0.0, 0.0, 0.0), (0.1, 10.0, 10.0)),
        dimensionality=Dimensionality.TWO_D,
        normal=0,
    )
    assert resolve_preset(CameraPreset.TOP, sideways)[0] is CameraPreset.RIGHT


# -- generated scripts -------------------------------------------------------------------------


def request_for(tmp_path: Path, **kwargs) -> RenderRequest:
    return RenderRequest(
        reader=tmp_path / "case.foam", output=tmp_path / "out", **kwargs
    )


def test_a_screenshot_script_is_valid_python(tmp_path: Path) -> None:
    """The cheapest guard against a template that renders but does not parse."""
    script = screenshot_script(
        request_for(tmp_path), camera_for(CUBE, CameraPreset.TOP, geometry=SOLID)
    )
    compile(script, "<screenshot>", "exec")


def test_an_animation_script_is_valid_python(tmp_path: Path) -> None:
    script = animation_script(
        request_for(tmp_path), camera_for(CUBE, CameraPreset.TOP, geometry=SOLID)
    )
    compile(script, "<animation>", "exec")


def test_the_script_opens_the_reader_it_was_given(tmp_path: Path) -> None:
    script = screenshot_script(request_for(tmp_path), None)
    assert repr(str(tmp_path / "case.foam")) in script
    assert "OpenFOAMReader" in script


def test_a_decomposed_case_is_read_as_one(tmp_path: Path) -> None:
    script = screenshot_script(request_for(tmp_path, decomposed=True), None)
    assert "Decomposed Case" in script
    assert "Decomposed Case" not in screenshot_script(request_for(tmp_path), None)


def test_the_camera_reaches_the_script(tmp_path: Path) -> None:
    camera = camera_for(CUBE, CameraPreset.TOP, geometry=SOLID)
    script = screenshot_script(request_for(tmp_path), camera)
    assert camera is not None
    assert "CameraPosition" in script and "ResetCamera" not in script
    assert repr(camera.parallel_scale) in script


def test_without_bounds_paraview_frames_it_but_the_angle_is_still_honoured(
    tmp_path: Path,
) -> None:
    """Direction from the request, distance from ParaView -- which has the mesh open.

    Dropping the angle too would mean a case whose mesh Dispatch cannot measure ignores the
    camera preset entirely, including one substituted because the case is planar.
    """
    script = screenshot_script(request_for(tmp_path, preset=CameraPreset.TOP), None)
    assert "ResetCamera()" in script
    assert "CameraParallelScale" not in script, "the distance is ParaView's to decide"
    assert "view.CameraPosition = [0.0, 0.0, 1.0]" in script, "looking down the z axis"


def test_the_fallback_angle_follows_the_preset(tmp_path: Path) -> None:
    looking_along_x = screenshot_script(request_for(tmp_path, preset=CameraPreset.RIGHT), None)
    assert "view.CameraPosition = [1.0, 0.0, 0.0]" in looking_along_x


def test_a_mesh_screenshot_shows_the_mesh(tmp_path: Path) -> None:
    """A plain surface tells you nothing about the grid, which is the point of the image."""
    assert "Surface With Edges" in screenshot_script(request_for(tmp_path), None)


def test_colouring_by_a_field_is_requested_when_asked(tmp_path: Path) -> None:
    script = screenshot_script(request_for(tmp_path, field="U"), None)
    assert "ColorBy" in script and "'U'" in script
    assert "ColorBy" not in screenshot_script(request_for(tmp_path), None)


def test_an_animation_writes_frames_into_a_directory_of_its_own(tmp_path: Path) -> None:
    """Frames, then a video from them: the frames are what took the hours, so they live
    apart from the video and survive a failed encode."""
    request = request_for(tmp_path, name="flow")
    script = animation_script(request, None)
    assert str(request.frame_pattern) in script
    assert request.frame_pattern.name == "frame.%05d.png"
    assert request.frame_dir == tmp_path / "out" / "flow.frames"


def test_an_animation_can_be_capped(tmp_path: Path) -> None:
    assert "[:25]" in animation_script(request_for(tmp_path, frames=25), None)
    assert "[:" not in animation_script(request_for(tmp_path), None)


def test_the_image_size_reaches_the_script(tmp_path: Path) -> None:
    script = screenshot_script(request_for(tmp_path, size=(800, 600)), None)
    assert "[800, 600]" in script


def test_every_preset_is_offered_by_name() -> None:
    assert set(presets()) == {
        "front", "back", "left", "right", "top", "bottom", "isometric",
    }


# -- through the adapter ------------------------------------------------------------------------


@pytest.fixture
def pvbatch(tmp_path: Path, monkeypatch):
    """A fake pvbatch on the PATH, so the plan can be built without ParaView installed."""
    binary = tmp_path / "bin" / "pvbatch"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return {"PATH": str(binary.parent)}


def test_the_adapter_plans_a_render(tmp_path: Path, pvbatch) -> None:
    root = case(tmp_path)
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest(kind=VisualKind.MESH))

    assert isinstance(plan, VisualPlan)
    assert plan.steps[0].program.endswith("pvbatch")
    assert plan.steps[0].cwd == root
    assert plan.steps[0].timeout_s is not None, "a renderer with no display waits forever"
    assert (root / OUTPUT_DIR).is_dir()


def test_a_missing_paraview_plans_nothing(tmp_path: Path) -> None:
    """So the caller can say "ParaView was not found" rather than report a failed command."""
    ctx = CaseContext(workdir=case(tmp_path), cores=1, env={"PATH": "/nonexistent"})
    assert OpenFOAMAdapter({}).visualise(ctx, VisualRequest()) is None


def test_an_unknown_preset_is_refused_with_the_valid_ones(tmp_path: Path, pvbatch) -> None:
    ctx = CaseContext(workdir=case(tmp_path), cores=1, env=pvbatch)
    with pytest.raises(ValidationError, match="isometric"):
        OpenFOAMAdapter({}).visualise(ctx, VisualRequest(preset="sideways"))


def test_the_substitution_is_reported_in_the_plan(tmp_path: Path, pvbatch) -> None:
    root = case(tmp_path)  # BOUNDARY declares an empty patch, so the case is planar
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest(preset="front"))
    assert plan is not None
    assert any("2D" in note and "edge-on" in note for note in plan.notes)


def test_a_three_dimensional_case_is_not_substituted(tmp_path: Path, pvbatch) -> None:
    root = case(tmp_path, boundary=BOUNDARY_3D)
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest(preset="front"))
    assert plan is not None
    assert not any("edge-on" in note for note in plan.notes)
    assert plan.outputs[0].name == "mesh-front.png"


def test_the_output_name_always_includes_the_angle(tmp_path: Path, pvbatch) -> None:
    """Predictable: it must not depend on whether a blockMeshDict happened to be readable."""
    root = case(tmp_path, block_mesh=None, boundary=BOUNDARY_3D)
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest(preset="left"))
    assert plan is not None and plan.outputs[0].name == "mesh-left.png"


def test_a_reader_stub_is_created_when_the_case_has_none(tmp_path: Path, pvbatch) -> None:
    root = case(tmp_path)
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    OpenFOAMAdapter({}).visualise(ctx, VisualRequest())
    assert (root / f"{root.name}.foam").is_file()


def test_an_existing_stub_is_reused(tmp_path: Path, pvbatch) -> None:
    root = case(tmp_path)
    # Deliberately not named after the directory, so "reused" is distinguishable from
    # "created with the name it would have had anyway".
    (root / "wing.foam").touch()
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest())
    assert plan is not None
    assert "wing.foam" in Path(plan.steps[0].argv[-1]).read_text()
    assert not (root / f"{root.name}.foam").exists()


def test_the_dispatch_log_is_never_handed_to_paraview(tmp_path: Path, pvbatch) -> None:
    """`log.foam` is Dispatch's own working-directory log (§6.4) and matches `*.foam`.

    Opening it as a case would fail in a way nobody would connect to the log convention.
    """
    root = case(tmp_path)
    (root / "log.foam").write_text("Time = 0.1\nsmoothSolver: Solving for Ux ...\n")
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest())

    assert plan is not None
    script = Path(plan.steps[0].argv[-1]).read_text()
    assert "log.foam" not in script
    assert f"{root.name}.foam" in script


def test_a_preview_writes_nothing_into_the_case(tmp_path: Path, pvbatch) -> None:
    root = case(tmp_path)
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch, dry_run=True)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest())

    assert plan is not None, "the plan is still described"
    assert not (root / OUTPUT_DIR).exists()
    assert not (root / f"{root.name}.foam").exists()


def test_a_decomposed_case_is_noted_and_read_as_one(tmp_path: Path, pvbatch) -> None:
    root = case(tmp_path, processors=4)
    ctx = CaseContext(workdir=root, cores=4, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest())
    assert plan is not None
    assert any("decomposed" in note for note in plan.notes)
    assert "Decomposed Case" in Path(plan.steps[0].argv[-1]).read_text()


def test_an_adapter_that_cannot_render_plans_nothing(tmp_path: Path) -> None:
    """The default, so no existing adapter had to change."""
    from dispatch.adapters.su2 import SU2Adapter

    assert SU2Adapter({}).visualise(CaseContext(workdir=tmp_path, cores=1), VisualRequest()) is None


def test_renders_land_under_post_processing(tmp_path: Path, pvbatch) -> None:
    """Where a case already keeps things derived from its results, in our own subdirectory."""
    root = case(tmp_path)
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest())
    assert plan is not None
    assert plan.outputs[0].parent == root / OUTPUT_DIR


# -- a planar case whose extent cannot be read ------------------------------------------------
#
# The common case, not an edge case: a snappyHexMesh or imported mesh declares itself 2-D in
# its boundary file and has no blockMeshDict to measure. Dispatch therefore knows the case is
# planar but not which way the plane faces, and refuses to guess -- so the decision is
# deferred into the script, where ParaView has the mesh open and its bounds are exact.


def test_a_planar_case_without_bounds_defers_the_camera_to_the_script(
    tmp_path: Path,
) -> None:
    script = screenshot_script(
        request_for(tmp_path, preset=CameraPreset.FRONT, planar=True), None
    )
    compile(script, "<planar>", "exec")
    assert "GetDataInformation().GetBounds()" in script
    assert "_extents.index(min(_extents))" in script, "the thinnest axis is the normal"
    assert "ResetCamera()" in script


def test_the_deferred_camera_reports_which_axis_it_chose(tmp_path: Path) -> None:
    """So a user can tell a correct render from a lucky one."""
    script = screenshot_script(request_for(tmp_path, planar=True), None)
    assert "2D case: rendering along axis" in script


def test_a_volumetric_case_without_bounds_keeps_the_requested_angle(
    tmp_path: Path,
) -> None:
    script = screenshot_script(
        request_for(tmp_path, preset=CameraPreset.RIGHT, planar=False), None
    )
    assert "GetDataInformation" not in script
    assert "view.CameraPosition = [1.0, 0.0, 0.0]" in script


def test_a_planar_case_with_bounds_needs_no_deferral(tmp_path: Path) -> None:
    """Dispatch already knows the normal, so the camera is fixed before ParaView starts."""
    camera = camera_for(WIDE, CameraPreset.FRONT, geometry=PLANAR)
    script = screenshot_script(request_for(tmp_path, planar=True), camera)
    assert "GetDataInformation" not in script
    assert "CameraParallelScale" in script


def test_the_adapter_flags_a_planar_case_to_the_script(tmp_path: Path, pvbatch) -> None:
    """End to end: no blockMeshDict, an empty patch, and the script works it out."""
    root = case(tmp_path, block_mesh=None)
    ctx = CaseContext(workdir=root, cores=1, env=pvbatch)
    plan = OpenFOAMAdapter({}).visualise(ctx, VisualRequest(preset="front"))

    assert plan is not None
    assert any("2D" in note for note in plan.notes)
    assert "GetDataInformation" in Path(plan.steps[0].argv[-1]).read_text()


def test_an_animation_of_a_planar_case_defers_too(tmp_path: Path) -> None:
    script = animation_script(request_for(tmp_path, planar=True), None)
    compile(script, "<planar-animation>", "exec")
    assert "GetDataInformation().GetBounds()" in script
