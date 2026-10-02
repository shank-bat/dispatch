"""Animations as video, frames as uncompressed PNG, and choosing what to colour by.

No ParaView or ffmpeg is needed: scripts and command lines are asserted as text and argv,
fields are read from text files, and the render manager is exercised with shell scripts
standing in for the tools. The real tools were exercised by hand against a hand-built mesh;
these tests pin the properties that matter so a later edit cannot quietly lose them.
"""

from __future__ import annotations

import asyncio
import gzip
import re
from pathlib import Path

import pytest

from dispatch.adapters.base import CaseContext
from dispatch.adapters.foammesh import render_fields
from dispatch.adapters.openfoam import OpenFOAMAdapter
from dispatch.adapters.paraview import (
    PNG_COMPRESSION,
    VIDEO_ENCODERS,
    CameraPreset,
    RenderRequest,
    animation_script,
    camera_for,
    encode_argv,
    find_ffmpeg,
    frame_bytes,
    safe_name,
    screenshot_script,
    video_encoder,
)
from dispatch.core.errors import ValidationError
from dispatch.core.geometry import Bounds
from dispatch.core.plan import CommandStep, StepKind
from dispatch.core.visual import VisualKind, VisualPlan, VisualRequest
from dispatch.daemon.events import EventBus
from dispatch.daemon.renders import RenderManager
from dispatch.ipc.protocol import Event
from tests.unit.test_caseinfo import BOUNDARY_3D, case

CUBE = Bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))


def request_for(tmp_path: Path, **kwargs) -> RenderRequest:
    return RenderRequest(reader=tmp_path / "case.foam", output=tmp_path / "out", **kwargs)


def screenshot_calls(script: str) -> list[str]:
    return re.findall(r"SaveScreenshot\([^\n]*\)", script)


# -- zero compression, always -------------------------------------------------------------


def test_the_compression_level_is_zero() -> None:
    assert PNG_COMPRESSION == "0"


@pytest.mark.parametrize("field", [None, "p"])
def test_every_still_is_saved_uncompressed(tmp_path: Path, field: str | None) -> None:
    calls = screenshot_calls(screenshot_script(request_for(tmp_path, field=field), None))
    assert calls, "the script saves an image"
    assert all("CompressionLevel='0'" in call for call in calls)


@pytest.mark.parametrize("planar", [False, True])
def test_every_animation_frame_is_saved_uncompressed(tmp_path: Path, planar: bool) -> None:
    script = animation_script(request_for(tmp_path, field="U", planar=planar), None)
    calls = screenshot_calls(script)
    assert calls
    assert all("CompressionLevel='0'" in call for call in calls)


def test_no_template_can_save_without_the_compression_setting(tmp_path: Path) -> None:
    """Every variant, with and without a camera, field and plane: none may omit it."""
    camera = camera_for(CUBE, CameraPreset.TOP)
    for builder in (screenshot_script, animation_script):
        for field in (None, "p"):
            for planar in (False, True):
                for cam in (camera, None):
                    script = builder(request_for(tmp_path, field=field, planar=planar), cam)
                    total = script.count("SaveScreenshot(") - script.count("import")
                    assert len(screenshot_calls(script)) >= 1
                    assert all(
                        "CompressionLevel='0'" in call for call in screenshot_calls(script)
                    ), (builder.__name__, field, planar, total)


def test_an_uncompressed_frame_is_budgeted_at_three_bytes_a_pixel() -> None:
    assert frame_bytes((1600, 1000)) == 4_800_000


# -- the animation script ------------------------------------------------------------------


def test_stale_frames_are_removed_before_rendering(tmp_path: Path) -> None:
    """A shorter re-render must not leave the old tail for ffmpeg to splice on."""
    script = animation_script(request_for(tmp_path), None)
    assert "os.remove(_stale)" in script
    assert script.index("os.remove(_stale)") < script.index("for index, time in enumerate")


def test_the_initial_condition_is_included(tmp_path: Path) -> None:
    """Setting SkipZeroTime is not enough: the reader must reload its time list."""
    script = screenshot_script(request_for(tmp_path), None)
    assert "reader.SkipZeroTime = 0" in script
    assert script.index("SkipZeroTime = 0") < script.index("ReloadFiles(reader)")


def test_the_colour_scale_is_fixed_over_the_whole_run(tmp_path: Path) -> None:
    """Otherwise the bar moves under the flow and every feature appears to pulse."""
    script = animation_script(request_for(tmp_path, field="p"), None)
    assert "RescaleTransferFunctionToDataRangeOverTime" in script
    assert 'AutomaticRescaleRangeMode = "Never"' in script


def test_a_still_is_scaled_to_its_own_data(tmp_path: Path) -> None:
    script = screenshot_script(request_for(tmp_path, field="p"), None)
    assert "OverTime" not in script


def test_point_or_cell_data_is_chosen_at_render_time(tmp_path: Path) -> None:
    """The OpenFOAM reader offers fields as cell arrays; a fixed "POINTS" fails on them."""
    script = screenshot_script(request_for(tmp_path, field="U"), None)
    assert '_association = "CELLS"' in script
    assert '_association = "POINTS"' in script
    assert "ColorBy(display, (_association, _field))" in script


def test_a_missing_field_stops_the_render_with_the_alternatives(tmp_path: Path) -> None:
    script = screenshot_script(request_for(tmp_path, field="U"), None)
    assert "raise SystemExit" in script and "is not in this case" in script


def test_frame_progress_is_printed_for_the_daemon_to_read(tmp_path: Path) -> None:
    script = animation_script(request_for(tmp_path), None)
    assert 'print("frame %d/%d' in script


@pytest.mark.parametrize("builder", [screenshot_script, animation_script])
@pytest.mark.parametrize("field", [None, "U", "alpha.water"])
def test_every_generated_script_is_valid_python(tmp_path: Path, builder, field) -> None:
    compile(builder(request_for(tmp_path, field=field), None), "<script>", "exec")


# -- encoding ---------------------------------------------------------------------------------


def argv(tmp_path: Path, **kwargs) -> list[str]:
    return encode_argv(
        "/usr/bin/ffmpeg",
        request_for(tmp_path),
        tmp_path / "out" / "video.mp4",
        encoder=VIDEO_ENCODERS[0],
        **kwargs,
    )


def test_the_encode_reads_the_frames_the_script_writes(tmp_path: Path) -> None:
    command = argv(tmp_path)
    assert command[command.index("-i") + 1] == str(request_for(tmp_path).frame_pattern)
    assert command[command.index("-start_number") + 1] == "0"


def test_the_video_plays_everywhere(tmp_path: Path) -> None:
    """yuv420p is the pixel format every player decodes, and it needs even dimensions."""
    command = argv(tmp_path)
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert "pad=ceil(iw/2)*2:ceil(ih/2)*2" in command
    assert "+faststart" in command


def test_the_encode_overwrites_rather_than_prompting(tmp_path: Path) -> None:
    """A prompt with no terminal would wait forever."""
    assert "-y" in argv(tmp_path)


def test_the_frame_rate_is_honoured(tmp_path: Path) -> None:
    command = argv(tmp_path, fps=12)
    assert command[command.index("-framerate") + 1] == "12"
    assert argv(tmp_path, fps=0)[argv(tmp_path, fps=0).index("-framerate") + 1] == "1"


def test_the_output_is_the_last_argument(tmp_path: Path) -> None:
    assert argv(tmp_path)[-1] == str(tmp_path / "out" / "video.mp4")


def fake_ffmpeg(tmp_path: Path, encoders: str) -> str:
    binary = tmp_path / "bin" / "ffmpeg"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(f"#!/bin/sh\ncat <<'EOF'\nEncoders:\n{encoders}\nEOF\n")
    binary.chmod(0o755)
    return str(binary)


def test_h264_is_preferred_when_available(tmp_path: Path) -> None:
    video_encoder.cache_clear()
    ffmpeg = fake_ffmpeg(tmp_path, " V....D libx264  H.264\n V.S..D mpeg4  MPEG-4")
    assert video_encoder(ffmpeg)[0] == "libx264"


def test_openh264_is_the_second_choice(tmp_path: Path) -> None:
    video_encoder.cache_clear()
    ffmpeg = fake_ffmpeg(tmp_path, " V....D libopenh264  OpenH264\n V.S..D mpeg4  MPEG-4")
    assert video_encoder(ffmpeg)[0] == "libopenh264"


def test_mpeg4_is_the_fallback_when_probing_fails(tmp_path: Path) -> None:
    video_encoder.cache_clear()
    assert video_encoder(str(tmp_path / "does-not-exist"))[0] == "mpeg4"


def test_ffmpeg_is_looked_for_on_the_job_s_path(tmp_path: Path) -> None:
    ffmpeg = fake_ffmpeg(tmp_path, "")
    assert find_ffmpeg({"PATH": str(Path(ffmpeg).parent)}) == ffmpeg
    assert find_ffmpeg({"PATH": "/nonexistent"}) is None


@pytest.mark.parametrize(
    ("name", "safe"), [("p", "p"), ("alpha.water", "alpha.water"), ("grad(p)", "grad_p_")]
)
def test_field_names_are_made_safe_for_filenames(name: str, safe: str) -> None:
    assert safe_name(name) == safe


# -- which fields a case offers --------------------------------------------------------------


def field_file(path: Path, name: str, cls: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(f"FoamFile\n{{\n    class {cls};\n    object {name};\n}}\n")


def test_fields_come_from_the_latest_written_time(tmp_path: Path) -> None:
    """A field the solver derives as it runs exists only in written times."""
    root = tmp_path / "case"
    field_file(root / "0", "p", "volScalarField")
    field_file(root / "0.5", "p", "volScalarField")
    field_file(root / "0.5", "vorticity", "volVectorField")
    assert render_fields(root) == [("p", "scalar"), ("vorticity", "vector")]


def test_a_case_that_has_not_run_offers_its_initial_fields(tmp_path: Path) -> None:
    root = tmp_path / "case"
    field_file(root / "0", "U", "volVectorField")
    assert render_fields(root) == [("U", "vector")]


def test_surface_fields_are_not_offered(tmp_path: Path) -> None:
    """The reader does not load them onto the mesh; offering one would fail the render."""
    root = tmp_path / "case"
    field_file(root / "1", "phi", "surfaceScalarField")
    field_file(root / "1", "p", "volScalarField")
    assert render_fields(root) == [("p", "scalar")]


def test_compressed_fields_are_read(tmp_path: Path) -> None:
    root = tmp_path / "case" / "1"
    root.mkdir(parents=True)
    with gzip.open(root / "U.gz", "wb") as handle:
        handle.write(b"FoamFile\n{\n    class volVectorField;\n}\n")
    assert render_fields(tmp_path / "case") == [("U", "vector")]


def test_a_decomposed_case_s_fields_are_read_from_processor0(tmp_path: Path) -> None:
    root = tmp_path / "case"
    field_file(root / "processor0" / "2", "T", "volScalarField")
    assert render_fields(root) == [("T", "scalar")]


def test_a_case_with_no_time_directories_offers_nothing(tmp_path: Path) -> None:
    (tmp_path / "case").mkdir()
    assert render_fields(tmp_path / "case") == []


def test_the_adapter_orders_and_labels_fields_for_a_chooser(tmp_path: Path) -> None:
    root = tmp_path / "case"
    for name, cls in (
        ("k", "volScalarField"),
        ("U_0", "volVectorField"),
        ("p", "volScalarField"),
        ("U", "volVectorField"),
        ("zeta", "volScalarField"),
    ):
        field_file(root / "1", name, cls)

    fields = OpenFOAMAdapter({}).visual_fields(CaseContext(workdir=root, cores=1))
    assert [item.name for item in fields] == ["U", "p", "k", "zeta", "U_0"]
    labels = {item.name: item.display for item in fields}
    assert labels["U"] == "U (velocity, magnitude)"
    assert labels["p"] == "p (pressure)"
    assert labels["zeta"] == "zeta"


# -- the plan -----------------------------------------------------------------------------------


@pytest.fixture
def tools(tmp_path: Path) -> dict[str, str]:
    """Fake pvbatch and ffmpeg, so a plan can be built with neither installed."""
    bin_dir = tmp_path / "tools"
    bin_dir.mkdir()
    for name in ("pvbatch", "ffmpeg"):
        binary = bin_dir / name
        binary.write_text("#!/bin/sh\necho ' V....D libx264  H.264'\n")
        binary.chmod(0o755)
    video_encoder.cache_clear()
    return {"PATH": str(bin_dir)}


def foam(tmp_path: Path) -> Path:
    root = case(tmp_path, boundary=BOUNDARY_3D, fields=())
    for time in ("0", "0.1", "0.2"):
        field_file(root / time, "p", "volScalarField")
        field_file(root / time, "U", "volVectorField")
    return root


def plan_for(tmp_path: Path, env: dict[str, str], **kwargs) -> VisualPlan:
    plan = OpenFOAMAdapter({}).visualise(
        CaseContext(workdir=foam(tmp_path), cores=1, env=env), VisualRequest(**kwargs)
    )
    assert plan is not None
    return plan


def test_an_animation_renders_then_encodes(tmp_path: Path, tools) -> None:
    plan = plan_for(tmp_path, tools, kind=VisualKind.ANIMATION, field="U", preset="front")
    assert [Path(step.program).name for step in plan.steps] == ["pvbatch", "ffmpeg"]
    assert plan.outputs[0].name == "animation-U-front.mp4"
    assert "libx264" in plan.steps[1].description


def test_the_frames_are_kept_by_default(tmp_path: Path, tools) -> None:
    plan = plan_for(tmp_path, tools, kind=VisualKind.ANIMATION, field="p")
    assert len(plan.steps) == 2
    assert any("kept" in note for note in plan.notes)


def test_frames_are_only_deleted_after_a_successful_encode(tmp_path: Path, tools) -> None:
    """The delete is the last step, so a failed encode -- which stops the plan -- never
    costs the frames that took the hours."""
    plan = plan_for(tmp_path, tools, kind=VisualKind.ANIMATION, field="p", keep_frames=False)
    assert [Path(step.program).name for step in plan.steps] == ["pvbatch", "ffmpeg", "rm"]
    assert plan.steps[-1].kind is StepKind.CLEANUP
    assert plan.steps[-1].argv[-1].endswith("animation-p-isometric.frames")


def test_an_animation_without_ffmpeg_is_refused_before_rendering(tmp_path: Path) -> None:
    """Discovering it after hours of frames would leave PNGs and no video."""
    only_paraview = tmp_path / "pv"
    only_paraview.mkdir()
    (only_paraview / "pvbatch").write_text("#!/bin/sh\n")
    (only_paraview / "pvbatch").chmod(0o755)
    with pytest.raises(ValidationError, match="ffmpeg was not found"):
        plan_for(tmp_path, {"PATH": str(only_paraview)}, kind=VisualKind.ANIMATION)


def test_a_still_needs_no_ffmpeg(tmp_path: Path) -> None:
    only_paraview = tmp_path / "pv"
    only_paraview.mkdir()
    (only_paraview / "pvbatch").write_text("#!/bin/sh\n")
    (only_paraview / "pvbatch").chmod(0o755)
    plan = plan_for(tmp_path, {"PATH": str(only_paraview)}, kind=VisualKind.MESH)
    assert len(plan.steps) == 1


def test_an_unknown_field_is_refused_with_the_real_ones(tmp_path: Path, tools) -> None:
    with pytest.raises(ValidationError) as excinfo:
        plan_for(tmp_path, tools, kind=VisualKind.MESH, field="velocity")
    assert "U" in str(excinfo.value) and "p" in str(excinfo.value)


def test_an_animation_with_no_field_named_takes_pressure(tmp_path: Path, tools) -> None:
    """A plain grey mesh for every frame is a video of nothing."""
    plan = plan_for(tmp_path, tools, kind=VisualKind.ANIMATION)
    assert plan.outputs[0].name == "animation-p-isometric.mp4"


def test_a_still_with_no_field_is_the_plain_mesh(tmp_path: Path, tools) -> None:
    plan = plan_for(tmp_path, tools, kind=VisualKind.MESH, preset="top")
    assert plan.outputs[0].name == "mesh-top.png"


def test_a_coloured_still_names_its_field(tmp_path: Path, tools) -> None:
    plan = plan_for(tmp_path, tools, kind=VisualKind.MESH, field="U", preset="top")
    assert plan.outputs[0].name == "mesh-U-top.png"


def test_an_animation_that_cannot_fit_on_disk_is_refused(
    tmp_path: Path, tools, monkeypatch
) -> None:
    """Uncompressed frames are large; filling the disk can take running simulations' output
    down with it."""
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**9, 10**9, 1000))
    with pytest.raises(ValidationError, match="GB is free"):
        plan_for(tmp_path, tools, kind=VisualKind.ANIMATION, field="p")


def test_the_disk_cost_is_reported(tmp_path: Path, tools) -> None:
    plan = plan_for(tmp_path, tools, kind=VisualKind.ANIMATION, field="p", width=800, height=500)
    assert any("uncompressed frame" in note and "GB" in note for note in plan.notes)


def test_the_animation_timeout_is_long_enough_for_a_real_run(tmp_path: Path, tools) -> None:
    plan = plan_for(tmp_path, tools, kind=VisualKind.ANIMATION, field="p")
    assert plan.steps[0].timeout_s is not None and plan.steps[0].timeout_s >= 3600


# -- the render manager ------------------------------------------------------------------------


def step(script: str, tmp_path: Path, *, timeout: float | None = 30.0) -> CommandStep:
    return CommandStep(
        argv=["/bin/sh", "-c", script],
        cwd=tmp_path,
        description="fake render",
        kind=StepKind.SOLVE,
        timeout_s=timeout,
    )


def plan_of(*steps: CommandStep, output: Path) -> VisualPlan:
    return VisualPlan(steps=steps, outputs=(output,), tool="fake")


class Recorder:
    def __init__(self, bus: EventBus) -> None:
        self.events: list[tuple[str, dict]] = []
        bus.publish = lambda event, data: self.events.append((str(event), data))  # type: ignore[method-assign,assignment,misc]

    def of(self, event: Event) -> list[dict]:
        return [data for name, data in self.events if name == str(event)]


async def finished(recorder: Recorder, timeout: float = 20.0) -> dict:
    for _ in range(int(timeout / 0.05)):
        done = recorder.of(Event.RENDER_FINISHED)
        if done:
            return done[0]
        await asyncio.sleep(0.05)
    raise AssertionError("render never finished")


async def test_a_render_starts_without_waiting_for_it(tmp_path: Path) -> None:
    """The point: an hours-long render must not hold up the request that asked for it."""
    bus = EventBus()
    recorder = Recorder(bus)
    manager = RenderManager(bus)
    entry = manager.start(
        tmp_path, "animation", plan_of(step("sleep 1", tmp_path), output=tmp_path / "v.mp4")
    )
    assert entry.task is not None and not entry.task.done()
    assert manager.active
    result = await finished(recorder)
    assert result["ok"] is True
    assert not manager.active


async def test_progress_is_read_from_the_renderer_s_own_output(tmp_path: Path) -> None:
    bus = EventBus()
    recorder = Recorder(bus)
    manager = RenderManager(bus)
    script = 'for i in 1 2 3; do echo "frame $i/3  t = 0.$i"; sleep 1.1; done'
    manager.start(tmp_path, "animation", plan_of(step(script, tmp_path), output=tmp_path / "v"))
    await finished(recorder)
    frames = [data["frame"] for data in recorder.of(Event.RENDER_PROGRESS) if data["frame"]]
    assert frames and frames[-1] == 3
    assert all(data["frames"] == 3 for data in recorder.of(Event.RENDER_PROGRESS) if data["frame"])


async def test_a_failed_step_is_reported_with_its_last_words(tmp_path: Path) -> None:
    bus = EventBus()
    recorder = Recorder(bus)
    manager = RenderManager(bus)
    manager.start(
        tmp_path,
        "mesh",
        plan_of(step("echo 'no GL context'; exit 4", tmp_path), output=tmp_path / "m"),
    )
    result = await finished(recorder)
    assert result["ok"] is False
    assert "exit 4" in result["error"] and "no GL context" in result["error"]


async def test_a_failed_render_stops_before_the_next_step(tmp_path: Path) -> None:
    """So a failed encode never runs the frame-deleting step after it."""
    marker = tmp_path / "ran"
    bus = EventBus()
    recorder = Recorder(bus)
    manager = RenderManager(bus)
    manager.start(
        tmp_path,
        "animation",
        plan_of(step("exit 1", tmp_path), step(f"touch {marker}", tmp_path), output=tmp_path / "v"),
    )
    await finished(recorder)
    assert not marker.exists()


async def test_a_cancelled_render_kills_its_process(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid"
    bus = EventBus()
    recorder = Recorder(bus)
    manager = RenderManager(bus)
    entry = manager.start(
        tmp_path,
        "animation",
        plan_of(step(f"echo $$ > {pidfile}; exec sleep 30", tmp_path), output=tmp_path / "v"),
    )
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        await asyncio.sleep(0.05)
    assert manager.cancel(entry.id[:8]), "cancellable by an id prefix"
    result = await finished(recorder)
    assert result["error"] == "cancelled"

    import os

    pid = int(pidfile.read_text())
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_a_render_that_runs_too_long_is_stopped(tmp_path: Path) -> None:
    """A renderer with no display does not fail; it waits."""
    bus = EventBus()
    recorder = Recorder(bus)
    manager = RenderManager(bus)
    manager.start(
        tmp_path, "mesh", plan_of(step("sleep 30", tmp_path, timeout=0.5), output=tmp_path / "m")
    )
    result = await finished(recorder)
    assert result["ok"] is False and "limit" in result["error"]


async def test_two_renders_of_the_same_output_are_refused(tmp_path: Path) -> None:
    """They would share one frames directory and splice a video from both."""
    manager = RenderManager(EventBus())
    plan = plan_of(step("sleep 2", tmp_path), output=tmp_path / "same.mp4")
    manager.start(tmp_path, "animation", plan)
    with pytest.raises(ValidationError, match="already running"):
        manager.start(tmp_path, "animation", plan)
    await manager.shutdown()


async def test_shutdown_stops_every_render(tmp_path: Path) -> None:
    manager = RenderManager(EventBus())
    manager.start(tmp_path, "a", plan_of(step("sleep 30", tmp_path), output=tmp_path / "a"))
    manager.start(tmp_path, "b", plan_of(step("sleep 30", tmp_path), output=tmp_path / "b"))
    await asyncio.wait_for(manager.shutdown(), timeout=10)
    assert not manager.active


async def test_inline_rendering_returns_the_output(tmp_path: Path) -> None:
    """The CLI's path: it has nothing else to do while it waits."""
    produced = await RenderManager(EventBus()).run_inline(
        tmp_path, "mesh", plan_of(step("echo wrote it", tmp_path), output=tmp_path / "m")
    )
    assert produced == ["wrote it"]


async def test_a_render_cancelled_before_it_begins_is_still_reported(tmp_path: Path) -> None:
    """A task cancelled before its first step never runs its own cleanup."""
    bus = EventBus()
    recorder = Recorder(bus)
    manager = RenderManager(bus)
    entry = manager.start(
        tmp_path, "mesh", plan_of(step("sleep 30", tmp_path), output=tmp_path / "m")
    )
    manager.cancel(entry.id)
    result = await finished(recorder)
    assert result["error"] == "cancelled"
    assert not manager.active
    assert len(recorder.of(Event.RENDER_FINISHED)) == 1
