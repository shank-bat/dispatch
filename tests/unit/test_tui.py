"""The interface: state handling, log tailing, and formatting.

Most of this is testable without running Textual, because the parts that could be wrong --
what order jobs appear in, how a growing file is followed, how a duration is rendered --
were deliberately kept out of the widgets.

The last section does drive the real application, headless, to check the one behaviour that
only shows up when everything is wired together: the interface starts and stays usable when
no daemon is running.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from dispatch.core.config import Config, DaemonConfig, PathsConfig
from dispatch.ipc.protocol import Method
from dispatch.tui.app import DispatchApp
from dispatch.tui.screens.submit import PlanScreen, SubmitScreen
from dispatch.tui.state import AppState
from dispatch.tui.tailer import AsyncTailer, read_last_lines
from dispatch.tui.widgets.jobtable import format_duration, state_text, step_label


def job(
    job_id: str,
    *,
    name: str = "case",
    state: str = "QUEUED",
    position: int | None = None,
    priority: int = 0,
    seq: int = 1,
    finished_at: float | None = None,
    cores: int = 4,
) -> dict:
    return {
        "id": job_id,
        "name": name,
        "state": state,
        "queue_position": position,
        "priority": priority,
        "seq": seq,
        "finished_at": finished_at,
        "cores": cores,
        "solver": "fake",
        "solver_binary": None,
        "tags": [],
        "metrics": {"runtime_s": None, "peak_rss_mb": None, "mean_cpu_pct": None},
    }


# -- client-side state ------------------------------------------------------------------


def test_jobs_are_grouped_by_state() -> None:
    state = AppState()
    state.replace_jobs(
        [
            job("a", state="RUNNING"),
            job("b", state="QUEUED", position=1),
            job("c", state="COMPLETED", finished_at=100.0),
        ]
    )
    assert [j["id"] for j in state.running] == ["a"]
    assert [j["id"] for j in state.queued] == ["b"]
    assert [j["id"] for j in state.finished] == ["c"]


def test_preparing_counts_as_running() -> None:
    """It holds an allocation, so the dashboard must show it as using the machine."""
    state = AppState()
    state.replace_jobs([job("a", state="PREPARING")])
    assert [j["id"] for j in state.running] == ["a"]


def test_the_queue_is_ordered_by_position() -> None:
    state = AppState()
    state.replace_jobs(
        [
            job("third", state="QUEUED", position=3),
            job("first", state="QUEUED", position=1),
            job("second", state="QUEUED", position=2),
        ]
    )
    assert [j["id"] for j in state.queued] == ["first", "second", "third"]


def test_held_jobs_sort_after_queued_ones() -> None:
    """A held job has no queue position, so it must not appear to be next."""
    state = AppState()
    state.replace_jobs(
        [job("held", state="HELD"), job("next", state="QUEUED", position=1)]
    )
    assert [j["id"] for j in state.queued] == ["next", "held"]


def test_next_job_ignores_held_jobs() -> None:
    state = AppState()
    state.replace_jobs([job("held", state="HELD")])
    assert state.next_job is None


def test_finished_jobs_are_newest_first() -> None:
    state = AppState()
    state.replace_jobs(
        [
            job("old", state="COMPLETED", finished_at=100.0),
            job("new", state="FAILED", finished_at=500.0),
        ]
    )
    assert [j["id"] for j in state.finished] == ["new", "old"]


def test_an_update_replaces_a_job_in_place() -> None:
    state = AppState()
    state.replace_jobs([job("a", state="QUEUED")])
    state.update_job(job("a", state="RUNNING"))
    assert state.jobs["a"]["state"] == "RUNNING"
    assert len(state.jobs) == 1


def test_forgetting_a_job_drops_its_progress_too() -> None:
    state = AppState()
    state.replace_jobs([job("a", state="RUNNING")])
    state.update_progress({"id": "a", "rss_mb": 100})
    state.forget("a")
    assert state.get("a") is None
    assert state.progress_for("a") is None


def test_progress_updates_merge_instead_of_replacing() -> None:
    """Two samplers publish here at different rates, each carrying only its own fields.

    A plain assignment would let the frequent time-step update erase the memory reading and
    the slow resource sample erase the time step, so the display would alternate between
    halves of the truth.
    """
    state = AppState()
    state.replace_jobs([job("a", state="RUNNING")])
    state.update_progress({"id": "a", "rss_mb": 100, "cpu_pct": 800.0})
    state.update_progress({"id": "a", "progress": {"current": 0.35, "label": "Time"}})

    latest = state.progress_for("a")
    assert latest is not None
    assert latest["rss_mb"] == 100
    assert latest["progress"]["current"] == 0.35


def test_the_time_step_is_rendered_with_its_units() -> None:
    assert step_label({"current": 0.35, "label": "Time"}) == "t 0.35"
    assert step_label({"current": 1250, "label": "Iteration"}) == "iter 1250"


def test_a_known_end_time_is_shown_alongside_the_current_one() -> None:
    assert step_label({"current": 0.35, "total": 2.0, "label": "Time"}) == "t 0.35/2"


def test_small_and_large_time_steps_stay_readable() -> None:
    """A CFD time step is as likely to be 1e-05 as 12.5, and both have to fit the column."""
    assert step_label({"current": 1e-05, "label": "Time"}) == "t 1e-05"
    assert step_label({"current": 125000.0, "label": "Time"}) == "t 125000"


def test_an_unparseable_reading_renders_as_nothing() -> None:
    assert step_label({"current": None, "label": "Time"}) == ""


def test_counts_by_state() -> None:
    state = AppState()
    state.replace_jobs(
        [job("a", state="RUNNING"), job("b", state="QUEUED"), job("c", state="QUEUED")]
    )
    assert state.counts == {"RUNNING": 1, "QUEUED": 2}


# -- log tailing --------------------------------------------------------------------------


def test_read_last_lines_returns_the_tail(tmp_path: Path) -> None:
    path = tmp_path / "log"
    path.write_text("\n".join(f"line {i}" for i in range(1000)) + "\n")
    assert read_last_lines(path, 3) == ["line 997", "line 998", "line 999"]


def test_read_last_lines_handles_a_short_file(tmp_path: Path) -> None:
    path = tmp_path / "log"
    path.write_text("only one\n")
    assert read_last_lines(path, 50) == ["only one"]


def test_read_last_lines_on_a_missing_file(tmp_path: Path) -> None:
    assert read_last_lines(tmp_path / "absent", 10) == []


def test_read_last_lines_does_not_read_the_whole_file(tmp_path: Path) -> None:
    """A three-day residual log can be hundreds of megabytes."""
    path = tmp_path / "big"
    path.write_text("x" * 200_000 + "\ntail line\n")
    assert read_last_lines(path, 1, block=1024) == ["tail line"]


async def test_the_tailer_yields_appended_text(tmp_path: Path) -> None:
    path = tmp_path / "growing"
    path.write_text("existing\n")

    tailer = AsyncTailer(path, from_start=True)
    chunks: list[str] = []

    async def collect() -> None:
        async for chunk in tailer.follow():
            chunks.append(chunk)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.3)
    with path.open("a") as handle:
        handle.write("appended\n")
    await asyncio.sleep(0.5)

    tailer.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "appended" in "".join(chunks)


async def test_the_tailer_recovers_from_truncation(tmp_path: Path) -> None:
    """A restarted job reopens its log; the viewer must not sit at a stale offset."""
    path = tmp_path / "rotating"
    path.write_text("first run output\n")

    tailer = AsyncTailer(path, from_start=True)
    chunks: list[str] = []

    async def collect() -> None:
        async for chunk in tailer.follow():
            chunks.append(chunk)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.3)
    path.write_text("second\n")  # truncate and rewrite
    await asyncio.sleep(0.6)

    tailer.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "second" in "".join(chunks)


async def test_the_tailer_waits_for_a_file_that_does_not_exist_yet(tmp_path: Path) -> None:
    """Normal right after submission, and not an error."""
    tailer = AsyncTailer(tmp_path / "later", from_start=True)
    chunks: list[str] = []

    async def collect() -> None:
        async for chunk in tailer.follow():
            chunks.append(chunk)

    task: asyncio.Task[None] = asyncio.create_task(collect())
    await asyncio.sleep(0.2)
    assert not task.done(), "the tailer gave up on a file that does not exist yet"
    assert chunks == []

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# -- formatting -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, ""),
        (0, ""),
        (45, "45s"),
        (90, "1m 30s"),
        (3600, "1h 0m"),
        (7 * 3600 + 900, "7h 15m"),
        (3 * 86400 + 4 * 3600, "3d 4h"),
    ],
)
def test_durations_render_compactly(seconds: float | None, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_state_text_is_styled_per_state() -> None:
    assert state_text("RUNNING").plain == "running"
    assert state_text("FAILED").style != state_text("COMPLETED").style


def test_an_unknown_state_still_renders() -> None:
    """History written by a future version must not crash the table."""
    assert state_text("TELEPORTED").plain == "teleported"


# -- the application ---------------------------------------------------------------------------


@pytest.fixture
def offline_config(tmp_path: Path) -> Config:
    """A config pointing at a socket that does not exist, with autostart off."""
    return Config(
        paths=PathsConfig(
            database=tmp_path / "dispatch.db",
            log_dir=tmp_path / "logs",
            runtime_dir=tmp_path / "run",
        ),
        daemon=DaemonConfig(autostart=False),
    )


async def test_the_interface_starts_with_no_daemon_running(offline_config: Config) -> None:
    """It must report the problem, not crash.

    This is the realistic first-run failure -- and the realistic upgrade failure, when the
    daemon is restarting while a TUI is open.
    """
    from dispatch.tui.app import DispatchApp

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.state.connected
        assert app.state.error is not None
        assert app.is_running


async def test_a_running_job_shows_its_time_step_next_to_the_bar() -> None:
    """Rendered through the real widget, because the failure mode is structural.

    The step column adds a cell as well as a header; a table whose cell count stops
    matching its column count raises at the moment a job appears, which no amount of
    testing the formatting function in isolation would catch.
    """
    from textual.app import App, ComposeResult

    from dispatch.tui.widgets.jobtable import JobTable

    running = job("a1", name="foamlonger", state="RUNNING")
    progress = {
        "a1": {
            "rss_mb": 2048,
            "progress": {"current": 28.35, "total": 50.0, "label": "Time", "fraction": 0.567},
        }
    }

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield JobTable(id="active")
            yield JobTable(id="recent", show_progress=False)

    app = Harness()
    async with app.run_test() as pilot:
        app.query_one("#active", JobTable).show([running], progress)
        app.query_one("#recent", JobTable).show([job("b2", state="FAILED")])
        await pilot.pause()

        active = app.query_one("#active", JobTable)
        cells = [str(c) for c in active.get_row_at(0)]
        assert "t 28.35/50" in cells  # the solver's own number, verbatim
        assert any("57%" in c for c in cells)  # and the estimate, still there

        # The history table has no progress, so it must gain neither column nor cell.
        recent = app.query_one("#recent", JobTable)
        assert "step" not in [str(c.label) for c in recent.columns.values()]
        assert len(list(recent.get_row_at(0))) == len(recent.columns)


async def test_the_screens_can_all_be_reached(offline_config: Config) -> None:
    """Composition errors in a screen would otherwise only surface when a user pressed a key."""
    from dispatch.tui.app import DispatchApp

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        for key in ("2", "3", "question_mark", "escape", "1"):
            await pilot.press(key)
            await pilot.pause()
        assert app.is_running


@pytest.mark.parametrize(
    "screen_name",
    ["DashboardScreen", "QueueScreen", "HistoryScreen", "HelpScreen", "LogScreen", "PlotScreen"],
)
async def test_every_screen_mounts_its_children(offline_config: Config, screen_name: str) -> None:
    """A screen that composes but never mounts renders as a blank page.

    This is not hypothetical: `LogScreen` once assigned `self._task`, which shadows
    Textual's own `MessagePump._task`, and the screen silently showed nothing at all. The
    check is cheap and catches the whole class of attribute collisions.
    """
    from dispatch.tui import screens
    from dispatch.tui.app import DispatchApp
    from dispatch.tui.screens.dashboard import DashboardScreen
    from dispatch.tui.screens.help import HelpScreen
    from dispatch.tui.screens.history import HistoryScreen
    from dispatch.tui.screens.logs import LogScreen
    from dispatch.tui.screens.plot import PlotScreen
    from dispatch.tui.screens.queue import QueueScreen

    lookup = {
        "DashboardScreen": lambda: DashboardScreen(),
        "QueueScreen": lambda: QueueScreen(),
        "HistoryScreen": lambda: HistoryScreen(),
        "HelpScreen": lambda: HelpScreen(),
        "LogScreen": lambda: LogScreen("some-job-id"),
        "PlotScreen": lambda: PlotScreen("some-job-id"),
    }
    del screens

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(lookup[screen_name]())
        await pilot.pause()

        screen = app.screen
        assert type(screen).__name__ == screen_name
        assert screen.children, f"{screen_name} mounted no children"
        assert screen.query("#topbar"), f"{screen_name} has no top bar"


async def test_the_run_after_picker_defaults_to_none(offline_config: Config) -> None:
    """The default must be "none", and reachable without choosing anything."""
    from dispatch.tui.app import DispatchApp
    from dispatch.tui.screens.submit import RunAfterScreen

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        chosen: list[str | None] = []
        await app.push_screen(
            RunAfterScreen([job("a1", name="parent", state="RUNNING")], current=None),
            chosen.append,
        )
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert chosen == [""]  # "" clears the dependency; the first entry is "none"


async def test_the_run_after_picker_returns_the_chosen_job(offline_config: Config) -> None:
    from dispatch.tui.app import DispatchApp
    from dispatch.tui.screens.submit import RunAfterScreen

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        chosen: list[str | None] = []
        await app.push_screen(
            RunAfterScreen([job("a1", name="parent", state="RUNNING")], current=None),
            chosen.append,
        )
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

    assert chosen == ["a1"]


async def test_backing_out_of_the_run_after_picker_changes_nothing(
    offline_config: Config,
) -> None:
    from dispatch.tui.app import DispatchApp
    from dispatch.tui.screens.submit import RunAfterScreen

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        chosen: list[str | None] = []
        await app.push_screen(
            RunAfterScreen([job("a1", state="RUNNING")], current=None), chosen.append
        )
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert chosen == [None]  # None means "leave the current choice alone"


async def test_a_new_submission_waits_for_nothing_by_default(offline_config: Config) -> None:
    from dispatch.tui.app import DispatchApp
    from dispatch.tui.screens.submit import SubmitScreen

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = SubmitScreen(Path.home())
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.run_after is None


async def test_no_widget_shadows_a_textual_internal() -> None:
    """Screens must not assign attributes that Textual's own machinery owns.

    `_task`, `_nodes`, and friends belong to MessagePump and Widget. Shadowing one breaks
    the widget in ways that produce no error -- just an empty screen.
    """
    import ast

    from textual.message_pump import MessagePump
    from textual.widget import Widget

    reserved = {
        name
        for cls in (MessagePump, Widget)
        for name in dir(cls)
        if name.startswith("_") and not name.startswith("__")
    }

    offenders: list[str] = []
    root = Path(__file__).resolve().parents[2] / "src" / "dispatch" / "tui"
    for module in sorted(root.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
                continue
            for stmt in ast.walk(node):
                targets = []
                if isinstance(stmt, ast.Assign):
                    targets = stmt.targets
                elif isinstance(stmt, ast.AnnAssign):
                    targets = [stmt.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr in reserved
                    ):
                        offenders.append(f"{module.name} sets self.{target.attr}")

    assert not offenders, "Attributes shadowing Textual internals:\n  " + "\n  ".join(offenders)


# -- the new surface ---------------------------------------------------------------------


def test_the_job_table_shows_what_kind_of_work_a_job_is() -> None:
    """"Is the GPU busy" and "are the cores busy" are different questions."""
    from dispatch.tui.widgets.jobtable import resource_text

    assert resource_text({"resource_kind": "cpu", "cores": 20, "gpus": 0}).plain == "CPU 20c"
    assert resource_text({"resource_kind": "gpu", "cores": 1, "gpus": 1}).plain == "GPU 1G"
    assert resource_text({"resource_kind": "gpu", "cores": 4, "gpus": 2}).plain == "GPU 2G·4c"


def test_a_job_from_before_the_resource_column_still_renders() -> None:
    """History written by an older Dispatch has neither field."""
    from dispatch.tui.widgets.jobtable import resource_text

    assert resource_text({"cores": 8}).plain == "CPU 8c"


def rendered(widget) -> str:
    """A widget's rendered text, as plain characters."""
    from rich.text import Text

    output = widget.render()
    assert isinstance(output, Text)
    return output.plain


def test_the_meters_show_gpus_only_on_a_machine_that_has_them() -> None:
    """A permanent "gpu 0/0" on most workstations is a column reporting an absence."""
    from dispatch.tui.widgets.meters import ResourceMeters

    meters = ResourceMeters()
    meters.snapshot = {
        "total_cores": 8, "allocated_cores": 0, "free_cores": 8, "cpu_percent": 5.0,
        "used_ram_mb": 4000, "total_ram_mb": 32000, "load_average": [0.1],
        "total_gpus": 0, "allocated_gpus": 0, "free_gpus": 0,
    }
    assert "gpu" not in rendered(meters)

    meters.snapshot = {**meters.snapshot, "total_gpus": 2, "allocated_gpus": 1, "free_gpus": 1}
    assert "gpu" in rendered(meters)


async def test_the_plot_screen_opens_offline_without_crashing(offline_config: Config) -> None:
    """`p` must not be able to take the interface down when the daemon is unreachable."""
    from dispatch.tui.app import DispatchApp
    from dispatch.tui.screens.plot import PlotScreen

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(PlotScreen("nobody"))
        await pilot.pause()
        assert app.is_running
        assert isinstance(app.screen, PlotScreen)
        assert app.screen.error is not None


async def test_the_project_search_modal_mounts(offline_config: Config) -> None:
    from dispatch.tui.app import DispatchApp
    from dispatch.tui.screens.projects import ProjectSearchScreen

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(ProjectSearchScreen())
        await pilot.pause()
        assert app.is_running
        assert app.screen.query("#project-query"), "the search box did not mount"


def test_a_plot_selection_defaults_to_something_worth_looking_at() -> None:
    """Opening the screen should show a chart, not an empty pair of lists."""
    from dispatch.core.series import PlotData, Series
    from dispatch.tui.screens.plot import PlotScreen

    screen = PlotScreen("job")
    screen.data = PlotData(
        series=(
            Series("iteration", "Iteration", (1.0, 2.0), (0, 1), axis=True),
            Series("residual(Ux)", "residual(Ux)", (0.1, 0.01), (0, 1)),
        ),
        samples=2,
    )
    screen._choose_defaults()

    assert screen._x_key == "iteration"
    assert screen._y_keys == ["residual(Ux)"]
    assert screen._style.log_y is False, "two points spanning one decade do not need a log axis"


def test_a_plot_of_residuals_defaults_to_a_log_axis() -> None:
    """Four decades on a linear axis collapse onto the bottom row."""
    from dispatch.core.series import PlotData, Series
    from dispatch.tui.screens.plot import PlotScreen

    values = tuple(10.0**-exponent for exponent in range(1, 6))
    screen = PlotScreen("job")
    screen.data = PlotData(
        series=(
            Series("iteration", "Iteration", tuple(float(i) for i in range(5)), tuple(range(5)),
                   axis=True),
            Series("residual(p)", "residual(p)", values, tuple(range(5))),
        ),
        samples=5,
    )
    screen._choose_defaults()
    assert screen._style.log_y is True


def test_decoding_skips_a_series_it_cannot_understand() -> None:
    """The interface renders data written by other versions; one bad entry is not fatal."""
    from dispatch.tui.screens.plot import _decode

    data = _decode(
        {
            "series": [
                {"key": "good", "label": "good", "values": [1.0], "samples": [0]},
                {"key": "bad", "label": "bad", "values": ["not-a-number"], "samples": [0]},
                {"key": "short", "label": "short"},
            ],
            "samples": 1,
        }
    )
    assert [item.key for item in data.series] == ["good"]


# -- submitting a folder of cases as a sweep -----------------------------------------------
#
# The submit screen is the only place a sweep can be configured, so these drive the real
# screen rather than the helpers underneath it. What they check is that the screen routes
# to the existing sweep endpoint with the settings the user chose -- not that it schedules
# anything, which is the daemon's business and is tested there.


def recording_app(config: Config, replies: dict[str, Any]) -> DispatchApp:
    """A real DispatchApp whose daemon calls are recorded and answered from a script.

    The recorder is attached to the instance rather than added by a subclass: Textual
    resolves ``CSS_PATH`` relative to the module a class is *defined* in, so an App
    subclass declared in a test file goes looking for a stylesheet beside the test.
    """
    app = DispatchApp(config=config, autostart=False)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _call(method: str, **params: Any) -> Any:
        calls.append((method, params))
        reply = replies.get(method)
        if isinstance(reply, Exception):
            raise reply
        return reply

    app.call = _call  # type: ignore[method-assign]
    app.calls = calls  # type: ignore[attr-defined]
    app.calls_to = lambda method: [  # type: ignore[attr-defined]
        params for name, params in calls if name == method
    ]
    return app


CAVITY_SWEEP = {
    "solver": "openfoam",
    "count": 5,
    "cases": ["Re100", "Re200", "Re400", "Re800", "Re1600"],
}


def sweep_replies(**extra: Any) -> dict[str, Any]:
    """Daemon answers for a directory the daemon has recognised as a sweep."""
    replies: dict[str, Any] = {
        Method.FS_LIST: {
            "path": "/cases/cavity",
            "parent": "/cases",
            "entries": [],
            "case": None,
            "sweep": CAVITY_SWEEP,
        },
        Method.SWEEP_SUBMIT: {
            "sweep": {
                "id": "s1",
                "name": "cavity",
                "total": 5,
                "cores_per_job": 3,
                "concurrency": 2,
            },
            "jobs": [],
        },
    }
    replies.update(extra)
    return replies


def dry_run_report(**extra: Any) -> dict[str, Any]:
    """A dry-run report shaped like the daemon's, for driving the plan screen."""
    report: dict[str, Any] = {
        "workdir": "/cases/cavity/Re100",
        "cores": 3,
        "resources": "3 cores",
        "detections": [],
        "validation": {"summary": "ok", "passed": True, "findings": []},
        "plan": None,
        "suggested_tags": [],
        "projection": None,
        "log_path": None,
    }
    report.update(extra)
    return report


async def open_submit(app: DispatchApp, pilot: Any, path: Path) -> SubmitScreen:
    """Push the submit screen and let its initial listing settle."""
    screen = SubmitScreen(start=path)
    await app.push_screen(screen)
    for _ in range(20):
        await pilot.pause()
        if screen.sweep is not None or screen._inspection is not None:
            break
    return screen


async def test_a_folder_of_cases_is_recognised_as_a_sweep_in_the_wizard(
    offline_config: Config, tmp_path: Path
) -> None:
    """Step 2 of the flow: the browser learns it from the listing it already fetches."""
    app = recording_app(offline_config, sweep_replies())
    async with app.run_test() as pilot:
        screen = await open_submit(app, pilot, tmp_path)
        assert screen.sweep == CAVITY_SWEEP


async def test_the_sweep_pane_shows_the_cases_in_execution_order(
    offline_config: Config, tmp_path: Path
) -> None:
    """Step 3: the order is a property of the submission, so it is shown before committing."""
    app = recording_app(offline_config, sweep_replies())
    async with app.run_test() as pilot:
        screen = await open_submit(app, pilot, tmp_path)
        pane = str(screen._sweep_detail())

    for position, name in enumerate(CAVITY_SWEEP["cases"], start=1):
        assert f"{position:>3} {name}" in pane


def test_the_sweep_pane_separates_every_label_from_its_value() -> None:
    """``concurrent`` is exactly as wide as the default label column, and ran into it."""
    from dispatch.tui.screens.submit import SWEEP_LABEL_WIDTH

    assert len("concurrent") < SWEEP_LABEL_WIDTH


async def test_reviewing_a_sweep_shows_every_case_and_the_cap(
    offline_config: Config, tmp_path: Path
) -> None:
    """Step 5: the review lists the whole sweep, not the pane's preview of it."""
    plan_report = dry_run_report()
    app = recording_app(offline_config, sweep_replies(**{Method.CASE_DRYRUN: plan_report}))
    async with app.run_test() as pilot:
        screen = await open_submit(app, pilot, tmp_path)
        screen.cores, screen.concurrency = 3, 2
        await screen.action_dry_run()
        await pilot.pause()

        assert isinstance(app.screen, PlanScreen)
        body = str(app.screen._body())

    assert "a hard cap" in body
    assert "at most 6 cores in use" in body
    for name in CAVITY_SWEEP["cases"]:
        assert name in body


async def test_reviewing_a_sweep_asks_for_a_real_case_plan(
    offline_config: Config, tmp_path: Path
) -> None:
    """The plan comes from the ordinary dry-run endpoint, so it cannot drift from reality."""
    plan_report = dry_run_report()
    app = recording_app(offline_config, sweep_replies(**{Method.CASE_DRYRUN: plan_report}))
    async with app.run_test() as pilot:
        screen = await open_submit(app, pilot, tmp_path)
        screen.cores = 3
        await screen.action_dry_run()
        await pilot.pause()

    dry_runs = app.calls_to(Method.CASE_DRYRUN)
    assert len(dry_runs) == 1
    assert dry_runs[0]["workdir"].endswith("Re100")  # the first case, standing in for all
    assert dry_runs[0]["cores"] == 3


async def test_submitting_a_sweep_uses_the_sweep_endpoint(
    offline_config: Config, tmp_path: Path
) -> None:
    """Step 6: one call to the existing sweep path, carrying the settings the user chose."""
    app = recording_app(offline_config, sweep_replies())
    async with app.run_test() as pilot:
        screen = await open_submit(app, pilot, tmp_path)
        screen.cores, screen.concurrency, screen.gpus = 3, 2, 1
        screen.tags = ["study"]
        await screen.action_submit()
        await pilot.pause()

    submissions = app.calls_to(Method.SWEEP_SUBMIT)
    assert len(submissions) == 1
    assert submissions[0]["cores_per_job"] == 3
    assert submissions[0]["concurrency"] == 2
    assert submissions[0]["gpus"] == 1, "a GPU count set in the wizard must not be dropped"
    assert submissions[0]["tags"] == ["study"]
    assert app.calls_to(Method.JOB_SUBMIT) == []


async def test_an_ordinary_case_still_submits_as_a_single_job(
    offline_config: Config, tmp_path: Path
) -> None:
    """The path that must not change: a case directory is unaffected by any of this."""
    replies = {
        Method.FS_LIST: {
            "path": "/cases/cavity",
            "parent": "/cases",
            "entries": [],
            "case": "openfoam",
            "sweep": None,
        },
        Method.CASE_VALIDATE: {
            "solver": "openfoam",
            "solver_binary": "icoFoam",
            "validation": {"summary": "ok", "passed": True, "findings": []},
            "case": {},
        },
        Method.JOB_SUBMIT: {"job": {"id": "j1", "name": "cavity", "cores": 4}},
    }
    app = recording_app(offline_config, replies)
    async with app.run_test() as pilot:
        screen = await open_submit(app, pilot, tmp_path)
        assert screen.sweep is None
        await screen.action_submit()
        await pilot.pause()

    assert len(app.calls_to(Method.JOB_SUBMIT)) == 1
    assert app.calls_to(Method.SWEEP_SUBMIT) == []


async def test_the_concurrency_key_does_not_shadow_row_navigation(
    offline_config: Config, tmp_path: Path
) -> None:
    """``j`` is documented globally as "next row", so it cannot also mean "set the cap".

    A vim-ish interface trains the reflex everywhere else; a screen where ``j`` opens a
    prompt instead of moving the cursor is a trap rather than a shortcut.
    """
    from dispatch.tui.screens.submit import SubmitScreen as Screen

    keys = {binding.key for binding in Screen.BINDINGS}
    assert "j" not in keys
    assert "m" in keys


async def test_setting_concurrency_outside_a_sweep_is_refused(
    offline_config: Config, tmp_path: Path
) -> None:
    """The key exists on every directory; it only means something on a sweep folder."""
    replies = {
        Method.FS_LIST: {
            "path": "/x",
            "parent": None,
            "entries": [],
            "case": None,
            "sweep": None,
        }
    }
    app = recording_app(offline_config, replies)
    async with app.run_test() as pilot:
        screen = await open_submit(app, pilot, tmp_path)
        screen.action_edit_concurrency()
        await pilot.pause()
        assert screen.concurrency == 1


# -- sweep membership in the queue and the dashboard -------------------------------------------


def test_a_sweep_members_row_is_marked_with_its_position() -> None:
    """Nothing else in the row says these jobs were submitted together."""
    from dispatch.tui.widgets.jobtable import _name_text

    member = job("a", name="Re400", state="RUNNING")
    member["sweep_id"], member["sweep_position"] = "s1", 2

    assert str(_name_text(member)) == "Re400 ·3"


def test_an_ordinary_jobs_row_is_unchanged() -> None:
    from dispatch.tui.widgets.jobtable import _name_text

    assert str(_name_text(job("a", name="cavity"))) == "cavity"


def test_the_sweep_summary_states_progress_and_the_cap() -> None:
    """Progress alone reads as a stall; the cap alone does not say how far along it is."""
    from dispatch.tui.state import sweep_summary

    state = AppState()
    state.replace_jobs(
        [
            {**job("a", state="RUNNING"), "sweep_id": "s1"},
            {**job("b", state="RUNNING"), "sweep_id": "s1"},
            {**job("c", state="QUEUED"), "sweep_id": "s1"},
        ]
    )
    line = sweep_summary(
        state, {"id": "s1", "name": "cavity", "total": 5, "finished": 2, "concurrency": 2}
    )

    assert line == "sweep cavity  2 running, 2/5 done  (max 2)"


async def test_the_dashboard_shows_an_active_sweep(offline_config: Config) -> None:
    """Step 7, on the screen that answers "what is the machine doing right now"."""
    from dispatch.tui.screens.dashboard import DashboardScreen

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        app.state.replace_jobs([{**job("a", state="RUNNING"), "sweep_id": "s1"}])
        app.state.replace_sweeps(
            [{"id": "s1", "name": "cavity", "total": 5, "finished": 0, "concurrency": 2}]
        )
        screen = app.screen
        assert isinstance(screen, DashboardScreen)
        screen.refresh_view()
        await pilot.pause()
        assert "sweep cavity" in str(screen._sweeps())


async def test_the_dashboard_says_nothing_when_no_sweep_is_active(
    offline_config: Config,
) -> None:
    """A machine running ordinary jobs must not grow an empty sweep section."""
    from dispatch.tui.screens.dashboard import DashboardScreen

    app = DispatchApp(config=offline_config, autostart=False)
    async with app.run_test() as pilot:
        app.state.replace_jobs([job("a", state="RUNNING")])
        screen = app.screen
        assert isinstance(screen, DashboardScreen)
        screen.refresh_view()
        await pilot.pause()
        assert str(screen._sweeps()) == ""
