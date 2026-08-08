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

import pytest

from dispatch.core.config import Config, DaemonConfig, PathsConfig
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
    ["DashboardScreen", "QueueScreen", "HistoryScreen", "HelpScreen", "LogScreen"],
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
    from dispatch.tui.screens.queue import QueueScreen

    lookup = {
        "DashboardScreen": lambda: DashboardScreen(),
        "QueueScreen": lambda: QueueScreen(),
        "HistoryScreen": lambda: HistoryScreen(),
        "HelpScreen": lambda: HelpScreen(),
        "LogScreen": lambda: LogScreen("some-job-id"),
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
