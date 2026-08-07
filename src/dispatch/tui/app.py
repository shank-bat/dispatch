"""The Textual application.

Owns the daemon connection and the cached state; the screens render it. After the initial
paint the interface is entirely **push-driven** -- it never polls the daemon. The only
local timers are the ones that make the wall clock and elapsed-time columns advance, which
touch no I/O at all.

Disconnection is treated as normal, because it is: upgrading Dispatch means restarting the
daemon while a TUI is open. The client reconnects with backoff, a banner appears, and a
full resync happens when it comes back.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App
from textual.binding import Binding

from dispatch.core.config import Config, load_config
from dispatch.core.errors import DispatchError
from dispatch.ipc.client import DaemonClient
from dispatch.ipc.protocol import Event, Method, Notification, Topic
from dispatch.tui.screens.dashboard import DashboardScreen
from dispatch.tui.screens.help import HelpScreen
from dispatch.tui.screens.history import HistoryScreen
from dispatch.tui.screens.logs import LogScreen
from dispatch.tui.screens.queue import QueueScreen
from dispatch.tui.screens.submit import SubmitScreen
from dispatch.tui.state import AppState
from dispatch.tui.theme import DISPATCH_THEME, Palette, state_style

__all__ = ["DispatchApp"]


class DispatchApp(App[None]):
    """Dispatch's interactive interface."""

    TITLE = "Dispatch"
    CSS_PATH = "dispatch.tcss"
    """The whole stylesheet, in one file next to this module.

    Styling lives there rather than in DEFAULT_CSS blocks so the design can be read, and
    changed, without reading Python.
    """

    BINDINGS = [
        Binding("n", "new_job", "new job"),
        Binding("question_mark", "help", "help", priority=True),
        Binding("q", "quit", "quit", priority=True),
        # Screen switches are shown in the top bar, so they are bound but not repeated in
        # the footer -- otherwise they crowd out the screen-specific keys, which are the
        # ones worth reminding somebody of.
        Binding("1", "dashboard", "dashboard", show=False),
        Binding("2", "queue", "queue", show=False),
        Binding("3", "history", "history", show=False),
        Binding("r", "resync", "refresh", show=False),
    ]

    def __init__(self, *, config: Config | None = None, autostart: bool = True) -> None:
        super().__init__()
        self.config = config or load_config()
        self.state = AppState()
        self.client = DaemonClient(
            self.config.paths.socket,
            autostart=autostart and self.config.daemon.autostart,
            config_path=self.config.source,
            on_event=self._on_event,
            on_state_change=self._on_connection_change,
        )
        self._watchdog: asyncio.Task[None] | None = None

    # -- lifecycle ------------------------------------------------------------------------

    async def on_mount(self) -> None:
        self.register_theme(DISPATCH_THEME)
        self.theme = DISPATCH_THEME.name
        await self.push_screen(DashboardScreen())
        await self._connect()
        self._watchdog = asyncio.create_task(self._reconnect_forever(), name="dispatch-reconnect")

    async def on_unmount(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog
        await self.client.close()

    async def _connect(self) -> None:
        try:
            await self.client.connect()
        except DispatchError as exc:
            self.state.connected = False
            self.state.error = str(exc)
            self._refresh_screen()
            self.notify(str(exc), severity="error", timeout=10)
            return
        await self._after_connect()

    async def _after_connect(self) -> None:
        """Subscribe, then take one full snapshot. Everything after this is pushed."""
        self.state.connected = True
        self.state.hostname = str(self.client.server_info.get("hostname", ""))
        self.state.daemon_version = str(self.client.server_info.get("version", ""))
        await self.client.subscribe([str(t) for t in Topic])
        await self.resync()

    async def _reconnect_forever(self) -> None:
        """Reconnect after the daemon restarts, without the interface noticing."""
        with contextlib.suppress(asyncio.CancelledError):
            async for _ in self.client.watch_connection():
                await self._after_connect()
                self.notify("Reconnected to the daemon.", timeout=3)

    def _on_connection_change(self, connected: bool) -> None:
        self.state.connected = connected
        self._refresh_screen()

    # -- data ------------------------------------------------------------------------------

    async def resync(self) -> None:
        """Refetch everything. Used on connect, on request, and after an event overflow."""
        try:
            snapshot = await self.client.call(Method.SYSTEM_SNAPSHOT)
            page = await self.client.call(Method.JOB_LIST, limit=500)
        except DispatchError as exc:
            self.state.error = str(exc)
            return
        self.state.snapshot = snapshot
        self.state.replace_jobs(page["items"])
        self.state.error = None
        self._refresh_screen()

    def _on_event(self, notification: Notification) -> None:
        """Apply a pushed event to the cached state."""
        event = notification.event
        data = notification.data

        if event == Event.SYSTEM_STATS:
            self.state.snapshot = data
        elif event == Event.JOB_STATE:
            self.state.update_job(data)
        elif event == Event.JOB_PROGRESS:
            self.state.update_progress(data)
        elif event == Event.QUEUE_CHANGED:
            if deleted := data.get("deleted"):
                self.state.forget(str(deleted))
            self.call_later(self.resync)
            return
        elif event == Event.RESYNC:
            # The daemon dropped our backlog because we fell behind. Refetch rather than
            # rendering a state we know is incomplete.
            self.call_later(self.resync)
            return
        elif event == Event.DAEMON_SHUTDOWN:
            self.state.connected = False
            self.notify("The daemon is shutting down. Simulations keep running.", timeout=8)

        self._refresh_screen()

    def _refresh_screen(self) -> None:
        with contextlib.suppress(Exception):
            screen = self.screen
            if hasattr(screen, "refresh_view"):
                screen.refresh_view()

    # -- helpers for screens ------------------------------------------------------------------

    async def call(self, method: str, **params: Any) -> Any:
        """Make a request, surfacing errors as notifications rather than crashes."""
        return await self.client.call(method, **params)

    def send(self, method: str, **params: Any) -> None:
        """Fire a request without waiting, reporting failure as a notification."""

        async def _run() -> None:
            try:
                await self.client.call(method, **params)
            except DispatchError as exc:
                self.notify(str(exc), severity="error", timeout=8)

        self.call_later(_run)

    def status_summary(self) -> Text:
        """The live half of the top bar: host, job counts, clock.

        Counts appear only when non-zero. An idle machine should show an almost empty bar
        rather than a row of zeros, so that a number appearing means something happened.
        """
        state = self.state
        counts = state.counts

        # No "offline" marker here: the banner above already says so, and saying it twice
        # in one glance is how a status bar becomes noise.
        text = Text()
        for key, label in (
            ("RUNNING", "running"),
            ("PREPARING", "preparing"),
            ("QUEUED", "queued"),
            ("HELD", "held"),
        ):
            count = counts.get(key, 0)
            if not count:
                continue
            text.append(str(count), style=state_style(key))
            text.append(f" {label}", style=Palette.MUTED)
            text.append("   ")

        if state.hostname:
            text.append(state.hostname, style=Palette.MUTED)
            text.append("   ")
        text.append(datetime.now().strftime("%H:%M"), style=Palette.FAINT)
        return text

    def open_logs(self, job_id: str) -> None:
        """Open the live log viewer for a job."""
        self.push_screen(LogScreen(job_id))

    # -- actions --------------------------------------------------------------------------------

    async def action_dashboard(self) -> None:
        await self._switch(DashboardScreen())

    async def action_queue(self) -> None:
        await self._switch(QueueScreen())

    async def action_history(self) -> None:
        await self._switch(HistoryScreen())

    async def action_new_job(self) -> None:
        start = self._start_directory()
        await self.push_screen(SubmitScreen(start))

    async def action_help(self) -> None:
        await self.push_screen(HelpScreen())

    async def action_resync(self) -> None:
        await self.resync()
        self.notify("Refreshed.", timeout=2)

    async def _switch(self, screen: Any) -> None:
        """Replace the base screen, dismissing anything stacked above it.

        Textual's own default screen sits at the bottom of the stack and must not be
        swapped out, so the base Dispatch screen is the one at index 1 -- everything above
        it is popped, and then that one is replaced.
        """
        while len(self.screen_stack) > 2:
            self.pop_screen()
        if len(self.screen_stack) > 1:
            await self.switch_screen(screen)
        else:
            await self.push_screen(screen)

    def _start_directory(self) -> Path:
        """Where the browser opens.

        The configured start path, else the current directory, else home -- so that
        ``cd my-case && dispatch`` lands somewhere useful.
        """
        configured = self.config.adapters.get("browser", {}).get("start")
        if configured:
            candidate = Path(str(configured)).expanduser()
            if candidate.is_dir():
                return candidate
        with contextlib.suppress(OSError):
            return Path.cwd()
        return Path.home()
