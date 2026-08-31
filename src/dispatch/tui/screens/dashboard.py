"""The dashboard: what the machine is doing, right now.

Push-driven. There is no polling of the daemon anywhere in this screen -- it renders from
:class:`~dispatch.tui.state.AppState`, which events update. The one local timer drives the
wall clock and the elapsed-time column, both of which advance on their own.

The layout is three bands with space between them: machine status, what is running, what
recently finished. Section labels are quiet lowercase text rather than boxed panel titles,
because the tables beneath them are already unmistakable.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static

from dispatch.tui.screens.base import DispatchScreen
from dispatch.tui.theme import Palette
from dispatch.tui.widgets.jobtable import JobTable
from dispatch.tui.widgets.meters import ResourceMeters

__all__ = ["DashboardScreen"]


class DashboardScreen(DispatchScreen):
    """Running jobs, the queue head, recent completions, and machine load."""

    TITLE = "Dashboard"
    nav_key = "1"

    BINDINGS = [
        Binding("enter", "open", "logs"),
        Binding("p", "plot", "plot"),
    ]
    """The dashboard is where a running job is being watched, so the two things worth
    doing to one from here -- read its output, plot its numbers -- are bound directly."""

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical():
            yield ResourceMeters(id="meters")
            yield Static(_section("active"), classes="section")
            yield JobTable(id="running")
            yield Static("", id="next-up")
            yield Static(_section("recent"), classes="section")
            yield JobTable(id="recent", show_progress=False)
        yield from self.compose_footer()

    async def on_mount(self) -> None:
        await super().on_mount()
        # One second: enough for a wall clock and a ticking elapsed column, and cheap
        # because it touches only local state.
        self.set_interval(1.0, self.refresh_view)
        self.refresh_view()
        self.query_one("#running", JobTable).focus()

    def refresh_view(self) -> None:
        """Re-render from the cached state."""
        state = self.app_state

        self.query_one("#meters", ResourceMeters).snapshot = state.snapshot
        self.query_one("#running", JobTable).show(state.running, state.progress)
        self.query_one("#recent", JobTable).show(state.finished[:8])
        self.query_one("#next-up", Static).update(self._next_up())
        self.update_status()

    def _selected(self) -> str | None:
        """The job under the cursor in whichever table has focus.

        Two tables share this screen, so "the selected job" depends on where the cursor
        is; falling back to the active table matches what somebody glancing at the screen
        would mean by it.
        """
        for widget_id in ("#running", "#recent"):
            table = self.query_one(widget_id, JobTable)
            if table.has_focus and table.selected_job_id:
                return table.selected_job_id
        return self.query_one("#running", JobTable).selected_job_id

    def action_open(self) -> None:
        job_id = self._selected()
        if job_id:
            self.dispatch_app.open_logs(job_id)

    def action_plot(self) -> None:
        job_id = self._selected()
        if job_id:
            self.dispatch_app.open_plot(job_id)

    def _next_up(self) -> Text:
        """One line describing what runs next, and why it has not started."""
        state = self.app_state
        queued = state.queued
        if not queued:
            return Text("queue empty", style=Palette.FAINT)

        head = state.next_job
        if head is None:
            return Text(f"{len(queued)} held", style=Palette.WARNING)

        text = Text("next  ", style=Palette.FAINT)
        text.append(head["name"], style=Palette.TEXT)
        text.append(f"  {head['cores']} cores", style=Palette.MUTED)

        free = int(state.snapshot.get("free_cores", 0))
        if head["cores"] > free:
            text.append(
                f"  waiting for {head['cores'] - free} more", style=Palette.WARNING
            )
        else:
            text.append("  starting shortly", style=Palette.SUCCESS)

        remaining = len(queued) - 1
        if remaining > 0:
            text.append(f"   +{remaining} queued", style=Palette.FAINT)
        return text


def _section(label: str) -> Text:
    """A section label: lowercase, faint, no rule beneath it."""
    return Text(label, style=Palette.FAINT)
