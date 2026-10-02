"""The dashboard: what the machine is doing, right now.

Push-driven. There is no polling of the daemon anywhere in this screen -- it renders from
:class:`~dispatch.tui.state.AppState`, which events update. The one local timer drives the
wall clock and the elapsed-time column, both of which advance on their own.

The layout is three bands with space between them: machine status, what is running, what
recently finished. Section labels are quiet lowercase text rather than boxed panel titles,
because the tables beneath them are already unmistakable.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static

from dispatch.ipc.protocol import Method
from dispatch.tui.screens.base import DispatchScreen
from dispatch.tui.state import sweep_summary
from dispatch.tui.theme import Palette
from dispatch.tui.widgets.jobtable import JobTable

__all__ = ["DashboardScreen"]


DETAIL_VALUES = 2
"""How many measured values the collapsed-by-default expander shows.

Two, because the dashboard's worth is that it fits on one screen and the first two are the
ones asked for. The full set is one keypress away in the plot view.
"""


class DashboardScreen(DispatchScreen):
    """Running jobs, the queue head, recent completions, and machine load."""

    TITLE = "Dashboard"
    nav_key = "1"

    BINDINGS = [
        Binding("enter", "open", "logs"),
        Binding("p", "plot", "plot"),
        Binding("i", "case_info", "info"),
        Binding("space", "expand", "details"),
    ]
    """The dashboard is where a running job is being watched, so the things worth doing to
    one from here -- read its output, plot its numbers, glance at its coefficients -- are
    bound directly."""

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical():
            # No meters line of its own: the header's stats grid (see
            # screens/base.py:compose_header, widgets/meters.py:HeaderStats) already shows
            # cores, CPU, memory, GPU, load and core-hours beside the logo on every screen,
            # this one included. Repeating the same six numbers immediately below it would
            # be exactly the clutter the grid was built to avoid.
            yield Static(_section("active"), classes="section")
            yield JobTable(id="running")
            yield Static("", id="details")
            yield Static("", id="next-up")
            yield Static("", id="sweeps")
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

    def __init__(self) -> None:
        super().__init__()
        self._expanded: str | None = None
        """Which job's details are showing, if any. ``None`` keeps the dashboard compact."""

        self._metrics: dict[str, dict[str, Any]] = {}
        """Latest case values by job id, as the daemon last reported them.

        Cached so the one-second repaint renders from memory. Fetched only when a job is
        expanded and when its row changes -- never on a timer, because reading a case's
        output files on every tick for a job nobody is looking at is exactly the kind of
        background work Dispatch does not do (§6.1).
        """

    def refresh_view(self) -> None:
        """Re-render from the cached state."""
        state = self.app_state

        self.query_one("#running", JobTable).show(state.running, state.progress)
        self.query_one("#recent", JobTable).show(state.finished[:8])
        self.query_one("#details", Static).update(self._details())
        self.query_one("#next-up", Static).update(self._next_up())
        self.query_one("#sweeps", Static).update(self._sweeps())
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

    def action_case_info(self) -> None:
        job_id = self._selected()
        if job_id:
            self.dispatch_app.open_case_info(job_id=job_id)

    # -- the expander ---------------------------------------------------------------------

    def action_expand(self) -> None:
        """Show, or hide, the selected job's latest case values.

        Collapsed by default and collapsible again, because the dashboard's value is that it
        fits on one screen. Expanding one job is a question about that job, not a change of
        view.
        """
        job_id = self._selected()
        if job_id is None:
            return
        if self._expanded == job_id:
            self._expanded = None
            self.refresh_view()
            return
        self._expanded = job_id
        self.app.call_later(self._fetch_metrics, job_id)

    async def _fetch_metrics(self, job_id: str) -> None:
        """Ask the daemon for the case's latest values.

        Reads the case's own output files and not the log, so this costs a stat and a short
        read rather than parsing a run's worth of residuals.
        """
        try:
            payload = await self.dispatch_app.call(Method.JOB_METRICS, id=job_id)
        except Exception as exc:
            self._metrics[job_id] = {}
            self.notify_error(str(exc))
            return
        self._metrics[job_id] = dict(payload.get("datasets") or {})
        self.refresh_view()

    def _details(self) -> Text:
        """The expanded job's latest case values, or an honest line saying there are none.

        The labels and their order come from the daemon, which got them from the adapter:
        this screen renders ``Cl (lift) +0.714`` without knowing that lift exists, which is
        what keeps the interface solver-ignorant (§3).
        """
        if self._expanded is None:
            return Text("")
        job = self.app_state.get(self._expanded)
        if job is None:
            return Text("")

        text = Text("  ")
        text.append(str(job["name"]), style=Palette.TEXT)
        text.append("   ", style=Palette.FAINT)

        if self._expanded not in self._metrics:
            text.append("reading the case…", style=Palette.FAINT)
            return text

        datasets = self._metrics[self._expanded]
        if not datasets:
            # Most cases write no post-processing output, and that is not a failure.
            text.append("no case output to summarise", style=Palette.FAINT)
            return text

        for dataset in datasets.values():
            values = [
                entry for entry in (dataset.get("values") or []) if isinstance(entry, dict)
            ]
            axis = next((entry for entry in values if entry.get("axis")), None)
            measured = [entry for entry in values if not entry.get("axis")][:DETAIL_VALUES]

            for entry in measured:
                text.append(f"{entry['label']} ", style=Palette.FAINT)
                text.append(f"{float(entry['value']):+.5g}", style=Palette.ACCENT_TEXT)
                text.append("   ")
            if axis is not None:
                text.append(
                    f"at {axis['label'].lower()} {float(axis['value']):g}", style=Palette.FAINT
                )
                text.append("   ")

        text.append("space to collapse · p to plot", style=Palette.FAINT)
        return text

    def _sweeps(self) -> Text:
        """A line per active sweep, or nothing at all when none are running.

        The dashboard's job is what the machine is doing right now, and a sweep running two
        of eight cases is a fact about the machine that the rows alone do not convey: the
        row marker says a job belongs to a sweep, this says why the other six are not
        moving.
        """
        state = self.app_state
        unfinished = {
            job.get("sweep_id")
            for job in state.jobs.values()
            if job.get("sweep_id")
            and job["state"] in ("QUEUED", "HELD", "PREPARING", "RUNNING")
        }
        active = [sweep for sweep in state.sweeps.values() if sweep["id"] in unfinished]
        if not active:
            return Text("")

        text = Text()
        for index, sweep in enumerate(active):
            if index:
                text.append("\n")
            text.append(sweep_summary(state, sweep), style=Palette.ACCENT)
        return text

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
