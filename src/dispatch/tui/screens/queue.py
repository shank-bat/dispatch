"""The queue screen: everything waiting or running, and the keys to change it."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static

from dispatch.ipc.protocol import Method
from dispatch.tui.screens.base import DispatchScreen, run_when_confirmed
from dispatch.tui.theme import Palette
from dispatch.tui.widgets.jobtable import JobTable

__all__ = ["QueueScreen"]


class QueueScreen(DispatchScreen):
    """Active and queued jobs, with hold, release, priority, and cancel."""

    TITLE = "Queue"
    nav_key = "2"

    BINDINGS = [
        Binding("x", "cancel", "cancel"),
        Binding("h", "hold", "hold"),
        Binding("H", "release", "release"),
        Binding("plus,equals_sign,k", "raise_priority", "priority"),
        Binding("minus,underscore,j", "lower_priority", "", show=False),
        Binding("enter", "open", "logs"),
    ]

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical():
            yield Static("", id="heading")
            yield JobTable(id="queue-table")
            yield Static("", id="why")
        yield from self.compose_footer()

    async def on_mount(self) -> None:
        await super().on_mount()
        self.set_interval(1.0, self.refresh_view)
        self.refresh_view()
        self.query_one("#queue-table", JobTable).focus()

    def refresh_view(self) -> None:
        state = self.app_state
        jobs = state.running + state.queued
        self.query_one("#queue-table", JobTable).show(jobs, state.progress)
        self.query_one("#why", Static).update(self._explanation())
        self.update_status()

    def heading(self) -> Text:
        """Counts, as a sentence rather than a row of badges."""
        counts = self.app_state.counts
        active = counts.get("RUNNING", 0) + counts.get("PREPARING", 0)
        queued = counts.get("QUEUED", 0)
        held = counts.get("HELD", 0)

        text = Text("queue", style=f"bold {Palette.TEXT}")
        text.append("   ")
        text.append(f"{active} active", style=Palette.MUTED)
        text.append("   ")
        text.append(f"{queued} waiting", style=Palette.MUTED)
        if held:
            text.append("   ")
            text.append(f"{held} held", style=Palette.WARNING)
        return text

    def _explanation(self) -> Text:
        """Why the selected job is not running.

        Shown as a footnote under the table rather than a panel, because it applies to
        whichever row the cursor is on and changes as you move.
        """
        job = self._selected()
        if job is None:
            return Text("")
        if job["state"] in ("RUNNING", "PREPARING"):
            return Text(f"running on {job['cores']} cores", style=Palette.FAINT)
        if job["state"] == "HELD":
            return Text("held — press H to release", style=Palette.WARNING)

        parent_id = job.get("depends_on_job_id")
        if parent_id:
            parent = self.app_state.get(parent_id)
            if parent is None or parent["state"] != "COMPLETED":
                name = parent["name"] if parent else parent_id[:8]
                return Text(f"waiting for {name}", style=Palette.WARNING)

        free = int(self.app_state.snapshot.get("free_cores", 0))
        if job["cores"] > free:
            return Text(
                f"needs {job['cores']} cores, {free} free", style=Palette.WARNING
            )
        return Text("next in line", style=Palette.FAINT)

    # -- actions ------------------------------------------------------------------------

    def _selected(self) -> dict[str, Any] | None:
        job_id = self.query_one("#queue-table", JobTable).selected_job_id
        return self.app_state.get(job_id) if job_id else None

    def action_cancel(self) -> None:
        job = self._selected()
        if job is None:
            return
        run_when_confirmed(
            self,
            f"Cancel {job['name']}?",
            lambda: self.dispatch_app.send(Method.JOB_CANCEL, id=job["id"]),
            detail="The solver is asked to stop cleanly where it can.",
        )

    def action_hold(self) -> None:
        job = self._selected()
        if job and job["state"] == "QUEUED":
            self.dispatch_app.send(Method.JOB_HOLD, id=job["id"])

    def action_release(self) -> None:
        job = self._selected()
        if job and job["state"] == "HELD":
            self.dispatch_app.send(Method.JOB_RELEASE, id=job["id"])

    def action_raise_priority(self) -> None:
        self._nudge_priority(+1)

    def action_lower_priority(self) -> None:
        self._nudge_priority(-1)

    def _nudge_priority(self, delta: int) -> None:
        job = self._selected()
        if job is None:
            return
        self.dispatch_app.send(
            Method.JOB_PRIORITY, id=job["id"], priority=int(job["priority"]) + delta
        )

    def action_open(self) -> None:
        job = self._selected()
        if job is not None:
            self.dispatch_app.open_logs(job["id"])
