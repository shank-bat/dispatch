"""Searching the history.

The search box takes the same query language as ``dispatch search``, because two
implementations of a query language is one too many.

It is docked at the top and looks like a search field rather than a form control: no
label, no border, just a prompt character and the caret. The syntax reminder lives at the
bottom in the faintest text on screen, where it is available without being read every time.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Static

from dispatch.ipc.protocol import Method
from dispatch.tui.screens.base import DispatchScreen, run_when_confirmed
from dispatch.tui.theme import Palette
from dispatch.tui.widgets.jobtable import JobTable

__all__ = ["HistoryScreen"]

EXAMPLES = "tag:paper   solver:openfoam   resource:gpu   cores>=16   endTime>500   after:7d"


class HistoryScreen(DispatchScreen):
    """Full-text and structured search across every job ever run."""

    TITLE = "History"
    nav_key = "3"

    BINDINGS = [
        Binding("slash", "focus_search", "search"),
        Binding("enter", "open", "logs"),
        Binding("p", "plot", "plot"),
        Binding("t", "edit_tags", "tags"),
        Binding("n", "add_note", "note"),
        Binding("delete,d", "delete", "delete"),
        Binding("r", "refresh_results", "reload"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []
        self.total = 0
        self._prompt_mode = ""

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical():
            yield Input(placeholder="search jobs…", id="search")
            yield Static("", id="heading")
            yield JobTable(id="results", show_progress=False)
            yield Static(Text(EXAMPLES, style=Palette.FAINT), id="hint")
            yield Input(id="prompt", classes="prompt")
        yield from self.compose_footer()

    async def on_mount(self) -> None:
        await super().on_mount()
        await self.run_search("")
        self.query_one("#results", JobTable).focus()

    async def run_search(self, query: str) -> None:
        """Ask the daemon; a query error is shown rather than swallowed."""
        try:
            page = await self.dispatch_app.call(Method.HISTORY_SEARCH, query=query, limit=200)
        except Exception as exc:
            self.notify_error(str(exc))
            self.query_one("#hint", Static).update(Text(str(exc), style=Palette.ERROR))
            self.update_status()
            return

        self.results = page["items"]
        self.total = page["total"]
        self.query_one("#results", JobTable).show(self.results)

        summary = Text()
        summary.append(f"{self.total} result{'' if self.total == 1 else 's'}", style=Palette.MUTED)
        if page["has_more"]:
            summary.append("  showing first 200", style=Palette.FAINT)
        else:
            summary.append(f"   {EXAMPLES}", style=Palette.FAINT)
        self.query_one("#hint", Static).update(summary)
        self.update_status()

    def heading(self) -> Text:
        return Text("history", style=f"bold {Palette.TEXT}")

    # -- actions -----------------------------------------------------------------------

    def _selected(self) -> dict[str, Any] | None:
        job_id = self.query_one("#results", JobTable).selected_job_id
        if job_id is None:
            return None
        return next((job for job in self.results if job["id"] == job_id), None)

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    async def action_refresh_results(self) -> None:
        await self.run_search(self.query_one("#search", Input).value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search":
            await self.run_search(event.value)
            self.query_one("#results", JobTable).focus()
            return

        value = event.value.strip()
        box = self.query_one("#prompt", Input)
        box.remove_class("visible")
        self.query_one("#results", JobTable).focus()
        job = self._selected()
        if job is None or not value:
            return

        if self._prompt_mode == "tags":
            add = [t for t in value.split() if not t.startswith("-")]
            remove = [t[1:] for t in value.split() if t.startswith("-")]
            await self._send(Method.JOB_TAG, id=job["id"], add=add, remove=remove)
        elif self._prompt_mode == "note":
            await self._send(Method.JOB_NOTE, id=job["id"], body=value)
        await self.action_refresh_results()

    async def _send(self, method: str, **params: Any) -> None:
        try:
            await self.dispatch_app.call(method, **params)
        except Exception as exc:
            self.notify_error(str(exc))

    def action_edit_tags(self) -> None:
        """Tags are editable long after a job finishes -- which is when you know."""
        job = self._selected()
        if job is None:
            return
        self._open_prompt("tags", f"tags for {job['name']}  (-tag removes)")

    def action_add_note(self) -> None:
        job = self._selected()
        if job is None:
            return
        self._open_prompt("note", f"note for {job['name']}")

    def _open_prompt(self, mode: str, placeholder: str) -> None:
        self._prompt_mode = mode
        box = self.query_one("#prompt", Input)
        box.placeholder = placeholder
        box.value = ""
        box.add_class("visible")
        box.focus()

    def action_open(self) -> None:
        job = self._selected()
        if job is not None:
            self.dispatch_app.open_logs(job["id"])

    def action_plot(self) -> None:
        """Plot a finished run's residuals, months after it finished."""
        job = self._selected()
        if job is not None:
            self.dispatch_app.open_plot(job["id"])

    def action_delete(self) -> None:
        job = self._selected()
        if job is None:
            return
        if job["state"] in ("QUEUED", "HELD", "PREPARING", "RUNNING"):
            self.notify_error("Cancel the job before deleting it.")
            return

        async def _delete() -> None:
            await self._send(Method.JOB_DELETE, id=job["id"])
            await self.action_refresh_results()

        run_when_confirmed(
            self,
            f"Delete {job['name']} from the history?",
            lambda: self.app.call_later(_delete),
            detail="The record goes permanently. Its log files are kept.",
        )
