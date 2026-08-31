"""Finding a case by typing part of its name.

The submit wizard is a directory browser, and that is the right default: Dispatch does not
guess where your work lives, so you show it. But the case you want is often four levels
down a tree you last opened in March, and browsing to it is eight keystrokes of
remembering.

So: ``/`` from the new-job screen, type ``cavity``, and the directories under
``~/projects`` whose names contain it appear, best match first. Enter puts one in the
browser, and the wizard carries on exactly as before -- detection, validation, cores,
confirm. Nothing about submission changes; this only replaces the walking.

Search runs in the daemon (§9.5), which is also what lets each result carry the "this is
an OpenFOAM case" mark: the marks come from the real adapters, and this screen stays as
solver-ignorant as the rest of the interface.

Typing is debounced by a keystroke's worth of scheduling rather than by a timer: each
input change starts a search, and a search that finishes after a newer one has started
discards its own result. That keeps the list matching what has been typed without a
polling loop anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView, Static

from dispatch.ipc.protocol import Method
from dispatch.tui.theme import Palette

__all__ = ["ProjectSearchScreen"]


class ProjectSearchScreen(ModalScreen[str | None]):
    """Search project directories by name.

    Dismisses with the chosen directory, or ``None`` when the user backs out and whatever
    the wizard was showing should stand.
    """

    BINDINGS = [
        Binding("escape", "cancel", "cancel"),
        Binding("down,ctrl+n", "next", "next", show=False),
        Binding("up,ctrl+p", "previous", "previous", show=False),
        Binding("enter", "choose", "open"),
    ]

    def __init__(self, initial: str = "") -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []
        self.root = ""
        self._initial = initial
        self._generation = 0
        """Increments per search, so a slow reply for stale input is discarded."""

    def compose(self) -> ComposeResult:
        with Vertical(id="project-search"):
            yield Static(Text("search projects", style=f"bold {Palette.TEXT}"))
            yield Input(placeholder="part of a directory name…", id="project-query")
            yield ListView(id="project-results")
            yield Static("", id="project-status")

    async def on_mount(self) -> None:
        box = self.query_one("#project-query", Input)
        box.value = self._initial
        box.focus()
        await self.search(self._initial)

    # -- searching ------------------------------------------------------------------------

    async def search(self, query: str) -> None:
        """Ask the daemon, unless a newer keystroke has already superseded this one."""
        self._generation += 1
        generation = self._generation
        try:
            payload = await self.app.call(Method.PROJECTS_SEARCH, query=query)  # type: ignore[attr-defined]
        except Exception as exc:
            if generation == self._generation:
                self._show_status(Text(str(exc), style=Palette.ERROR))
            return
        if generation != self._generation:
            return

        self.root = str(payload.get("root") or "")
        self.results = list(payload.get("results") or [])
        await self._fill()
        self._show_status(_summary(payload))

    async def _fill(self) -> None:
        view = self.query_one("#project-results", ListView)
        await view.clear()
        for entry in self.results:
            view.append(ListItem(Static(_entry_text(entry))))
        if self.results:
            view.index = 0

    def _show_status(self, text: Text) -> None:
        self.query_one("#project-status", Static).update(text)

    async def on_input_changed(self, event: Input.Changed) -> None:
        await self.search(event.value.strip())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter from the box takes the best match, which is what the ordering is for."""
        self.action_choose()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.action_choose()

    # -- actions ---------------------------------------------------------------------------

    def action_choose(self) -> None:
        view = self.query_one("#project-results", ListView)
        index = view.index or 0
        if not self.results or not 0 <= index < len(self.results):
            return
        self.dismiss(str(self.results[index]["path"]))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_next(self) -> None:
        self._move(1)

    def action_previous(self) -> None:
        self._move(-1)

    def _move(self, delta: int) -> None:
        """Move the cursor without leaving the input box.

        The arrows have to work while typing -- search, look, adjust, choose is one motion
        -- so they are handled here rather than by focusing the list.
        """
        view = self.query_one("#project-results", ListView)
        if not self.results:
            return
        position = (view.index or 0) + delta
        view.index = max(0, min(len(self.results) - 1, position))


def _entry_text(entry: dict[str, Any]) -> Text:
    """One result: its path below the root, and what kind of case it looks like.

    The relative path rather than the absolute one. Every result shares the same root, so
    repeating it on every line spends the width that distinguishes the results on the part
    that does not.
    """
    text = Text()
    solver = entry.get("case")
    text.append("● " if solver else "  ", style=Palette.SUCCESS)

    relative = str(entry.get("relative") or entry.get("path") or "")
    parent, _, name = relative.rpartition("/")
    if parent:
        text.append(f"{parent}/", style=Palette.FAINT)
    text.append(name or relative, style=Palette.TEXT)
    if solver:
        text.append(f"  {solver}", style=Palette.MUTED)
    return text


def _summary(payload: dict[str, Any]) -> Text:
    """The line under the results: where it looked, and what it did not look at."""
    error = payload.get("error")
    if error:
        return Text(str(error), style=Palette.WARNING)

    count = len(payload.get("results") or [])
    text = Text()
    text.append(f"{count} match{'' if count == 1 else 'es'}", style=Palette.MUTED)
    text.append(f"   in {_tilde(str(payload.get('root') or ''))}", style=Palette.FAINT)
    if payload.get("truncated"):
        # Never implied: a bounded walk that found more than it returned has to say so, or
        # "not in the list" reads as "not on the machine".
        text.append("   (more exist; narrow the search)", style=Palette.WARNING)
    return text


def _tilde(path: str) -> str:
    """Abbreviate the user's home, the way every other path in the interface is written."""
    try:
        return f"~/{Path(path).relative_to(Path.home())}"
    except ValueError:
        return path
