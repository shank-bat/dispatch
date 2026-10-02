"""The case information view: everything about a case, on one page.

``i`` on a directory in the submit browser, or on a job anywhere else. What appears is
assembled by the *adapter* -- see :mod:`dispatch.core.caseinfo` -- and this screen only lays
it out. It does not know what a ``controlDict`` is, how to count cells, or that force
coefficients exist, which is what keeps it from being an OpenFOAM screen that another solver
would have to reimplement.

So the layout code here is genuinely generic: titled sections, aligned label/value pairs, a
faint note beside a value, and a warning block at the top. A new solver that implements
``describe_case`` gets this page with no change to this file.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from dispatch.ipc.protocol import Method
from dispatch.tui.screens.base import DispatchScreen
from dispatch.tui.theme import Palette

__all__ = ["CaseInfoScreen"]

LABEL_WIDTH = 20
"""Label column width. Values align regardless of label length, which is what lets the eye
run down the values without re-finding the left edge on every line."""


class CaseInfoScreen(DispatchScreen):
    """A full, readable description of one case."""

    TITLE = "Case"
    nav_key = ""

    BINDINGS = [
        Binding("escape,q,i", "back", "back"),
        Binding("r", "reload", "reload"),
        Binding("v", "render", "render image"),
        Binding("V", "render_animation", "render video"),
        Binding("x", "cancel_render", "cancel render"),
    ]

    def __init__(self, path: str | None = None, *, job_id: str | None = None) -> None:
        """Args:
        path: The case directory. Used by the submit browser, where nothing is submitted yet.
        job_id: A job whose working directory to describe. Used everywhere else.
        """
        super().__init__()
        self.path = path
        self.job_id = job_id
        self.report: dict[str, Any] | None = None
        self.error: str | None = None

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical():
            yield Static("", id="heading")
            yield Static("", id="case-info")
        yield from self.compose_footer()

    async def on_mount(self) -> None:
        await super().on_mount()
        await self.load()

    async def load(self) -> None:
        """Ask the daemon to describe the case."""
        params: dict[str, Any] = {}
        if self.path is not None:
            params["path"] = self.path
        if self.job_id is not None:
            params["id"] = self.job_id
        try:
            self.report = await self.dispatch_app.call(Method.CASE_INFO, **params)
            self.error = None
        except Exception as exc:
            self.report = None
            self.error = str(exc)
        self._redraw()

    async def action_reload(self) -> None:
        """Re-read the case. The obvious key to press on a case that is still running."""
        await self.load()

    def action_back(self) -> None:
        self.dismiss()

    def action_render(self) -> None:
        """Render a still of the mesh: choose an angle, then what to colour it by."""
        self._choose_render("mesh")

    def action_render_animation(self) -> None:
        """Render a video over every written time: choose an angle, then a field."""
        self._choose_render("animation")

    def action_cancel_render(self) -> None:
        """Stop the render running for this case, keeping any frames already written."""
        running = self._renders()
        if not running:
            self.notify_error("No render is running for this case.")
            return
        self.dispatch_app.send(Method.RENDER_CANCEL, id=running[0]["id"])

    def _choose_render(self, kind: str) -> None:
        def _angle(preset: str | None) -> None:
            if preset:
                self.app.call_later(self._choose_field, kind, preset)

        self.app.push_screen(
            ChoiceScreen("camera angle", [(name, name) for name in PRESETS], initial="isometric"),
            _angle,
        )

    async def _choose_field(self, kind: str, preset: str) -> None:
        """Offer exactly the fields this case has, as the adapter read them.

        A still may also be the plain mesh -- that is what a mesh screenshot usually is. An
        animation needs something that changes, so the plain mesh is not offered for one
        unless the case has no fields at all.
        """
        try:
            listing = await self.dispatch_app.call(Method.CASE_FIELDS, **self._target())
        except Exception as exc:
            self.notify_error(str(exc))
            return
        fields = [(str(item["name"]), str(item["label"])) for item in listing["fields"]]

        options: list[tuple[str, str]] = []
        if kind == "mesh" or not fields:
            options.append(("", "none (plain mesh, edges shown)"))
        options.extend(fields)

        def _field(choice: str | None) -> None:
            if choice is not None:
                self.app.call_later(self._start_render, kind, preset, choice or None)

        title = "colour by" if kind == "mesh" else "animate which field"
        self.app.push_screen(ChoiceScreen(title, options), _field)

    async def _start_render(self, kind: str, preset: str, field: str | None) -> None:
        """Start the render in the background; progress shows here, the result anywhere.

        Not awaited to completion: an animation can take hours, and the daemon answers one
        interface's requests in turn, so waiting here would freeze everything else this
        interface asks for until it finished.
        """
        params: dict[str, Any] = {**self._target(), "kind": kind, "preset": preset}
        if field:
            params["field"] = field
        try:
            result = await self.dispatch_app.call(Method.CASE_RENDER, **params)
        except Exception as exc:
            self.notify_error(str(exc))
            return
        for note in result.get("notes") or []:
            self.notify_ok(str(note))
        what = "video" if kind == "animation" else "image"
        self.notify_ok(f"Rendering the {what} in the background; x cancels")
        self.refresh_view()

    def _target(self) -> dict[str, Any]:
        if self.path is not None:
            return {"path": self.path}
        return {"id": self.job_id}

    def _renders(self) -> list[dict[str, Any]]:
        case = (self.report or {}).get("path")
        return self.app_state.renders_for(str(case)) if case else []

    def refresh_view(self) -> None:
        """Repaint when a render's progress arrives."""
        if self.report is not None:
            self._redraw()


    def heading(self) -> Text:
        text = Text("case", style=f"bold {Palette.TEXT}")
        if self.report:
            text.append(f"   {self.report.get('title', '')}", style=Palette.TEXT)
            text.append(f"   {self.report.get('solver', '')}", style=Palette.MUTED)
        return text

    def _redraw(self) -> None:
        """Repaint from the loaded report.

        Not called ``_render``: Textual's ``Widget`` already owns that name for producing a
        renderable, and shadowing it makes a screen that silently fails to paint.
        """
        body = self.query_one("#case-info", Static)
        if self.error is not None:
            body.update(Text(self.error, style=Palette.ERROR))
            self.update_status()
            return
        if not self.report:
            body.update(Text("reading the case…", style=Palette.FAINT))
            self.update_status()
            return
        body.update(_render_report(self.report, self._renders()))
        self.update_status()


def _render_report(report: dict[str, Any], renders: list[dict[str, Any]] | None = None) -> Text:
    """Lay out a case report: renders in progress, warnings, then each section's fields."""
    text = Text()
    for item in renders or []:
        frame, frames = item.get("frame"), item.get("frames")
        detail = f"frame {frame}/{frames}" if frame and frames else str(item.get("description", ""))
        text.append("  ▶ ", style=Palette.ACCENT)
        text.append(f"rendering {item.get('kind')}: {detail}", style=Palette.TEXT)
        text.append(
            f"   step {item.get('step')}/{item.get('steps')} · x cancels\n", style=Palette.FAINT
        )

    path = report.get("path")
    if path:
        text.append(f"{path}\n", style=Palette.FAINT)

    warnings = list(report.get("warnings") or [])
    if warnings:
        # At the top, because the usual reason to open this page is that something is wrong.
        text.append("\n")
        for warning in warnings:
            text.append("  ! ", style=Palette.WARNING)
            text.append(f"{warning}\n", style=Palette.TEXT)

    for section in report.get("sections") or []:
        fields = list(section.get("fields") or [])
        missing = str(section.get("missing") or "")
        if not fields and not missing:
            continue

        text.append("\n")
        text.append(f"{section.get('title', '')}\n", style=Palette.FAINT)
        if missing:
            text.append(f"  {missing}\n", style=Palette.MUTED)
        for item in fields:
            _field(text, item)

    if len(text) == 0:
        text.append("nothing to report about this case", style=Palette.MUTED)
    return text


def _field(text: Text, item: dict[str, Any]) -> None:
    """One aligned ``label   value   note`` line."""
    label = str(item.get("label", ""))
    text.append(f"  {label:<{LABEL_WIDTH}}", style=Palette.FAINT)
    text.append(
        str(item.get("value", "")),
        # Emphasis is the adapter's call: it marks the handful of fields that answer "is this
        # case the one I think it is".
        style=Palette.ACCENT_TEXT if item.get("important") else Palette.TEXT,
    )
    note = str(item.get("note") or "")
    if note:
        text.append(f"   {note}", style=Palette.FAINT)
    text.append("\n")


PRESETS = ("front", "back", "left", "right", "top", "bottom", "isometric")
"""Kept in step with :class:`dispatch.adapters.paraview.CameraPreset`, which the interface may
not import -- the TUI does not depend on adapters (§3). A name this list has wrong is refused
by the daemon with the valid ones, rather than silently rendering something else."""


class ChoiceScreen(ModalScreen[str | None]):
    """Pick one of a short list of named options.

    Dismisses with the chosen value, or ``None`` when the user backs out. A list rather than a
    typed name: the options are the whole vocabulary, and nobody should have to remember which
    spellings -- or which of this case's fields -- exist.
    """

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(
        self, title: str, options: list[tuple[str, str]], *, initial: str | None = None
    ) -> None:
        """Args:
        title: What is being chosen.
        options: ``(value, label)`` pairs, in display order.
        initial: Value to start the cursor on.
        """
        super().__init__()
        self.title_text = title
        self.options = options
        self._initial = initial

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(Text(self.title_text, style=f"bold {Palette.TEXT}"))
            yield ListView(
                *(ListItem(Label(Text(label))) for _, label in self.options),
                id="choice-list",
            )

    def on_mount(self) -> None:
        view = self.query_one("#choice-list", ListView)
        values = [value for value, _ in self.options]
        view.index = values.index(self._initial) if self._initial in values else 0
        view.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self.options:
            self.dismiss(self.options[event.list_view.index or 0][0])

    def action_cancel(self) -> None:
        self.dismiss(None)
