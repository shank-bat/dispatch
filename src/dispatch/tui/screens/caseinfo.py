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
        Binding("v", "render", "render mesh"),
        Binding("V", "render_animation", "animate"),
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
        """Render a mesh screenshot, after asking which angle."""
        self.app.push_screen(_PresetScreen(), self._render_mesh)

    def action_render_animation(self) -> None:
        """Render a frame per written time step."""
        self.app.push_screen(_PresetScreen(), self._render_animation)

    def _render_mesh(self, preset: str | None) -> None:
        if preset:
            self.app.call_later(self._run_render, "mesh", preset)

    def _render_animation(self, preset: str | None) -> None:
        if preset:
            self.app.call_later(self._run_render, "animation", preset)

    async def _run_render(self, kind: str, preset: str) -> None:
        """Ask the daemon to render, and say where the result went.

        Rendering an animation over a long run takes minutes, so the notification says it has
        started and the result replaces it when it arrives -- rather than the screen appearing
        to do nothing.
        """
        self.notify_ok(f"rendering the {kind}… this can take a while")
        params: dict[str, Any] = {"kind": kind, "preset": preset}
        if self.path is not None:
            params["path"] = self.path
        if self.job_id is not None:
            params["id"] = self.job_id
        try:
            result = await self.dispatch_app.call(Method.CASE_RENDER, **params)
        except Exception as exc:
            self.notify_error(str(exc))
            return
        for note in result.get("notes") or []:
            self.notify_ok(str(note))
        outputs = result.get("outputs") or []
        self.notify_ok(f"wrote {outputs[0]}" if outputs else "rendered")

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
        body.update(_render_report(self.report))
        self.update_status()


def _render_report(report: dict[str, Any]) -> Text:
    """Lay out a case report: warnings, then each section's fields."""
    text = Text()

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


class _PresetScreen(ModalScreen[str | None]):
    """Pick a camera angle.

    A list rather than a typed name: there are seven, they are the whole vocabulary, and
    nobody should have to remember which spellings are accepted.
    """

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    PRESETS = ("front", "back", "left", "right", "top", "bottom", "isometric")
    """Kept in step with :class:`dispatch.adapters.paraview.CameraPreset`, which the
    interface may not import -- the TUI does not depend on adapters (§3). A name this list
    has wrong is refused by the daemon with the valid ones, rather than silently rendering
    something else."""

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(Text("camera angle", style=f"bold {Palette.TEXT}"))
            yield ListView(
                *(ListItem(Label(Text(name))) for name in self.PRESETS),
                id="preset-list",
            )

    def on_mount(self) -> None:
        view = self.query_one("#preset-list", ListView)
        view.index = len(self.PRESETS) - 1  # isometric: the one that shows a 3-D mesh best
        view.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(self.PRESETS[event.list_view.index or 0])

    def action_cancel(self) -> None:
        self.dismiss(None)
