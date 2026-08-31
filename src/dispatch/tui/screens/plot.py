"""Plotting a job's numbers, in the terminal.

Press ``p`` on any job, anywhere. Dispatch asks the daemon what that job's log contains,
the daemon asks the job's adapter, and whatever comes back is offered as a list of
quantities to put on each axis. Nothing here knows what a residual is; it knows that some
series are usually X and the rest are usually Y, and that a human should pick.

**Selection and plot on one screen, not two.** The brief this was built from described a
selector followed by a chart, and building it that way makes every comparison -- is Uy
converging like Ux? -- a round trip through a menu. With both visible, moving the cursor
redraws immediately, which turns "choose a plot" into "look through the data".

**Y is a multi-select.** Residuals are read against each other, so ``space`` overlays a
series and ``space`` again removes it. That is the one piece of state worth carrying
beyond a single chart.

The rendering itself is :mod:`dispatch.tui.plot`, which is pure arithmetic over numbers
and a size -- no widgets, no files, and above all no images.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import ListItem, ListView, Static

from dispatch.core.series import PlotData, Series, align
from dispatch.ipc.protocol import Method
from dispatch.tui.plot import (
    Charset,
    Curve,
    PlotStyle,
    hidden_by_log,
    render_plot,
    series_colour,
    suggests_log,
)
from dispatch.tui.screens.base import DispatchScreen
from dispatch.tui.theme import Palette

__all__ = ["PlotScreen"]

MAX_OVERLAY = 4
"""Y series drawn at once.

Not a technical limit. Four curves is where a terminal chart stops being readable, and a
cap that says so is kinder than one that lets the user discover it.
"""


class PlotScreen(DispatchScreen):
    """Choose an X and one or more Y series, and see them drawn."""

    TITLE = "Plot"
    nav_key = ""

    BINDINGS = [
        Binding("escape,q", "back", "back"),
        Binding("tab", "switch_axis", "x/y"),
        Binding("space", "toggle_y", "overlay"),
        Binding("l", "toggle_log", "log scale"),
        Binding("m", "toggle_marker", "marks"),
        Binding("r", "reload", "reload"),
        Binding("p", "focus_axes", "series", show=False),
    ]

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.data = PlotData()
        self.job_name = ""
        self.error: str | None = None
        self._x_key: str | None = None
        self._y_keys: list[str] = []
        self._style = PlotStyle()
        self._log_chosen = False
        """Whether the user has overridden the automatic log-scale guess."""

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical():
            yield Static("", id="heading")
            with Horizontal(id="plot-panes"):
                with Vertical(id="plot-axes"):
                    yield Static(_section("x axis"), classes="section")
                    yield ListView(id="x-list")
                    yield Static(_section("y axis"), classes="section")
                    yield ListView(id="y-list")
                yield Static("", id="plot-canvas")
            yield Static("", id="plot-status")
        yield from self.compose_footer()

    async def on_mount(self) -> None:
        await super().on_mount()
        await self.load()

    # -- data ---------------------------------------------------------------------------

    async def load(self) -> None:
        """Ask the daemon what this job's log contains."""
        try:
            payload = await self.dispatch_app.call(Method.JOB_SERIES, id=self.job_id)
        except Exception as exc:
            self.error = str(exc)
            self.data = PlotData()
            self._redraw()
            return

        self.error = None
        self.job_name = str(payload.get("name") or "")
        self.data = _decode(payload)
        self._choose_defaults()
        self._fill_lists()
        self._redraw()
        self.query_one("#y-list", ListView).focus()

    def _choose_defaults(self) -> None:
        """Pick a first plot worth looking at, so the screen opens on a chart.

        The first axis-ish series for X -- iteration, if the parser offered one -- and the
        first non-axis series for Y. For a CFD log that lands on iteration against the
        first residual, which is what somebody pressing ``p`` on a running job wanted.
        """
        axes = self.data.axes or self.data.series
        others = self.data.others or self.data.series
        self._x_key = axes[0].key if axes else None
        self._y_keys = [others[0].key] if others else []
        if others and self._x_key == others[0].key and len(others) > 1:
            self._y_keys = [others[1].key]
        self._log_chosen = False
        self._apply_auto_log()

    def _apply_auto_log(self) -> None:
        """Default the Y axis to log scale when the data plainly wants it.

        A residual history is unreadable linearly -- four decades collapse onto the bottom
        row -- and execution time is unreadable logarithmically. Guessing from the data
        gets the common case right; ``l`` settles it when the guess is wrong, and once the
        user has pressed it the guess stops applying.
        """
        if self._log_chosen:
            return
        values = [value for series in self._selected_y() for value in series.values]
        self._style = PlotStyle(
            charset=self._style.charset, log_y=suggests_log(values), log_x=self._style.log_x
        )

    # -- selection lists ------------------------------------------------------------------

    def _fill_lists(self) -> None:
        """Rebuild both selectors from the loaded series."""
        self._fill(self.query_one("#x-list", ListView), self._x_candidates(), self._x_key)
        self._fill(self.query_one("#y-list", ListView), self._y_candidates(), None)

    def _fill(self, view: ListView, series: list[Series], current: str | None) -> None:
        view.clear()
        index = 0
        for position, item in enumerate(series):
            view.append(ListItem(Static(self._entry(item))))
            if item.key == current:
                index = position
        if series:
            view.index = index

    def _entry(self, series: Series) -> Text:
        """One row of a selector: a marker, the name, and how many points it has."""
        text = Text()
        if series.key in self._y_keys:
            text.append("● ", style=series_colour(self._y_keys.index(series.key)))
        elif series.key == self._x_key:
            text.append("→ ", style=Palette.ACCENT)
        else:
            text.append("  ")
        text.append(series.display, style=Palette.TEXT)
        text.append(f"  {len(series)}", style=Palette.FAINT)
        return text

    def _x_candidates(self) -> list[Series]:
        """Every series, with the natural axes first.

        All of them, deliberately: ``execution time`` against ``iteration`` answers "is it
        slowing down", and a selector that only offered monotonic quantities on X could
        not express it.
        """
        return [*self.data.axes, *self.data.others]

    def _y_candidates(self) -> list[Series]:
        """Every series, with the measured quantities first."""
        return [*self.data.others, *self.data.axes]

    def _selected_y(self) -> list[Series]:
        found = [self.data.get(key) for key in self._y_keys]
        return [series for series in found if series is not None]

    # -- rendering ---------------------------------------------------------------------

    def _redraw(self) -> None:
        """Redraw the chart and the status line from the current selection.

        Not called ``_render``: Textual's ``Widget`` already owns that name for producing
        a renderable, and shadowing it makes a screen that silently fails to paint.
        """
        canvas = self.query_one("#plot-canvas", Static)
        status = self.query_one("#plot-status", Static)

        if self.error is not None:
            canvas.update(Text(self.error, style=Palette.ERROR))
            status.update(Text(""))
            self.update_status()
            return

        if not self.data:
            canvas.update(_nothing_to_plot(self.data))
            status.update(Text(""))
            self.update_status()
            return

        x_series = self.data.get(self._x_key or "")
        y_series = self._selected_y()
        if x_series is None or not y_series:
            canvas.update(Text("choose an x and a y series", style=Palette.MUTED))
            status.update(Text(""))
            self.update_status()
            return

        curves: list[Curve] = []
        notes: list[str] = []
        for series in y_series:
            xs, ys = align(x_series, series)
            if not xs:
                notes.append(f"{series.label} shares no samples with {x_series.label}")
                continue
            # Said out loud rather than silently truncated: two series of different
            # lengths is exactly where a plot starts lying about what it shows.
            if len(xs) < min(len(x_series), len(series)):
                notes.append(f"{series.label}: {len(xs)} of {len(series)} points align")
            # A log axis cannot draw a zero, and a solver prints them. Dropping the point
            # is right; dropping it quietly is not.
            if self._style.log_y and (hidden := hidden_by_log(ys)):
                notes.append(f"{series.label}: {hidden} non-positive point(s) not shown")
            curves.append(Curve(series.display, xs, ys))

        size = self.query_one("#plot-canvas").size
        lines = render_plot(
            curves,
            width=max(20, size.width - 1),
            height=max(6, size.height - 2),
            style=self._style,
            x_label=x_series.display,
        )

        body = Text()
        body.append(self._legend())
        body.append("\n")
        for line in lines:
            body.append(line)
            body.append("\n")
        canvas.update(body)
        status.update(self._status(notes))
        self.update_status()

    def _legend(self) -> Text:
        """Which colour is which series."""
        text = Text()
        for index, series in enumerate(self._selected_y()):
            if index:
                text.append("   ")
            text.append("━ ", style=series_colour(index))
            text.append(series.display, style=Palette.MUTED)
        return text

    def _status(self, notes: list[str]) -> Text:
        """Scale, sample count, and anything the pairing had to say."""
        text = Text()
        text.append("y ", style=Palette.FAINT)
        text.append("log" if self._style.log_y else "linear", style=Palette.MUTED)
        text.append("   marks ", style=Palette.FAINT)
        text.append(self._style.charset.value, style=Palette.MUTED)
        text.append(f"   {self.data.samples} samples", style=Palette.FAINT)
        if self.data.truncated:
            text.append("   (log truncated; showing the end of the run)", style=Palette.WARNING)
        for note in notes:
            text.append(f"   {note}", style=Palette.WARNING)
        return text

    def heading(self) -> Text:
        text = Text("plot", style=f"bold {Palette.TEXT}")
        if self.job_name:
            text.append(f"   {self.job_name}", style=Palette.MUTED)
        return text

    def on_resize(self) -> None:
        """Re-render at the new size; the chart is sized in characters."""
        if self.data:
            self._redraw()

    # -- actions --------------------------------------------------------------------------

    def action_back(self) -> None:
        self.dismiss()

    def action_focus_axes(self) -> None:
        """``p`` again returns to the series list, per the key's meaning everywhere else."""
        self.query_one("#y-list", ListView).focus()

    def action_switch_axis(self) -> None:
        lists = [self.query_one("#x-list", ListView), self.query_one("#y-list", ListView)]
        target = lists[1] if lists[0].has_focus else lists[0]
        target.focus()

    def action_toggle_log(self) -> None:
        self._log_chosen = True
        self._style = PlotStyle(
            charset=self._style.charset, log_y=not self._style.log_y, log_x=self._style.log_x
        )
        self._redraw()

    def action_toggle_marker(self) -> None:
        """Cycle the glyph family, for terminals whose font lacks braille."""
        following = (
            Charset.BLOCKS if self._style.charset is Charset.BRAILLE else Charset.BRAILLE
        )
        self._style = PlotStyle(
            charset=following, log_y=self._style.log_y, log_x=self._style.log_x
        )
        self._redraw()

    async def action_reload(self) -> None:
        """Re-read the log. The obvious thing to press while watching a running job."""
        await self.load()

    def action_toggle_y(self) -> None:
        """Add or remove the highlighted series from the overlay."""
        series = self._highlighted("#y-list", self._y_candidates())
        if series is None:
            return
        if series.key in self._y_keys:
            if len(self._y_keys) > 1:
                self._y_keys.remove(series.key)
        elif len(self._y_keys) < MAX_OVERLAY:
            self._y_keys.append(series.key)
        else:
            self.notify_error(f"At most {MAX_OVERLAY} series can be overlaid.")
            return
        self._apply_auto_log()
        self._refresh_lists()
        self._redraw()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Moving the cursor changes the plot, so the data can be browsed rather than queried."""
        if not self.data:
            return
        if event.list_view.id == "x-list":
            series = self._highlighted("#x-list", self._x_candidates())
            if series is not None and series.key != self._x_key:
                self._x_key = series.key
                self._refresh_lists()
                self._redraw()
        elif event.list_view.id == "y-list":
            series = self._highlighted("#y-list", self._y_candidates())
            # A single highlighted series replaces the selection; an overlay built with
            # `space` is left alone, or moving the cursor would dismantle it.
            if series is not None and len(self._y_keys) <= 1 and series.key not in self._y_keys:
                self._y_keys = [series.key]
                self._apply_auto_log()
                self._refresh_lists()
                self._redraw()

    def _highlighted(self, selector: str, series: list[Series]) -> Series | None:
        index = self.query_one(selector, ListView).index
        if index is None or not 0 <= index < len(series):
            return None
        return series[index]

    def _refresh_lists(self) -> None:
        """Redraw the markers without moving either cursor."""
        for selector, candidates in (
            ("#x-list", self._x_candidates()),
            ("#y-list", self._y_candidates()),
        ):
            view = self.query_one(selector, ListView)
            position = view.index
            self._fill(view, candidates, self._x_key if selector == "#x-list" else None)
            if position is not None and 0 <= position < len(candidates):
                view.index = position


def _decode(payload: dict[str, Any]) -> PlotData:
    """Rebuild plot data from the wire.

    Tolerant: a series the client cannot make sense of is skipped rather than taking the
    screen down with it. The interface renders history written by other versions.
    """
    series: list[Series] = []
    for raw in payload.get("series") or []:
        try:
            series.append(
                Series(
                    key=str(raw["key"]),
                    label=str(raw.get("label") or raw["key"]),
                    values=tuple(float(value) for value in raw["values"]),
                    samples=tuple(int(index) for index in raw["samples"]),
                    unit=raw.get("unit"),
                    axis=bool(raw.get("axis")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return PlotData(
        series=tuple(series),
        samples=int(payload.get("samples") or 0),
        truncated=bool(payload.get("truncated")),
    )


def _nothing_to_plot(data: PlotData) -> Text:
    """Explain an empty result, which is an ordinary outcome rather than a failure."""
    text = Text("nothing plottable in this job's output\n\n", style=Palette.MUTED)
    text.append(
        "Dispatch asks the job's adapter what its log contains. This one found no\n"
        "numerical series -- the job may not have produced output yet, or its solver\n"
        "may print nothing this adapter recognises.\n",
        style=Palette.FAINT,
    )
    if data.truncated:
        text.append("\nOnly the end of the log was read.\n", style=Palette.FAINT)
    return text


def _section(label: str) -> Text:
    return Text(label, style=Palette.FAINT)
