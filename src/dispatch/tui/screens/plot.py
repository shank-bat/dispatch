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

**Three ways to read the chart more closely**, each a typed value because each is a number
the user already has in mind:

* ``z`` moves the dotted **reference line**, at y = 0 until moved, or turns it off. A lift
  coefficient is read against zero and a residual against its tolerance.
* ``w`` shows only the **last N** of the x axis. The start of a run is routinely orders of
  magnitude from where it settles, and the y scale fits whatever is drawn, so this is what
  makes the converged part legible.
* ``f`` **finds the value at an x** -- the lift at iteration 500 -- for every plotted series,
  interpolated between samples when none sits exactly there, and marks it on the chart. It
  is kept across reloads, so it reads off a running job as it advances.

The rendering itself is :mod:`dispatch.tui.plot`, which is pure arithmetic over numbers
and a size -- no widgets, no files, and above all no images.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, ListItem, ListView, Static

from dispatch.core.series import Dataset, PlotData, Series, align
from dispatch.ipc.protocol import Method
from dispatch.tui.plot import (
    Charset,
    Curve,
    Lookup,
    PlotStyle,
    hidden_by_log,
    last_window,
    render_plot,
    series_colour,
    suggests_log,
    value_at,
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
        Binding("d", "next_dataset", "dataset"),
        Binding("z", "reference", "ref line"),
        Binding("w", "window", "last N"),
        Binding("f", "find", "value at x"),
        Binding("p", "focus_axes", "series", show=False),
    ]

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.datasets: list[Dataset] = []
        """Every source of numbers this job has: its log, and the files its case writes.

        Separate rather than merged because their rows do not correspond -- a log records
        one per time step, a function object one per write -- so pairing across them by
        position would plot one quantity against a different moment of another.
        """

        self._dataset = 0
        self.job_name = ""
        self.error: str | None = None
        self._x_key: str | None = None
        self._y_keys: list[str] = []
        self._style = PlotStyle()
        self._log_chosen = False
        """Whether the user has overridden the automatic log-scale guess."""
        self.reference: float | None = 0.0
        """Where the dotted reference line is drawn, or ``None`` for no line."""
        self.window: float | None = None
        """Show only this much of the end of the x axis; ``None`` shows everything."""
        self.lookup_x: float | None = None
        """The x whose values are read out under the legend, if one was asked for."""
        self._prompt_mode = ""

    @property
    def data(self) -> PlotData:
        """The dataset currently being plotted. Empty when the job has none."""
        if not self.datasets:
            return PlotData()
        return self.datasets[self._dataset % len(self.datasets)].data

    @data.setter
    def data(self, value: PlotData) -> None:
        """Replace everything with a single unnamed dataset.

        A convenience for "just plot this", used by tests and by any caller that has numbers
        rather than a source. Assigning an empty :class:`PlotData` clears the screen, which
        is what "this job has nothing to plot" looks like.
        """
        self.datasets = (
            [Dataset(key="log", label="Solver log", data=value)] if value else []
        )
        self._dataset = 0

    @property
    def dataset(self) -> Dataset | None:
        """The current dataset itself, for its label and source."""
        if not self.datasets:
            return None
        return self.datasets[self._dataset % len(self.datasets)]

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
            yield Input(id="plot-prompt", classes="prompt")
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
            self.datasets = []
            self._redraw()
            return

        self.error = None
        self.job_name = str(payload.get("name") or "")
        self.datasets = _decode_datasets(payload)
        self._dataset = 0
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
        # From what will be drawn: the transient that `w` hides should not choose the scale
        # of the part that is left.
        values = [value for _, _, ys in self._pairs() for value in ys]
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

    def _pairs(
        self, *, windowed: bool = True
    ) -> list[tuple[Series, tuple[float, ...], tuple[float, ...]]]:
        """Each selected y series paired with x, cut to the window unless asked not to be."""
        x_series = self.data.get(self._x_key or "")
        if x_series is None:
            return []
        pairs = []
        for series in self._selected_y():
            xs, ys = align(x_series, series)
            if windowed and self.window is not None:
                xs, ys = last_window(xs, ys, self.window)
            pairs.append((series, xs, ys))
        return pairs

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
        full = {series.key: (xs, ys) for series, xs, ys in self._pairs(windowed=False)}
        for series, xs, ys in self._pairs():
            aligned = len(full[series.key][0])
            if not aligned:
                notes.append(f"{series.label} shares no samples with {x_series.label}")
                continue
            if not xs:
                notes.append(f"{series.label} has no samples in the window")
                continue
            # Said out loud rather than silently truncated: two series of different
            # lengths is exactly where a plot starts lying about what it shows.
            if aligned < min(len(x_series), len(series)):
                notes.append(f"{series.label}: {aligned} of {len(series)} points align")
            # A log axis cannot draw a zero, and a solver prints them. Dropping the point
            # is right; dropping it quietly is not.
            if self._style.log_y and (hidden := hidden_by_log(ys)):
                notes.append(f"{series.label}: {hidden} non-positive point(s) not shown")
            curves.append(Curve(series.display, xs, ys))

        readout = self._readout(x_series, full)
        size = self.query_one("#plot-canvas").size
        lines = render_plot(
            curves,
            width=max(20, size.width - 1),
            height=max(6, size.height - 2 - (1 if readout else 0)),
            style=self._style,
            x_label=x_series.display,
            reference=self.reference,
            marker_x=self.lookup_x,
        )

        body = Text()
        body.append(self._legend())
        body.append("\n")
        if readout:
            body.append(readout)
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

    def _readout(
        self, x_series: Series, full: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]
    ) -> Text | None:
        """Every plotted series' value at the looked-up x, in its own colour.

        From the whole run, not the window: the question is what the value *was* at that x,
        and hiding the start of the plot does not change the answer.
        """
        if self.lookup_x is None:
            return None
        text = Text(f"at {x_series.display} = {_number(self.lookup_x)}:", style=Palette.MUTED)
        for index, series in enumerate(self._selected_y()):
            xs, ys = full.get(series.key, ((), ()))
            found: Lookup | None = value_at(xs, ys, self.lookup_x)
            text.append("   ")
            text.append(f"{series.display} ", style=series_colour(index))
            if found is None:
                span = f" ({_number(min(xs))} to {_number(max(xs))})" if xs else ""
                text.append(f"no data there{span}", style=Palette.WARNING)
            else:
                # Approximate is said, not hidden: between two writes the value is a line
                # drawn between them, not something the solver reported.
                text.append(
                    f"{'' if found.exact else '≈ '}{_number(found.value)}", style=Palette.TEXT
                )
        return text

    def _status(self, notes: list[str]) -> Text:
        """Scale, sample count, and anything the pairing had to say."""
        text = Text()
        current = self.dataset
        if current is not None:
            text.append(current.label, style=Palette.ACCENT)
            if len(self.datasets) > 1:
                text.append(
                    f" ({self._dataset + 1}/{len(self.datasets)}, d to change)",
                    style=Palette.FAINT,
                )
            text.append("   ")
        text.append("y ", style=Palette.FAINT)
        text.append("log" if self._style.log_y else "linear", style=Palette.MUTED)
        text.append("   marks ", style=Palette.FAINT)
        text.append(self._style.charset.value, style=Palette.MUTED)
        if self.window is not None:
            text.append("   last ", style=Palette.FAINT)
            text.append(_number(self.window), style=Palette.ACCENT)
        text.append("   ref ", style=Palette.FAINT)
        if self.reference is None:
            text.append("off", style=Palette.MUTED)
        else:
            text.append(f"y = {_number(self.reference)}", style=Palette.MUTED)
            if not self._reference_drawn():
                text.append(" (off the scale)", style=Palette.FAINT)
        text.append(f"   {self.data.samples} samples", style=Palette.FAINT)
        if self.data.truncated:
            text.append("   (log truncated; showing the end of the run)", style=Palette.WARNING)
        for note in notes:
            text.append(f"   {note}", style=Palette.WARNING)
        return text

    def _reference_drawn(self) -> bool:
        """Whether the reference line falls inside the y range being drawn."""
        if self.reference is None:
            return False
        values = [value for _, _, ys in self._pairs() for value in ys]
        if self._style.log_y:
            if self.reference <= 0:
                return False
            values = [value for value in values if value > 0]
        return bool(values) and min(values) <= self.reference <= max(values)

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
        # Escape while typing a value closes the prompt rather than the whole screen.
        box = self.query_one("#plot-prompt", Input)
        if box.has_class("visible"):
            self._close_prompt()
            return
        self.dismiss()

    def action_reference(self) -> None:
        """Move the dotted reference line, or turn it off."""
        now = "off" if self.reference is None else _number(self.reference)
        self._open_prompt("reference", f"reference line at y = ?  (now {now}; 'off' hides it)")

    def action_window(self) -> None:
        """Show only the end of the run, so its start stops deciding the scale."""
        x_series = self.data.get(self._x_key or "")
        unit = x_series.display if x_series else "the x axis"
        self._open_prompt("window", f"show only the last how much of {unit}?  (empty shows all)")

    def action_find(self) -> None:
        """Read every plotted series' value at a chosen x."""
        x_series = self.data.get(self._x_key or "")
        unit = x_series.display if x_series else "x"
        self._open_prompt("lookup", f"value at {unit} = ?  (empty clears)")

    def _open_prompt(self, mode: str, placeholder: str) -> None:
        self._prompt_mode = mode
        box = self.query_one("#plot-prompt", Input)
        box.placeholder = placeholder
        box.value = ""
        box.add_class("visible")
        box.focus()

    def _close_prompt(self) -> None:
        self.query_one("#plot-prompt", Input).remove_class("visible")
        self._prompt_mode = ""
        self.query_one("#y-list", ListView).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        mode = self._prompt_mode
        raw = event.value.strip()
        self._close_prompt()
        event.stop()

        if mode == "reference" and raw.lower() in ("off", "none", "-"):
            self.reference = None
        elif not raw:
            # Empty means "back to normal" for the two that have one; for the reference line,
            # which defaults to zero, it leaves the line where it was.
            if mode == "window":
                self.window = None
            elif mode == "lookup":
                self.lookup_x = None
            else:
                return
        else:
            try:
                value = float(raw)
            except ValueError:
                self.notify_error(f"{raw!r} is not a number")
                return
            if mode == "reference":
                self.reference = value
            elif mode == "window":
                if value <= 0:
                    self.notify_error("The window must be larger than zero.")
                    return
                self.window = value
            elif mode == "lookup":
                self.lookup_x = value
            else:
                return
        self._apply_auto_log()
        self._redraw()

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

    def action_next_dataset(self) -> None:
        """Move to the next source of numbers -- the log, the force coefficients, ...

        A key rather than a third list: most jobs have one dataset and a selector that is
        usually a single row would be chrome. The status line names the current one, and the
        key is only interesting on a case that has more than one.
        """
        if len(self.datasets) < 2:
            self.notify_error("This job has only one set of data to plot.")
            return
        self._dataset = (self._dataset + 1) % len(self.datasets)
        self._choose_defaults()
        self._fill_lists()
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


def _decode_datasets(payload: dict[str, Any]) -> list[Dataset]:
    """Rebuild the dataset list, falling back to the flat payload.

    The fallback is what makes the field additive: a daemon from before datasets existed
    answers with ``series`` alone, and that is read as the one log dataset rather than as an
    empty screen.
    """
    raw = payload.get("datasets")
    if not raw:
        data = _decode(payload)
        if not data:
            return []
        return [Dataset(key="log", label="Solver log", data=data, source=payload.get("path"))]

    datasets: list[Dataset] = []
    for item in raw:
        data = _decode(item)
        if not data:
            continue
        datasets.append(
            Dataset(
                key=str(item.get("key") or "data"),
                label=str(item.get("label") or item.get("key") or "data"),
                data=data,
                source=item.get("source"),
            )
        )
    return datasets


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


def _number(value: float) -> str:
    """A typed or read-off value, in full enough precision to be worth having asked for.

    Not the axis's compact format: the axis is read for its scale, but a value the user
    asked for is read for its digits, and ``0.4`` for a lift of ``0.41237`` would be wrong.
    """
    return f"{value:.6g}"
