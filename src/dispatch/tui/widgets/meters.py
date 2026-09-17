"""Machine status, as one line.

Previously this was four labelled bars stacked in a block — a dashboard widget. It is now
a status bar: cores, CPU, memory, load, and — on a machine that has any — GPUs, on a
single row, each a compact figure with a short sparkline-style gauge beside it.

The reason is proportion. The machine's state is context for the job list, not the subject
of the screen, and four rows of chrome above the content said otherwise.

One thing is kept from the old design and is not negotiable: **two different CPU numbers
are shown.** ``allocated`` is what Dispatch has promised out of its ledger and is what
admission decisions use; ``cpu`` is what the cores are actually doing. They disagree
whenever a solver blocks on I/O, and collapsing them into one tidy number would hide the
single most confusing thing about the scheduler.

The GPU figure has no measured counterpart at all. It is the ledger, and only the ledger:
a card's utilisation reads near zero between training steps, so a measurement here would
say "idle" about a GPU that is fully committed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.console import RenderableType
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from dispatch.tui.theme import Palette, usage_style

__all__ = ["HeaderStats", "ResourceMeters"]

GAUGE_WIDTH = 8
"""Cells per gauge. Enough to read at a glance, small enough that four fit on one line."""

GAUGE_FULL = "━"
GAUGE_EMPTY = "─"
"""A heavy rule over a light one.

Block characters (``█░``) render as a chunky bar chart, which is the 1990s look this
design is moving away from. Two weights of the same rule read as a measure.
"""

SEPARATOR = "   "


class ResourceMeters(Static):
    """A single-line summary of the machine."""

    snapshot: reactive[dict[str, Any]] = reactive(dict, always_update=True)

    def render(self) -> RenderableType:
        data = self.snapshot
        if not data:
            return Text("connecting…", style=Palette.FAINT)

        total = int(data.get("total_cores", 0)) or 1
        allocated = int(data.get("allocated_cores", 0))
        free = int(data.get("free_cores", 0))
        cpu = float(data.get("cpu_percent", 0.0))
        used_ram = int(data.get("used_ram_mb", 0))
        total_ram = int(data.get("total_ram_mb", 0)) or 1

        text = Text()
        _field(
            text,
            "cores",
            f"{allocated}/{total}",
            allocated / total,
            note=f"{free} free",
        )
        text.append(SEPARATOR)
        _field(text, "cpu", f"{cpu:.0f}%", cpu / 100)
        text.append(SEPARATOR)
        _field(
            text,
            "mem",
            f"{used_ram / 1024:.1f}/{total_ram / 1024:.0f}G",
            used_ram / total_ram,
        )

        # Shown only on a machine that has one. A permanent "gpu 0/0" on the great
        # majority of workstations would be a column of noise reporting the absence of a
        # feature, which is the opposite of what a status bar is for.
        total_gpus = int(data.get("total_gpus", 0))
        if total_gpus:
            allocated_gpus = int(data.get("allocated_gpus", 0))
            text.append(SEPARATOR)
            _field(
                text,
                "gpu",
                f"{allocated_gpus}/{total_gpus}",
                allocated_gpus / total_gpus,
                note=f"{int(data.get('free_gpus', 0))} free",
            )

        load = data.get("load_average") or [0.0]
        text.append(SEPARATOR)
        text.append("load ", style=Palette.FAINT)
        text.append(f"{float(load[0]):.2f}", style=Palette.MUTED)
        return text


def _field(
    text: Text, label: str, value: str, fraction: float, *, note: str = ""
) -> None:
    """Append ``label gauge value`` to the line.

    The label is faint, the gauge carries the colour, the value is plain. Reading order is
    therefore gauge first (is anything wrong?) then value (how wrong?), which is the order
    someone glancing at a status bar actually wants.
    """
    text.append(f"{label} ", style=Palette.FAINT)
    text.append(_gauge(fraction))
    text.append(" ")
    text.append(value, style=Palette.TEXT)
    if note:
        text.append(f" ({note})", style=Palette.FAINT)


def _gauge(fraction: float, *, width: int = GAUGE_WIDTH) -> Text:
    """A short two-weight rule showing a fraction.

    Colour comes from :func:`~dispatch.tui.theme.usage_style`, which leaves anything under
    70% grey. A machine running comfortably should not light up.
    """
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    style = usage_style(fraction * 100)

    gauge = Text()
    gauge.append(GAUGE_FULL * filled, style=style)
    gauge.append(GAUGE_EMPTY * (width - filled), style=Palette.BORDER)
    return gauge


class HeaderStats(Static):
    """The machine's numbers, as a grid of sparklines beside the logo, filling the row.

    Six cells, two columns by three rows -- core / cpu, mem / gpu, load / core hrs -- laid
    out through one shared function so every cell is the same shape and every column
    starts at the same character position no matter which row it is in. The earlier
    version built its third row by hand, appending text ad hoc instead of going through
    that function, and the two columns of numbers it produced did not actually line up
    with the two above them -- which is the whole reason this went through one function
    for all six instead of "most of them".

    Width is claimed with ``width: 1fr`` in the stylesheet, not a fixed guess: the block
    fills whatever is left of the header row beside the logo, and the sparklines are
    stretched to fill it in turn (:meth:`_column_width`), so a wide terminal gets a
    genuinely bigger, more detailed trend line rather than a fixed-size grid stranded in
    the middle of empty space.

    The sparkline is the last :data:`~dispatch.tui.state.HISTORY_LENGTH` readings
    :meth:`~dispatch.tui.state.AppState.push_snapshot` has recorded -- about the last
    minute, at the daemon's own sampling rate -- rather than the single instant reading a
    plain gauge would show. A machine climbing towards saturation and one that spiked and
    came back down both read as "70%" on a bar; the shape of recent history is what tells
    them apart, which is the reason to plot one here instead of a bar.

    All six cells are always in the same positions, GPU included even on a machine with
    none -- unlike :class:`ResourceMeters`' single line, which drops the GPU segment
    entirely there. A dropped segment just shortens a line; a dropped *cell* would leave a
    hole in a shape the eye has learned is a fixed grid, which reads as broken rather than
    absent. Load and the core-hour total get a cell the same shape as the rest, flat and
    muted rather than plotted -- load is already itself a running average and core-hours
    only ever climbs, so there is no trend in either worth the same ink -- for the same
    reason: a grid where two cells are a different shape looks like a mistake.
    """

    snapshot: reactive[dict[str, Any]] = reactive(dict, always_update=True)
    core_hours: reactive[float] = reactive(0.0)
    history: reactive[dict[str, Sequence[float]]] = reactive(dict, always_update=True)

    def render(self) -> RenderableType:
        data = self.snapshot
        if not data:
            return Text("connecting…", style=Palette.FAINT)

        total = int(data.get("total_cores", 0)) or 1
        allocated = int(data.get("allocated_cores", 0))
        cpu = float(data.get("cpu_percent", 0.0))
        used_ram = int(data.get("used_ram_mb", 0))
        total_ram = int(data.get("total_ram_mb", 0)) or 1
        load = data.get("load_average") or [0.0]
        total_gpus = int(data.get("total_gpus", 0))

        if total_gpus:
            allocated_gpus = int(data.get("allocated_gpus", 0))
            gpu_history: Sequence[float] = self.history.get("gpu", ())
            gpu_value = f"{allocated_gpus}/{total_gpus}"
        else:
            # No GPU on this machine: there is nothing to plot, so the cell carries no
            # sparkline rather than one honestly reading zero forever.
            gpu_history, gpu_value = (), "-"

        spark_width = self._spark_width()

        # Clip rather than wrap. The block has a fixed height to match the logo beside it,
        # and a cell Rich decided to wrap lands its overflow on the *next* line of the same
        # block instead of a line of its own -- which does not raise anything, it just
        # quietly slides every row below it down by one. An ellipsis on an implausibly
        # large number is a far more honest failure than that.
        text = Text(no_wrap=True, overflow="crop")
        rows = (
            (
                ("core", self.history.get("cores", ()), f"{allocated}/{total}"),
                ("cpu", self.history.get("cpu", ()), f"{cpu:.0f}%"),
            ),
            (
                (
                    "mem",
                    self.history.get("mem", ()),
                    f"{used_ram / 1024:.1f}/{total_ram / 1024:.0f}G",
                ),
                ("gpu", gpu_history, gpu_value),
            ),
            (
                ("load", (), f"{float(load[0]):.2f}"),
                ("core hrs", (), f"{self.core_hours:.1f}"),
            ),
        )
        for row_index, (left, right) in enumerate(rows):
            if row_index:
                text.append("\n")
            _cell(text, *left, spark_width=spark_width)
            text.append(GRID_GAP)
            _cell(text, *right, spark_width=spark_width)
        return text

    def _spark_width(self) -> int:
        """How wide each sparkline should be, given the width Textual has actually given
        this widget this frame.

        ``self.size.width`` is only meaningful once the widget has been through layout, so
        a still-unmeasured ``0`` falls back to the width the grid used before it could
        claim the row's leftover space at all -- narrow, not wrong.
        """
        available = self.size.width or GRID_MIN_WIDTH
        per_column = (available - GRID_GAP_WIDTH) // 2
        spark = per_column - GRID_LABEL_WIDTH - 1 - GRID_VALUE_WIDTH - 1
        return max(GRID_MIN_SPARK, min(GRID_MAX_SPARK, spark))



GRID_LABEL_WIDTH = 8
"""Width of a cell's label. Fixed at the widest one needs -- "core hrs" -- so every label
column, in both grid columns, starts and ends at the same place."""

GRID_VALUE_WIDTH = 9
"""Width of a cell's value. Fixed, unlike the sparkline: a number that grew with the
terminal would not get more readable, only more randomly positioned."""

GRID_GAP = "   "
GRID_GAP_WIDTH = len(GRID_GAP)
"""Blank columns between the grid's two columns."""

GRID_MIN_WIDTH = 56
"""Fallback total width, for the one frame before Textual has measured this widget.

The width ``#header-stats`` claimed before it could ask the stylesheet for the row's
leftover space instead (``width: 1fr``) -- narrow rather than wrong, and only ever used
for the render that happens before layout has run once.
"""

GRID_MIN_SPARK = 6
"""Floor on sparkline width. Below this the shape of a trend stops being legible at all,
so a genuinely narrow terminal gets a short-but-real sparkline rather than nothing."""

GRID_MAX_SPARK = 40
"""Ceiling on sparkline width, so an ultrawide terminal gets a bigger, more legible trend
line rather than one line stretched to an ungainly and no more informative length."""

SPARK_LEVELS = "▁▂▃▄▅▆▇█"
"""Eighth-step block heights, low to high. Widely supported -- these eight are also what
``btop``, ``htop`` and friends draw their own graphs in."""


def _cell(text: Text, label: str, values: Sequence[float], value: str, *, spark_width: int) -> None:
    """Append one ``label sparkline value`` cell, padded to a fixed width.

    ``values`` are recent percentage readings, oldest first, as :attr:`AppState.history`
    stores them; an empty sequence -- a machine with no GPU, load, or the core-hour total,
    none of which have a trend worth plotting -- draws a flat, muted baseline rather than
    an empty gap, so every cell keeps the same shape whether or not it has anything to
    plot. ``spark_width`` is decided once per render, by :meth:`HeaderStats._spark_width`,
    and passed to every cell in that render so all six stay the same width as each other.
    """
    text.append(f"{label:<{GRID_LABEL_WIDTH}}", style=Palette.FAINT)
    text.append(" ")
    text.append(_sparkline(values, width=spark_width))
    text.append(" ")
    style = usage_style(values[-1]) if values else Palette.MUTED
    text.append(f"{value:<{GRID_VALUE_WIDTH}}", style=style)


def _sparkline(values: Sequence[float], *, width: int) -> Text:
    """The last ``width`` readings as a block-height trend line, coloured by the latest.

    Padded on the left with the oldest available reading -- rather than with blanks --
    when there is not yet a full window of history, so a sparkline started thirty seconds
    ago reads as "flat so far" instead of half-erased.
    """
    if not values:
        return Text(SPARK_LEVELS[0] * width, style=Palette.BORDER)

    tail = list(values)[-width:]
    if len(tail) < width:
        tail = [tail[0]] * (width - len(tail)) + tail

    style = usage_style(values[-1])
    chars = []
    for reading in tail:
        clamped = max(0.0, min(100.0, reading))
        index = round(clamped / 100 * (len(SPARK_LEVELS) - 1))
        chars.append(SPARK_LEVELS[index])
    return Text("".join(chars), style=style)
