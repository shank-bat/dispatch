"""Drawing a line chart with characters.

No image is produced, no file is written, and no plotting library is imported. The output
is a list of :class:`rich.text.Text` lines, which is the only form that works over the
connection this is actually used on: an SSH session to a headless machine, where the
answer to "how are the residuals looking" has to arrive as text or not at all.

**Resolution comes from Unicode, not from the terminal.** A braille cell carries a 2x4
grid of independently addressable dots, so a plot 60 columns wide has 120 horizontal
samples rather than 60 -- enough for a residual curve to look like a curve. Quadrant
blocks give 2x2 and are offered as a fallback, because braille depends on the terminal
font having the block and a few do not; ``m`` cycles between them, so a user on such a
terminal fixes the display in one keystroke rather than filing a bug.

**Log scale is chosen, not imposed.** Residuals span six decades and are unreadable
linearly; execution time spans one and is unreadable logarithmically. So a log axis is
offered automatically when the data is strictly positive and spans more than two decades,
and ``l`` overrides the guess either way.

Everything here is a pure function of numbers and a size. That is deliberate: the hard
part of a terminal chart is the arithmetic at the edges -- a series of constant value, a
single point, a range of zero, a NaN that got this far -- and all of it is testable
without a terminal, a widget, or a running application.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Final

from rich.text import Text

from dispatch.tui.theme import Palette

__all__ = ["Charset", "PlotStyle", "hidden_by_log", "render_plot", "series_colour"]

MIN_WIDTH: Final = 20
MIN_HEIGHT: Final = 5
"""Below this there is no chart worth drawing, and the caller is told to widen instead."""

LOG_DECADES: Final = 2.0
"""Dynamic range, in decades, above which a log axis is offered by default."""

MOSTLY_POSITIVE: Final = 0.9
"""Fraction of values that must be positive before a log axis is offered.

Nine in ten rather than all of them: solvers print an exact zero for a field that has not
moved yet, and one such point should not decide the scale of the other ninety-nine. Below
this fraction the negatives are the data -- continuity errors, say -- and a log axis would
hide most of the series."""

GUTTER: Final = 2
"""Spaces between the y tick labels and the axis."""


class Charset(StrEnum):
    """How a data point becomes part of a character."""

    BRAILLE = "braille"
    """2x4 dots per cell. The default: the highest resolution available in text."""

    BLOCKS = "blocks"
    """2x2 quadrant blocks. Coarser, and drawn from a much older part of Unicode."""


_BRAILLE_BASE: Final = 0x2800
_BRAILLE_BITS: Final = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)
"""Dot bit per (column, row) within a braille cell.

The order is not sequential: braille numbers its dots 1-6 down two columns and appends 7
and 8 at the bottom, so the fourth row's bits are 0x40 and 0x80 rather than what a reader
would guess. Written out as a table because deriving it inline is how this gets subtly
wrong.
"""

_BLOCK_BITS: Final = ((0x01, 0x04), (0x02, 0x08))
_BLOCK_GLYPHS: Final = (
    " ", "▘", "▖", "▌", "▝", "▀", "▞", "▛",
    "▗", "▚", "▄", "▙", "▐", "▜", "▟", "█",
)  # fmt: skip
"""Quadrant glyphs indexed by the bitmask of filled quarters."""

SERIES_COLOURS: Final[tuple[str, ...]] = (
    Palette.ACCENT,
    Palette.WARNING,
    Palette.SUCCESS,
    Palette.ERROR,
    Palette.ACCENT_TEXT,
    Palette.MUTED,
)
"""Colours for overlaid series, in order.

Six, and they repeat after that: more than a handful of curves on one terminal chart is
unreadable however they are coloured, and the selector limits the overlay anyway.
"""


def series_colour(index: int) -> str:
    """The colour for the nth overlaid series."""
    return SERIES_COLOURS[index % len(SERIES_COLOURS)]


@dataclass(frozen=True, slots=True)
class PlotStyle:
    """How to draw. Everything the keyboard can change lives here.

    Attributes:
        charset: Which glyph family to draw points with.
        log_y: Logarithmic Y axis.
        log_x: Logarithmic X axis.
    """

    charset: Charset = Charset.BRAILLE
    log_y: bool = False
    log_x: bool = False

    @property
    def cell(self) -> tuple[int, int]:
        """Sub-cell resolution as ``(columns, rows)``."""
        return (2, 4) if self.charset is Charset.BRAILLE else (2, 2)


@dataclass(frozen=True, slots=True)
class Curve:
    """One series to draw, already paired and ready.

    Attributes:
        label: What to call it in the legend.
        xs: X values.
        ys: Y values, the same length as ``xs``.
    """

    label: str
    xs: Sequence[float]
    ys: Sequence[float]


def suggests_log(values: Sequence[float]) -> bool:
    """Whether these values are better read on a logarithmic axis.

    The positive values must span more than :data:`LOG_DECADES` decades, and almost all of
    the values must be positive. A residual history satisfies both and is meaningless
    without a log axis; a Courant number satisfies neither and would be made worse by one.

    **A handful of zeros does not disqualify a series**, and that tolerance is not a
    nicety. A real ``icoFoam`` cavity log prints ``Initial residual = 0`` for ``Uy`` on its
    first step, because the field starts at zero -- one value in a hundred. Requiring every
    value to be positive meant that single point turned off the log axis for a plot whose
    entire reason for existing is six decades of decay. Non-positive points are dropped
    from a log plot and their number is reported (§9.6), which is the honest handling; they
    are not a reason to render the plot uselessly.
    """
    positive = [value for value in values if value > 0]
    if not positive or len(positive) < len(values) * MOSTLY_POSITIVE:
        return False
    low, high = min(positive), max(positive)
    return bool(high / low > 10.0**LOG_DECADES)


def hidden_by_log(values: Sequence[float]) -> int:
    """How many values a logarithmic axis cannot show. Reported, never silently dropped."""
    return sum(1 for value in values if value <= 0)


class _Canvas:
    """A grid of sub-cell dots that renders to characters.

    Owns nothing but a bytearray of bitmasks and a parallel array of which curve last
    touched each cell -- which is how a cell gets a colour when two curves cross inside
    it. Last writer wins, so the curve drawn most recently is the one that shows; with
    the legend beside it that reads correctly, and the alternative (blending, or
    splitting the cell) is not available in a terminal.
    """

    __slots__ = ("_bits", "_cols", "_owner", "_rows", "_style", "_sub")

    def __init__(self, cols: int, rows: int, style: PlotStyle) -> None:
        self._cols = cols
        self._rows = rows
        self._style = style
        self._sub = style.cell
        self._bits = bytearray(cols * rows)
        self._owner = [-1] * (cols * rows)

    def set(self, x: int, y: int, curve: int) -> None:
        """Light the dot at sub-cell coordinates ``(x, y)``, origin top-left."""
        sub_cols, sub_rows = self._sub
        col, row = divmod(x, sub_cols)[0], divmod(y, sub_rows)[0]
        if not (0 <= col < self._cols and 0 <= row < self._rows):
            return
        table = _BRAILLE_BITS if self._style.charset is Charset.BRAILLE else _BLOCK_BITS
        index = row * self._cols + col
        self._bits[index] |= table[x % sub_cols][y % sub_rows]
        self._owner[index] = curve

    def line(self, x0: int, y0: int, x1: int, y1: int, curve: int) -> None:
        """Draw a straight segment between two sub-cell points.

        Integer Bresenham. Segments rather than dots because a downsampled series has far
        fewer points than the canvas has columns, and a scatter of unconnected dots does
        not read as a curve -- which is the entire thing being looked at.
        """
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            self.set(x0, y0, curve)
            if x0 == x1 and y0 == y1:
                return
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += step_x
            if doubled <= dx:
                error += dx
                y0 += step_y

    def row(self, row: int) -> Text:
        """One rendered line of the canvas."""
        text = Text()
        for col in range(self._cols):
            index = row * self._cols + col
            mask = self._bits[index]
            if not mask:
                text.append(" ")
                continue
            glyph = (
                chr(_BRAILLE_BASE + mask)
                if self._style.charset is Charset.BRAILLE
                else _BLOCK_GLYPHS[mask]
            )
            text.append(glyph, style=series_colour(max(0, self._owner[index])))
        return text


def render_plot(
    curves: Sequence[Curve],
    *,
    width: int,
    height: int,
    style: PlotStyle | None = None,
    x_label: str = "",
    y_label: str = "",
) -> list[Text]:
    """Render one or more curves as terminal lines.

    Args:
        curves: What to draw. Overlaid on shared axes, each in its own colour.
        width: Total columns available, including the y tick gutter.
        height: Total rows available, including the x axis and its labels.
        style: Charset and axis scaling. Defaults to braille, linear.
        x_label: Axis caption, shown under the x tick labels.
        y_label: Axis caption, shown above the chart.

    Returns:
        Lines ready to write into a widget. A message rather than a chart when there is
        nothing to draw or no room to draw it -- both are ordinary situations and both
        deserve a sentence rather than a blank rectangle.
    """
    style = style or PlotStyle()
    drawable = [curve for curve in curves if curve.xs and curve.ys]
    if not drawable:
        return [Text("no data points in the selected series", style=Palette.MUTED)]
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return [Text("the terminal is too small for a plot", style=Palette.MUTED)]

    x_values = [value for curve in drawable for value in curve.xs]
    y_values = [value for curve in drawable for value in curve.ys]
    x_range = _range(x_values, log=style.log_x)
    y_range = _range(y_values, log=style.log_y)
    if x_range is None or y_range is None:
        return [
            Text(
                "these values cannot be shown on a logarithmic axis (they are not all "
                "positive) — press l for a linear axis",
                style=Palette.WARNING,
            )
        ]

    ticks = _y_ticks(y_range, rows=height - 2, log=style.log_y)
    gutter = max((len(label) for label, _ in ticks), default=0) + GUTTER
    plot_cols = width - gutter - 1
    plot_rows = height - 2  # one row for the x axis, one for its labels
    if plot_cols < MIN_WIDTH // 2 or plot_rows < 1:
        return [Text("the terminal is too small for a plot", style=Palette.MUTED)]

    canvas = _Canvas(plot_cols, plot_rows, style)
    sub_cols, sub_rows = style.cell
    for index, curve in enumerate(drawable):
        _draw(
            canvas,
            curve,
            index,
            x_range,
            y_range,
            plot_cols * sub_cols,
            plot_rows * sub_rows,
            style,
        )

    lines: list[Text] = []
    if y_label:
        lines.append(Text(y_label, style=Palette.TEXT))

    tick_rows = {row: label for label, row in ticks}
    for row in range(plot_rows):
        line = Text()
        label = tick_rows.get(row)
        if label is not None:
            line.append(label.rjust(gutter - GUTTER), style=Palette.FAINT)
            line.append(" " * (GUTTER - 1))
            line.append("┤", style=Palette.BORDER)
        else:
            line.append(" " * (gutter - 1))
            line.append("│", style=Palette.BORDER)
        line.append(canvas.row(row))
        lines.append(line)

    lines.append(_axis(gutter, plot_cols))
    lines.append(_x_labels(x_range, gutter, plot_cols, log=style.log_x, caption=x_label))
    return lines


def _draw(
    canvas: _Canvas,
    curve: Curve,
    index: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    sub_width: int,
    sub_height: int,
    style: PlotStyle,
) -> None:
    """Project one curve into canvas coordinates and stroke it."""
    points: list[tuple[int, int]] = []
    for raw_x, raw_y in zip(curve.xs, curve.ys, strict=False):
        x = _project(raw_x, x_range, sub_width - 1, log=style.log_x)
        y = _project(raw_y, y_range, sub_height - 1, log=style.log_y)
        if x is None or y is None:
            continue
        points.append((x, sub_height - 1 - y))

    if len(points) == 1:
        canvas.set(points[0][0], points[0][1], index)
        return
    for (x0, y0), (x1, y1) in pairwise(points):
        canvas.line(x0, y0, x1, y1, index)


def _project(
    value: float, span: tuple[float, float], extent: int, *, log: bool
) -> int | None:
    """Map a value onto ``0..extent``, or ``None`` if it cannot be shown."""
    low, high = span
    if log:
        if value <= 0:
            return None
        value, low, high = math.log10(value), math.log10(low), math.log10(high)
    if high <= low:
        # A constant series still deserves a line, drawn down the middle rather than
        # collapsed onto an edge where it reads as a boundary.
        return extent // 2
    return max(0, min(extent, round((value - low) / (high - low) * extent)))


def _range(values: Sequence[float], *, log: bool) -> tuple[float, float] | None:
    """The span to draw, or ``None`` when a log axis was asked for and is impossible."""
    if log:
        positive = [value for value in values if value > 0]
        if not positive:
            return None
        values = positive
    low, high = min(values), max(values)
    if low == high:
        # Widen a flat series so it has somewhere to sit. Multiplicatively on a log axis,
        # additively otherwise, so both produce a sensible pair of ticks.
        return (low / 10, high * 10) if log and low > 0 else (low - 1.0, high + 1.0)
    return (low, high)


def _y_ticks(span: tuple[float, float], *, rows: int, log: bool) -> list[tuple[str, int]]:
    """Labels for the y axis, paired with the canvas row each sits on.

    At most five, and never more than there are rows: a tick on every row of a 20-row
    chart is a wall of numbers, and the axis is read for its scale rather than for exact
    values.
    """
    if rows < 1:
        return []
    count = max(2, min(5, rows))
    low, high = span
    ticks: list[tuple[str, int]] = []
    for step in range(count):
        fraction = step / (count - 1)
        row = round((1 - fraction) * (rows - 1))
        if log:
            value = 10 ** (math.log10(low) + fraction * (math.log10(high) - math.log10(low)))
        else:
            value = low + fraction * (high - low)
        ticks.append((_format(value), row))
    # Rounding can put two ticks on one row on a very short chart; the highest wins.
    seen: dict[int, str] = {}
    for label, row in ticks:
        seen.setdefault(row, label)
    return [(label, row) for row, label in sorted(seen.items())]


def _axis(gutter: int, cols: int) -> Text:
    """The horizontal rule under the plot."""
    line = Text(" " * (gutter - 1), style=Palette.BORDER)
    line.append("└", style=Palette.BORDER)
    line.append("─" * cols, style=Palette.BORDER)
    return line


def _x_labels(
    span: tuple[float, float], gutter: int, cols: int, *, log: bool, caption: str
) -> Text:
    """Tick labels along the bottom, plus the axis caption.

    Three ticks -- start, middle, end. More would collide on a narrow terminal, and the
    x axis of a residual plot is read for "how far through" rather than for exact values.
    """
    low, high = span
    if log and low > 0 and high > 0:
        middle = 10 ** ((math.log10(low) + math.log10(high)) / 2)
    else:
        middle = (low + high) / 2

    labels = [_format(low), _format(middle), _format(high)]
    row = [" "] * cols
    for position, label in zip((0, cols // 2, cols), labels, strict=False):
        start = min(max(0, position - len(label) // 2), max(0, cols - len(label)))
        for offset, character in enumerate(label):
            if start + offset < cols:
                row[start + offset] = character

    text = Text(" " * gutter)
    text.append("".join(row).rstrip(), style=Palette.FAINT)
    if caption:
        text.append(f"   {caption}", style=Palette.MUTED)
    return text


def _format(value: float) -> str:
    """Render an axis value compactly.

    ``%g`` throughout: a solver's time is as likely to be ``1e-05`` as ``12.5``, and
    neither a fixed number of decimals nor ``str`` renders both readably.
    """
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.0e}"
    return f"{value:g}" if magnitude >= 1 else f"{value:.4g}"
