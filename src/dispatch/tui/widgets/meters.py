"""Machine status, as one line.

Previously this was four labelled bars stacked in a block — a dashboard widget. It is now
a status bar: cores, CPU, memory, and load on a single row, each a compact figure with a
short sparkline-style gauge beside it.

The reason is proportion. The machine's state is context for the job list, not the subject
of the screen, and four rows of chrome above the content said otherwise.

One thing is kept from the old design and is not negotiable: **two different CPU numbers
are shown.** ``allocated`` is what Dispatch has promised out of its ledger and is what
admission decisions use; ``cpu`` is what the cores are actually doing. They disagree
whenever a solver blocks on I/O, and collapsing them into one tidy number would hide the
single most confusing thing about the scheduler.
"""

from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from dispatch.tui.theme import Palette, usage_style

__all__ = ["ResourceMeters"]

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


def _gauge(fraction: float) -> Text:
    """A short two-weight rule showing a fraction.

    Colour comes from :func:`~dispatch.tui.theme.usage_style`, which leaves anything under
    70% grey. A machine running comfortably should not light up.
    """
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * GAUGE_WIDTH)
    style = usage_style(fraction * 100)

    gauge = Text()
    gauge.append(GAUGE_FULL * filled, style=style)
    gauge.append(GAUGE_EMPTY * (GAUGE_WIDTH - filled), style=Palette.BORDER)
    return gauge
