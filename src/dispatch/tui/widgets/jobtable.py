"""The job table, used by the dashboard, the queue, and history.

One widget rather than three: the columns differ only in which are shown, and three
near-identical tables would drift apart within a month.

Visually this is deliberately plain. There are no grid lines, no zebra striping and no
border — separation comes from column spacing, and the only strong element on the row is
the selection highlight. Colour appears in exactly one column, the state, because that is
the field a glance is looking for.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.widgets import DataTable

from dispatch.tui.theme import Palette, state_style

__all__ = ["JobTable", "format_duration", "state_text"]

PROGRESS_WIDTH = 10
PROGRESS_FULL = "━"
PROGRESS_EMPTY = "─"


class JobTable(DataTable[Any]):
    """A keyboard-navigable table of jobs.

    Rows are keyed by job id, so a state change updates in place and the cursor does not
    jump — which matters when a job finishes while you are pointing at the one below it.
    """

    COLUMNS = (
        ("", 3),
        ("job", 26),
        ("state", 11),
        ("cores", 5),
        ("solver", 13),
        ("progress", 17),
        ("step", 14),
        ("time", 8),
        ("tags", 20),
    )

    def __init__(self, *, show_progress: bool = True, **kwargs: Any) -> None:
        super().__init__(
            cursor_type="row",
            zebra_stripes=False,
            show_cursor=True,
            **kwargs,
        )
        self._show_progress = show_progress
        self._ids: list[str] = []

    def on_mount(self) -> None:
        for label, width in self.COLUMNS:
            if label in ("progress", "step") and not self._show_progress:
                continue
            # Lowercase, dim, no styling: a header should label a column without
            # competing with the data underneath it.
            self.add_column(
                Text(label, style=Palette.FAINT), width=width, key=label or "pos"
            )

    @property
    def selected_job_id(self) -> str | None:
        """The job under the cursor, or ``None`` when the table is empty."""
        if not self._ids or self.cursor_row < 0:
            return None
        if self.cursor_row >= len(self._ids):
            return None
        return self._ids[self.cursor_row]

    def show(
        self, jobs: Sequence[dict[str, Any]], progress: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """Replace the table's contents, preserving the cursor position where possible."""
        previous = self.selected_job_id
        self.clear()
        self._ids = []

        for job in jobs:
            self._ids.append(job["id"])
            self.add_row(*self._cells(job, (progress or {}).get(job["id"])))

        if previous in self._ids:
            self.move_cursor(row=self._ids.index(previous))

    def _cells(self, job: dict[str, Any], progress: dict[str, Any] | None) -> list[Any]:
        position = job.get("queue_position")
        cells: list[Any] = [
            Text(str(position) if position else "", style=Palette.FAINT),
            Text(_ellipsise(job["name"], 26), style=Palette.TEXT),
            state_text(job["state"]),
            Text(str(job["cores"]), justify="right", style=Palette.MUTED),
            Text(
                _ellipsise(job.get("solver_binary") or job["solver"], 13),
                style=Palette.MUTED,
            ),
        ]
        if self._show_progress:
            cells.append(_progress_text(job, progress))
            cells.append(_step_text(job, progress))
        cells.extend(
            [
                Text(_runtime(job), justify="right", style=Palette.MUTED),
                Text(_ellipsise(" ".join(job.get("tags", [])), 20), style=Palette.FAINT),
            ]
        )
        return cells


def state_text(state: str) -> Text:
    """A state name in lowercase, coloured by what it means.

    Lowercase because ALL CAPS in a dense column reads as shouting, and the states are
    already distinguished by colour and position.
    """
    return Text(state.lower(), style=state_style(state))


def _progress_text(job: dict[str, Any], progress: dict[str, Any] | None) -> Text:
    """Render progress, falling back to memory usage when the solver reports none.

    A solver with no declared end time genuinely has no percentage. Showing an invented
    one would be worse than showing something else useful.
    """
    if job["state"] not in ("RUNNING", "PREPARING"):
        return Text("")
    if not progress:
        return Text("·", style=Palette.FAINT)

    detail = progress.get("progress")
    if detail and detail.get("fraction") is not None:
        fraction = max(0.0, min(1.0, float(detail["fraction"])))
        filled = round(fraction * PROGRESS_WIDTH)
        bar = Text()
        bar.append(PROGRESS_FULL * filled, style=Palette.ACCENT_TEXT)
        bar.append(PROGRESS_EMPTY * (PROGRESS_WIDTH - filled), style=Palette.BORDER)
        bar.append(f" {fraction * 100:3.0f}%", style=Palette.MUTED)
        return bar
    if detail:
        return Text(step_label(detail), style=Palette.MUTED)

    rss = progress.get("rss_mb")
    return Text(f"{rss} MB" if rss else "·", style=Palette.FAINT)


def _step_text(job: dict[str, Any], progress: dict[str, Any] | None) -> Text:
    """The solver's own current time or iteration, verbatim.

    Separate from the progress bar rather than crammed into it. A percentage answers "how
    much longer"; the time step answers "where is the solution now", which is the number
    the user is actually watching for -- it is what they compare against the physics, and
    it is the only one of the two that exists when a case has no declared end time.
    """
    if job["state"] not in ("RUNNING", "PREPARING"):
        return Text("")
    detail = (progress or {}).get("progress")
    if not detail:
        return Text("")
    return Text(step_label(detail), justify="right", style=Palette.TEXT)


def step_label(detail: dict[str, Any]) -> str:
    """Format one progress reading as ``t 0.35`` or ``iter 1250``.

    ``%g`` because a solver's time is as likely to be ``1e-05`` as ``12.5``, and neither a
    fixed number of decimals nor a bare ``str`` renders both readably.
    """
    label = str(detail.get("label") or "").lower()
    short = {"time": "t", "iteration": "iter", "step": "step"}.get(label, label[:4] or "t")
    try:
        current = f"{float(detail['current']):g}"
    except (TypeError, ValueError, KeyError):
        return ""
    total = detail.get("total")
    if total:
        try:
            return f"{short} {current}/{float(total):g}"
        except (TypeError, ValueError):
            pass
    return f"{short} {current}"


def _runtime(job: dict[str, Any]) -> str:
    runtime = (job.get("metrics") or {}).get("runtime_s")
    if runtime:
        return format_duration(runtime)
    started = job.get("started_at")
    if started and job["state"] in ("RUNNING", "PREPARING"):
        return format_duration(datetime.now().timestamp() - started)
    return ""


def format_duration(seconds: float | None) -> str:
    """Render a duration compactly: ``3d 4h``, ``2h 14m``, ``45s``."""
    if not seconds or seconds < 0:
        return ""
    total = int(seconds)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _ellipsise(value: str, width: int) -> str:
    """Truncate with an ellipsis rather than a hard cut.

    A name cut mid-word looks like corrupted data; one ending in ``…`` obviously continues.
    """
    if len(value) <= width:
        return value
    return value[: width - 1] + "…"
