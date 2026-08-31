"""The live log viewer.

Reads the log file directly from disk, never through the daemon. Two consequences worth
knowing about:

* Opening a log costs the daemon nothing, so viewing a job cannot slow it down.
* Logs remain readable when the daemon is down.

The in-memory buffer is bounded. A solver that emits a two-gigabyte log must not be able
to take the interface out with it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, RichLog, Static

from dispatch.tui.screens.base import DispatchScreen
from dispatch.tui.tailer import AsyncTailer, read_last_lines
from dispatch.tui.theme import Palette

__all__ = ["LogScreen"]

MAX_LINES = 5000
"""Lines kept in memory. Beyond this, the oldest are dropped."""

HEADING_REASON_CHARS = 200
"""How much of a failure reason fits on the heading's second line."""


def _first_line(text: str) -> str:
    """The failure reason, collapsed to one line for the heading.

    The whole reason may be several lines; the heading has room for the first, and the
    rest is in the log directly below it.
    """
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    if len(line) > HEADING_REASON_CHARS:
        line = line[: HEADING_REASON_CHARS - 1].rstrip() + "…"
    return line


class LogScreen(DispatchScreen):
    """``tail -f`` for one job, with search."""

    TITLE = "Logs"

    nav_key = ""

    BINDINGS = [
        Binding("escape,q", "back", "back"),
        Binding("e", "toggle_stream", "output/steps"),
        Binding("p", "plot", "plot"),
        Binding("G,end", "follow", "follow"),
        Binding("slash", "search", "search"),
        Binding("n", "next_match", "next match"),
        Binding("N", "previous_match", "previous match"),
        Binding("w", "wrap", "wrap"),
    ]

    def __init__(self, job_id: str) -> None:
        # Note the `_tail_` prefix on the task attribute below: Textual's MessagePump owns
        # `self._task` for the widget's own message loop, and shadowing it stops the screen
        # from ever mounting its children.
        super().__init__()
        self.job_id = job_id
        self._stream = "output"
        self._tail_task: asyncio.Task[None] | None = None
        self._tailer: AsyncTailer | None = None
        self._lines: deque[str] = deque(maxlen=MAX_LINES)
        self._matches: list[int] = []
        self._match_index = 0
        self._query = ""
        self._following = True

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical():
            yield Static("", id="heading")
            yield RichLog(id="log", highlight=False, markup=False, wrap=False, auto_scroll=True)
            yield Static("", id="log-status")
            yield Input(placeholder="search buffered lines…", id="search-box", classes="prompt")
        yield from self.compose_footer()

    async def on_mount(self) -> None:
        await super().on_mount()
        self.query_one("#log", RichLog).focus()
        self._start_tail()
        self.set_interval(1.0, self.update_status)
        self.update_status()

    def on_unmount(self) -> None:
        self._stop_tail()

    # -- tailing -------------------------------------------------------------------------

    @property
    def _streams(self) -> list[str]:
        """Which streams this job actually has, in toggle order.

        ``output`` is the solver's log and ``steps`` is the preparation transcript --
        decomposition, compilation, whatever ran before the solver. That second one used
        to be invisible in the interface even though it holds the explanation for most
        preparation failures.

        ``stderr`` appears only for jobs old enough to have a separate one. New jobs
        write both streams to a single file (§6.4), so offering an always-identical
        second view of it would be a key that does nothing.
        """
        streams = ["output", "steps"]
        job = self.app_state.get(self.job_id) or {}
        if job.get("stderr_path") and job.get("stderr_path") != job.get("stdout_path"):
            streams.insert(1, "stderr")
        return streams

    @property
    def _path(self) -> Path | None:
        job = self.app_state.get(self.job_id)
        if job is None:
            return None
        if self._stream == "steps":
            # Derived rather than carried on the job: the transcript is internal
            # bookkeeping in a directory Dispatch owns, and the interface reads log files
            # off the disk directly anyway (§9.4).
            return self.dispatch_app.config.paths.job_dir(self.job_id) / "steps.log"
        key = "stderr_path" if self._stream == "stderr" else "stdout_path"
        raw = job.get(key) or job.get("output_path")
        return Path(raw) if raw else None

    def _start_tail(self) -> None:
        self._stop_tail()
        widget = self.query_one("#log", RichLog)
        widget.clear()
        self._lines.clear()

        path = self._path
        if path is None or not path.exists():
            widget.write(Text("no output yet", style=Palette.FAINT))
            return

        # Seed with the tail of the file, so the viewer opens with context rather than
        # waiting for the solver's next line.
        for line in read_last_lines(path, 400):
            self._lines.append(line)
            widget.write(line)

        self._tailer = AsyncTailer(path)
        self._tailer._offset = path.stat().st_size
        self._tail_task = asyncio.create_task(self._follow(), name="dispatch-log-tail")

    async def _follow(self) -> None:
        assert self._tailer is not None
        widget = self.query_one("#log", RichLog)
        buffer = ""
        with contextlib.suppress(asyncio.CancelledError):
            async for chunk in self._tailer.follow():
                buffer += chunk
                *complete, buffer = buffer.split("\n")
                for line in complete:
                    self._lines.append(line)
                    widget.write(line)

    def _stop_tail(self) -> None:
        if self._tailer is not None:
            self._tailer.stop()
            self._tailer = None
        if self._tail_task is not None:
            self._tail_task.cancel()
            self._tail_task = None

    # -- status ---------------------------------------------------------------------------

    def heading(self) -> Text:
        """Which job this is, and how far along it is."""
        job = self.app_state.get(self.job_id)
        if job is None:
            return Text("job not found", style=Palette.ERROR)

        from dispatch.tui.widgets.jobtable import state_text, step_label

        text = Text()
        text.append(job["name"], style=f"bold {Palette.TEXT}")
        text.append("   ")
        text.append(state_text(job["state"]))
        text.append("   ")
        text.append(self._stream, style=Palette.MUTED)

        progress = self.app_state.progress_for(self.job_id)
        if progress:
            detail = progress.get("progress")
            if detail:
                # Both, not one or the other: the percentage is the estimate and the time
                # step is the fact, and the log viewer is where the fact is wanted.
                step = step_label(detail)
                if step:
                    text.append(f"   {step}", style=f"bold {Palette.TEXT}")
                if detail.get("fraction") is not None:
                    text.append(f"   {detail['fraction'] * 100:.0f}%", style=Palette.MUTED)
            if progress.get("rss_mb"):
                text.append(f"   {progress['rss_mb']} MB", style=Palette.FAINT)

        reason = job.get("exit_detail")
        if reason:
            text.append("\n")
            text.append(_first_line(reason), style=Palette.ERROR)
        return text

    def update_status(self) -> None:
        super().update_status()
        with contextlib.suppress(Exception):
            path = self._path
            line = Text(str(path) if path else "no log file yet", style=Palette.FAINT)
            line.append(f"   {len(self._lines)} lines", style=Palette.FAINT)
            if self._query:
                line.append(
                    f"   /{self._query} {len(self._matches)} matches", style=Palette.MUTED
                )
            self.query_one("#log-status", Static).update(line)

    # -- actions ----------------------------------------------------------------------------

    def action_back(self) -> None:
        self.dismiss()

    def action_toggle_stream(self) -> None:
        """Cycle through whichever streams this job has."""
        streams = self._streams
        try:
            position = streams.index(self._stream)
        except ValueError:
            position = -1
        self._stream = streams[(position + 1) % len(streams)]
        self._start_tail()
        self.update_status()

    def action_plot(self) -> None:
        """Plot the numbers in the log currently being read."""
        self.dispatch_app.open_plot(self.job_id)

    def action_follow(self) -> None:
        widget = self.query_one("#log", RichLog)
        widget.scroll_end(animate=False)
        widget.auto_scroll = True
        self._following = True

    def action_wrap(self) -> None:
        widget = self.query_one("#log", RichLog)
        widget.wrap = not widget.wrap
        widget.refresh()

    def action_search(self) -> None:
        box = self.query_one("#search-box", Input)
        box.add_class("visible")
        box.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._query = event.value.strip()
        box = self.query_one("#search-box", Input)
        box.remove_class("visible")
        self.query_one("#log", RichLog).focus()

        if not self._query:
            self._matches = []
            return

        needle = self._query.lower()
        self._matches = [i for i, line in enumerate(self._lines) if needle in line.lower()]
        self._match_index = 0
        if self._matches:
            self._scroll_to_match()
            self.notify_ok(f"{len(self._matches)} match(es)")
        else:
            self.notify_error(f"no match for {self._query!r} in the buffered lines")
        self.update_status()

    def action_next_match(self) -> None:
        if self._matches:
            self._match_index = (self._match_index + 1) % len(self._matches)
            self._scroll_to_match()

    def action_previous_match(self) -> None:
        if self._matches:
            self._match_index = (self._match_index - 1) % len(self._matches)
            self._scroll_to_match()

    def _scroll_to_match(self) -> None:
        """Jump to the current match, leaving follow mode.

        Scrolling away from the tail must stop auto-scroll, or the next line of solver
        output would yank the view back and make search unusable.
        """
        widget = self.query_one("#log", RichLog)
        widget.auto_scroll = False
        self._following = False
        widget.scroll_to(y=max(0, self._matches[self._match_index] - 5), animate=False)
