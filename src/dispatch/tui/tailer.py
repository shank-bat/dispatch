"""Following a growing log file.

Log files are read **directly from disk**, never proxied through the daemon: same machine,
same user, same filesystem. Streaming gigabytes of residuals through a JSON socket to
render them in a terminal would be absurd.

Growth is detected by ``stat``-ing the file on an adaptive interval rather than with
inotify. The stdlib has no inotify binding, so that would mean either a new dependency or
a hand-rolled ``ctypes`` wrapper with real edge cases (queue overflow, watch-descriptor
limits, NFS). The wakeups it would save happen only while a human has a log open, and one
``fstat`` costs microseconds. :class:`Tailer` is a Protocol, so an inotify backend can be
dropped in later without touching the widget.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Protocol

__all__ = ["AsyncTailer", "Tailer", "read_last_lines"]

FAST_INTERVAL = 0.15
"""Poll interval while the file is actively growing."""

SLOW_INTERVAL = 2.0
"""Poll interval once it has been quiet, so an idle viewer costs almost nothing."""

CHUNK_LIMIT = 1024 * 1024
"""Most bytes read in one go, so a burst cannot block the UI for a whole frame."""


class Tailer(Protocol):
    """Yields text as it is appended to a file."""

    def follow(self) -> AsyncIterator[str]: ...


class AsyncTailer:
    """Follows a file, yielding newly appended text.

    Handles the three things that actually happen to log files: they grow, they get
    truncated (a job restarted and reopened its log), and they briefly do not exist yet.

    Args:
        path: File to follow.
        from_start: Start at the beginning rather than at the current end.
        tail_bytes: When not starting from the beginning, how much of the end to emit
            first, so a viewer opens with context instead of a blank pane.
    """

    def __init__(
        self, path: Path, *, from_start: bool = False, tail_bytes: int = 64 * 1024
    ) -> None:
        self.path = path
        self._from_start = from_start
        self._tail_bytes = tail_bytes
        self._offset = 0
        self._stopped = False

    def stop(self) -> None:
        """Ask the follow loop to end."""
        self._stopped = True

    async def follow(self) -> AsyncIterator[str]:
        """Yield appended text until stopped.

        The interval adapts between :data:`FAST_INTERVAL` while output is flowing and
        :data:`SLOW_INTERVAL` once it stops, so a solver that prints once a minute does
        not cost a poll every 150 ms all night.
        """
        interval = FAST_INTERVAL
        primed = False

        while not self._stopped:
            try:
                size = self.path.stat().st_size
            except OSError:
                # The job has not created its log yet, which is normal right after submit.
                await asyncio.sleep(SLOW_INTERVAL)
                continue

            if not primed:
                self._offset = 0 if self._from_start else max(0, size - self._tail_bytes)
                primed = True

            if size < self._offset:
                # Truncated or rotated: start again from the beginning of the new file.
                self._offset = 0

            if size > self._offset:
                chunk = self._read(size)
                if chunk:
                    yield chunk
                interval = FAST_INTERVAL
            else:
                interval = min(SLOW_INTERVAL, interval * 1.5)

            await asyncio.sleep(interval)

    def _read(self, size: int) -> str:
        """Read the new bytes, capped so one burst cannot stall the interface."""
        want = min(size - self._offset, CHUNK_LIMIT)
        try:
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                data = handle.read(want)
        except OSError:
            return ""
        self._offset += len(data)
        return data.decode("utf-8", errors="replace")


def read_last_lines(path: Path, count: int, *, block: int = 64 * 1024) -> list[str]:
    """Read the last ``count`` lines of a file without reading all of it.

    Walks backwards a block at a time. A three-day residual log can be hundreds of
    megabytes, and reading it whole to show fifty lines would stall the interface and
    spike memory for no reason.
    """
    if count <= 0:
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            data = b""
            newlines = 0
            position = end

            while position > 0 and newlines <= count:
                step = min(block, position)
                position -= step
                handle.seek(position)
                data = handle.read(step) + data
                newlines = data.count(b"\n")

            lines = data.decode("utf-8", errors="replace").splitlines()
            return lines[-count:]
    except OSError:
        return []


async def watch_file(
    path: Path, on_text: Callable[[str], None], *, from_start: bool = False
) -> None:
    """Follow a file, handing each new chunk to a callback. Cancel the task to stop."""
    tailer = AsyncTailer(path, from_start=from_start)
    with contextlib.suppress(asyncio.CancelledError):
        async for chunk in tailer.follow():
            on_text(chunk)
