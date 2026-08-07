"""Notifications.

Sinks subscribe to the event bus that already exists for client push, so the scheduler and
executor contain no notification code at all -- which is also why a new sink can be a
third-party package registered through the ``dispatch.sinks`` entry-point group.

Delivery is fire-and-forget on a bounded queue with a per-sink timeout. A dead webhook is a
normal Tuesday, and it must never delay a job transition or accumulate work in memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, ClassVar, Protocol, runtime_checkable

from dispatch.core.config import NotificationConfig
from dispatch.core.states import JobState

__all__ = ["Notification", "NotificationSink", "Notifier", "build_sinks"]

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "dispatch.sinks"
DELIVERY_TIMEOUT_S = 10.0
QUEUE_SIZE = 128


@dataclass(frozen=True, slots=True)
class Notification:
    """Something worth telling the user about."""

    event: str
    title: str
    body: str
    job_id: str | None = None
    urgent: bool = False


@runtime_checkable
class NotificationSink(Protocol):
    """Somewhere notifications can be delivered."""

    name: ClassVar[str]

    async def deliver(self, note: Notification) -> None: ...


class LogSink:
    """Writes notifications to the daemon log. The only sink shipped enabled.

    Unglamorous, but it works on a headless machine with no network, which is the target
    environment.
    """

    name: ClassVar[str] = "log"

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings = dict(settings or {})

    async def deliver(self, note: Notification) -> None:
        level = logging.WARNING if note.urgent else logging.INFO
        log.log(level, "%s: %s", note.title, note.body)


class DesktopSink:
    """Sends a desktop notification via ``notify-send``.

    Useful when Dispatch runs on a workstation someone also sits at. Silently inert when
    ``notify-send`` is absent, which is the normal case on a headless box -- a missing
    optional tool is not an error worth reporting on every job.
    """

    name: ClassVar[str] = "desktop"

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings = dict(settings or {})
        self._binary = shutil.which("notify-send")

    async def deliver(self, note: Notification) -> None:
        if not self._binary:
            return
        argv = [
            self._binary,
            "--app-name=Dispatch",
            f"--urgency={'critical' if note.urgent else 'normal'}",
            note.title,
            note.body,
        ]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), DELIVERY_TIMEOUT_S)


class CommandSink:
    """Runs an arbitrary command, with the notification in its environment.

    The escape hatch that makes ntfy, Discord, email, and anything else possible without
    Dispatch growing an HTTP client or a dependency on one::

        [notifications]
        sinks = ["command"]

        [notifications.command]
        argv = ["curl", "-d", "$DISPATCH_BODY", "https://ntfy.sh/eddy-dispatch"]

    Arguments are passed through unexpanded; the notification is available to the command
    as ``DISPATCH_EVENT``, ``DISPATCH_TITLE``, ``DISPATCH_BODY``, and ``DISPATCH_JOB_ID``.
    """

    name: ClassVar[str] = "command"

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings = dict(settings or {})
        argv = self.settings.get("argv") or []
        self._argv = [str(part) for part in argv] if isinstance(argv, (list, tuple)) else []
        if not self._argv:
            log.warning(
                "The 'command' notification sink has no argv configured; it will do nothing"
            )

    async def deliver(self, note: Notification) -> None:
        if not self._argv:
            return
        import os

        env = dict(os.environ)
        env.update(
            {
                "DISPATCH_EVENT": note.event,
                "DISPATCH_TITLE": note.title,
                "DISPATCH_BODY": note.body,
                "DISPATCH_JOB_ID": note.job_id or "",
            }
        )
        process = await asyncio.create_subprocess_exec(
            *self._argv,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), DELIVERY_TIMEOUT_S)
        except TimeoutError:
            log.warning("Notification command timed out; killing it")
            with contextlib.suppress(ProcessLookupError):
                process.kill()


BUILTIN_SINKS: dict[str, type] = {
    LogSink.name: LogSink,
    DesktopSink.name: DesktopSink,
    CommandSink.name: CommandSink,
}


def build_sinks(config: NotificationConfig) -> list[NotificationSink]:
    """Construct the configured sinks.

    An unknown sink name is logged and skipped rather than raising: the daemon starting
    without one notification channel is much better than it not starting at all.
    """
    plugins = _discover_plugins()
    available = {**BUILTIN_SINKS, **plugins}

    sinks: list[NotificationSink] = []
    for name in config.sinks:
        sink_cls = available.get(str(name))
        if sink_cls is None:
            log.warning(
                "Unknown notification sink %r; known sinks: %s",
                name,
                ", ".join(sorted(available)),
            )
            continue
        try:
            sinks.append(sink_cls(config.options.get(str(name), {})))
        except Exception as exc:
            log.warning("Could not construct notification sink %r: %s", name, exc)
    return sinks


def _discover_plugins() -> dict[str, type]:
    found: dict[str, type] = {}
    try:
        entries = importlib_metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - environmental
        return found
    for entry in entries:
        try:
            sink_cls = entry.load()
        except Exception as exc:
            log.warning("Could not load notification sink %r: %s", entry.name, exc)
            continue
        found[getattr(sink_cls, "name", entry.name)] = sink_cls
    return found


class Notifier:
    """Delivers notifications to the configured sinks, off the hot path.

    Args:
        config: Which events to report and where.
        sinks: Sinks to use. Built from config when omitted.
    """

    def __init__(
        self, config: NotificationConfig, *, sinks: Sequence[NotificationSink] | None = None
    ) -> None:
        self._config = config
        self._sinks = list(sinks) if sinks is not None else build_sinks(config)
        self._queue: asyncio.Queue[Notification] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._task: asyncio.Task[None] | None = None
        self._enabled = {str(event) for event in config.on}
        self.dropped = 0

    @property
    def sink_names(self) -> list[str]:
        """Names of the active sinks."""
        return [sink.name for sink in self._sinks]

    def start(self) -> None:
        """Begin the delivery loop."""
        if self._sinks and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._loop(), name="dispatch-notifier")

    async def stop(self) -> None:
        """Stop delivering."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def notify_job(self, job: Any, *, event: str) -> None:
        """Queue a notification about a job, if this event is enabled.

        Non-blocking and never raises. A full queue drops the notification and counts it:
        a backed-up webhook must not be able to stall a state transition.
        """
        if event not in self._enabled or not self._sinks:
            return

        state = getattr(job, "state", None)
        urgent = state in (JobState.FAILED, JobState.UNKNOWN)
        runtime = getattr(getattr(job, "metrics", None), "runtime_s", None)
        detail = f" in {_duration(runtime)}" if runtime else ""

        # A failure notification that does not say what failed sends the user to the
        # terminal to find out. The first line of the reason usually saves that trip.
        reason = getattr(job, "exit_detail", None)
        because = f"\n{_first_line(reason)}" if urgent and reason else ""

        note = Notification(
            event=event,
            title=f"Dispatch: {job.name} {str(state).lower()}",
            body=(
                f"{job.name} ({job.solver}, {job.cores} cores) "
                f"{str(state).lower()}{detail}. {job.workdir}{because}"
            ),
            job_id=job.id,
            urgent=urgent,
        )
        try:
            self._queue.put_nowait(note)
        except asyncio.QueueFull:
            self.dropped += 1
            log.warning("Notification queue is full; dropped %d so far", self.dropped)

    async def _loop(self) -> None:
        while True:
            note = await self._queue.get()
            for sink in self._sinks:
                try:
                    await asyncio.wait_for(sink.deliver(note), DELIVERY_TIMEOUT_S)
                except TimeoutError:
                    log.warning("Notification sink %r timed out", sink.name)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("Notification sink %r failed: %s", sink.name, exc)


def event_for_state(state: JobState) -> str | None:
    """The notification event name for a terminal state, or ``None``."""
    return {
        JobState.COMPLETED: "job.completed",
        JobState.FAILED: "job.failed",
        JobState.CANCELLED: "job.cancelled",
        JobState.UNKNOWN: "job.failed",
    }.get(state)


def _first_line(text: str | None, *, limit: int = 160) -> str:
    """The first line of a failure reason, for a notification that has room for one.

    Desktop notification daemons truncate silently and inconsistently, so the trimming
    happens here where the ellipsis at least tells the user there is more to read.
    """
    if not text:
        return ""
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _duration(seconds: float | None) -> str:
    """Render a duration compactly, e.g. ``2h 14m``."""
    if not seconds:
        return "0s"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def check_notify_send() -> bool:
    """Whether desktop notifications are possible here. Used by ``dispatch doctor``."""
    return shutil.which("notify-send") is not None
