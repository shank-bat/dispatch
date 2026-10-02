"""Running renders in the background, and telling clients how they are going.

A render is not a job (§13.32): it holds no allocation, has no place in the history, and
must not wait in the queue behind a week-long simulation. But an animation can still take
hours, and the daemon answers each client's requests **one at a time** -- so a render run
inside the request that asked for it would freeze every other request from that interface
for as long as it took. The dashboard would stop updating its queue, cancelling a job would
hang, and nothing would say why.

So a render is started, given an id, and left to run as a task. The request returns at once;
progress and completion arrive as events on the ``renders`` topic, which is how the interface
learns about everything else that happens while nobody is waiting on it.

Progress comes from the renderer's own output, read a line at a time as it is written -- the
generated ParaView script prints ``frame i/n`` -- and published at most once a second, so a
ten-thousand-frame animation does not become ten thousand events.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from dispatch.core.errors import ValidationError
from dispatch.core.plan import CommandStep
from dispatch.core.visual import VisualPlan
from dispatch.daemon.events import EventBus
from dispatch.ipc.protocol import Event

__all__ = ["RenderManager", "RenderTask"]

log = logging.getLogger(__name__)

PROGRESS_INTERVAL_S: Final = 1.0
"""Minimum time between progress events for one render."""

OUTPUT_TAIL: Final = 40
"""Lines of a renderer's output kept for the error message if it fails."""

_FRAME = re.compile(r"frame (\d+)/(\d+)")


@dataclass
class RenderTask:
    """One render in flight, and what clients are told about it."""

    id: str
    case: Path
    kind: str
    plan: VisualPlan
    task: asyncio.Task[None] | None = None
    step: int = 0
    frame: int | None = None
    frames: int | None = None
    started_at: float = field(default_factory=time.time)
    output: list[str] = field(default_factory=list)
    last_progress: float = 0.0
    """Monotonic time of the last progress event, for throttling."""
    finished: bool = False
    """Whether ``render.finished`` has been published, so it is published exactly once."""

    @property
    def outputs(self) -> list[str]:
        return [str(path) for path in self.plan.outputs]

    def describe(self) -> dict[str, Any]:
        """The wire form, shared by progress events and ``render.list``."""
        step = self.plan.steps[min(self.step, len(self.plan.steps) - 1)]
        return {
            "id": self.id,
            "case": str(self.case),
            "kind": self.kind,
            "step": self.step + 1,
            "steps": len(self.plan.steps),
            "description": step.description,
            "frame": self.frame,
            "frames": self.frames,
            "outputs": self.outputs,
            "elapsed_s": time.time() - self.started_at,
        }


class RenderManager:
    """Starts, tracks and cancels background renders.

    Args:
        bus: Where progress and completion are published.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._active: dict[str, RenderTask] = {}

    @property
    def active(self) -> Sequence[RenderTask]:
        return tuple(self._active.values())

    def start(self, case: Path, kind: str, plan: VisualPlan) -> RenderTask:
        """Begin a render and return immediately.

        A second render writing the same output while the first is still going is refused:
        both would write the same frames directory, and the video would be whichever
        happened to finish last, spliced from both.
        """
        wanted = {str(path) for path in plan.outputs}
        for other in self._active.values():
            if wanted & set(other.outputs):
                raise ValidationError(
                    f"A render of {', '.join(sorted(wanted))} is already running "
                    f"(id {other.id[:8]}). Wait for it, or cancel it first."
                )
        entry = RenderTask(id=str(uuid.uuid4()), case=case, kind=kind, plan=plan)
        entry.task = asyncio.create_task(self._run(entry), name=f"dispatch-render-{entry.id[:8]}")
        # A task cancelled before it first runs never enters _run, so its cleanup is not
        # there to rely on: without this it would stay listed as running for ever.
        entry.task.add_done_callback(lambda _: self._settle(entry, False, "cancelled"))
        self._active[entry.id] = entry
        log.info("Render %s started for %s (%s)", entry.id[:8], case, kind)
        return entry

    async def run_inline(self, case: Path, kind: str, plan: VisualPlan) -> list[str]:
        """Run a render to completion in the caller's request. For the CLI, which waits."""
        entry = RenderTask(id=str(uuid.uuid4()), case=case, kind=kind, plan=plan)
        await self._steps(entry)
        return entry.output

    def cancel(self, render_id: str) -> bool:
        """Stop a render. Its process is killed; frames already written are left."""
        entry = self._find(render_id)
        if entry is None or entry.task is None:
            return False
        entry.task.cancel()
        return True

    async def shutdown(self) -> None:
        """Stop every render. A renderer must not outlive the daemon that supervises it."""
        tasks = [entry.task for entry in self._active.values() if entry.task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _find(self, render_id: str) -> RenderTask | None:
        """By full id or by an unambiguous prefix, as job ids are."""
        matches = [entry for key, entry in self._active.items() if key.startswith(render_id)]
        return matches[0] if len(matches) == 1 else None

    # -- running -------------------------------------------------------------------------

    async def _run(self, entry: RenderTask) -> None:
        ok, error = True, None
        try:
            await self._steps(entry)
        except asyncio.CancelledError:
            ok, error = False, "cancelled"
        except ValidationError as exc:
            ok, error = False, str(exc)
        except Exception as exc:  # a render must never take the daemon down with it
            log.exception("Render %s failed unexpectedly", entry.id[:8])
            ok, error = False, f"internal error: {exc}"
        self._settle(entry, ok, error)

    def _settle(self, entry: RenderTask, ok: bool, error: str | None) -> None:
        """Forget a finished render and tell clients how it ended. Idempotent."""
        self._active.pop(entry.id, None)
        if entry.finished:
            return
        entry.finished = True
        self._bus.publish(
            Event.RENDER_FINISHED,
            {
                **entry.describe(),
                "ok": ok,
                "error": error,
                "produced": entry.output[-OUTPUT_TAIL:],
            },
        )
        log.info("Render %s finished: %s", entry.id[:8], "ok" if ok else error)

    async def _steps(self, entry: RenderTask) -> None:
        for index, step in enumerate(entry.plan.steps):
            entry.step = index
            self._publish_progress(entry, force=True)
            code = await self._run_step(entry, step)
            if code != 0:
                tail = next((line for line in reversed(entry.output) if line.strip()), "")
                raise ValidationError(
                    f"{step.description} failed (exit {code})" + (f": {tail}" if tail else "")
                )

    async def _run_step(self, entry: RenderTask, step: CommandStep) -> int:
        """Run one command, reading its output as it arrives, bounded by its timeout.

        The process is killed on timeout and on cancellation: a renderer that cannot find a
        GL context does not fail, it waits, and a cancelled render must actually stop using
        the machine.
        """
        process = await asyncio.create_subprocess_exec(
            *step.argv,
            cwd=str(step.cwd),
            env=dict(step.env) if step.env is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(self._read(entry, process), timeout=step.timeout_s)
            return await process.wait()
        except TimeoutError:
            raise ValidationError(
                f"{step.description} exceeded its {step.timeout_s:.0f}s limit and was stopped"
            ) from None
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await process.wait()

    async def _read(self, entry: RenderTask, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        while True:
            raw = await process.stdout.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip()
            # ParaView's own diagnostics are long; keeping only a tail bounds the memory a
            # chatty render can take.
            entry.output.append(line)
            del entry.output[:-OUTPUT_TAIL]
            match = _FRAME.search(line)
            if match is not None:
                entry.frame, entry.frames = int(match.group(1)), int(match.group(2))
                self._publish_progress(entry)

    def _publish_progress(self, entry: RenderTask, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - entry.last_progress < PROGRESS_INTERVAL_S:
            return
        entry.last_progress = now
        self._bus.publish(Event.RENDER_PROGRESS, entry.describe())
