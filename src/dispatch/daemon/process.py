"""Spawning, supervising, signalling, and re-adopting job processes.

The primitives here are what make two of Dispatch's hard requirements true: a simulation
survives the daemon exiting, and its exit code is recovered anyway.

Three techniques carry that weight:

``start_new_session=True``
    Every job gets its own session and process group. It is detached from the daemon's
    controlling terminal (a stray Ctrl-C cannot reach it), the whole process tree can be
    signalled at once, and the daemon can exit without hanging up on its children.

Exit sentinels
    A process reparented to init cannot be ``wait()``ed, so its exit status is
    unrecoverable through normal means. The command is therefore wrapped in a tiny shell
    that writes its exit code to a file before exiting. Slightly ugly; it is the only way
    to answer "what happened" after a daemon restart.

``pidfd_open``
    Lets the daemon wait for a process it did not fork, so a re-adopted job still notifies
    on exit instead of being polled.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from dispatch.core.clock import Clock, SystemClock
from dispatch.core.plan import CommandStep

__all__ = ["ProcessHandle", "ProcessManager", "SpawnError", "system_boot_time"]

log = logging.getLogger(__name__)

SIGNAL_LADDER: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)
"""Cancellation escalation.

SIGINT first because most CFD solvers treat it as "write the current state and stop
cleanly" -- a cancelled run that leaves a usable final write is worth far more than one
killed outright.
"""


class SpawnError(OSError):
    """A step's program could not be started."""


@dataclass(frozen=True, slots=True)
class ProcessHandle:
    """A running job process."""

    pid: int
    start_time: float
    """The process's creation time from ``/proc``.

    Stored with the pid because a pid alone is not an identity: after a daemon restart
    that number may belong to something else entirely. The pair is what makes re-adoption
    safe.
    """

    process: asyncio.subprocess.Process | None = None
    """``None`` for a re-adopted process, which the daemon did not fork."""

    exit_file: Path | None = None


class ProcessManager:
    """Starts and supervises the processes a plan describes.

    Args:
        clock: Time source, injected for tests.
        shell: Shell used for the exit-sentinel wrapper.
    """

    def __init__(self, *, clock: Clock | None = None, shell: str = "/bin/sh") -> None:
        self._clock = clock or SystemClock()
        self._shell = shell

    # -- spawning -------------------------------------------------------------------------

    async def spawn(
        self,
        step: CommandStep,
        *,
        stdout: IO[bytes] | int | None = None,
        stderr: IO[bytes] | int | None = None,
        exit_file: Path | None = None,
    ) -> ProcessHandle:
        """Start ``step`` in its own session.

        ``stdout`` and ``stderr`` are file objects or descriptors, handed straight to the
        child. The daemon never sees the output: the kernel writes solver output directly
        to disk, so a job that prints 40 MB of residuals over three days costs the daemon
        nothing (§13.4).

        Args:
            exit_file: When given, the command is wrapped so its exit code is written here
                before the process exits, surviving a daemon crash.

        Raises:
            SpawnError: If the program does not exist or cannot be executed.
        """
        argv = self._wrap(step.argv, exit_file) if exit_file else list(step.argv)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(step.cwd),
                env=dict(step.env) if step.env is not None else None,
                stdout=stdout if stdout is not None else asyncio.subprocess.DEVNULL,
                stderr=stderr if stderr is not None else asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise SpawnError(f"{step.program}: command not found") from exc
        except PermissionError as exc:
            raise SpawnError(f"{step.program}: not executable") from exc
        except OSError as exc:
            raise SpawnError(f"{step.program}: {exc}") from exc

        return ProcessHandle(
            pid=process.pid,
            start_time=process_start_time(process.pid) or self._clock.now(),
            process=process,
            exit_file=exit_file,
        )

    def _wrap(self, argv: list[str] | tuple[str, ...] | Any, exit_file: Path) -> list[str]:
        """Wrap a command so its exit code lands on disk before the process exits.

        ``sh -c 'script' <exit_file> <program> <args...>`` puts the exit file in ``$0`` and
        the real command in ``"$@"``, so nothing needs quoting into the script text and a
        case directory containing a space stays intact.
        """
        script = 'ec=0; "$@" || ec=$?; printf %s "$ec" > "$0" 2>/dev/null; exit $ec'
        return [self._shell, "-c", script, str(exit_file), *argv]

    # -- waiting ----------------------------------------------------------------------------

    async def wait(self, handle: ProcessHandle, *, timeout: float | None = None) -> int | None:
        """Wait for a process to exit.

        Returns:
            Its exit code, or ``None`` if ``timeout`` elapsed first. A process killed by a
            signal reports the shell's ``128 + signum`` convention, because the wrapper
            propagates it.
        """
        if handle.process is not None:
            if timeout is None:
                return await handle.process.wait()
            try:
                return await asyncio.wait_for(handle.process.wait(), timeout)
            except TimeoutError:
                return None
        return await self.wait_adopted(handle, timeout=timeout)

    async def wait_adopted(
        self, handle: ProcessHandle, *, timeout: float | None = None
    ) -> int | None:
        """Wait for a process the daemon did not fork.

        Uses ``pidfd_open`` so the wait is event-driven rather than a poll. Falls back to
        a slow poll where pidfds are unavailable (Linux before 5.3), which affects nothing
        else -- a re-adopted job's exit code comes from its sentinel file either way.
        """
        try:
            fd = os.pidfd_open(handle.pid)
        except (AttributeError, OSError):
            return await self._poll_until_gone(handle.pid, timeout=timeout)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def _on_exit() -> None:
            if not future.done():
                future.set_result(None)

        try:
            loop.add_reader(fd, _on_exit)
            if timeout is None:
                await future
            else:
                try:
                    await asyncio.wait_for(asyncio.shield(future), timeout)
                except TimeoutError:
                    return None
        finally:
            with contextlib.suppress(Exception):
                loop.remove_reader(fd)
            os.close(fd)
        return None  # the code comes from the sentinel; we cannot reap a non-child

    async def _poll_until_gone(self, pid: int, *, timeout: float | None) -> int | None:
        deadline = None if timeout is None else self._clock.monotonic() + timeout
        while is_alive(pid):
            if deadline is not None and self._clock.monotonic() >= deadline:
                return None
            await asyncio.sleep(2.0)
        return None

    # -- signalling ---------------------------------------------------------------------------

    def signal_group(self, pid: int, sig: signal.Signals) -> bool:
        """Signal a job's entire process group.

        The group is what matters: an MPI launcher's children are the actual solver ranks,
        and signalling only the launcher leaves them running.

        Returns:
            ``True`` if the signal was delivered, ``False`` if the group was already gone.
        """
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            return False
        except PermissionError:
            log.warning("Not permitted to signal process group %d", pid)
            return False
        except OSError as exc:  # pragma: no cover - unusual kernel states
            log.warning("Could not signal process group %d: %s", pid, exc)
            return False
        return True

    async def terminate(
        self, handle: ProcessHandle, *, grace_s: float = 10.0
    ) -> signal.Signals | None:
        """Stop a job, escalating through the signal ladder.

        Each rung is given ``grace_s`` to work before the next is tried. Returns the signal
        that finally worked, or ``None`` if the process was already gone.
        """
        for sig in SIGNAL_LADDER:
            if not is_alive(handle.pid):
                return None
            log.info("Sending %s to process group %d", sig.name, handle.pid)
            if not self.signal_group(handle.pid, sig):
                return None
            if await self._wait_for_exit(handle, grace_s):
                return sig
        return SIGNAL_LADDER[-1]

    async def _wait_for_exit(self, handle: ProcessHandle, timeout: float) -> bool:
        """Wait up to ``timeout`` for a process to die. Returns whether it did."""
        if handle.process is not None:
            try:
                await asyncio.wait_for(handle.process.wait(), timeout)
            except TimeoutError:
                return False
            return True

        deadline = self._clock.monotonic() + timeout
        while self._clock.monotonic() < deadline:
            if not is_alive(handle.pid):
                return True
            await asyncio.sleep(0.2)
        return not is_alive(handle.pid)

    # -- exit codes -----------------------------------------------------------------------------

    @staticmethod
    def read_exit_file(path: Path) -> int | None:
        """Read an exit sentinel.

        Returns ``None`` if the file is missing or unreadable, which is exactly the
        situation that makes a job's outcome UNKNOWN rather than assumed.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        try:
            return int(text)
        except ValueError:
            return None


def is_alive(pid: int) -> bool:
    """Whether a process with this pid currently exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by somebody else
    return True


def process_start_time(pid: int) -> float | None:
    """Read a process's creation time, for use as a pid-reuse guard.

    Read from ``/proc/<pid>/stat`` field 22 directly rather than through psutil, because
    this is called on the hot path of every spawn and the import is not worth it.
    """
    try:
        with Path(f"/proc/{pid}/stat").open("rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    # The comm field is parenthesised and may itself contain spaces or parentheses, so
    # split from the last ')' rather than tokenising the whole line.
    close = raw.rfind(b")")
    if close == -1:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        ticks = float(fields[19])
    except ValueError:
        return None
    return ticks / os.sysconf("SC_CLK_TCK")


def matches_start_time(pid: int, expected: float | None, *, tolerance: float = 1.0) -> bool:
    """Whether a live pid is still the process Dispatch started.

    Without this check, re-adoption after a restart could attach to whatever unrelated
    program happened to inherit the pid -- and then signal it on cancel.
    """
    if expected is None:
        return False
    actual = process_start_time(pid)
    if actual is None:
        return False
    return abs(actual - expected) <= tolerance


def describe_exit(code: int | None) -> tuple[int | None, str | None]:
    """Split a shell-convention exit code into ``(code, signal name)``.

    ``128 + n`` means "killed by signal n", which is how the sentinel wrapper reports a
    signalled solver.
    """
    if code is None:
        return None, None
    if code > 128:
        try:
            return code, signal.Signals(code - 128).name.removeprefix("SIG")
        except ValueError:
            return code, None
    return code, None


def render_command(argv: list[str] | tuple[str, ...]) -> str:
    """Shell-quote a command for logs. Display only; nothing runs through a shell."""
    return shlex.join(argv)


def system_boot_time() -> float | None:
    """This machine's boot identity: the kernel's ``btime``, in seconds since the epoch.

    Read from ``/proc/stat`` rather than derived from ``/proc/uptime``, because ``btime`` is
    a fixed integer the kernel reports identically on every read, while ``now - uptime``
    drifts by a fraction of a second each time it is computed. The value is compared
    against a copy of itself recorded hours or days earlier, so a stable reading matters
    more than a precise one.

    ``None`` when it cannot be read, which callers treat as no evidence of a reboot.
    """
    try:
        with Path("/proc/stat").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):  # pragma: no cover - environmental
        return None
    return None
