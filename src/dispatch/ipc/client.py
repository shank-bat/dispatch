"""The client side of the daemon protocol.

Used by both the TUI and the CLI, so there is exactly one implementation of connecting,
handshaking, pipelining, and reconnecting.

Two behaviours here matter more than they look:

* **Requests are pipelined.** Responses are matched to futures by id, so a slow
  ``history.search`` cannot delay a ``job.cancel`` queued behind it.
* **Disconnection is normal, not exceptional.** Upgrading Dispatch means restarting the
  daemon while a TUI is open. The client reconnects with backoff and reports the state
  rather than raising into a render loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Self

from dispatch.core.errors import DispatchError
from dispatch.ipc.codec import ProtocolError, read_message, write_message
from dispatch.ipc.protocol import PROTOCOL_VERSION, Method, Notification, Request, Topic
from dispatch.ipc.socketpath import connect_unix

__all__ = ["DaemonClient", "DaemonUnavailable", "RemoteError"]

log = logging.getLogger(__name__)


class DaemonUnavailable(DispatchError):
    """No daemon is listening, and one could not be started."""

    code = "DAEMON_UNAVAILABLE"


class RemoteError(DispatchError):
    """The daemon refused a request.

    Carries the daemon's own error code, so callers can branch on what went wrong without
    matching on English.
    """

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message, detail=detail)
        self.code = code


class DaemonClient:
    """An async client for one daemon.

    Args:
        socket_path: Path to the daemon's Unix socket.
        autostart: Start a daemon with ``--detach`` if none is listening. On by default so
            a new user runs ``dispatch`` and it simply works, with no systemd knowledge
            required (§2.1).
        config_path: Configuration file this client was started with, forwarded to any
            daemon it spawns. Without it an autostarted daemon would read the default
            configuration and bind a different socket than the one being waited on.
        on_event: Called for every server-pushed notification.
        on_state_change: Called with ``True`` on connect and ``False`` on disconnect, so a
            UI can show a banner without polling.
    """

    def __init__(
        self,
        socket_path: Path,
        *,
        autostart: bool = True,
        config_path: Path | None = None,
        on_event: Callable[[Notification], None] | None = None,
        on_state_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.autostart = autostart
        self.config_path = config_path
        self._on_event = on_event
        self._on_state_change = on_state_change

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._reader_task: asyncio.Task[None] | None = None
        self._topics: set[str] = set()
        self.server_info: dict[str, Any] = {}

    @property
    def connected(self) -> bool:
        """Whether a connection is currently established."""
        return self._writer is not None and not self._writer.is_closing()

    # -- lifecycle -----------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self, *, timeout: float = 5.0) -> None:
        """Connect and complete the handshake, starting a daemon if permitted.

        Raises:
            DaemonUnavailable: If no daemon is listening and autostart is off or failed.
            RemoteError: On a protocol version mismatch.
        """
        try:
            await self._open()
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            if not self.autostart:
                raise DaemonUnavailable(
                    f"No Dispatch daemon is listening on {self.socket_path}. "
                    "Start one with `dispatchd --detach`."
                ) from exc
            await self._spawn_daemon(timeout=timeout)
            try:
                await self._open()
            except OSError as retry_exc:
                raise DaemonUnavailable(
                    f"Started a daemon but could not connect to {self.socket_path}: {retry_exc}"
                ) from retry_exc

        self.server_info = await self.call(
            Method.HELLO,
            protocol=PROTOCOL_VERSION,
            client=f"dispatch/{_client_version()}",
        )
        if self._on_state_change:
            self._on_state_change(True)

    async def _open(self) -> None:
        # Reconnecting must not orphan the previous transport; a StreamWriter dropped with
        # a live socket raises from __del__ at an unpredictable moment later.
        await self._close_transport()
        reader, writer = await connect_unix(self.socket_path)
        self._reader, self._writer = reader, writer
        self._reader_task = asyncio.create_task(self._read_loop(), name="dispatch-client-reader")

    async def _close_transport(self) -> None:
        """Close the socket and wait for it, tolerating a peer that already went away."""
        writer, self._writer, self._reader = self._writer, None, None
        if writer is None:
            return
        with contextlib.suppress(Exception):
            writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    async def _spawn_daemon(self, *, timeout: float) -> None:
        """Start a detached daemon and wait for its socket to appear."""
        executable = shutil.which("dispatchd")
        argv = [executable] if executable else [sys.executable, "-m", "dispatch.daemon.main"]
        if self.config_path is not None:
            argv.extend(["--config", str(self.config_path)])
        argv.append("--detach")
        log.info("No daemon listening; starting one with %s", " ".join(argv))
        try:
            subprocess.run(argv, check=True, capture_output=True, timeout=timeout + 5)
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", b"") or b""
            raise DaemonUnavailable(
                f"Could not start a Dispatch daemon: {exc}. "
                f"{detail.decode(errors='replace').strip()}".strip()
            ) from exc

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if self.socket_path.exists():
                return
            await asyncio.sleep(0.05)

    async def close(self) -> None:
        """Close the connection and cancel every in-flight request."""
        task, self._reader_task = self._reader_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._close_transport()
        self._fail_pending(DaemonUnavailable("The connection was closed"))

    # -- requests -------------------------------------------------------------------------

    async def call(self, method: str, **params: Any) -> Any:
        """Send a request and await its response.

        Raises:
            DaemonUnavailable: If not connected, or the connection drops mid-request.
            RemoteError: If the daemon returns an error.
        """
        if self._writer is None:
            raise DaemonUnavailable("Not connected to a daemon")

        self._next_id += 1
        request = Request(id=self._next_id, method=str(method), params=params)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request.id] = future

        try:
            await write_message(self._writer, request.to_json())
        except (OSError, ProtocolError) as exc:
            self._pending.pop(request.id, None)
            raise DaemonUnavailable(f"Could not send {method}: {exc}") from exc

        return await future

    async def subscribe(self, topics: Sequence[str] = ()) -> None:
        """Subscribe to event topics, remembering them across reconnects."""
        wanted = [str(t) for t in (topics or list(Topic))]
        self._topics = set(wanted)
        await self.call(Method.SUBSCRIBE, topics=wanted)

    async def _read_loop(self) -> None:
        """Dispatch incoming messages to waiting futures or the event callback."""
        assert self._reader is not None
        try:
            while True:
                message = await read_message(self._reader)
                if message is None:
                    break
                self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except (ProtocolError, OSError) as exc:
            log.debug("Client read loop ended: %s", exc)
        finally:
            # Deliberately does not clear `_writer`: `close()` owns the transport, and
            # dropping the reference here is what leaves a socket to be reaped by the
            # garbage collector instead. `connected` already reports False once the
            # writer is closing.
            self._fail_pending(DaemonUnavailable("The daemon closed the connection"))
            if self._on_state_change:
                self._on_state_change(False)

    def _dispatch(self, message: dict[str, Any]) -> None:
        kind = message.get("t")
        if kind == "res":
            future = self._pending.pop(int(message.get("id", -1)), None)
            if future is None or future.done():
                return
            if message.get("ok"):
                future.set_result(message.get("result"))
            else:
                error = message.get("error") or {}
                future.set_exception(
                    RemoteError(
                        str(error.get("code", "ERROR")),
                        str(error.get("message", "The daemon reported an error")),
                        error.get("detail"),
                    )
                )
        elif kind == "evt" and self._on_event:
            self._on_event(
                Notification(event=str(message.get("event", "")), data=message.get("data") or {})
            )

    def _fail_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    # -- reconnection -----------------------------------------------------------------------

    async def watch_connection(
        self, *, initial_delay: float = 0.5, max_delay: float = 10.0
    ) -> AsyncIterator[bool]:
        """Keep the connection alive, yielding ``True`` on every successful (re)connect.

        Backoff is capped rather than unbounded: a daemon that is down for an hour should
        be retried every ten seconds, not once more at the end of the hour.
        """
        delay = initial_delay
        while True:
            if self.connected:
                await asyncio.sleep(initial_delay)
                continue
            try:
                await self.connect()
            except DispatchError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            delay = initial_delay
            if self._topics:
                with contextlib.suppress(DispatchError):
                    await self.call(Method.SUBSCRIBE, topics=sorted(self._topics))
            yield True


def _client_version() -> str:
    from dispatch.version import __version__

    return __version__


def default_socket_path() -> Path:
    """The socket path from configuration, for callers that have no config loaded."""
    from dispatch.core.config import load_config

    return load_config().paths.socket


def socket_exists(path: Path) -> bool:
    """Whether a socket file is present, cheaply and without connecting."""
    try:
        return path.exists()
    except OSError:  # pragma: no cover - permission oddities
        return False
