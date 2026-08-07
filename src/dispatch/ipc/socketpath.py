"""Binding and connecting to Unix sockets whose paths are too long.

``sockaddr_un.sun_path`` is a fixed 108-byte buffer on Linux. A socket under a deeply
nested home directory -- or under a test's temporary directory -- overflows it, and the
failure is an opaque ``OSError: AF_UNIX path too long`` from inside asyncio.

The fix is the standard one: ``bind`` and ``connect`` resolve relative paths against the
current working directory, so changing directory to the socket's parent and using its bare
name keeps the path far under the limit. The directory change is held only for the syscall
itself, under a lock, and restored afterwards.

The alternative -- telling the user to move their home directory -- is not a fix.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

__all__ = ["MAX_SOCKET_PATH", "connect_unix", "is_too_long", "start_unix_server"]

MAX_SOCKET_PATH = 100
"""Conservative ceiling. The kernel's limit is 108 including the NUL terminator; leaving
headroom avoids depending on exactly how the runtime counts it.
"""

_CHDIR_LOCK = threading.Lock()


def is_too_long(path: Path | str) -> bool:
    """Whether this path will not fit in a ``sockaddr_un``."""
    return len(str(path).encode()) >= MAX_SOCKET_PATH


@contextlib.contextmanager
def _in_directory(directory: Path) -> Iterator[None]:
    """Temporarily change the working directory, under a lock.

    ``chdir`` is process-global, so the window is kept to a single syscall and serialised.
    Both users of this -- one bind at daemon startup, one connect per client -- are brief
    and infrequent.
    """
    with _CHDIR_LOCK:
        previous = Path.cwd()
        os.chdir(directory)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                os.chdir(previous)


async def start_unix_server(
    handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Any], path: Path
) -> asyncio.Server:
    """Serve on a Unix socket, working around the path length limit.

    The socket is still created at ``path``; only the string handed to ``bind`` is short.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_too_long(path):
        return await asyncio.start_unix_server(handler, path=str(path))

    with _in_directory(path.parent):
        return await asyncio.start_unix_server(handler, path=path.name)


async def connect_unix(path: Path) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to a Unix socket, working around the path length limit."""
    if not is_too_long(path):
        return await asyncio.open_unix_connection(str(path))

    with _in_directory(path.parent):
        return await asyncio.open_unix_connection(path.name)
