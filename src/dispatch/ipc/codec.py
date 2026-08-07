"""Framing and serialisation for the NDJSON protocol.

One JSON object per line. ``json.dumps`` escapes embedded newlines, so ``\\n`` is an
unambiguous frame delimiter and the format stays readable in a terminal.

Both directions enforce a size ceiling. An unbounded reader on a long-lived daemon is how
a malformed or hostile client turns into an out-of-memory kill at 3am, and the ceiling
costs one comparison per message.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from dispatch.core.errors import DispatchError
from dispatch.ipc.protocol import MAX_MESSAGE_BYTES

__all__ = ["ProtocolError", "encode_line", "read_message", "write_message"]


class ProtocolError(DispatchError):
    """A message could not be framed, parsed, or was too large."""

    code = "PROTOCOL_ERROR"


def encode_line(payload: Any) -> bytes:
    """Serialise one message to a single NDJSON line.

    ``ensure_ascii=False`` keeps case names with non-ASCII characters readable on the
    wire; UTF-8 is the only encoding either side uses.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Cannot serialise message: {exc}") from exc
    data = text.encode("utf-8") + b"\n"
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            f"Message is {len(data)} bytes; the limit is {MAX_MESSAGE_BYTES}. "
            "Use a smaller page size.",
            detail={"size": len(data), "limit": MAX_MESSAGE_BYTES},
        )
    return data


async def write_message(writer: asyncio.StreamWriter, payload: Any) -> None:
    """Write one message and flush it."""
    writer.write(encode_line(payload))
    await writer.drain()


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one message.

    Returns:
        The decoded object, or ``None`` at clean end of stream.

    Raises:
        ProtocolError: On an oversized line, malformed JSON, or a JSON value that is not
            an object. Each is a distinct, reportable client bug rather than something to
            paper over.
    """
    try:
        line = await reader.readuntil(b"\n")
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise ProtocolError("Connection closed mid-message") from exc
    except asyncio.LimitOverrunError as exc:
        raise ProtocolError(
            f"Message exceeds the {MAX_MESSAGE_BYTES} byte limit before a newline appeared"
        ) from exc
    except (ConnectionResetError, BrokenPipeError):
        return None

    if not line:
        return None
    if len(line) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"Message is {len(line)} bytes; the limit is {MAX_MESSAGE_BYTES}")

    try:
        parsed = json.loads(line)
    except ValueError as exc:
        raise ProtocolError(f"Malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ProtocolError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed
