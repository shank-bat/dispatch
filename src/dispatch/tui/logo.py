"""The Dispatch logo: the bird, and the wordmark beside it.

On a terminal confirmed to speak the kitty graphics protocol, both are drawn: the real
``logo.png`` artwork beside the ``logo.txt`` block-letter wordmark. Everywhere else, the
wordmark stands alone. There used to be a third tier -- an ASCII/block-art rendering of the
bird for any terminal with plain colour and no graphics protocol -- and it is gone; see
:func:`_bird` for why.

Why the image is not simply printed
-----------------------------------
Graphics escape codes are not text, and Textual owns the screen: it composites widgets,
repaints damage and scrolls. An image written into a widget's text would be escaped, or
drawn once and painted over by the next frame. So the kitty path uses **unicode
placeholders**, which exist for this: the picture is transmitted once with ``U=1`` (stored,
displaying nothing), and the widget then renders ordinary characters that name a row and
column of it. The terminal composites the picture behind those cells itself, so resizing,
scrolling and repainting all behave normally.

Nothing here is required. An unreadable asset, or a terminal that cannot be identified,
ends with the wordmark alone -- which is plain text, and cannot render badly.
"""

from __future__ import annotations

import base64
import contextlib
import os
import selectors
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Final, Protocol

from rich.style import Style
from rich.text import Text

from dispatch.tui.theme import Palette

__all__ = ["LOGO_ROWS", "Logo", "graphics_supported", "load", "prime"]

LOGO_ROWS: Final = 10
"""Height of the header's logo block, in cells.

The wordmark is centred against it, and the chrome sizes ``#logo`` to match (see
``dispatch.tcss``), so this is the one number to change to resize the mark.
"""

BIRD_COLUMNS: Final = round(LOGO_ROWS * 1.6)
"""Width of the bird, in cells.

Terminal cells are about twice as tall as they are wide, and the artwork is a little taller
than it is wide, so the two factors give roughly 1.6 columns per row. Without this the bird
comes out smeared.
"""

GAP: Final = 2
"""Blank columns between the bird and the wordmark."""

PLACEHOLDER: Final = "\U0010eeee"
"""The cell kitty composites an image into. Ordinary text as far as Textual is concerned."""

IMAGE_ID: Final = 0x445A11
"""Identifier for the transmitted logo.

Deliberately under 24 bits: kitty reads the id from the cell's foreground colour, and an id
needing a fourth byte has to smuggle it through an extra diacritic. Every byte is non-zero
so no colour channel is absent.
"""

ROWCOLUMN_DIACRITICS: Final = (
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F,
    0x0346, 0x034A, 0x034B, 0x034C, 0x0350, 0x0351, 0x0352, 0x0357,
)
"""Diacritics naming a row or column index, in kitty's order.

The first sixteen, which is all a logo this size can use. Read off the bytes kitty's own
``icat`` emits rather than copied from a table, so they are known rather than believed.
"""

GRAPHICS_TERMINALS: Final = frozenset({"ghostty", "wezterm", "kitty", "wayst"})
"""``TERM_PROGRAM`` values whose terminals implement the kitty graphics protocol.

Konsole is deliberately not in this set -- see :data:`DENIED_TERMINALS`.
"""

DENIED_TERMINALS: Final = frozenset({"konsole"})
"""Terminals confirmed *not* to work, checked before anything else gets a chance to guess.

Konsole answers the capability query in :func:`_query_terminal` with an acknowledgement --
it does implement image transmission -- but does not composite the Unicode placeholder
cells :class:`KittyBird` actually depends on. Left to the query, it reports a false
positive and the result is not a missing bird but a visible one: a grid of the placeholder
glyphs themselves, tinted in :data:`IMAGE_ID`'s colour, on top of the artwork. That is
worse than the plain-wordmark fallback ever is, which is why this is checked ahead of the
query rather than left to it.
"""

PROBE_TIMEOUT_S: Final = 0.25
"""How long to wait for a terminal to answer the capability query.

Generous locally and still imperceptible. A terminal silent by then is treated as having no
graphics, which costs a fallback rather than a hang.
"""


class Bird(Protocol):
    """The picture, however this terminal can manage it."""

    columns: int

    def prepare(self) -> None:
        """One-time work before the cells are drawn."""

    def row(self, index: int) -> Text:
        """One row of the picture."""


@dataclass(frozen=True, slots=True)
class NoBird:
    """No usable image. The wordmark stands alone."""

    columns: int = 0

    def prepare(self) -> None:
        return None

    def row(self, index: int) -> Text:
        return Text()


@dataclass(frozen=True, slots=True)
class KittyBird:
    """The image itself, composited by the terminal behind placeholder cells."""

    payload: bytes
    columns: int

    def prepare(self) -> None:
        """Transmit the image once, stored against :data:`IMAGE_ID`.

        Displays nothing -- ``U=1`` makes the placement virtual -- so this is safe to write
        while Textual owns the screen. ``q=2`` suppresses the acknowledgement, which would
        otherwise arrive as keyboard input.
        """
        _transmit(self.payload, columns=self.columns, rows=LOGO_ROWS)

    def row(self, index: int) -> Text:
        """One placeholder per cell, coloured with the image id.

        The colour carries the id and the diacritics carry the cell's row and column, so
        each character tells the terminal which piece of the picture it is holding.
        """
        # Style(colour=...) parses a string; Style.from_color expects an already-parsed
        # Color and stores whatever it is given, which leaves a str where Textual later
        # expects a colour object.
        style = Style(
            color=_rich_colour(((IMAGE_ID >> 16) & 0xFF, (IMAGE_ID >> 8) & 0xFF, IMAGE_ID & 0xFF))
        )
        cells = "".join(
            PLACEHOLDER + chr(ROWCOLUMN_DIACRITICS[index]) + chr(ROWCOLUMN_DIACRITICS[column])
            for column in range(self.columns)
        )
        return Text(cells, style=style)


@dataclass(frozen=True, slots=True)
class Logo:
    """The bird and the wordmark, side by side."""

    bird: Bird
    wordmark: tuple[str, ...]
    width: int
    height: int

    def prepare(self) -> None:
        """Hand the terminal whatever it needs before the first draw."""
        self.bird.prepare()

    def render(self) -> Text:
        """The whole mark, as one block Textual can put in a widget.

        The wordmark is centred against the bird rather than sitting on the top line, so
        the two read as one mark instead of as a picture with a caption.
        """
        offset = max(0, (self.height - len(self.wordmark)) // 2)
        text = Text()
        for row in range(self.height):
            if row:
                text.append("\n")
            if self.bird.columns:
                text.append_text(self.bird.row(row))
                text.append(" " * GAP)
            index = row - offset
            if 0 <= index < len(self.wordmark):
                text.append(self.wordmark[index], style=Palette.TEXT)
        return text


def prime() -> None:
    """Do the slow and the delicate work before Textual takes the screen.

    Asking the terminal what it supports means writing to it and reading the reply, which
    is only safe while nothing else owns it -- hence doing it here, before ``load()`` is
    ever called from inside a mounted screen.
    """
    graphics_supported()
    with contextlib.suppress(Exception):
        load()


@lru_cache(maxsize=1)
def load() -> Logo:
    """The mark this terminal should draw, built once per process."""
    wordmark = _wordmark()
    bird = _bird()
    width = (bird.columns + GAP if bird.columns else 0) + max(
        (len(line) for line in wordmark), default=0
    )
    return Logo(bird=bird, wordmark=wordmark, width=width, height=LOGO_ROWS)


@lru_cache(maxsize=1)
def graphics_supported() -> bool:
    """Whether this terminal can draw kitty graphics.

    The question is about the *terminal*, not about what is installed: the protocol is
    implemented by several terminals and by none of the binaries on ``PATH``. So the
    environment is read for terminals known to implement it, and anything unrecognised is
    asked directly.

    Conservative in two places. Under a multiplexer the escape codes need a passthrough
    that may not be configured, and a wrong answer there is a screenful of replacement
    glyphs; and any failure resolves to ``False``. Guessing wrong downwards costs the plain
    wordmark instead of the bird, which is not worth a risk.
    """
    env = os.environ
    if not sys.__stdout__ or not sys.__stdout__.isatty():
        return False
    if env.get("TMUX") or env.get("STY") or env.get("TERM", "").startswith("screen"):
        return False
    if env.get("KONSOLE_VERSION") or env.get("TERM_PROGRAM", "").lower() in DENIED_TERMINALS:
        return False
    if env.get("KITTY_WINDOW_ID") or "kitty" in env.get("TERM", "").lower():
        return True
    if env.get("TERM_PROGRAM", "").lower() in GRAPHICS_TERMINALS:
        return True
    if env.get("WEZTERM_PANE"):
        return True
    # Nothing names a terminal known either way, which is the normal case over SSH where
    # TERM describes the far end rather than the near one. Ask it.
    return _query_terminal()


def _bird() -> Bird:
    """The picture, on a terminal confirmed to render it properly. The wordmark alone
    everywhere else.

    There used to be a middle tier here: the PNG reduced to coloured block characters, for
    any terminal with plain colour and no graphics protocol. It is gone. On at least one
    real terminal (Konsole) the safest available glyph for it turned out not to be safe
    either -- a single flat ``█`` per cell renders crisply, no anti-aliasing, but at the
    resolution that fits beside the wordmark it reads as a coarse, blocky mess rather than
    a bird, which is its own kind of wrong answer. Rather than chase a rendering that works
    acceptably on some unknown subset of "any terminal with colour", the wordmark by itself
    is what every terminal not confirmed to speak kitty graphics gets. It is plain text; it
    cannot render badly.
    """
    try:
        payload = _asset_bytes("logo.png")
    except OSError:
        return NoBird()
    if not payload:
        return NoBird()

    if graphics_supported():
        return KittyBird(payload=payload, columns=BIRD_COLUMNS)
    return NoBird()


def _rich_colour(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


# -- assets ---------------------------------------------------------------------------------


def _asset_bytes(name: str) -> bytes:
    """Read a packaged asset.

    Through ``importlib.resources`` rather than path arithmetic on ``__file__``, so the
    assets are found the same way from a source tree, an installed wheel or a zip.
    """
    return (resources.files("dispatch.tui") / name).read_bytes()


def _wordmark() -> tuple[str, ...]:
    """``logo.txt``, padded to a rectangle so the block has one width."""
    try:
        raw = _asset_bytes("logo.txt").decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return ("dispatch",)
    lines = [line.rstrip("\n") for line in raw.splitlines() if line.strip()]
    if not lines:
        return ("dispatch",)
    width = max(len(line) for line in lines)
    return tuple(line.ljust(width) for line in lines)


# -- talking to the terminal ------------------------------------------------------------------


def _write_terminal(data: str) -> bool:
    """Write an escape sequence straight to the terminal. Returns whether it landed.

    Straight to the real stdout rather than through Textual, because this is not content:
    it is addressed to the emulator, produces no output, and must not be laid out.
    """
    stream = sys.__stdout__
    if stream is None:
        return False
    try:
        stream.write(data)
        stream.flush()
    except (OSError, ValueError):
        return False
    return True


def _transmit(payload: bytes, *, columns: int, rows: int) -> None:
    """Send the image in chunks, as the protocol requires.

    Only the first chunk carries the parameters; the rest carry ``m`` alone, which says
    whether more is coming. A failed write is dropped -- the placeholders then have nothing
    behind them, and the next start tries again.
    """
    encoded = base64.b64encode(payload).decode("ascii")
    chunk_size = 4096
    chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)] or [""]
    header = (
        f"a=T,U=1,q=2,f=100,t=d,i={IMAGE_ID},c={columns},r={rows},"
        f"m={1 if len(chunks) > 1 else 0}"
    )
    if not _write_terminal(f"\x1b_G{header};{chunks[0]}\x1b\\"):
        return
    for index, chunk in enumerate(chunks[1:], start=2):
        more = 1 if index < len(chunks) else 0
        if not _write_terminal(f"\x1b_Gm={more};{chunk}\x1b\\"):
            return


def _query_terminal() -> bool:
    """Ask the terminal whether it speaks the graphics protocol, and wait briefly.

    Sends a one-pixel query alongside a plain device-attributes request. Every terminal
    answers the second; only one with graphics answers the first. So a reply carrying no
    graphics acknowledgement is a definite *no* rather than a timeout, and the wait ends as
    soon as the attributes arrive.

    Raw mode is restored in a ``finally``: leaving a user's shell without an echo would be a
    far worse bug than a missing logo.
    """
    try:
        import termios
        import tty
    except ImportError:  # pragma: no cover - not POSIX
        return False

    stdin = sys.__stdin__
    if stdin is None or not stdin.isatty():
        return False
    try:
        fd = stdin.fileno()
        saved = termios.tcgetattr(fd)
    except (OSError, ValueError, termios.error):
        return False

    try:
        tty.setraw(fd)
        if not _write_terminal(f"\x1b_Gi={IMAGE_ID},s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\\x1b[c"):
            return False
        return _await_graphics_reply(fd)
    except (OSError, ValueError, termios.error):
        return False
    finally:
        with contextlib.suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _await_graphics_reply(fd: int) -> bool:
    """Read until the device attributes arrive; say whether graphics were acknowledged."""
    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_READ)
    reply = b""
    try:
        while selector.select(PROBE_TIMEOUT_S):
            reply += os.read(fd, 1024)
            if b"\x1b[?" in reply and b"c" in reply:
                break
            if len(reply) > 4096:
                break
    except OSError:
        return False
    finally:
        selector.close()
    return b"_Gi=" in reply and b";OK" in reply
