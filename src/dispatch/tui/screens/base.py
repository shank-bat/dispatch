"""Shared behaviour and chrome for every screen.

Holds the pieces each screen would otherwise reimplement: access to the cached state, the
top bar, the disconnection notice, and the confirmation prompt used before anything
destructive.

All styling lives in ``dispatch.tcss``. What is built here is *structure* — which regions
exist and what they contain — and the small amount of Rich text where colour carries
meaning that CSS cannot express, such as a job state.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Footer, Label, Static

from dispatch.tui.theme import Palette

if TYPE_CHECKING:
    from dispatch.tui.app import DispatchApp
    from dispatch.tui.state import AppState

__all__ = ["ConfirmScreen", "DispatchScreen", "run_when_confirmed"]

SEPARATOR = "  ·  "
"""Between fields in the top bar.

A middle dot with generous space either side, rather than a pipe. It separates without
drawing a line, which is the whole idea.
"""


class DispatchScreen(Screen[None]):
    """Base for Dispatch's screens.

    Provides a two-part header: a docked top bar that is identical everywhere, and an
    optional heading that names the screen and gives its content some room to breathe.
    """

    NAV = (("1", "dashboard"), ("2", "queue"), ("3", "history"))
    """Screen switches, shown in the top bar so navigation is always visible."""

    nav_key = ""
    """Which entry of :data:`NAV` this screen is. Empty for screens outside the rotation."""

    @property
    def dispatch_app(self) -> DispatchApp:
        """The running application, typed."""
        from dispatch.tui.app import DispatchApp

        assert isinstance(self.app, DispatchApp)
        return self.app

    @property
    def app_state(self) -> AppState:
        """The cached view of the daemon's state."""
        return self.dispatch_app.state

    # -- chrome ---------------------------------------------------------------------

    def compose_header(self) -> Iterator[Widget]:
        """The top bar and the disconnection notice, as one docked block.

        Both live inside a single container rather than being docked individually.
        Textual places every widget docked to the same edge at that edge, so two docked
        siblings overlap -- which meant the notice silently covered the top bar exactly
        when the top bar mattered most.
        """
        chrome = Vertical(id="chrome")
        chrome.compose_add_child(Static("", id="banner"))
        chrome.compose_add_child(Static("", id="topbar"))
        yield chrome

    def compose_footer(self) -> Iterator[Widget]:
        """The key hints."""
        yield Footer()

    def heading(self) -> Text | None:
        """The screen's heading, or ``None`` for screens that supply their own header.

        Kept separate from the top bar so a screen can name itself and give one line of
        context without duplicating the machine status.
        """
        return None

    def topbar_left(self) -> Text:
        """Application identity and navigation.

        The navigation reads as tabs: the current screen in full-strength text, the others
        muted, each preceded by the key that reaches it. Nothing here changes as data
        arrives, so the eye can ignore it once learned.
        """
        text = Text()
        text.append("dispatch", style=f"bold {Palette.TEXT}")
        text.append(SEPARATOR, style=Palette.FAINT)

        for index, (key, label) in enumerate(self.NAV):
            if index:
                text.append("   ", style=Palette.FAINT)
            current = key == self.nav_key
            text.append(key, style=Palette.ACCENT_TEXT if current else Palette.FAINT)
            text.append(" ")
            text.append(label, style=Palette.TEXT if current else Palette.MUTED)
        return text

    def topbar_right(self) -> Text:
        """Live status: host, job counts, and the clock.

        Deliberately terse. Counts appear only when non-zero, so an idle machine shows an
        almost empty bar rather than a row of zeros.
        """
        return self.dispatch_app.status_summary()

    def update_status(self) -> None:
        """Re-render the top bar, heading, and connection notice."""
        try:
            topbar = self.query_one("#topbar", Static)
            banner = self.query_one("#banner", Static)
        except Exception:  # pragma: no cover - screen not mounted yet
            return

        topbar.update(_justify(self.topbar_left(), self.topbar_right(), self.size.width - 4))

        heading = self.heading()
        if heading is not None:
            # Not every screen has a heading region; the log viewer, for one, has its own
            # docked status line and does not need a second one.
            with contextlib.suppress(Exception):
                self.query_one("#heading", Static).update(heading)

        if self.app_state.connected:
            banner.remove_class("visible")
        else:
            banner.update(
                Text(
                    "Daemon unreachable — reconnecting. Running simulations are unaffected.",
                    style=Palette.WARNING,
                )
            )
            banner.add_class("visible")

    async def on_mount(self) -> None:
        """Paint the chrome as soon as the screen exists.

        Subclasses that define their own ``on_mount`` must ``await super().on_mount()``;
        without it a static screen such as the help or plan view would show an empty top
        bar, because nothing else ever asks it to draw. Async so that screens whose setup
        needs the daemon can override it without changing the signature.
        """
        self.update_status()

    def refresh_view(self) -> None:
        """Re-render from cached state. Called when events arrive."""

    # -- feedback ---------------------------------------------------------------------

    def notify_error(self, message: str) -> None:
        """Show an error without disturbing the layout."""
        self.notify(message, severity="error", timeout=8)

    def notify_ok(self, message: str) -> None:
        """Show a confirmation."""
        self.notify(message, timeout=4)


class ConfirmScreen(ModalScreen[bool]):
    """A yes/no prompt.

    Used before cancelling a job or deleting history — the two actions a mistyped key
    should not be able to perform. Everything else is immediate, because confirming
    routine actions trains people to confirm without reading.
    """

    BINDINGS = [
        ("escape,n", "dismiss_no", "cancel"),
        ("y,enter", "dismiss_yes", "confirm"),
    ]

    def __init__(self, question: str, *, detail: str = "") -> None:
        super().__init__()
        self._question = question
        self._detail = detail

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(Text(self._question))
            if self._detail:
                yield Label(Text(self._detail), classes="confirm-detail")
            yield Label(_keys(("y", "confirm"), ("esc", "cancel")), classes="confirm-keys")

    def action_dismiss_yes(self) -> None:
        self.dismiss(True)

    def action_dismiss_no(self) -> None:
        self.dismiss(False)


def run_when_confirmed(
    screen: DispatchScreen, question: str, action: Callable[[], Any], *, detail: str = ""
) -> None:
    """Ask, then run ``action`` if the answer is yes."""

    def _answered(confirmed: bool | None) -> None:
        if confirmed:
            action()

    screen.app.push_screen(ConfirmScreen(question, detail=detail), _answered)


# -- text helpers -------------------------------------------------------------------


def _justify(left: Text, right: Text, width: int) -> Text:
    """Place ``right`` against the right edge, or drop it if the terminal is too narrow.

    Dropping beats wrapping: the top bar is one line by definition, and a wrapped bar
    would push every screen's content down by a row on a narrow terminal.
    """
    gap = width - left.cell_len - right.cell_len
    if gap < 2:
        return left
    return Text.assemble(left, " " * gap, right)


def _keys(*pairs: tuple[str, str]) -> Text:
    """Render key hints as ``key label`` groups, keys in the accent colour."""
    text = Text()
    for index, (key, label) in enumerate(pairs):
        if index:
            text.append("   ")
        text.append(key, style=Palette.ACCENT_TEXT)
        text.append(f" {label}", style=Palette.MUTED)
    return text
