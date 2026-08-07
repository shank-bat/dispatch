"""The key reference."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from dispatch.tui.screens.base import DispatchScreen
from dispatch.tui.theme import Palette
from dispatch.version import __version__

__all__ = ["HelpScreen"]

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Anywhere",
        [
            ("1", "dashboard"),
            ("2", "queue"),
            ("3", "history"),
            ("n", "new job"),
            ("?", "this help"),
            ("q", "quit — simulations keep running"),
            ("ctrl+c", "quit"),
        ],
    ),
    (
        "Moving around",
        [
            ("j / down", "next row"),
            ("k / up", "previous row"),
            ("g / G", "first / last row"),
            ("enter", "open the job's logs"),
            ("escape", "go back"),
        ],
    ),
    (
        "Queue",
        [
            ("x", "cancel the job"),
            ("h / H", "hold / release"),
            ("+ / -", "raise / lower priority"),
        ],
    ),
    (
        "History",
        [
            ("/", "search"),
            ("t", "edit tags"),
            ("n", "add a note"),
            ("d", "delete a finished job"),
            ("r", "re-run the search"),
        ],
    ),
    (
        "Submitting",
        [
            ("enter", "open a directory"),
            ("backspace", "go up"),
            ("p", "type a path"),
            ("c", "set cores"),
            ("t", "set tags"),
            ("d", "preview the plan"),
            ("s", "submit"),
            ("f", "submit despite errors"),
            (".", "show hidden"),
        ],
    ),
    (
        "Log viewer",
        [
            ("e", "stdout / stderr"),
            ("/", "search the buffer"),
            ("n / N", "next / previous match"),
            ("G", "follow new output"),
            ("w", "toggle line wrapping"),
        ],
    ),
]

SEARCH_HELP = """\
naca0018                 anywhere in the name, directory, notes, or metadata
tag:paper  -tag:scratch  has, or lacks, a tag
solver:openfoam          the adapter that ran it
app:interFoam            the specific application
state:failed             job state
cores>=16  runtime>2h    numeric comparisons, with units
endTime>500              any metadata field the adapter declared
after:7d  before:2026-06 date ranges, absolute or relative
dirty:true               launched from a repository with uncommitted changes
"""


class HelpScreen(DispatchScreen):
    """Every key, and the search syntax."""

    TITLE = "Help"

    BINDINGS = [Binding("escape,q,question_mark", "back", "back")]

    nav_key = ""

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        yield Static("", id="heading")
        yield Static(self._body(), id="help-body")
        yield from self.compose_footer()

    def _body(self) -> Table:
        """Two columns of key groups, then the search syntax.

        Keys are right-aligned against their descriptions so the gap between the two is
        constant regardless of key length -- which is what makes a long list scannable.
        """
        outer = Table.grid(padding=(0, 8))
        outer.add_column()
        outer.add_column()

        # Split by accumulated height rather than alternating, so the two columns end at
        # roughly the same line instead of one running far past the other.
        columns = [_column(), _column()]
        heights = [0, 0]
        for title, bindings in SECTIONS:
            index = 0 if heights[0] <= heights[1] else 1
            heights[index] += len(bindings) + 2
            target = columns[index]
            if target.row_count:
                target.add_row("", "")
            target.add_row("", Text(title, style=Palette.MUTED))
            for key, description in bindings:
                target.add_row(
                    Text(key, style=Palette.ACCENT_TEXT),
                    Text(description, style=Palette.TEXT),
                )

        outer.add_row(*columns)

        body = Table.grid()
        body.add_column()
        body.add_row(outer)
        body.add_row("")
        body.add_row(Text("search", style=Palette.MUTED))
        body.add_row(Text(SEARCH_HELP.rstrip("\n"), style=Palette.FAINT))
        return body

    def heading(self) -> Text:
        text = Text("keys", style=f"bold {Palette.TEXT}")
        text.append(f"   dispatch {__version__}", style=Palette.FAINT)
        return text

    def action_back(self) -> None:
        self.dismiss()


def _column() -> Table:
    """One half of the two-column key list: keys right-aligned against their descriptions."""
    column = Table.grid(padding=(0, 2))
    column.add_column(justify="right", width=14)
    column.add_column()
    return column
