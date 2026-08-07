"""The colour palette, in one place.

Colour reaches the screen by two routes: Textual CSS, and Rich ``Text`` styles built in
Python. If those two drifted apart the interface would be subtly inconsistent in ways that
are miserable to chase, so both read from the constants here — the CSS via a
:class:`~textual.theme.Theme` whose variables the stylesheet references, and the
renderables via :data:`STATE_STYLES` and friends.

The palette is VS Code's Dark Modern. It is deliberately narrow: two greys for text, one
blue for selection, and three semantic colours used only where they carry meaning. Nothing
here is fully saturated, because a long SSH session over a mediocre connection is not the
place for neon.
"""

from __future__ import annotations

from typing import Final

from textual.theme import Theme

__all__ = ["DISPATCH_THEME", "SEVERITY_STYLES", "STATE_STYLES", "Palette"]


class Palette:
    """Named colours."""

    BACKGROUND: Final = "#282828"
    SURFACE: Final = "#32302f"
    ELEVATED: Final = "#3c3836"

    BORDER: Final = "#504945"

    TEXT: Final = "#ebdbb2"
    MUTED: Final = "#bdae93"
    FAINT: Final = "#928374"

    # Selection should be warm brown instead of VS Code blue
    SELECTION: Final = "#665c54"

    # Muted teal accent (Gruvbox Material)
    ACCENT: Final = "#7daea3"
    ACCENT_TEXT: Final = "#89b482"

    SUCCESS: Final = "#a9b665"
    WARNING: Final = "#d8a657"
    ERROR: Final = "#ea6962"

    # Info is intentionally no longer bright blue
    INFO: Final = "#89b482"

DISPATCH_THEME: Final = Theme(
    name="dispatch-dark",
    dark=True,
    primary=Palette.ACCENT,
    secondary=Palette.ACCENT_TEXT,
    accent=Palette.ACCENT_TEXT,
    background=Palette.BACKGROUND,
    surface=Palette.SURFACE,
    panel=Palette.SURFACE,
    foreground=Palette.TEXT,
    success=Palette.SUCCESS,
    warning=Palette.WARNING,
    error=Palette.ERROR,
    # Textual derives a great many variables from the base colours above. The handful that
    # matter to this design are pinned rather than derived, so the stylesheet and the Rich
    # renderables cannot disagree about, say, exactly which blue a selected row is.
    variables={
        "border": Palette.BORDER,
        "border-blurred": Palette.BORDER,
        "text": Palette.TEXT,
        "text-muted": Palette.MUTED,
        "text-disabled": Palette.FAINT,
        "surface-lighten-1": Palette.ELEVATED,
        "panel-lighten-1": Palette.ELEVATED,
        "block-cursor-background": Palette.SELECTION,
        "block-cursor-foreground": Palette.TEXT,
        "block-cursor-text-style": "none",
        "block-cursor-blurred-background": Palette.SURFACE,
        "block-cursor-blurred-foreground": Palette.TEXT,
        "block-cursor-blurred-text-style": "none",
        "input-cursor-background": Palette.TEXT,
        "input-cursor-foreground": Palette.BACKGROUND,
        "input-selection-background": f"{Palette.SELECTION} 60%",
        "scrollbar": Palette.BORDER,
        "scrollbar-hover": Palette.FAINT,
        "scrollbar-active": Palette.FAINT,
        "scrollbar-background": Palette.BACKGROUND,
        "scrollbar-corner-color": Palette.BACKGROUND,
        "scrollbar-background-hover": Palette.BACKGROUND,
        "scrollbar-background-active": Palette.BACKGROUND,
        "footer-key-foreground": Palette.ACCENT_TEXT,
        "footer-description-foreground": Palette.MUTED,
        "footer-background": Palette.SURFACE,
        "footer-key-background": Palette.SURFACE,
        "footer-description-background": Palette.SURFACE,
    },
)
"""The application theme. Registered and selected in :class:`~dispatch.tui.app.DispatchApp`."""


STATE_STYLES: Final[dict[str, str]] = {
    "RUNNING": Palette.SUCCESS,
    "PREPARING": Palette.ACCENT_TEXT,
    "QUEUED": Palette.TEXT,
    "HELD": Palette.WARNING,
    "COMPLETED": Palette.MUTED,
    "FAILED": Palette.ERROR,
    "CANCELLED": Palette.WARNING,
    "REJECTED": Palette.ERROR,
    "UNKNOWN": Palette.WARNING,
}
"""Colour per job state.

``COMPLETED`` is deliberately the *muted* grey rather than a success green. A finished
job is the resting state of almost every row in the history, and colouring the common case
means colouring everything. Green is reserved for what is happening now.
"""

SEVERITY_STYLES: Final[dict[str, str]] = {
    "ERROR": Palette.ERROR,
    "WARNING": Palette.WARNING,
    "INFO": Palette.INFO,
}
"""Colour per validation severity."""


def state_style(state: str) -> str:
    """The colour for a job state, falling back to muted for one we do not know.

    History written by a future version must render, not raise.
    """
    return STATE_STYLES.get(state, Palette.MUTED)


def severity_style(severity: str) -> str:
    """The colour for a validation severity."""
    return SEVERITY_STYLES.get(severity.upper(), Palette.TEXT)


def usage_style(percent: float) -> str:
    """Colour a utilisation figure by how close to saturated it is.

    Three bands rather than a gradient: below 70% is unremarkable and should not draw the
    eye at all, so it takes the muted grey rather than a colour.
    """
    if percent >= 90:
        return Palette.ERROR
    if percent >= 70:
        return Palette.WARNING
    return Palette.MUTED
