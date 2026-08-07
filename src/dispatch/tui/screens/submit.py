"""The submit wizard: browse, detect, validate, confirm.

Dispatch never guesses where your projects are, so this starts with a directory browser
rather than a path prompt. Directories that adapters recognise are marked as you browse,
and the marking comes from the daemon -- the TUI has no idea what a case looks like.

When detection is unambiguous the solver is shown, never asked about. The only question
the wizard asks about solvers is when two adapters claim a directory equally well.

Before submitting, ``d`` shows the actual execution plan -- the same plan the executor
will run, not a description of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, ListItem, ListView, Static

from dispatch.ipc.protocol import Method
from dispatch.tui.screens.base import DispatchScreen
from dispatch.tui.theme import Palette, severity_style

__all__ = ["SubmitScreen"]


class SubmitScreen(DispatchScreen):
    """Choose a case, then queue it."""

    TITLE = "Submit"

    BINDINGS = [
        Binding("escape", "back", "back"),
        Binding("backspace,left", "up", "parent"),
        Binding("enter,right", "enter", "open"),
        Binding("d", "dry_run", "plan"),
        Binding("s", "submit", "submit"),
        Binding("f", "force_submit", "force"),
        Binding("c", "edit_cores", "cores"),
        Binding("t", "edit_tags", "tags"),
        Binding("p", "edit_path", "go to path"),
        Binding("period", "toggle_hidden", "hidden"),
    ]

    def __init__(self, start: Path | None = None) -> None:
        super().__init__()
        self.path = (start or Path.home()).expanduser().resolve()
        self.entries: list[dict[str, Any]] = []
        self.cores = 1
        self.tags: list[str] = []
        self.show_hidden = False
        self._inspection: dict[str, Any] | None = None
        self._prompt_mode = ""

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical():
            yield Static("", id="path-line")
            with Horizontal(id="panes"):
                with Vertical(id="browser"):
                    yield ListView(id="entries")
                with Vertical(id="detail"):
                    yield Static("", id="detail-text")
            yield Input(id="prompt", classes="prompt")
        yield from self.compose_footer()

    async def on_mount(self) -> None:
        await super().on_mount()
        self.cores = max(1, int(self.app_state.snapshot.get("free_cores", 1)) or 1)
        await self.load()
        self.query_one("#entries", ListView).focus()

    # -- browsing ---------------------------------------------------------------------

    async def load(self) -> None:
        """List the current directory, marking recognised cases."""
        try:
            listing = await self.dispatch_app.call(
                Method.FS_LIST, path=str(self.path), hidden=self.show_hidden
            )
        except Exception as exc:
            self.notify_error(str(exc))
            return

        self.entries = listing["entries"]
        view = self.query_one("#entries", ListView)
        await view.clear()

        for entry in self.entries:
            label = Text()
            if entry["case"]:
                # A dot in the gutter, the way a file tree marks modified files. It reads
                # as an annotation on the row rather than as a differently-styled row.
                label.append("● ", style=Palette.SUCCESS)
                label.append(entry["name"], style=Palette.TEXT)
                label.append(f"  {entry['case']}", style=Palette.FAINT)
            else:
                label.append("  ")
                label.append(entry["name"], style=Palette.MUTED)
            view.append(ListItem(Label(label)))

        self.query_one("#path-line", Static).update(_path_text(self.path))
        await self.inspect(listing.get("case"))

    async def inspect(self, detected: str | None) -> None:
        """Validate the current directory, if it is a case."""
        self._inspection = None
        if not detected:
            self._render_detail()
            return
        try:
            self._inspection = await self.dispatch_app.call(
                Method.CASE_VALIDATE, path=str(self.path), cores=self.cores
            )
        except Exception as exc:
            self._inspection = {"error": str(exc)}
        self._render_detail()

    def _render_detail(self) -> None:
        """Render the right-hand pane: what this directory is and whether it will run.

        A definition list of aligned label/value pairs, then findings, then the case's own
        settings. Labels are faint and left-aligned to a fixed column so the eye can run
        down the values without re-finding the left edge on every line.
        """
        text = Text()
        result = self._inspection

        if result is None:
            text.append("no case here\n\n", style=Palette.MUTED)
            text.append(
                "Browse into a directory containing a case.\n\n", style=Palette.FAINT
            )
            text.append(_keyline(("enter", "open"), ("backspace", "up"), ("p", "path")))
            self.query_one("#detail-text", Static).update(text)
            return

        if "error" in result:
            text.append("could not inspect this directory\n\n", style=Palette.ERROR)
            text.append(result["error"], style=Palette.MUTED)
            self.query_one("#detail-text", Static).update(text)
            return

        detected = result["solver"]
        if result.get("solver_binary"):
            detected += f"  {result['solver_binary']}"
        _row(text, "solver", detected)

        free = self.app_state.snapshot.get("free_cores")
        _row(
            text,
            "cores",
            str(self.cores),
            note=f"{free} free" if free is not None else "",
        )
        if self.tags:
            _row(text, "tags", " ".join(self.tags))

        validation = result["validation"]
        _row(
            text,
            "checks",
            validation["summary"],
            style=Palette.SUCCESS if validation["passed"] else Palette.ERROR,
        )

        if validation["findings"]:
            width = self._detail_width()
            text.append("\n")
            for finding in validation["findings"]:
                text.append("  ● ", style=severity_style(finding["severity"]))
                text.append(_wrapped(finding["message"], width, indent=4), style=Palette.TEXT)
                if finding.get("hint"):
                    text.append("    ")
                    text.append(
                        _wrapped(finding["hint"], width, indent=4), style=Palette.FAINT
                    )

        case = (result.get("metadata") or {}).get("case") or {}
        if case:
            text.append("\n")
            for key, value in list(case.items())[:10]:
                _row(text, key, str(value), width=20)

        text.append("\n")
        if validation["passed"]:
            text.append(_keyline(("s", "submit"), ("d", "plan"), ("c", "cores"), ("t", "tags")))
        else:
            text.append(
                _keyline(("f", "submit anyway"), ("d", "plan"), ("c", "cores"))
            )

        self.query_one("#detail-text", Static).update(text)

    def on_resize(self) -> None:
        """Re-wrap the detail pane when its width changes.

        The first render happens before layout has settled, so the pane's width is not yet
        known; without this the findings stay wrapped to the initial guess.
        """
        if self._inspection is not None:
            self._render_detail()

    def _detail_width(self) -> int:
        """Usable width of the detail pane, for hanging-indent wrapping.

        Textual wraps long text on its own, but without a hanging indent -- a wrapped
        bullet then starts back at the margin and reads as a new item. Wrapping here keeps
        continuation lines aligned under the text they continue.
        """
        try:
            return max(30, self.query_one("#detail").size.width - 6)
        except Exception:  # pragma: no cover - before layout settles
            return 52

    def heading(self) -> Text:
        return Text("new job", style=f"bold {Palette.TEXT}")

    # -- navigation -----------------------------------------------------------------------

    async def action_enter(self) -> None:
        view = self.query_one("#entries", ListView)
        index = view.index
        if index is None or index >= len(self.entries):
            return
        self.path = Path(self.entries[index]["path"])
        await self.load()

    async def action_up(self) -> None:
        if self.path.parent != self.path:
            self.path = self.path.parent
            await self.load()

    async def action_toggle_hidden(self) -> None:
        self.show_hidden = not self.show_hidden
        await self.load()

    def action_back(self) -> None:
        self.dismiss()

    # -- editing ---------------------------------------------------------------------------

    def action_edit_cores(self) -> None:
        self._open_prompt("cores", f"cores (currently {self.cores}): ")

    def action_edit_tags(self) -> None:
        self._open_prompt("tags", "tags, space separated: ")

    def action_edit_path(self) -> None:
        self._open_prompt("path", "path: ")

    def _open_prompt(self, mode: str, placeholder: str) -> None:
        self._prompt_mode = mode
        box = self.query_one("#prompt", Input)
        box.placeholder = placeholder
        box.value = ""
        box.add_class("visible")
        box.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        box = self.query_one("#prompt", Input)
        box.remove_class("visible")
        self.query_one("#entries", ListView).focus()

        if self._prompt_mode == "cores" and value:
            try:
                self.cores = max(1, int(value))
            except ValueError:
                self.notify_error(f"{value!r} is not a number")
                return
            await self.inspect(self._inspection.get("solver") if self._inspection else None)
        elif self._prompt_mode == "tags":
            self.tags = value.split()
            self._render_detail()
        elif self._prompt_mode == "path" and value:
            candidate = Path(value).expanduser()
            if candidate.is_dir():
                self.path = candidate.resolve()
                await self.load()
            else:
                self.notify_error(f"{candidate} is not a directory")

    # -- submission -------------------------------------------------------------------------

    async def action_dry_run(self) -> None:
        """Show the execution plan without submitting anything."""
        if self._inspection is None or "error" in self._inspection:
            self.notify_error("This directory is not a recognised case.")
            return
        try:
            report = await self.dispatch_app.call(
                Method.CASE_DRYRUN, workdir=str(self.path), cores=self.cores
            )
        except Exception as exc:
            self.notify_error(str(exc))
            return
        self.app.push_screen(PlanScreen(report))

    async def action_submit(self) -> None:
        await self._submit(force=False)

    async def action_force_submit(self) -> None:
        await self._submit(force=True)

    async def _submit(self, *, force: bool) -> None:
        if self._inspection is None or "error" in self._inspection:
            self.notify_error("This directory is not a recognised case.")
            return
        if not self._inspection["validation"]["passed"] and not force:
            self.notify_error("Validation failed. Press f to queue it anyway.")
            return

        try:
            result = await self.dispatch_app.call(
                Method.JOB_SUBMIT,
                workdir=str(self.path),
                cores=self.cores,
                tags=self.tags,
                force=force,
            )
        except Exception as exc:
            self.notify_error(str(exc))
            return

        job = result["job"]
        self.notify_ok(f"Queued {job['name']} ({job['cores']} cores)")
        self.dismiss()


class PlanScreen(DispatchScreen):
    """The dry-run report: exactly what would run, and nothing was started to find out."""

    TITLE = "Execution plan"

    BINDINGS = [Binding("escape,q,enter", "back", "back")]

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__()
        self.report = report

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        yield Static("", id="heading")
        yield Static(self._body(), id="plan")
        yield from self.compose_footer()

    def _body(self) -> Text:
        """The dry-run report.

        The plan is the point of this screen, so it gets the most space and the only
        emphasis: the solve step is highlighted, preparation steps are muted, and each
        command is shown exactly as it will be run.
        """
        report = self.report
        text = Text()

        _row(text, "case", str(report["workdir"]), width=12)
        if report["detections"]:
            best = report["detections"][0]
            _row(
                text,
                "detected",
                best["label"],
                note=f"confidence {best['confidence']:.2f}",
                width=12,
            )
        _row(text, "cores", str(report["cores"]), width=12)

        validation = report["validation"]
        _row(
            text,
            "checks",
            validation["summary"],
            style=Palette.SUCCESS if validation["passed"] else Palette.ERROR,
            width=12,
        )
        for finding in validation["findings"]:
            text.append("  ● ", style=severity_style(finding["severity"]))
            text.append(f"{finding['message']}\n", style=Palette.TEXT)

        if report["plan"]:
            text.append("\n")
            text.append("plan\n", style=Palette.FAINT)
            for index, step in enumerate(report["plan"]["steps"], start=1):
                solve = step["kind"] == "SOLVE"
                text.append(f"  {index}  ", style=Palette.FAINT)
                text.append(
                    f"{step['kind'].lower():<8}",
                    style=Palette.ACCENT_TEXT if solve else Palette.FAINT,
                )
                text.append(
                    f"{step['command']}\n",
                    style=Palette.TEXT if solve else Palette.MUTED,
                )
                text.append(f"          {step['description']}\n", style=Palette.FAINT)
            text.append(
                f"\n  in {report['plan']['steps'][0]['cwd']}\n", style=Palette.FAINT
            )

        projection = report.get("projection")
        if projection:
            text.append("\n")
            if projection["would_start_immediately"]:
                _row(
                    text,
                    "schedule",
                    "would start immediately",
                    note=f"{projection['cores_free']} of {projection['cores_total']} cores free",
                    style=Palette.SUCCESS,
                    width=12,
                )
            else:
                _row(
                    text,
                    "schedule",
                    "would wait",
                    note=projection["blocking_reason"] or "",
                    style=Palette.WARNING,
                    width=12,
                )

        if report["suggested_tags"]:
            _row(
                text,
                "tags",
                " ".join(report["suggested_tags"]),
                note="suggested",
                width=12,
            )

        text.append("\n")
        text.append("Nothing was submitted.  ", style=Palette.MUTED)
        text.append(_keyline(("esc", "back")))
        return text

    def heading(self) -> Text:
        return Text("execution plan", style=f"bold {Palette.TEXT}")

    def action_back(self) -> None:
        self.dismiss()


# -- text helpers -------------------------------------------------------------------


def _row(
    text: Text,
    label: str,
    value: str,
    *,
    note: str = "",
    style: str = Palette.TEXT,
    width: int = 10,
) -> None:
    """Append an aligned ``label   value`` line to a definition list."""
    text.append(f"{label:<{width}}", style=Palette.FAINT)
    text.append(value, style=style)
    if note:
        text.append(f"   {note}", style=Palette.FAINT)
    text.append("\n")


def _keyline(*pairs: tuple[str, str]) -> Text:
    """Render key hints inline, keys in the accent colour."""
    text = Text()
    for index, (key, label) in enumerate(pairs):
        if index:
            text.append("   ")
        text.append(key, style=Palette.ACCENT_TEXT)
        text.append(f" {label}", style=Palette.MUTED)
    return text


def _wrapped(message: str, width: int, *, indent: int) -> str:
    """Wrap ``message`` to ``width`` with continuation lines indented, ending in a newline."""
    import textwrap

    lines = textwrap.wrap(message, width=width) or [""]
    pad = " " * indent
    return lines[0] + "".join(f"\n{pad}{line}" for line in lines[1:]) + "\n"


def _path_text(path: Path) -> Text:
    """The current directory, with the final component emphasised.

    A long path is mostly context; the part that changes as you browse is the tail, so
    that is the part given full-strength text.
    """
    text = Text()
    parent = str(path.parent).rstrip("/")
    if parent and path.parent != path:
        text.append(f"{parent}/", style=Palette.FAINT)
    text.append(path.name or str(path), style=Palette.TEXT)
    return text
