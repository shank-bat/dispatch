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
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

from dispatch.ipc.protocol import Method
from dispatch.tui.screens.base import DispatchScreen
from dispatch.tui.screens.projects import ProjectSearchScreen
from dispatch.tui.theme import Palette, severity_style

__all__ = ["SubmitScreen"]


SWEEP_LABEL_WIDTH = 12
"""Label column for the sweep pane. ``concurrent`` alone needs more than the default."""

SWEEP_PREVIEW = 6
"""How many of a sweep's cases to name in the detail pane.

Enough to confirm the folder is the one intended and that the ordering looks right,
without turning the pane into a file listing.
"""


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
        Binding("slash", "find_project", "find"),
        Binding("c", "edit_cores", "cores"),
        Binding("g", "edit_gpus", "gpus"),
        Binding("t", "edit_tags", "tags"),
        Binding("a", "run_after", "run after"),
        Binding("m", "edit_concurrency", "max at once"),
        Binding("p", "edit_path", "go to path"),
        Binding("period", "toggle_hidden", "hidden"),
    ]

    def __init__(self, start: Path | None = None) -> None:
        super().__init__()
        self.path = (start or Path.home()).expanduser().resolve()
        self.entries: list[dict[str, Any]] = []
        self.cores = 1
        self.gpus = 0
        """GPUs to request. Zero means CPU work, which is what almost every job is."""

        self.tags: list[str] = []
        self.run_after: dict[str, Any] | None = None
        """The job this one should wait for. ``None`` -- the default -- means none."""

        self.sweep: dict[str, Any] | None = None
        """The sweep this directory was recognised as, if it was. ``None`` for a case."""

        self.concurrency = 1
        """How many of a sweep's jobs may run at once. Ignored unless this is a sweep."""
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

        self.sweep = listing.get("sweep")
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
                Method.CASE_VALIDATE, path=str(self.path), cores=self.cores, gpus=self.gpus
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

        if result is None and self.sweep is not None:
            self.query_one("#detail-text", Static).update(self._sweep_detail())
            return

        if result is None:
            text.append("no case here\n\n", style=Palette.MUTED)
            text.append(
                "Browse into a directory containing a case.\n\n", style=Palette.FAINT
            )
            text.append(
                _keyline(
                    ("enter", "open"), ("backspace", "up"), ("/", "find"), ("p", "path")
                )
            )
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
        total_gpus = int(self.app_state.snapshot.get("total_gpus", 0) or 0)
        if self.gpus or total_gpus:
            free_gpus = self.app_state.snapshot.get("free_gpus")
            _row(
                text,
                "gpus",
                str(self.gpus),
                note=f"{free_gpus} free" if free_gpus is not None else "",
                style=Palette.ACCENT if self.gpus else Palette.MUTED,
            )
        if self.tags:
            _row(text, "tags", " ".join(self.tags))
        # Always shown, including its default, so that "this job starts when it fits" is
        # visible rather than assumed.
        _row(
            text,
            "run after",
            self.run_after["name"] if self.run_after else "none",
            style=Palette.TEXT if self.run_after else Palette.MUTED,
        )

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
            text.append(
                _keyline(
                    ("s", "submit"),
                    ("d", "plan"),
                    ("c", "cores"),
                    ("g", "gpus"),
                    ("t", "tags"),
                    ("a", "after"),
                )
            )
        else:
            text.append(
                _keyline(("f", "submit anyway"), ("d", "plan"), ("c", "cores"), ("g", "gpus"))
            )

        self.query_one("#detail-text", Static).update(text)

    def _sweep_detail(self) -> Text:
        """The right-hand pane for a sweep folder.

        Deliberately the same shape as the case pane -- the same aligned label/value rows,
        the same key line -- because a sweep is not a different kind of thing to submit, it
        is the same thing with two extra numbers on it.
        """
        assert self.sweep is not None
        text = Text()
        text.append("sweep folder\n\n", style=Palette.ACCENT)

        # `concurrent` is exactly ten characters, so the default label column leaves no gap
        # between it and its value. Widened here rather than globally: the case pane's
        # labels are shorter and its alignment is already right.
        _row(text, "solver", str(self.sweep["solver"]), width=SWEEP_LABEL_WIDTH)
        _row(text, "cases", str(self.sweep["count"]), width=SWEEP_LABEL_WIDTH)
        free = self.app_state.snapshot.get("free_cores")
        _row(
            text,
            "cores/job",
            str(self.cores),
            note=f"{free} free" if free is not None else "",
            width=SWEEP_LABEL_WIDTH,
        )
        total_gpus = int(self.app_state.snapshot.get("total_gpus", 0) or 0)
        if self.gpus or total_gpus:
            free_gpus = self.app_state.snapshot.get("free_gpus")
            _row(
                text,
                "gpus/job",
                str(self.gpus),
                note=f"{free_gpus} free" if free_gpus is not None else "",
                style=Palette.ACCENT if self.gpus else Palette.MUTED,
                width=SWEEP_LABEL_WIDTH,
            )
        # The setting the whole feature exists for, so it is stated as a limit rather than
        # left to be inferred from a bare number.
        _row(
            text,
            "concurrent",
            str(self.concurrency),
            note=f"at most {self.concurrency * self.cores} cores in use",
            width=SWEEP_LABEL_WIDTH,
        )
        if self.tags:
            _row(text, "tags", " ".join(self.tags), width=SWEEP_LABEL_WIDTH)

        text.append("\n")
        for position, name in enumerate(self.sweep["cases"][:SWEEP_PREVIEW], start=1):
            # Numbered, because the order is a property of the sweep the user is about to
            # commit to and not merely how the directory happened to list.
            text.append(f"  {position:>3} ", style=Palette.FAINT)
            text.append(f"{name}\n", style=Palette.MUTED)
        remaining = int(self.sweep["count"]) - SWEEP_PREVIEW
        if remaining > 0:
            text.append(f"      and {remaining} more — press d to review\n", style=Palette.FAINT)

        text.append("\n")
        text.append(
            _keyline(
                ("s", "submit sweep"),
                ("d", "review"),
                ("c", "cores/job"),
                ("m", "max at once"),
            )
        )
        return text

    def action_edit_concurrency(self) -> None:
        """Set how many of a sweep's jobs may run at once."""
        if self.sweep is None:
            self.notify_error("This directory is not a sweep folder")
            return
        self._open_prompt(
            "concurrency", f"concurrent jobs (currently {self.concurrency}): "
        )

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

    def action_edit_gpus(self) -> None:
        """Ask for GPUs. Setting a non-zero count is what makes this a GPU job."""
        self._open_prompt("gpus", f"gpus (currently {self.gpus}): ")

    def action_find_project(self) -> None:
        """Search for a case by name instead of browsing to it."""

        def _chosen(path: str | None) -> None:
            if path is None:
                return
            self.path = Path(path)
            self.app.call_later(self.load)

        self.app.push_screen(ProjectSearchScreen(), _chosen)

    def action_edit_tags(self) -> None:
        self._open_prompt("tags", "tags, space separated: ")

    def action_edit_path(self) -> None:
        self._open_prompt("path", "path: ")

    def action_run_after(self) -> None:
        """Choose a job this one should wait for, or clear the choice.

        A picker rather than a typed id: the jobs worth waiting for are the ones already on
        screen, and nobody wants to retype a UUID prefix from the queue view.
        """
        candidates = self.app_state.running + self.app_state.queued
        if not candidates:
            self.notify_error("There are no unfinished jobs to wait for.")
            return

        def _chosen(job_id: str | None) -> None:
            if job_id is None:  # backed out; leave the current choice alone
                return
            self.run_after = self.app_state.get(job_id) if job_id else None
            self._render_detail()

        self.app.push_screen(RunAfterScreen(candidates, current=self.run_after), _chosen)

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

        if self._prompt_mode == "concurrency" and value:
            try:
                self.concurrency = max(1, int(value))
            except ValueError:
                self.notify_error(f"{value!r} is not a number")
            else:
                self._render_detail()
            return
        if self._prompt_mode == "cores" and value:
            try:
                self.cores = max(1, int(value))
            except ValueError:
                self.notify_error(f"{value!r} is not a number")
                return
            await self.inspect(self._inspection.get("solver") if self._inspection else None)
        elif self._prompt_mode == "gpus" and value:
            try:
                self.gpus = max(0, int(value))
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
        if self._inspection is None and self.sweep is not None:
            await self._review_sweep()
            return
        if self._inspection is None or "error" in self._inspection:
            self.notify_error("This directory is not a recognised case.")
            return
        try:
            report = await self.dispatch_app.call(
                Method.CASE_DRYRUN, workdir=str(self.path), cores=self.cores, gpus=self.gpus
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
        if self._inspection is None and self.sweep is not None:
            await self._submit_sweep()
            return
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
                gpus=self.gpus,
                tags=self.tags,
                force=force,
                depends_on_job_id=self.run_after["id"] if self.run_after else None,
            )
        except Exception as exc:
            self.notify_error(str(exc))
            return

        job = result["job"]
        self.notify_ok(f"Queued {job['name']} ({job['cores']} cores)")
        self.dismiss()

    async def _review_sweep(self) -> None:
        """Show what submitting this folder as a sweep would do, without doing it.

        The per-case plan comes from the ordinary ``case.dryrun`` path, run against the
        first case: every member of a sweep is the same solver with the same resources, so
        one case's plan is the plan, and asking for it through the existing endpoint means
        the review cannot drift from what submission will actually do. What the sweep adds
        on top -- the full ordered case list, the cap, the ceiling on cores -- is carried
        alongside it rather than recomputed anywhere.
        """
        assert self.sweep is not None
        first = self.path / self.sweep["cases"][0]
        try:
            report = await self.dispatch_app.call(
                Method.CASE_DRYRUN,
                workdir=str(first),
                cores=self.cores,
                gpus=self.gpus,
            )
        except Exception as exc:
            self.notify_error(str(exc))
            return

        self.app.push_screen(
            PlanScreen(
                report,
                sweep={
                    "root": str(self.path),
                    "solver": self.sweep["solver"],
                    "cases": list(self.sweep["cases"]),
                    "cores_per_job": self.cores,
                    "gpus_per_job": self.gpus,
                    "concurrency": self.concurrency,
                },
            )
        )

    async def _submit_sweep(self) -> None:
        """Queue the whole folder as one sweep.

        No force variant: a sweep is refused only when the daemon no longer sees it as one,
        which force could not make true anyway.
        """
        try:
            result = await self.dispatch_app.call(
                Method.SWEEP_SUBMIT,
                root=str(self.path),
                cores_per_job=self.cores,
                concurrency=self.concurrency,
                gpus=self.gpus,
                tags=self.tags,
            )
        except Exception as exc:
            self.notify_error(str(exc))
            return

        sweep = result["sweep"]
        self.notify_ok(
            f"Queued sweep {sweep['name']}: {sweep['total']} cases, "
            f"{sweep['cores_per_job']} cores each, {sweep['concurrency']} at a time"
        )
        self.dismiss()


class RunAfterScreen(ModalScreen[str | None]):
    """Pick the job a submission should wait for.

    Dismisses with the chosen job's id, with ``""`` for "none" — the first entry, and the
    default — or with ``None`` when the user backs out and the current choice should stand.

    Deliberately just a list: choosing one job to wait for is the whole feature, and
    anything more would be a dependency editor.
    """

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, jobs: list[dict[str, Any]], *, current: dict[str, Any] | None) -> None:
        super().__init__()
        self.jobs = jobs
        self._current = current

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(Text("run after", style=f"bold {Palette.TEXT}"))
            items = [ListItem(Label(Text("none", style=Palette.MUTED)))]
            for job in self.jobs:
                label = Text()
                label.append(f"{job['id'][:8]}  ", style=Palette.FAINT)
                label.append(job["name"], style=Palette.TEXT)
                label.append(f"  {job['state'].lower()}", style=Palette.MUTED)
                items.append(ListItem(Label(label)))
            view = ListView(*items, id="run-after-list")
            view.index = self._initial_index()
            yield view
            yield Label(_keyline(("enter", "choose"), ("esc", "cancel")), classes="confirm-keys")

    def _initial_index(self) -> int:
        """Start on the current choice, so reopening the picker shows what is set."""
        if self._current is None:
            return 0
        for offset, job in enumerate(self.jobs, start=1):
            if job["id"] == self._current["id"]:
                return offset
        return 0

    def on_mount(self) -> None:
        self.query_one("#run-after-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index or 0
        self.dismiss("" if index == 0 else self.jobs[index - 1]["id"])

    def action_cancel(self) -> None:
        self.dismiss(None)


class PlanScreen(DispatchScreen):
    """The dry-run report: exactly what would run, and nothing was started to find out."""

    TITLE = "Execution plan"

    BINDINGS = [Binding("escape,q,enter", "back", "back")]

    def __init__(self, report: dict[str, Any], *, sweep: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.report = report
        self.sweep = sweep
        """Sweep context, when the report is one case standing in for a whole folder."""

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        yield Static("", id="heading")
        yield Static(self._body(), id="plan")
        yield from self.compose_footer()

    def _sweep_summary(self) -> Text:
        """What the sweep would submit: every case, in order, and the limits on them.

        The whole list, not a preview. This is the screen the user opens *to check* the
        order and membership before committing, so truncating it would remove the only
        reason to open it.
        """
        assert self.sweep is not None
        sweep = self.sweep
        cases = sweep["cases"]
        cores = int(sweep["cores_per_job"])
        concurrency = int(sweep["concurrency"])

        text = Text()
        text.append("sweep\n\n", style=Palette.ACCENT)
        _row(text, "folder", str(sweep["root"]), width=12)
        _row(text, "solver", str(sweep["solver"]), width=12)
        _row(text, "cases", f"{len(cases)} jobs", width=12)
        _row(text, "cores/job", str(cores), width=12)
        if sweep.get("gpus_per_job"):
            _row(text, "gpus/job", str(sweep["gpus_per_job"]), width=12)
        _row(
            text,
            "concurrent",
            str(concurrency),
            note=f"a hard cap; at most {concurrency * cores} cores in use at once",
            width=12,
        )

        text.append("\norder\n", style=Palette.FAINT)
        for position, name in enumerate(cases, start=1):
            text.append(f"  {position:>3} ", style=Palette.FAINT)
            text.append(f"{name}\n", style=Palette.TEXT)
        return text

    def _body(self) -> Text:
        """The dry-run report.

        The plan is the point of this screen, so it gets the most space and the only
        emphasis: the solve step is highlighted, preparation steps are muted, and each
        command is shown exactly as it will be run.
        """
        report = self.report
        text = Text()

        if self.sweep is not None:
            text.append(self._sweep_summary())
            text.append("\nplan for each case\n\n", style=Palette.FAINT)

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
        _row(text, "resources", str(report.get("resources") or report["cores"]), width=12)
        if report.get("log_path"):
            # Where the output will appear, said before anything is submitted: the whole
            # point of the working-directory log is that it can be found without asking.
            _row(text, "log", str(report["log_path"]), width=12)

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
