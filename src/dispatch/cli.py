"""The ``dispatch`` command: a TUI by default, a scriptable CLI when given a verb.

Both are thin skins over :class:`~dispatch.ipc.client.DaemonClient`, so there is one
implementation of talking to the daemon and no way for the two front ends to disagree.

The CLI exists because ``ssh eddy dispatch ls`` should work, and because a shell script
that queues fifty parameter variations should not have to drive a TUI.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dispatch.core.config import Config, load_config
from dispatch.core.errors import DispatchError
from dispatch.ipc.client import DaemonClient, DaemonUnavailable
from dispatch.ipc.protocol import Method
from dispatch.version import __version__

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dispatch",
        description="Job scheduler for a scientific workstation. Run with no arguments "
        "for the interactive interface.",
    )
    parser.add_argument("--config", type=Path, help="configuration file to use")
    parser.add_argument("--version", action="version", version=f"dispatch {__version__}")
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="fail rather than starting a daemon if none is running",
    )

    sub = parser.add_subparsers(dest="command")

    submit = sub.add_parser("submit", help="queue a case")
    submit.add_argument("path", type=Path, help="the case directory")
    submit.add_argument("-c", "--cores", type=int, default=1, help="cores to request")
    submit.add_argument(
        "-g",
        "--gpus",
        type=int,
        metavar="N",
        help="GPUs to request; implies --resource gpu",
    )
    submit.add_argument(
        "--resource",
        choices=["cpu", "gpu"],
        help="which pool this job draws from (default: gpu when --gpus is given, else cpu)",
    )
    submit.add_argument("-n", "--name", help="job name (defaults to the directory name)")
    submit.add_argument("-p", "--priority", type=int, default=0, help="higher runs first")
    submit.add_argument("--ram", type=int, metavar="MB", help="RAM estimate in megabytes")
    submit.add_argument("--solver", help="force a solver instead of detecting one")
    submit.add_argument("-t", "--tag", action="append", default=[], help="tag (repeatable)")
    submit.add_argument(
        "--after",
        metavar="ID",
        help="run only once this job has completed (id, or a unique prefix of one)",
    )
    submit.add_argument("--note", help="a note to attach")
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would happen, then do nothing",
    )
    submit.add_argument(
        "--force", action="store_true", help="queue even if validation reports errors"
    )

    ls = sub.add_parser("ls", help="list jobs")
    ls.add_argument("--state", action="append", default=[], help="filter by state (repeatable)")
    ls.add_argument("-n", "--limit", type=int, default=50, help="how many to show")
    ls.add_argument("-a", "--all", action="store_true", help="include finished jobs")

    show = sub.add_parser("show", help="show one job in detail")
    show.add_argument("id", help="job id, or a unique prefix of one")

    search = sub.add_parser("search", help="search the history")
    search.add_argument(
        "query", nargs="*", help="e.g. tag:paper solver:openfoam cores>=16 resource:gpu"
    )
    search.add_argument("-n", "--limit", type=int, default=50)

    find = sub.add_parser("find", help="find a project directory by name")
    find.add_argument("name", nargs="*", help="part of a directory name, e.g. cavity")
    find.add_argument("-n", "--limit", type=int, default=20, help="how many to show")
    find.add_argument(
        "-0",
        "--paths-only",
        action="store_true",
        help="print bare paths, one per line, for use in a shell pipeline",
    )

    for name, help_text in (
        ("cancel", "stop a job"),
        ("hold", "withhold a queued job"),
        ("release", "return a held job to the queue"),
        ("rm", "delete a finished job"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("id", help="job id, or a unique prefix of one")
        if name == "cancel":
            command.add_argument("-f", "--force", action="store_true", help="skip the grace period")
        if name == "rm":
            command.add_argument("--purge-logs", action="store_true", help="delete its logs too")

    priority = sub.add_parser("priority", help="change a job's priority")
    priority.add_argument("id")
    priority.add_argument("priority", type=int)

    tag = sub.add_parser("tag", help="add or remove tags")
    tag.add_argument("id")
    tag.add_argument("tags", nargs="*", help="tags to add; prefix with - to remove")

    note = sub.add_parser("note", help="attach a note to a job")
    note.add_argument("id")
    note.add_argument("body", nargs="+")

    logs = sub.add_parser("logs", help="print or follow a job's output")
    logs.add_argument("id")
    logs.add_argument("-f", "--follow", action="store_true", help="follow, like tail -f")
    logs.add_argument(
        "-e",
        "--stderr",
        action="store_true",
        help="show stderr instead (only differs for jobs from before merged logs)",
    )
    logs.add_argument(
        "--steps",
        action="store_true",
        help="show the preparation transcript instead of the solver's output",
    )
    logs.add_argument("-n", "--lines", type=int, default=50, help="how many lines to show")

    sub.add_parser("status", help="show the machine and the queue")
    sub.add_parser("tags", help="list tags in use")
    sub.add_parser("info", help="show daemon information")
    sub.add_parser("doctor", help="check this machine's setup")
    sub.add_parser("stop", help="stop the daemon (running jobs keep going)")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``dispatch`` command."""
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config, required=args.config is not None)
    except DispatchError as exc:
        print(f"dispatch: {exc}", file=sys.stderr)
        return 2

    if args.command is None:
        return _run_tui(config, autostart=not args.no_autostart)
    if args.command == "doctor":
        return _doctor(config)

    try:
        return asyncio.run(_run_command(args, config))
    except KeyboardInterrupt:
        return 130
    except DispatchError as exc:
        print(f"dispatch: {exc}", file=sys.stderr)
        return 1


def _run_tui(config: Config, *, autostart: bool) -> int:
    """Launch the interactive interface.

    Imported lazily so that ``dispatch ls`` in a script does not pay Textual's import cost.
    """
    if not sys.stdout.isatty():
        print(
            "dispatch: this is the interactive interface and needs a terminal. "
            "Try `dispatch ls` or `dispatch status`.",
            file=sys.stderr,
        )
        return 2

    from dispatch.tui.app import DispatchApp

    app = DispatchApp(config=config, autostart=autostart)
    app.run()
    return 0


async def _run_command(args: argparse.Namespace, config: Config) -> int:
    """Connect to the daemon and run one CLI verb."""
    from rich.console import Console

    console = Console()
    client = DaemonClient(
        config.paths.socket,
        autostart=config.daemon.autostart and not args.no_autostart,
        config_path=config.source,
    )

    try:
        await client.connect()
    except DaemonUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        return 3

    try:
        handler = _HANDLERS[args.command]
        return await handler(client, args, console, config)
    finally:
        await client.close()


# -- verbs ---------------------------------------------------------------------------------


async def _submit(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    path = args.path.expanduser().resolve()

    if args.dry_run:
        report = await client.call(
            Method.CASE_DRYRUN,
            workdir=str(path),
            cores=args.cores,
            gpus=args.gpus or 0,
            resource=args.resource,
            ram_mb=args.ram,
            solver=args.solver,
            name=args.name or "",
        )
        _print_dry_run(console, report)
        return 0 if report["would_submit"] else 1

    result = await client.call(
        Method.JOB_SUBMIT,
        workdir=str(path),
        cores=args.cores,
        gpus=args.gpus,
        resource=args.resource,
        ram_mb=args.ram,
        solver=args.solver,
        name=args.name or "",
        priority=args.priority,
        tags=args.tag,
        note=args.note,
        force=args.force,
        depends_on_job_id=args.after,
    )
    job = result["job"]
    _print_findings(console, result.get("validation"))
    position = job.get("queue_position")
    where = f" at queue position {position}" if position else ""
    console.print(
        f"[green]Queued[/green] {job['name']} ({job['solver']}, {_resources(job)}){where}"
    )
    console.print(f"  id {job['id']}")
    if job.get("log_path"):
        # The single most useful line of the output: where to look while it runs.
        console.print(f"  log {job['log_path']}")
    return 0


async def _ls(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    states = [s.upper() for s in args.state] or (
        None if args.all else ["QUEUED", "HELD", "PREPARING", "RUNNING"]
    )
    page = await client.call(Method.JOB_LIST, states=states, limit=args.limit)
    _print_jobs(console, page["items"])
    if not page["items"]:
        console.print("[dim]No jobs.[/dim]")
    elif page["has_more"]:
        console.print(f"[dim]showing {len(page['items'])} of {page['total']}[/dim]")
    return 0


async def _search(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    page = await client.call(Method.HISTORY_SEARCH, query=" ".join(args.query), limit=args.limit)
    _print_jobs(console, page["items"])
    console.print(f"[dim]{page['total']} match(es)[/dim]")
    return 0


async def _find(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    """Find a project directory by name under the configured root.

    ``--paths-only`` exists so this composes: ``dispatch submit "$(dispatch find -0 cavity
    | head -1)" --cores 20`` is a reasonable thing to type, and it only works if the
    output is a bare path rather than a table.
    """
    payload = await client.call(
        Method.PROJECTS_SEARCH, query=" ".join(args.name), limit=args.limit
    )
    if payload.get("error"):
        console.print(f"[red]{payload['error']}[/red]")
        return 1

    results = payload.get("results") or []
    if args.paths_only:
        for entry in results:
            print(entry["path"])
        return 0 if results else 1

    if not results:
        console.print(f"[dim]No match under {payload['root']}.[/dim]")
        return 1

    from rich.table import Table

    table = Table(box=None, padding=(0, 2, 0, 0), header_style="bold")
    table.add_column("")
    table.add_column("project")
    table.add_column("path")
    for entry in results:
        table.add_row(
            "[green]*[/green]" if entry.get("case") else " ",
            entry.get("case") or "",
            entry["path"],
        )
    console.print(table)
    if payload.get("truncated"):
        console.print("[dim]more matches exist; narrow the search[/dim]")
    return 0


async def _show(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    from rich.table import Table

    detail = await client.call(Method.JOB_GET, id=args.id)
    job = detail["job"]

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_row("id", job["id"])
    table.add_row("name", job["name"])
    table.add_row("state", _state_markup(job["state"]))
    table.add_row("solver", f"{job['solver']} ({job['solver_binary'] or 'unknown'})")
    table.add_row("directory", job["workdir"])
    table.add_row("resources", _resources(job))
    if job["ram_estimate_mb"]:
        table.add_row("ram estimate", f"{job['ram_estimate_mb']} MB")
    table.add_row("priority", str(job["priority"]))
    if job.get("depends_on_job_id"):
        table.add_row("run after", job["depends_on_job_id"][:8])
    if job["tags"]:
        table.add_row("tags", " ".join(job["tags"]))
    if detail.get("waiting_because"):
        table.add_row("waiting", detail["waiting_because"])
    if job["exit_code"] is not None:
        table.add_row("exit", f"{job['exit_code']} ({job['exit_reason'] or 'unknown'})")
    if job.get("exit_detail"):
        # Indented under its own label rather than inlined: these are the solver's words,
        # they run to several lines, and they are the first thing worth reading here.
        table.add_row("failed because", _indent(job["exit_detail"]))
    if job["metrics"]["runtime_s"]:
        table.add_row("runtime", _duration(job["metrics"]["runtime_s"]))
    if job["metrics"]["peak_rss_mb"]:
        table.add_row("peak memory", f"{job['metrics']['peak_rss_mb']} MB")
    if job.get("log_path"):
        table.add_row("log", job["log_path"])
    elif job["stdout_path"]:
        # Pre-dates the working-directory convention; its output is where it was written.
        table.add_row("stdout", job["stdout_path"])
    console.print(table)

    case = job["metadata"].get("case") or {}
    if case:
        console.print("\n[bold]Case[/bold]")
        meta = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
        for key, value in case.items():
            meta.add_row(key, str(value))
        console.print(meta)

    if detail["notes"]:
        console.print("\n[bold]Notes[/bold]")
        for note in detail["notes"]:
            console.print(f"  {note['body']}")

    if detail["events"]:
        console.print("\n[bold]History[/bold]")
        for event in detail["events"][-15:]:
            console.print(f"  [dim]{_time(event['ts'])}[/dim]  {event['detail']}")

    provenance = await client.call(Method.JOB_PROVENANCE, id=args.id)
    if provenance:
        console.print("\n[bold]Provenance[/bold]")
        prov = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
        prov.add_row("host", f"{provenance['hostname']} ({provenance['kernel_version']})")
        if provenance.get("solver_version"):
            prov.add_row("solver", provenance["solver_version"])
        git = provenance.get("git") or {}
        if git.get("commit"):
            dirty = " [yellow](uncommitted changes)[/yellow]" if git.get("dirty") else ""
            prov.add_row("git", f"{git['commit'][:10]} {git.get('branch') or ''}{dirty}")
        prov.add_row("dispatch", provenance["dispatch_version"])
        if provenance.get("argv"):
            prov.add_row("command", " ".join(provenance["argv"]))
        console.print(prov)
    return 0


async def _status(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    from rich.table import Table

    snapshot = await client.call(Method.SYSTEM_SNAPSHOT)
    page = await client.call(
        Method.JOB_LIST, states=["QUEUED", "HELD", "PREPARING", "RUNNING"], limit=100
    )

    console.print(
        f"[bold]{snapshot['hostname']}[/bold]  "
        f"{snapshot['free_cores']}/{snapshot['total_cores']} cores free  "
        f"(reserved {snapshot['reserved_cores']}, allocated {snapshot['allocated_cores']})"
    )
    if snapshot.get("total_gpus"):
        console.print(
            f"[bold]gpus[/bold]  {snapshot['free_gpus']}/{snapshot['total_gpus']} free  "
            f"(allocated {snapshot['allocated_gpus']})"
        )
    console.print(
        f"cpu {snapshot['cpu_percent']:.0f}%   "
        f"ram {snapshot['used_ram_mb'] // 1024}/{snapshot['total_ram_mb'] // 1024} GB   "
        f"load {', '.join(f'{value:.2f}' for value in snapshot['load_average'])}"
    )
    console.print()
    if page["items"]:
        _print_jobs(console, page["items"])
    else:
        console.print("[dim]Nothing queued or running.[/dim]")
    del Table
    return 0


async def _cancel(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    result = await client.call(Method.JOB_CANCEL, id=args.id, force=args.force)
    console.print(
        "[yellow]Cancelling[/yellow]" if result["cancelling"] else "[yellow]Cancelled[/yellow]"
    )
    return 0


async def _hold(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    job = await client.call(Method.JOB_HOLD, id=args.id)
    console.print(f"[yellow]Held[/yellow] {job['name']}")
    return 0


async def _release(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    job = await client.call(Method.JOB_RELEASE, id=args.id)
    console.print(f"[green]Released[/green] {job['name']}")
    return 0


async def _priority(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    job = await client.call(Method.JOB_PRIORITY, id=args.id, priority=args.priority)
    console.print(f"{job['name']} priority is now {job['priority']}")
    return 0


async def _rm(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    result = await client.call(Method.JOB_DELETE, id=args.id, purge_logs=args.purge_logs)
    console.print(f"Deleted {result['name']}")
    return 0


async def _tag(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    add = [t for t in args.tags if not t.startswith("-")]
    remove = [t[1:] for t in args.tags if t.startswith("-")]
    job = await client.call(Method.JOB_TAG, id=args.id, add=add, remove=remove)
    console.print(f"{job['name']}: {' '.join(job['tags']) or '(no tags)'}")
    return 0


async def _note(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    await client.call(Method.JOB_NOTE, id=args.id, body=" ".join(args.body))
    console.print("Note added.")
    return 0


async def _tags(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    for entry in await client.call(Method.TAGS_LIST):
        console.print(f"{entry['count']:>5}  {entry['name']}")
    return 0


async def _logs(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    """Print or follow a job's output.

    The log file is read **directly**, not proxied through the daemon: same machine, same
    user, same filesystem. Streaming gigabytes of residuals through a JSON socket would be
    absurd.
    """
    detail = await client.call(Method.JOB_GET, id=args.id)
    job = detail["job"]
    if args.steps:
        path_str = str(config.paths.job_dir(job["id"]) / "steps.log")
    elif args.stderr:
        path_str = job["stderr_path"]
    else:
        path_str = job.get("log_path") or job["stdout_path"]
    if not path_str:
        console.print("[red]This job has no log file yet.[/red]")
        return 1

    path = Path(path_str)
    if not path.exists():
        console.print(f"[red]{path} does not exist.[/red]")
        return 1

    from dispatch.tui.tailer import read_last_lines

    for line in read_last_lines(path, args.lines):
        print(line)

    if not args.follow:
        return 0

    with contextlib.suppress(KeyboardInterrupt):
        offset = path.stat().st_size
        while True:
            await asyncio.sleep(0.25)
            size = path.stat().st_size
            if size < offset:
                offset = 0  # truncated or rotated
            if size > offset:
                with path.open("rb") as handle:
                    handle.seek(offset)
                    chunk = handle.read(size - offset)
                offset = size
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
    return 0


async def _info(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    from rich.table import Table

    info = await client.call(Method.DAEMON_INFO)
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_row("version", info["version"])
    table.add_row("pid", str(info["pid"]))
    table.add_row("uptime", _duration(info["uptime_s"]))
    table.add_row("policy", info["policy"] + (" (paused)" if info["paused"] else ""))
    table.add_row("database", info["database"])
    table.add_row("socket", info["socket"])
    table.add_row("logs", info["log_dir"])
    table.add_row("clients", str(info["clients"]))
    table.add_row("adapters", ", ".join(a["adapter"] for a in info["adapters"]))
    console.print(table)

    counts = {k: v for k, v in info["counts"].items() if v}
    if counts:
        console.print("\n" + "  ".join(f"{state.lower()} {n}" for state, n in counts.items()))
    for rejected in info["rejected_adapters"]:
        console.print(
            f"[yellow]adapter not loaded:[/yellow] {rejected['name']}: {rejected['reason']}"
        )
    return 0


async def _stop(client: DaemonClient, args: Any, console: Any, config: Config) -> int:
    await client.call(Method.DAEMON_SHUTDOWN)
    console.print("Daemon stopping. Running simulations keep going.")
    return 0


def _doctor(config: Config) -> int:
    """Check this machine's setup. Deliberately works with no daemon running."""
    from rich.console import Console

    from dispatch.daemon.selfcheck import CheckStatus, run_checks

    console = Console()
    console.print(f"[bold]Dispatch {__version__}[/bold] on {os.uname().nodename}\n")

    worst = CheckStatus.OK
    for check in run_checks(config):
        colour = {"ok": "green", "warn": "yellow", "fail": "red"}[check.status.value]
        console.print(f"[{colour}]{check.symbol}[/{colour}] {check.name:<10} {check.detail}")
        if check.remedy:
            console.print(f"    [dim]{check.remedy}[/dim]")
        if check.status is CheckStatus.FAIL:
            worst = CheckStatus.FAIL
        elif check.status is CheckStatus.WARN and worst is not CheckStatus.FAIL:
            worst = CheckStatus.WARN

    socket = config.paths.socket
    console.print(
        f"\n[green]+[/green] daemon     running at {socket}"
        if socket.exists()
        else f"\n[yellow]![/yellow] daemon     not running ({socket} does not exist)"
    )
    return 1 if worst is CheckStatus.FAIL else 0


# -- rendering -------------------------------------------------------------------------------


def _print_jobs(console: Any, jobs: list[dict[str, Any]]) -> None:
    from rich.table import Table

    if not jobs:
        return
    table = Table(box=None, padding=(0, 2, 0, 0), header_style="bold")
    table.add_column("id")
    table.add_column("name")
    table.add_column("state")
    table.add_column("res")
    table.add_column("solver")
    table.add_column("runtime", justify="right")
    table.add_column("tags")

    for job in jobs:
        position = job.get("queue_position")
        state = _state_markup(job["state"]) + (f" #{position}" if position else "")
        table.add_row(
            job["id"][:8],
            job["name"][:28],
            state,
            _resource_cell(job),
            job["solver_binary"] or job["solver"],
            _duration(job["metrics"]["runtime_s"]) if job["metrics"]["runtime_s"] else "",
            " ".join(job["tags"][:3]),
        )
    console.print(table)


def _resource_cell(job: dict[str, Any]) -> str:
    """The compact form used in listings: ``20c`` or ``1G`` (with cores when they matter)."""
    gpus = int(job.get("gpus") or 0)
    cores = int(job.get("cores") or 0)
    if not gpus:
        return f"{cores}c"
    return f"[cyan]{gpus}G[/cyan]" + (f"\u00b7{cores}c" if cores > 1 else "")


def _print_findings(console: Any, validation: dict[str, Any] | None) -> None:
    if not validation:
        return
    for finding in validation.get("findings", []):
        colour = {"ERROR": "red", "WARNING": "yellow", "INFO": "cyan"}.get(
            finding["severity"], "white"
        )
        console.print(f"  [{colour}]{finding['severity'].lower()}[/{colour}] {finding['message']}")
        if finding.get("hint"):
            console.print(f"    [dim]{finding['hint']}[/dim]")


def _print_dry_run(console: Any, report: dict[str, Any]) -> None:
    """Render the dry-run report -- generated by the code that would run, not a description."""
    console.print(f"  [bold]Case[/bold]        {report['workdir']}")
    if report["detections"]:
        best = report["detections"][0]
        console.print(
            f"  [bold]Detected[/bold]    {best['label']}  "
            f"[dim](confidence {best['confidence']:.2f})[/dim]"
        )
    console.print(f"  [bold]Requested[/bold]   {report.get('resources') or report['cores']}")
    if report.get("log_path"):
        console.print(f"  [bold]Log[/bold]         {report['log_path']}")
    console.print()

    validation = report["validation"]
    colour = "green" if validation["passed"] else "red"
    console.print(f"  [bold]Validation[/bold]  [{colour}]{validation['summary']}[/{colour}]")
    _print_findings(console, validation)
    console.print()

    if report["plan"]:
        console.print("  [bold]Execution plan[/bold]")
        for index, step in enumerate(report["plan"]["steps"], start=1):
            marker = "[cyan]SOLVE  [/cyan]" if step["kind"] == "SOLVE" else "[dim]PREPARE[/dim]"
            console.print(f"    {index}  {marker}  {step['command']}")
        console.print(f"       [dim]cwd  {report['plan']['steps'][0]['cwd']}[/dim]")
        console.print()

    projection = report.get("projection")
    if projection:
        if projection["would_start_immediately"]:
            console.print(
                f"  [bold]Scheduling[/bold]  {projection['cores_free']} of "
                f"{projection['cores_total']} cores free -- would start immediately"
            )
        else:
            console.print(f"  [bold]Scheduling[/bold]  would wait: {projection['blocking_reason']}")
    if report["suggested_tags"]:
        console.print(
            f"  [bold]Tags[/bold]        {' '.join(report['suggested_tags'])} "
            "[dim](suggested)[/dim]"
        )
    console.print("\n  [dim]Nothing was submitted.[/dim]")


def _resources(job: dict[str, Any]) -> str:
    """What a job holds, in one phrase: ``20 cores`` or ``1 GPU, 4 cores``."""
    gpus = int(job.get("gpus") or 0)
    cores = int(job.get("cores") or 0)
    cores_text = f"{cores} core{'' if cores == 1 else 's'}"
    if not gpus:
        return cores_text
    return f"{gpus} GPU{'' if gpus == 1 else 's'}, {cores_text}"


def _state_markup(state: str) -> str:
    colour = {
        "QUEUED": "white",
        "HELD": "yellow",
        "PREPARING": "cyan",
        "RUNNING": "green",
        "COMPLETED": "blue",
        "FAILED": "red",
        "CANCELLED": "yellow",
        "REJECTED": "red",
        "UNKNOWN": "magenta",
    }.get(state, "white")
    return f"[{colour}]{state.lower()}[/{colour}]"


def _indent(text: str) -> str:
    """Align a multi-line value under the first line of its table cell.

    Rich squares the columns for us; what it will not do is stop solver output containing
    square brackets from being read as markup, so the value is escaped.
    """
    from rich.markup import escape

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return escape("\n".join(lines))


def _duration(seconds: float | None) -> str:
    """Render a duration compactly."""
    if not seconds:
        return "-"
    total = int(seconds)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _time(timestamp: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


_HANDLERS = {
    "submit": _submit,
    "ls": _ls,
    "show": _show,
    "search": _search,
    "find": _find,
    "status": _status,
    "cancel": _cancel,
    "hold": _hold,
    "release": _release,
    "priority": _priority,
    "rm": _rm,
    "tag": _tag,
    "note": _note,
    "tags": _tags,
    "logs": _logs,
    "info": _info,
    "stop": _stop,
}


if __name__ == "__main__":
    raise SystemExit(main())
