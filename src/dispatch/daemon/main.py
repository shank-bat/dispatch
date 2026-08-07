"""The ``dispatchd`` entry point: wiring, lifecycle, and signals.

This module is the dependency-injection root. Every component above it takes its
collaborators as constructor arguments and none of them reach for a global, which is what
makes the scheduler testable with a fake executor and the executor testable with a fake
process manager.

Startup order is deliberate:

1. Load configuration and set up logging.
2. Take the single-instance lock, so two daemons cannot fight over one database.
3. Open and migrate the database.
4. Reconcile jobs left over from a previous life, **before** the socket is bound, so no
   client ever observes an inconsistent world.
5. Bind the socket and start scheduling.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import errno
import fcntl
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TextIO

from dispatch.adapters.registry import build_default_registry
from dispatch.core.config import Config, load_config
from dispatch.core.errors import DispatchError
from dispatch.daemon.dryrun import CaseInspector
from dispatch.daemon.events import EventBus
from dispatch.daemon.executor import JobExecutor
from dispatch.daemon.logsetup import configure_logging
from dispatch.daemon.monitor import JobSampler, SystemMonitor
from dispatch.daemon.notify import Notifier, event_for_state
from dispatch.daemon.policies import build_policy
from dispatch.daemon.provenance import ProvenanceCollector
from dispatch.daemon.recovery import recover
from dispatch.daemon.resources import ResourceModel
from dispatch.daemon.scheduler import Scheduler
from dispatch.daemon.selfcheck import survives_logout
from dispatch.daemon.server import IpcServer
from dispatch.db.connection import connect, migrate
from dispatch.db.repository import JobRepository
from dispatch.version import __version__

__all__ = ["Daemon", "main"]

log = logging.getLogger(__name__)


class SingleInstanceLock:
    """An advisory lock ensuring only one daemon owns a given database.

    Two daemons sharing one database would both schedule against the same queue and both
    believe they own the machine's cores. The lock is held on an open file descriptor, so
    the kernel releases it if the process dies -- unlike a pidfile, which survives a crash
    and then blocks the next start for no reason.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: TextIO | None = None

    def acquire(self) -> bool:
        """Take the lock. Returns ``False`` if another daemon holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return True

    def holder(self) -> int | None:
        """The pid recorded in the lock file, for a useful error message."""
        try:
            return int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        """Release the lock and remove the file."""
        if self._handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
        with contextlib.suppress(OSError):
            self.path.unlink()


class Daemon:
    """The assembled daemon.

    Args:
        config: Configuration.
        load_plugins: Whether to discover third-party adapters. Off in tests.
    """

    def __init__(self, config: Config, *, load_plugins: bool = True) -> None:
        self.config = config
        self._lock = SingleInstanceLock(config.paths.pidfile)
        self._stopping = asyncio.Event()
        self._load_plugins = load_plugins

        self.registry = build_default_registry(config.adapters, load_plugins=load_plugins)
        self.bus = EventBus()
        self.resources = ResourceModel(config.scheduler, log_dir=config.paths.log_dir)

        self.conn = connect(config.paths.database)
        migrate(self.conn)
        self.repo = JobRepository(self.conn, specs=self.registry.specs())

        self.monitor = SystemMonitor(resources=self.resources, bus=self.bus, config=config.daemon)
        self.bus._on_demand_change = self.monitor.on_demand_change

        self.notifier = Notifier(config.notifications)
        self.executor = JobExecutor(
            repo=self.repo,
            registry=self.registry,
            resources=self.resources,
            config=config,
            bus=self.bus,
            provenance=ProvenanceCollector(),
            on_finished=self._on_job_finished,
        )
        self.scheduler = Scheduler(
            repo=self.repo,
            resources=self.resources,
            executor=self.executor,
            policy=build_policy(config.scheduler),
            config=config.scheduler,
            bus=self.bus,
        )
        self.sampler = JobSampler(
            repo=self.repo,
            registry=self.registry,
            bus=self.bus,
            config=config.daemon,
            running_ids=lambda: self.executor.running_ids,
        )
        self.inspector = CaseInspector(registry=self.registry, resources=self.resources)
        self.server = IpcServer(
            config=config,
            repo=self.repo,
            scheduler=self.scheduler,
            executor=self.executor,
            resources=self.resources,
            registry=self.registry,
            inspector=self.inspector,
            monitor=self.monitor,
            bus=self.bus,
            on_shutdown=self.request_stop,
        )

    # -- lifecycle ------------------------------------------------------------------------

    async def run(self) -> int:
        """Start everything and serve until asked to stop."""
        if not self._lock.acquire():
            holder = self._lock.holder()
            log.error(
                "Another Dispatch daemon is already running%s. Only one may own %s.",
                f" (pid {holder})" if holder else "",
                self.config.paths.database,
            )
            return 1

        self._install_signal_handlers()
        self._report_logout_hazard()

        try:
            report = recover(
                repo=self.repo,
                resources=self.resources,
                executor=self.executor,
                config=self.config,
            )
            if report.total:
                log.info("Recovered %d job(s): %s", report.total, report.describe())

            self.notifier.start()
            self.sampler.start()
            self.scheduler.start()
            await self.server.start()

            log.info(
                "Dispatch %s ready: %d cores (%d reserved), policy %r, adapters: %s",
                __version__,
                self.resources.total_cores,
                self.resources.reserved_cores,
                self.config.scheduler.policy,
                ", ".join(self.registry.names) or "none",
            )
            await self._stopping.wait()
        finally:
            await self._shutdown()
        return 0

    async def _shutdown(self) -> None:
        """Stop cleanly, leaving simulations running.

        Stopping Dispatch and stopping a week-long run are different actions. The jobs are
        in their own sessions, so they survive; recovery re-adopts them next start.
        """
        log.info("Shutting down")
        await self.server.stop()
        await self.scheduler.stop()
        self.sampler.stop()
        self.monitor.stop()
        await self.executor.shutdown(kill_jobs=self.config.daemon.shutdown_kills_jobs)
        await self.notifier.stop()
        with contextlib.suppress(Exception):
            self.conn.close()
        self._lock.release()
        log.info("Stopped")

    def request_stop(self) -> None:
        """Ask the daemon to shut down. Safe from a signal handler."""
        self._stopping.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._on_terminate, sig)
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal.SIGHUP, self._on_reload)

    def _on_terminate(self, sig: signal.Signals) -> None:
        log.info("Received %s", sig.name)
        self.request_stop()

    def _on_reload(self) -> None:
        """Reload configuration in place, without restarting.

        Only the values that can safely change while jobs are running are applied: core
        reservations, margins, and the policy. Paths are not, because moving the database
        under a live daemon is not a reload, it is a different daemon.
        """
        log.info("Received SIGHUP; reloading configuration")
        try:
            fresh = load_config(self.config.source)
        except DispatchError as exc:
            log.error("Configuration not reloaded: %s", exc)
            return

        self.resources.reserved_cores = min(
            fresh.scheduler.reserved_cores, max(0, self.resources.total_cores - 1)
        )
        try:
            self.scheduler._policy = build_policy(fresh.scheduler)
        except DispatchError as exc:
            log.error("Policy not changed: %s", exc)
        self.scheduler.nudge()
        log.info("Configuration reloaded")

    def _on_job_finished(self) -> None:
        """Called by the executor whenever a job reaches a terminal state."""
        self.scheduler.nudge()
        for job in self.repo.recent(limit=1):
            event = event_for_state(job.state)
            if event:
                self.notifier.notify_job(job, event=event)

    def _report_logout_hazard(self) -> None:
        """Warn once if this daemon would not survive logout -- then carry on.

        Detecting and explaining beats mandating ``loginctl enable-linger`` for every user,
        most of whom do not have the problem: the Debian default is ``KillUserProcesses=no``.
        """
        survives, reason = survives_logout()
        if not survives:
            user = os.environ.get("USER", "$USER")
            log.warning(
                "This daemon will be killed when you log out (%s). To keep it running, "
                "use `loginctl enable-linger %s` or start it with "
                "`systemd-run --user --scope dispatchd`. Running simulations are "
                "unaffected while you stay logged in.",
                reason,
                user,
            )


# -- entry point ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dispatchd",
        description="The Dispatch scheduling daemon.",
        epilog=(
            "Runs in the foreground by default, which suits systemd, tmux, and debugging. "
            "Use --detach to background it. No systemd or linger configuration is required."
        ),
    )
    parser.add_argument("--config", type=Path, help="configuration file to use")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run in the background, with output redirected to the daemon log",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging threshold (default: INFO)",
    )
    parser.add_argument("--version", action="version", version=f"dispatchd {__version__}")
    return parser


def _detach(config: Config) -> None:
    """Daemonise with the classic double fork.

    The second fork guarantees the process cannot reacquire a controlling terminal, so a
    closing SSH session cannot send it SIGHUP. Standard, and about fifteen lines -- versus
    requiring every user to configure a service manager.
    """
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    config.paths.log_dir.mkdir(parents=True, exist_ok=True)
    os.chdir("/")
    os.umask(0o077)

    with open(os.devnull, "rb") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    target = open(config.paths.daemon_log, "ab", buffering=0)  # noqa: SIM115
    os.dup2(target.fileno(), sys.stdout.fileno())
    os.dup2(target.fileno(), sys.stderr.fileno())


def main(argv: list[str] | None = None) -> int:
    """Run the daemon. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config, required=args.config is not None)
    except DispatchError as exc:
        print(f"dispatchd: {exc}", file=sys.stderr)
        return 2

    if args.detach:
        _detach(config)

    configure_logging(
        log_file=config.paths.daemon_log,
        level=args.log_level,
        console=not args.detach,
    )

    try:
        daemon = Daemon(config)
    except DispatchError as exc:
        log.error("%s", exc)
        return 2

    try:
        return asyncio.run(daemon.run())
    except KeyboardInterrupt:
        return 0
    except Exception:
        log.exception("The daemon stopped unexpectedly")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
