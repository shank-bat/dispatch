"""Collecting the reproducibility record.

Captured once, between the last preparation step and the solver spawn. That timing is the
whole point: capturing at submission would miss changes made while the job sat in the
queue for two days, and capturing at finish would miss a checkout during the run.

Two rules govern everything here.

**Nothing may fail a job.** Every probe is wrapped. A missing ``git``, an unreadable
``/etc/os-release``, an adapter that cannot determine its solver version -- each yields a
null field and a warning in the job's event log. Reproducibility metadata is valuable, but
not more valuable than the run it describes.

**The environment is filtered, not dumped.** A full ``environ`` is large, churns for
irrelevant reasons, and can contain credentials. Adapters declare the keys that matter to
them; everything else is reduced to a hash, which still answers "were these two runs
identical?" exactly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from dispatch.adapters.base import CaseContext, SolverAdapter
from dispatch.core.clock import Clock, SystemClock
from dispatch.core.provenance import GitInfo, Provenance, hash_environment
from dispatch.version import __version__

__all__ = ["ProvenanceCollector"]

log = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 5.0
"""Cap on any external probe.

A case directory on a network mount can make ``git`` hang indefinitely, and the event loop
is not somewhere that can be allowed to happen.
"""


class ProvenanceCollector:
    """Builds a :class:`Provenance` for a job about to start.

    Args:
        clock: Time source.
        git: Path to the git binary. ``None`` disables version-control probing.
    """

    def __init__(self, *, clock: Clock | None = None, git: str = "git") -> None:
        self._clock = clock or SystemClock()
        self._git = git
        self._machine: dict[str, object] | None = None
        self._solver_versions: dict[str, str | None] = {}

    async def capture(
        self,
        *,
        ctx: CaseContext,
        adapter: SolverAdapter | None,
        argv: Sequence[str],
        env: Mapping[str, str],
        warn: object = None,
    ) -> Provenance:
        """Collect everything known about this run, at this instant.

        Args:
            ctx: The case being run.
            adapter: Adapter handling it, asked for its solver version.
            argv: The exact solve command, recorded verbatim.
            env: The environment the job will run in.
            warn: Optional callable taking a message, used to record probe failures in the
                job's event log.

        Returns:
            A record with nulls wherever a probe failed. Never raises.
        """
        machine = self._machine_facts()
        git_info = await self._git_info(ctx.workdir, warn=warn)
        solver_version = self._solver_version(adapter, ctx, warn=warn)
        declared = tuple(getattr(adapter, "env_keys", ()) or ())

        return Provenance(
            captured_at=self._clock.now(),
            dispatch_version=__version__,
            python_version=platform.python_version(),
            hostname=str(machine["hostname"]),
            kernel_version=str(machine["kernel"]),
            os_release=machine["os_release"],  # type: ignore[arg-type]
            cpu_model=machine["cpu_model"],  # type: ignore[arg-type]
            total_ram_mb=machine["total_ram_mb"],  # type: ignore[arg-type]
            solver_name=getattr(adapter, "name", None),
            solver_version=solver_version,
            adapter_version=getattr(adapter, "adapter_version", None),
            git=git_info,
            env_snapshot={key: env[key] for key in declared if key in env},
            env_hash=hash_environment(env),
            argv=tuple(argv),
        )

    # -- machine ---------------------------------------------------------------------------

    def _machine_facts(self) -> dict[str, object]:
        """Facts that cannot change while the daemon runs, so read once and cache."""
        if self._machine is None:
            uname = os.uname()
            self._machine = {
                "hostname": uname.nodename,
                "kernel": uname.release,
                "os_release": _os_release(),
                "cpu_model": _cpu_model(),
                "total_ram_mb": _total_ram_mb(),
            }
        return self._machine

    # -- solver ------------------------------------------------------------------------------

    def _solver_version(
        self, adapter: SolverAdapter | None, ctx: CaseContext, *, warn: object
    ) -> str | None:
        """Ask the adapter for its solver build, caching per daemon lifetime.

        Solver versions do not change under a running daemon often enough to justify
        re-running ``--version`` for every job in a hundred-job sweep.
        """
        if adapter is None:
            return None
        name = getattr(adapter, "name", "")
        if name in self._solver_versions:
            return self._solver_versions[name]
        try:
            version = adapter.solver_version(ctx)
        except Exception as exc:
            _warn(warn, f"could not determine solver version: {exc}")
            version = None
        self._solver_versions[name] = version
        return version

    # -- git ------------------------------------------------------------------------------------

    async def _git_info(self, workdir: Path, *, warn: object) -> GitInfo:
        """Read the case directory's version-control state.

        The *case*, not Dispatch: six months later the question is what the case looked
        like when it ran, and ``dirty`` is the field that will actually save you.
        """
        if not self._git:
            return GitInfo()

        inside = await self._run_git(workdir, "rev-parse", "--is-inside-work-tree")
        if inside != "true":
            return GitInfo()

        commit = await self._run_git(workdir, "rev-parse", "HEAD")
        if commit is None:
            _warn(warn, "case directory is a git repository but HEAD could not be read")
            return GitInfo()

        branch = await self._run_git(workdir, "rev-parse", "--abbrev-ref", "HEAD")
        status = await self._run_git(workdir, "status", "--porcelain")
        remote = await self._run_git(workdir, "remote", "get-url", "origin")

        return GitInfo(
            commit=commit,
            branch=branch if branch and branch != "HEAD" else None,
            dirty=bool(status),
            remote=remote,
        )

    async def _run_git(self, workdir: Path, *args: str) -> str | None:
        """Run one git command, returning stripped stdout or ``None`` on any failure."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._git,
                "-C",
                str(workdir),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except (OSError, ValueError):
            return None

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), PROBE_TIMEOUT_S)
        except TimeoutError:
            log.warning("git %s timed out in %s", " ".join(args), workdir)
            process.kill()
            return None
        except Exception:  # pragma: no cover - defensive
            return None

        if process.returncode != 0:
            return None
        return stdout.decode(errors="replace").strip() or None


def _warn(warn: object, message: str) -> None:
    if callable(warn):
        warn(message)
    else:
        log.debug("provenance: %s", message)


def _os_release() -> str | None:
    """The distribution's pretty name, e.g. ``Debian GNU/Linux 13 (trixie)``."""
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.partition("=")[2].strip().strip('"')
    except OSError:
        pass
    return None


def _cpu_model() -> str | None:
    """The first CPU model name from ``/proc/cpuinfo``."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or None


def _total_ram_mb() -> int | None:
    """Total system memory in megabytes."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):  # pragma: no cover - unavailable on exotic kernels
        return None
    return int(pages * page_size / (1024 * 1024))


def interpreter_summary() -> str:
    """A one-line description of the running interpreter, for ``dispatch doctor``."""
    return f"{platform.python_implementation()} {platform.python_version()} at {sys.executable}"
