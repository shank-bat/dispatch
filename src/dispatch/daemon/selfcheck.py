"""Environment diagnosis: what works here, what will not, and what to do about it.

Backs ``dispatch doctor`` and the single startup warning the daemon emits.

The logout question is the reason this module exists. Dispatch must be installable on any
Debian box without the user learning what ``loginctl`` is, so rather than mandating
``enable-linger``, the daemon detects the one configuration in which it would not survive
logout and says so -- then starts anyway, because the user may well be about to stay
logged in.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = ["Check", "CheckStatus", "run_checks", "survives_logout"]

log = logging.getLogger(__name__)


class CheckStatus(StrEnum):
    """Outcome of one diagnostic."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnostic result."""

    name: str
    status: CheckStatus
    detail: str
    remedy: str | None = None

    @property
    def symbol(self) -> str:
        """A single character for compact output."""
        return {CheckStatus.OK: "+", CheckStatus.WARN: "!", CheckStatus.FAIL: "x"}[self.status]


def run_checks(config: object | None = None) -> list[Check]:
    """Run every diagnostic and return the results.

    Never raises: this is what a user runs *because* something is wrong, and it failing
    would be worse than useless.
    """
    checks: list[Check] = [
        _python_check(),
        _sqlite_check(),
        _pidfd_check(),
        _resource_check(),
        _linger_check(),
    ]
    if config is not None:
        checks.extend(_path_checks(config))
    return checks


def _python_check() -> Check:
    """Report the interpreter.

    No version gate here: ``requires-python`` means an older interpreter cannot install
    Dispatch at all, so a runtime check could never fire.
    """
    import platform
    import sys

    return Check(
        "python",
        CheckStatus.OK,
        f"Python {platform.python_version()} at {sys.executable}",
    )


def _sqlite_check() -> Check:
    """Verify the two SQLite features the schema depends on.

    Both are compile-time options. Finding out at the first search -- months in -- would be
    far worse than finding out now. The probe itself lives with the rest of the database
    knowledge rather than being duplicated here.
    """
    from dispatch.db.connection import missing_capabilities

    missing = missing_capabilities()
    if missing:
        return Check(
            "sqlite",
            CheckStatus.FAIL,
            f"SQLite {sqlite3.sqlite_version} is missing {' and '.join(missing)}",
            "Install a Python built against a full SQLite, or rebuild SQLite with "
            "-DSQLITE_ENABLE_FTS5 -DSQLITE_ENABLE_JSON1.",
        )
    return Check("sqlite", CheckStatus.OK, f"SQLite {sqlite3.sqlite_version} with FTS5 and JSON1")


def _resource_check() -> Check:
    """What Dispatch will schedule against.

    Reported rather than judged. Zero GPUs is the correct and common answer, and the check
    exists so that a user whose GPU job is refused can see immediately whether Dispatch
    can see the card at all -- which is otherwise a confusing thing to have to guess.
    """
    from dispatch.core.config import installed_gpus, physical_cores

    cores = physical_cores() or os.cpu_count() or 1
    gpus = installed_gpus()
    detail = f"{cores} physical cores, {gpus} GPU(s) detected"
    if gpus:
        return Check("resources", CheckStatus.OK, detail)
    return Check(
        "resources",
        CheckStatus.OK,
        detail,
        remedy="Set [scheduler] total_gpus in the configuration if this machine has GPUs.",
    )


def _pidfd_check() -> Check:
    """Whether re-adopted jobs can be waited on rather than polled."""
    if not hasattr(os, "pidfd_open"):
        return Check(
            "pidfd",
            CheckStatus.WARN,
            "os.pidfd_open is unavailable",
            "Re-adopted jobs will be polled every 2s instead. Nothing else is affected.",
        )
    return Check("pidfd", CheckStatus.OK, "pidfd_open available for re-adopted jobs")


def survives_logout() -> tuple[bool, str]:
    """Whether a detached daemon would survive this user logging out.

    Two things decide it: logind's ``KillUserProcesses`` setting, and whether this user has
    lingering enabled. The Debian and Ubuntu default is ``KillUserProcesses=no``, under
    which a detached daemon survives with no systemd involvement at all -- which is why
    linger is a recommendation here and not a requirement.
    """
    kill_user_processes = _logind_kill_user_processes()
    if kill_user_processes is False:
        return True, "logind is configured with KillUserProcesses=no"
    if _lingering():
        return True, "lingering is enabled for this user"
    if kill_user_processes is None:
        return True, "logind configuration not found; assuming processes survive logout"
    return False, "logind has KillUserProcesses=yes and lingering is not enabled"


def _linger_check() -> Check:
    survives, reason = survives_logout()
    if survives:
        return Check("logout", CheckStatus.OK, f"the daemon survives logout: {reason}")
    user = os.environ.get("USER", "$USER")
    return Check(
        "logout",
        CheckStatus.WARN,
        f"the daemon would be killed at logout: {reason}",
        f"Run `loginctl enable-linger {user}`, or start the daemon with "
        "`systemd-run --user --scope dispatchd`. Running jobs are unaffected either way "
        "while you stay logged in.",
    )


def _logind_kill_user_processes() -> bool | None:
    """Read ``KillUserProcesses`` from logind's configuration.

    Returns ``None`` when no configuration file states it, which means the compiled-in
    default applies -- and on Debian-family systems that default is ``no``.
    """
    candidates = [Path("/etc/systemd/logind.conf")]
    for directory in (
        Path("/etc/systemd/logind.conf.d"),
        Path("/usr/lib/systemd/logind.conf.d"),
    ):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*.conf")))

    result: bool | None = None
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if key.strip() == "KillUserProcesses":
                result = value.strip().lower() in ("yes", "true", "1", "on")
    return result


def _lingering() -> bool:
    """Whether this user has a lingering systemd user manager."""
    user = os.environ.get("USER") or ""
    if user and Path(f"/var/lib/systemd/linger/{user}").exists():
        return True
    if not shutil.which("loginctl"):
        return False
    import subprocess

    try:
        result = subprocess.run(
            ["loginctl", "show-user", user or str(os.getuid()), "--property=Linger"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "Linger=yes" in result.stdout


def _path_checks(config: object) -> list[Check]:
    """Verify the daemon's directories are usable."""
    paths = getattr(config, "paths", None)
    if paths is None:  # pragma: no cover - defensive
        return []

    checks: list[Check] = []
    for label, path in (
        ("database", Path(paths.database).parent),
        ("logs", Path(paths.log_dir)),
        ("runtime", Path(paths.runtime_dir)),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".dispatch-write-test"
            probe.touch()
            probe.unlink()
        except OSError as exc:
            checks.append(Check(label, CheckStatus.FAIL, f"{path} is not writable: {exc}", None))
        else:
            checks.append(Check(label, CheckStatus.OK, str(path)))

    usage = _free_mb(Path(paths.log_dir))
    if usage is not None and usage < 1024:
        checks.append(
            Check(
                "disk",
                CheckStatus.WARN,
                f"only {usage} MB free where logs are written",
                "Dispatch stops admitting jobs below the configured floor.",
            )
        )
    return checks


def _free_mb(path: Path) -> int | None:
    try:
        return shutil.disk_usage(path).free // (1024 * 1024)
    except OSError:
        return None
