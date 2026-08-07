"""Capturing an environment from a shell script that must be sourced.

Some solvers exist only after a setup script has been sourced -- OpenFOAM's ``etc/bashrc``
is the archetype, defining a hundred variables and putting the solver binaries on the PATH.
Dispatch cannot source a script into its own process, and re-sourcing it for every job
would mean a shell invocation per job.

So it is sourced **once per daemon lifetime**, its resulting environment is captured, and
that dictionary is handed to every job that needs it.

``env -0`` is used rather than plain ``env`` because environment values legitimately
contain newlines (OpenFOAM's own aliases do), and splitting on them would corrupt exactly
the variables that matter.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

__all__ = ["capture", "clear_cache"]

log = logging.getLogger(__name__)

SOURCE_TIMEOUT_S = 30.0
"""Generous: a cold NFS-mounted solver installation can genuinely take this long."""

_cache: dict[str, dict[str, str]] = {}


def capture(
    script: Path,
    *,
    shell: str = "/bin/bash",
    base: dict[str, str] | None = None,
    use_cache: bool = True,
) -> dict[str, str]:
    """Source ``script`` in a login shell and return the resulting environment.

    Args:
        script: The setup script to source.
        shell: Shell to use. Bash by default, because these scripts are almost always
            bash-specific in practice.
        base: Environment to start from. Defaults to the daemon's own.
        use_cache: Reuse a previously captured result for the same script.

    Returns:
        The environment after sourcing. On any failure, ``base`` unchanged -- a solver that
        cannot be set up produces a clear validation error later, which is far more useful
        than an exception here.
    """
    key = str(script)
    if use_cache and key in _cache:
        return dict(_cache[key])

    environment = dict(base if base is not None else os.environ)
    if not script.exists():
        log.warning("Environment script %s does not exist", script)
        return environment

    # `-l` so login-shell profile setup runs; stdout of the script itself is discarded so
    # that a chatty setup script cannot corrupt the NUL-separated output.
    command = f'. "{script}" >/dev/null 2>&1 && env -0'
    try:
        result = subprocess.run(
            [shell, "-lc", command],
            capture_output=True,
            timeout=SOURCE_TIMEOUT_S,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Could not source %s: %s", script, exc)
        return environment

    if result.returncode != 0:
        log.warning(
            "Sourcing %s failed with exit code %d: %s",
            script,
            result.returncode,
            result.stderr.decode(errors="replace").strip()[:200],
        )
        return environment

    captured = _parse(result.stdout)
    if not captured:
        log.warning("Sourcing %s produced no environment", script)
        return environment

    log.info("Captured %d environment variables from %s", len(captured), script)
    if use_cache:
        _cache[key] = dict(captured)
    return captured


def _parse(raw: bytes) -> dict[str, str]:
    """Parse NUL-separated ``KEY=VALUE`` records."""
    environment: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        key, sep, value = record.partition(b"=")
        if not sep:
            continue
        environment[key.decode(errors="replace")] = value.decode(errors="replace")
    return environment


def clear_cache() -> None:
    """Forget every captured environment. Used on configuration reload and in tests."""
    _cache.clear()


def find_first(candidates: list[Path]) -> Path | None:
    """Return the first path that exists, for probing well-known install locations."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
