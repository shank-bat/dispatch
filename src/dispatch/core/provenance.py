"""The reproducibility record captured for every job.

Six months after a run produced a figure, the questions are: what code was this, what
solver build, what machine, and had I committed my changes? :class:`Provenance` is the
answer, captured at the moment the solver is spawned.

This module is pure data. Collection -- running ``git``, reading ``/etc/os-release``,
asking an adapter for its solver version -- lives in ``dispatch.daemon.provenance``,
because it touches the world and this layer does not.

See ``docs/ARCHITECTURE.md`` §6.9.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GitInfo", "Provenance", "hash_environment"]


@dataclass(frozen=True, slots=True)
class GitInfo:
    """Version-control state of the *case directory* at launch.

    Not of Dispatch itself -- that is recorded separately in
    :attr:`Provenance.dispatch_version`. The question worth answering later is what the
    case looked like, and :attr:`dirty` is the field that will actually save you: a clean
    commit hash recorded against a modified working tree is a lie shaped like provenance.
    """

    commit: str | None = None
    branch: str | None = None
    dirty: bool = False
    remote: str | None = None

    @property
    def is_present(self) -> bool:
        """Whether the case directory was under version control at all."""
        return self.commit is not None

    def describe(self) -> str:
        """A one-line human summary, e.g. ``a1b2c3d (main, dirty)``."""
        if self.commit is None:
            return "not a git repository"
        short = self.commit[:7]
        parts = [self.branch] if self.branch else []
        if self.dirty:
            parts.append("dirty")
        return f"{short} ({', '.join(parts)})" if parts else short


@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything needed to understand what produced a job's output.

    Every field except the versions is optional: collection must never fail a job (§6.9),
    so a missing ``git`` binary or an adapter that cannot determine its solver version
    yields ``None`` here and a warning in the job's event log.
    """

    captured_at: float
    dispatch_version: str
    python_version: str
    hostname: str
    kernel_version: str

    os_release: str | None = None
    cpu_model: str | None = None
    total_ram_mb: int | None = None

    solver_name: str | None = None
    solver_version: str | None = None
    adapter_version: int | None = None

    git: GitInfo = field(default_factory=GitInfo)

    env_snapshot: Mapping[str, str] = field(default_factory=dict)
    """Adapter-declared environment variables only.

    A full ``environ`` dump would be large, would churn for irrelevant reasons
    (``SSH_AUTH_SOCK``, ``TMUX``), and could contain credentials. Adapters declare the
    keys that matter to them; everything else is reduced to :attr:`env_hash`.
    """

    env_hash: str | None = None
    """SHA-256 of the *complete* environment.

    Preserves the ability to answer "did these two runs have identical environments?"
    exactly, without storing the environments.
    """

    argv: Sequence[str] = ()
    """The exact SOLVE command executed, verbatim."""

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-serialisable form used by the IPC layer."""
        return {
            "captured_at": self.captured_at,
            "dispatch_version": self.dispatch_version,
            "python_version": self.python_version,
            "hostname": self.hostname,
            "kernel_version": self.kernel_version,
            "os_release": self.os_release,
            "cpu_model": self.cpu_model,
            "total_ram_mb": self.total_ram_mb,
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
            "adapter_version": self.adapter_version,
            "git": {
                "commit": self.git.commit,
                "branch": self.git.branch,
                "dirty": self.git.dirty,
                "remote": self.git.remote,
            },
            "env_snapshot": dict(self.env_snapshot),
            "env_hash": self.env_hash,
            "argv": list(self.argv),
        }


def hash_environment(env: Mapping[str, str]) -> str:
    """Return a stable SHA-256 digest of an entire environment mapping.

    Keys are sorted, so dictionary ordering cannot affect the result, and key and value
    are both terminated with NUL rather than joined with ``=``.

    The NUL matters. Joining with ``=`` would make ``{"A": "1=B"}`` and ``{"A=1": "B"}``
    hash identically -- both serialise to ``A=1=B`` -- and two genuinely different
    environments would compare as equal. POSIX forbids NUL in both names and values, so
    it is the one byte that cannot appear inside a field.
    """
    digest = hashlib.sha256()
    for key in sorted(env):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(env[key].encode())
        digest.update(b"\0")
    return digest.hexdigest()
