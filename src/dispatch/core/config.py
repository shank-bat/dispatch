"""Configuration: defaults, TOML loading, and path resolution.

Dispatch runs with no configuration file at all. Every value here has a default that works
on a normal Linux workstation, and ``~/.config/dispatch/config.toml`` overrides only what
the user cares about.

Unknown keys are an error rather than being ignored. A typo in ``config.toml`` that
silently leaves the machine scheduling differently than the file says is the kind of bug
that gets discovered six months later, while wondering why a large job never ran (§13.16).
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Final, Self

from dispatch.core.errors import ConfigError

__all__ = [
    "Config",
    "DaemonConfig",
    "NotificationConfig",
    "PathsConfig",
    "SchedulerConfig",
    "load_config",
    "physical_cores",
]

APP_NAME: Final = "dispatch"


def physical_cores() -> int | None:
    """The machine's physical core count, ignoring SMT siblings.

    ``None`` when it cannot be determined, which is a reason to fall back to the logical
    count rather than to guess. psutil is imported lazily because this is called a handful
    of times per daemon lifetime and ``config`` is imported by the CLI on every invocation.
    """
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:  # pragma: no cover - psutil is a hard dependency, but never fatal
        return None


def _xdg(var: str, default: str) -> Path:
    """Resolve an XDG base directory, honouring the environment variable when set."""
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else Path.home() / default


def _data_home() -> Path:
    """Dispatch's directory under the XDG data home."""
    return _xdg("XDG_DATA_HOME", ".local/share") / APP_NAME


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Where Dispatch keeps its state.

    The runtime directory defaults under ``~/.local/run`` rather than ``$XDG_RUNTIME_DIR``
    (``/run/user/1000``), because the latter is cleared at logout on some configurations
    and the daemon has to outlive logout (§2.2).
    """

    database: Path = field(
        default_factory=lambda: _data_home() / "dispatch.db",
    )
    log_dir: Path = field(default_factory=lambda: _data_home() / "logs")
    runtime_dir: Path = field(default_factory=lambda: Path.home() / ".local/run" / APP_NAME)

    @property
    def socket(self) -> Path:
        """The daemon's Unix socket."""
        return self.runtime_dir / "daemon.sock"

    @property
    def pidfile(self) -> Path:
        """The flock'd single-instance guard."""
        return self.runtime_dir / "daemon.pid"

    @property
    def job_log_dir(self) -> Path:
        """Root of the per-job log tree."""
        return self.log_dir / "jobs"

    @property
    def daemon_log(self) -> Path:
        """The daemon's own log file."""
        return self.log_dir / "daemon.log"

    def job_dir(self, job_id: str) -> Path:
        """Log directory for one job."""
        return self.job_log_dir / job_id


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Admission policy and resource reservations."""

    policy: str = "backfill"
    """Named policy from the registry: ``fifo`` or ``backfill``.

    An unknown name is a startup error listing the valid ones, never a silent fallback.
    """

    reserved_cores: int = 1
    """Cores held back from jobs so the machine stays responsive over SSH.

    One is enough on a workstation; the alternative is a box you cannot log in to in order
    to cancel the job that is making it unresponsive.
    """

    total_cores: int | None = None
    """Override the detected core count. ``None`` means the machine's *physical* cores."""

    ram_margin_mb: int = 2048
    """RAM kept free above the sum of running jobs' estimates."""

    min_free_disk_mb: int = 1024
    """Admission stops below this much free space on the log filesystem.

    A solver that fills the disk mid-run loses its output and may corrupt its case; not
    starting is strictly better.
    """

    heartbeat_s: float = 30.0
    """Safety re-check interval. The scheduler is event-driven; this only catches drift."""

    def __post_init__(self) -> None:
        if self.reserved_cores < 0:
            raise ConfigError(f"scheduler.reserved_cores must be >= 0, got {self.reserved_cores}")
        if self.total_cores is not None and self.total_cores < 1:
            raise ConfigError(f"scheduler.total_cores must be >= 1, got {self.total_cores}")
        if self.heartbeat_s <= 0:
            raise ConfigError(f"scheduler.heartbeat_s must be positive, got {self.heartbeat_s}")

    def resolve_total_cores(self) -> int:
        """The core count to schedule against.

        **Physical** cores, not the logical count ``os.cpu_count()`` reports. Two
        independent reasons, and they happen to give the same answer:

        * MPI agrees with this number and not the other one. Open MPI sizes its default
          slot count by physical cores, so a job launched with more ranks than the machine
          has cores dies instantly with "not enough slots" -- before the solver runs at
          all. Scheduling against the logical count on any SMT machine therefore admits
          jobs that cannot start.
        * It is the right number anyway. CFD solvers are memory-bandwidth bound, so a
          second rank on the same physical core competes for the same cache and load/store
          units rather than adding throughput.

        A machine that genuinely wants hyperthreads sets ``scheduler.total_cores``
        explicitly; that override is honoured, and the MPI launch adapts to it (§8.7).
        """
        return self.total_cores or physical_cores() or os.cpu_count() or 1


@dataclass(frozen=True, slots=True)
class DaemonConfig:
    """Daemon lifecycle and supervision behaviour."""

    autostart: bool = True
    """Whether the TUI and CLI start a daemon when none is listening.

    On by default so that a new user runs ``dispatch`` and it works, with no systemd
    knowledge required (§2.1). Turn off where something else owns the lifecycle.
    """

    sample_interval_s: float = 2.0
    """Dashboard sampling period, active only while a client is subscribed."""

    job_sample_interval_s: float = 30.0
    """Per-job resource sampling period while a job runs."""

    progress_interval_s: float = 5.0
    """How often a running job's log is read for its current time step.

    Much shorter than the resource sampling period above, and affordable for a different
    reason: this is one ``pread`` of the last few kilobytes of a file, whereas a resource
    sample walks the whole process tree. Tying the two together would mean either walking
    process trees every five seconds or watching a solver's time step update twice a
    minute, and neither is a good trade.
    """

    cancel_grace_s: float = 10.0
    """Seconds between rungs of the SIGINT -> SIGTERM -> SIGKILL cancellation ladder."""

    shutdown_kills_jobs: bool = False
    """Whether stopping the daemon also stops simulations.

    Default ``False``, and changing it should feel like a decision: stopping Dispatch and
    stopping a week-long run are different actions (§13.13).
    """

    sample_retention_days: int = 7
    """Full-resolution retention for per-job samples; older ones are downsampled."""

    def __post_init__(self) -> None:
        for name in (
            "sample_interval_s",
            "job_sample_interval_s",
            "progress_interval_s",
            "cancel_grace_s",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"daemon.{name} must be positive")


@dataclass(frozen=True, slots=True)
class NotificationConfig:
    """Which events reach which sinks (§6.10)."""

    on: Sequence[str] = ("job.completed", "job.failed")
    sinks: Sequence[str] = ("log",)
    options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Per-sink settings, e.g. ``{"ntfy": {"topic": "eddy-dispatch"}}``."""


@dataclass(frozen=True, slots=True)
class Config:
    """The complete configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    adapters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Per-adapter settings, e.g. ``{"openfoam": {"bashrc": "/opt/openfoam/etc/bashrc"}}``.

    Not validated here: this layer must not know which adapters exist. Each adapter reads
    and validates its own section.
    """

    source: Path | None = None
    """The file this was loaded from, or ``None`` for pure defaults."""

    @classmethod
    def default_path(cls) -> Path:
        """The standard configuration file location."""
        return _xdg("XDG_CONFIG_HOME", ".config") / APP_NAME / "config.toml"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: Path | None = None) -> Self:
        """Build a config from parsed TOML, rejecting unknown keys."""
        known = {"paths", "scheduler", "daemon", "notifications", "adapters"}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"Unknown configuration section(s): {', '.join(sorted(unknown))}. "
                f"Valid sections: {', '.join(sorted(known))}",
                detail={"unknown": sorted(unknown)},
            )
        return cls(
            paths=_build(PathsConfig, data.get("paths", {}), "paths"),
            scheduler=_build(SchedulerConfig, data.get("scheduler", {}), "scheduler"),
            daemon=_build(DaemonConfig, data.get("daemon", {}), "daemon"),
            notifications=_build(
                NotificationConfig, data.get("notifications", {}), "notifications"
            ),
            adapters=dict(data.get("adapters", {})),
            source=source,
        )


def load_config(path: Path | None = None, *, required: bool = False) -> Config:
    """Load configuration from disk.

    Args:
        path: File to read. Defaults to :meth:`Config.default_path`.
        required: When ``True``, a missing file is an error. When ``False`` (the default)
            it simply yields defaults -- Dispatch must work before the user has written
            any configuration at all.

    Raises:
        ConfigError: If the file is unreadable, malformed, or contains unknown keys.
    """
    target = path or Config.default_path()
    if not target.exists():
        if required:
            raise ConfigError(f"Configuration file not found: {target}")
        return Config()

    try:
        with target.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{target}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{target}: cannot read configuration: {exc}") from exc

    return Config.from_mapping(data, source=target)


def _build[T](cls: type[T], data: Mapping[str, Any], section: str) -> T:
    """Instantiate a config dataclass from a TOML table, with clear errors.

    Handles the two conversions TOML cannot express: strings to :class:`Path`, and lists
    to tuples for the frozen dataclasses' sequence fields.
    """
    if not isinstance(data, Mapping):
        raise ConfigError(f"[{section}] must be a table, got {type(data).__name__}")
    if not is_dataclass(cls):  # pragma: no cover - programming error, not user error
        raise TypeError(f"{cls} is not a dataclass")

    declared = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(declared)
    if unknown:
        raise ConfigError(
            f"[{section}]: unknown key(s) {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(declared))}",
            detail={"section": section, "unknown": sorted(unknown)},
        )

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        kwargs[key] = _coerce(value, declared[key].type, f"{section}.{key}")
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"[{section}]: {exc}") from exc


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    """Convert a TOML value to the type a dataclass field declares.

    ``from __future__ import annotations`` means field annotations arrive as strings, so
    they are matched textually. That is coarse, but the config surface is small, fixed,
    and fully covered by tests -- a resolver using :func:`typing.get_type_hints` would be
    more general and would earn nothing here.
    """
    declared = str(annotation)

    if "Path" in declared:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a path string, got {value!r}")
        return Path(value).expanduser()

    if "Sequence" in declared:
        if isinstance(value, list):
            return tuple(value)
        # A bare string where a list is expected is a common and harmless slip.
        if isinstance(value, str):
            return (value,)
        raise ConfigError(f"{where}: expected a list, got {value!r}")

    if "bool" in declared:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value

    if ("int" in declared or "float" in declared) and isinstance(value, bool):
        # bool is an int subclass, so `reserved_cores = true` would otherwise mean 1.
        raise ConfigError(f"{where}: expected a number, got a boolean")

    return value
