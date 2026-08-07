"""Adapter discovery, registration, and detection ranking.

Built-in adapters register at import. Third-party ones are found through the
``dispatch.adapters`` entry-point group, so ``uv pip install dispatch-calculix`` is the
whole integration story.

Registration is where :data:`~dispatch.adapters.base.ADAPTER_API_VERSION` is enforced, and
where a bad adapter is contained: a plugin that declares the wrong version, or explodes on
import, is recorded in :attr:`AdapterRegistry.rejected` and skipped. The daemon starts.
Losing the queue because someone's plugin is stale would be a much worse failure than
losing the plugin.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from dispatch.adapters.base import ADAPTER_API_VERSION, CaseContext, SolverAdapter
from dispatch.core.errors import AdapterError
from dispatch.core.models import Detection

__all__ = ["AdapterRegistry", "RejectedAdapter", "build_default_registry"]

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "dispatch.adapters"


@dataclass(frozen=True, slots=True)
class RejectedAdapter:
    """An adapter that could not be loaded, and why. Surfaced by ``dispatch doctor``."""

    name: str
    source: str
    reason: str


class AdapterRegistry:
    """Holds the adapters this daemon knows about.

    Args:
        settings: Per-adapter configuration, i.e. the ``[adapters]`` table.
    """

    def __init__(self, settings: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._classes: dict[str, type[SolverAdapter]] = {}
        self._instances: dict[str, SolverAdapter] = {}
        self._settings = dict(settings or {})
        self.rejected: list[RejectedAdapter] = []

    # -- registration -------------------------------------------------------------------

    def register(self, adapter_cls: type[SolverAdapter], *, source: str = "built-in") -> bool:
        """Add an adapter class.

        Returns:
            ``True`` if it was registered, ``False`` if it was rejected. Rejection is never
            an exception: one bad plugin must not prevent the daemon from starting.
        """
        name = getattr(adapter_cls, "name", "")
        if not name:
            self._reject("<unnamed>", source, "adapter class has no `name`")
            return False

        declared = getattr(adapter_cls, "api_version", None)
        if declared != ADAPTER_API_VERSION:
            self._reject(
                name,
                source,
                f"declares adapter API version {declared!r}, but this Dispatch speaks "
                f"version {ADAPTER_API_VERSION}. The adapter needs updating.",
            )
            return False

        if name in self._classes:
            self._reject(name, source, f"an adapter named {name!r} is already registered")
            return False

        self._classes[name] = adapter_cls
        log.debug("Registered adapter %s (%s)", name, source)
        return True

    def load_entry_points(self) -> None:
        """Discover third-party adapters from the ``dispatch.adapters`` group."""
        try:
            entries = importlib_metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:  # pragma: no cover - importlib failure is environmental
            log.warning("Could not read adapter entry points: %s", exc)
            return

        for entry in entries:
            source = f"entry point {entry.name!r}"
            try:
                adapter_cls = entry.load()
            except Exception as exc:
                # Deliberately broad: a third-party import can raise anything at all, and
                # whatever it raises must not reach the event loop.
                self._reject(entry.name, source, f"failed to import: {exc}")
                continue
            self.register(adapter_cls, source=source)

    def _reject(self, name: str, source: str, reason: str) -> None:
        self.rejected.append(RejectedAdapter(name=name, source=source, reason=reason))
        log.warning("Adapter %s (%s) not loaded: %s", name, source, reason)

    # -- access -------------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._classes

    def __len__(self) -> int:
        return len(self._classes)

    def __iter__(self) -> Iterator[SolverAdapter]:
        return (self.get(name) for name in sorted(self._classes))

    @property
    def names(self) -> Sequence[str]:
        """Registered adapter names, sorted."""
        return tuple(sorted(self._classes))

    def get(self, name: str) -> SolverAdapter:
        """Return the adapter instance for ``name``.

        Instances are cached: adapters may memoise expensive environment probes, and
        rebuilding one per job would throw that away.

        Raises:
            AdapterError: If no adapter is registered under that name.
        """
        if name not in self._classes:
            known = ", ".join(self.names) or "none"
            raise AdapterError(
                f"No solver adapter named {name!r}. Registered adapters: {known}",
                detail={"requested": name, "available": list(self.names)},
            )
        if name not in self._instances:
            adapter_cls = self._classes[name]
            self._instances[name] = adapter_cls(self._settings.get(name, {}))  # type: ignore[call-arg]
        return self._instances[name]

    def specs(self) -> Mapping[str, Any]:
        """Metadata specs by adapter name, for the repository's typed index."""
        return {name: self._classes[name].metadata_spec for name in self._classes}

    # -- detection ------------------------------------------------------------------------

    def detect(self, path: Path) -> Sequence[Detection]:
        """Ask every adapter whether ``path`` is its kind of case.

        Returns:
            Detections sorted by descending confidence. An adapter that raises is skipped
            with a warning rather than failing the whole detection -- the user is trying to
            browse a directory, and one broken adapter should not make that impossible.
        """
        found: list[Detection] = []
        for name in sorted(self._classes):
            adapter_cls = self._classes[name]
            try:
                detection = adapter_cls.detect(path)
            except Exception as exc:
                log.warning("Adapter %s raised while detecting %s: %s", name, path, exc)
                continue
            if detection is not None:
                found.append(detection)
        return tuple(sorted(found, key=lambda d: d.confidence, reverse=True))

    def best_detection(self, path: Path, *, margin: float = 0.15) -> Detection | None:
        """Return the single unambiguous detection for ``path``, if there is one.

        A winner must lead the runner-up by ``margin``. Ambiguity returns ``None``, which
        is the only case that prompts the user to choose -- when detection succeeds
        cleanly, the solver is never asked about.
        """
        detections = self.detect(path)
        if not detections:
            return None
        if len(detections) == 1:
            return detections[0]
        if detections[0].confidence - detections[1].confidence >= margin:
            return detections[0]
        return None

    def context(
        self,
        path: Path,
        *,
        cores: int = 1,
        ram_mb: int | None = None,
        entry: Path | None = None,
        env: Mapping[str, str] | None = None,
        adapter: str | None = None,
        job_name: str = "",
    ) -> CaseContext:
        """Build a :class:`CaseContext` with the right settings section attached."""
        return CaseContext(
            workdir=path,
            cores=cores,
            ram_mb=ram_mb,
            entry=entry,
            env=dict(env or {}),
            settings=self._settings.get(adapter or "", {}),
            job_name=job_name or path.name,
        )


def build_default_registry(
    settings: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    load_plugins: bool = True,
) -> AdapterRegistry:
    """Construct a registry with every built-in adapter, plus discovered plugins.

    The imports are local so that this module -- which the daemon's wiring imports at
    startup -- does not drag in every solver adapter simply by being imported.
    """
    from dispatch.adapters.basilisk import BasiliskAdapter
    from dispatch.adapters.calculix import CalculiXAdapter
    from dispatch.adapters.openfoam import OpenFOAMAdapter
    from dispatch.adapters.su2 import SU2Adapter

    registry = AdapterRegistry(settings)
    for adapter_cls in (OpenFOAMAdapter, SU2Adapter, BasiliskAdapter, CalculiXAdapter):
        registry.register(adapter_cls)
    if load_plugins:
        registry.load_entry_points()
    return registry
