"""The search query language.

One input box accepts free text mixed with typed terms, because typing
``tag:paper solver:openfoam cores>=16`` beats operating four dropdowns::

    naca0018 tag:paper -tag:scratch state:completed endTime>500 after:2026-01

Parsing is pure and lives here; compiling a :class:`SearchQuery` into SQL is the
repository's job. That split is what makes the grammar exhaustively unit-testable without
a database.

A deliberate strictness: an unrecognised ``key:`` prefix is an error, not free text. If
``taag:paper`` silently degraded to a text search it would return a confident, wrong
answer to the question "which runs made this figure" -- and the user would never know.

See ``docs/ARCHITECTURE.md`` §5.2.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

from dispatch.core.errors import QueryError
from dispatch.core.states import JobState
from dispatch.core.tags import normalise_tag

__all__ = [
    "CompareOp",
    "Comparison",
    "SearchQuery",
    "parse_query",
]


class CompareOp(StrEnum):
    """Comparison operators available on numeric fields."""

    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    @property
    def sql(self) -> str:
        """The SQL spelling of this operator."""
        return "<>" if self is CompareOp.NE else self.value


class ValueKind(StrEnum):
    """How a comparison's right-hand side should be interpreted."""

    NUMBER = "number"
    DURATION = "duration"
    """Accepts ``90``, ``30m``, ``2h``, ``3d``. Normalised to seconds."""

    SIZE_MB = "size_mb"
    """Accepts ``512``, ``512M``, ``16G``. Normalised to megabytes."""


@dataclass(frozen=True, slots=True)
class Comparison:
    """A numeric constraint, on either a job column or a metadata field.

    Attributes:
        field: Column name for job comparisons, or the metadata key.
        op: The operator.
        value: Right-hand side, already normalised to the field's canonical unit.
        is_metadata: Whether ``field`` names a declared metadata key rather than a
            column. Metadata comparisons resolve against the derived ``job_metadata``
            index, which is why they can use an index at all (§13.18).
    """

    field: str
    op: CompareOp
    value: float
    is_metadata: bool = False


# Job columns that accept comparisons, mapped to (SQL column, value interpretation).
_COLUMNS: Final[dict[str, tuple[str, ValueKind]]] = {
    "cores": ("cores", ValueKind.NUMBER),
    "priority": ("priority", ValueKind.NUMBER),
    "ram": ("ram_estimate_mb", ValueKind.SIZE_MB),
    "runtime": ("runtime_s", ValueKind.DURATION),
    "rss": ("peak_rss_mb", ValueKind.SIZE_MB),
    "cpu": ("mean_cpu_pct", ValueKind.NUMBER),
    "exit": ("exit_code", ValueKind.NUMBER),
}

# `key:value` filters. Anything not listed here is a query error.
_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "tag",
        "solver",
        "state",
        "app",
        "name",
        "dir",
        "workdir",
        "id",
        "after",
        "before",
        "dirty",
        "meta",
    }
)

_COMPARISON_RE: Final = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)(>=|<=|!=|>|<|=)(.+)$")
_DURATION_RE: Final = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhdw]?)$", re.IGNORECASE)
_SIZE_RE: Final = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgt]?)b?$", re.IGNORECASE)

_DURATION_UNITS: Final = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}
_SIZE_UNITS_MB: Final = {"": 1.0, "k": 1 / 1024, "m": 1.0, "g": 1024.0, "t": 1024.0 * 1024}


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """A parsed query, ready for the repository to compile into SQL."""

    text: str = ""
    """Free-text terms, joined by spaces, for the FTS5 ``MATCH``."""

    tags_include: frozenset[str] = frozenset()
    tags_exclude: frozenset[str] = frozenset()
    states: frozenset[JobState] = frozenset()
    states_exclude: frozenset[JobState] = frozenset()
    solvers: frozenset[str] = frozenset()
    apps: frozenset[str] = frozenset()
    names: Sequence[str] = ()
    """Substring constraints on the job name."""

    dirs: Sequence[str] = ()
    """Substring constraints on the working directory."""

    ids: frozenset[str] = frozenset()
    """Exact or prefix job-id constraints."""

    created_after: float | None = None
    created_before: float | None = None
    dirty: bool | None = None
    comparisons: Sequence[Comparison] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        """Whether the query constrains nothing, and so matches every job."""
        return not any(
            (
                self.text,
                self.tags_include,
                self.tags_exclude,
                self.states,
                self.states_exclude,
                self.solvers,
                self.apps,
                self.names,
                self.dirs,
                self.ids,
                self.created_after is not None,
                self.created_before is not None,
                self.dirty is not None,
                self.comparisons,
            )
        )


def parse_query(raw: str, *, now: float) -> SearchQuery:
    """Parse a query string.

    Args:
        raw: The user's input.
        now: Current Unix time, used to resolve relative dates such as ``after:7d``.
            Injected rather than read from the clock so that tests are deterministic.

    Returns:
        The parsed query.

    Raises:
        QueryError: On an unknown keyword, a malformed value, or unbalanced quotes.
    """
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        raise QueryError(f"Could not parse query: {exc}") from exc

    acc = _Accumulator(now=now)
    for token in tokens:
        acc.consume(token)
    return acc.build()


class _Accumulator:
    """Mutable state while parsing. Kept private -- :class:`SearchQuery` is the interface."""

    __slots__ = (
        "apps",
        "comparisons",
        "created_after",
        "created_before",
        "dirs",
        "dirty",
        "ids",
        "names",
        "now",
        "solvers",
        "states",
        "states_exclude",
        "tags_exclude",
        "tags_include",
        "text",
    )

    def __init__(self, *, now: float) -> None:
        self.now = now
        self.text: list[str] = []
        self.tags_include: set[str] = set()
        self.tags_exclude: set[str] = set()
        self.states: set[JobState] = set()
        self.states_exclude: set[JobState] = set()
        self.solvers: set[str] = set()
        self.apps: set[str] = set()
        self.names: list[str] = []
        self.dirs: list[str] = []
        self.ids: set[str] = set()
        self.created_after: float | None = None
        self.created_before: float | None = None
        self.dirty: bool | None = None
        self.comparisons: list[Comparison] = []

    def consume(self, token: str) -> None:
        """Classify and absorb one whitespace-delimited token."""
        if not token:
            return

        negated = token.startswith("-") and len(token) > 1
        body = token[1:] if negated else token

        match = _COMPARISON_RE.match(body)
        if match:
            if negated:
                raise QueryError(f"Cannot negate a comparison: {token!r}; invert the operator")
            self._comparison(*match.groups())
            return

        if ":" in body:
            key, _, value = body.partition(":")
            self._keyword(key.lower(), value, negated=negated, token=token)
            return

        if negated:
            raise QueryError(
                f"Cannot negate free text: {token!r}; negation applies to tag:, state:, "
                "and solver: only"
            )
        self.text.append(body)

    # -- term handlers -----------------------------------------------------------------

    def _keyword(self, key: str, value: str, *, negated: bool, token: str) -> None:
        if key not in _KEYWORDS:
            raise QueryError(
                f"Unknown search field {key!r}. Valid fields: {', '.join(sorted(_KEYWORDS))}",
                detail={"field": key},
            )
        if not value:
            raise QueryError(f"Search term {token!r} is missing a value")

        match key:
            case "tag":
                target = self.tags_exclude if negated else self.tags_include
                target.add(normalise_tag(value))
            case "state":
                target_states = self.states_exclude if negated else self.states
                target_states.add(_parse_state(value))
            case "solver":
                self._no_negation(negated, token)
                self.solvers.add(value.lower())
            case "app":
                self._no_negation(negated, token)
                self.apps.add(value)
            case "name":
                self._no_negation(negated, token)
                self.names.append(value)
            case "dir" | "workdir":
                self._no_negation(negated, token)
                self.dirs.append(value)
            case "id":
                self._no_negation(negated, token)
                self.ids.add(value.lower())
            case "after":
                self._no_negation(negated, token)
                self.created_after = _parse_instant(value, now=self.now)
            case "before":
                self._no_negation(negated, token)
                self.created_before = _parse_instant(value, now=self.now)
            case "dirty":
                self._no_negation(negated, token)
                self.dirty = _parse_bool(value)
            case "meta":
                # `meta:key>value` -- the explicit form, for a metadata key that collides
                # with a job column name such as `cores`.
                inner = _COMPARISON_RE.match(value)
                if not inner:
                    raise QueryError(
                        f"meta: needs a comparison, e.g. meta:endTime>500 (got {token!r})"
                    )
                name, op, rhs = inner.groups()
                self.comparisons.append(
                    Comparison(name, CompareOp(op), _parse_number(rhs, name), is_metadata=True)
                )

    def _comparison(self, name: str, op: str, rhs: str) -> None:
        column = _COLUMNS.get(name.lower())
        if column is not None:
            sql_name, kind = column
            self.comparisons.append(
                Comparison(sql_name, CompareOp(op), _parse_value(rhs, kind, name))
            )
            return
        # Not a job column, so it is a metadata field. Metadata keys are declared by
        # adapters and cannot be enumerated here, so this cannot be validated at parse
        # time; the repository resolves it against the derived index and simply matches
        # nothing if the key was never produced.
        self.comparisons.append(
            Comparison(name, CompareOp(op), _parse_number(rhs, name), is_metadata=True)
        )

    def _no_negation(self, negated: bool, token: str) -> None:
        """Reject ``-`` on fields where exclusion has no useful meaning.

        Negation is supported for ``tag:`` and ``state:``, where "everything except" is a
        natural thing to want. For ``after:`` or ``name:`` it is not, and accepting it
        silently would be worse than saying so.
        """
        if negated:
            raise QueryError(
                f"{token!r} cannot be negated; negation applies to tag: and state: only"
            )

    def build(self) -> SearchQuery:
        """Freeze the accumulated state into a :class:`SearchQuery`."""
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after > self.created_before
        ):
            raise QueryError("The after: date is later than the before: date, so nothing can match")
        return SearchQuery(
            text=" ".join(self.text),
            tags_include=frozenset(self.tags_include),
            tags_exclude=frozenset(self.tags_exclude),
            states=frozenset(self.states),
            states_exclude=frozenset(self.states_exclude),
            solvers=frozenset(self.solvers),
            apps=frozenset(self.apps),
            names=tuple(self.names),
            dirs=tuple(self.dirs),
            ids=frozenset(self.ids),
            created_after=self.created_after,
            created_before=self.created_before,
            dirty=self.dirty,
            comparisons=tuple(self.comparisons),
        )


# -- value parsers ---------------------------------------------------------------------


def _parse_state(value: str) -> JobState:
    try:
        return JobState(value.upper())
    except ValueError as exc:
        valid = ", ".join(s.value.lower() for s in JobState)
        raise QueryError(f"Unknown job state {value!r}. Valid states: {valid}") from exc


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "yes", "1", "y"):
        return True
    if lowered in ("false", "no", "0", "n"):
        return False
    raise QueryError(f"Expected true or false, got {value!r}")


def _parse_number(raw: str, field_name: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise QueryError(f"{field_name}: expected a number, got {raw!r}") from exc


def _parse_value(raw: str, kind: ValueKind, field_name: str) -> float:
    match kind:
        case ValueKind.DURATION:
            return _parse_duration(raw, field_name)
        case ValueKind.SIZE_MB:
            return _parse_size_mb(raw, field_name)
        case _:
            return _parse_number(raw, field_name)


def _parse_duration(raw: str, field_name: str) -> float:
    """Parse ``90``, ``30m``, ``2h``, ``3d``, ``1w`` into seconds. Bare numbers are seconds."""
    match = _DURATION_RE.match(raw.strip())
    if not match:
        raise QueryError(
            f"{field_name}: expected a duration such as 90, 30m, 2h, or 3d; got {raw!r}"
        )
    amount, unit = match.groups()
    return float(amount) * _DURATION_UNITS[unit.lower()]


def _parse_size_mb(raw: str, field_name: str) -> float:
    """Parse ``512``, ``512M``, ``16G`` into megabytes. Bare numbers are megabytes."""
    match = _SIZE_RE.match(raw.strip())
    if not match:
        raise QueryError(f"{field_name}: expected a size such as 512, 512M, or 16G; got {raw!r}")
    amount, unit = match.groups()
    return float(amount) * _SIZE_UNITS_MB[unit.lower()]


def _parse_instant(raw: str, *, now: float) -> float:
    """Parse an absolute date or a relative offset into a Unix timestamp.

    Accepted absolute forms are ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD``, and
    ``YYYY-MM-DDTHH:MM``, each resolving to the *start* of the period named. Relative
    forms are durations meaning "ago": ``after:7d`` is seven days before now.

    Both ``after:`` and ``before:`` resolve to the start of the named period, so
    ``after:2026-01 before:2026-02`` is exactly January -- a composition that stops
    working the moment one of the two silently means "end of".
    """
    value = raw.strip()
    relative = _DURATION_RE.match(value)
    if relative and relative.group(2):
        return now - _parse_duration(value, "date")

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=UTC).timestamp()

    if value.lower() == "today":
        midnight = datetime.fromtimestamp(now, UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return midnight.timestamp()
    if value.lower() == "yesterday":
        midnight = datetime.fromtimestamp(now, UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (midnight - timedelta(days=1)).timestamp()

    raise QueryError(
        f"Could not read {raw!r} as a date. Use YYYY-MM-DD, YYYY-MM, YYYY, "
        "a relative offset such as 7d, or today/yesterday"
    )
