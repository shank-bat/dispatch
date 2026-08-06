"""Structured, adapter-declared case metadata.

A bare ``dict[str, Any]`` is where information goes to become unsearchable. Every adapter
instead declares a :class:`MetadataSpec` describing the fields it extracts, their types,
labels, and units. That single declaration drives three things at once:

* **Storage** -- values are type-checked and coerced on write.
* **Search** -- declared fields are indexed typed, so ``endTime>500`` is an index seek.
* **Display** -- the TUI renders a labelled, unit-annotated panel without knowing what
  OpenFOAM is.

Anything an adapter wants to keep but has not committed to a schema for goes in
:attr:`CaseMetadata.extra`, which is stored and displayed but not typed-searchable.

See ``docs/ARCHITECTURE.md`` §4.4.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from dispatch.core.errors import ValidationError

__all__ = [
    "CaseMetadata",
    "FieldType",
    "MetadataField",
    "MetadataSpec",
    "MetadataValue",
    "SpecRef",
]

MetadataValue = str | int | float | bool
"""The value types a declared metadata field may hold. JSON-representable by construction."""


class FieldType(StrEnum):
    """The declared type of a metadata field."""

    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    PATH = "path"
    """A filesystem path. Stored and searched as text; distinguished from STR so the TUI
    can shorten it for display (``~/projects/...``) rather than truncating mid-word."""

    @property
    def is_numeric(self) -> bool:
        """Whether values of this type support ``<``/``>`` comparisons in search."""
        return self in (FieldType.INT, FieldType.FLOAT)


@dataclass(frozen=True, slots=True)
class MetadataField:
    """One field an adapter promises to produce.

    Attributes:
        key: Machine name, used in search queries. Conventionally the solver's own
            spelling (``endTime``, not ``end_time``) so that searching matches what the
            user reads in ``controlDict``.
        type: Declared type. Values are coerced to it on write.
        label: Human-readable name for the TUI.
        unit: Physical or logical unit, appended in display. ``None`` when unitless.
        searchable: Whether to index this field for typed search. Set ``False`` for
            values that are long, high-cardinality, and never queried.
        display_order: Ascending sort key for the TUI metadata panel.
    """

    key: str
    type: FieldType
    label: str
    unit: str | None = None
    searchable: bool = True
    display_order: int = 100

    def coerce(self, value: object) -> MetadataValue:
        """Convert ``value`` to this field's declared type.

        Raises:
            ValidationError: If the value cannot be represented as the declared type.
        """
        try:
            match self.type:
                case FieldType.INT:
                    # bool is an int subclass; accepting it here would silently store
                    # True as 1 for a field the adapter declared numeric.
                    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                        raise TypeError(f"cannot read {type(value).__name__} as an integer")
                    return int(value)
                case FieldType.FLOAT:
                    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                        raise TypeError(f"cannot read {type(value).__name__} as a float")
                    return float(value)
                case FieldType.BOOL:
                    if isinstance(value, str):
                        lowered = value.strip().lower()
                        if lowered in ("true", "yes", "on", "1"):
                            return True
                        if lowered in ("false", "no", "off", "0"):
                            return False
                        raise ValueError(f"not a boolean: {value!r}")
                    return bool(value)
                case _:
                    return str(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Metadata field {self.key!r} expects {self.type}, got {value!r}",
                detail={"key": self.key, "expected": str(self.type)},
            ) from exc


@dataclass(frozen=True, slots=True)
class SpecRef:
    """Identifies the spec a stored metadata envelope was written against.

    Persisted with every job so that an adapter may rename or retype a field in a later
    revision without silently corrupting the interpretation of older history.
    """

    adapter: str
    version: int = 1


@dataclass(frozen=True, slots=True)
class MetadataSpec:
    """The complete set of fields an adapter declares.

    Args:
        ref: Adapter name and spec version.
        fields: Declared fields. Order is irrelevant; ``display_order`` governs the TUI.
    """

    ref: SpecRef
    fields: Sequence[MetadataField] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.fields:
            if item.key in seen:
                raise ValidationError(
                    f"Duplicate metadata field {item.key!r} in spec for {self.ref.adapter!r}"
                )
            seen.add(item.key)

    def __iter__(self) -> Iterator[MetadataField]:
        return iter(self.fields)

    def get(self, key: str) -> MetadataField | None:
        """Return the field declared under ``key``, or ``None``."""
        return next((f for f in self.fields if f.key == key), None)

    @property
    def ordered(self) -> Sequence[MetadataField]:
        """Fields sorted for display."""
        return sorted(self.fields, key=lambda f: (f.display_order, f.key))

    def build(self, values: Mapping[str, object], **extra: Any) -> CaseMetadata:
        """Construct a validated :class:`CaseMetadata` from raw adapter output.

        Keys present in ``values`` but absent from the spec are an error rather than a
        silent move to ``extra``: an adapter emitting an undeclared field is a bug in the
        adapter, and discovering it at write time is far cheaper than discovering it when
        a search returns nothing.

        Args:
            values: Declared field values. Missing fields are simply absent -- a case
                that does not set ``writeInterval`` should not store a fabricated one.
            **extra: Undeclared extras to preserve verbatim.

        Raises:
            ValidationError: On an undeclared key or an uncoercible value.
        """
        coerced: dict[str, MetadataValue] = {}
        for key, value in values.items():
            declared = self.get(key)
            if declared is None:
                raise ValidationError(
                    f"Adapter {self.ref.adapter!r} produced undeclared metadata key {key!r}",
                    detail={"key": key, "declared": [f.key for f in self.fields]},
                )
            if value is None:
                continue
            coerced[key] = declared.coerce(value)
        return CaseMetadata(spec=self.ref, case=coerced, extra=dict(extra))


EMPTY_SPEC: Final = MetadataSpec(ref=SpecRef(adapter="", version=0))
"""Placeholder for jobs whose adapter declares nothing (and for tests)."""


@dataclass(frozen=True, slots=True)
class CaseMetadata:
    """The stored metadata envelope for one job.

    Serialised to the ``jobs.metadata`` JSON column as::

        {"spec": {"adapter": "openfoam", "version": 1},
         "case": {"application": "interFoam", "endTime": 600.0},
         "extra": {}}
    """

    spec: SpecRef
    case: Mapping[str, MetadataValue] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, adapter: str = "") -> CaseMetadata:
        """An envelope with no values, for jobs whose metadata is not yet collected."""
        return cls(spec=SpecRef(adapter=adapter, version=0))

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-serialisable form written to the database."""
        return {
            "spec": {"adapter": self.spec.adapter, "version": self.spec.version},
            "case": dict(self.case),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> CaseMetadata:
        """Rebuild an envelope from stored JSON.

        Tolerant by design: this reads data written by *past* versions of Dispatch, and
        refusing to load a job because its envelope predates a field would make the
        history unreadable. Unrecognised shapes degrade to empty rather than raising.
        """
        spec_raw = payload.get("spec")
        spec = (
            SpecRef(
                adapter=str(spec_raw.get("adapter", "")),
                version=int(spec_raw.get("version", 0)),
            )
            if isinstance(spec_raw, Mapping)
            else SpecRef(adapter="", version=0)
        )
        case = payload.get("case")
        extra = payload.get("extra")
        return cls(
            spec=spec,
            case=dict(case) if isinstance(case, Mapping) else {},
            extra=dict(extra) if isinstance(extra, Mapping) else {},
        )

    def searchable_pairs(self, spec: MetadataSpec | None = None) -> Iterator[
        tuple[str, str | None, float | None]
    ]:
        """Yield ``(key, text_value, numeric_value)`` rows for the derived search index.

        Exactly one of the two value slots is populated per row, matching the
        ``job_metadata`` table's shape (§13.18). Numeric fields get the numeric slot so
        that ``endTime>500`` compares as a number, not as the string ``"600"`` -- which
        would place ``"1000"`` before ``"600"``.

        Args:
            spec: The adapter's spec, used to decide numeric vs text and to honour
                ``searchable=False``. When ``None``, types are inferred from the stored
                Python values, which is the right fallback for historical rows whose
                adapter is no longer installed.
        """
        for key, value in self.case.items():
            declared = spec.get(key) if spec is not None else None
            if declared is not None and not declared.searchable:
                continue
            numeric = declared.type.is_numeric if declared is not None else _looks_numeric(value)
            # A field the spec calls numeric may still hold text if it was written by an
            # older spec revision. Fall back to the text slot rather than raising: this
            # runs while indexing history, where a hard failure would be far worse than an
            # imprecise comparison for one stale row.
            if numeric and isinstance(value, (int, float)) and not isinstance(value, bool):
                yield key, None, float(value)
            else:
                yield key, _as_text(value), None

    def flatten_for_fts(self) -> str:
        """Render as ``key value key value ...`` for the full-text index.

        Both keys and values are emitted so that a bare search for ``interFoam`` and a
        bare search for ``application`` each find the job.
        """
        parts: list[str] = []
        for key, value in self.case.items():
            parts.append(key)
            parts.append(_as_text(value))
        for key, value in self.extra.items():
            parts.append(key)
            parts.append(_as_text(value))
        return " ".join(parts)


def _looks_numeric(value: object) -> bool:
    """Whether a stored value should be treated as a number without a spec to consult."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_text(value: object) -> str:
    """Render a metadata value as searchable text.

    Booleans become ``true``/``false`` rather than Python's ``True``/``False`` so that a
    query typed in the obvious lowercase form matches.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
