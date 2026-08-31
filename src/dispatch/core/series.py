"""Numerical series extracted from a job's own output.

The chain this module sits in the middle of::

    job -> its log -> the adapter's parser -> series -> X/Y choice -> terminal plot

Only the middle arrow is solver-specific, and it belongs to the adapter, exactly as
:class:`~dispatch.core.plan.ExecutionPlan` keeps command construction there. The daemon
reads bytes off a disk and hands them to an adapter; it does not know that a residual
exists. The interface renders numbers; it does not know that OpenFOAM does.

Two design points are worth stating, because the obvious shapes are both wrong:

**A flat pool of series, not one X with a list of Ys.** Pinning the independent variable
at parse time would decide for the user that iteration is the X axis, and then ``execution
time vs. iteration`` -- a genuinely useful plot -- could not be expressed at all. Every
series is a candidate for either axis; :attr:`Series.axis` only says which ones are
*likely* to be wanted as X, so the selector can put them first.

**Every value carries the sample it came from.** A solver that prints ``residual(p)`` on
every time step but ``continuity`` only when it feels like it produces two series of
different lengths, and zipping those together silently plots one quantity against the
wrong one. Pairing is an inner join on :attr:`Series.samples`, so a mismatch is visible
and correct rather than invisible and wrong.

See ``docs/ARCHITECTURE.md`` §9.6.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["PlotData", "Series", "align", "downsample", "series_from_records"]


@dataclass(frozen=True, slots=True)
class Series:
    """One named column of numbers read out of a job's output.

    Attributes:
        key: Stable identifier, e.g. ``residual(p)``. Used to remember a selection.
        label: What to show a human. Usually the same as ``key``.
        values: The numbers, in the order they appeared.
        samples: The record index each value came from, same length as ``values``. For a
            series present on every record this is simply ``0, 1, 2, ...``; for a
            sporadic one it has gaps, which is precisely the information that makes
            pairing honest.
        unit: Physical unit, when there is one.
        axis: Whether this reads naturally as an X axis -- iteration, simulated time,
            wall-clock time. Ordering only; any series may be chosen for either axis.
    """

    key: str
    label: str
    values: tuple[float, ...]
    samples: tuple[int, ...]
    unit: str | None = None
    axis: bool = False

    def __post_init__(self) -> None:
        if len(self.values) != len(self.samples):
            raise ValueError(
                f"Series {self.key!r} has {len(self.values)} values but "
                f"{len(self.samples)} sample indices"
            )

    def __len__(self) -> int:
        return len(self.values)

    @property
    def display(self) -> str:
        """The label with its unit, when it has one."""
        return f"{self.label} ({self.unit})" if self.unit else self.label

    @property
    def span(self) -> tuple[float, float] | None:
        """Smallest and largest value, or ``None`` when the series is empty."""
        if not self.values:
            return None
        return (min(self.values), max(self.values))

    @property
    def is_positive(self) -> bool:
        """Whether every value is strictly positive, i.e. whether a log axis is possible."""
        return bool(self.values) and all(value > 0.0 for value in self.values)


@dataclass(frozen=True, slots=True)
class PlotData:
    """Everything plottable that could be read out of one job's output.

    Attributes:
        series: The available columns, in the order the parser produced them.
        samples: How many records the parser saw. Zero means the log said nothing this
            parser recognised -- a first-class answer, not a failure.
        truncated: Whether only the end of the log was read, because it exceeded the
            configured limit. Shown to the user rather than hidden: a plot of the last
            fifth of a run is useful, and quietly presenting it as the whole run is not.
    """

    series: Sequence[Series] = field(default_factory=tuple)
    samples: int = 0
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.series)

    def __bool__(self) -> bool:
        return bool(self.series)

    def get(self, key: str) -> Series | None:
        """One series by key, or ``None`` if this log did not contain it.

        A log with no ``Uy`` residual has no ``residual(Uy)`` series, and asking for one
        is answered with ``None`` rather than an empty line on a chart.
        """
        return next((item for item in self.series if item.key == key), None)

    @property
    def axes(self) -> Sequence[Series]:
        """Series that read naturally as an X axis, in order."""
        return tuple(item for item in self.series if item.axis)

    @property
    def others(self) -> Sequence[Series]:
        """Everything else -- the quantities usually wanted on Y."""
        return tuple(item for item in self.series if not item.axis)


def align(x: Series, y: Series) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pair two series on the records they share.

    An inner join on :attr:`Series.samples`, not a ``zip``. When both series were recorded
    on every step -- the normal case -- this is exactly ``zip`` and costs one dictionary.
    When they were not, it is the difference between a correct plot over the overlap and a
    confident plot of the wrong pairs.

    Returns:
        The x and y values for the shared records, in record order. Both are empty when
        the two series share nothing, which the caller should say out loud.
    """
    lookup = dict(zip(y.samples, y.values, strict=True))
    xs: list[float] = []
    ys: list[float] = []
    for sample, value in zip(x.samples, x.values, strict=True):
        paired = lookup.get(sample)
        if paired is not None:
            xs.append(value)
            ys.append(paired)
    return tuple(xs), tuple(ys)


def downsample(series: Series, limit: int) -> Series:
    """Reduce a series to at most ``limit`` points, keeping both ends.

    Uniform stride rather than averaging: a residual history is read for its shape and its
    final value, and an averaged tail would show a run converging further than it did.
    """
    count = len(series.values)
    if limit < 2 or count <= limit:
        return series
    step = (count - 1) / (limit - 1)
    indices = sorted({min(count - 1, round(i * step)) for i in range(limit)})
    return Series(
        key=series.key,
        label=series.label,
        values=tuple(series.values[i] for i in indices),
        samples=tuple(series.samples[i] for i in indices),
        unit=series.unit,
        axis=series.axis,
    )


def series_from_records(
    records: Sequence[Mapping[str, float]],
    *,
    labels: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    axes: Iterable[str] = (),
    order: Sequence[str] = (),
) -> tuple[Series, ...]:
    """Turn a parser's per-step records into series, keeping the gaps honest.

    The shape every log parser naturally produces is "one mapping per step, with whatever
    that step happened to print". This turns that into columns, recording for each value
    which step it came from -- so a quantity that appeared on only half the steps stays
    pairable with one that appeared on all of them.

    Args:
        records: One mapping per step, in order.
        labels: Display names by key. Defaults to the key.
        units: Units by key.
        axes: Keys that read naturally as an X axis.
        order: Preferred key ordering. Keys not listed follow, in first-seen order.

    Returns:
        One series per key that appeared at least once. Keys that never appeared produce
        nothing at all, which is how "this log has no Uy residual" is expressed.
    """
    label_map = dict(labels or {})
    unit_map = dict(units or {})
    axis_keys = set(axes)

    columns: dict[str, tuple[list[float], list[int]]] = {}
    for index, record in enumerate(records):
        for key, value in record.items():
            values, samples = columns.setdefault(key, ([], []))
            values.append(float(value))
            samples.append(index)

    ranking = {key: position for position, key in enumerate(order)}
    keys = sorted(columns, key=lambda key: (ranking.get(key, len(ranking)), key))
    return tuple(
        Series(
            key=key,
            label=label_map.get(key, key),
            values=tuple(columns[key][0]),
            samples=tuple(columns[key][1]),
            unit=unit_map.get(key),
            axis=key in axis_keys,
        )
        for key in keys
    )
