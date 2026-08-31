"""Plot data and terminal rendering.

Three layers, all testable without a terminal, a solver, or a running application:

* :mod:`dispatch.core.series` -- the algebra. Pairing, downsampling, and what happens when
  two series were not recorded on the same steps.
* :mod:`dispatch.adapters.foamlog` -- reading numbers out of real solver output, including
  the output that is malformed, truncated, or diverging.
* :mod:`dispatch.tui.plot` -- turning numbers into characters, and only into characters.

The last of those is checked explicitly for the thing it must never do: write a file.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dispatch.adapters.base import CaseContext
from dispatch.adapters.foamlog import parse_foam_log
from dispatch.adapters.openfoam import OpenFOAMAdapter
from dispatch.core.series import PlotData, Series, align, downsample, series_from_records
from dispatch.tui.plot import (
    Charset,
    Curve,
    PlotStyle,
    hidden_by_log,
    render_plot,
    series_colour,
    suggests_log,
)

# -- a realistic OpenFOAM log ---------------------------------------------------------------

STEP = """
Time = {time}

Courant Number mean: 0.018 max: {courant}
smoothSolver:  Solving for Ux, Initial residual = {ux}, Final residual = 1e-06, No It 3
smoothSolver:  Solving for Uy, Initial residual = {uy}, Final residual = 1e-07, No It 3
GAMG:  Solving for p, Initial residual = {p}, Final residual = 1e-05, No It 12
GAMG:  Solving for p, Initial residual = 0.0001, Final residual = 1e-08, No It 4
time step continuity errors : sum local = 1e-09, global = -1e-19, cumulative = {cont}
ExecutionTime = {exec_time} s  ClockTime = {clock} s
"""

HEADER = """\
/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
\\*---------------------------------------------------------------------------*/
Build  : v2312
Create mesh for time = 0

Starting time loop
"""


def foam_log(steps: int = 5, *, header: str = HEADER) -> str:
    body = "".join(
        STEP.format(
            time=0.005 * (index + 1),
            courant=0.4 + index * 0.01,
            ux=10 ** (-2 - index),
            uy=10 ** (-2.5 - index),
            p=10 ** (-1 - index),
            cont=-1e-19 * (index + 1),
            exec_time=0.05 * (index + 1),
            clock=index,
        )
        for index in range(steps)
    )
    return header + body


# -- core: the series algebra ---------------------------------------------------------------


def test_records_become_columns_that_remember_their_gaps() -> None:
    """A quantity printed on only some steps must stay pairable with one printed on all."""
    series = series_from_records(
        [{"i": 1.0, "loss": 0.5}, {"i": 2.0}, {"i": 3.0, "loss": 0.1}],
        axes=["i"],
        order=["i", "loss"],
    )
    by_key = {item.key: item for item in series}
    assert by_key["i"].values == (1.0, 2.0, 3.0)
    assert by_key["loss"].values == (0.5, 0.1)
    assert by_key["loss"].samples == (0, 2), "the gap is recorded, not filled in"


def test_a_key_that_never_appeared_produces_no_series() -> None:
    """A 2-D case has no Uz, so it must not be offered an empty residual(Uz)."""
    series = series_from_records([{"a": 1.0}, {"a": 2.0}])
    assert [item.key for item in series] == ["a"]


def test_pairing_is_an_inner_join_not_a_zip() -> None:
    """Zipping series of different lengths silently plots the wrong pairs."""
    x = Series("i", "i", (1.0, 2.0, 3.0), (0, 1, 2), axis=True)
    y = Series("loss", "loss", (0.5, 0.1), (0, 2))
    assert align(x, y) == ((1.0, 3.0), (0.5, 0.1))


def test_series_that_share_nothing_pair_to_nothing() -> None:
    """Empty, so the interface can say so rather than drawing a misleading line."""
    x = Series("a", "a", (1.0,), (0,))
    y = Series("b", "b", (2.0,), (7,))
    assert align(x, y) == ((), ())


def test_a_series_of_mismatched_lengths_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="sample indices"):
        Series("x", "x", (1.0, 2.0), (0,))


def test_downsampling_keeps_both_ends() -> None:
    """A residual plot is read for its shape and its final value."""
    full = Series("r", "r", tuple(float(i) for i in range(1000)), tuple(range(1000)))
    reduced = downsample(full, 50)
    assert len(reduced) <= 50
    assert reduced.values[0] == 0.0
    assert reduced.values[-1] == 999.0
    assert reduced.samples[-1] == 999


def test_downsampling_a_short_series_changes_nothing() -> None:
    short = Series("r", "r", (1.0, 2.0), (0, 1))
    assert downsample(short, 500) is short


def test_plot_data_reports_what_it_does_not_have() -> None:
    data = PlotData(series=(Series("a", "a", (1.0,), (0,), axis=True),), samples=1)
    assert data.get("a") is not None
    assert data.get("residual(Uy)") is None
    assert [item.key for item in data.axes] == ["a"]
    assert data.others == ()


# -- the OpenFOAM parser -----------------------------------------------------------------


def test_a_foam_log_yields_the_series_it_contains() -> None:
    data = parse_foam_log(foam_log(steps=4))
    keys = {item.key for item in data.series}
    assert data.samples == 4
    assert {
        "iteration",
        "time",
        "execution_time",
        "clock_time",
        "continuity",
        "courant_mean",
        "courant_max",
        "residual(Ux)",
        "residual(Uy)",
        "residual(p)",
    } <= keys


def test_multiple_residuals_are_kept_apart() -> None:
    data = parse_foam_log(foam_log(steps=3))
    ux = data.get("residual(Ux)")
    p = data.get("residual(p)")
    assert ux is not None and p is not None
    assert ux.values == (1e-2, 1e-3, 1e-4)
    assert p.values == (1e-1, 1e-2, 1e-3)


def test_only_the_first_residual_of_each_step_is_taken() -> None:
    """A PIMPLE run solves for p several times per step; the plotted one is the first."""
    data = parse_foam_log(foam_log(steps=1))
    p = data.get("residual(p)")
    assert p is not None
    assert p.values == (1e-1,), "the inner-loop 1e-04 is a different quantity"


def test_a_missing_residual_is_simply_absent() -> None:
    """A 2-D case has no Uz. It must not appear in the selector at all."""
    data = parse_foam_log(foam_log(steps=3))
    assert data.get("residual(Uz)") is None
    assert "residual(Uz)" not in {item.key for item in data.series}


def test_iteration_and_time_are_both_offered_as_axes() -> None:
    data = parse_foam_log(foam_log(steps=3))
    assert {item.key for item in data.axes} >= {"iteration", "time"}
    assert data.get("residual(p)") is not None
    assert not data.get("residual(p)").axis  # type: ignore[union-attr]


def test_arbitrary_x_and_y_can_be_paired() -> None:
    """``execution_time`` against ``iteration`` -- "is it slowing down" -- must work."""
    data = parse_foam_log(foam_log(steps=5))
    x, y = data.get("iteration"), data.get("execution_time")
    assert x is not None and y is not None
    xs, ys = align(x, y)
    assert xs == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert ys[-1] == pytest.approx(0.25)


def test_parallel_rank_prefixes_are_stripped() -> None:
    """In practice every log worth plotting is a parallel one."""
    labelled = "\n".join(f"[0] {line}" for line in foam_log(steps=3).splitlines())
    data = parse_foam_log(labelled)
    assert data.samples == 3
    assert data.get("residual(Ux)") is not None


def test_an_empty_log_yields_nothing_rather_than_failing() -> None:
    data = parse_foam_log("")
    assert not data
    assert data.samples == 0
    assert data.series == ()


def test_a_log_with_only_a_header_yields_nothing() -> None:
    """A case that has not reached its first time step is an ordinary state."""
    assert parse_foam_log(HEADER).samples == 0


def test_malformed_lines_are_skipped_not_fatal() -> None:
    """A log truncated by a kill, or interleaved by mpirun, still plots what it has."""
    broken = foam_log(steps=3) + "\nTime = not-a-number\nSolving for Ux, Initial residual = ,\n"
    data = parse_foam_log(broken)
    assert data.samples == 3
    assert data.get("residual(Ux)") is not None


def test_a_truncated_final_line_does_not_lose_the_run() -> None:
    partial = foam_log(steps=3) + "\nTime = 0.02\n\nsmoothSolver:  Solving for Ux, Initial re"
    data = parse_foam_log(partial)
    assert data.samples == 4
    ux = data.get("residual(Ux)")
    assert ux is not None and len(ux) == 3, "the incomplete step contributed no residual"


def test_nan_and_infinity_are_dropped() -> None:
    """A diverging solver prints them, and they poison every axis calculation."""
    diverging = foam_log(steps=2) + (
        "\nTime = 0.015\n"
        "smoothSolver:  Solving for Ux, Initial residual = nan, Final residual = nan, No It 3\n"
        "smoothSolver:  Solving for Uy, Initial residual = inf, Final residual = inf, No It 3\n"
    )
    data = parse_foam_log(diverging)
    ux = data.get("residual(Ux)")
    assert ux is not None
    assert all(math.isfinite(value) for value in ux.values)
    assert len(ux) == 2


def test_truncation_is_reported_rather_than_hidden() -> None:
    data = parse_foam_log(foam_log(steps=2), truncated=True)
    assert data.truncated


def test_the_adapter_exposes_the_parser(tmp_path: Path) -> None:
    """The daemon calls this, not the module: the plumbing has to be connected."""
    ctx = CaseContext(workdir=tmp_path, cores=1, env={"PATH": "/nonexistent"})
    data = OpenFOAMAdapter({}).parse_series(foam_log(steps=3), ctx)
    assert data.samples == 3
    assert data.get("residual(p)") is not None


def test_an_adapter_with_no_parser_returns_empty_data(tmp_path: Path) -> None:
    """Returning nothing is a valid answer, and the default."""
    from dispatch.adapters.basilisk import BasiliskAdapter

    ctx = CaseContext(workdir=tmp_path, cores=1)
    assert not BasiliskAdapter({}).parse_series("t = 0.5\n", ctx)


# -- the renderer ---------------------------------------------------------------------------


def line_text(lines: list) -> str:
    return "\n".join(line.plain for line in lines)


def decaying(count: int = 200) -> Curve:
    xs = [float(i) for i in range(count)]
    ys = [10 ** (-1 - 3 * i / count) for i in range(count)]
    return Curve("residual(Ux)", xs, ys)


def test_a_plot_is_characters_and_nothing_else(tmp_path: Path) -> None:
    """The requirement, stated as a test: no image file of any kind is produced."""
    before = set(tmp_path.rglob("*"))
    lines = render_plot([decaying()], width=70, height=14, style=PlotStyle(log_y=True))
    assert set(tmp_path.rglob("*")) == before

    rendered = line_text(lines)
    assert rendered.strip()
    for extension in (".png", ".svg", ".jpg", ".jpeg", ".pdf", ".gif"):
        assert extension not in rendered


def test_no_plotting_library_is_imported() -> None:
    """The dependency footprint is part of the design, and is checked rather than trusted."""
    import sys

    import dispatch.tui.plot  # noqa: F401

    for forbidden in ("matplotlib", "plotly", "numpy", "PIL", "plotext"):
        assert forbidden not in sys.modules


def test_the_plot_has_axes_and_tick_labels() -> None:
    lines = render_plot(
        [decaying()], width=70, height=14, style=PlotStyle(log_y=True), x_label="Iteration"
    )
    rendered = line_text(lines)
    assert "└" in rendered and "┤" in rendered
    assert "Iteration" in rendered
    assert "1e-04" in rendered or "0.0001" in rendered


def test_the_curve_actually_descends() -> None:
    """A decaying residual must be drawn high on the left and low on the right."""
    lines = render_plot([decaying()], width=70, height=16, style=PlotStyle(log_y=True))
    body = [line.plain for line in lines][:-2]
    first_ink = [index for index, line in enumerate(body) if line.strip().split("┤")[-1].strip()]
    top, bottom = first_ink[0], first_ink[-1]

    def rightmost(row: str) -> int:
        return len(row.rstrip())

    assert rightmost(body[top]) < rightmost(body[bottom])


def test_both_charsets_render() -> None:
    """``m`` exists for terminals whose font lacks braille; both must produce a chart."""
    for charset in (Charset.BRAILLE, Charset.BLOCKS):
        lines = render_plot(
            [decaying()], width=60, height=12, style=PlotStyle(charset=charset, log_y=True)
        )
        assert any(line.plain.strip() for line in lines)


def test_several_curves_are_overlaid_in_different_colours() -> None:
    curves = [decaying(), Curve("residual(p)", [float(i) for i in range(200)], [0.5] * 200)]
    lines = render_plot(curves, width=70, height=14)
    styles = {str(span.style) for line in lines for span in line.spans}
    assert series_colour(0) in styles
    assert series_colour(1) in styles


def test_an_empty_selection_says_so_rather_than_drawing_a_blank_box() -> None:
    lines = render_plot([Curve("empty", [], [])], width=70, height=14)
    assert "no data points" in line_text(lines)


def test_a_terminal_too_small_says_so() -> None:
    lines = render_plot([decaying()], width=10, height=3)
    assert "too small" in line_text(lines)


def test_a_single_point_renders() -> None:
    lines = render_plot([Curve("one", [1.0], [1.0])], width=40, height=8)
    assert any(char != "⠀" for line in lines for char in line.plain)


def test_a_constant_series_renders_without_dividing_by_zero() -> None:
    flat = Curve("steady", [float(i) for i in range(50)], [3.0] * 50)
    lines = render_plot([flat], width=50, height=10)
    assert line_text(lines).strip()


def test_a_log_axis_is_refused_for_non_positive_data_with_an_explanation() -> None:
    """Continuity errors are negative. Silently drawing nothing would be worse."""
    negative = Curve("continuity", [1.0, 2.0, 3.0], [-1e-9, -2e-9, -3e-9])
    lines = render_plot([negative], width=50, height=10, style=PlotStyle(log_y=True))
    assert "logarithmic" in line_text(lines)


def test_a_negative_series_renders_linearly() -> None:
    negative = Curve("continuity", [1.0, 2.0, 3.0], [-1e-9, -2e-9, -3e-9])
    lines = render_plot([negative], width=50, height=10)
    assert line_text(lines).strip()


def test_log_scale_is_suggested_only_for_wide_positive_ranges() -> None:
    assert suggests_log([1e-1, 1e-5])
    assert not suggests_log([1.0, 2.0, 3.0])
    assert not suggests_log([])


def test_one_zero_does_not_disqualify_a_residual_history() -> None:
    """Taken from a real icoFoam cavity log: Uy's first initial residual is exactly 0.

    Requiring every value to be positive meant that single point turned off the log axis
    for a plot whose whole purpose is six decades of decay.
    """
    residuals = [0.0, *(10.0**-exponent for exponent in range(1, 100))]
    assert suggests_log(residuals)
    assert hidden_by_log(residuals) == 1


def test_a_mostly_negative_series_still_rules_a_log_axis_out() -> None:
    """Continuity errors are negative; hiding most of the series would be worse."""
    assert not suggests_log([-1e-9, -2e-9, -3e-9, 1e-9])


def test_points_a_log_axis_cannot_show_are_counted() -> None:
    assert hidden_by_log([1.0, 0.0, -1.0, 2.0]) == 2
    assert hidden_by_log([1.0, 2.0]) == 0


def test_a_series_with_one_zero_still_renders_on_a_log_axis() -> None:
    """The zero is dropped from the drawing; the other points are the plot."""
    xs = [float(i) for i in range(50)]
    ys = [0.0, *(10 ** (-1 - 3 * i / 50) for i in range(1, 50))]
    lines = render_plot([Curve("residual(Uy)", xs, ys)], width=60, height=12,
                        style=PlotStyle(log_y=True))
    assert "logarithmic" not in line_text(lines)
    assert line_text(lines).strip()
