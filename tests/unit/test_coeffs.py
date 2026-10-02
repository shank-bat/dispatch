"""Finding and reading OpenFOAM force coefficients.

Most of this is about *finding* the file. There is no single path to assume: the function
object's directory is named by the user, the time directory beneath it is named by whenever
the object started, and the filename changed spelling in OpenFOAM v2012 and gains a suffix
on restart. Every layout exercised here was taken from a real projects tree.

The parsing tests are mostly about files that are wrong: half-written rows, headers with no
data, diverged runs full of ``nan``. A solver writes these while running, and a plot that
crashes on one is worse than no plot.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from dispatch.adapters.base import CaseContext
from dispatch.adapters.foamcoeffs import (
    DRAG_KEYS,
    LIFT_KEYS,
    find_coefficient_file,
    latest_coefficients,
    parse_coefficients,
    pick,
)
from dispatch.adapters.openfoam import OpenFOAMAdapter

# Real header text, trailing whitespace and all: OpenFOAM pads every column, and a parser
# that only copes with stripped lines would not cope with any actual file. Built by joining
# rather than written literally so the padding survives both this file's linter and anyone
# whose editor trims line ends.
HEADER = (
    "# Force and moment coefficients\n"
    "# liftDir       : (-3.48994967e-02 9.99390827e-01 0.00000000e+00)\n"
    "# magUInf       : 2.43455782e+01\n"
    "# Aref          : 2.00728993e-03\n"
    "#\n"
    "# Time         \tCd             \tCd(f)          \tCl             \tCl(f)          "
    "\tCmPitch        \n"
)

OLD_HEADER = "# Time\tCm\tCd\tCl\tCl(f)\tCl(r)\n"


def rows(count: int, *, start: int = 1) -> str:
    return "".join(
        f"{index}\t{0.05 + index * 0.001:.6e}\t{0.025:.6e}\t"
        f"{0.4 + index * 0.01:.6e}\t{0.2:.6e}\t{-0.3:.6e}\n"
        for index in range(start, start + count)
    )


def coeff_file(case: Path, relative: str, text: str, *, age: float = 0.0) -> Path:
    """Write a coefficient file, optionally backdated so mtime ordering is deterministic."""
    path = case / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path


# -- discovery ----------------------------------------------------------------------------


def test_the_modern_layout_is_found(tmp_path: Path) -> None:
    case = tmp_path / "case"
    expected = coeff_file(case, "postProcessing/forceCoeffs/0/coefficient.dat", HEADER + rows(3))
    assert find_coefficient_file(case) == expected


def test_the_older_filename_is_found(tmp_path: Path) -> None:
    """``forceCoeffs.dat`` predates the ``coefficient.dat`` spelling."""
    case = tmp_path / "case"
    expected = coeff_file(
        case, "postProcessing/forceCoeffs/0/forceCoeffs.dat", OLD_HEADER + rows(3)
    )
    assert find_coefficient_file(case) == expected


def test_a_restart_suffix_is_found(tmp_path: Path) -> None:
    """A function object restarting inside one time directory appends ``_0``."""
    case = tmp_path / "case"
    expected = coeff_file(case, "postProcessing/forceCoeffs/0/coefficient_0.dat", HEADER + rows(3))
    assert find_coefficient_file(case) == expected


def test_a_renamed_function_object_is_found(tmp_path: Path) -> None:
    """The directory is named by the user, so it is only conventionally ``forceCoeffs``."""
    case = tmp_path / "case"
    expected = coeff_file(
        case, "postProcessing/forceCoeffs1/0/coefficient.dat", HEADER + rows(3)
    )
    assert find_coefficient_file(case) == expected


def test_the_most_recently_written_file_wins(tmp_path: Path) -> None:
    """The requirement: not a hardcoded path, the file that is actually being written."""
    case = tmp_path / "case"
    coeff_file(case, "postProcessing/forceCoeffs/0/coefficient.dat", HEADER + rows(3), age=600)
    newest = coeff_file(
        case, "postProcessing/forceCoeffs/0/coefficient_0.dat", HEADER + rows(9)
    )
    assert find_coefficient_file(case) == newest


def test_a_non_integer_time_directory_is_handled(tmp_path: Path) -> None:
    """A restart's directory is named for its start time, which is rarely a whole number.

    Taken from a real case: sorting these *names* means deciding whether
    ``0.0004319995976`` sorts above ``4.5`` as text or as a number, which is why the
    filesystem's own modification time is used instead.
    """
    case = tmp_path / "case"
    coeff_file(case, "postProcessing/forceCoeffs/0/coefficient.dat", HEADER + rows(3), age=600)
    restart = coeff_file(
        case,
        "postProcessing/forceCoeffs/0.0004319995976/coefficient.dat",
        HEADER + rows(5, start=4),
    )
    assert find_coefficient_file(case) == restart


def test_a_case_with_no_post_processing_finds_nothing(tmp_path: Path) -> None:
    case = tmp_path / "case"
    case.mkdir()
    assert find_coefficient_file(case) is None


def test_a_missing_case_finds_nothing(tmp_path: Path) -> None:
    assert find_coefficient_file(tmp_path / "nope") is None


def test_an_empty_file_is_passed_over_for_one_with_data(tmp_path: Path) -> None:
    """A started-but-not-yet-written function object: the previous run is more use."""
    case = tmp_path / "case"
    older = coeff_file(
        case, "postProcessing/forceCoeffs/0/coefficient.dat", HEADER + rows(4), age=600
    )
    coeff_file(case, "postProcessing/forceCoeffs/1/coefficient.dat", HEADER)
    assert find_coefficient_file(case) == older


def test_an_unrelated_dat_file_is_not_mistaken_for_coefficients(tmp_path: Path) -> None:
    case = tmp_path / "case"
    coeff_file(
        case,
        "postProcessing/forceCoeffs/0/coefficient.dat",
        "# Time\tsomethingElse\n1\t2.0\n",
    )
    assert find_coefficient_file(case) is None


def test_other_post_processing_output_is_ignored(tmp_path: Path) -> None:
    """A probes or residuals function object must not be read as coefficients."""
    case = tmp_path / "case"
    coeff_file(case, "postProcessing/probes/0/p", "# Probe 0\n1 2.0\n")
    coeff_file(case, "postProcessing/residuals/0/residuals.dat", "# Time\tp\n1\t1e-5\n")
    assert find_coefficient_file(case) is None


# -- parsing --------------------------------------------------------------------------------


def test_columns_come_from_the_file_s_own_header(tmp_path: Path) -> None:
    data = parse_coefficients(HEADER + rows(4))
    assert [item.key for item in data.series] == [
        "Time", "Cl", "Cd", "Cd(f)", "Cl(f)", "CmPitch",
    ]
    assert data.samples == 4


def test_lift_and_drag_are_exposed(tmp_path: Path) -> None:
    """The two the feature exists for."""
    data = parse_coefficients(HEADER + rows(3))
    assert data.get("Cl") is not None
    assert data.get("Cd") is not None
    assert data.get("Cl").display == "Cl (lift)"  # type: ignore[union-attr]
    assert data.get("Cd").display == "Cd (drag)"  # type: ignore[union-attr]


def test_time_is_the_axis() -> None:
    data = parse_coefficients(HEADER + rows(3))
    assert [item.key for item in data.axes] == ["Time"]


def test_values_are_read_in_order() -> None:
    data = parse_coefficients(HEADER + rows(3))
    assert data.get("Time").values == (1.0, 2.0, 3.0)  # type: ignore[union-attr]
    assert data.get("Cl").values == pytest.approx((0.41, 0.42, 0.43))  # type: ignore[union-attr]


def test_the_older_column_layout_parses() -> None:
    data = parse_coefficients(OLD_HEADER + rows(3))
    assert data.samples == 3
    assert data.get("Cl") is not None and data.get("Cm") is not None


def test_a_half_written_trailing_row_is_dropped() -> None:
    """The solver has flushed part of a line and not the newline.

    Keeping it would read a column's value from under a different column's name.
    """
    partial = HEADER + rows(3) + "4\t0.054\t0.025\n"
    data = parse_coefficients(partial)
    assert data.samples == 3


def test_a_file_with_a_header_and_no_rows_yields_nothing() -> None:
    assert not parse_coefficients(HEADER)


def test_an_empty_file_yields_nothing() -> None:
    assert not parse_coefficients("")


def test_a_file_with_no_recognisable_header_yields_nothing() -> None:
    assert not parse_coefficients("1 2 3\n4 5 6\n")


def test_malformed_numbers_do_not_crash() -> None:
    broken = HEADER + "1\tnot-a-number\t0.025\t0.41\t0.2\t-0.3\n" + rows(2, start=2)
    data = parse_coefficients(broken)
    assert data.samples == 3
    # The bad field is simply absent from its series rather than poisoning it.
    assert len(data.get("Cd")) == 2  # type: ignore[arg-type]


def test_nan_and_infinity_are_dropped() -> None:
    """A diverging run writes them, and they poison every axis calculation."""
    import math

    diverged = HEADER + "1\tnan\t0.025\tinf\t0.2\t-0.3\n" + rows(2, start=2)
    data = parse_coefficients(diverged)
    for item in data.series:
        assert all(math.isfinite(value) for value in item.values)


def test_a_second_header_from_a_restart_is_the_one_that_counts() -> None:
    """A restarted function object appends a new header describing the rows after it."""
    restarted = HEADER + rows(2) + OLD_HEADER + rows(2, start=3)
    data = parse_coefficients(restarted)
    assert data.get("Cm") is not None, "the later header's columns are in use"


def test_truncation_is_reported() -> None:
    assert parse_coefficients(HEADER + rows(2), truncated=True).truncated


def test_the_latest_values_are_the_last_row() -> None:
    data = parse_coefficients(HEADER + rows(5))
    latest = latest_coefficients(data)
    assert latest["Time"] == 5.0
    assert pick(latest, LIFT_KEYS) == pytest.approx(0.45)
    assert pick(latest, DRAG_KEYS) == pytest.approx(0.055)


def test_picking_a_missing_coefficient_returns_nothing() -> None:
    assert pick({"Time": 1.0}, LIFT_KEYS) is None


# -- through the adapter ---------------------------------------------------------------------


def test_the_adapter_offers_coefficients_as_their_own_dataset(tmp_path: Path) -> None:
    """Separate from the log, because a function object writes on its own schedule: row 5
    of one is not row 5 of the other."""
    case = tmp_path / "case"
    coeff_file(case, "postProcessing/forceCoeffs/0/coefficient.dat", HEADER + rows(6))

    datasets = OpenFOAMAdapter({}).case_datasets(CaseContext(workdir=case, cores=1))
    assert [item.key for item in datasets] == ["forceCoeffs"]
    assert datasets[0].label == "Force coefficients"
    assert datasets[0].data.samples == 6
    assert datasets[0].source is not None and datasets[0].source.endswith("coefficient.dat")


def test_a_case_without_coefficients_offers_no_dataset(tmp_path: Path) -> None:
    case = tmp_path / "case"
    case.mkdir()
    assert OpenFOAMAdapter({}).case_datasets(CaseContext(workdir=case, cores=1)) == ()


def test_an_adapter_without_case_output_offers_nothing(tmp_path: Path) -> None:
    """The default, so no existing adapter had to change."""
    from dispatch.adapters.su2 import SU2Adapter

    assert SU2Adapter({}).case_datasets(CaseContext(workdir=tmp_path, cores=1)) == ()
