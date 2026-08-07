"""The search query language.

Parsing is pure, so this is where the grammar is pinned down. The cases that matter most
are the rejections: a query that silently means something other than what the user typed
returns a confident wrong answer, and in a research history that is the expensive failure.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dispatch.core.errors import QueryError
from dispatch.core.query import CompareOp, parse_query
from dispatch.core.states import JobState

NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC).timestamp()


def parse(text: str):
    return parse_query(text, now=NOW)


# -- free text ----------------------------------------------------------------------------


def test_bare_words_become_free_text() -> None:
    assert parse("naca0018 wing").text == "naca0018 wing"


def test_empty_query_is_empty() -> None:
    assert parse("").is_empty
    assert parse("   ").is_empty


def test_quoted_phrases_survive_as_one_term() -> None:
    assert parse('"angle of attack"').text == "angle of attack"


def test_a_constrained_query_is_not_empty() -> None:
    assert not parse("tag:paper").is_empty


# -- tags ---------------------------------------------------------------------------------


def test_tag_inclusion_and_exclusion() -> None:
    query = parse("tag:paper -tag:scratch")
    assert query.tags_include == {"paper"}
    assert query.tags_exclude == {"scratch"}


def test_tags_are_normalised_in_queries_exactly_as_on_write() -> None:
    """Otherwise `tag:Paper` would silently fail to match a job tagged `paper`."""
    assert parse("tag:PAPER").tags_include == {"paper"}


def test_multiple_include_tags_accumulate() -> None:
    assert parse("tag:paper tag:re100000").tags_include == {"paper", "re100000"}


# -- enumerated fields ----------------------------------------------------------------------


def test_state_filter_is_case_insensitive() -> None:
    assert parse("state:completed").states == {JobState.COMPLETED}
    assert parse("state:RUNNING").states == {JobState.RUNNING}


def test_state_can_be_negated() -> None:
    assert parse("-state:cancelled").states_exclude == {JobState.CANCELLED}


def test_unknown_state_is_rejected_with_the_valid_list() -> None:
    with pytest.raises(QueryError, match="Unknown job state"):
        parse("state:finished")


def test_solver_and_app_filters() -> None:
    query = parse("solver:OpenFOAM app:interFoam")
    assert query.solvers == {"openfoam"}
    assert query.apps == {"interFoam"}


def test_dir_and_workdir_are_the_same_field() -> None:
    assert parse("dir:foam").dirs == ("foam",)
    assert parse("workdir:foam").dirs == ("foam",)


# -- the strictness that matters ---------------------------------------------------------------


def test_unknown_keyword_is_an_error_not_free_text() -> None:
    """A typo must not quietly return the wrong set of runs."""
    with pytest.raises(QueryError, match="Unknown search field"):
        parse("taag:paper")


def test_unknown_keyword_error_names_the_valid_fields() -> None:
    with pytest.raises(QueryError) as excinfo:
        parse("bogus:1")
    assert "tag" in str(excinfo.value)


def test_keyword_without_a_value_is_an_error() -> None:
    with pytest.raises(QueryError, match="missing a value"):
        parse("tag:")


def test_free_text_cannot_be_negated() -> None:
    with pytest.raises(QueryError, match="Cannot negate free text"):
        parse("-naca0018")


def test_negation_is_rejected_where_it_has_no_meaning() -> None:
    with pytest.raises(QueryError, match="cannot be negated"):
        parse("-after:2026")


def test_unbalanced_quotes_are_reported_clearly() -> None:
    with pytest.raises(QueryError, match="Could not parse query"):
        parse('name:"unterminated')


def test_impossible_date_range_is_rejected_up_front() -> None:
    """Better to say "nothing can match" than to return an empty page and look correct."""
    with pytest.raises(QueryError, match="nothing can match"):
        parse("after:2026-06 before:2026-01")


# -- comparisons -------------------------------------------------------------------------------


def test_column_comparison_maps_to_the_sql_column() -> None:
    (comparison,) = parse("cores>=16").comparisons
    assert (comparison.field, comparison.op, comparison.value) == ("cores", CompareOp.GE, 16.0)
    assert not comparison.is_metadata


def test_unknown_key_with_an_operator_is_treated_as_metadata() -> None:
    """Metadata keys are adapter-declared and cannot be enumerated at parse time."""
    (comparison,) = parse("endTime>500").comparisons
    assert comparison.is_metadata
    assert comparison.field == "endTime"


def test_meta_prefix_disambiguates_a_key_that_collides_with_a_column() -> None:
    (comparison,) = parse("meta:cores>8").comparisons
    assert comparison.is_metadata and comparison.field == "cores"


def test_all_comparison_operators_parse() -> None:
    for text, op in [
        ("cores=8", CompareOp.EQ),
        ("cores!=8", CompareOp.NE),
        ("cores<8", CompareOp.LT),
        ("cores<=8", CompareOp.LE),
        ("cores>8", CompareOp.GT),
        ("cores>=8", CompareOp.GE),
    ]:
        assert parse(text).comparisons[0].op is op


def test_not_equal_uses_sql_spelling() -> None:
    assert CompareOp.NE.sql == "<>"


def test_comparisons_cannot_be_negated() -> None:
    with pytest.raises(QueryError, match="invert the operator"):
        parse("-cores>8")


def test_non_numeric_comparison_value_is_rejected() -> None:
    with pytest.raises(QueryError, match="expected a number"):
        parse("cores>many")


# -- units --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("runtime>90", 90), ("runtime>30m", 1800), ("runtime>2h", 7200), ("runtime>3d", 259200)],
)
def test_durations_normalise_to_seconds(text: str, seconds: float) -> None:
    assert parse(text).comparisons[0].value == seconds


@pytest.mark.parametrize(
    ("text", "megabytes"), [("ram>512", 512), ("ram>512M", 512), ("ram>16G", 16384)]
)
def test_sizes_normalise_to_megabytes(text: str, megabytes: float) -> None:
    assert parse(text).comparisons[0].value == megabytes


def test_bad_duration_names_the_accepted_forms() -> None:
    with pytest.raises(QueryError, match="30m"):
        parse("runtime>2fortnights")


def test_a_space_splits_tokens_so_a_bare_number_is_still_valid() -> None:
    """`runtime>2 fortnights` is two tokens: a valid comparison and a free-text word.

    Worth pinning down, because it means a stray space changes the meaning of a query
    rather than producing an error.
    """
    query = parse("runtime>2 fortnights")
    assert query.comparisons[0].value == 2.0
    assert query.text == "fortnights"


# -- dates ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("after:2026", datetime(2026, 1, 1, tzinfo=UTC)),
        ("after:2026-06", datetime(2026, 6, 1, tzinfo=UTC)),
        ("after:2026-06-15", datetime(2026, 6, 15, tzinfo=UTC)),
        ("after:2026-06-15T09:30", datetime(2026, 6, 15, 9, 30, tzinfo=UTC)),
    ],
)
def test_absolute_dates_resolve_to_the_start_of_the_period(text: str, expected) -> None:
    assert parse(text).created_after == expected.timestamp()


def test_before_also_resolves_to_the_start_of_the_period() -> None:
    """So that `after:2026-01 before:2026-02` is exactly January.

    If one of the two silently meant "end of", that composition would quietly include or
    exclude a month's worth of runs.
    """
    query = parse("after:2026-01 before:2026-02")
    assert query.created_after == datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    assert query.created_before == datetime(2026, 2, 1, tzinfo=UTC).timestamp()


def test_relative_dates_are_resolved_against_the_injected_now() -> None:
    assert parse("after:7d").created_after == NOW - 7 * 86400


def test_today_and_yesterday() -> None:
    midnight = datetime(2026, 6, 15, tzinfo=UTC).timestamp()
    assert parse("after:today").created_after == midnight
    assert parse("after:yesterday").created_after == midnight - 86400


def test_unreadable_date_is_rejected() -> None:
    with pytest.raises(QueryError, match="Could not read"):
        parse("after:last-tuesday")


# -- misc ------------------------------------------------------------------------------------------


def test_dirty_flag() -> None:
    assert parse("dirty:true").dirty is True
    assert parse("dirty:no").dirty is False


def test_bad_boolean_is_rejected() -> None:
    with pytest.raises(QueryError, match="Expected true or false"):
        parse("dirty:maybe")


def test_id_prefixes_are_lowercased() -> None:
    assert parse("id:A1B2").ids == {"a1b2"}


def test_a_realistic_mixed_query() -> None:
    query = parse("naca0018 tag:paper -tag:scratch solver:openfoam cores>=16 endTime>500 after:7d")
    assert query.text == "naca0018"
    assert query.tags_include == {"paper"}
    assert query.tags_exclude == {"scratch"}
    assert query.solvers == {"openfoam"}
    assert {c.field for c in query.comparisons} == {"cores", "endTime"}
    assert query.created_after == NOW - 7 * 86400
