"""Search: FTS, tags, typed metadata, and the derived indexes staying in sync.

The index-drift tests are the important ones. The canonical data is the ``jobs`` row; the
FTS row and ``job_metadata`` are copies. A code path that updates one without the other
produces a search that is silently, permanently wrong -- so every mutation is checked to
see that the copies followed.
"""

from __future__ import annotations

import sqlite3

import pytest

from dispatch.core.clock import FakeClock
from dispatch.core.metadata import CaseMetadata, MetadataSpec
from dispatch.core.query import parse_query
from dispatch.core.states import ExitReason, JobState
from dispatch.db.repository import JobRepository


@pytest.fixture
def populated(repo: JobRepository, make_spec, foam_spec: MetadataSpec, clock: FakeClock):
    """Three jobs with distinct tags, solvers, metadata, and outcomes."""
    acoustic = repo.create(
        make_spec(
            "foamacoustic",
            solver="openfoam",
            solver_binary="interFoam",
            cores=20,
            tags={"paper", "naca0018"},
            metadata=foam_spec.build(
                {"application": "interFoam", "endTime": 600.0, "decomposition": 20}
            ),
            note="final run for figure 4",
        )
    )
    cavity = repo.create(
        make_spec(
            "cavity",
            solver="openfoam",
            solver_binary="icoFoam",
            cores=4,
            tags={"scratch"},
            metadata=foam_spec.build(
                {"application": "icoFoam", "endTime": 0.5, "decomposition": 4}
            ),
        )
    )
    wing = repo.create(
        make_spec("onera-m6", solver="su2", solver_binary="SU2_CFD", cores=8, tags={"paper"})
    )
    return {"acoustic": acoustic, "cavity": cavity, "wing": wing}


def find(repo: JobRepository, text: str, clock: FakeClock) -> set[str]:
    """Run a query and return the matching job names."""
    page = repo.search(parse_query(text, now=clock.now()))
    return {job.name for job in page.items}


# -- free text ---------------------------------------------------------------------------


def test_matches_job_name(repo, populated, clock) -> None:
    assert find(repo, "cavity", clock) == {"cavity"}


def test_matches_a_prefix(repo, populated, clock) -> None:
    """`naca` should find `naca0018`; requiring whole words makes search useless."""
    assert find(repo, "foamac", clock) == {"foamacoustic"}


def test_matches_a_note(repo, populated, clock) -> None:
    assert find(repo, "figure", clock) == {"foamacoustic"}


def test_matches_a_metadata_value(repo, populated, clock) -> None:
    assert find(repo, "icoFoam", clock) == {"cavity"}


def test_matches_a_metadata_key(repo, populated, clock) -> None:
    """Both keys and values are indexed, so `decomposition` finds jobs that declare it."""
    assert find(repo, "decomposition", clock) == {"foamacoustic", "cavity"}


def test_matches_a_tag_as_free_text(repo, populated, clock) -> None:
    assert find(repo, "naca0018", clock) == {"foamacoustic"}


def test_matches_the_working_directory(repo, populated, clock) -> None:
    assert find(repo, "onera-m6", clock) == {"onera-m6"}


def test_terms_are_combined_with_and(repo, populated, clock) -> None:
    assert find(repo, "cavity figure", clock) == set()


def test_special_characters_do_not_break_fts_syntax(repo, populated, clock) -> None:
    """`re-100` is a legal thing to search for, not an FTS5 NOT expression."""
    assert find(repo, "re-100", clock) == set()  # no match, but importantly no crash


def test_quoted_phrase_with_punctuation(repo, populated, clock) -> None:
    assert find(repo, '"figure 4"', clock) == {"foamacoustic"}


# -- structured filters --------------------------------------------------------------------


def test_tag_filter(repo, populated, clock) -> None:
    assert find(repo, "tag:paper", clock) == {"foamacoustic", "onera-m6"}


def test_multiple_tags_are_conjunctive(repo, populated, clock) -> None:
    """`tag:paper tag:naca0018` means both, not either."""
    assert find(repo, "tag:paper tag:naca0018", clock) == {"foamacoustic"}


def test_tag_exclusion(repo, populated, clock) -> None:
    assert find(repo, "-tag:scratch", clock) == {"foamacoustic", "onera-m6"}


def test_solver_filter(repo, populated, clock) -> None:
    assert find(repo, "solver:su2", clock) == {"onera-m6"}


def test_app_filter(repo, populated, clock) -> None:
    assert find(repo, "app:icoFoam", clock) == {"cavity"}


def test_state_filter(repo, populated, clock) -> None:
    repo.mark_preparing(populated["cavity"].id)
    assert find(repo, "state:preparing", clock) == {"cavity"}


def test_state_exclusion(repo, populated, clock) -> None:
    repo.mark_preparing(populated["cavity"].id)
    assert find(repo, "-state:queued", clock) == {"cavity"}


def test_column_comparison(repo, populated, clock) -> None:
    assert find(repo, "cores>=8", clock) == {"foamacoustic", "onera-m6"}


def test_directory_filter(repo, populated, clock) -> None:
    assert find(repo, "dir:cavity", clock) == {"cavity"}


def test_id_prefix_filter(repo, populated, clock) -> None:
    target = populated["wing"]
    assert find(repo, f"id:{target.id[:8]}", clock) == {"onera-m6"}


def test_filters_combine(repo, populated, clock) -> None:
    assert find(repo, "tag:paper solver:openfoam", clock) == {"foamacoustic"}


def test_empty_query_returns_everything(repo, populated, clock) -> None:
    assert len(find(repo, "", clock)) == 3


# -- typed metadata comparisons ------------------------------------------------------------


def test_metadata_comparison_is_numeric_not_lexical(repo, populated, clock) -> None:
    """The whole reason for the derived typed index.

    As strings, "0.5" sorts after "600". If this ever regresses to text comparison, this
    test flips and the failure is unmistakable.
    """
    assert find(repo, "endTime>500", clock) == {"foamacoustic"}
    assert find(repo, "endTime<1", clock) == {"cavity"}


def test_metadata_comparison_on_an_integer_field(repo, populated, clock) -> None:
    assert find(repo, "decomposition>=20", clock) == {"foamacoustic"}


def test_metadata_comparison_ignores_jobs_lacking_the_field(repo, populated, clock) -> None:
    """The SU2 job has no `endTime`, so it is absent rather than treated as zero."""
    assert "onera-m6" not in find(repo, "endTime<1000000", clock)


def test_unknown_metadata_key_matches_nothing(repo, populated, clock) -> None:
    assert find(repo, "nosuchfield>1", clock) == set()


def test_fields_marked_unsearchable_are_not_indexed(
    repo: JobRepository, make_spec, foam_spec, conn: sqlite3.Connection, clock
) -> None:
    job = repo.create(
        make_spec("blobby", metadata=foam_spec.build({"notes_blob": "x", "endTime": 5.0}))
    )
    keys = {
        row["key"]
        for row in conn.execute("SELECT key FROM job_metadata WHERE job_id = ?", (job.id,))
    }
    assert keys == {"endTime"}


# -- index synchronisation -------------------------------------------------------------------


def test_adding_a_note_updates_the_search_index(repo, populated, clock) -> None:
    repo.add_note(populated["cavity"].id, "diverged at iteration 400")
    assert find(repo, "diverged", clock) == {"cavity"}


def test_adding_a_tag_updates_the_search_index(repo, populated, clock) -> None:
    repo.edit_tags(populated["cavity"].id, add=["validation"])
    assert find(repo, "tag:validation", clock) == {"cavity"}
    assert find(repo, "validation", clock) == {"cavity"}


def test_removing_a_tag_updates_the_search_index(repo, populated, clock) -> None:
    repo.edit_tags(populated["acoustic"].id, remove=["paper"])
    assert find(repo, "tag:paper", clock) == {"onera-m6"}


def test_replacing_tags_updates_the_search_index(repo, populated, clock) -> None:
    repo.set_tags(populated["acoustic"].id, ["archived"])
    assert find(repo, "tag:paper", clock) == {"onera-m6"}
    assert find(repo, "tag:archived", clock) == {"foamacoustic"}


def test_updating_metadata_updates_both_indexes(repo, populated, foam_spec, clock) -> None:
    repo.update_metadata(
        populated["cavity"].id,
        foam_spec.build({"application": "pisoFoam", "endTime": 999.0}),
    )
    # The new values are searchable in both the text index and the typed index...
    assert find(repo, "pisoFoam", clock) == {"cavity"}
    assert find(repo, "endTime>900", clock) == {"cavity"}
    # ...and the superseded value is gone, rather than lingering as a stale copy.
    # (`icoFoam` itself still matches, but via the `solver_binary` column -- see below.)
    assert find(repo, "endTime<1", clock) == set()


def test_metadata_update_does_not_touch_solver_binary(repo, populated, foam_spec, clock) -> None:
    """`solver_binary` is a job column, not metadata.

    So the job stays findable by the application it was submitted with, even after its
    case metadata is re-read. Worth pinning down, since the two hold the same string.
    """
    repo.update_metadata(populated["cavity"].id, foam_spec.build({"application": "pisoFoam"}))
    assert repo.get(populated["cavity"].id).solver_binary == "icoFoam"
    assert find(repo, "icoFoam", clock) == {"cavity"}


def test_deleting_a_job_removes_it_from_search(repo, populated, clock) -> None:
    job = populated["cavity"]
    repo.transition(job.id, JobState.CANCELLED)
    repo.delete(job.id)
    assert find(repo, "cavity", clock) == set()


def test_rebuild_search_index_reproduces_the_same_results(
    repo: JobRepository, populated, clock
) -> None:
    """A rebuild after a manual database edit must not change what search returns."""
    before = find(repo, "tag:paper", clock)
    assert repo.rebuild_search_index() == 3
    assert find(repo, "tag:paper", clock) == before
    assert find(repo, "endTime>500", clock) == {"foamacoustic"}


# -- dates and provenance ----------------------------------------------------------------------


def test_date_filters_use_creation_time(repo: JobRepository, make_spec, clock: FakeClock) -> None:
    old = repo.create(make_spec("old"))
    clock.advance(10 * 86400)
    new = repo.create(make_spec("new"))
    assert find(repo, "after:3d", clock) == {"new"}
    assert {j.name for j in repo.search(parse_query("before:3d", now=clock.now())).items} == {"old"}
    assert {old.name, new.name} == {"old", "new"}


def test_dirty_filter_uses_provenance(repo: JobRepository, make_spec, clock: FakeClock) -> None:
    from dispatch.core.provenance import GitInfo, Provenance

    clean = repo.create(make_spec("clean"))
    dirty = repo.create(make_spec("dirty"))
    def record(git: GitInfo) -> Provenance:
        return Provenance(
            captured_at=clock.now(),
            dispatch_version="0.1.0",
            python_version="3.13.0",
            hostname="eddy",
            kernel_version="6.16.3",
            git=git,
        )

    repo.save_provenance(clean.id, record(GitInfo(commit="a" * 40, dirty=False)))
    repo.save_provenance(dirty.id, record(GitInfo(commit="b" * 40, dirty=True)))

    assert find(repo, "dirty:true", clock) == {"dirty"}
    assert find(repo, "dirty:false", clock) == {"clean"}


# -- paging --------------------------------------------------------------------------------------


def test_search_pages_and_reports_the_true_total(
    repo: JobRepository, make_spec, clock: FakeClock
) -> None:
    for i in range(7):
        repo.create(make_spec(f"case{i}", tags={"bulk"}))
    page = repo.search(parse_query("tag:bulk", now=clock.now()), limit=3)
    assert page.limit == 3
    assert page.total == 7
    assert page.has_more


def test_search_results_are_newest_first(repo: JobRepository, make_spec, clock: FakeClock) -> None:
    first = repo.create(make_spec("first"))
    clock.advance(60)
    second = repo.create(make_spec("second"))
    page = repo.search(parse_query("", now=clock.now()))
    assert [j.id for j in page.items] == [second.id, first.id]


def test_metadata_written_without_a_registered_spec_still_indexes(
    conn: sqlite3.Connection, clock: FakeClock, make_spec
) -> None:
    """History whose adapter is no longer installed must remain searchable.

    Types are then inferred from the stored values rather than from a spec.
    """
    bare = JobRepository(conn, clock=clock)  # no specs registered
    metadata = CaseMetadata.from_json(
        {"spec": {"adapter": "gone", "version": 1}, "case": {"endTime": 750.0}, "extra": {}}
    )
    job = bare.create(make_spec("orphan", metadata=metadata))
    found = bare.search(parse_query("endTime>500", now=clock.now()))
    assert [j.id for j in found.items] == [job.id]


def test_exit_code_comparison(repo: JobRepository, make_spec, clock: FakeClock) -> None:
    job = repo.create(make_spec("failed"))
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)
    repo.mark_finished(job.id, state=JobState.FAILED, exit_code=1, reason=ExitReason.NONZERO)
    assert find(repo, "exit!=0", clock) == {"failed"}
