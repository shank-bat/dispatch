"""Finding a project directory by name.

The interesting cases are all the ones that are not "it found the directory": a symlink
pointing at its own ancestor, a directory the user cannot read, a tree deeper than the
limit, and a query that matches nothing. Each has to produce an answer rather than an
exception or a hang.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dispatch.core.config import ProjectsConfig
from dispatch.daemon.projects import CONTAINS, EXACT, PATH_MATCH, PREFIX, WORD, score_name
from dispatch.daemon.projects import search_projects as search


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A projects tree shaped like a real one."""
    root = tmp_path / "projects"
    for relative in (
        "openfoam/cavity",
        "openfoam/cavity/system",
        "openfoam/pitzDaily",
        "paper1/cavityRe100",
        "validation/cavity3D",
        "validation/re100-cavity",
        "ml/pinn_burgers",
        "notes",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def names(result) -> list[str]:
    return [hit.relative for hit in result.hits]


# -- matching ------------------------------------------------------------------------------


def test_directory_names_are_searched_recursively(tree: Path) -> None:
    found = names(search(ProjectsConfig(root=tree), "cavity"))
    assert "openfoam/cavity" in found
    assert "paper1/cavityRe100" in found
    assert "validation/cavity3D" in found


def test_an_exact_name_match_comes_first(tree: Path) -> None:
    """The thing called exactly what was typed is nearly always the thing meant."""
    assert names(search(ProjectsConfig(root=tree), "cavity"))[0] == "openfoam/cavity"


def test_names_outrank_paths(tree: Path) -> None:
    """"Search directory names first", made concrete.

    ``openfoam/cavity/system`` matches only because of its parent, so it must sort below
    every directory actually called something like ``cavity``.
    """
    found = names(search(ProjectsConfig(root=tree), "cavity"))
    assert found.index("openfoam/cavity/system") == len(found) - 1


def test_matching_is_case_insensitive(tree: Path) -> None:
    assert names(search(ProjectsConfig(root=tree), "CAVITY3D")) == ["validation/cavity3D"]


@pytest.mark.parametrize(
    ("name", "query", "score"),
    [
        ("cavity", "cavity", EXACT),
        ("cavityRe100", "cavity", PREFIX),
        ("re100-cavity", "cavity", WORD),
        ("mycavitytest", "cavity", CONTAINS),
        ("pitzDaily", "cavity", 0),
    ],
)
def test_match_strength_reflects_the_evidence(name: str, query: str, score: int) -> None:
    assert score_name(name, query) == score


def test_a_path_only_match_scores_far_below_any_name_match() -> None:
    assert PATH_MATCH < CONTAINS < WORD < PREFIX < EXACT


def test_no_match_returns_an_empty_result_rather_than_an_error(tree: Path) -> None:
    result = search(ProjectsConfig(root=tree), "nothing-called-this")
    assert result.hits == ()
    assert result.error is None


def test_an_empty_query_offers_the_top_level(tree: Path) -> None:
    """So the picker is useful the instant it opens."""
    found = names(search(ProjectsConfig(root=tree), ""))
    assert set(found) == {"openfoam", "paper1", "validation", "ml", "notes"}


# -- bounds ---------------------------------------------------------------------------------


def test_the_search_never_leaves_its_root(tmp_path: Path) -> None:
    """A search that could wander into / is a search that will eventually hang on NFS."""
    root = tmp_path / "projects"
    (root / "inside" / "cavity").mkdir(parents=True)
    (tmp_path / "elsewhere" / "cavity").mkdir(parents=True)

    found = [str(hit.path) for hit in search(ProjectsConfig(root=root), "cavity").hits]
    assert found == [str(root / "inside" / "cavity")]


def test_depth_is_limited(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    (root / "a/b/c/d/e/f/cavity").mkdir(parents=True)
    assert search(ProjectsConfig(root=root, max_depth=3), "cavity").hits == ()
    assert search(ProjectsConfig(root=root, max_depth=8), "cavity").hits


def test_the_walk_is_bounded_and_says_when_it_stopped(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    for index in range(50):
        (root / f"case{index}").mkdir(parents=True)
    result = search(ProjectsConfig(root=root, max_entries=10), "case")
    assert result.truncated
    assert result.scanned <= 11


def test_results_are_limited_and_the_truncation_is_reported(tmp_path: Path) -> None:
    """"Not in the list" must not be able to read as "not on the machine"."""
    root = tmp_path / "projects"
    for index in range(30):
        (root / f"cavity{index}").mkdir(parents=True)
    result = search(ProjectsConfig(root=root), "cavity", limit=5)
    assert len(result.hits) == 5
    assert result.truncated


def test_uninteresting_directories_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    (root / "node_modules" / "cavity").mkdir(parents=True)
    (root / ".git" / "cavity").mkdir(parents=True)
    (root / "real" / "cavity").mkdir(parents=True)
    assert names(search(ProjectsConfig(root=root), "cavity")) == ["real/cavity"]


# -- hazards ----------------------------------------------------------------------------------


def test_a_symlink_loop_does_not_hang_the_search(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    (root / "cases" / "cavity").mkdir(parents=True)
    (root / "cases" / "self").symlink_to(root / "cases")
    (root / "cases" / "up").symlink_to(root)

    result = search(ProjectsConfig(root=root), "cavity")
    assert names(result) == ["cases/cavity"]


def test_symlinked_directories_are_not_offered(tmp_path: Path) -> None:
    """Selecting one would submit a job against a path the user did not choose."""
    root = tmp_path / "projects"
    (root / "real" / "cavity").mkdir(parents=True)
    (root / "shortcut").symlink_to(root / "real" / "cavity")

    assert names(search(ProjectsConfig(root=root), "cavity")) == ["real/cavity"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read an unreadable directory")
def test_an_unreadable_directory_is_skipped_not_fatal(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    (root / "open" / "cavity").mkdir(parents=True)
    closed = root / "closed"
    (closed / "cavity").mkdir(parents=True)
    closed.chmod(0o000)
    try:
        result = search(ProjectsConfig(root=root), "cavity")
    finally:
        closed.chmod(0o700)

    assert names(result) == ["open/cavity"]
    assert result.error is None


def test_a_missing_root_explains_itself(tmp_path: Path) -> None:
    """With the remedy, since the default root does not exist on many machines."""
    result = search(ProjectsConfig(root=tmp_path / "nope"), "cavity")
    assert result.hits == ()
    assert result.error is not None
    assert "projects" in result.error and "root" in result.error


def test_a_directory_that_vanishes_mid_walk_is_skipped(tmp_path: Path) -> None:
    """Ordinary on a live tree; it must not raise."""
    root = tmp_path / "projects"
    (root / "cavity").mkdir(parents=True)
    doomed = root / "cavity2"
    doomed.mkdir()
    doomed.rmdir()
    assert names(search(ProjectsConfig(root=root), "cavity")) == ["cavity"]


def test_the_root_is_expanded(tmp_path: Path) -> None:
    """``root = "~/projects"`` in a config file has to mean the user's home."""
    config = ProjectsConfig(root=Path("~/definitely-not-a-real-directory"))
    assert "~" not in str(config.root)


# -- regression: the search was unusable on a real tree ------------------------------------
#
# Reproduced against ~/projects before the fix: a query for "foam" returned 50 results of
# which 47 were the insides of one matching case -- time directories like `0.orig` and
# `4.5200001`, `constant`, `system` -- and the page was reported truncated because the
# genuine matches had been pushed off the end. The cause was that the walk treated a case's
# own subdirectories as projects, and that a path-only match competed for the same page as
# a name match.


def foam_case(path: Path, *, times: tuple[str, ...] = ("0", "0.orig", "4.52")) -> Path:
    """A directory with the marker that identifies an OpenFOAM case, plus its clutter."""
    (path / "system").mkdir(parents=True, exist_ok=True)
    (path / "system" / "controlDict").write_text("application icoFoam;\n")
    (path / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    for name in times:
        (path / name).mkdir(exist_ok=True)
    for rank in range(4):
        (path / f"processor{rank}" / "constant").mkdir(parents=True, exist_ok=True)
    (path / "postProcessing" / "forceCoeffs" / "0").mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def recognise():
    """The daemon's real stat-only case screen."""
    from dispatch.adapters.registry import build_default_registry

    return build_default_registry({}, load_plugins=False).looks_like_case


def test_a_case_s_own_directories_are_not_offered_as_projects(tmp_path, recognise) -> None:
    """The bug, directly: `cavity/0.orig` and `cavity/processor2` are not projects."""
    root = tmp_path / "projects"
    foam_case(root / "openfoam" / "cavity")

    found = names(search(ProjectsConfig(root=root), "cavity", is_case=recognise))
    assert found == ["openfoam/cavity"]


def test_matches_are_not_crowded_out_by_one_case_s_internals(tmp_path, recognise) -> None:
    """Three cases under a folder called `foam` must not lose the page to time directories."""
    root = tmp_path / "projects"
    for name in ("alpha", "beta", "gamma"):
        foam_case(root / "foam" / name, times=tuple(f"{step}" for step in range(40)))

    result = search(ProjectsConfig(root=root), "foam", is_case=recognise)
    assert not result.truncated, "nothing genuine was dropped, so nothing should be claimed"
    assert "foam" in names(result)
    # The three cases are reachable because they are cases, not because of their names.
    assert {"foam/alpha", "foam/beta", "foam/gamma"} <= set(names(result))
    assert not any("/0" in entry or "processor" in entry for entry in names(result))


def test_a_case_outranks_a_mere_name_match(tmp_path, recognise) -> None:
    """The search exists to find something submittable."""
    root = tmp_path / "projects"
    (root / "cavity-notes").mkdir(parents=True)
    foam_case(root / "cavity")

    assert names(search(ProjectsConfig(root=root), "cavity", is_case=recognise))[0] == "cavity"


def test_cases_inside_a_matching_folder_are_offered(tmp_path, recognise) -> None:
    """Searching the folder name should reach the runnable cases inside it."""
    root = tmp_path / "projects"
    foam_case(root / "paper1" / "run-a")
    foam_case(root / "paper1" / "run-b")

    found = names(search(ProjectsConfig(root=root), "paper1", is_case=recognise))
    assert found[0] == "paper1"
    assert {"paper1/run-a", "paper1/run-b"} <= set(found)


def test_path_only_noise_never_consumes_the_page(tmp_path, recognise) -> None:
    """Genuine matches fill the limit first; context gets only what is left."""
    root = tmp_path / "projects"
    for index in range(6):
        foam_case(root / "sweep" / f"case{index}")
    # A directory that is not a case, with children that match only by path.
    plain = root / "sweep" / "scratch"
    for index in range(30):
        (plain / f"junk{index}").mkdir(parents=True)

    result = search(ProjectsConfig(root=root), "sweep", is_case=recognise, limit=8)
    shown = names(result)
    assert len(shown) == 8
    assert sum(1 for entry in shown if "junk" in entry) <= 1, shown
    assert sum(1 for entry in shown if entry.startswith("sweep/case")) == 6


def test_truncation_is_reported_only_when_real_matches_are_dropped(
    tmp_path, recognise
) -> None:
    root = tmp_path / "projects"
    for index in range(10):
        foam_case(root / f"cavity{index}")

    assert search(ProjectsConfig(root=root), "cavity", is_case=recognise, limit=20).truncated is (
        False
    )
    assert search(ProjectsConfig(root=root), "cavity", is_case=recognise, limit=4).truncated


def test_nested_cases_are_found_several_levels_down(tmp_path, recognise) -> None:
    """A real tree puts cases at `work/group/project/study/case`."""
    root = tmp_path / "projects"
    foam_case(root / "work" / "group" / "project" / "study" / "wing")

    found = search(ProjectsConfig(root=root), "wing", is_case=recognise)
    assert names(found) == ["work/group/project/study/wing"]
    assert found.hits[0].is_case


def test_spaces_and_special_characters_in_names_are_searchable(
    tmp_path, recognise
) -> None:
    root = tmp_path / "projects"
    foam_case(root / "my cases" / "wing (v2) [final]")

    found = search(ProjectsConfig(root=root), "wing (v2)", is_case=recognise)
    assert names(found) == ["my cases/wing (v2) [final]"]

    assert names(search(ProjectsConfig(root=root), "my cases", is_case=recognise))[0] == (
        "my cases"
    )


def test_pruning_at_cases_makes_the_walk_cheaper(tmp_path, recognise) -> None:
    """Not an optimisation detail: it is why the search can run on every keystroke."""
    root = tmp_path / "projects"
    for index in range(5):
        foam_case(root / f"case{index}", times=tuple(str(step) for step in range(30)))

    without = search(ProjectsConfig(root=root), "case")
    with_cases = search(ProjectsConfig(root=root), "case", is_case=recognise)
    assert with_cases.scanned * 4 < without.scanned


def test_the_case_flag_reaches_the_results(tmp_path, recognise) -> None:
    root = tmp_path / "projects"
    foam_case(root / "runnable")
    (root / "folder").mkdir()

    hits = {hit.relative: hit.is_case for hit in search(
        ProjectsConfig(root=root), "", is_case=recognise
    ).hits}
    assert hits == {"runnable": True, "folder": False}
