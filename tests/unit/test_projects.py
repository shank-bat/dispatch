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
