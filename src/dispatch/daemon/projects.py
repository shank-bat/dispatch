"""Finding a project directory by name, without scanning the machine.

The submit wizard opens a directory browser, because Dispatch refuses to guess where your
work lives (§9.3). That is still the right default and it is still slow when the case you
want is four levels down a tree you half remember. So there is a search: type ``cavity``,
get the handful of directories under ``~/projects`` whose names contain it, press enter.

Four constraints shape the implementation, and each rules something out:

**One configured root, never the filesystem.** ``~/projects`` by default, and only that.
A search that could wander into ``/`` is a search that will eventually sit on an NFS mount
for thirty seconds, and Dispatch's whole idea about locations is that the user says where.

**No index and no daemon.** Building one would mean a background walker, inotify watches,
and a cache to invalidate -- a periodic loop, in a program whose defining property is that
it has none (§13.26). A bounded walk of a projects tree takes single-digit milliseconds
warm, and it is triggered by a keystroke, so nothing runs when nobody is searching.

**Bounded in every direction.** Depth, directories visited, and results returned all have
limits, and the walk reports when it hit one rather than pretending the answer is complete.

**Nothing may throw.** A directory the user cannot read, a symlink pointing at its own
parent, a mount that has gone away: each is skipped, and the search returns what it found.

The walk runs in a worker thread (the handler awaits it), so a cold page cache cannot
stall the event loop.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from dispatch.core.config import ProjectsConfig

__all__ = ["ProjectHit", "ProjectSearch", "search_projects"]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProjectHit:
    """One directory that matched.

    Attributes:
        path: The absolute directory.
        name: Its final component, which is what was matched against first.
        relative: Its path below the root, for display -- ``paper1/cavity`` reads far
            better in a list than the same thing with the user's home prefixed to it.
        score: Match quality. Higher is better; used only for ordering.
        depth: How far below the root it sits.
        is_case: Whether a solver adapter's cheap marker recognises this directory as one
            of its cases. A directory that is a case is something the user can actually
            submit, so it outranks a directory that merely matched.
    """

    path: Path
    name: str
    relative: str
    score: int
    depth: int
    is_case: bool = False


@dataclass(frozen=True, slots=True)
class ProjectSearch:
    """The result of one search.

    Attributes:
        root: The directory that was searched.
        query: What was searched for.
        hits: Matches, best first.
        scanned: Directories visited.
        truncated: Whether a limit stopped the walk before it finished, so the interface
            can say "showing the first N" instead of implying there are no others.
        error: Why nothing could be searched at all, e.g. the root does not exist.
    """

    root: Path
    query: str
    hits: Sequence[ProjectHit] = ()
    scanned: int = 0
    truncated: bool = False
    error: str | None = None


# Match strengths. The gaps are wide so that later tie-breaks -- depth, then name --
# only ever reorder results of genuinely equal quality.
EXACT = 100
"""The directory is called exactly what was typed."""

PREFIX = 80
"""``cavity`` matching ``cavityRe100``: the common way a case family is named."""

WORD = 60
"""``cavity`` matching ``re100-cavity``, at a word boundary inside the name."""

CONTAINS = 40
"""``cavity`` anywhere in the name, e.g. ``mycavitytest``."""

PATH_MATCH = 10
"""Not in the name at all, but somewhere in the path below the root.

Ranked far below every name match, which is what "search directory names first" means in
practice: a directory called ``cavity`` always outranks one that merely lives inside a
folder called ``cavity``.

Ranking alone turned out not to be enough. Measured against a real projects tree, one
query produced fifty results of which forty-seven were path-only -- every descendant of
the single folder that matched -- so the genuine matches were correctly sorted to the top
of a page that was otherwise useless, and the "more exist" warning fired because real
matches had been pushed off the end. Score decides *order*; :data:`CASE_BONUS` and the tiering in
:func:`search_projects` decide what is allowed to take up room.
"""

CASE_BONUS = 1000
"""Added to any directory a solver adapter recognises as one of its cases.

Large enough to dominate every name score, because it answers a different and more
important question. The search exists to find something submittable; a directory called
``cavity`` that is an actual case is a better answer than one called ``cavity`` that is a
folder of notes, whatever their names score.
"""

SEPARATORS = "-_. "


def score_name(name: str, query: str) -> int:
    """How well a directory name matches a query. Zero means it does not.

    Case-insensitive throughout: nobody typing into a search box is thinking about
    whether the case they want is ``cavity3D`` or ``Cavity3d``.
    """
    lowered, needle = name.lower(), query.lower()
    if not needle:
        return 0
    if lowered == needle:
        return EXACT
    if lowered.startswith(needle):
        return PREFIX
    position = lowered.find(needle)
    if position < 0:
        return 0
    return WORD if lowered[position - 1] in SEPARATORS else CONTAINS


def search_projects(
    config: ProjectsConfig,
    query: str,
    *,
    limit: int | None = None,
    is_case: Callable[[Path], bool] | None = None,
) -> ProjectSearch:
    """Find directories under the configured root whose names match ``query``.

    Args:
        config: Root, depth, and the bounds on the walk.
        query: What to look for. Empty returns the root's immediate children, which makes
            the search box useful before anything has been typed into it.
        limit: Override the configured result count.
        is_case: Cheap test for "this directory is a submittable case", supplied by the
            caller so this module stays solver-agnostic -- the daemon passes
            :meth:`~dispatch.adapters.registry.AdapterRegistry.looks_like_case`. It does
            two things: a case outranks a mere name match, and the walk **stops** at one,
            because a case's own time directories and ``processorN`` folders are not
            projects and offering them is what made this search unusable. Omitted, the
            walk behaves exactly as it did before this argument existed.

    Returns:
        The matches, best first, with enough context for the interface to be honest about
        what it did not look at.
    """
    root = config.root.expanduser()
    ceiling = limit or config.limit
    recognise = is_case or (lambda _: False)
    try:
        resolved = root.resolve()
    except OSError as exc:
        return ProjectSearch(root=root, query=query, error=f"cannot read {root}: {exc}")
    if not resolved.is_dir():
        return ProjectSearch(
            root=resolved,
            query=query,
            error=(
                f"{resolved} does not exist. Set [projects] root in the configuration to "
                "where your work lives."
            ),
        )

    named: list[ProjectHit] = []
    case_paths: list[ProjectHit] = []
    other_paths: list[ProjectHit] = []
    scanned = 0
    truncated = False

    for directory, depth, case in _walk(resolved, config, recognise):
        scanned += 1
        if scanned > config.max_entries:
            truncated = True
            break
        by_name = _name_score(directory, resolved, query)
        if by_name:
            score = by_name + (CASE_BONUS if case else 0)
            named.append(_hit(directory, resolved, score, depth, case))
        elif _path_matches(directory, resolved, query):
            bucket = case_paths if case else other_paths
            bucket.append(_hit(directory, resolved, PATH_MATCH, depth, case))

    for bucket in (named, case_paths, other_paths):
        bucket.sort(key=_order)

    # Tiered rather than one sorted list, so that path-only noise can never consume the
    # page. A name match is what was asked for; a case inside a matching folder is a real,
    # selectable answer; everything else is context and gets only the room left over.
    wanted = named + case_paths
    if len(wanted) > ceiling:
        # Genuine matches were dropped: the user needs to know there are more.
        wanted, truncated = wanted[:ceiling], True
    hits = wanted + other_paths[: max(0, ceiling - len(wanted))]

    return ProjectSearch(
        root=resolved, query=query, hits=tuple(hits), scanned=scanned, truncated=truncated
    )


def _hit(directory: Path, root: Path, score: int, depth: int, is_case: bool) -> ProjectHit:
    """Build one result."""
    return ProjectHit(
        path=directory,
        name=directory.name,
        relative=_relative(directory, root),
        score=score,
        depth=depth,
        is_case=is_case,
    )


def _order(hit: ProjectHit) -> tuple[int, int, str, str]:
    """Best match, then shallowest, then alphabetical.

    Depth second because a project directory is nearly always above its own sub-cases, so
    the thing the user meant sorts above the things inside it.
    """
    return (-hit.score, hit.depth, hit.name.lower(), str(hit.path))


def _name_score(directory: Path, root: Path, query: str) -> int:
    """How well a directory's own name matches, ignoring its path. Zero means it does not."""
    if not query:
        # No query: offer the top level, so the picker is not an empty box.
        return EXACT if directory.parent == root else 0
    return score_name(directory.name, query)


def _path_matches(directory: Path, root: Path, query: str) -> bool:
    """Whether the query appears anywhere in the path below the root but not in the name."""
    return bool(query) and query.lower() in _relative(directory, root).lower()


def _relative(directory: Path, root: Path) -> str:
    """The directory's path below the root, or its full path if it is not below it."""
    try:
        return str(directory.relative_to(root))
    except ValueError:  # pragma: no cover - only reachable via a symlink out of the root
        return str(directory)


def _walk(
    root: Path, config: ProjectsConfig, is_case: Callable[[Path], bool]
) -> Iterator[tuple[Path, int, bool]]:
    """Yield ``(directory, depth, is_case)`` below ``root``, bounded and loop-proof.

    Breadth first rather than depth first so that a bound cut short by ``max_entries``
    truncates the *deepest* level rather than an arbitrary branch -- the shallow
    directories, which are the ones people are usually looking for, are always visited.

    **A case is yielded but not descended into.** Its time directories, ``processorN``
    folders and ``postProcessing`` tree are part of one case rather than projects of their
    own, and offering them is what made this search unusable on a real tree: a query that
    should return three cases returned fifty rows, forty-seven of them the insides of one
    of them. Pruning there also makes the walk dramatically cheaper, since a meshed case
    holds far more directories than the project tree above it.

    Symlinked directories are not followed. ``a/b -> a`` is enough to make a naive walk run
    until the depth limit, and a pair of symlinks pointing at each other defeats the depth
    limit too; the visited-inode set below catches the remaining cases, such as a bind
    mount of a directory into itself.
    """
    skip = set(config.skip)
    visited: set[tuple[int, int]] = set()
    frontier: list[tuple[Path, int]] = [(root, 0)]

    start = _identity(root)
    if start is not None:
        visited.add(start)

    while frontier:
        directory, depth = frontier.pop(0)
        if depth >= config.max_depth:
            continue
        for child in _children(directory):
            if child.name.startswith(".") or child.name in skip:
                continue
            marker = _identity(child)
            if marker is None or marker in visited:
                continue
            visited.add(marker)
            case = is_case(child)
            yield child, depth + 1, case
            if not case:
                frontier.append((child, depth + 1))


def _children(directory: Path) -> list[Path]:
    """Real subdirectories of ``directory``, sorted, skipping what cannot be read.

    ``follow_symlinks=False`` on the type check is what keeps a symlinked directory out of
    the walk entirely, rather than merely out of the recursion: a symlink into a huge tree
    should not appear as a result either, because selecting it would submit a job against
    a path whose real location the user did not choose.
    """
    try:
        with os.scandir(directory) as entries:
            found = [
                Path(entry.path)
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
            ]
    except (PermissionError, OSError) as exc:
        # A directory the user cannot read is normal on a shared machine and is not worth
        # a warning on every keystroke; it is simply not part of the answer.
        log.debug("Skipping %s during project search: %s", directory, exc)
        return []
    return sorted(found, key=lambda path: path.name.lower())


def _identity(path: Path) -> tuple[int, int] | None:
    """A directory's ``(device, inode)``, or ``None`` if it cannot be stat'd.

    The pair, not the path: it is what makes a bind mount of a tree into itself, or any
    other way the same directory appears twice, visible as the same directory. ``None``
    for a directory that vanished between being listed and being examined, which on a
    live projects tree happens often enough to matter.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_dev, info.st_ino)
