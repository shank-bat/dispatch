"""Where a job's output goes, and what happens to the last run's.

Dispatch used to write every job's stdout to ``~/.local/share/dispatch/logs/jobs/<uuid>/``.
That is a fine place for a machine to keep a file and a poor place for a person to find
one: nobody browsing ``~/projects/cavity`` can see their own solver output, and nobody
remembers a UUID. The log now lives in the case directory under a name that solver's own
users already recognise -- ``log.<something>``, and each adapter says which -- so nothing
in this module knows that any particular solver exists.

The per-job directory does not go away. It still holds the preparation transcript, the
exit sentinel that makes exit codes survive a daemon crash (§6.5), and it is still where
output falls back to when the case directory cannot take it. Those are internal
bookkeeping and belong out of the user's way.

**Rotation.** Two runs of the same case would otherwise share one file, and the older
job's history would silently start showing the newer job's output -- a quiet corruption of
the record, which is the thing Dispatch is least allowed to get wrong. So an existing log
is renamed aside before a new run starts, and the job that wrote it has its record
repointed at the new name. Nothing is deleted, and ``dispatch logs`` on a job from last
month still shows what that job printed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dispatch.core.config import Config
from dispatch.core.models import Job
from dispatch.db.repository import JobRepository

__all__ = ["LogDestination", "assign_log_paths", "choose_log_path", "rotate"]

log = logging.getLogger(__name__)

MAX_ROTATIONS = 1000
"""How many rotated logs one case directory may accumulate before rotation gives up.

Not a cleanup policy -- the files are the user's -- just a bound on how long the search
for a free name may run. A case run a thousand times has other problems.
"""


@dataclass(frozen=True, slots=True)
class LogDestination:
    """Where one job's output will be written.

    Attributes:
        path: The file the solver's stdout and stderr are opened onto.
        in_workdir: Whether that file is in the case directory, i.e. whether the user will
            find it where they expect. ``False`` means the fallback was used.
        reason: Why the fallback was used, for a job event the user can read. ``None`` on
            the normal path.
    """

    path: Path
    in_workdir: bool
    reason: str | None = None


def choose_log_path(workdir: Path, name: str, fallback: Path) -> LogDestination:
    """Decide where a job's solver output should go.

    The case directory when it will accept the file, and the job's own log directory when
    it will not -- a read-only mount, a directory owned by somebody else, a full disk. A
    case that cannot hold its log is not a reason to refuse to run it; it is a reason to
    put the log somewhere else and say so.

    Args:
        workdir: The case directory.
        name: The adapter's log filename, e.g. ``log.<solver>``.
        fallback: Where to write if the case directory cannot take it.

    Returns:
        The chosen destination.
    """
    candidate = workdir / name
    if not workdir.is_dir():
        return LogDestination(fallback, False, f"{workdir} is not a directory")
    if not os.access(workdir, os.W_OK):
        return LogDestination(fallback, False, f"{workdir} is not writable")
    if candidate.exists() and not candidate.is_file():
        return LogDestination(fallback, False, f"{candidate} exists and is not a regular file")
    return LogDestination(candidate, True)


def rotate(path: Path) -> Path | None:
    """Move an existing log aside so a new run starts with an empty one.

    Renamed to ``<name>.1``, ``<name>.2``, and so on -- the first free number, rather than
    cascading every existing file up by one. Cascading would rewrite N paths and invalidate
    N job records on every run; taking the next free name touches exactly one.

    An empty or missing file is left alone: there is nothing to preserve, and rotating it
    would leave a directory full of empty ``log.<name>.7`` files after a few failed starts.

    Returns:
        The path the old log now has, or ``None`` if there was nothing to move.
    """
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
    except OSError:
        return None

    for index in range(1, MAX_ROTATIONS):
        target = path.with_name(f"{path.name}.{index}")
        if target.exists():
            continue
        try:
            path.rename(target)
        except OSError as exc:
            # Losing the previous log's *name* must never cost the new run its start. The
            # new job then appends to the existing file, which is untidy but keeps both
            # runs' output rather than discarding either.
            log.warning("Could not rotate %s aside: %s", path, exc)
            return None
        log.info("Rotated %s to %s", path, target.name)
        return target

    log.warning("Not rotating %s: %d rotated logs already exist", path, MAX_ROTATIONS)
    return None


def assign_log_paths(
    repo: JobRepository, config: Config, job: Job, log_name: str
) -> Job:
    """Decide and record where a job's output will go, immediately after it is created.

    Done at submission rather than at launch so the answer is visible in the queue, in
    ``dispatch show``, and in the dry run *before* the job starts -- "where will this
    write" is a question people ask while deciding whether to submit at all. The executor
    re-checks at launch, because a case directory can change in the days a job spends
    queued.

    The per-job directory is created either way: it holds the step transcript and the exit
    sentinel regardless of where the solver's own output lands.

    Args:
        repo: Persistence.
        config: For the per-job log directory.
        job: The freshly created job.
        log_name: The adapter's log filename.

    Returns:
        The job, with its paths recorded.
    """
    job_dir = config.paths.job_dir(job.id)
    job_dir.mkdir(parents=True, exist_ok=True)
    destination = choose_log_path(job.workdir, log_name, job_dir / "stdout.log")
    if destination.reason:
        log.info("Job %s logs to %s: %s", job.id[:8], destination.path, destination.reason)
    return repo.set_log_paths(
        job.id,
        stdout=destination.path,
        stderr=destination.path,
        log=destination.path if destination.in_workdir else None,
    )
