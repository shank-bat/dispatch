"""Where a job's log goes, and what happens to the previous one.

The behaviour under test is mostly about the awkward cases -- an unwritable case
directory, a second run of the same case, a log that is not a regular file -- because the
happy path is one ``open`` and the risk is entirely in the rest.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dispatch.adapters.base import DEFAULT_LOG_NAME, BaseAdapter
from dispatch.adapters.registry import build_default_registry
from dispatch.core.models import JobSpec, ResourceRequest
from dispatch.core.states import ExitReason, JobState
from dispatch.daemon.joblog import assign_log_paths, choose_log_path, rotate


def case(tmp_path: Path, name: str = "cavity") -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_job(repo, workdir: Path, *, solver: str = "fake"):
    return repo.create(
        JobSpec(
            workdir=workdir,
            solver=solver,
            resources=ResourceRequest(cores=1),
            name=workdir.name,
        )
    )


# -- the filename comes from the adapter -------------------------------------------------


def test_every_builtin_adapter_names_its_log() -> None:
    """The name is a solver convention, so the adapter supplies it -- not the daemon."""
    expected = {
        "openfoam": "log.foam",
        "su2": "log.su2",
        "basilisk": "log.basilisk",
        "calculix": "log.calculix",
        "ml": "log.ml",
        "pinn": "log.pinn",
    }
    for adapter in build_default_registry({}, load_plugins=False):
        assert adapter.log_name == expected[adapter.name]


def test_every_log_name_is_discoverable_in_a_case_directory() -> None:
    """``log.something``: recognisable to somebody who has never heard of Dispatch."""
    for adapter in build_default_registry({}, load_plugins=False):
        assert adapter.log_name.startswith("log.")
        assert "/" not in adapter.log_name


def test_an_adapter_that_says_nothing_gets_a_generic_name() -> None:
    """Additive change: an adapter written before this existed still works."""

    class Old(BaseAdapter):
        name = "old"

        @classmethod
        def detect(cls, path: Path):
            return None

        def validate(self, ctx):
            raise NotImplementedError

        def plan(self, ctx):
            raise NotImplementedError

    assert Old.log_name == DEFAULT_LOG_NAME


# -- choosing the destination -------------------------------------------------------------


def test_the_log_goes_in_the_working_directory(tmp_path: Path) -> None:
    workdir = case(tmp_path)
    destination = choose_log_path(workdir, "log.foam", tmp_path / "fallback.log")
    assert destination.path == workdir / "log.foam"
    assert destination.in_workdir
    assert destination.reason is None


def test_an_unwritable_case_directory_falls_back_and_says_why(tmp_path: Path) -> None:
    """A case that cannot hold its log is not a reason to refuse to run it."""
    workdir = case(tmp_path)
    fallback = tmp_path / "jobs" / "stdout.log"
    workdir.chmod(0o500)
    try:
        destination = choose_log_path(workdir, "log.foam", fallback)
    finally:
        workdir.chmod(0o700)
    assert destination.path == fallback
    assert not destination.in_workdir
    assert destination.reason is not None and "not writable" in destination.reason


def test_a_missing_case_directory_falls_back(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback.log"
    destination = choose_log_path(tmp_path / "gone", "log.foam", fallback)
    assert destination.path == fallback
    assert destination.reason is not None


def test_a_log_name_taken_by_a_directory_falls_back(tmp_path: Path) -> None:
    """Refusing to write is better than failing at spawn time with an IsADirectoryError."""
    workdir = case(tmp_path)
    (workdir / "log.foam").mkdir()
    fallback = tmp_path / "fallback.log"
    destination = choose_log_path(workdir, "log.foam", fallback)
    assert destination.path == fallback
    assert destination.reason is not None and "not a regular file" in destination.reason


# -- rotation -----------------------------------------------------------------------------


def test_a_previous_log_is_renamed_rather_than_deleted(tmp_path: Path) -> None:
    """Historical output is the user's; nothing here may destroy it."""
    target = tmp_path / "log.foam"
    target.write_text("first run\n")

    moved = rotate(target)
    assert moved == tmp_path / "log.foam.1"
    assert moved.read_text() == "first run\n"
    assert not target.exists()


def test_rotation_takes_the_next_free_number(tmp_path: Path) -> None:
    """Not a cascade: renaming N files on every run would invalidate N job records."""
    target = tmp_path / "log.foam"
    (tmp_path / "log.foam.1").write_text("older")
    (tmp_path / "log.foam.2").write_text("old")
    target.write_text("current")

    assert rotate(target) == tmp_path / "log.foam.3"
    assert (tmp_path / "log.foam.1").read_text() == "older"


def test_an_empty_log_is_not_rotated(tmp_path: Path) -> None:
    """Otherwise a few failed starts leave a directory full of empty numbered files."""
    target = tmp_path / "log.foam"
    target.touch()
    assert rotate(target) is None
    assert target.exists()


def test_rotating_a_missing_log_does_nothing(tmp_path: Path) -> None:
    assert rotate(tmp_path / "never-existed") is None


# -- assignment, and what history sees ------------------------------------------------------


def test_assignment_records_the_working_directory_log(repo, dispatch_config, tmp_path) -> None:
    job = make_job(repo, case(tmp_path))
    assigned = assign_log_paths(repo, dispatch_config, job, "log.fake")

    expected = job.workdir / "log.fake"
    assert assigned.log_path == expected
    assert assigned.stdout_path == expected
    assert assigned.stderr_path == expected, "one file: stdout and stderr are merged"
    assert assigned.output_path == expected


def test_the_job_directory_is_still_created(repo, dispatch_config, tmp_path) -> None:
    """It holds the step transcript and the exit sentinel however the solver logs."""
    job = make_job(repo, case(tmp_path))
    assign_log_paths(repo, dispatch_config, job, "log.fake")
    assert dispatch_config.paths.job_dir(job.id).is_dir()


def test_a_fallback_leaves_log_path_empty(repo, dispatch_config, tmp_path) -> None:
    """``log_path`` means "in the case directory"; claiming one that is not would lie."""
    workdir = case(tmp_path)
    workdir.chmod(0o500)
    try:
        job = make_job(repo, workdir)
        assigned = assign_log_paths(repo, dispatch_config, job, "log.fake")
    finally:
        workdir.chmod(0o700)

    assert assigned.log_path is None
    assert assigned.stdout_path == dispatch_config.paths.job_dir(job.id) / "stdout.log"
    assert assigned.output_path == assigned.stdout_path


def test_history_written_before_the_convention_stays_readable(repo, tmp_path) -> None:
    """The upgrade path: an old row has no log_path, and its output is still found."""
    job = make_job(repo, case(tmp_path))
    old = tmp_path / "logs" / "jobs" / job.id
    old.mkdir(parents=True)
    updated = repo.set_log_paths(job.id, stdout=old / "stdout.log", stderr=old / "stderr.log")

    assert updated.log_path is None
    assert updated.output_path == old / "stdout.log"
    assert updated.stderr_path == old / "stderr.log"


def test_rotation_repoints_the_job_that_wrote_the_old_log(repo, dispatch_config, tmp_path) -> None:
    """Otherwise last week's job starts showing this morning's output."""
    workdir = case(tmp_path)
    first = make_job(repo, workdir)
    first = assign_log_paths(repo, dispatch_config, first, "log.fake")
    assert first.log_path is not None
    first.log_path.write_text("first run output\n")
    repo.mark_preparing(first.id)
    repo.mark_started(first.id, pid=1, pid_start_time=1.0)
    repo.mark_finished(first.id, state=JobState.COMPLETED, exit_code=0, reason=ExitReason.OK)

    moved = rotate(first.log_path)
    assert moved is not None
    assert repo.repoint_logs(first.log_path, moved) == 1

    reloaded = repo.get(first.id)
    assert reloaded.log_path == moved
    assert reloaded.stdout_path == moved
    assert reloaded.output_path is not None
    assert reloaded.output_path.read_text() == "first run output\n"


def test_repointing_leaves_an_active_job_alone(repo, dispatch_config, tmp_path) -> None:
    """A running job's log is not the one being rotated away, and must not be redirected."""
    workdir = case(tmp_path)
    job = assign_log_paths(repo, dispatch_config, make_job(repo, workdir), "log.fake")
    assert job.log_path is not None
    job.log_path.write_text("live\n")
    repo.mark_preparing(job.id)
    repo.mark_started(job.id, pid=1, pid_start_time=1.0)

    assert repo.repoint_logs(job.log_path, job.log_path.with_name("log.fake.1")) == 0
    assert repo.get(job.id).log_path == job.log_path


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to an unwritable directory")
def test_the_fallback_is_only_used_when_it_has_to_be(repo, dispatch_config, tmp_path) -> None:
    """A writable case directory always wins; the fallback is not a preference."""
    job = assign_log_paths(repo, dispatch_config, make_job(repo, case(tmp_path)), "log.fake")
    assert job.log_path is not None
    assert dispatch_config.paths.job_dir(job.id) not in job.log_path.parents
