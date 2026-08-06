"""Mapping between SQLite rows and domain dataclasses.

Kept separate from the repository so that "how a job is stored" and "how a job is queried"
are two files rather than one long one, and so that a schema change touches the smaller of
the two.

Reading is deliberately tolerant: these functions parse rows written by *past* versions of
Dispatch, and refusing to load a job because one column has an unexpected value would make
history unreadable at exactly the moment it matters. Unparseable values degrade to ``None``
rather than raising.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dispatch.core.metadata import CaseMetadata
from dispatch.core.models import Job, JobEvent, JobMetrics, Note, ResourceRequest, Sample
from dispatch.core.provenance import GitInfo, Provenance
from dispatch.core.states import ExitReason, JobState

__all__ = [
    "event_from_row",
    "job_from_row",
    "note_from_row",
    "provenance_from_row",
    "sample_from_row",
]


def job_from_row(row: sqlite3.Row, *, tags: frozenset[str] = frozenset()) -> Job:
    """Build a :class:`Job` from a ``jobs`` row.

    Tags live in a separate table and are passed in rather than re-queried per row: a
    listing of 200 jobs should cost one tag query, not 200.
    """
    return Job(
        id=row["id"],
        seq=row["seq"],
        name=row["name"],
        workdir=Path(row["workdir"]),
        solver=row["solver"],
        solver_binary=row["solver_binary"],
        resources=ResourceRequest(cores=row["cores"], ram_mb=row["ram_estimate_mb"]),
        state=JobState(row["state"]),
        priority=row["priority"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        exit_code=row["exit_code"],
        exit_reason=_exit_reason(row["exit_reason"]),
        exit_signal=row["exit_signal"],
        stdout_path=_optional_path(row["stdout_path"]),
        stderr_path=_optional_path(row["stderr_path"]),
        pid=row["pid"],
        pid_start_time=row["pid_start_time"],
        tags=tags,
        metadata=_metadata(row["metadata"]),
        metrics=JobMetrics(
            peak_rss_mb=row["peak_rss_mb"],
            mean_cpu_pct=row["mean_cpu_pct"],
            runtime_s=row["runtime_s"],
        ),
    )


def event_from_row(row: sqlite3.Row) -> JobEvent:
    """Build a :class:`JobEvent` from a ``job_events`` row."""
    return JobEvent(
        id=row["id"], job_id=row["job_id"], ts=row["ts"], kind=row["kind"], detail=row["detail"]
    )


def note_from_row(row: sqlite3.Row) -> Note:
    """Build a :class:`Note` from a ``notes`` row."""
    return Note(id=row["id"], job_id=row["job_id"], ts=row["ts"], body=row["body"])


def sample_from_row(row: sqlite3.Row) -> Sample:
    """Build a :class:`Sample` from a ``job_samples`` row."""
    return Sample(ts=row["ts"], rss_mb=row["rss_mb"], cpu_pct=row["cpu_pct"])


def provenance_from_row(row: sqlite3.Row) -> Provenance:
    """Build a :class:`Provenance` from a ``job_provenance`` row."""
    return Provenance(
        captured_at=row["captured_at"],
        dispatch_version=row["dispatch_version"],
        python_version=row["python_version"],
        hostname=row["hostname"],
        kernel_version=row["kernel_version"],
        os_release=row["os_release"],
        cpu_model=row["cpu_model"],
        total_ram_mb=row["total_ram_mb"],
        solver_name=row["solver_name"],
        solver_version=row["solver_version"],
        adapter_version=row["adapter_version"],
        git=GitInfo(
            commit=row["git_commit"],
            branch=row["git_branch"],
            dirty=bool(row["git_dirty"]),
            remote=row["git_remote"],
        ),
        env_snapshot=_json_object(row["env_snapshot"]),
        env_hash=row["env_hash"],
        argv=tuple(_json_array(row["argv"])),
    )


def provenance_to_params(job_id: str, prov: Provenance) -> dict[str, Any]:
    """Flatten a :class:`Provenance` into named bind parameters."""
    return {
        "job_id": job_id,
        "captured_at": prov.captured_at,
        "dispatch_version": prov.dispatch_version,
        "python_version": prov.python_version,
        "hostname": prov.hostname,
        "kernel_version": prov.kernel_version,
        "os_release": prov.os_release,
        "cpu_model": prov.cpu_model,
        "total_ram_mb": prov.total_ram_mb,
        "solver_name": prov.solver_name,
        "solver_version": prov.solver_version,
        "adapter_version": prov.adapter_version,
        "git_commit": prov.git.commit,
        "git_branch": prov.git.branch,
        "git_dirty": 1 if prov.git.dirty else 0,
        "git_remote": prov.git.remote,
        "env_snapshot": json.dumps(dict(prov.env_snapshot), separators=(",", ":")),
        "env_hash": prov.env_hash,
        "argv": json.dumps(list(prov.argv), separators=(",", ":")),
    }


# -- tolerant scalar parsers -------------------------------------------------------------


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _exit_reason(value: str | None) -> ExitReason | None:
    if not value:
        return None
    try:
        return ExitReason(value)
    except ValueError:
        # A reason written by a future version. Better to show the job without its reason
        # than to refuse to show the job.
        return None


def _metadata(raw: str | None) -> CaseMetadata:
    payload = _json_object(raw)
    return CaseMetadata.from_json(payload) if payload else CaseMetadata.empty()


def _json_object(raw: str | None) -> Mapping[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _json_array(raw: str | None) -> list[Any]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []
